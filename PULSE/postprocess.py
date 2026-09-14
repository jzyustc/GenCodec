"""Fit Meta Prior, export integer Linear CDF decoding, and calibrate y CDFs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

from codec.checkpoint import load_model, save_model, META_PRIOR_BANKS
from codec.config import load_config, recipe_path
from codec.models.entropy.entropy_coding import build_cdf_tables, MAX_ENTROPY_CODING_VALUE, require_entropy_extension
from codec.models.entropy.int8scale import IntegerLinearCDFIndexDecoder
from codec.models.entropy.meta_prior import MetaPrior
from codec.training.data import read_records
from codec.training.meta_prior import collect_meta_prior_training_data, fit_meta_prior
from codec.training.y_cdf import calibrate_bundle, calibration_settings, calibration_crop
from codec.training.bundle import new_bundle_output, validate_bundle
from codec.utils.model_io import atomic_torch_save, sha256


def tensor_only(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.astype(np.int32) if value.dtype == np.uint16 else value.copy())
    if isinstance(value, torch.Tensor):
        return value.int() if value.dtype == torch.uint16 else value.detach().cpu()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: tensor_only(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_only(v) for v in value)
    return value


def export_integer(model, config, data, output):
    """Quantize only the entropy-control projection; neural transforms stay float."""
    channels = model.z_ch
    channel_max = np.zeros(channels, dtype=np.float32)
    for values in data["z_by_qp"].values():
        channel_max = np.maximum(channel_max, np.abs(values.astype(np.float32)).max(axis=0))
    divisor = torch.from_numpy(np.maximum(1, np.ceil(channel_max / 127)).astype(np.int32))
    em = model.entropy_model
    integer = IntegerLinearCDFIndexDecoder.from_float(
        em.scale_dec, input_divisor=divisor.to(next(model.parameters()).device)).cpu().eval()
    artifact = dict(
        format_version=1,
        implementation=IntegerLinearCDFIndexDecoder.artifact_implementation,
        algorithm="Meta Prior + Linear CDF Index Decoding",
        constructor=dict(M=integer.M, z_ch=integer.z_ch, qp_num=integer.qp_num,
                         up_factor=integer.up_factor, input_clip=integer.input_clip),
        state_dict={k: v.cpu() for k, v in integer.state_dict().items()
                    if k != "last_input_saturation_count"},
        cdf_tables=build_cdf_tables(em),
        entropy_coding_version=2,
        max_entropy_coding_value=MAX_ENTROPY_CODING_VALUE,
        y_cdf_mode="gaussian",
        calibration=dict(num_images=data["num_images"], crop=data["crop_size"],
                         seed=data["seed"], input_divisor=divisor,
                         channel_max_abs=torch.from_numpy(channel_max)),
    )
    if em.meta_prior is not None and bool(em.meta_prior.index_merge_ready.item()):
        artifact["meta_prior_index_merge"] = dict(
            format_version=1, algorithm="Meta-Prior Index Merge Coding",
            adaptation_shift=int(em.meta_prior.index_merge_adaptation_shift.item()),
            bank_count=em.meta_prior.bank_count,
            probabilities=em.meta_prior.index_merge_probabilities.detach().cpu())
        artifact["format_version"] = 2
        artifact["entropy_coding_version"] = 3
    checkpoint = save_model(model, config, output)
    artifact["y_cdf_calibration"] = {"checkpoint_sha256": sha256(checkpoint)}
    atomic_torch_save(tensor_only(artifact), Path(output) / "entropy_control_int.pt")


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(recipe_path("postprocess")))
    selected, _ = config_parser.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=selected.config)
    parser.add_argument("--model", required=True, help="trained model bundle")
    parser.add_argument("--data", required=True, help="training images/list; never an evaluation split")
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True, help="new complete checkpoint directory; must not exist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-images", type=int, default=4035)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--em-iters", type=int, default=6)
    parser.add_argument("--init-steps", type=int, default=500)
    parser.add_argument("--refine-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--symbol-radius", type=int, default=127)
    parser.add_argument("--assign-chunk", type=int, default=8192)
    parser.add_argument("--index-merge", action=argparse.BooleanOptionalAction,
                        help="fit Index Merge initial probabilities (enabled by default)")
    settings = load_config(selected.config)
    y_cdf_settings = settings.pop("y_cdf")
    unknown = settings.keys() - {action.dest for action in parser._actions}
    if unknown:
        parser.error(f"unknown post-training parameters: {sorted(unknown)}")
    parser.set_defaults(**settings)
    args = parser.parse_args()
    if min(args.num_images, args.threads, args.init_steps, args.refine_steps) < 1 or args.em_iters < 0:
        parser.error("image/step/thread counts must be positive and --em-iters nonnegative")
    if args.crop < 64 or args.crop % 64:
        parser.error("--crop must be a multiple of 64")
    require_entropy_extension()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint output must be a new directory: {output}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model = load_model(args.model, device, entropy=False)
    records = read_records(args.data, args.data_root)
    fitting_paths = [r["image"] for r in records]
    y_settings = calibration_settings(model.release_config, y_cdf_settings)
    if len(fitting_paths) < y_settings["num_images"]:
        parser.error("insufficient y CDF fitting images")
    records = random.Random(args.seed).sample(records, min(args.num_images, len(records)))
    data = collect_meta_prior_training_data(
        model, [r["image"] for r in records], device=device,
        qps=tuple(range(model.qp_num)), crop_size=args.crop, seed=args.seed)
    payload = fit_meta_prior(
        model, data, device=device, bank_count=META_PRIOR_BANKS,
        em_iters=args.em_iters, first_steps=args.init_steps,
        later_steps=args.refine_steps, lr=args.lr, symbol_radius=args.symbol_radius,
        assign_chunk=args.assign_chunk, index_merge=args.index_merge)
    model.entropy_model.meta_prior = MetaPrior(
        qp_num=model.qp_num, bank_count=META_PRIOR_BANKS, channel=model.z_ch).to(device)
    model.entropy_model.meta_prior.load_fitted(payload)
    config = deepcopy(model.release_config)
    config["coding"] = {"skip_index_cutoff": 2}
    with new_bundle_output(output) as staged:
        export_integer(model, config, data, staged)
        del model
        y_cdf_mode = calibrate_bundle(staged, fitting_paths, y_cdf_settings, device=device)
        sample = calibration_crop(fitting_paths[0], 128, args.seed)
        validate_bundle(staged, sample, device=device)
    print(json.dumps(dict(output=str(output), images=len(records), bank_count=META_PRIOR_BANKS,
                          integer_linear_cdf=True, index_merge=args.index_merge,
                          y_cdf_mode=y_cdf_mode, validated_qps=list(range(8)))))


if __name__ == "__main__":
    main()
