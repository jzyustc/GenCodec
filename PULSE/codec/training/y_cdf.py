"""Training-only calibration of the two checkerboard residual CDFs."""
import argparse
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from codec.checkpoint import load_model
from codec.config import load_config, recipe_path
from codec.models.entropy.y_cdf import validate_part_cdf
from codec.training.data import read_records
from codec.training.bundle import write_bundle_checksums
from codec.utils.model_io import atomic_torch_save, sha256

CALIBRATION_QPS = (0, 2, 5, 7)
CALIBRATION_SEED = 26090604
GAUSSIAN_PRIOR_COUNT = 65536


def calibration_crop(path, crop_size, seed):
    with Image.open(path) as source:
        image = source.convert("RGB")
    if min(image.size) < crop_size:
        image = TF.resize(image, crop_size, antialias=True)
    rng = random.Random(seed)
    left = rng.randint(0, image.width - crop_size) if image.width > crop_size else 0
    top = rng.randint(0, image.height - crop_size) if image.height > crop_size else 0
    image = TF.crop(image, top, left, crop_size, crop_size)
    if rng.random() < 0.5:
        image = TF.hflip(image)
    return TF.to_tensor(image).unsqueeze(0)


def calibration_settings(config, settings):
    """All sizes use their objective's calibrated-CDF recipe."""
    selected = dict(settings[config["objective"]])
    if set(selected) != {"num_images", "crop"}:
        raise ValueError("y CDF recipe requires num_images and crop")
    if selected["num_images"] < 1 or selected["crop"] < 64 or selected["crop"] % 64:
        raise ValueError("positive image count and crop divisible by 64 required")
    return selected


def accumulate_counts(counts, samples, lengths):
    """Count transported symbols, preserving the original escape/bypass bin."""
    for part, packed in enumerate(samples):
        packed = packed.astype(np.int32)
        indexes, symbols = packed & 255, packed >> 8
        if np.any(indexes >= 64):
            raise ValueError("calibration requires raw scale indexes in [0,63]")
        leaf = np.minimum(np.abs(symbols) * 2 - (symbols > 0), lengths[indexes] - 2)
        counts[part] += np.bincount(indexes * 128 + leaf, minlength=64 * 128).reshape(64, 128)


def quantize_probabilities(probabilities):
    n = len(probabilities)
    scaled = (65536 - n) * probabilities / probabilities.sum()
    frequencies = 1 + np.floor(scaled).astype(np.int64)
    order = np.argsort(-(scaled - np.floor(scaled)), kind="stable")
    frequencies[order[:65536 - int(frequencies.sum())]] += 1
    return np.r_[0, frequencies.cumsum()].astype(np.int32)


def fit_part_cdf(counts, cdf, lengths):
    if counts.shape != (2, 64, 128) or np.any(counts < 0):
        raise ValueError("expected nonnegative counts with shape [2,64,128]")
    values = np.stack([cdf.copy(), cdf.copy()])
    for part in range(2):
        for scale in range(64):
            n = int(lengths[scale]) - 1
            if counts[part, scale, :n].sum():
                prior = GAUSSIAN_PRIOR_COUNT * np.diff(cdf[scale, :n + 1]).astype(np.float64) / 65536
                values[part, scale, :n + 1] = quantize_probabilities(counts[part, scale, :n] + prior)
    return validate_part_cdf(np.concatenate(values), np.tile(lengths, 2), lengths)


@torch.inference_mode()
def calibrate_artifact(model, artifact, paths, *, num_images, crop):
    if num_images < 1 or crop < 64 or crop % 64:
        raise ValueError("positive image count and crop divisible by 64 required")
    paths = list(paths)[:num_images]
    if len(paths) != num_images:
        raise ValueError("insufficient y CDF fitting images")
    cdf, lengths = [np.asarray(value, dtype=np.int32) for value in artifact["cdf_tables"]["y"]]
    if cdf.shape[0] != 64 or lengths.shape != (64,):
        raise ValueError("expected 64 Gaussian scale CDFs")
    counts = np.zeros((2, 64, 128), dtype=np.int64)
    checks = []
    cutoff = model.release_config.get("coding", {}).get("skip_index_cutoff", -1)
    cutoff = None if cutoff < 0 else cutoff
    previous = (torch.get_num_threads(), torch.get_float32_matmul_precision(),
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic)
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model.eval()
        device = next(model.parameters()).device
        for index, path in enumerate(paths):
            image = calibration_crop(Path(path), crop, CALIBRATION_SEED + index).to(device)
            for qp in CALIBRATION_QPS:
                result = model.compress(image, qp=qp, skip_index_cutoff=cutoff,
                                        y_cdf_mode="gaussian", collect_y_symbols=True)
                if index < 2:
                    decoded = model.decompress(
                        result["stream"], result["z_shape"], result["y_shape"], qp=qp,
                        skip_index_cutoff=cutoff, y_cdf_mode="gaussian")
                    checks.append((image, qp, decoded.cpu()))
                accumulate_counts(counts, result["y_cdf_samples"], lengths)
            print(f"[y CDF] calibrated images {index + 1}/{len(paths)}", flush=True)
        values, fitted_lengths = fit_part_cdf(counts, cdf, lengths)
        entropy = model.entropy_model
        previous_part_cdf = getattr(entropy, "_cached_y_part_cdf", None)
        entropy._cached_y_part_cdf = (values, fitted_lengths)
        try:
            # Compare actual decoder outputs, not the encoder's local FP32
            # reconstruction (which can differ by roundoff on CPU).
            for image, qp, expected in checks:
                result = model.compress(image, qp=qp, skip_index_cutoff=cutoff,
                                        y_cdf_mode="part_specific")
                decoded = model.decompress(
                    result["stream"], result["z_shape"], result["y_shape"], qp=qp,
                    skip_index_cutoff=cutoff, y_cdf_mode="part_specific")
                if not torch.equal(decoded.cpu(), expected):
                    raise RuntimeError("y CDF calibration changed decoded latents")
        finally:
            if previous_part_cdf is None:
                del entropy._cached_y_part_cdf
            else:
                entropy._cached_y_part_cdf = previous_part_cdf
    finally:
        threads, matmul_precision, cudnn_tf32, benchmark, deterministic = previous
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
    output = deepcopy(artifact)
    output["cdf_tables"]["y_part_specific"] = (
        torch.from_numpy(values), torch.from_numpy(fitted_lengths))
    output["y_cdf_mode"] = "part_specific"
    output["y_cdf_format_version"] = 1
    label = " + Part-specific y CDF"
    output["algorithm"] = artifact.get("algorithm", "Integer entropy control").removesuffix(label) + label
    output["y_cdf_calibration"] = dict(
        checkpoint_sha256=artifact["y_cdf_calibration"]["checkpoint_sha256"],
        images=num_images, crop=crop, seed=CALIBRATION_SEED, qps=list(CALIBRATION_QPS),
        prior=GAUSSIAN_PRIOR_COUNT, skip_index_cutoff=-1 if cutoff is None else cutoff)
    return output


def calibrate_bundle(directory, paths, settings, *, device="cpu"):
    directory = Path(directory)
    config = load_config(directory / "config.json")
    selected = calibration_settings(config, settings)
    model = load_model(directory, device)
    artifact_path = directory / "entropy_control_int.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    checkpoint_hash = sha256(directory / "model.pt")
    output = calibrate_artifact(model, artifact, paths, **selected)
    if sha256(directory / "model.pt") != checkpoint_hash:
        raise RuntimeError("network checkpoint changed during y CDF calibration")
    atomic_torch_save(output, artifact_path)
    write_bundle_checksums(directory)
    return output["y_cdf_mode"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="exported bundle; updates only its entropy artifact")
    parser.add_argument("--data", required=True, help="training images; manifest order is preserved")
    parser.add_argument("--data-root")
    parser.add_argument("--config", default=str(recipe_path("postprocess")))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    settings = load_config(args.config)["y_cdf"]
    records = read_records(args.data, args.data_root)
    mode = calibrate_bundle(args.model, [r["image"] for r in records], settings, device=args.device)
    print(f"y CDF: {mode}", flush=True)


if __name__ == "__main__":
    main()
