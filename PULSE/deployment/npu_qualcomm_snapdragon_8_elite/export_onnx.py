"""Export whole-image encoder/receiver graphs and their matched mobile assets."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import onnx
import torch

from deployment.common import load_bundle
from codec.models.entropy import entropy_coding as ec
from .graphs import RuntimeQpEncoder, RuntimeQpDecoder, qp_controls, fold_hardgelu_for_qnn


def write_array(stream, value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    stream.write(struct.pack("<I", array.size))
    stream.write(array.tobytes())


def export_entropy(artifact, fingerprint, config, path, qp, height, width):
    state, cfg = artifact["state_dict"], artifact["constructor"]
    meta_cdf, meta_lengths = artifact["cdf_tables"]["meta_prior"][qp]
    if "y_part_specific" not in artifact["cdf_tables"]:
        raise ValueError("mobile export requires calibrated y CDFs; run postprocess.py first")
    mode = True
    y_cdf, y_lengths = artifact["cdf_tables"]["y_part_specific"]
    merge = artifact.get("meta_prior_index_merge")
    probabilities = ec.default_meta_prior_index_merge_probabilities(64) if merge is None else merge["probabilities"][qp]
    shift = 5 if merge is None else merge["adaptation_shift"]
    cdf = meta_cdf.to(torch.int64).reshape(64, cfg["z_ch"], -1)
    lengths = meta_lengths.long().reshape(64, cfg["z_ch"])
    values = torch.minimum(torch.arange(128)[None, None], (lengths - 2)[..., None])
    total = torch.gather(cdf, 2, (lengths - 1)[..., None]).squeeze(2).float()
    freq = torch.gather(cdf, 2, values + 1) - torch.gather(cdf, 2, values)
    cost = -torch.log2((freq.float() / total[..., None]).clamp_min(1e-12))
    with path.open("wb") as stream:
        stream.write(struct.pack(
            "<8s16s10I", b"PULSENT2", bytes.fromhex(fingerprint)[:16], 2,
            width, height, cfg["M"], cfg["z_ch"], qp, cfg["input_clip"],
            3, mode, shift))
        for value, dtype in [
            (state["input_divisor"], "<i4"), (state["weight_q"], "i1"),
            (state["multiplier"], "<i4"), (state["affine_bias"][qp], "<i4"),
            (meta_cdf, "<i4"), (meta_lengths, "<i4"),
            (y_cdf, "<i4"), (y_lengths, "<i4"), (cost, "<f4"), (probabilities, "<u2"),
        ]:
            write_array(stream, value, dtype)


def export_controls(model, path, height, width):
    dico, render = model.decoder.q_scale_dico.shape[1], model.decoder.renderer.q_scale.shape[1]
    data = bytearray(struct.pack("<8s6I", b"PQPCTRL1", 1, 8, dico, render, height // 16, width // 16))
    for qp in range(8):
        controls = qp_controls(model, qp, height=height, width=width)
        values = [float(c.flatten()[0]) for c in controls[:6]]
        data.extend(np.concatenate([values, controls[6].flatten().numpy(),
                                    controls[7].flatten().numpy()]).astype("<f2").tobytes())
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--width", type=int, default=1920)
    args = parser.parse_args()
    if min(args.height, args.width) < 64 or args.height % 64 or args.width % 64:
        parser.error("whole-image height/width must be positive multiples of 64")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, artifact, fingerprint, config = load_bundle(args.model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    controls = qp_controls(model, 3, height=args.height, width=args.width)
    export_controls(model, out / "qp_controls.bin", args.height, args.width)
    for qp in range(8):
        export_entropy(artifact, fingerprint, config, out / f"entropy_qp{qp}.bin", qp, args.height, args.width)
    image = torch.rand(1, 3, args.height, args.width)
    reference_encoder = RuntimeQpEncoder(model, height=args.height, width=args.width).eval()
    with torch.no_grad():
        symbols = tuple(t.clone() for t in reference_encoder(image, *controls[:5]))
    records = {}
    for kind in ("encoder", "decoder"):
        network = copy.deepcopy(model)
        folded = fold_hardgelu_for_qnn(network)
        if kind == "encoder":
            module = RuntimeQpEncoder(network, height=args.height, width=args.width).eval()
            examples = (image, *controls[:5])
            names = ["image", "q_image", "q_hyper_z", "q_hyper_m", "q_prior", "q_basic"]
            outputs = ["z_hat", "symbols0_packed", "symbols1_packed"]
        else:
            module = RuntimeQpDecoder(network, height=args.height, width=args.width).eval()
            examples = (*symbols, *controls[1:])
            names = ["z_hat", "symbols0_packed", "symbols1_packed", "q_hyper_z",
                     "q_hyper_m", "q_prior", "q_basic", "q_codec_dec", "q_dico", "q_render"]
            outputs = ["rgb"]
        path = out / f"{kind}.onnx"
        torch.onnx.export(module, examples, str(path), input_names=names, output_names=outputs,
                          opset_version=17, do_constant_folding=True, dynamo=False)
        onnx.checker.check_model(str(path))
        with torch.inference_mode():
            result = module(*examples)
        result = result if isinstance(result, tuple) else (result,)
        np.savez(out / f"{kind}_reference.npz",
                 **{name: tensor.detach().numpy() for name, tensor in zip(names, examples)},
                 **{f"out_{name}": tensor.detach().numpy() for name, tensor in zip(outputs, result)})
        records[kind] = dict(inputs={n: list(t.shape) for n, t in zip(names, examples)},
                             outputs=outputs, folded_blocks=folded)
    files = [*out.glob("*.onnx"), *out.glob("entropy_qp*.bin"), out / "qp_controls.bin"]
    model_id = config.get("model_id", f"pulse{'' if config['size'] == 'base' else '-' + config['size']}-{config['objective']}")
    manifest = dict(format_version=1, model_id=model_id, fingerprint=fingerprint,
                    height=args.height, width=args.width, qp_count=8, graphs=records,
                    scope="whole-image neural encoder/receiver plus Meta Prior, integer CDF and z4/y3/y3 rANS",
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
