"""Create a phone-compatible whole-image .pulse stream on the host."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from codec.images import read_image, pad_image
from deployment.common import load_bundle
from .graphs import RuntimeQpEncoder, qp_controls, fold_hardgelu_for_qnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--binary", required=True, help="host pulse_entropy executable")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qp", type=int, choices=range(8), default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((Path(args.assets) / "manifest.json").read_text())
    model, _, fingerprint, _ = load_bundle(args.model)
    if fingerprint != manifest["fingerprint"]:
        raise ValueError("model and mobile assets differ")
    image = pad_image(read_image(args.input))
    if list(image.shape[-2:]) != [manifest["height"], manifest["width"]]:
        raise ValueError("padded image must match the exported whole-image shape")
    controls = qp_controls(model, args.qp, height=image.shape[-2], width=image.shape[-1])
    fold_hardgelu_for_qnn(model)
    module = RuntimeQpEncoder(model, height=image.shape[-2], width=image.shape[-1]).eval()
    with torch.inference_mode():
        symbols = module(image, *controls[:5])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pulse-symbols-") as temp:
        temp = Path(temp)
        for name, tensor in zip(("z", "y0", "y1"), symbols):
            tensor.cpu().numpy().astype("<i2").tofile(temp / name)
        subprocess.run([args.binary, "encode-symbols", "--bundle",
                        str(Path(args.assets) / f"entropy_qp{args.qp}.bin"),
                        "--z", str(temp / "z"), "--y0", str(temp / "y0"),
                        "--y1", str(temp / "y1"), "--output", str(temp / "image.pulse")], check=True)
        output.write_bytes((temp / "image.pulse").read_bytes())
    print(output)


if __name__ == "__main__":
    main()
