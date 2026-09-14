"""Stage and run full-image compression, decompression or timing on REDMI via ADB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
from PIL import Image
from codec import bitstream


def prepare_rgb(path, height, width):
    with Image.open(path) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = image.shape[:2]
    if h > height or w > width:
        raise ValueError("image exceeds the exported whole-image shape; export a larger graph")
    return np.pad(image, ((0, height - h), (0, width - w), (0, 0)), mode="edge").tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["compress", "decompress", "benchmark"], required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--binary", required=True, help="Android pulse_mobile executable")
    parser.add_argument("--sdk", default=os.environ.get("QNN_SDK_ROOT"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--serial")
    parser.add_argument("--qp", type=int, choices=range(8), default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.sdk or min(args.warmup, args.repeats) < 1:
        parser.error("provide --sdk/QNN_SDK_ROOT and positive warmup/repeats")
    assets, sdk = Path(args.assets).resolve(), Path(args.sdk).resolve()
    manifest = json.loads((assets / "manifest.json").read_text())
    files = [assets / "qp_controls.bin", *[assets / f"entropy_qp{q}.bin" for q in range(8)]]
    files += [assets / f"{kind}.bin" for kind in ("encoder", "decoder")]
    libraries = [sdk / "lib/aarch64-android" / name for name in
                 ("libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV79Stub.so")]
    libraries += [sdk / "lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"]
    optional = sdk / "lib/aarch64-android/libQnnHtpPrepare.so"
    if optional.is_file():
        libraries.append(optional)
    if not args.dry_run:
        for path in files:
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
                raise ValueError(f"asset checksum mismatch: {path.name}")
        for path in [Path(args.binary), *libraries]:
            if not path.is_file():
                raise FileNotFoundError(path)
    remote = "/data/local/tmp/pulse_release/" + manifest["fingerprint"][:16]
    adb = ["adb"] + (["-s", args.serial] if args.serial else [])
    commands = [
        adb + ["shell", f"mkdir -p {remote}/lib"],
        adb + ["push", str(Path(args.binary).resolve()), remote + "/pulse_mobile"],
        *[adb + ["push", str(path), remote + "/" + path.name] for path in files],
        *[adb + ["push", str(path), remote + "/lib/" + path.name] for path in libraries],
    ]
    command = [remote + "/pulse_mobile", args.action, "--assets", remote,
               "--lib", remote + "/lib", "--input", remote + "/input",
               "--output", remote + "/result", "--qp", str(args.qp),
               "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
    adsp = remote + "/lib;/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp"
    shell = (f"chmod +x {remote}/pulse_mobile && "
             f"LD_LIBRARY_PATH={remote}/lib:/vendor/lib64 ADSP_LIBRARY_PATH={shlex.quote(adsp)} "
             + shlex.join(command))
    if args.dry_run:
        commands += [adb + ["push", "<prepared RGB8 or .pulse>", remote + "/input"],
                     adb + ["shell", shell], adb + ["pull", remote + "/result", args.output]]
        print(json.dumps(commands, indent=2))
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pulse-mobile-") as temp:
        temp = Path(temp)
        if args.action == "decompress":
            data = Path(args.input).read_bytes()
            metadata, _ = bitstream.unpack(data)
            if (metadata["model_id"] != bytes.fromhex(manifest["fingerprint"])[:16]
                    or metadata.get("entropy_sections") != [4, 3, 3]):
                raise ValueError("input is not a matching mobile .pulse stream")
        else:
            data = prepare_rgb(args.input, manifest["height"], manifest["width"])
        (temp / "input").write_bytes(data)
        commands += [adb + ["push", str(temp / "input"), remote + "/input"],
                     adb + ["shell", shell], adb + ["pull", remote + "/result", str(temp / "result")]]
        for command in commands:
            subprocess.run(command, check=True)
        data = (temp / "result").read_bytes()
        if args.action == "decompress":
            pixels = np.frombuffer(data, dtype="<f2").reshape(1, 3, manifest["height"], manifest["width"])
            if not np.isfinite(pixels).all():
                raise FloatingPointError("non-finite reconstruction")
            pixels = np.rint((np.clip(pixels.astype(np.float32), -1, 1) + 1) * 127.5).astype(np.uint8)
            Image.fromarray(pixels[0].transpose(1, 2, 0)).save(output)
        else:
            if args.action == "compress":
                bitstream.unpack(data)
            else:
                json.loads(data)
            output.write_bytes(data)
    print(output)


if __name__ == "__main__":
    main()
