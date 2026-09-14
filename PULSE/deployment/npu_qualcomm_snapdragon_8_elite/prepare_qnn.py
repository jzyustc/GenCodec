"""Convert both whole-image graphs to FP16 HTP contexts with shared-buffer IO."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--sdk", default=os.environ.get("QNN_SDK_ROOT"))
    parser.add_argument("--sdk-python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.sdk:
        parser.error("set QNN_SDK_ROOT or provide --sdk")
    sdk, out = Path(args.sdk).resolve(), Path(args.assets).resolve()
    manifest = json.loads((out / "manifest.json").read_text())
    commands = []
    for kind in ("encoder", "decoder"):
        commands.extend([
            [args.sdk_python, str(sdk / "bin/x86_64-linux-clang/qairt-converter"),
             "--input_network", str(out / f"{kind}.onnx"), "--output_path", str(out / f"{kind}.dlc"),
             "--float_bitwidth", "16", "--target_backend", "HTP"],
            [str(sdk / "bin/x86_64-linux-clang/qnn-context-binary-generator"),
             "--backend", str(sdk / "lib/x86_64-linux-clang/libQnnHtp.so"),
             "--dlc_path", str(out / f"{kind}.dlc"), "--output_dir", str(out),
             "--binary_file", kind, "--config_file", str(out / "backend_extensions.json"),
             "--input_output_tensor_mem_type", "memhandle", "--profiling_level", "basic",
             "--log_level", "error"],
        ])
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return
    for name, digest in manifest["files"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"asset checksum mismatch: {name}")
    (out / "htp_device.json").write_text(json.dumps(
        {"devices": [{"soc_model": 69, "dsp_arch": "v79", "pd_session": "unsigned"}]}, indent=2))
    (out / "backend_extensions.json").write_text(json.dumps(
        {"backend_extensions": {
            "shared_library_path": str(sdk / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"),
            "config_file_path": str(out / "htp_device.json")}}, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(sdk / "lib/python") + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = str(sdk / "lib/x86_64-linux-clang") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    for index, command in enumerate(commands):
        with (out / f"conversion_{index}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for kind in ("encoder", "decoder"):
        path = out / f"{kind}.bin"
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["qnn"] = dict(soc_model=69, dsp_arch="v79", precision="FP16", io_memory="memhandle")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
