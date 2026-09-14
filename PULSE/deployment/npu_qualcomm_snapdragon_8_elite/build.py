"""Build the native host entropy checker or the Android full-codec executable."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["host", "android"], required=True)
    parser.add_argument("--output", default="outputs/mobile_build")
    parser.add_argument("--ndk", default=os.environ.get("ANDROID_NDK_ROOT"))
    parser.add_argument("--sdk", default=os.environ.get("QNN_SDK_ROOT"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    rans = here.parents[1] / "codec/models/entropy/extensions/cpp/py_rans"
    native = here / "native"
    out = Path(args.output).resolve()
    sources = [str(native / "main.cpp"), str(rans / "rans.cpp")]
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-DNO_CUDA_PINNED_MEMORY",
             "-pthread", "-I" + str(native), "-I" + str(rans)]
    if args.target == "android":
        if not args.ndk or not args.sdk:
            parser.error("Android requires --ndk/ANDROID_NDK_ROOT and --sdk/QNN_SDK_ROOT")
        toolchain = Path(args.ndk) / "toolchains/llvm/prebuilt/linux-x86_64/bin"
        compiler = str(toolchain / "aarch64-linux-android29-clang++")
        sdk = Path(args.sdk)
        flags += ["-march=armv8.6-a+i8mm+fp16", "-static-libstdc++",
                  "-I" + str(sdk / "include/QNN"),
                  "-I" + str(sdk / "examples/QNN/SampleApp/SampleAppSharedBuffer/src")]
        sources += [str(native / "qnn_runner.cpp"), str(native / "qp_controls.cpp")]
        libraries = ["-landroid", "-llog", "-ldl"]
        binary = out / "pulse_mobile"
    else:
        compiler = os.environ.get("CXX", "g++")
        flags += ["-DPULSE_HOST_ONLY"]
        libraries = []
        binary = out / "pulse_entropy"
    command = [compiler, *flags, *sources, *libraries, "-o", str(binary)]
    print(shlex.join(command))
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
