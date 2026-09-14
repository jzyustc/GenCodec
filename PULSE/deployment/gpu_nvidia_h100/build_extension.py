#!/usr/bin/env python3
"""Build the isolated high-throughput entropy extension in place."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "cpp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove the local build directory and extension before rebuilding",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="force recompilation even when setuptools considers outputs current",
    )
    args = parser.parse_args()

    if args.clean:
        shutil.rmtree(CPP_ROOT / "build", ignore_errors=True)
        for path in CPP_ROOT.glob("pulse_gpu_entropy_ext*.so"):
            path.unlink()
        for path in CPP_ROOT.glob("pulse_gpu_entropy_ext*.pyd"):
            path.unlink()

    command = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
    ]
    if args.force:
        command.append("--force")
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    subprocess.run(command, cwd=CPP_ROOT, env=environment, check=True)

    environment["PYTHONPATH"] = str(CPP_ROOT)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pulse_gpu_entropy_ext as ext; "
                "print(ext.__file__); "
                "print('max API smoke: ', hasattr(ext, 'MetaPriorSelector'))"
            ),
        ],
        cwd=CPP_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
