"""Build the shared AMD entropy kernels and size-specialized BF16 receivers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["s", "base", "l", "all"], default="all")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parent
    env = dict(os.environ, MAX_JOBS=str(args.jobs))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"],
                   cwd=root / "entropy_cpp", env=env, check=True)
    for size in ("s", "base", "l") if args.size == "all" else (args.size,):
        subprocess.run([sys.executable, "setup.py", "--size", size, "build_ext", "--inplace"],
                       cwd=root / "cpp", env=env, check=True)


if __name__ == "__main__":
    main()
