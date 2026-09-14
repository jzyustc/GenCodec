"""H100 full-codec benchmark, including CPU entropy coding and device transfers."""
from deployment.common import run
from .runtime import Codec


def main():
    run("gpu_nvidia_h100", Codec, "benchmark")


if __name__ == "__main__":
    main()
