"""Single-thread AMD CPU full-codec benchmark."""
from deployment.common import run
from .runtime import Codec


def main():
    run("cpu_amd_epyc_9v84_single_thread", Codec, "benchmark")


if __name__ == "__main__":
    main()
