"""Compress an image with the optimized AMD full codec."""
from deployment.common import run
from .runtime import Codec

if __name__ == "__main__":
    run("cpu_amd_epyc_9v84_single_thread", Codec, "compress")
