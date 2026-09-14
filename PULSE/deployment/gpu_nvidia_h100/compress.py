"""Compress an image with the optimized H100 full codec."""
from deployment.common import run
from .runtime import Codec

if __name__ == "__main__":
    run("gpu_nvidia_h100", Codec, "compress")
