"""Decode a .pulse file with the compiled H100 BF16 receiver."""
from deployment.common import run
from .runtime import Codec

if __name__ == "__main__":
    run("gpu_nvidia_h100", Codec, "decompress")
