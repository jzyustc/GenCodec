"""Decode a .pulse file with the native AMD BF16 receiver."""
from deployment.common import run
from .runtime import Codec

if __name__ == "__main__":
    run("cpu_amd_epyc_9v84_single_thread", Codec, "decompress")
