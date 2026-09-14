"""Build the portable CPU rANS and integer-CDF kernels; CUDA is not required."""
from pathlib import Path
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

source = Path("codec/models/entropy/extensions/cpp/py_rans")
flags = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]
setup(
    packages=find_packages(include=["codec*", "deployment*"]) + ["codec.recipes"],
    package_dir={"codec.recipes": "configs"},
    package_data={"codec.recipes": ["mse/*.yaml", "perceptual/*.yaml", "*.yaml"],
                  "deployment.cpu_amd_epyc_9v84_single_thread": ["README.md"],
                  "deployment.gpu_nvidia_h100": ["README.md"],
                  "deployment.npu_qualcomm_snapdragon_8_elite": ["README.md"]},
    ext_modules=[
        Pybind11Extension(
            "MLCodec_extensions_cpp",
            [str(p) for p in sorted(source.glob("*.cpp"))],
            define_macros=[("NO_CUDA_PINNED_MEMORY", "1")],
            extra_compile_args=flags,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
