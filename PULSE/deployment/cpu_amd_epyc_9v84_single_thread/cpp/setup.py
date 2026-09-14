# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Build one size-specialized extension from shared AVX-512 kernels."""
import argparse
from pathlib import Path
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--size", choices=["s", "base", "l"], required=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
root = Path(__file__).resolve().parent
variant = root / args.size
sources = {p.name: p for p in (root / "common").glob("*.cpp")}
sources.update({p.name: p for p in variant.glob("*.cpp")})
module = {"s": "pulse_cpu_amd_pulse_s_decoder_ext",
          "base": "pulse_cpu_amd_decoder_ext",
          "l": "pulse_cpu_amd_pulse_l_decoder_ext"}[args.size]

setup(
    name=module,
    ext_modules=[CppExtension(
        name=module, sources=[str(sources[n]) for n in sorted(sources)],
        include_dirs=[str(variant), str(root / "common")],
        extra_compile_args={"cxx": ["-O3", "-std=c++17", "-fopenmp", "-march=native",
                                     "-Wno-deprecated-declarations"]})],
    cmdclass={"build_ext": BuildExtension},
    options={"build_ext": {"build_temp": str(root / "build" / f"temp_{args.size}")}},
)
