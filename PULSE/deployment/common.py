"""Shared bundle loading and commands for the optimized platform runtimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from codec import bitstream
from codec.checkpoint import load_model
from codec.config import SIZES
from codec.images import read_image, save_image


def load_bundle(directory):
    directory = Path(directory)
    validated = load_model(directory)
    if validated.release_config["architecture"] != SIZES[validated.release_config["size"]]:
        raise ValueError("optimized kernels require an unmodified S/base/L architecture")
    fingerprint, config = validated.bundle_fingerprint, validated.release_config
    del validated
    # Native neural kernels use the original floating weights, including the
    # scale module's state layout; integer entropy control is loaded separately.
    model = load_model(directory, entropy=False)
    artifact = torch.load(directory / "entropy_control_int.pt", map_location="cpu", weights_only=True)
    return model, artifact, fingerprint, config


def run(platform_name, runtime_class, action):
    parser = argparse.ArgumentParser(description=f"PULSE optimized {platform_name} {action}")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=action != "benchmark")
    parser.add_argument("--output", required=action != "benchmark")
    parser.add_argument("--qp", type=int, choices=range(8), default=3)
    if platform_name == "cpu_amd_epyc_9v84_single_thread":
        parser.add_argument("--core", type=int, default=min(os.sched_getaffinity(0)))
        parser.add_argument("--export-dir", default="outputs/amd_ir")
    else:
        parser.add_argument("--device", default="cuda:0")
    if action == "benchmark":
        parser.add_argument("--height", type=int, default=1088)
        parser.add_argument("--width", type=int, default=1920)
        parser.add_argument("--warmup", type=int, default=3)
        parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if platform_name == "cpu_amd_epyc_9v84_single_thread":
        if args.core not in os.sched_getaffinity(0):
            parser.error("--core is outside the allowed CPU affinity")
        os.sched_setaffinity(0, {args.core})
    torch.manual_seed(0)
    stream = None
    if action == "decompress":
        stream = Path(args.input).read_bytes()
        metadata, _ = bitstream.unpack(stream)
        height, width, qp = metadata["height"], metadata["width"], metadata["qp"]
        image = None
    else:
        if args.input:
            image = read_image(args.input)
        else:
            if min(args.height, args.width, args.warmup, args.repeats) < 1:
                parser.error("shape, warmup and repeats must be positive")
            image = torch.rand(1, 3, args.height, args.width)
        height, width, qp = *image.shape[-2:], args.qp
    kwargs = (dict(export_dir=args.export_dir) if platform_name == "cpu_amd_epyc_9v84_single_thread"
              else dict(device=args.device))
    if action == "decompress":
        kwargs["stream_metadata"] = metadata
    codec = runtime_class(args.model, height=height, width=width, qp=qp, **kwargs)
    if action == "compress":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_bytes(codec.compress(image))
        return
    if action == "decompress":
        save_image(codec.decompress(stream), args.output)
        return
    if min(args.warmup, args.repeats) < 1:
        parser.error("warmup and repeats must be positive")
    # Stage the input before timing; compact-symbol transfers remain timed.
    codec.set_image(image)
    encode, decode, hashes = [], [], []
    for i in range(args.warmup + args.repeats):
        start = time.perf_counter()
        stream = codec.compress()
        end_encode = time.perf_counter()
        output = codec.decompress(stream)
        end_decode = time.perf_counter()
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite reconstruction")
        if i >= args.warmup:
            encode.append(1000 * (end_encode - start))
            decode.append(1000 * (end_decode - end_encode))
            hashes.append(hashlib.sha256(stream).hexdigest())
    if len(set(hashes)) != 1:
        raise RuntimeError("repeated compression did not produce identical streams")
    metadata, payload = bitstream.unpack(stream)
    if metadata["y_cdf_mode"] != "part_specific" or metadata["skip_index_cutoff"] != 2 or payload[:5] != b"MPRI\x02":
        raise RuntimeError("benchmark did not use calibrated CDF, skip=2 and Index Merge")
    def summary(values):
        return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                    p90_ms=float(np.percentile(values, 90)), samples_ms=values)
    result = dict(model=str(args.model), backend=platform_name, precision="BF16",
                  shape=list(image.shape), qp=qp, warmup=args.warmup, repeats=args.repeats,
                  bytes=len(stream), bpp=len(stream) * 8 / (height * width),
                  encoding=summary(encode), decoding=summary(decode),
                  input=args.input or "synthetic RGB, CPU seed 0",
                  input_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                  stream_sha256=hashes[0], deterministic_stream=True,
                  y_cdf_mode=metadata["y_cdf_mode"], skip_index_cutoff=2,
                  meta_prior_coding="index_merge",
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  torch_threads=torch.get_num_threads(), python=platform.python_version(),
                  torch=torch.__version__, platform=platform.platform(),
                  entropy_lanes=codec.entropy.lanes, entropy_threads=codec.entropy.threads,
                  entropy_sections=getattr(codec.entropy, "sections", None),
                  entropy_partitioning=("dense_positions" if hasattr(codec.entropy, "sections")
                                        else "combined"),
                  timing_loop="alternating_encode_decode",
                  device="cpu" if platform_name == "cpu_amd_epyc_9v84_single_thread" else args.device,
                  timing_scope="full codec including .pulse envelope, Meta Prior, integer Linear CDF, rANS and compact-symbol transfers; excludes model loading, compilation, disk IO, initial image upload and warmup")
    if platform_name == "gpu_nvidia_h100":
        result["rans_cpu_affinity"] = codec.host_cpus
        result["accelerator"] = torch.cuda.get_device_name(codec.device)
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
