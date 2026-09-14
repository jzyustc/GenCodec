# Deployment and timing

Always use the matching `model.pt`, `config.json`, and `entropy_control_int.pt`.
The neural transforms may run on accelerators; integer CDF prediction and
rANS coding remain on the CPU. Floating-point pixel equality across backends
is not required for bit-exact entropy transport.

All encoders default to calibrated part-specific y CDFs, skip-index cutoff 2,
and Meta Prior Index Merge (including small images; no fixed-width selection).
The same defaults apply to the standard CLI, AMD/H100 runtimes and mobile
exports. Calibrate older Gaussian-only bundles before encoding. Index Merge
uses a fitted initialization when available, otherwise the defined P=0.5
initialization. Decoders use the settings carried by each file.

## Platforms

Run the commands in each guide from the repository root.

| Target | Directory and guide | Timing scope |
|---|---|---|
| AMD EPYC 9V84, one thread | [cpu_amd_epyc_9v84_single_thread](../deployment/cpu_amd_epyc_9v84_single_thread/README.md) | Full codec, OpenVINO BF16 sender + AVX-512 BF16 receiver |
| NVIDIA H100 | [gpu_nvidia_h100](../deployment/gpu_nvidia_h100/README.md) | Full codec, compiled BF16 CUDA graphs + multi-lane CPU entropy coding |
| REDMI K80 Pro NPU | [npu_qualcomm_snapdragon_8_elite](../deployment/npu_qualcomm_snapdragon_8_elite/README.md) | Whole-image QNN FP16 + ARM I8MM + z4/y3/y3 entropy coding; device validation pending |

CPU and H100 use dedicated optimized runtimes with shared entropy transport.
They time complete compression and decompression, including Meta Prior
selection/decoding, integer Linear CDF prediction, rANS, the `.pulse` envelope,
and device transfers within the codec. CUDA is synchronized at timing boundaries.
Disk IO, model loading, graph conversion, input creation and warmup are excluded.

Use `--input image.png` for content-dependent measurements, otherwise the same
CPU-seeded synthetic RGB input is used on each backend. JSON output reports
mean/median/p90, all samples, actual file bpp, input/stream hashes, thread counts,
CPU affinity, entropy partitioning, versions, and timing scope. Encoding and
decoding alternate in the measured loop; do not compare this directly with
separate encode-only/decode-only loops without accounting for cache effects.

## Scope

- Timings depend on model, image content, hardware and runtime versions.
  Compare identical inputs, QPs and timing boundaries.
- Single-lane `.pulse` files retain the `PLS1` format. Parallel entropy uses
  `PLS2`, adding one checked lane-count byte before the native image header.
  Both carry the same Meta Prior and integer-CDF syntax. No external lane
  configuration is required; use the updated decoder for `PLS2` files.
- Whole-image split streams use `PLS3`, with three lane-count bytes and
  independent z/y0/y1 rANS sections. Mobile uses 4/3/3 lanes partitioned over
  retained symbols. H100 uses 16/32/32 lanes partitioned over dense positions,
  marked by the high bits of both y lane bytes. The standard decoder reads
  both profiles; H100 files additionally require the H100 CPU entropy extension
  (no GPU required). The optimized AMD reader accepts combined `PLS1`/`PLS2`;
  the optimized H100 reader also accepts its dense `PLS3` profile.
