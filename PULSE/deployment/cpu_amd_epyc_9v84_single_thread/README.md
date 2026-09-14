# AMD EPYC 9V84: single-thread full codec

OpenVINO BF16 sender → Meta Prior + integer Linear CDF + rANS →
native AVX-512 BF16 receiver. Run all commands from the repository root.
Requires Linux x86-64, AVX-512 BF16/VNNI, PyTorch and a C++17 compiler.

## Build

```bash
pip install -r requirements-deploy.txt
python -m deployment.cpu_amd_epyc_9v84_single_thread.build_extension --jobs 8
```

This builds the entropy extension and all three receiver sizes.
Use `--size s`, `base`, or `l` to build only the required receiver.

## Benchmark

```bash
python -m deployment.cpu_amd_epyc_9v84_single_thread.benchmark \
  --model checkpoints/pulse-mse --core 0 \
  --height 1088 --width 1920 --qp 3 --warmup 3 --repeats 20 \
  --output outputs/amd_cpu.json
```

The process is pinned to `--core`; choose an allowed, idle CPU core.
PyTorch, OpenMP/BLAS and OpenVINO compute threads are fixed to one.
CPU-encoded streams use one rANS lane.
Use `--input image.png` instead of synthetic RGB. All four model bundles are supported.

## Compress / decompress

```bash
python -m deployment.cpu_amd_epyc_9v84_single_thread.compress \
  --model checkpoints/pulse-mse --input image.png --output outputs/image.pulse \
  --qp 3 --core 0

python -m deployment.cpu_amd_epyc_9v84_single_thread.decompress \
  --model checkpoints/pulse-mse --input outputs/image.pulse \
  --output outputs/reconstruction.png --core 0
```

Decompression reads the shape and QP from the stream and does not need the original.
Encoder IR is prepared on first use and cached under `outputs/amd_ir`, keyed by
model identity, QP and padded shape. Use `--export-dir` to change this directory.
Image padding, entropy coding and rendering are included in the codec;
benchmark input preparation, compilation and warmup are excluded.

## Native code

```text
cpp/
  common/   shared AVX-512 primitives and wrappers
  s/        S-specific blocks, dispatch and fixed dimensions
  base/     base-specific receiver and dispatch
  l/        L-specific receiver, dispatch and fused renderer
entropy_cpp/
```

Shared kernels are compiled separately for each size using its fixed
`model_config.h`. S's plain blocks, base/L channel attention and L's wider
fused renderer retain their original specialized implementations.
There is no FP32 deployment fallback. See [timing scope](../../docs/deployment.md).
