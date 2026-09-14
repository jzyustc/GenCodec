# NVIDIA H100: full codec

Compiled BF16 CUDA graphs → pinned compact-symbol transfers →
CPU integer Linear CDF + Meta Prior + multi-lane rANS → compiled BF16 receiver.
Run all commands from the repository root.
Requires CUDA-enabled PyTorch, Linux x86-64 and an AVX-512/VNNI-capable host CPU.

## Build

```bash
pip install -r requirements-deploy.txt
python -m deployment.gpu_nvidia_h100.build_extension
```

This builds the optimized CPU entropy extension. Neural graphs are compiled
by `torch.compile` on first use; do not include compilation in latency measurements.

## Benchmark

```bash
python -m deployment.gpu_nvidia_h100.benchmark --model checkpoints/pulse-mse \
  --device cuda:0 --height 1088 --width 1920 --qp 3 \
  --warmup 3 --repeats 20 --output outputs/h100.json
```

Use `--device cuda:1` for the second GPU, or `CUDA_VISIBLE_DEVICES=1`
with `--device cuda:0`. Avoid concurrent GPU/CPU workloads.
Use `--input image.png` for real content. All four model bundles are supported.

The implementation uses `torch.compile(..., mode="reduce-overhead")`, BF16
neural transforms, independent z/y0/y1 rANS streams with 16/32/32 lanes,
and 16-thread integer Linear CDF kernels. Both y parts are queued concurrently;
native dense-position skip coding and zero-copy decoded views avoid NumPy
filtering and scatter copies.
Pinned int16 hyperlatents and int8 checkerboard symbols reduce transfer volume.
Decoding overlaps symbol uploads with CPU entropy work. CUDA is synchronized
before returning each complete encoding/decoding result.

## Compress / decompress

```bash
python -m deployment.gpu_nvidia_h100.compress --model checkpoints/pulse-mse \
  --input image.png --output outputs/image.pulse --qp 3 --device cuda:0

python -m deployment.gpu_nvidia_h100.decompress --model checkpoints/pulse-mse \
  --input outputs/image.pulse --output outputs/reconstruction.png --device cuda:0
```

The `PLS3` stream records its three lane counts, dense skip profile, shape
and QP; no original image or external lane setting is required for decoding.
The standard `decompress.py` also accepts these files with the same model bundle
after building the H100 CPU entropy extension above (no GPU is needed to build
or run that extension). This runtime also reads combined `PLS1`/`PLS2` streams;
use the standard decoder for mobile retained-symbol `PLS3` streams.
Floating-point reconstructed pixels may differ between BF16 and FP32 backends.

There is no FP32 deployment fallback. See [timing scope](../../docs/deployment.md).
