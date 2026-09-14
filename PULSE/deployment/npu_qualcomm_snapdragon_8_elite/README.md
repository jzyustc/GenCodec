# Snapdragon 8 Elite: whole-image codec

REDMI K80 Pro / SM8750 / HTP v79. The complete image is processed by one
QNN FP16 encoder/receiver graph, with ARM I8MM integer Linear CDF prediction,
Meta Prior (including Index Merge), and independent z4/y3/y3 rANS sections.
**There is no spatial tiling.**

Run host commands from the repository root. All four released models and
QP 0–7 are supported. The default graph shape is **1920×1088**.
Android hardware execution must be checked on the target phone; host validation
does not substitute for an NDK build or a QNN device run.

## 1. Export matched model assets

```bash
pip install -r requirements-deploy.txt

python -m deployment.npu_qualcomm_snapdragon_8_elite.export_onnx \
  --model checkpoints/pulse-perceptual --output outputs/mobile
```

This writes whole-image `encoder.onnx` / `decoder.onnx`, eight entropy bundles,
runtime QP controls, reference tensors and `manifest.json`.
The receiver graph includes hyperlatent and checkerboard reconstruction, not
only the pixel renderer. Hard-GELU is weight-folded into Hardswish for QNN.
Use `--height` / `--width` (multiples of 64) for a different fixed graph shape.

## 2. Build QNN contexts

Install Qualcomm AI Runtime **2.40** and its supported Python environment.
Use the same SDK for conversion, compilation headers and device libraries.

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
export ANDROID_NDK_ROOT=/path/to/android-ndk

python -m deployment.npu_qualcomm_snapdragon_8_elite.prepare_qnn \
  --assets outputs/mobile --sdk "$QNN_SDK_ROOT" \
  --sdk-python /path/to/qairt-venv/bin/python
```

Both graphs use FP16 HTP and registered shared-buffer IO. The helper produces
`encoder.bin` / `decoder.bin` and records their hashes in the manifest.
`--dry-run` prints commands without invoking the SDK.

## 3. Build the Android executable

Use the Linux x86-64 Android NDK toolchain:

```bash
python -m deployment.npu_qualcomm_snapdragon_8_elite.build --target android \
  --ndk "$ANDROID_NDK_ROOT" --sdk "$QNN_SDK_ROOT" --output outputs/mobile_build
```

This builds `pulse_mobile` for arm64-v8a, API 29+, with I8MM/FP16 enabled.
The QNN runner retains registered RPC buffers and reuses contexts between frames.
No APK, camera, display or network-streaming application is required.

## 4. Benchmark on the phone

Enable USB debugging and verify the device with `adb devices`.
Use a 1920×1080 or 1920×1088 input for the default graph.

```bash
python -m deployment.npu_qualcomm_snapdragon_8_elite.run --action benchmark \
  --assets outputs/mobile --binary outputs/mobile_build/pulse_mobile \
  --sdk "$QNN_SDK_ROOT" --input image.png --qp 3 \
  --warmup 3 --repeats 20 --output outputs/mobile_timing.json
```

The helper verifies asset hashes, pushes the executable, matched model assets
and SDK libraries, then runs the **complete codec** on the phone.
Use `--serial DEVICE_SERIAL` to select a device or `--dry-run` to inspect commands.

Timings are measured inside the native process:
- Encode: RGB8 in memory → FP16 preparation → QNN → integer CDF / Meta Prior /
  parallel rANS → checked `.pulse` in memory.
- Decode: `.pulse` in memory → parallel entropy decoding → packed FP16 inputs →
  QNN → FP16 RGB in memory.

ADB transfers, file IO, model/context loading, warmup, display and final
correctness checks are excluded. Repeated streams and finite RGB are checked.
Smaller inputs are edge-padded on the host; no resizing is performed. Decoded
images retain the exported padded shape.

## 5. Separate compression and decompression

```bash
python -m deployment.npu_qualcomm_snapdragon_8_elite.run --action compress \
  --assets outputs/mobile --binary outputs/mobile_build/pulse_mobile \
  --sdk "$QNN_SDK_ROOT" --input image.png --qp 3 --output outputs/image.pulse

python -m deployment.npu_qualcomm_snapdragon_8_elite.run --action decompress \
  --assets outputs/mobile --binary outputs/mobile_build/pulse_mobile \
  --sdk "$QNN_SDK_ROOT" --input outputs/image.pulse --output outputs/phone.png

# Independent host verification of the phone's bitstream.
python decompress.py --model checkpoints/pulse-perceptual \
  --input outputs/image.pulse --output outputs/host.png --device cpu
```

The `.pulse` `PLS3` envelope identifies the model, checks data integrity and
records the three entropy lane counts. The updated standard decoder accepts
these streams without a device-specific flag. FP16/FP32 reconstructed pixels
need not be bit-identical.

To create a phone-compatible stream on the host without QNN:

```bash
python -m deployment.npu_qualcomm_snapdragon_8_elite.build \
  --target host --output outputs/mobile_host

python -m deployment.npu_qualcomm_snapdragon_8_elite.compress \
  --model checkpoints/pulse-perceptual --assets outputs/mobile \
  --binary outputs/mobile_host/pulse_entropy \
  --input image.png --qp 3 --output outputs/host_encoded.pulse
```

The host build uses the scalar integer reference, not the phone's ARM I8MM kernel.
SDK headers/libraries and generated QNN binaries are not distributed in this repository.
