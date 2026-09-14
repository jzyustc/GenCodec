# Training

Run the commands below from `PULSE/`. Install the training dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

Use an image directory, a text list (one image path per line), or ROI JSONL.
Relative paths resolve against `--data-root`, or the list's directory if omitted.
Images are decoded as RGB.

### Data sources by stage

| Training stage | Argument | Data source |
|---|---|---|
| MSE LR / Perceptual Stage I LR | `--train-data` | Open Images training images and SAM-1B, prepared for 512 × 512 training |
| MSE HR / Perceptual Stage I HR | `--hr-data` | Flickr2K, DIV2K training, and CLIC training |
| Perceptual Stage II / ROI | `--roi-data` | The same HR pool, with face/text annotations and ROI-biased sampling |

**LR (512 × 512).** Follow the **512-resolution portion** of the
[CoD preparation pipeline](https://github.com/microsoft/GenCodec/blob/main/CoD/scripts/prepare_datasets/README.md)
and use `anno_512/hq_anno.txt`. Images are resized to a 512-pixel short edge,
then center-cropped to 512 × 512 during training. The 256-resolution pool,
including ImageNet-21K, is not used.

**HR (1024 × 1024).** Following [AEIC](https://github.com/LuizScarlet/AEIC),
we use the sources below, **excluding CLIC validation/test images to prevent
data leakage**. LSDIR is not included.

| Source | Training images |
|---|---:|
| Flickr2K | 2,650 |
| DIV2K training | 800 |
| CLIC training | 585 |
| **Total** | **4,035** |

Removing 41 CLIC validation and 60 test images from the original 4,136-image
list leaves 4,035 training images. HR training uses native-resolution random
crops; only images smaller than the crop are resized.

Use [`tools/prepare_roi_data.py`](../tools/prepare_roi_data.py) to filter a local
HR path list before training:

```bash
# Run from PULSE/. Entries in the input list are relative to /path/to/data.
python tools/prepare_roi_data.py \
  --metadata /path/to/data/high_res/anno/anno_aeic_stage2.txt \
  --data-root /path/to/data \
  --output-dir data/hr_clean \
  --prepare-only
```

The script writes `hr_train.txt`, `excluded.jsonl`, and `manifest.json`.
It excludes validation/test and Kodak/Tecnick paths, and requires an explicit
`train` component for CLIC. Add `--exclude-list path/to/heldout.txt` for other
held-out files.

**No evaluation data:** filtering checks paths, not image content. Also remove
renamed/copied evaluation images from training, ROI preparation, Meta Prior
fitting, and integer calibration.

### Preparing face/text ROI annotations

Annotate the HR images offline; crops and horizontal flips are applied jointly
to images and boxes during training. Original images are left unchanged.

**1. Install dependencies and download annotation models.**

```bash
pip install -r requirements-data.txt
```

Download these ONNX models from **OpenCV Zoo**, following their READMEs and
licenses. They are annotation models, not the training loss's feature teachers.

- [YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet):
  `face_detection_yunet_2023mar.onnx`.
- [PP-OCRv3 text detector](https://github.com/opencv/opencv_zoo/tree/main/models/text_detection_ppocr):
  `text_detection_en_ppocrv3_2023may.onnx`.
- [CRNN text recognizer](https://github.com/opencv/opencv_zoo/tree/main/models/text_recognition_crnn):
  `text_recognition_CRNN_CH_2021sep.onnx` (**CH** version).

**2. Generate annotations.**

```bash
python tools/prepare_roi_data.py \
  --metadata data/hr_clean/hr_train.txt \
  --data-root /path/to/data \
  --face-model /path/to/models/face_detection_yunet_2023mar.onnx \
  --text-model /path/to/models/text_detection_en_ppocrv3_2023may.onnx \
  --recognition-model /path/to/models/text_recognition_CRNN_CH_2021sep.onnx \
  --output-dir data/hr_roi \
  --threads 1
```

Default filters:

| Filter | Threshold |
|---|---|
| Face / text detection score | ≥ 0.75 / ≥ 0.85 |
| CRNN text verification | Confidence ≥ 0.35; ≥ 2 nonblank collapsed CTC tokens |
| Box area | ≤ 10% of the image |
| Box extent at a 1024-pixel image short edge | 8–96 pixels: face long side, text short side |

The script runs on CPU, records model hashes and package versions, and requires
a new or empty output directory. See `--help` for adjustable thresholds.

Outputs:

| File | Contents |
|---|---|
| `roi_all.jsonl` | All HR images, including those without ROIs |
| `roi_only.jsonl` | Images with verified face/text boxes |
| `roi_mixed.jsonl` | Training list: 75% draws from ROI images, 25% from all HR images |
| `roi_evidence.jsonl` | Detection and recognition evidence |
| `hr_train.txt`, `excluded.jsonl`, `manifest.json` | Image lists, settings, counts, and hashes |

The mixed list has the same length as the HR pool and is sampled with replacement
using a fixed seed. Its whole-pool draws may also contain ROIs. This image-sampling
ratio is separate from `roi_crop_prob=0.75`, which controls crop placement.

**3. Use the annotations for training.**

Pass `roi_mixed.jsonl` as `--roi-data`. Boxes use half-open `[x0,y0,x1,y1]`
coordinates in the original image:

```json
{"image":"high_res/CLIC/train/example.png","faces":[[30,20,110,130]],"texts":[[10,150,230,190]]}
```

Keep the original images and use a common `--data-root` for all three lists
(or use absolute image paths). Prepare feature teachers as described under
**Perceptual** below.

```bash
python train_perceptual.py --size base \
  --train-data /path/to/data/anno_512/hq_anno.txt \
  --hr-data data/hr_clean/hr_train.txt \
  --roi-data data/hr_roi/roi_mixed.jsonl \
  --data-root /path/to/data \
  --face-teacher teachers/face.pt --ocr-teacher teachers/ocr.pt \
  --output runs/base-perceptual
```

## Recipes

Recipes are in `configs/mse/{s,base}.yaml` and
`configs/perceptual/{base,l}.yaml`. They define model sizes, losses, optimizers,
schedules, and crops; fixed architecture choices live in the code.
Batch sizes are **global**; use `--micro-batch-size` for gradient accumulation.

Check a recipe without loading data or starting training:

```bash
python train_mse.py --size s --check-config
python train_perceptual.py --size l --check-config
```

| Objective/stage | Crop | Steps | Batch | Lambda |
|---|---:|---:|---:|---:|
| MSE LR | 512 | 500k | 8 | 8–80, weighted MSE |
| MSE HR | 1024 | 50k | 8 | 8–80, weighted MSE |
| Perceptual I LR | 512 | 300k | 16 | 0.5–12.5 |
| Perceptual I HR | 1024 | 50k | 16 | 0.5–12.5 |
| Perceptual II / ROI | 1024 | 20k | 32 | 3–25 |

- **MSE:** SOAP with `mse_weight = 5 × (lambda / sqrt(8 × 80))^0.4435`.
  The effective MSE coefficient is approximately 24.005–666.515; raw lambda
  and its weight are logged separately.
- **Perceptual Stage I:** SOAP with MSE and `LPIPS-VGG + 0.5 × LPIPS-Alex`.
- **Stage II:** scales LPIPS by 0.25 and adds a latent-conditioned DINOv2
  discriminator, ROI pixel/perceptual losses, and FaceNet/CRNN feature losses.
  ROI LPIPS has an additional multiplier of 0.125. Generator SOAP uses zero
  weight decay; discriminator Adam uses `1e-4`, with hinge loss on local logits.
  **20k alternating steps means 10k generator and 10k discriminator updates.**

## MSE

```bash
python train_mse.py --size s --train-data data/train.txt \
  --hr-data data/train_hr.txt --output runs/s-mse

# Base model: --size base
```

## Perceptual

Stage II requires frozen face/OCR feature teachers. Export them once:

```bash
pip install -e ".[teachers]"
python tools/export_teachers.py face --output teachers/face.pt

# Download a CRNN ONNX model from the official OpenCV Zoo first.
python tools/export_teachers.py ocr --onnx teachers/text_recognition_CRNN_CH_2021sep.onnx \
  --output teachers/ocr.pt
```

Use OpenCV Zoo's [CRNN **CH** model](https://github.com/opencv/opencv_zoo/tree/main/models/text_recognition_crnn)
with BGR `[-1,1]` input. The exporter takes a `[N,C,1,T]` feature before the
recurrent head; override `--feature-node` for a different graph.
The EN model is supported but is not the reference teacher.
Teacher weights are not redistributed; follow their upstream licenses.

```bash
python train_perceptual.py --size base \
  --train-data data/train.txt --hr-data data/train_hr.txt \
  --roi-data data/train_roi.jsonl \
  --face-teacher teachers/face.pt --ocr-teacher teachers/ocr.pt \
  --output runs/base-perceptual
```

Options:

- `--size l`: train PULSE-L.
- `--init runs/base-mse/model`: initialize the base model from MSE weights.
- `--dino-source /path/to/dinov2`: use an offline DINOv2 checkout.
- `--stop-after-stage perceptual_hr`: run Stage I without ROI teachers.

## Multi-GPU and resume

Use `torchrun` for multi-GPU training and `--resume` to restore a training state:

```bash
torchrun --standalone --nproc_per_node=2 train_mse.py --size base \
  --train-data data/train.txt --hr-data data/train_hr.txt \
  --micro-batch-size 2 --output runs/base-mse

python train_mse.py --size base --train-data data/train.txt \
  --hr-data data/train_hr.txt --output runs/base-mse \
  --resume runs/base-mse/training-state.pt
```

Resume restores model, optimizer, and stage state, but does not guarantee
bit-identical data-loader replay. For a smoke test only, use
`--max-steps 2 --crop-size 64 --workers 0`.

<details>
<summary>Runtime defaults</summary>

- BF16 autocast; `medium` FP32 matmul precision for SOAP.
- 16 persistent loader workers, prefetch factor 8, and no dropped final batch.
- Each stage restores the seeded post-model-initialization random state before
  constructing losses.
- Surrogate gradients are selected automatically: rounded-affine hard sigmoid
  for LR MSE, input-domain surrogate for HR/perceptual. Forward inference is unchanged.

</details>

## Postprocessing

After MSE or perceptual training, fit Meta Prior and calibrate the integer
Linear CDF and y CDFs using **training images only**:

```bash
python postprocess.py --model runs/base-perceptual/model \
  --data data/train_hr.txt --output checkpoints/my-pulse-perceptual
```

The network stays frozen. `--output` must be a new directory; the input is
never overwritten. The script validates the bundle and an all-eight-QP
training-image roundtrip before publishing:

```text
checkpoints/my-pulse-perceptual/
  config.json
  model.pt
  entropy_control_int.pt
  SHA256SUMS
```

`entropy_control_int.pt` includes the calibrated y CDFs. Failed runs do not
publish partial bundles. Use the output with `compress.py`, `decompress.py`,
and `evaluate.py`.

### Default calibration recipe

Settings are in `configs/postprocess.yaml`; use `--config` for another recipe.

- **Meta Prior:** 64 banks per QP, 4,035 sampled training images, 512 crops,
  500 complexity-based initialization steps, and six 300-step hard-EM refinements.
  Unused banks retain their parameters. Use `--num-images` to change the sample count.
- **Integer export:** INT8 linear projection, INT32 accumulation, and fixed
  integer CDF tables. Index Merge probability fitting is enabled by default;
  inference always uses Index Merge.
- **y CDF:** calibrated separately for both checkerboard parts and 64 scale
  buckets, using QPs 0/2/5/7, seed 26090604, and 65,536 Gaussian prior counts.
  Image selection follows manifest order:

| Models | First training images | Crop |
|---|---:|---:|
| S/base MSE | 64 | 1024 |
| Base/L perceptual | 128 | 512 |

All four models use calibrated y CDFs and skip-index cutoff 2. Calibration
changes entropy probabilities, not CDF support or bypass coding.

### Recalibrating an existing bundle

To update only y CDFs, without retraining or repeating Meta Prior/integer export:

```bash
python -m codec.training.y_cdf --model checkpoints/my-pulse-perceptual \
  --data data/train_hr.txt --device cuda:0
```

This **updates the existing bundle in place** (`entropy_control_int.pt` and
`SHA256SUMS`), preserving the network, skip policy, Meta Prior, and Index Merge data.
