# Data Preparation

This directory contains scripts to download, preprocess, and organize all training datasets for CoD.

## Prerequisites

Before running, you need to obtain the raw data for each dataset:

| Dataset | Source | Place at |
|---------|--------|----------|
| **ImageNet-21K** | Download `winter21_whole.tar.gz` from [image-net.org](https://image-net.org/download.php) | `<DATA_DIR>/ImageNet-21K/winter21_whole.tar.gz` |
| **OpenImage** | Download train splits from [OpenImages](https://storage.googleapis.com/openimages/web/download_v7.html) | `<DATA_DIR>/OpenImage/train_0/` .. `train_f/` |
| **SAM-1B** | Get the tar URL list from [SAM](https://ai.meta.com/datasets/segment-anything/) | `<DATA_DIR>/SAM-1B/tar_list.txt` |

## Quick Start

```bash
# 1. Place raw data (see Prerequisites above)
# 2. Run
bash scripts/prepare_datasets/prepare_all.sh /path/to/data
```

## Output Structure

After running, `<DATA_DIR>` will contain:

```
<DATA_DIR>/
├── anno_256/
│   └── anno.txt                        # merged annotation for 256×256 training
├── anno_512/
│   └── hq_anno.txt                     # merged annotation for 512×512 training
│
├── ImageNet-21K/
│   └── images_256/                     # resized, short edge = 256
│
├── OpenImage/
│   ├── train256_0/ .. train256_f/      # resized, short edge = 256
│   └── train512_0/ .. train512_f/      # resized, short edge = 512
│
└── SAM-1B/
    ├── sam_images_256_0/ .. sam_images_256_4/  # resized, short edge = 256
    └── sam_images_512_0/ .. sam_images_512_4/  # resized, short edge = 512
```

Intermediate files (archives, original-resolution images, per-dataset annotations) are automatically cleaned up after each stage.

Annotation files list one relative path per line (relative to `<DATA_DIR>`):

```
ImageNet-21K/images_256/n01440764/n01440764_10026.JPEG
OpenImage/train256_0/02a0b26f3e94e09f.jpg
SAM-1B/sam_images_256_0/sa_12345.jpg
```

## Per-Dataset Pipeline

### ImageNet-21K

| Step | Script | Description |
|:----:|--------|-------------|
| 1 | `step1_untar.py` | Extract `winter21_whole.tar.gz` → `winter21_whole/`, then extract each per-class `.tar` → `images/` |
| 2 | `step2_resize.py` | Resize images with short edge > 256 → `images_256/` |
| 3 | `step3_anno.py` | Scan `images_256/`, write annotation with data_dir-relative paths |

### OpenImage

| Step | Script | Description |
|:----:|--------|-------------|
| 1 | `step1_stat.py` | List images per subdirectory (diagnostic) |
| 2a | `step2_resize.py --size 256` | Resize to 256 short edge → `train256_X/` |
| 2b | `step2_resize.py --size 512` | Resize to 512 short edge → `train512_X/` |
| 3a | `step3_anno.py --size 256` | Scan `train256_X/`, write annotation |
| 3b | `step3_anno.py --size 512` | Scan `train512_X/`, write annotation |

### SAM-1B

| Step | Script | Description |
|:----:|--------|-------------|
| 1 | `step1_download.py` | Download tar archives, extract images, delete tars (grouped, resumable) |
| 2a | `step2_resize.py --size 256` | Resize to 256 short edge → `sam_images_256_X/`, remove JSON metadata |
| 2b | `step2_resize.py --size 512` | Resize to 512 short edge → `sam_images_512_X/` |
| 3a | `step3_anno.py --size 256` | Scan `sam_images_256_X/`, write annotation |
| 3b | `step3_anno.py --size 512` | Scan `sam_images_512_X/`, write annotation |

### Final Merge

| Script | Description |
|--------|-------------|
| `step4_merge_anno.py` | Combine per-dataset annotations into `anno_256/anno.txt` and `anno_512/hq_anno.txt` |

## Running Individual Steps

Each script supports `--help` for full argument documentation. Example:

```bash
# Resize only OpenImage to 512
python scripts/prepare_datasets/OpenImage/step2_resize.py \
    --image_dir /data/OpenImage \
    --output_dir /data/OpenImage \
    --size 512

# Re-generate merged annotations
python scripts/prepare_datasets/step4_merge_anno.py \
    --data_dir /data \
    --anno_dir /data/anno_per_dataset
```

## Notes

- **Disk space**: ImageNet-21K and SAM-1B are very large. Ensure sufficient disk space before starting.
- **Parallelism**: Most scripts support `--workers` to control concurrency. Defaults to CPU count.
- **Resumability**: SAM-1B download is resumable — re-running skips completed files.
- **Idempotency**: `prepare_all.sh` checks for existing outputs and skips completed steps where possible.
