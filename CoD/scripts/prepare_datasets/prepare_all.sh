#!/bin/bash
# Prepare all training datasets: ImageNet-21K, OpenImage, SAM-1B.
#
# Prerequisites:
#   - ImageNet-21K: Download winter21_whole.tar.gz from https://image-net.org/download.php
#                   and place it at <DATA_DIR>/ImageNet-21K/winter21_whole.tar.gz
#   - OpenImage:    Place raw images in <DATA_DIR>/OpenImage/train_0/ .. train_f/
#   - SAM-1B:       Place tar_list.txt in <DATA_DIR>/SAM-1B/tar_list.txt
#
# Produces:
#   <DATA_DIR>/
#   ├── anno_256/anno.txt              # merged annotation (256 training)
#   ├── anno_512/hq_anno.txt           # merged annotation (512 training)
#   ├── ImageNet-21K/
#   │   └── images_256/                # resized to 256
#   ├── OpenImage/
#   │   ├── train256_0/ .. train256_f/ # resized to 256
#   │   └── train512_0/ .. train512_f/ # resized to 512
#   └── SAM-1B/
#       ├── sam_images_256_0/ .. sam_images_256_4/
#       └── sam_images_512_0/ .. sam_images_512_4/
#
# Usage:
#     bash prepare_all.sh <DATA_DIR>

set -e

# ============================================================
# Configuration
# ============================================================

if [ -z "$1" ]; then
    echo "Usage: bash prepare_all.sh <DATA_DIR>"
    exit 1
fi

DATA_DIR="$1"

# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANNO_INTERMEDIATE="${DATA_DIR}/anno_per_dataset"

run_step() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: Required file not found: $1"
        echo "  $2"
        exit 1
    fi
}

require_dir() {
    if [ ! -d "$1" ]; then
        echo "ERROR: Required directory not found: $1"
        echo "  $2"
        exit 1
    fi
}

# ============================================================
# Pre-flight checks
# ============================================================
echo "Data directory: ${DATA_DIR}"

# ============================================================
# 1. ImageNet-21K
# ============================================================
IN21K_DIR="${DATA_DIR}/ImageNet-21K"
IN21K_ARCHIVE="${IN21K_DIR}/winter21_whole.tar.gz"
IN21K_TAR_DIR="${IN21K_DIR}/winter21_whole"
IN21K_IMAGE_DIR="${IN21K_DIR}/images"
IN21K_IMAGE_256_DIR="${IN21K_DIR}/images_256"

run_step "ImageNet-21K: Step 1 — Extract archives"
require_file "${IN21K_ARCHIVE}" \
    "Download winter21_whole.tar.gz from https://image-net.org/download.php and place it at ${IN21K_ARCHIVE}"
python "${SCRIPT_DIR}/ImageNet-21K/step1_untar.py" \
    --archive "${IN21K_ARCHIVE}" \
    --dst_dir "${IN21K_DIR}" \
    --image_dir "${IN21K_IMAGE_DIR}"

run_step "ImageNet-21K: Step 2 — Resize to 256"
require_dir "${IN21K_IMAGE_DIR}" \
    "Step 1 should have created ${IN21K_IMAGE_DIR}"
python "${SCRIPT_DIR}/ImageNet-21K/step2_resize.py" \
    --image_dir "${IN21K_IMAGE_DIR}" \
    --output_dir "${IN21K_IMAGE_256_DIR}" \
    --size 256

run_step "ImageNet-21K: Step 3 — Generate annotation"
require_dir "${IN21K_IMAGE_256_DIR}" \
    "Step 2 should have created ${IN21K_IMAGE_256_DIR}"
python "${SCRIPT_DIR}/ImageNet-21K/step3_anno.py" \
    --image_dir "${IN21K_IMAGE_256_DIR}" \
    --anno_prefix "ImageNet-21K/images_256" \
    --output "${ANNO_INTERMEDIATE}/ImageNet-21K_256_anno.txt"

run_step "ImageNet-21K: Cleanup — Remove intermediate files"
rm -rf "${IN21K_ARCHIVE}" "${IN21K_TAR_DIR}" "${IN21K_IMAGE_DIR}"
echo "Removed: winter21_whole.tar.gz, winter21_whole/, images/"

# ============================================================
# 2. OpenImage
# ============================================================
OI_DIR="${DATA_DIR}/OpenImage"

require_dir "${OI_DIR}" \
    "Place OpenImage raw images in ${OI_DIR}/train_0/ .. train_f/"

run_step "OpenImage: Step 1 — List images per directory"
python "${SCRIPT_DIR}/OpenImage/step1_stat.py" \
    --image_dir "${OI_DIR}" \
    --output_dir "${ANNO_INTERMEDIATE}/OpenImage_lists"

run_step "OpenImage: Step 2a — Resize to 256"
python "${SCRIPT_DIR}/OpenImage/step2_resize.py" \
    --image_dir "${OI_DIR}" \
    --output_dir "${OI_DIR}" \
    --size 256

run_step "OpenImage: Step 2b — Resize to 512"
python "${SCRIPT_DIR}/OpenImage/step2_resize.py" \
    --image_dir "${OI_DIR}" \
    --output_dir "${OI_DIR}" \
    --size 512

run_step "OpenImage: Step 3a — Generate 256 annotation"
python "${SCRIPT_DIR}/OpenImage/step3_anno.py" \
    --data_dir "${DATA_DIR}" --dataset OpenImage --size 256 \
    --output_dir "${ANNO_INTERMEDIATE}"

run_step "OpenImage: Step 3b — Generate 512 annotation"
python "${SCRIPT_DIR}/OpenImage/step3_anno.py" \
    --data_dir "${DATA_DIR}" --dataset OpenImage --size 512 \
    --output_dir "${ANNO_INTERMEDIATE}"

run_step "OpenImage: Cleanup — Remove original images"
for SPLIT in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
    rm -rf "${OI_DIR}/train_${SPLIT}"
done
echo "Removed: train_0/ .. train_f/"

# ============================================================
# 3. SAM-1B
# ============================================================
SAM_DIR="${DATA_DIR}/SAM-1B"
SAM_TAR_LIST="${SAM_DIR}/tar_list.txt"

require_file "${SAM_TAR_LIST}" \
    "Place SAM-1B tar_list.txt at ${SAM_TAR_LIST}"

run_step "SAM-1B: Step 1 — Download and extract (all groups)"
for GROUP_ID in 0 1 2 3 4; do
    echo "--- Group ${GROUP_ID} ---"
    python "${SCRIPT_DIR}/SAM-1B/step1_download.py" \
        --group ${GROUP_ID} --num_groups 5 --workers 10 \
        --output_dir "${SAM_DIR}" \
        --tar_list "${SAM_TAR_LIST}"
done

run_step "SAM-1B: Step 2a — Resize to 256"
for GROUP_ID in 0 1 2 3 4; do
    require_dir "${SAM_DIR}/sam_images_${GROUP_ID}" \
        "Step 1 should have created ${SAM_DIR}/sam_images_${GROUP_ID}"
    python "${SCRIPT_DIR}/SAM-1B/step2_resize.py" \
        --image_dir "${SAM_DIR}/sam_images_${GROUP_ID}" \
        --output_dir "${SAM_DIR}/sam_images_256_${GROUP_ID}" \
        --size 256
done

run_step "SAM-1B: Step 2b — Resize to 512"
for GROUP_ID in 0 1 2 3 4; do
    python "${SCRIPT_DIR}/SAM-1B/step2_resize.py" \
        --image_dir "${SAM_DIR}/sam_images_${GROUP_ID}" \
        --output_dir "${SAM_DIR}/sam_images_512_${GROUP_ID}" \
        --size 512
done

run_step "SAM-1B: Step 3a — Generate 256 annotation"
python "${SCRIPT_DIR}/SAM-1B/step3_anno.py" \
    --data_dir "${DATA_DIR}" --size 256 --num_groups 5 \
    --output "${ANNO_INTERMEDIATE}/SAM-1B_256_anno.txt"

run_step "SAM-1B: Step 3b — Generate 512 annotation"
python "${SCRIPT_DIR}/SAM-1B/step3_anno.py" \
    --data_dir "${DATA_DIR}" --size 512 --num_groups 5 \
    --output "${ANNO_INTERMEDIATE}/SAM-1B_512_anno.txt"

run_step "SAM-1B: Cleanup — Remove original images and tar list"
for GROUP_ID in 0 1 2 3 4; do
    rm -rf "${SAM_DIR}/sam_images_${GROUP_ID}"
done
rm -f "${SAM_TAR_LIST}"
echo "Removed: sam_images_0/ .. sam_images_4/, tar_list.txt"

# ============================================================
# 4. Merge all dataset annotations
# ============================================================
run_step "Final: Merge all annotations"
python "${SCRIPT_DIR}/step4_merge_anno.py" \
    --data_dir "${DATA_DIR}" \
    --anno_dir "${ANNO_INTERMEDIATE}"

run_step "Final: Cleanup — Remove intermediate annotations"
rm -rf "${ANNO_INTERMEDIATE}"
echo "Removed: anno_per_dataset/"

echo ""
echo "================================================================"
echo "  Done. Final annotations:"
echo "    ${DATA_DIR}/anno_256/anno.txt"
echo "    ${DATA_DIR}/anno_512/hq_anno.txt"
echo "================================================================"
