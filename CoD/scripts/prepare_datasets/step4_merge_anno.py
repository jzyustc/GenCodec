"""Merge per-dataset annotations into final training annotation files.

Combines annotations from OpenImage, ImageNet-21K, and SAM-1B into:
  - anno_256/anno.txt   (for 256x256 training)
  - anno_512/hq_anno.txt (for 512x512 training)

All paths are relative to data_dir.

Usage:
    python step4_merge_anno.py --data_dir /data
"""

import os
import argparse
from glob import glob


def collect_paths(pattern):
    """Read all annotation files matching pattern and return combined entries."""
    entries = []
    for fpath in sorted(glob(pattern)):
        with open(fpath, "r") as f:
            entries += [line.strip() for line in f if line.strip()]
    return entries


def main():
    parser = argparse.ArgumentParser(description="Merge per-dataset annotations into final training annotations.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base data directory")
    parser.add_argument("--anno_dir", type=str, default=None, help="Annotation input directory (default: <data_dir>/anno_per_dataset)")
    args = parser.parse_args()

    anno_dir = args.anno_dir or os.path.join(args.data_dir, "anno_per_dataset")

    # --- anno_256: OpenImage(256) + ImageNet-21K(256) + SAM-1B ---
    entries_256 = []

    oi_256 = os.path.join(anno_dir, "OpenImage_256_anno.txt")
    if os.path.exists(oi_256):
        with open(oi_256, "r") as f:
            entries_256 += [line.strip() for line in f if line.strip()]
        print(f"OpenImage 256: {len(entries_256)} entries")

    in21k_256 = os.path.join(anno_dir, "ImageNet-21K_256_anno.txt")
    if os.path.exists(in21k_256):
        before = len(entries_256)
        with open(in21k_256, "r") as f:
            entries_256 += [line.strip() for line in f if line.strip()]
        print(f"ImageNet-21K 256: {len(entries_256) - before} entries")

    sam_anno = os.path.join(anno_dir, "SAM-1B_256_anno.txt")
    if os.path.exists(sam_anno):
        before = len(entries_256)
        with open(sam_anno, "r") as f:
            entries_256 += [line.strip() for line in f if line.strip()]
        print(f"SAM-1B 256: {len(entries_256) - before} entries")

    # --- anno_512: OpenImage(512) + SAM-1B(512) ---
    entries_512 = []

    oi_512 = os.path.join(anno_dir, "OpenImage_512_anno.txt")
    if os.path.exists(oi_512):
        with open(oi_512, "r") as f:
            entries_512 += [line.strip() for line in f if line.strip()]
        print(f"OpenImage 512: {len(entries_512)} entries")

    sam_512 = os.path.join(anno_dir, "SAM-1B_512_anno.txt")
    if os.path.exists(sam_512):
        before = len(entries_512)
        with open(sam_512, "r") as f:
            entries_512 += [line.strip() for line in f if line.strip()]
        print(f"SAM-1B 512: {len(entries_512) - before} entries")

    # --- Write final annotations ---
    out_256_dir = os.path.join(args.data_dir, "anno_256")
    out_512_dir = os.path.join(args.data_dir, "anno_512")
    os.makedirs(out_256_dir, exist_ok=True)
    os.makedirs(out_512_dir, exist_ok=True)

    out_256 = os.path.join(out_256_dir, "anno.txt")
    with open(out_256, "w") as f:
        f.write("\n".join(entries_256))
    print(f"\nanno_256/anno.txt: {len(entries_256)} total entries")

    out_512 = os.path.join(out_512_dir, "hq_anno.txt")
    with open(out_512, "w") as f:
        f.write("\n".join(entries_512))
    print(f"anno_512/hq_anno.txt: {len(entries_512)} total entries")


if __name__ == "__main__":
    main()
