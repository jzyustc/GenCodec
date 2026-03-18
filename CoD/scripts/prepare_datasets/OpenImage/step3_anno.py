"""Step 3: Generate annotation files listing resized images.

Scans train{size}_X directories and writes paths relative to data_dir.

Usage:
    python step3_anno.py --data_dir /data --dataset OpenImage --size 256 \
        --output_dir /data/anno_per_dataset
"""

import os
import glob
import argparse

HEX_SPLITS = [str(i) for i in range(10)] + list("abcdef")


def main():
    parser = argparse.ArgumentParser(description="Generate OpenImage annotation file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base data directory")
    parser.add_argument("--dataset", type=str, default="OpenImage", help="Dataset subdirectory name")
    parser.add_argument("--size", type=int, required=True, help="Short-edge size (256 or 512)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for annotation file")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    all_entries = []

    for split in HEX_SPLITS:
        subdir = f"train{args.size}_{split}"
        full_dir = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(full_dir):
            continue
        images = sorted(glob.glob(os.path.join(full_dir, "*.jpg")))
        entries = [os.path.join(args.dataset, subdir, os.path.basename(p)) for p in images]
        all_entries += entries
        print(f"[{subdir}] {len(entries)} images")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.size}_anno.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(all_entries))

    print(f"Total: {len(all_entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
