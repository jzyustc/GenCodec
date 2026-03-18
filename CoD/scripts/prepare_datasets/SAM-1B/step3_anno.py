"""Step 3: Generate SAM-1B annotation file by scanning resized image directories.

Writes one relative path per line (relative to data_dir).

Usage:
    python step3_anno.py --data_dir /data --size 256 --num_groups 5 \
        --output /data/anno_per_dataset/SAM-1B_256_anno.txt
"""

import os
import argparse


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Generate SAM-1B annotation file.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Base data directory")
    parser.add_argument("--dataset", type=str, default="SAM-1B",
                        help="Dataset subdirectory name")
    parser.add_argument("--size", type=int, required=True,
                        help="Short-edge size (256 or 512)")
    parser.add_argument("--num_groups", type=int, default=5,
                        help="Number of groups (default: 5)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output annotation file path")
    args = parser.parse_args()

    all_entries = []
    for group in range(args.num_groups):
        subdir = f"sam_images_{args.size}_{group}"
        full_dir = os.path.join(args.data_dir, args.dataset, subdir)
        if not os.path.isdir(full_dir):
            print(f"[{subdir}] not found, skipping")
            continue
        count = 0
        for dirpath, _, filenames in os.walk(full_dir):
            for fname in sorted(filenames):
                if os.path.splitext(fname)[1].lower() in EXTS:
                    rel = os.path.relpath(os.path.join(dirpath, fname), start=full_dir)
                    all_entries.append(os.path.join(args.dataset, subdir, rel))
                    count += 1
        print(f"[{subdir}] {count} images")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_entries.sort()
    with open(args.output, "w") as f:
        f.write("\n".join(all_entries))

    print(f"Annotation saved to {args.output}: {len(all_entries)} entries")


if __name__ == "__main__":
    main()
