"""Step 3: Generate annotation file by scanning resized image directory.

Writes one relative path per line (relative to data_dir).

Usage:
    python step3_anno.py --image_dir ./images_256 \
        --anno_prefix ImageNet-21K/images_256 \
        --output /data/anno_per_dataset/ImageNet-21K_256_anno.txt
"""

import os
import argparse


EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description="Generate ImageNet-21K annotation file.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root directory of resized images")
    parser.add_argument("--anno_prefix", type=str, default="",
                        help="Prefix for paths in output annotation")
    parser.add_argument("--output", type=str, required=True,
                        help="Output annotation file path")
    args = parser.parse_args()

    entries = []
    for dirpath, _, filenames in os.walk(args.image_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in EXTS:
                rel_path = os.path.relpath(os.path.join(dirpath, fname), start=args.image_dir)
                if args.anno_prefix:
                    entries.append(os.path.join(args.anno_prefix, rel_path))
                else:
                    entries.append(rel_path)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    entries.sort()
    with open(args.output, "w") as f:
        f.write("\n".join(entries))

    print(f"Annotation saved to {args.output}: {len(entries)} entries")


if __name__ == "__main__":
    main()
