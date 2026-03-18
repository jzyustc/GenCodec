"""List images per subdirectory and save per-directory file lists.

Usage:
    python stat.py --image_dir /path/to/OpenImage --output_dir ./lists
"""

import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="List images per OpenImage subdirectory.")
    parser.add_argument("--image_dir", type=str, required=True, help="Root OpenImage directory")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for per-dir lists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for item in sorted(os.listdir(args.image_dir)):
        sub_path = os.path.join(args.image_dir, item)
        if os.path.isdir(sub_path):
            images = sorted(os.listdir(sub_path))
            out_path = os.path.join(args.output_dir, f"{item}.txt")
            with open(out_path, "w") as f:
                f.write("\n".join(images))
            print(f"{item}: {len(images)} images")


if __name__ == "__main__":
    main()
