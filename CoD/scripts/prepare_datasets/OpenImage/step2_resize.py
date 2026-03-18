"""Step 2: Resize OpenImage images so the short edge equals the target size.

Processes subdirectories train_0 .. train_f (16 hex-named splits).

Usage:
    python step2_resize.py --image_dir /path/to/OpenImage \
        --output_dir /path/to/OpenImage --size 256
"""

import os
import glob
import argparse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

HEX_SPLITS = [str(i) for i in range(10)] + list("abcdef")


def resize_and_save(args_tuple):
    src_path, dst_path, size = args_tuple
    try:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with Image.open(src_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            short_edge = min(w, h)
            if short_edge == 0:
                raise ValueError("Image has a zero dimension.")
            scale = size / short_edge
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(dst_path, quality=95)
        return dst_path
    except Exception as e:
        print(f"Failed: {src_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Resize OpenImage images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Root OpenImage directory (contains train_X dirs)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for resized images")
    parser.add_argument("--size", type=int, default=256, help="Target short-edge size (default: 256)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count()
    total_success = 0

    for split in HEX_SPLITS:
        src_dir = os.path.join(args.image_dir, f"train_{split}")
        dst_dir = os.path.join(args.output_dir, f"train{args.size}_{split}")

        img_paths = glob.glob(os.path.join(src_dir, "*.jpg"))
        print(f"[{split}] Found {len(img_paths)} images.")

        tasks = [(p, os.path.join(dst_dir, os.path.basename(p)), args.size) for p in img_paths]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(resize_and_save, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Resizing train_{split}"):
                if f.result() is not None:
                    total_success += 1

        print(f"[{split}] Done.\n")

    print(f"Successfully resized {total_success} images in total.")


if __name__ == "__main__":
    main()
