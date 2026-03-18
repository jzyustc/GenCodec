"""Step 2: Resize images so the short edge equals the target size.

Scans the source image directory, skips images whose short edge is already
<= the target size, and saves resized images to the output directory.

Usage:
    python step2_resize.py --image_dir ./images --output_dir ./images_256 --size 256
"""

import os
import argparse
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resize_and_save(args_tuple):
    rel_path, image_dir, output_dir, size = args_tuple
    src_path = os.path.join(image_dir, rel_path)
    dst_path = os.path.join(output_dir, rel_path)
    try:
        with Image.open(src_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            short_edge = min(w, h)
            if short_edge <= size:
                return None
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            scale = size / short_edge
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(dst_path, quality=95)
        return rel_path
    except Exception as e:
        print(f"Failed: {src_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Resize ImageNet-21K images.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Source image root directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for resized images")
    parser.add_argument("--size", type=int, default=256,
                        help="Target short-edge size (default: 256)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    all_paths = [
        os.path.relpath(os.path.join(dirpath, fname), start=args.image_dir)
        for dirpath, _, filenames in os.walk(args.image_dir)
        for fname in filenames
        if os.path.splitext(fname)[1].lower() in EXTS
    ]
    print(f"Found {len(all_paths)} images to process.")

    workers = args.workers or os.cpu_count()
    tasks = [(p, args.image_dir, args.output_dir, args.size) for p in all_paths]

    success_paths = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(resize_and_save, t) for t in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resizing"):
            result = future.result()
            if result is not None:
                success_paths.append(result)

    print(f"Successfully resized {len(success_paths)}/{len(all_paths)} images "
          f"(skipped {len(all_paths) - len(success_paths)} with short edge <= {args.size}).")


if __name__ == "__main__":
    main()
