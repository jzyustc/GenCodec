"""Step 1: Extract ImageNet-21K archives.

First extracts the top-level winter21_whole.tar.gz to get per-class .tar files,
then extracts each per-class .tar into the images directory.

Usage:
    python step1_untar.py --archive ./winter21_whole.tar.gz --dst_dir ./ \
        --image_dir ./images
"""

import os
import sys
import argparse
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def extract_class_tar(tar_path, dst_dir):
    class_name = os.path.splitext(os.path.basename(tar_path))[0]
    out_dir = os.path.join(dst_dir, class_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=out_dir)
        return class_name, True
    except Exception as e:
        return class_name, str(e)


def main():
    parser = argparse.ArgumentParser(description="Extract ImageNet-21K archives.")
    parser.add_argument("--archive", type=str, required=True,
                        help="Path to winter21_whole.tar.gz")
    parser.add_argument("--dst_dir", type=str, required=True,
                        help="Destination directory for extracted winter21_whole/")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Output directory for per-class images")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for per-class extraction")
    args = parser.parse_args()

    # Step 0: Extract top-level archive
    tar_dir = os.path.join(args.dst_dir, "winter21_whole")
    if os.path.isdir(tar_dir):
        print(f"winter21_whole/ already exists at {tar_dir}, skipping top-level extraction.")
    else:
        if not os.path.isfile(args.archive):
            print(f"ERROR: Archive not found: {args.archive}")
            print("Download it from https://image-net.org/download.php")
            sys.exit(1)
        os.makedirs(args.dst_dir, exist_ok=True)
        print(f"Extracting {args.archive} -> {args.dst_dir} ...")
        with tarfile.open(args.archive, "r:gz") as tar:
            tar.extractall(path=args.dst_dir)
        print("Top-level extraction done.")

    # Step 1: Extract per-class tar files
    os.makedirs(args.image_dir, exist_ok=True)
    tar_files = [os.path.join(tar_dir, f) for f in os.listdir(tar_dir) if f.endswith(".tar")]
    print(f"Found {len(tar_files)} per-class tar files.")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_class_tar, p, args.image_dir) for p in tar_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            name, status = future.result()
            if status is not True:
                print(f"[FAILED] {name}: {status}")

    print("Extraction complete.")


if __name__ == "__main__":
    main()
