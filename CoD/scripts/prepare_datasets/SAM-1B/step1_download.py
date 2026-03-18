"""Step 1: Download, extract, and clean up SAM-1B tar archives.

Requires a tar_list.txt file with tab-separated (filename, url) entries.
Files are split into groups for parallel downloading across machines.

Usage:
    python step1_download.py --group 0 --output_dir /data/SAM-1B
"""

import os
import sys
import argparse
import tarfile
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_extract_delete(fname_url, group_id, output_dir, tmp_dir):
    fname, url = fname_url
    final_path = os.path.join(tmp_dir, fname)
    temp_path = final_path + ".part"
    extract_dir = os.path.join(output_dir, f"sam_images_{group_id}")

    # Download with resume support
    resume_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    if os.path.exists(final_path):
        print(f"[SKIP] {fname} already exists")
    else:
        try:
            headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0)) + resume_pos
                mode = "ab" if resume_pos > 0 else "wb"
                with open(temp_path, mode) as f, tqdm(
                    total=total_size, initial=resume_pos,
                    unit="B", unit_scale=True, desc=fname, leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            os.rename(temp_path, final_path)
            print(f"[DOWNLOADED] {fname}")
        except Exception as e:
            return f"[ERROR] {fname} download failed: {e}"

    # Extract
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(final_path, "r") as tar:
            tar.extractall(path=extract_dir)
        print(f"[EXTRACTED] {fname} -> {extract_dir}")
    except Exception as e:
        return f"[ERROR] {fname} extraction failed: {e}"

    # Delete tar to save space
    try:
        os.remove(final_path)
        print(f"[DELETED] {fname}")
    except Exception as e:
        return f"[ERROR] {fname} deletion failed: {e}"

    # Log completion
    log_file = os.path.join(output_dir, f"finished_{group_id}.txt")
    with open(log_file, "a") as f:
        f.write(fname + "\n")

    return f"[DONE] {fname}"


def main():
    parser = argparse.ArgumentParser(description="Download and extract SAM-1B archives.")
    parser.add_argument("--group", type=int, required=True, help="Group index (0-based)")
    parser.add_argument("--num_groups", type=int, default=5, help="Total number of groups (default: 5)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel download workers")
    parser.add_argument("--tar_list", type=str, default="tar_list.txt", help="Tab-separated file list (fname, url)")
    parser.add_argument("--output_dir", type=str, default=".", help="Base output directory for extracted images")
    args = parser.parse_args()

    if not (0 <= args.group < args.num_groups):
        print(f"Group index must be in [0, {args.num_groups - 1}]")
        sys.exit(1)

    tmp_dir = os.path.join(args.output_dir, "sam_data")
    os.makedirs(tmp_dir, exist_ok=True)

    # Load completed files
    finished_file = os.path.join(args.output_dir, f"finished_{args.group}.txt")
    finished_set = set()
    if os.path.exists(finished_file):
        with open(finished_file, "r") as f:
            finished_set = set(line.strip() for line in f if line.strip())

    # Load tar list (skip header line)
    with open(args.tar_list, "r") as f:
        lines = f.read().split("\n")[1:]
        links = [line.split("\t") for line in lines if line.strip()]

    # Split into groups
    total_files = len(links)
    group_size = total_files // args.num_groups
    start_idx = args.group * group_size
    end_idx = start_idx + group_size if args.group < args.num_groups - 1 else total_files
    links_subset = [link for link in links[start_idx:end_idx] if link[0] not in finished_set]

    print(f"Group {args.group}: {len(links_subset)} pending / {end_idx - start_idx} total files")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_extract_delete, link, args.group, args.output_dir, tmp_dir)
            for link in links_subset
        ]
        for future in as_completed(futures):
            print(future.result())

    print("Group complete.")


if __name__ == "__main__":
    main()
