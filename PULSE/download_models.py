"""Download model bundles from Hugging Face and verify their SHA256 manifests."""
import argparse
import json
from pathlib import Path

from codec.utils.model_io import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="zhaoyangjia/PULSE")
    parser.add_argument("--revision", default="main", help="pin a commit/tag for reproducibility")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--output", default="checkpoints")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    out = Path(args.output)
    manifest = hf_hub_download(args.repo, "manifest.json", revision=args.revision, local_dir=out)
    models = json.loads(Path(manifest).read_text())["models"]
    selected = list(models) if args.models == ["all"] else args.models
    unknown = set(selected) - models.keys()
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}; available: {list(models)}")
    for model in selected:
        for name, record in models[model]["files"].items():
            path = hf_hub_download(args.repo, f"{model}/{name}", revision=args.revision, local_dir=out)
            if sha256(path) != record["sha256"]:
                raise ValueError(f"checksum mismatch: {model}/{name}")
        print(out / model)


if __name__ == "__main__":
    main()
