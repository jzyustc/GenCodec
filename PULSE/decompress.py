"""Decompress .pulse files without access to the original images."""
import argparse
import json
from pathlib import Path

from codec.checkpoint import load_model
from codec.images import save_image
from codec.runtime import configure_runtime, decompress_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help=".pulse file or directory")
    parser.add_argument("--output", required=True, help=".png file or directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allow-legacy", action="store_true", help="accept trusted native files without integrity/model checks")
    args = parser.parse_args()
    configure_runtime(args.device, args.threads)
    model = load_model(args.model, args.device)
    source, output = Path(args.input), Path(args.output)
    paths = [source] if source.is_file() else sorted(source.rglob("*.pulse"))
    if not paths:
        raise FileNotFoundError(f"no PULSE bitstreams in {source}")
    for path in paths:
        target = output if source.is_file() and output.suffix.lower() == ".png" else (
            output / (path.name if source.is_file() else path.relative_to(source))).with_suffix(".png")
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        image, elapsed = decompress_image(model, path.read_bytes(), allow_legacy=args.allow_legacy)
        save_image(image, target)
        print(json.dumps(dict(input=str(path), output=str(target), decode_ms=elapsed)))


if __name__ == "__main__":
    main()
