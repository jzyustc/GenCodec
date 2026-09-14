"""Compress one image or a directory. PNG/JPEG loading is outside codec timing."""
import argparse
import json
from pathlib import Path

from codec.checkpoint import load_model
from codec.images import image_paths, read_image
from codec.runtime import compress_image, configure_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="downloaded model bundle directory")
    parser.add_argument("--input", required=True, help="image or directory")
    parser.add_argument("--output", required=True, help=".pulse file or directory")
    parser.add_argument("--qp", type=int, default=3, choices=range(8))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--legacy", action="store_true", help="write unchecked native format")
    args = parser.parse_args()
    configure_runtime(args.device, args.threads)
    model = load_model(args.model, args.device)
    source, output = Path(args.input), Path(args.output)
    for path in image_paths(source):
        target = output if source.is_file() and output.suffix == ".pulse" else (
            output / (path.name if source.is_file() else path.relative_to(source))).with_suffix(".pulse")
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        image = read_image(path, args.device)
        data, elapsed = compress_image(model, image, args.qp, legacy=args.legacy)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(json.dumps(dict(input=str(path), output=str(target), qp=args.qp,
                              bytes=len(data), bpp=8 * len(data) / (image.shape[-2] * image.shape[-1]),
                              encode_ms=elapsed)))


if __name__ == "__main__":
    main()
