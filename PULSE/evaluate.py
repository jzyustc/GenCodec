"""Evaluate actual .pulse rates (including headers) and reconstruction quality."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from codec import bitstream
from codec.checkpoint import load_model
from codec.images import image_paths, read_image, to_uint8, save_image
from codec.runtime import compress_image, configure_runtime, decompress_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="JSON results")
    parser.add_argument("--qps", type=int, nargs="+", default=list(range(8)), choices=range(8))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--legacy", action="store_true", help="evaluate native headers instead of the checked envelope")
    parser.add_argument("--save-reconstructions", help="optional PNG directory")
    parser.add_argument("--perceptual-metrics", action="store_true", help="LPIPS-VGG and LPIPS-Alex")
    parser.add_argument("--dists", action="store_true")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--fid-patch", type=int, default=256, help="patch side (paper: 256 for CLIC)")
    parser.add_argument("--fid-batch", type=int, default=32)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    if args.save_reconstructions and source.is_dir():
        destination = Path(args.save_reconstructions).resolve()
        if destination.is_relative_to(source):
            parser.error("save reconstructions outside the input dataset")
    paths = image_paths(args.input)
    configure_runtime(args.device, args.threads)
    model = load_model(args.model, args.device)
    perceptual = {}
    if args.perceptual_metrics:
        import lpips
        perceptual = {name: lpips.LPIPS(net=name).to(args.device).eval()
                      for name in ("alex", "vgg")}
    dists = None
    if args.dists:
        from DISTS_pytorch import DISTS
        dists = DISTS().to(args.device).eval().requires_grad_(False)
    fid = None
    if args.fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        if args.fid_patch < 2 or args.fid_batch < 1:
            parser.error("--fid-patch must be >=2 and --fid-batch positive")
        fid = FrechetInceptionDistance(feature=2048, normalize=False,
                                      reset_real_features=False).to(args.device)

    def update_fid(image, real):
        import torch.nn.functional as F
        p = args.fid_patch
        image = image.float()
        h, w = image.shape[-2:]
        image = F.pad(image, (0, max(0, p - w), 0, max(0, p - h)), mode="replicate")
        for offset in (0, p // 2):
            cropped = image[..., offset:, offset:]
            if min(cropped.shape[-2:]) < p:
                continue
            patches = F.unfold(cropped, kernel_size=p, stride=p).transpose(1, 2).reshape(-1, 3, p, p)
            patches = patches.round().clamp(0, 255).to(torch.uint8)
            for chunk in patches.split(args.fid_batch):
                fid.update(chunk, real=real)
    rows, summaries = [], {}
    with torch.inference_mode():
        for qp_index, qp in enumerate(args.qps):
            if fid is not None and qp_index:
                fid.reset()
            qp_rows = []
            for path in paths:
                image = read_image(path, args.device)
                data, encode_ms = compress_image(model, image, qp, legacy=args.legacy)
                metadata, payload = bitstream.unpack(data, allow_legacy=args.legacy)
                if metadata["y_cdf_mode"] != "part_specific" or metadata["skip_index_cutoff"] != 2 or payload[:5] != b"MPRI\x02":
                    raise RuntimeError("evaluation did not use calibrated CDF, skip=2 and Index Merge")
                reconstruction, decode_ms = decompress_image(model, data, allow_legacy=args.legacy)
                quantized = to_uint8(reconstruction).float() / 255
                mse = (quantized - image).square().mean().item()
                row = dict(image=str(path), qp=qp, bytes=len(data),
                           pixels=image.shape[-2] * image.shape[-1],
                           bpp=8 * len(data) / (image.shape[-2] * image.shape[-1]),
                           psnr=-10 * torch.log10(torch.tensor(max(mse, 1e-12))).item(),
                           encode_ms=encode_ms, decode_ms=decode_ms)
                for name, metric in perceptual.items():
                    row["lpips_" + name] = float(metric(quantized * 2 - 1, image * 2 - 1).mean())
                if dists is not None:
                    row["dists"] = float(dists(quantized, image).mean())
                if fid is not None:
                    if qp_index == 0:
                        update_fid((image * 255).round(), True)
                    update_fid(to_uint8(reconstruction), False)
                if args.save_reconstructions:
                    relative = Path(path.name) if Path(args.input).is_file() else path.relative_to(args.input)
                    destination = Path(args.save_reconstructions) / f"qp{qp}" / relative.with_suffix(".png")
                    if destination.resolve() == path.resolve():
                        raise ValueError("refusing to overwrite the input image")
                    save_image(reconstruction, destination)
                rows.append(row)
                qp_rows.append(row)
                print(json.dumps(row), flush=True)
            summaries[str(qp)] = {
                key: statistics.mean(r[key] for r in qp_rows)
                for key in ("bpp", "psnr", "encode_ms", "decode_ms",
                            *["lpips_" + k for k in perceptual], *(["dists"] if dists is not None else []))
            }
            summaries[str(qp)]["pixel_weighted_bpp"] = (
                8 * sum(r["bytes"] for r in qp_rows) / sum(r["pixels"] for r in qp_rows))
            if fid is not None:
                summaries[str(qp)]["fid"] = float(fid.compute())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(model=str(args.model), rate="actual .pulse file bytes including header",
                                     y_cdf_mode="part_specific", skip_index_cutoff=2,
                                     meta_prior_coding="index_merge",
                                     container="native" if args.legacy else "PLS1",
                                     fid_patch=args.fid_patch if args.fid else None,
                                     fid_offsets="(0,0) and (p/2,p/2); stride p" if args.fid else None,
                                     summary=summaries, images=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
