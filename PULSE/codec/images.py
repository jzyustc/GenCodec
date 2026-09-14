"""RGB image IO. Encoders consume [0,1]; decoder outputs are in [-1,1]."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def image_paths(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"no supported images in {path}")
    return paths


def read_image(path, device="cpu"):
    with Image.open(path) as image:
        image = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255


def pad_image(image):
    h, w = image.shape[-2:]
    return F.pad(image, (0, (-w) % 64, 0, (-h) % 64), mode="replicate")


def to_uint8(reconstruction):
    return ((reconstruction.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def save_image(reconstruction, path):
    pixels = to_uint8(reconstruction)[0].permute(1, 2, 0).cpu().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
