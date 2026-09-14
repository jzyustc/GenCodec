"""Directory, text-list or JSONL image datasets with synchronized ROI crops."""
import json
from pathlib import Path
import random
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from codec.images import IMAGE_SUFFIXES


def read_records(source, root=None):
    source = Path(source)
    if source.is_dir():
        records = [{"image": str(p.resolve())} for p in sorted(source.rglob("*"))
                   if p.suffix.lower() in IMAGE_SUFFIXES]
    else:
        base = Path(root) if root else source.parent
        records = []
        for line in source.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line) if line.startswith("{") else {"image": line}
            path = Path(item["image"])
            item["image"] = str(path if path.is_absolute() else base / path)
            records.append(item)
    if not records:
        raise ValueError(f"no training images in {source}")
    for item in records:
        if not Path(item["image"]).is_file():
            raise FileNotFoundError(item["image"])
        for kind in ("faces", "texts"):
            for box in item.get(kind, []):
                if len(box) != 4 or not all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"invalid {kind} box in {item['image']}: {box}")
    return records


class ImageDataset(Dataset):
    def __init__(self, records, crop_size=512, flip=True, *, hflip=None,
                 random_crop=True, crop_from_native=True, roi_crop_prob=0.0,
                 roi_context_scale=1.0, min_mask_side=1.0):
        self.records, self.crop_size = records, crop_size
        self.flip = flip if hflip is None else hflip
        self.random_crop, self.crop_from_native = random_crop, crop_from_native
        self.roi_crop_prob, self.roi_context_scale = roi_crop_prob, roi_context_scale
        self.min_mask_side = min_mask_side
        if crop_size < 64 or crop_size % 64:
            raise ValueError("crop size must be a positive multiple of 64")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        original_w, original_h = image.size
        if self.random_crop and (not self.crop_from_native or min(image.size) < self.crop_size):
            if original_w <= original_h:
                shape = (self.crop_size, int(self.crop_size * original_h / original_w))
            else:
                shape = (int(self.crop_size * original_w / original_h), self.crop_size)
            image = image.resize(shape, Image.Resampling.BILINEAR)
        sx, sy = image.width / original_w, image.height / original_h
        candidates = [b for k in ("faces", "texts") for b in record.get(k, [])]
        if not self.random_crop:
            left, top = (image.width - self.crop_size) // 2, (image.height - self.crop_size) // 2
        elif self.roi_crop_prob and candidates and random.random() < self.roi_crop_prob:
            x0, y0, x1, y1 = random.choice(candidates)
            def origin(a, b, scale, maximum):
                a, b = a * scale, b * scale
                lo, hi = max(0, math.ceil(b - self.crop_size)), min(maximum, math.floor(a))
                if lo > hi:
                    lo = hi = round(min(max((a + b - self.crop_size) / 2, 0), maximum))
                return random.randint(lo, hi) if hi > lo else lo
            left = origin(x0, x1, sx, image.width - self.crop_size)
            top = origin(y0, y1, sy, image.height - self.crop_size)
        elif self.roi_crop_prob:
            left = random.randint(0, image.width - self.crop_size) if image.width > self.crop_size else 0
            top = random.randint(0, image.height - self.crop_size) if image.height > self.crop_size else 0
        elif image.height == self.crop_size and image.width == self.crop_size:
            top = left = 0
        else:
            top = torch.randint(0, image.height - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, image.width - self.crop_size + 1, (1,)).item()
        do_flip = self.flip and random.random() < 0.5
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        if do_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        metadata = {}
        mask = torch.zeros(1, self.crop_size, self.crop_size)
        for plural, singular in (("faces", "face"), ("texts", "text")):
            boxes = []
            for x0, y0, x1, y1 in record.get(plural, []):
                x0, x1 = x0 * sx - left, x1 * sx - left
                y0, y1 = y0 * sy - top, y1 * sy - top
                if min(self.crop_size, x1) <= max(0, x0) or min(self.crop_size, y1) <= max(0, y0):
                    continue
                if do_flip:
                    x0, x1 = self.crop_size - x1, self.crop_size - x0
                boxes.append([max(0, x0), max(0, y0), min(self.crop_size, x1), min(self.crop_size, y1)])
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                width = max((x1 - x0) * self.roi_context_scale, self.min_mask_side)
                height = max((y1 - y0) * self.roi_context_scale, self.min_mask_side)
                mask[:, max(0, math.floor(cy - height / 2)):min(self.crop_size, math.ceil(cy + height / 2)),
                     max(0, math.floor(cx - width / 2)):min(self.crop_size, math.ceil(cx + width / 2))] = 1
            metadata[singular + "_present"] = torch.tensor(bool(boxes))
            metadata[singular + "_box"] = torch.tensor(
                random.choice(boxes) if boxes else [0., 0., 0., 0.]
            )
        metadata["roi_mask"] = mask
        return image, metadata
