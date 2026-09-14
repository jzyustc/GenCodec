#!/usr/bin/env python3
"""Prepare a training-only HR manifest and offline face/text ROI JSONL.

Run from PULSE with --help. Detection is CPU-only; images are never modified.
The detector/recognition/filter settings follow the source ROI preparation
pipeline. Output JSONL uses the public training loader's faces/texts schema.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random


HOLDOUT_PARTS = {"val", "valid", "validation", "test", "testing", "kodak",
                 "kodak_ori", "kodak512", "tecnick"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def training_images(manifest, root, exclude_list=None):
    """Fail on missing training files; omit known holdouts before opening them."""
    root = Path(root).resolve()
    excluded_paths = set()
    if exclude_list:
        excluded_paths = {(root / line).resolve() for line in read_manifest(exclude_list)}
    images, excluded, seen = [], [], set()
    for entry in read_manifest(manifest):
        path = (root / entry).resolve()
        parts = {part.lower() for part in Path(entry).parts}
        # Inspect the resolved path too, so symlinks cannot hide named holdouts.
        parts.update(part.lower() for part in path.parts)
        reason = None
        if parts & HOLDOUT_PARTS:
            reason = "validation/test or benchmark path"
        elif "clic" in parts and "train" not in parts:
            reason = "CLIC path without an explicit train split"
        elif path in excluded_paths:
            reason = "explicit exclusion list"
        if reason:
            excluded.append({"image": entry, "reason": reason})
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in seen:
            continue
        seen.add(path)
        # Public loader receives --data-root; preserve portable root-relative paths.
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"training image is outside --data-root: {path}") from exc
        images.append(relative)
    if not images:
        raise ValueError("no training images remain after exclusion")
    return images, excluded


def clamp_box(coords, width, height):
    x0, y0, x1, y1 = coords
    x0 = max(0, min(width - 1, round(float(x0))))
    y0 = max(0, min(height - 1, round(float(y0))))
    x1 = max(x0 + 1, min(width, round(float(x1))))
    y1 = max(y0 + 1, min(height, round(float(y1))))
    return [x0, y0, x1, y1]


def keep_box(kind, box, score, width, height, args):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    threshold = args.face_score if kind == "face" else args.text_score
    # Source detector records round score/area before the verification pass.
    if round(float(score), 6) < threshold:
        return False
    if round(w * h / (width * height), 8) > args.max_area_fraction:
        return False
    extent = max(w, h) if kind == "face" else min(w, h)
    canonical_side = extent * 1024 / min(width, height)
    return args.min_canonical_side <= canonical_side <= args.max_canonical_side


def ctc_evidence(log_probs):
    """Count nonblank CTC tokens and their geometric-mean confidence.

    OpenCV Zoo's CRNN_CH model outputs log probabilities, [T, 1, C].
    Text transcripts/character dictionaries are not needed for this filter.
    """
    import numpy as np
    probs = np.asarray(log_probs)
    if probs.ndim != 3 or probs.shape[1] != 1 or not np.isfinite(probs).all():
        raise ValueError(f"expected finite CRNN [T,1,C] log probabilities: {probs.shape}")
    probs = probs[:, 0, :]
    if (probs > 1e-4).any():
        raise ValueError("CRNN output is not log probabilities; use the documented CH model")
    classes = probs.argmax(axis=1)
    selected, previous = [], -1
    for i, token in enumerate(classes):
        if token != 0 and token != previous:
            selected.append(float(probs[i, token]))
        previous = token
    confidence = float(np.exp(np.mean(selected))) if selected else 0.0
    return len(selected), round(confidence, 6)


class Detectors:
    def __init__(self, args):
        import cv2
        import numpy as np
        self.cv2, self.np, self.args = cv2, np, args
        cv2.setNumThreads(args.threads)
        cv2.ocl.setUseOpenCL(False)
        self.face = cv2.FaceDetectorYN.create(
            str(args.face_model), "", (320, 320), 0.6, 0.3, 5000,
            cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        net = cv2.dnn.readNet(str(args.text_model))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.text = cv2.dnn_TextDetectionModel_DB(net)
        self.text.setBinaryThreshold(0.3)
        self.text.setPolygonThreshold(0.5)
        self.text.setUnclipRatio(1.5)
        self.text.setMaxCandidates(300)
        self.text.setInputSize((1024, 1024))
        self.text.setInputMean((123.675, 116.28, 103.53))
        self.text.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
        self.recognizer = cv2.dnn.readNet(str(args.recognition_model))
        self.recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def recognize(self, image, box):
        cv2 = self.cv2
        height, width = image.shape[:2]
        x0, y0, x1, y1 = box
        dx, dy = (x1 - x0) * 0.05, (y1 - y0) * 0.05
        crop = image[max(0, math.floor(y0 - dy)):min(height, math.ceil(y1 + dy)),
                     max(0, math.floor(x0 - dx)):min(width, math.ceil(x1 + dx))]
        blob = cv2.dnn.blobFromImage(crop, scalefactor=1 / 127.5,
                                     size=(100, 32), mean=127.5)
        self.recognizer.setInput(blob)
        return ctc_evidence(self.recognizer.forward())

    def annotate(self, path, relative):
        cv2, np, args = self.cv2, self.np, self.args
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        height, width = image.shape[:2]
        record = {"image": relative, "width": width, "height": height,
                  "faces": [], "texts": []}
        details = []
        scale = min(1.0, 1920 / max(width, height))
        face_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self.face.setInputSize(face_size)
        _, faces = self.face.detect(cv2.resize(image, face_size, interpolation=cv2.INTER_AREA))
        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4] / scale
                if min(w, h) < 6:
                    continue
                box = clamp_box([x, y, x + w, y + h], width, height)
                if keep_box("face", box, face[-1], width, height, args):
                    details.append({"type": "face", "bbox": box,
                                    "score": round(float(face[-1]), 6)})
        polygons, scores = self.text.detect(
            cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA))
        for polygon, score in zip(polygons, scores):
            points = np.asarray(polygon, dtype=np.float32)
            x0, y0 = points.min(axis=0) * [width / 1024, height / 1024]
            x1, y1 = points.max(axis=0) * [width / 1024, height / 1024]
            if min(x1 - x0, y1 - y0) < 6:
                continue
            box = clamp_box([x0, y0, x1, y1], width, height)
            if not keep_box("text", box, score, width, height, args):
                continue
            length, confidence = self.recognize(image, box)
            if length >= args.text_min_length and confidence >= args.text_min_confidence:
                details.append({"type": "text", "bbox": box, "score": round(float(score), 6),
                                "recognition_length": length, "recognition_confidence": confidence})
        details.sort(key=lambda item: (item["type"], -item["score"], item["bbox"]))
        for item in details:
            record["faces" if item["type"] == "face" else "texts"].append(item["bbox"])
        return record, {"image": relative, "boxes": details}


def mixed_records(records, fraction, seed):
    roi = [r for r in records if r["faces"] or r["texts"]]
    count = round(len(records) * fraction)
    if count and not roi:
        raise ValueError("no verified ROI images; inspect detector inputs/thresholds")
    rng = random.Random(seed)
    mixed = ([rng.choice(roi) for _ in range(count)]
             + [rng.choice(records) for _ in range(len(records) - count)])
    rng.shuffle(mixed)
    return roi, mixed


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, required=True, help="one image path per line")
    ap.add_argument("--data-root", type=Path, required=True, help="root for input/output image paths")
    ap.add_argument("--output-dir", type=Path, required=True, help="new or empty output directory")
    ap.add_argument("--exclude-list", type=Path, help="additional holdout paths, relative to data-root")
    ap.add_argument("--prepare-only", action="store_true", help="write filtered HR list without detection")
    ap.add_argument("--face-model", type=Path)
    ap.add_argument("--text-model", type=Path)
    ap.add_argument("--recognition-model", type=Path)
    ap.add_argument("--threads", type=int, default=1, help="OpenCV CPU threads; no GPU is used")
    ap.add_argument("--face-score", type=float, default=0.75)
    ap.add_argument("--text-score", type=float, default=0.85)
    ap.add_argument("--text-min-confidence", type=float, default=0.35)
    ap.add_argument("--text-min-length", type=int, default=2)
    ap.add_argument("--min-canonical-side", type=float, default=8)
    ap.add_argument("--max-canonical-side", type=float, default=96)
    ap.add_argument("--max-area-fraction", type=float, default=0.10)
    ap.add_argument("--mixed-roi-fraction", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=20260807)
    args = ap.parse_args(argv)
    if args.threads < 1 or args.text_min_length < 1:
        ap.error("threads and text-min-length must be positive")
    if not 0 < args.min_canonical_side <= args.max_canonical_side:
        ap.error("invalid canonical-side bounds")
    for key in ("face_score", "text_score", "text_min_confidence",
                "max_area_fraction", "mixed_roi_fraction"):
        if not 0 <= getattr(args, key) <= 1:
            ap.error(f"{key} must be in [0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error("output-dir must be new or empty; existing artifacts are never overwritten")
    images, excluded = training_images(args.metadata, args.data_root, args.exclude_list)
    models = {}
    detector = None
    if not args.prepare_only:
        for key in ("face_model", "text_model", "recognition_model"):
            path = getattr(args, key)
            if path is None or not path.is_file():
                ap.error(f"--{key.replace('_', '-')} must name an existing ONNX model")
            models[key] = {"name": path.name, "sha256": sha256(path)}
        detector = Detectors(args)
    records, evidence = [], []
    for i, image in enumerate(images):
        if detector:
            record, details = detector.annotate(args.data_root / image, image)
            records.append(record)
            evidence.append(details)
            if (i + 1) % 100 == 0 or i + 1 == len(images):
                print(f"annotated {i + 1}/{len(images)}", flush=True)
    if detector:
        roi, mixed = mixed_records(records, args.mixed_roi_fraction, args.seed)
    # Write only after all images have succeeded; retain original images unchanged.
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "hr_train.txt").write_text("".join(p + "\n" for p in images), encoding="utf-8")
    write_jsonl(out / "excluded.jsonl", excluded)
    summary = {"training_images": len(images), "excluded_entries": len(excluded),
               "source_directories": dict(sorted(Counter(str(Path(p).parent) for p in images).items()))}
    if detector:
        write_jsonl(out / "roi_all.jsonl", records)
        write_jsonl(out / "roi_only.jsonl", roi)
        write_jsonl(out / "roi_mixed.jsonl", mixed)
        write_jsonl(out / "roi_evidence.jsonl", evidence)
        summary.update(roi_images=len(roi), mixed_records=len(mixed),
                       face_boxes=sum(len(r["faces"]) for r in records),
                       text_boxes=sum(len(r["texts"]) for r in records))
    manifest = {"summary": summary, "settings": vars(args),
                "input_list_sha256": sha256(args.metadata), "models": models,
                "exclude_list_sha256": sha256(args.exclude_list) if args.exclude_list else None,
                "script_sha256": sha256(__file__),
                "opencv": detector.cv2.__version__ if detector else None,
                "numpy": detector.np.__version__ if detector else None,
                "outputs": {p.name: sha256(p) for p in sorted(out.iterdir())}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
