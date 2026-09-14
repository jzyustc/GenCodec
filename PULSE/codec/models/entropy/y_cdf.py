"""Fixed part-specific y CDFs; no online adaptation or additional network.

Scale/skip indexes always remain in [0,63]. Map part1 to rows64..127 only
after selecting transported symbols. Keep Gaussian tables for old streams.
"""

import numpy as np

Y_CDF_MODES = ("gaussian", "part_specific")


def validate_part_cdf(cdf, lengths, gaussian_lengths):
    raw, lens = np.asarray(cdf), np.asarray(lengths)
    if raw.dtype.kind not in "iu" or lens.dtype.kind not in "iu":
        raise ValueError("y CDF and lengths must be integer arrays")
    if raw.ndim != 2 or raw.shape[0] != 128 or lens.shape != (128,):
        raise ValueError("part-specific y requires 128 CDF rows")
    if not np.array_equal(lens, np.tile(np.asarray(gaussian_lengths), 2)):
        raise ValueError(
            "part-specific y must preserve Gaussian support/bypass lengths"
        )
    for row, n in zip(raw, lens):
        if n < 3 or n > raw.shape[1]:
            raise ValueError("invalid y CDF length")
        if (
            row[0] != 0
            or row[n - 1] != 65536
            or np.any(np.diff(row[:n].astype(np.int64)) <= 0)
        ):
            raise ValueError("y CDF must have positive frequencies totaling65536")
    return np.ascontiguousarray(raw, dtype=np.int32), np.ascontiguousarray(
        lens, dtype=np.int32
    )


def resolve_y_cdf(codec, mode=None):
    from codec.models.entropy.entropy_coding import update_y_cdf

    mode = getattr(codec, "_y_cdf_mode", "gaussian") if mode is None else mode
    if mode not in Y_CDF_MODES:
        raise ValueError(f"unknown y CDF mode: {mode!r}")
    if mode == "part_specific":
        if not hasattr(codec, "_cached_y_part_cdf"):
            raise ValueError(
                "part-specific coding requires calibrated y CDFs; run postprocess.py "
                "or python -m codec.training.y_cdf on this bundle first"
            )
        cdf, lengths = codec._cached_y_part_cdf
    else:
        cdf, lengths = getattr(codec, "_cached_y_cdf", (None, None))
        if cdf is None:
            cdf, lengths = update_y_cdf()
    return mode, cdf, lengths


def part1_packed(packed, *, validate=True):
    """The raw low byte is a scale index; adding64 must not carry into symbol."""
    values = np.asarray(packed)
    if validate and (
        values.dtype != np.int16 or np.any((values.astype(np.int32) & 255) >= 64)
    ):
        raise ValueError("expected int16 packed symbols with raw scale indexes0..63")
    return np.ascontiguousarray(values.astype(np.int32) + 64, dtype=np.int16)


def part1_indexes(indexes, *, validate=True):
    indexes = np.asarray(indexes)
    if validate and (indexes.dtype != np.uint8 or np.any(indexes >= 64)):
        raise ValueError("expected raw uint8 scale indexes0..63")
    return np.ascontiguousarray(indexes + 64, dtype=np.uint8)
