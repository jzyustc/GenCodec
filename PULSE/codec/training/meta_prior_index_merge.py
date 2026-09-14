"""Fit the small context model used by Meta-Prior Index Merge Coding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math

import numpy as np


PROBABILITY_TOTAL = 2048
DEFAULT_ADAPTATION_SHIFT = 5


def probability_count(bank_count: int) -> int:
    bank_count = int(bank_count)
    if not 1 < bank_count <= 256:
        raise ValueError("bank_count must be in [2, 256]")
    bits = int(math.ceil(math.log2(bank_count)))
    return 4 + 4 * ((1 << bits) - 1)


def _tree_node(level: int, prefix: int) -> int:
    return (1 << level) - 1 + prefix


def _add(counts: np.ndarray, index: int, bit: bool) -> None:
    counts[index, 1 if bit else 0] += 1


def accumulate_index_merge_counts(
    counts: np.ndarray,
    indexes_hw,
    *,
    bank_count: int,
) -> None:
    """Accumulate the exact causal contexts used by the runtime syntax."""
    indexes = np.asarray(indexes_hw, dtype=np.uint8)
    if indexes.ndim != 2:
        raise ValueError(f"expected an HW index map, got {indexes.shape}")
    expected = probability_count(bank_count)
    if counts.shape != (expected, 2):
        raise ValueError(
            f"count table must have shape {(expected, 2)}, got {counts.shape}"
        )
    if indexes.size and int(indexes.max()) >= int(bank_count):
        raise ValueError("index map contains a bank outside bank_count")

    bits = int(math.ceil(math.log2(bank_count)))
    height, width = indexes.shape
    for h in range(height):
        for w in range(width):
            value = int(indexes[h, w])
            merged = False
            if w > 0:
                left = int(indexes[h, w - 1])
                up_matches_left = h > 0 and int(indexes[h - 1, w]) == left
                same_left = value == left
                _add(counts, 1 if up_matches_left else 0, same_left)
                merged = same_left
            if not merged and h > 0:
                up = int(indexes[h - 1, w])
                distinct_left = w > 0 and int(indexes[h, w - 1]) != up
                if w == 0 or distinct_left:
                    same_up = value == up
                    _add(counts, 2 + (1 if distinct_left else 0), same_up)
                    merged = same_up
            if merged:
                continue

            prefix = 0
            for level in range(bits):
                bit_position = bits - 1 - level
                context = 0
                if w > 0:
                    context |= (int(indexes[h, w - 1]) >> bit_position) & 1
                if h > 0:
                    context |= (
                        ((int(indexes[h - 1, w]) >> bit_position) & 1) << 1
                    )
                node = _tree_node(level, prefix)
                bit = ((value >> bit_position) & 1) != 0
                _add(counts, 4 + node * 4 + context, bit)
                prefix = (prefix << 1) | int(bit)


def quantize_probabilities(counts: np.ndarray) -> np.ndarray:
    """Laplace-smoothed P(bin=0), quantized to the runtime Q11 domain."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim < 2 or counts.shape[-1] != 2:
        raise ValueError("counts must end with [zero, one]")
    zero = counts[..., 0]
    one = counts[..., 1]
    probability = (zero + 1.0) / (zero + one + 2.0)
    return np.clip(
        np.rint(probability * PROBABILITY_TOTAL),
        1,
        PROBABILITY_TOTAL - 1,
    ).astype(np.uint16)


def fit_index_merge_model(
    records_by_qp: Mapping[int, Iterable[np.ndarray]],
    *,
    qp_num: int,
    bank_count: int,
    adaptation_shift: int = DEFAULT_ADAPTATION_SHIFT,
) -> dict:
    """Fit one deterministic initialization table per QP."""
    if not 0 <= int(adaptation_shift) <= 15:
        raise ValueError("adaptation_shift must be in [0, 15]")
    context_count = probability_count(bank_count)
    counts = np.zeros((int(qp_num), context_count, 2), dtype=np.int64)
    record_counts = np.zeros(int(qp_num), dtype=np.int64)
    position_counts = np.zeros(int(qp_num), dtype=np.int64)
    for qp, records in records_by_qp.items():
        qp = int(qp)
        if not 0 <= qp < int(qp_num):
            raise ValueError(f"QP {qp} is outside [0, {int(qp_num) - 1}]")
        for indexes in records:
            indexes = np.asarray(indexes, dtype=np.uint8)
            accumulate_index_merge_counts(
                counts[qp],
                indexes,
                bank_count=bank_count,
            )
            record_counts[qp] += 1
            position_counts[qp] += indexes.size
    missing = np.flatnonzero(record_counts == 0)
    if missing.size:
        raise ValueError(
            "no Index Merge fitting records for QPs "
            + ", ".join(map(str, missing.tolist()))
        )
    return {
        "format_version": 1,
        "algorithm": "Meta-Prior Index Merge Coding",
        "syntax": (
            "same-left flag; conditional same-up flag; contextual "
            "binary-tree escape index"
        ),
        "probability_precision_bits": 11,
        "adaptation_shift": int(adaptation_shift),
        "bank_count": int(bank_count),
        "probabilities": quantize_probabilities(counts),
        "record_counts": record_counts.tolist(),
        "position_counts": position_counts.tolist(),
    }
