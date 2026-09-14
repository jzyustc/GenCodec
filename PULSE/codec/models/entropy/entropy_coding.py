"""
Real entropy coding wrappers for the vendored BitEstimator / GaussianEncoder.

Produces a real bitstream (bytes) using the rANS coder vendored at
`codec/models/entropy/extensions/cpp` (`MLCodec_extensions_cpp`). Mirrors the
DCVC API:

  - `BitEstimator.update(coder, qp_offset=0)` builds per-channel CDFs for the
    factorized z prior, registered at coder slot 0.
  - `GaussianEncoder.update(coder, skip_thres=0.0)` builds per-scale-bucket
    CDFs for the Gaussian y conditional, registered at coder slot 1.
  - Encode side transports z as int16 (tails use rANS bypass coding) and
    packs y as int16 with the bucket index in the low byte.
  - Decode side calls `RansDecoder.decode_z` / `decode_y` and reshapes back.

Pure-pytorch / numpy. No CUDA. The actual rANS step lives in the C++ ext.
"""
from __future__ import annotations

import math
import struct
from typing import Optional

import numpy as np
import torch

try:
    from MLCodec_extensions_cpp import (
        RansEncoder,
        RansDecoder,
        decode_meta_prior_index_merge as _decode_meta_prior_index_merge,
        encode_meta_prior_index_merge as _encode_meta_prior_index_merge,
        pack_checkerboard_positions as _pack_checkerboard_positions,
        pack_nchw_offsets as _pack_nchw_offsets,
        pmf_to_quantized_cdf as _pmf_to_quantized_cdf,
        restore_checkerboard_positions as _restore_checkerboard_positions,
        restore_checkerboard_positions_into as _restore_checkerboard_positions_into,
        restore_nchw_offsets_into as _restore_nchw_offsets_into,
        select_checkerboard_indexes_above as _select_checkerboard_indexes_above,
        select_indexes_above as _select_indexes_above,
    )
    _RANS_AVAILABLE = True
except ImportError:  # pragma: no cover
    RansEncoder = RansDecoder = None  # type: ignore
    _decode_meta_prior_index_merge = None  # type: ignore
    _encode_meta_prior_index_merge = None  # type: ignore
    _pmf_to_quantized_cdf = None  # type: ignore
    _pack_checkerboard_positions = None  # type: ignore
    _pack_nchw_offsets = None  # type: ignore
    _restore_checkerboard_positions = None  # type: ignore
    _restore_checkerboard_positions_into = None  # type: ignore
    _restore_nchw_offsets_into = None  # type: ignore
    _select_checkerboard_indexes_above = None  # type: ignore
    _select_indexes_above = None  # type: ignore
    _RANS_AVAILABLE = False


# Core CDF range. Values outside a per-CDF range are represented by the
# rANS coder's existing bypass path. y remains int8-packed and is clipped to
# this range; z is transported as int16 and must NOT be clipped here because
# z is also the input to the hyper decoder. The C++ extension stores
# `2*range+1` in int8, so the modeled CDF core must remain <= 63.
MAX_ENTROPY_CODING_VALUE = 63
Z_SYMBOL_MIN = np.iinfo(np.int16).min
Z_SYMBOL_MAX = np.iinfo(np.int16).max
META_PRIOR_MAGIC = b"MPRI"
META_PRIOR_FIXED_WIDTH_VERSION = 1
META_PRIOR_INDEX_MERGE_VERSION = 2
META_PRIOR_VERSION = META_PRIOR_INDEX_MERGE_VERSION
META_PRIOR_HEADER = struct.Struct("<4sBBH")
META_PRIOR_INDEX_MERGE_LENGTH = struct.Struct("<I")
META_PRIOR_INDEX_MERGE_ADAPTATION_SHIFT = 5
META_PRIOR_PROBABILITY_TOTAL = 2048


# -----------------------------------------------------------------------------
# CDF helpers
# -----------------------------------------------------------------------------

def require_entropy_extension():
    if not _RANS_AVAILABLE:
        raise RuntimeError(
            "PULSE entropy extension is missing or outdated (Index Merge support "
            "is required); run pip install -e . from the PULSE code directory"
        )


def _pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
    """Convert (channel, max_length) pmf rows into int32 CDF rows.

    pmf:        (N, max_length) — rows may be shorter than max_length, padded.
    tail_mass:  (N, 1)
    pmf_length: (N,) int — true length of each row (≤ max_length)
    max_length: int — row width
    Returns:    (N, max_length + 2) int32 CDFs (extra 2 = tail + sentinel).
    """
    require_entropy_extension()
    cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
    for i, p in enumerate(pmf):
        prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
        prob1 = _reorder_prob(prob)
        _cdf = _pmf_to_quantized_cdf(prob1.tolist())
        _cdf = torch.IntTensor(_cdf)
        cdf[i, : _cdf.size(0)] = _cdf
    return cdf


def _reorder_prob(prob):
    """Reorder centered pmf (..., -2, -1, 0, 1, 2, ..., tail) into the order the
    rANS coder expects: 0, 1, -1, 2, -2, ..., tail."""
    length = prob.size(0)
    out = prob.clone()
    center = (length - 2) // 2  # length includes tail; center is index of 0
    out[0] = prob[center]
    for i in range(1, center + 1):
        out[2 * i - 1] = prob[center + i]
        out[2 * i] = prob[center - i]
    return out


# -----------------------------------------------------------------------------
# z prior: per-channel factorized CDF
# -----------------------------------------------------------------------------

@torch.inference_mode()
def update_z_cdf(bit_estimator, qp: int = 0):
    """Build per-(qp×channel) CDF tables for the factorized z prior.

    Returns (quantized_cdf, cdf_length) ready to pass to
    `RansEncoder.set_cdf(quantized_cdf, cdf_length, 0)`.
    """
    from codec.models.entropy.entropy_models import bit_estimator_z_prob

    qp_num = bit_estimator.qp_num
    channel = bit_estimator.channel
    device = bit_estimator.h.device
    h, b, a = bit_estimator._select(qp)

    # Per-channel symmetric symbol range. The truncation test compares the
    # CDF (`bit_estimator_z_prob`), not the PMF (`forward` returns PMF).
    zeros = torch.zeros((qp_num, channel, 1, 1), device=device)
    sym_range = zeros + MAX_ENTROPY_CODING_VALUE
    for i in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
        neg = bit_estimator_z_prob(zeros - i, h, b, a)
        pos = bit_estimator_z_prob(zeros + i, h, b, a)
        sym_range = torch.where(
            torch.logical_and(neg < 1e-3, pos > 1 - 1e-3), float(i), sym_range
        )
    sym_range = sym_range.int()
    pmf_length = sym_range * 2 + 1

    max_length = MAX_ENTROPY_CODING_VALUE * 2 + 1
    samples = torch.arange(max_length, device=device)
    samples = samples[None, None, None, :] - sym_range  # (qp, ch, 1, max_length)

    upper = bit_estimator_z_prob(samples + 0.5, h, b, a)
    lower = bit_estimator_z_prob(samples - 0.5, h, b, a)
    pmf = (upper - lower)[:, :, 0, :]

    upper_at_range = bit_estimator_z_prob(sym_range.float(), h, b, a)[:, :, 0, -1:]
    lower_at_neg_range = bit_estimator_z_prob(-sym_range.float() - 0.5, h, b, a)[:, :, 0, :1]
    tail_mass = lower_at_neg_range + (1.0 - upper_at_range)

    pmf = pmf.reshape(-1, max_length).cpu()
    tail_mass = tail_mass.reshape(-1, 1).cpu()
    pmf_length_flat = pmf_length.reshape(-1).cpu()
    quantized_cdf = _pmf_to_cdf(pmf, tail_mass, pmf_length_flat, max_length)
    cdf_length = pmf_length_flat + 2
    return quantized_cdf.numpy(), cdf_length.int().numpy()


@torch.inference_mode()
def update_meta_prior_cdf(meta_prior, qp: int = 0):
    """Build the ``K × C`` integer CDF rows for one Meta Prior QP."""
    from codec.models.entropy.entropy_models import bit_estimator_z_prob

    if not bool(meta_prior.ready.item()):
        raise RuntimeError("Meta Prior has not been fitted")
    if not 0 <= int(qp) < meta_prior.qp_num:
        raise ValueError(f"qp must be in [0, {meta_prior.qp_num - 1}]")
    h = meta_prior.h[int(qp)]
    b = meta_prior.b[int(qp)]
    a = meta_prior.a[int(qp)]
    bank_count, channel, _ = h.shape
    device = h.device

    zeros = torch.zeros((bank_count, channel, 1, 1), device=device)
    sym_range = zeros + MAX_ENTROPY_CODING_VALUE
    for value in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
        neg = bit_estimator_z_prob(zeros - value, h, b, a)
        pos = bit_estimator_z_prob(zeros + value, h, b, a)
        sym_range = torch.where(
            torch.logical_and(neg < 1e-3, pos > 1 - 1e-3),
            float(value),
            sym_range,
        )
    sym_range = sym_range.int()
    pmf_length = sym_range * 2 + 1
    max_length = MAX_ENTROPY_CODING_VALUE * 2 + 1
    samples = torch.arange(max_length, device=device)
    samples = samples[None, None, None, :] - sym_range
    upper = bit_estimator_z_prob(samples + 0.5, h, b, a)
    lower = bit_estimator_z_prob(samples - 0.5, h, b, a)
    pmf = (upper - lower)[:, :, 0, :]
    upper_at_range = bit_estimator_z_prob(
        sym_range.float(),
        h,
        b,
        a,
    )[:, :, 0, -1:]
    lower_at_neg_range = bit_estimator_z_prob(
        -sym_range.float() - 0.5,
        h,
        b,
        a,
    )[:, :, 0, :1]
    tail_mass = lower_at_neg_range + (1.0 - upper_at_range)

    pmf = pmf.reshape(-1, max_length).cpu()
    tail_mass = tail_mass.reshape(-1, 1).cpu()
    pmf_length_flat = pmf_length.reshape(-1).cpu()
    quantized_cdf = _pmf_to_cdf(
        pmf,
        tail_mass,
        pmf_length_flat,
        max_length,
    )
    cdf_length = pmf_length_flat + 2
    expected_rows = bank_count * channel
    if quantized_cdf.shape[0] != expected_rows:
        raise RuntimeError("invalid Meta Prior CDF row count")
    return quantized_cdf.numpy(), cdf_length.int().numpy()


def factorized_cdf_cost_table(
    quantized_cdf,
    cdf_length,
    *,
    symbol_radius: int = 127,
) -> np.ndarray:
    """Convert integer factorized CDF rows to exact symbol costs.

    The returned table includes the same bypass-bin cost used by the rANS
    implementation and is suitable for minimum-code-length bank selection.
    """
    cdf = np.asarray(quantized_cdf, dtype=np.int64)
    lengths = np.asarray(cdf_length, dtype=np.int64)
    if cdf.ndim != 2 or lengths.shape != (cdf.shape[0],):
        raise ValueError("invalid factorized CDF inputs")
    symbols = np.arange(-symbol_radius, symbol_radius + 1)
    output = np.empty((cdf.shape[0], len(symbols)), dtype=np.float32)
    for row_index in range(cdf.shape[0]):
        length = int(lengths[row_index])
        row = cdf[row_index, :length]
        frequencies = np.diff(row)
        total = int(row[-1])
        max_value = length - 2
        for symbol_index, symbol in enumerate(symbols):
            value = abs(int(symbol)) * 2 - (int(symbol) > 0)
            bypass_bits = 0
            if value >= max_value:
                raw_value = value - max_value
                value = max_value
                n_bypass = 0
                while (raw_value >> (n_bypass * 2)) != 0:
                    n_bypass += 1
                bypass_bits = 2 * (n_bypass // 3 + 1 + n_bypass)
            output[row_index, symbol_index] = (
                -math.log2(frequencies[value] / total) + bypass_bits
            )
    return output


@torch.inference_mode()
def select_meta_prior_indexes(
    z_hat: torch.Tensor,
    cost_table,
    *,
    bank_count: int,
    symbol_radius: int = 127,
    chunk: int = 2048,
) -> torch.Tensor:
    """Choose one minimum-code-length bank per z spatial position."""
    if z_hat.ndim != 4:
        raise ValueError(f"expected BCHW z, got {tuple(z_hat.shape)}")
    batch, channel, height, width = z_hat.shape
    table = torch.as_tensor(
        cost_table,
        device=z_hat.device,
        dtype=torch.float32,
    )
    if tuple(table.shape[:1]) != (bank_count * channel,):
        raise ValueError(
            "Meta Prior cost table row count does not match banks × channels"
        )
    table = table.reshape(bank_count, channel, -1)
    positions = (
        z_hat.round()
        .to(torch.long)
        .permute(0, 2, 3, 1)
        .reshape(-1, channel)
    )
    shifted = positions + int(symbol_radius)
    if bool(((shifted < 0) | (shifted >= table.shape[-1])).any().item()):
        raise RuntimeError(
            f"z symbol exceeds Meta Prior selection radius ±{symbol_radius}"
        )
    selected = torch.empty(
        positions.shape[0],
        device=z_hat.device,
        dtype=torch.uint8,
    )
    for start in range(0, len(positions), int(chunk)):
        stop = min(start + int(chunk), len(positions))
        symbol_index = shifted[start:stop][None]
        expanded = table[:, None].expand(
            -1,
            stop - start,
            -1,
            -1,
        )
        gathered = torch.gather(
            expanded,
            3,
            symbol_index[:, :, :, None].expand(
                bank_count,
                -1,
                -1,
                1,
            ),
        ).squeeze(3)
        selected[start:stop] = gathered.sum(dim=2).argmin(dim=0).to(
            torch.uint8
        )
    return selected.reshape(batch, height, width)


@torch.inference_mode()
def select_meta_prior_indexes_from_cdf(
    z_hat: torch.Tensor,
    quantized_cdf,
    cdf_length,
    *,
    bank_count: int,
    chunk: int = 2048,
) -> torch.Tensor:
    """Select banks by the exact quantized-CDF rANS symbol cost.

    Unlike a finite lookup table, this path handles the complete lossless
    int16 z range. Tail symbols include the bank-dependent bypass-bin cost
    used by the C++ coder.
    """
    if z_hat.ndim != 4:
        raise ValueError(f"expected BCHW z, got {tuple(z_hat.shape)}")
    batch, channel, height, width = z_hat.shape
    if not 1 < int(bank_count) <= 256:
        raise ValueError("Meta Prior bank_count must be in [2, 256]")

    positions = (
        z_hat.round()
        .to(torch.int64)
        .permute(0, 2, 3, 1)
        .reshape(-1, channel)
    )
    if bool(
        (
            (positions < int(Z_SYMBOL_MIN))
            | (positions > int(Z_SYMBOL_MAX))
        ).any().item()
    ):
        raise OverflowError(
            "z symbol exceeds the lossless int16 rANS transport range"
        )

    cdf = torch.as_tensor(
        quantized_cdf,
        device=z_hat.device,
        dtype=torch.int64,
    )
    lengths = torch.as_tensor(
        cdf_length,
        device=z_hat.device,
        dtype=torch.int64,
    )
    expected_rows = int(bank_count) * channel
    if cdf.ndim != 2 or tuple(lengths.shape) != (expected_rows,):
        raise ValueError("invalid Meta Prior quantized CDF inputs")
    if cdf.shape[0] != expected_rows:
        raise ValueError(
            "Meta Prior CDF row count does not match banks × channels"
        )
    cdf = cdf.reshape(int(bank_count), channel, -1)
    lengths = lengths.reshape(int(bank_count), channel)
    if bool(((lengths < 2) | (lengths > cdf.shape[-1])).any().item()):
        raise ValueError("invalid Meta Prior CDF lengths")

    max_value = lengths - 2
    total = torch.gather(
        cdf,
        2,
        (lengths - 1)[..., None],
    ).squeeze(2).float()
    selected = torch.empty(
        positions.shape[0],
        device=z_hat.device,
        dtype=torch.uint8,
    )
    for start in range(0, len(positions), int(chunk)):
        stop = min(start + int(chunk), len(positions))
        symbols = positions[start:stop]
        values = symbols.abs() * 2 - (symbols > 0).to(torch.int64)
        values = values[None].expand(int(bank_count), -1, -1)
        tail = values >= max_value[:, None, :]
        encoded = torch.minimum(values, max_value[:, None, :])

        expanded_cdf = cdf[:, None].expand(
            -1,
            stop - start,
            -1,
            -1,
        )
        low = torch.gather(
            expanded_cdf,
            3,
            encoded[..., None],
        ).squeeze(3)
        high = torch.gather(
            expanded_cdf,
            3,
            (encoded + 1)[..., None],
        ).squeeze(3)
        frequency = (high - low).float()
        cost = -torch.log2(
            (frequency / total[:, None, :]).clamp_min(1e-12)
        )

        raw = torch.where(
            tail,
            values - max_value[:, None, :],
            torch.zeros_like(values),
        )
        remaining = raw
        n_bypass = torch.zeros_like(raw)
        # int16 symbols map to at most nine two-bit bypass groups.
        while bool((remaining > 0).any().item()):
            n_bypass += (remaining > 0).to(n_bypass.dtype)
            remaining = torch.bitwise_right_shift(remaining, 2)
        bypass_bins = torch.div(
            n_bypass,
            3,
            rounding_mode="floor",
        ) + 1 + n_bypass
        cost = cost + torch.where(
            tail,
            bypass_bins.float() * 2.0,
            torch.zeros_like(cost),
        )
        selected[start:stop] = cost.sum(dim=2).argmin(dim=0).to(
            torch.uint8
        )
    return selected.reshape(batch, height, width)


def _pack_fixed_width(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint8).reshape(-1)
    if not 1 <= bits <= 8:
        raise ValueError("fixed-width field must use 1..8 bits")
    if values.size and int(values.max()) >= (1 << bits):
        raise ValueError("value exceeds fixed-width selector range")
    output = bytearray((values.size * bits + 7) // 8)
    accumulator = 0
    available = 0
    output_index = 0
    for value in values:
        accumulator |= int(value) << available
        available += bits
        while available >= 8:
            output[output_index] = accumulator & 0xFF
            output_index += 1
            accumulator >>= 8
            available -= 8
    if available:
        output[output_index] = accumulator & 0xFF
    return bytes(output)


def _unpack_fixed_width(payload: bytes, count: int, bits: int) -> np.ndarray:
    expected = (count * bits + 7) // 8
    if len(payload) != expected:
        raise ValueError(
            f"selector payload size mismatch: {len(payload)} != {expected}"
        )
    output = np.empty(count, dtype=np.uint8)
    accumulator = 0
    available = 0
    input_index = 0
    mask = (1 << bits) - 1
    for output_index in range(count):
        while available < bits:
            accumulator |= payload[input_index] << available
            input_index += 1
            available += 8
        output[output_index] = accumulator & mask
        accumulator >>= bits
        available -= bits
    return output


def meta_prior_index_merge_probability_count(bank_count: int) -> int:
    """Number of Q11 context probabilities for Index Merge Coding."""
    bank_count = int(bank_count)
    if not 1 < bank_count <= 256:
        raise ValueError("Meta Prior bank_count must be in [2, 256]")
    bits = int(math.ceil(math.log2(bank_count)))
    return 4 + 4 * ((1 << bits) - 1)


def default_meta_prior_index_merge_probabilities(
    bank_count: int,
) -> np.ndarray:
    """Return the deterministic no-training P=0.5 context initialization."""
    return np.full(
        meta_prior_index_merge_probability_count(bank_count),
        META_PRIOR_PROBABILITY_TOTAL // 2,
        dtype=np.uint16,
    )


def _normalize_meta_prior_index_merge_probabilities(
    probabilities,
    bank_count: int,
) -> np.ndarray:
    if probabilities is None:
        return default_meta_prior_index_merge_probabilities(bank_count)
    normalized = np.ascontiguousarray(
        np.asarray(probabilities, dtype=np.uint16).reshape(-1)
    )
    expected = meta_prior_index_merge_probability_count(bank_count)
    if normalized.size != expected:
        raise ValueError(
            "Meta Prior Index Merge probability count mismatch: "
            f"{normalized.size} != {expected}"
        )
    if normalized.size and (
        int(normalized.min()) < 1
        or int(normalized.max()) >= META_PRIOR_PROBABILITY_TOTAL
    ):
        raise ValueError(
            "Meta Prior Index Merge probabilities must be in [1, 2047]"
        )
    return normalized


def _meta_prior_spatial_shape(bank_indexes, spatial_shape):
    indexes = np.asarray(bank_indexes, dtype=np.uint8)
    if spatial_shape is None:
        if indexes.ndim == 3:
            batch, height, width = indexes.shape
        elif indexes.ndim == 2:
            batch, height, width = 1, *indexes.shape
        elif indexes.ndim == 1:
            batch, height, width = 1, 1, indexes.size
        else:
            raise ValueError(
                "bank_indexes must be flat, HW, or BHW when spatial_shape "
                "is omitted"
            )
    else:
        shape = tuple(int(value) for value in spatial_shape)
        if len(shape) == 2:
            batch, height, width = 1, *shape
        elif len(shape) == 3:
            batch, height, width = shape
        else:
            raise ValueError("spatial_shape must be HW or BHW")
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError("Meta Prior spatial dimensions must be positive")
    if indexes.size != batch * height * width:
        raise ValueError(
            "Meta Prior index count does not match spatial_shape: "
            f"{indexes.size} != {batch}x{height}x{width}"
        )
    return (
        np.ascontiguousarray(indexes.reshape(-1), dtype=np.uint8),
        batch,
        height,
        width,
    )


def wrap_meta_prior_stream(
    rans_stream: bytes,
    bank_indexes,
    *,
    bank_count: int,
    spatial_shape=None,
    index_merge_probabilities=None,
    adaptation_shift: int = META_PRIOR_INDEX_MERGE_ADAPTATION_SHIFT,
    coding: str = "index_merge",
) -> bytes:
    """Prefix rANS with a backward-compatible Meta Prior selector payload.

    Production encoding always uses ``index_merge``: causal same-left/same-up
    flags followed by a contextual binary-tree escape index. Missing fitted
    probabilities use the syntax's deterministic P=0.5 initialization.
    Explicit ``auto`` and ``fixed`` remain for legacy-format comparisons.
    """
    bits = int(math.ceil(math.log2(bank_count)))
    indexes, batch, height, width = _meta_prior_spatial_shape(
        bank_indexes,
        spatial_shape,
    )
    if indexes.size and int(indexes.max()) >= int(bank_count):
        raise ValueError("Meta Prior selector contains index >= bank_count")
    coding = str(coding).strip().lower()
    if coding == "fixed":
        packed = _pack_fixed_width(indexes, bits)
        header = META_PRIOR_HEADER.pack(
            META_PRIOR_MAGIC,
            META_PRIOR_FIXED_WIDTH_VERSION,
            bits,
            int(bank_count),
        )
        return header + packed + bytes(rans_stream)
    if coding not in {"auto", "index_merge"}:
        raise ValueError(
            "Meta Prior selector coding must be auto/fixed/index_merge"
        )
    if _encode_meta_prior_index_merge is None:
        raise RuntimeError(
            "MLCodec_extensions_cpp lacks Meta Prior Index Merge Coding; "
            "rebuild the entropy extension"
        )
    probabilities = _normalize_meta_prior_index_merge_probabilities(
        index_merge_probabilities,
        bank_count,
    )
    packed = bytes(
        _encode_meta_prior_index_merge(
            indexes,
            batch,
            height,
            width,
            bits,
            probabilities,
            int(adaptation_shift),
        )
    )
    header = META_PRIOR_HEADER.pack(
        META_PRIOR_MAGIC,
        META_PRIOR_INDEX_MERGE_VERSION,
        bits,
        int(bank_count),
    )
    merged = (
        header
        + META_PRIOR_INDEX_MERGE_LENGTH.pack(len(packed))
        + packed
        + bytes(rans_stream)
    )
    if coding == "index_merge":
        return merged
    fixed_prefix_size = (
        META_PRIOR_HEADER.size
        + (indexes.size * bits + 7) // 8
    )
    merged_prefix_size = (
        META_PRIOR_HEADER.size
        + META_PRIOR_INDEX_MERGE_LENGTH.size
        + len(packed)
    )
    if merged_prefix_size <= fixed_prefix_size:
        return merged
    return wrap_meta_prior_stream(
        rans_stream,
        indexes,
        bank_count=bank_count,
        spatial_shape=(batch, height, width),
        coding="fixed",
    )


def unwrap_meta_prior_stream(
    stream: bytes,
    z_shape,
    *,
    expected_bank_count: int | None,
    index_merge_probabilities=None,
    adaptation_shift: int = META_PRIOR_INDEX_MERGE_ADAPTATION_SHIFT,
):
    """Return ``(bank_indexes_or_none, raw_rans_stream)``.

    Streams without the Meta Prior magic are legacy factorized-prior streams
    and are returned unchanged.
    """
    stream = bytes(stream)
    if len(stream) < META_PRIOR_HEADER.size or stream[:4] != META_PRIOR_MAGIC:
        return None, stream
    magic, version, bits, bank_count = META_PRIOR_HEADER.unpack_from(stream)
    if magic != META_PRIOR_MAGIC or version not in {
        META_PRIOR_FIXED_WIDTH_VERSION,
        META_PRIOR_INDEX_MERGE_VERSION,
    }:
        raise ValueError("unsupported Meta Prior stream version")
    if expected_bank_count is None:
        raise ValueError("bitstream requires Meta Prior but model has none")
    if bank_count != int(expected_bank_count):
        raise ValueError(
            f"Meta Prior bank count mismatch: stream={bank_count}, "
            f"model={expected_bank_count}"
        )
    expected_bits = int(math.ceil(math.log2(bank_count)))
    if bits != expected_bits:
        raise ValueError("invalid Meta Prior selector bit width")
    batch, _channel, height, width = tuple(int(value) for value in z_shape)
    count = batch * height * width
    start = META_PRIOR_HEADER.size
    if version == META_PRIOR_FIXED_WIDTH_VERSION:
        payload_size = (count * bits + 7) // 8
        stop = start + payload_size
        if len(stream) < stop + 4:
            raise ValueError("truncated Meta Prior entropy stream")
        indexes = _unpack_fixed_width(stream[start:stop], count, bits)
    else:
        length_stop = start + META_PRIOR_INDEX_MERGE_LENGTH.size
        if len(stream) < length_stop:
            raise ValueError("truncated Meta Prior Index Merge length")
        (payload_size,) = META_PRIOR_INDEX_MERGE_LENGTH.unpack_from(
            stream,
            start,
        )
        start = length_stop
        stop = start + int(payload_size)
        if payload_size == 0 or len(stream) < stop + 4:
            raise ValueError("truncated Meta Prior Index Merge stream")
        if _decode_meta_prior_index_merge is None:
            raise RuntimeError(
                "MLCodec_extensions_cpp lacks Meta Prior Index Merge Coding; "
                "rebuild the entropy extension"
            )
        probabilities = _normalize_meta_prior_index_merge_probabilities(
            index_merge_probabilities,
            bank_count,
        )
        indexes = np.asarray(
            _decode_meta_prior_index_merge(
                stream[start:stop],
                batch,
                height,
                width,
                bits,
                bank_count,
                probabilities,
                int(adaptation_shift),
            ),
            dtype=np.uint8,
        )
    if indexes.size and int(indexes.max()) >= bank_count:
        raise ValueError("Meta Prior stream contains an invalid bank index")
    return indexes.reshape(batch, height, width), stream[stop:]


def quantize_z(z_hat: torch.Tensor) -> np.ndarray:
    """z_hat → lossless int16 symbols in flattened NHWC order.

    The rANS CDF models a compact core range and uses bypass bits for tails.
    Silently clipping z would also change the hyper-decoder input and can
    severely damage both rate and reconstruction quality at high-rate QPs.
    """
    z_rounded = z_hat.round()
    out_of_range = torch.logical_or(
        z_rounded < Z_SYMBOL_MIN,
        z_rounded > Z_SYMBOL_MAX,
    )
    if bool(out_of_range.any().item()):
        z_min = float(z_rounded.min().item())
        z_max = float(z_rounded.max().item())
        raise OverflowError(
            "z symbol exceeds the lossless int16 rANS transport range: "
            f"[{z_min}, {z_max}] vs [{Z_SYMBOL_MIN}, {Z_SYMBOL_MAX}]"
        )
    z_int = z_rounded.to(torch.int16)
    return z_int.permute(0, 2, 3, 1).reshape(-1).cpu().numpy()


def dequantize_z(arr: torch.Tensor, shape) -> torch.Tensor:
    """(B*H*W*C,) → (B, C, H, W)."""
    B, C, H, W = shape
    return arr.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous().to(torch.float32)


# -----------------------------------------------------------------------------
# y conditional: per-scale-bucket Gaussian CDF
# -----------------------------------------------------------------------------

# Scale bucket constants, matching DCVC defaults.
SCALE_MIN = 0.11
SCALE_MAX = 256.0
SCALE_LEVEL = 64
LOG_SCALE_MIN = math.log(SCALE_MIN)
LOG_SCALE_MAX = math.log(SCALE_MAX)
LOG_SCALE_STEP = (LOG_SCALE_MAX - LOG_SCALE_MIN) / (SCALE_LEVEL - 1)


def skip_index_cutoff_from_tau(tau: float) -> int:
    """Map a scale threshold to the largest skipped log-scale bucket."""
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("skip tau must be a finite positive value")
    if tau < SCALE_MIN:
        return -1
    if tau >= SCALE_MAX:
        return SCALE_LEVEL - 1
    return min(
        SCALE_LEVEL - 1,
        int(math.floor((math.log(tau) - LOG_SCALE_MIN) / LOG_SCALE_STEP)),
    )


# Production skip policy. The 64 CDF rows are log-spaced, so tau=0.15 maps
# to rows 0..2 (row 2 has representative sigma 0.140694...).
DEFAULT_Y_SKIP_TAU = 0.15
DEFAULT_Y_SKIP_INDEX_CUTOFF = skip_index_cutoff_from_tau(DEFAULT_Y_SKIP_TAU)
DEFAULT_Y_SKIP_CONTEXT_MODE = "entropy_only"
Y_SKIP_CONTEXT_MODES = ("entropy_only", "exact")


@torch.inference_mode()
def update_y_cdf():
    """Build per-scale-bucket CDFs for the Gaussian y conditional.

    Returns (quantized_cdf, cdf_length) ready for slot 1.
    """
    scale_table = torch.exp(torch.linspace(LOG_SCALE_MIN, LOG_SCALE_MAX, SCALE_LEVEL))
    sym_range = torch.full_like(scale_table, MAX_ENTROPY_CODING_VALUE)
    for i in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
        normal = torch.distributions.Normal(0.0, scale_table)
        probs = normal.cdf(torch.full_like(scale_table, float(i)))
        sym_range = torch.where(probs > 1 - 1e-3, float(i), sym_range)
    sym_range = sym_range.int()
    pmf_length = 2 * sym_range + 1
    max_length = 2 * MAX_ENTROPY_CODING_VALUE + 1

    samples = torch.arange(max_length).float() - sym_range[:, None].float()
    scales = scale_table[:, None]
    normal = torch.distributions.Normal(0.0, scales)
    upper = normal.cdf(samples + 0.5)
    lower = normal.cdf(samples - 0.5)
    pmf = upper - lower
    tail_mass = 2 * lower[:, :1]

    quantized_cdf = _pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
    cdf_length = pmf_length + 2
    return quantized_cdf.numpy(), cdf_length.int().numpy()


def build_cdf_tables(codec) -> dict:
    """Precompute the exact rANS CDF tables stored in INT8 artifacts."""
    z_tables = {}
    for qp in range(codec.qp_num):
        cdf, lengths = update_z_cdf(codec.entropy_bottleneck, qp)
        z_tables[qp] = (
            torch.as_tensor(cdf),
            torch.as_tensor(lengths),
        )
    y_cdf, y_lengths = update_y_cdf()
    output = {
        "z": z_tables,
        "y": (
            torch.as_tensor(y_cdf),
            torch.as_tensor(y_lengths),
        ),
    }
    if getattr(codec, "meta_prior_enabled", False):
        meta_tables = {}
        for qp in range(codec.qp_num):
            cdf, lengths = update_meta_prior_cdf(codec.meta_prior, qp)
            meta_tables[qp] = (
                torch.as_tensor(cdf),
                torch.as_tensor(lengths),
            )
        output["meta_prior"] = meta_tables
    return output


def quantize_y_centered(y_centered: torch.Tensor) -> torch.Tensor:
    """y - means → int8 symbols, clamped to [-MAX, MAX]."""
    return y_centered.round().clamp(
        -MAX_ENTROPY_CODING_VALUE, MAX_ENTROPY_CODING_VALUE
    ).to(torch.int8)


def pack_y_symbols(symbols_i8: torch.Tensor, scale_idx_u8: torch.Tensor,
                   keep_mask: Optional[torch.Tensor] = None) -> np.ndarray:
    """Pack (symbol, bucket_idx) into int16: (symbol << 8) | bucket_idx.

    Layout follows DCVC: permute to (B, H, W, C), flatten, optionally
    filter by `keep_mask`.
    """
    sym = symbols_i8.short()
    idx = scale_idx_u8.short()
    packed = (sym << 8) + idx
    packed = packed.permute(0, 2, 3, 1).reshape(-1).to(torch.int16)
    if keep_mask is not None:
        keep = keep_mask.permute(0, 2, 3, 1).reshape(-1).bool()
        packed = packed[keep]
    return packed.cpu().numpy()


def _select_checkerboard_part_cpu(
    symbols_nchw: np.ndarray,
    part: int,
) -> np.ndarray:
    """Select one two-step checkerboard part directly in NHWC coder order."""
    symbols = np.asarray(symbols_nchw)
    if symbols.ndim != 4:
        raise ValueError(f"expected NCHW symbols, got {symbols.shape}")
    batch, channels, height, width = symbols.shape
    if channels % 2:
        raise ValueError("checkerboard packing requires an even channel count")
    half = channels // 2
    selected = np.empty((batch, height, width, half), dtype=np.int16)
    if part == 0:
        selected[:, 0::2, 0::2, :] = symbols[
            :, :half, 0::2, 0::2
        ].transpose(0, 2, 3, 1)
        selected[:, 1::2, 1::2, :] = symbols[
            :, :half, 1::2, 1::2
        ].transpose(0, 2, 3, 1)
        selected[:, 0::2, 1::2, :] = symbols[
            :, half:, 0::2, 1::2
        ].transpose(0, 2, 3, 1)
        selected[:, 1::2, 0::2, :] = symbols[
            :, half:, 1::2, 0::2
        ].transpose(0, 2, 3, 1)
    elif part == 1:
        selected[:, 0::2, 1::2, :] = symbols[
            :, :half, 0::2, 1::2
        ].transpose(0, 2, 3, 1)
        selected[:, 1::2, 0::2, :] = symbols[
            :, :half, 1::2, 0::2
        ].transpose(0, 2, 3, 1)
        selected[:, 0::2, 0::2, :] = symbols[
            :, half:, 0::2, 0::2
        ].transpose(0, 2, 3, 1)
        selected[:, 1::2, 1::2, :] = symbols[
            :, half:, 1::2, 1::2
        ].transpose(0, 2, 3, 1)
    else:
        raise ValueError(f"checkerboard part must be 0 or 1, got {part}")
    return selected.reshape(-1)


def pack_y_symbols_checkerboard_cpu(
    symbols_i8: torch.Tensor,
    packed_scale_idx_u8: torch.Tensor,
    *,
    part: int,
) -> np.ndarray:
    """Pack one checkerboard part without materializing a boolean mask."""
    if symbols_i8.device.type != "cpu":
        raise ValueError("CPU checkerboard pack requires a CPU tensor")
    symbols = _select_checkerboard_part_cpu(
        symbols_i8.detach().numpy(),
        part,
    )
    indexes = (
        packed_scale_idx_u8.detach()
        .to(device="cpu", dtype=torch.uint8)
        .contiguous()
        .numpy()
        .astype(np.int16, copy=False)
    )
    if symbols.size != indexes.size:
        raise ValueError(
            f"checkerboard symbol/index mismatch: "
            f"{symbols.size} != {indexes.size}"
        )
    return np.ascontiguousarray(
        ((symbols << 8) + indexes).astype(np.int16, copy=False)
    )


def restore_y_symbols_checkerboard_cpu(
    flat_symbols_i16: np.ndarray,
    shape,
    *,
    part: int,
) -> torch.Tensor:
    """Restore one packed checkerboard part directly to contiguous NCHW."""
    batch, channels, height, width = tuple(int(value) for value in shape)
    if channels % 2:
        raise ValueError("checkerboard restore requires an even channel count")
    half = channels // 2
    packed = np.asarray(flat_symbols_i16, dtype=np.int16)
    expected = batch * height * width * half
    if packed.size != expected:
        raise ValueError(
            f"checkerboard symbol count mismatch: "
            f"{packed.size} != {expected}"
        )
    packed = packed.reshape(batch, height, width, half).transpose(0, 3, 1, 2)
    output = np.zeros(
        (batch, channels, height, width),
        dtype=np.float32,
    )
    if part == 0:
        output[:, :half, 0::2, 0::2] = packed[:, :, 0::2, 0::2]
        output[:, :half, 1::2, 1::2] = packed[:, :, 1::2, 1::2]
        output[:, half:, 0::2, 1::2] = packed[:, :, 0::2, 1::2]
        output[:, half:, 1::2, 0::2] = packed[:, :, 1::2, 0::2]
    elif part == 1:
        output[:, :half, 0::2, 1::2] = packed[:, :, 0::2, 1::2]
        output[:, :half, 1::2, 0::2] = packed[:, :, 1::2, 0::2]
        output[:, half:, 0::2, 0::2] = packed[:, :, 0::2, 0::2]
        output[:, half:, 1::2, 1::2] = packed[:, :, 1::2, 1::2]
    else:
        raise ValueError(f"checkerboard part must be 0 or 1, got {part}")
    return torch.from_numpy(output)


def select_y_indexes_above_cpu(
    packed_scale_idx_u8,
    index_cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return kept CDF indexes and their compact checkerboard positions."""
    indexes = np.ascontiguousarray(
        torch.as_tensor(packed_scale_idx_u8)
        .detach()
        .to(device="cpu", dtype=torch.uint8)
        .numpy(),
        dtype=np.uint8,
    )
    selected, positions = _select_indexes_above(indexes, int(index_cutoff))
    return np.asarray(selected), np.asarray(positions)


def select_y_indexes_nchw_offsets_cpu(
    packed_scale_idx_u8,
    index_cutoff: int,
    shape,
    *,
    part: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return kept indexes and direct flat NCHW symbol offsets."""
    batch, channels, height, width = tuple(int(value) for value in shape)
    indexes = np.ascontiguousarray(
        torch.as_tensor(packed_scale_idx_u8)
        .detach()
        .to(device="cpu", dtype=torch.uint8)
        .numpy(),
        dtype=np.uint8,
    )
    selected, offsets = _select_checkerboard_indexes_above(
        indexes,
        int(index_cutoff),
        batch,
        channels,
        height,
        width,
        int(part),
    )
    return np.asarray(selected), np.asarray(offsets)


def pack_y_symbols_checkerboard_positions_cpu(
    symbols,
    selected_indexes_u8: np.ndarray,
    selected_positions_u32: np.ndarray,
    *,
    part: int,
) -> tuple[np.ndarray, int]:
    """Gather and pack only non-skipped checkerboard symbols in C++."""
    if isinstance(symbols, torch.Tensor):
        if symbols.device.type != "cpu":
            raise ValueError("CPU checkerboard pack requires a CPU tensor")
        symbols = symbols.detach().float().contiguous().numpy()
    symbols = np.ascontiguousarray(symbols, dtype=np.float32)
    packed, outside = _pack_checkerboard_positions(
        symbols,
        np.ascontiguousarray(selected_indexes_u8, dtype=np.uint8),
        np.ascontiguousarray(selected_positions_u32, dtype=np.uint32),
        int(part),
        MAX_ENTROPY_CODING_VALUE,
    )
    return np.asarray(packed), int(outside)


def pack_y_symbols_nchw_offsets_cpu(
    symbols,
    selected_indexes_u8: np.ndarray,
    selected_offsets_u32: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Gather selected symbols through precomputed direct NCHW offsets."""
    if isinstance(symbols, torch.Tensor):
        if symbols.device.type != "cpu":
            raise ValueError("CPU NCHW-offset pack requires a CPU tensor")
        symbols = symbols.detach().float().contiguous().numpy()
    symbols = np.ascontiguousarray(symbols, dtype=np.float32)
    packed, outside = _pack_nchw_offsets(
        symbols,
        np.ascontiguousarray(selected_indexes_u8, dtype=np.uint8),
        np.ascontiguousarray(selected_offsets_u32, dtype=np.uint32),
        MAX_ENTROPY_CODING_VALUE,
    )
    return np.asarray(packed), int(outside)


def restore_y_symbols_checkerboard_positions_cpu(
    decoded_symbols_i16: np.ndarray,
    selected_positions_u32: np.ndarray,
    shape,
    *,
    part: int,
) -> torch.Tensor:
    """Restore non-skipped symbols directly into a full zero-filled NCHW tensor."""
    batch, channels, height, width = tuple(int(value) for value in shape)
    restored = _restore_checkerboard_positions(
        np.ascontiguousarray(decoded_symbols_i16, dtype=np.int16),
        np.ascontiguousarray(selected_positions_u32, dtype=np.uint32),
        batch,
        channels,
        height,
        width,
        int(part),
    )
    return torch.from_numpy(np.asarray(restored))


def restore_y_symbols_checkerboard_positions_into_cpu(
    output_nchw: np.ndarray,
    decoded_symbols_i16: np.ndarray,
    selected_positions_u32: np.ndarray,
    *,
    part: int,
) -> None:
    """Write one sparse checkerboard part into an existing FP32 NCHW array."""
    if output_nchw.dtype != np.float32 or not output_nchw.flags.c_contiguous:
        raise ValueError("output_nchw must be a C-contiguous FP32 array")
    _restore_checkerboard_positions_into(
        output_nchw,
        np.ascontiguousarray(decoded_symbols_i16, dtype=np.int16),
        np.ascontiguousarray(selected_positions_u32, dtype=np.uint32),
        int(part),
    )


def restore_y_symbols_nchw_offsets_into_cpu(
    output_nchw: np.ndarray,
    decoded_symbols_i16: np.ndarray,
    selected_offsets_u32: np.ndarray,
) -> None:
    """Write sparse symbols through precomputed direct flat NCHW offsets."""
    if output_nchw.dtype != np.float32 or not output_nchw.flags.c_contiguous:
        raise ValueError("output_nchw must be a C-contiguous FP32 array")
    _restore_nchw_offsets_into(
        output_nchw,
        np.ascontiguousarray(decoded_symbols_i16, dtype=np.int16),
        np.ascontiguousarray(selected_offsets_u32, dtype=np.uint32),
    )


def select_y_indexes(scale_idx_u8: torch.Tensor,
                     keep_mask: Optional[torch.Tensor] = None) -> np.ndarray:
    """Decode-side: bucket indexes (uint8), permuted to (B, H, W, C) and flat."""
    idx = scale_idx_u8.permute(0, 2, 3, 1).reshape(-1)
    if keep_mask is not None:
        keep = keep_mask.permute(0, 2, 3, 1).reshape(-1).bool()
        idx = idx[keep]
    return idx.to(torch.uint8).cpu().numpy()


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------

class EntropyCoder:
    """Thin wrapper over MLCodec_extensions_cpp's RansEncoder / RansDecoder.

    z uses slot 0 (per-channel CDF, encode_z / decode_z, int8 symbols).
    y uses slot 1 (per-scale CDF, encode_y / decode_y, packed int16).
    """

    def __init__(self):
        if not _RANS_AVAILABLE:
            raise RuntimeError(
                "MLCodec_extensions_cpp is not installed. Build it from "
                "codec/models/entropy/extensions/cpp/install.sh"
            )
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

    def set_z_cdf(self, cdf, cdf_length):
        self.encoder.set_cdf(cdf, cdf_length, 0)
        self.decoder.set_cdf(cdf, cdf_length, 0)

    def set_y_cdf(self, cdf, cdf_length):
        self.encoder.set_cdf(cdf, cdf_length, 1)
        self.decoder.set_cdf(cdf, cdf_length, 1)

    def reset(self):
        self.encoder.reset()

    def encode_z(self, symbols_i16: np.ndarray, cdf_offset: int, ch: int):
        self.encoder.encode_z(
            np.ascontiguousarray(symbols_i16, dtype=np.int16),
            cdf_offset,
            ch,
        )

    def encode_z_meta_prior(
        self,
        symbols_i16: np.ndarray,
        bank_indexes_u8: np.ndarray,
        ch: int,
    ):
        self.encoder.encode_z_meta_prior(
            np.ascontiguousarray(symbols_i16, dtype=np.int16),
            np.ascontiguousarray(bank_indexes_u8, dtype=np.uint8),
            int(ch),
        )

    def encode_y(self, packed_i16: np.ndarray):
        self.encoder.encode_y(
            np.ascontiguousarray(packed_i16, dtype=np.int16)
        )

    def encode_y_skip(self, packed_i16: np.ndarray, index_cutoff: int):
        """Encode only entries whose packed Gaussian CDF index exceeds cutoff."""
        self.encoder.encode_y_skip(
            np.ascontiguousarray(packed_i16, dtype=np.int16),
            int(index_cutoff),
        )

    def flush(self) -> bytes:
        self.encoder.flush()
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream: bytes):
        self.decoder.set_stream(np.frombuffer(stream, dtype=np.uint8))

    def decode_z(self, total_size: int, cdf_offset: int, ch: int):
        self.decoder.decode_z(total_size, cdf_offset, ch)

    def decode_z_meta_prior(
        self,
        total_size: int,
        bank_indexes_u8: np.ndarray,
        ch: int,
    ):
        self.decoder.decode_z_meta_prior(
            int(total_size),
            np.ascontiguousarray(bank_indexes_u8, dtype=np.uint8),
            int(ch),
        )

    def decode_y(self, indexes_u8: np.ndarray):
        self.decoder.decode_y(
            np.ascontiguousarray(indexes_u8, dtype=np.uint8)
        )

    def decode_and_get_y(self, indexes_u8: np.ndarray) -> np.ndarray:
        return self.decoder.decode_and_get_y(
            np.ascontiguousarray(indexes_u8, dtype=np.uint8)
        )

    def decode_and_get_y_skip(
        self,
        indexes_u8: np.ndarray,
        index_cutoff: int,
    ) -> np.ndarray:
        """Decode kept entries and return a full array with skipped values zero."""
        return self.decoder.decode_and_get_y_skip(
            np.ascontiguousarray(indexes_u8, dtype=np.uint8),
            int(index_cutoff),
        )

    def get_decoded(self, device, dtype) -> torch.Tensor:
        rv = self.decoder.get_decoded_tensor()
        rv = torch.as_tensor(rv).to(device=device, dtype=dtype)
        return rv


class SplitEntropyDecoder(EntropyCoder):
    """Independent z/y0/y1 sections, with explicit retained/dense partitioning."""

    def __init__(self, lanes, *, dense=False, cutoff=None):
        if len(lanes) != 3 or any(not 1 <= n <= (32 if dense else 8) for n in lanes):
            raise ValueError("invalid split entropy lane counts")
        decoder_type = RansDecoder
        if dense:
            from deployment.gpu_nvidia_h100.cpp.pulse_gpu_entropy_ext import RansDecoder as decoder_type
        self.dense, self.cutoff, self.part_offset = dense, cutoff, 0
        self.decoders = [decoder_type() for _ in lanes]
        for decoder, count in zip(self.decoders, lanes):
            decoder.set_entropy_coder_parallel(count)
        self.decoder = self.decoders[0]
        self.part = 0

    def set_z_cdf(self, cdf, cdf_length):
        self.decoders[0].set_cdf(cdf, cdf_length, 0)

    def set_y_cdf(self, cdf, cdf_length):
        if self.dense and len(cdf) == 128:
            self.part_offset = 64
            self.decoders[1].set_cdf(cdf[:64], cdf_length[:64], 1)
            self.decoders[2].set_cdf(cdf[64:], cdf_length[64:], 1)
        else:
            for decoder in self.decoders[1:]:
                decoder.set_cdf(cdf, cdf_length, 1)

    def set_stream(self, stream):
        if len(stream) < 12:
            raise ValueError("truncated split entropy header")
        lengths = struct.unpack_from("<III", stream)
        if min(lengths) < 4 or 12 + sum(lengths) != len(stream):
            raise ValueError("invalid split entropy section lengths")
        offset = 12
        for decoder, length in zip(self.decoders, lengths):
            decoder.set_stream(np.frombuffer(stream[offset:offset + length], dtype=np.uint8))
            offset += length
        self.decoder, self.part = self.decoders[0], 0

    def _next_part(self):
        if self.part >= 2:
            raise ValueError("too many checkerboard parts")
        self.part += 1
        self.decoder = self.decoders[self.part]

    def decode_y(self, indexes_u8):
        self._next_part()
        if self.part == 2 and self.part_offset:
            indexes_u8 = np.ascontiguousarray(indexes_u8 - self.part_offset)
        if self.dense and self.cutoff is not None:
            return self.decoder.decode_y_skip(np.ascontiguousarray(indexes_u8), self.cutoff)
        return super().decode_y(indexes_u8)

    def decode_and_get_y(self, indexes_u8):
        self._next_part()
        if self.part == 2 and self.part_offset:
            indexes_u8 = np.ascontiguousarray(indexes_u8 - self.part_offset)
        if self.dense and self.cutoff is not None:
            return self.decoder.decode_and_get_y_skip(np.ascontiguousarray(indexes_u8), self.cutoff)
        return super().decode_and_get_y(indexes_u8)
