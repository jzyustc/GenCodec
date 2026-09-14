"""Versioned PULSE image header; entropy payload syntax remains native rANS.

c0de: Gaussian/no skip; c0df: Gaussian/explicit skip.
c0e1: part CDF/default skip2 (19B).
c0e2: part CDF/no skip (19B); c0e3: part CDF/explicit skip (20B).
The matching network and entropy artifact are external shared model assets.
"""

import struct

# Wire-format constant, not a mutable encoder default. A future default other
# than2 must use the explicit-cutoff format rather than reinterpret old c0e1.
PART_DEFAULT_SKIP_INDEX_CUTOFF = 2

HEADER = struct.Struct("<2sHHBHHHHHH")
SKIP_HEADER = struct.Struct("<2sHHBBHHHHHH")


def pack_header(
    *, W, H, qp, y_shape, z_shape, skip_index_cutoff=-1, y_cdf_mode="gaussian"
):
    if y_cdf_mode not in ("gaussian", "part_specific"):
        raise ValueError("invalid y CDF mode")
    if not 0 <= qp <= 7 or min(W, H, *y_shape[-3:], *z_shape[-3:]) <= 0:
        raise ValueError("invalid image/QP/latent shape")
    if not -1 <= skip_index_cutoff <= 63:
        raise ValueError("invalid skip cutoff")
    fields = (W, H, qp, *y_shape[-3:], *z_shape[-3:])
    if y_cdf_mode == "part_specific":
        if skip_index_cutoff == PART_DEFAULT_SKIP_INDEX_CUTOFF:
            return HEADER.pack(b"\xc0\xe1", *fields)
        if skip_index_cutoff < 0:
            return HEADER.pack(b"\xc0\xe2", *fields)
        return SKIP_HEADER.pack(
            b"\xc0\xe3", W, H, qp, skip_index_cutoff, *y_shape[-3:], *z_shape[-3:]
        )
    if skip_index_cutoff < 0:
        return HEADER.pack(b"\xc0\xde", *fields)
    return SKIP_HEADER.pack(
        b"\xc0\xdf", W, H, qp, skip_index_cutoff, *y_shape[-3:], *z_shape[-3:]
    )


def unpack_header(data):
    magic = data[:2]
    if magic not in (b"\xc0\xde", b"\xc0\xdf", b"\xc0\xe1", b"\xc0\xe2", b"\xc0\xe3"):
        raise ValueError(f"bad image magic: {magic!r}")
    explicit = magic in (b"\xc0\xdf", b"\xc0\xe3")
    parser = SKIP_HEADER if explicit else HEADER
    if len(data) < parser.size:
        raise ValueError("truncated image header")
    fields = parser.unpack_from(data)
    if explicit:
        _, W, H, qp, cutoff, yc, yh, yw, zc, zh, zw = fields
    else:
        _, W, H, qp, yc, yh, yw, zc, zh, zw = fields
        cutoff = PART_DEFAULT_SKIP_INDEX_CUTOFF if magic == b"\xc0\xe1" else -1
    if min(W, H, yc, yh, yw, zc, zh, zw) <= 0 or qp > 7 or not -1 <= cutoff <= 63:
        raise ValueError("invalid image header values")
    return dict(
        W=W,
        H=H,
        qp=qp,
        y_shape=(1, yc, yh, yw),
        z_shape=(1, zc, zh, zw),
        skip_index_cutoff=cutoff,
        y_cdf_mode="part_specific"
        if magic in (b"\xc0\xe1", b"\xc0\xe2", b"\xc0\xe3")
        else "gaussian",
        header_bytes=parser.size,
        stream=data[parser.size :],
    )
