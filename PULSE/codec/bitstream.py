"""Checked .pulse envelope around the unchanged native entropy transport."""
import struct
import zlib

from codec.utils.pulse_header import pack_header, unpack_header

MAX_PIXELS = 100_000_000
MAX_STREAM_BYTES = 512 * 1024 * 1024
MAGIC = b"PLS1"
PARALLEL_MAGIC = b"PLS2"
SPLIT_MAGIC = b"PLS3"
ENVELOPE = struct.Struct("<4s16sQI")  # magic, model id, native length, CRC32


def pack(metadata, payload, *, fingerprint, legacy=False):
    native = pack_header(
        W=metadata["width"], H=metadata["height"], qp=metadata["qp"],
        y_shape=metadata["y_shape"], z_shape=metadata["z_shape"],
        skip_index_cutoff=metadata["skip_index_cutoff"],
        y_cdf_mode=metadata["y_cdf_mode"],
    ) + payload
    lanes = metadata.get("entropy_lanes", 1)
    sections = metadata.get("entropy_sections")
    dense = bool(metadata.get("entropy_dense", False))
    if not 1 <= lanes <= 8:
        raise ValueError("entropy lanes must be in [1,8]")
    if sections is not None:
        limit = 32 if dense else 8
        if legacy or lanes != 1 or len(sections) != 3 or any(not 1 <= n <= limit for n in sections):
            raise ValueError("split entropy requires three lane counts and a checked envelope")
        counts = [sections[0], sections[1] | (128 if dense else 0), sections[2] | (128 if dense else 0)]
        native = bytes(counts) + native
    elif dense:
        raise ValueError("dense skip partitioning requires split entropy")
    elif lanes != 1:
        if legacy:
            raise ValueError("parallel streams require the checked PULSE envelope")
        native = bytes([lanes]) + native
    if len(native) > MAX_STREAM_BYTES:
        raise ValueError("bitstream exceeds the size limit")
    if legacy:
        return native
    model_id = bytes.fromhex(fingerprint)[:16]
    if len(model_id) != 16:
        raise ValueError("invalid model fingerprint")
    magic = SPLIT_MAGIC if sections is not None else (MAGIC if lanes == 1 else PARALLEL_MAGIC)
    prefix = struct.pack("<4s16sQ", magic, model_id, len(native))
    checksum = zlib.crc32(native, zlib.crc32(prefix))
    return prefix + struct.pack("<I", checksum) + native


def unpack(data, *, allow_legacy=False):
    model_id = None
    lanes = 1
    sections = None
    dense = False
    if len(data) > MAX_STREAM_BYTES + ENVELOPE.size:
        raise ValueError("bitstream exceeds the size limit")
    if data[:4] in (MAGIC, PARALLEL_MAGIC, SPLIT_MAGIC):
        if len(data) < ENVELOPE.size:
            raise ValueError("truncated PULSE envelope")
        magic, model_id, size, checksum = ENVELOPE.unpack_from(data)
        native = data[ENVELOPE.size:]
        if len(native) != size:
            raise ValueError("bitstream length mismatch")
        if zlib.crc32(native, zlib.crc32(data[:ENVELOPE.size - 4])) != checksum:
            raise ValueError("bitstream checksum mismatch")
        data = native
        if magic == SPLIT_MAGIC:
            if len(data) < 3 or bool(data[1] & 128) != bool(data[2] & 128):
                raise ValueError("invalid split entropy lane counts")
            dense = bool(data[1] & 128)
            sections = [data[0], data[1] & 127, data[2] & 127]
            if any(not 1 <= n <= (32 if dense else 8) for n in sections):
                raise ValueError("invalid split entropy lane counts")
            data = data[3:]
        elif magic == PARALLEL_MAGIC:
            if not data or not 2 <= data[0] <= 8:
                raise ValueError("invalid parallel entropy lane count")
            lanes, data = data[0], data[1:]
    elif not allow_legacy:
        raise ValueError("missing PULSE envelope; native streams require explicit --allow-legacy")
    parsed = unpack_header(data)
    if parsed["H"] * parsed["W"] > MAX_PIXELS:
        raise ValueError("image exceeds the decoder pixel limit")
    if not parsed["stream"]:
        raise ValueError("missing entropy payload")
    metadata = dict(model_id=model_id, height=parsed["H"], width=parsed["W"], qp=parsed["qp"],
                    y_shape=list(parsed["y_shape"]), z_shape=list(parsed["z_shape"]),
                    skip_index_cutoff=parsed["skip_index_cutoff"],
                    y_cdf_mode=parsed["y_cdf_mode"])
    if lanes != 1:
        metadata["entropy_lanes"] = lanes
    if sections is not None:
        metadata["entropy_sections"] = sections
    if dense:
        metadata["entropy_dense"] = True
    return metadata, parsed["stream"]
