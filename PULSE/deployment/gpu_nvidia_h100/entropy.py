"""Original high-throughput split rANS path: z16, y0/y1 each 32 lanes."""
import struct
import numpy as np

from codec import bitstream
from codec.models.entropy import entropy_coding as ec
from deployment.entropy import EntropyCodec


class SplitEntropyCodec(EntropyCodec):
    def __init__(self, extension, artifact, fingerprint, *, height, width, qp, cutoff, cdf_mode=None):
        super().__init__(extension, artifact, fingerprint, height=height, width=width,
                         qp=qp, cutoff=cutoff, lanes=32, threads=16, cdf_mode=cdf_mode)
        self.sections = [16, 32, 32]
        y_cdf, y_len = self.cdfs[1]
        first = (y_cdf[:64], y_len[:64])
        second = (y_cdf[64:], y_len[64:]) if self.mode == "part_specific" else first
        self.part_offset = 0
        self.y1_encoder, self.y1_decoder = extension.RansEncoder(), extension.RansDecoder()
        self.z_encoder, self.z_decoder = extension.RansEncoder(), extension.RansDecoder()
        for coder in (self.y1_encoder, self.y1_decoder):
            coder.set_entropy_coder_parallel(32)
            coder.set_cdf(*second, 1)
        for coder in (self.z_encoder, self.z_decoder):
            coder.set_entropy_coder_parallel(16)
            coder.set_cdf(*self.cdfs[0], 0)
        for coder in (self.encoder, self.decoder):
            coder.set_cdf(*first, 1)
        self.readers = (self.z_decoder, self.decoder, self.y1_decoder)

    @staticmethod
    def flush(encoder):
        encoder.flush()
        return bytes(encoder.get_encoded_stream())

    def compress(self, z, s0, s1, meta_indexes=None):
        self.require_encoding_defaults()
        if meta_indexes is None:
            self.selector.select_into(z, self.meta_indexes, 8)
        else:
            np.copyto(self.meta_indexes, meta_indexes)
        self.linear.packed_symbols_into(z, s0, s1, self.qp,
                                        self.packed[0], self.packed[1], self.threads)
        np.copyto(self.z_nhwc.reshape(*self.meta_indexes.shape, self.z_channels), z.transpose(0, 2, 3, 1))
        for encoder in (self.encoder, self.y1_encoder, self.z_encoder):
            encoder.reset()
        for encoder, values in ((self.encoder, self.packed[0]), (self.y1_encoder, self.packed[1])):
            if self.cutoff < 0:
                encoder.encode_y_borrowed(values)
            else:
                encoder.encode_y_skip_borrowed(values, self.cutoff)
        # Queue both y parts before waiting, matching the source implementation.
        y0, y1 = self.flush(self.encoder), self.flush(self.y1_encoder)
        self.z_encoder.encode_z_meta_prior_borrowed(self.z_nhwc, self.meta_indexes.reshape(-1), self.z_channels)
        z_raw = self.flush(self.z_encoder)
        raw = struct.pack("<III", len(z_raw), len(y0), len(y1)) + z_raw + y0 + y1
        payload = ec.wrap_meta_prior_stream(
            raw, self.meta_indexes, bank_count=64, coding=self.meta_coding,
            index_merge_probabilities=self.probabilities, adaptation_shift=self.adaptation_shift)
        return bitstream.pack({**self.metadata, "entropy_sections": self.sections,
                               "entropy_dense": True}, payload, fingerprint=self.fingerprint)

    def decompress(self, stream, z, s0, s1, on_decoded=None):
        metadata, payload = bitstream.unpack(stream)
        if metadata.pop("model_id") != bytes.fromhex(self.fingerprint)[:16]:
            raise ValueError("bitstream was encoded with a different model bundle")
        if metadata.pop("entropy_sections", None) != self.sections or not metadata.pop("entropy_dense", False):
            raise ValueError("expected the H100 dense split profile")
        if metadata != self.metadata:
            raise ValueError("bitstream settings do not match this runtime")
        indexes, raw = ec.unwrap_meta_prior_stream(
            payload, self.z_shape, expected_bank_count=64,
            index_merge_probabilities=self.probabilities, adaptation_shift=self.adaptation_shift)
        if len(raw) < 12:
            raise ValueError("truncated split entropy header")
        lengths = struct.unpack_from("<III", raw)
        if min(lengths) < 4 or 12 + sum(lengths) != len(raw):
            raise ValueError("invalid split entropy section lengths")
        offset = 12
        for decoder, length in zip(self.readers, lengths):
            decoder.set_stream(np.frombuffer(raw[offset:offset + length], dtype=np.uint8))
            offset += length
        self.z_decoder.decode_z_meta_prior(z.size, indexes.reshape(-1), self.z_channels)
        decoded_z = np.asarray(self.z_decoder.get_decoded_tensor_view())
        np.copyto(z, decoded_z.reshape(*self.meta_indexes.shape, self.z_channels).transpose(0, 3, 1, 2))
        if on_decoded is not None:
            on_decoded(0)
        self.linear.packed_indexes_into(z, self.qp, self.indexes, self.threads)
        for decoder, idx in zip(self.readers[1:], self.indexes):
            if self.cutoff < 0:
                decoder.decode_y_borrowed(idx)
            else:
                decoder.decode_y_skip_borrowed(idx, self.cutoff)
        for part, (decoder, output) in enumerate(zip(self.readers[1:], (s0, s1))):
            decoded = np.asarray(decoder.get_decoded_tensor_view())
            self.ext.narrow_i16_to_i8_into(decoded, output.reshape(-1), 1)
            if on_decoded is not None:
                on_decoded(part + 1)
