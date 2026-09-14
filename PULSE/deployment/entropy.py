"""Optimized native Linear CDF, Meta Prior and rANS transport."""
import numpy as np
import torch

from codec import bitstream
from codec.models.entropy import entropy_coding as ec
from codec.models.entropy.y_cdf import validate_part_cdf


class EntropyCodec:
    def __init__(self, extension, artifact, fingerprint, *, height, width, qp, cutoff,
                 lanes=1, threads=1, cdf_mode=None):
        self.ext = extension
        self.qp, self.cutoff, self.lanes, self.threads = qp, cutoff, lanes, threads
        self.fingerprint = fingerprint
        cfg, state = artifact["constructor"], artifact["state_dict"]
        self.channels, self.z_channels = int(cfg["M"]), int(cfg["z_ch"])
        ph, pw = (height + 63) // 64 * 64, (width + 63) // 64 * 64
        self.z_shape = (1, self.z_channels, ph // 64, pw // 64)
        self.compact_shape = (1, ph // 16, pw // 16, self.channels // 2)
        self.mode = "part_specific" if "y_part_specific" in artifact["cdf_tables"] else "gaussian"
        if cdf_mode is not None:
            self.mode = cdf_mode
        self.metadata = dict(height=height, width=width, qp=qp,
                             y_shape=[1, self.channels, ph // 16, pw // 16],
                             z_shape=list(self.z_shape), skip_index_cutoff=cutoff,
                             y_cdf_mode=self.mode)
        self.linear = extension.LogIndexDecoder(
            self.channels, self.z_channels, int(cfg["qp_num"]), int(cfg["up_factor"]),
            int(cfg["input_clip"]), state["input_divisor"].numpy(),
            state["weight_q"].reshape(self.channels * 16, self.z_channels).numpy(),
            state["multiplier"].numpy(), state["affine_bias"].numpy())
        meta_cdf, meta_lengths = artifact["cdf_tables"]["meta_prior"][qp]
        # Same quantized-CDF costs as the original optimized selector.
        cdf = torch.as_tensor(meta_cdf, dtype=torch.int64).reshape(64, self.z_channels, -1)
        lengths = torch.as_tensor(meta_lengths, dtype=torch.int64).reshape(64, self.z_channels)
        maximum = lengths - 2
        total = torch.gather(cdf, 2, (lengths - 1)[..., None]).squeeze(2).float()
        values = torch.minimum(torch.arange(128)[None, None], maximum[..., None])
        freq = torch.gather(cdf, 2, values + 1) - torch.gather(cdf, 2, values)
        cost = -torch.log2((freq.float() / total[..., None]).clamp_min(1e-12))
        self.selector = extension.MetaPriorSelector(
            64, self.z_channels, cost.contiguous().numpy(), maximum.int().contiguous().numpy())
        merge = artifact.get("meta_prior_index_merge")
        self.meta_coding = "index_merge"
        self.probabilities = None if merge is None else merge["probabilities"][qp].numpy()
        self.adaptation_shift = 5 if merge is None else int(merge["adaptation_shift"])
        self.encoder, self.decoder = extension.RansEncoder(), extension.RansDecoder()
        self.encoder.set_entropy_coder_parallel(lanes)
        self.decoder.set_entropy_coder_parallel(lanes)
        y_cdf, y_lengths = artifact["cdf_tables"]["y"]
        self.part_offset = 0
        if self.mode == "part_specific":
            y_cdf, y_lengths = validate_part_cdf(
                *artifact["cdf_tables"]["y_part_specific"], y_lengths)
            self.part_offset = 64
        self.cdfs = ((np.asarray(meta_cdf), np.asarray(meta_lengths)),
                     (np.asarray(y_cdf), np.asarray(y_lengths)))
        for coder in (self.encoder, self.decoder):
            for slot, tables in enumerate(self.cdfs):
                coder.set_cdf(*tables, slot)
        self.decoders = {lanes: self.decoder}
        self.indexes = np.empty((2, int(np.prod(self.compact_shape))), dtype=np.uint8)
        self.packed = np.empty_like(self.indexes, dtype=np.int16)
        self.meta_indexes = np.empty((1, ph // 64, pw // 64), dtype=np.uint8)
        self.z_nhwc = np.empty(int(np.prod(self.z_shape)), dtype=np.int16)

    def require_encoding_defaults(self):
        if self.mode != "part_specific":
            raise ValueError("calibrated y CDFs are required; run postprocess.py or codec.training.y_cdf")
        if self.cutoff != 2:
            raise ValueError("normal encoding requires skip-index cutoff 2")

    def compress(self, z, s0, s1, meta_indexes=None):
        self.require_encoding_defaults()
        if meta_indexes is None:
            self.selector.select_into(z, self.meta_indexes, self.threads)
        else:
            np.copyto(self.meta_indexes, meta_indexes)
        self.linear.packed_symbols_into(
            z, s0, s1, self.qp, self.packed[0], self.packed[1], self.threads)
        if self.part_offset:
            # Offset the CDF byte without changing the signed symbol byte.
            self.ext.offset_cdf_symbols(self.packed[1], self.part_offset, self.cutoff)
        np.copyto(self.z_nhwc.reshape(*self.meta_indexes.shape, self.z_channels),
                  z.transpose(0, 2, 3, 1))
        self.encoder.reset()
        retained = []
        for part in (1, 0):
            packed = self.packed[part]
            cutoff = self.cutoff + (self.part_offset if part else 0)
            if self.cutoff < 0:
                self.encoder.encode_y_borrowed(packed)
            elif self.lanes == 1:
                self.encoder.encode_y_skip_borrowed(packed, self.cutoff)
            else:
                # The portable multi-lane syntax partitions retained symbols.
                kept = np.ascontiguousarray(packed[(packed & 255) > cutoff])
                retained.append(kept)  # borrowed inputs must live until flush completes
                self.encoder.encode_y_borrowed(kept)
        self.encoder.encode_z_meta_prior_borrowed(
            self.z_nhwc, self.meta_indexes.reshape(-1), self.z_channels)
        self.encoder.flush()
        raw = bytes(self.encoder.get_encoded_stream())
        payload = ec.wrap_meta_prior_stream(
            raw, self.meta_indexes, bank_count=64, coding=self.meta_coding,
            index_merge_probabilities=self.probabilities, adaptation_shift=self.adaptation_shift)
        return bitstream.pack({**self.metadata, "entropy_lanes": self.lanes}, payload,
                              fingerprint=self.fingerprint)

    def decompress(self, stream, z, s0, s1, on_decoded=None):
        metadata, payload = bitstream.unpack(stream)
        lanes = metadata.pop("entropy_lanes", 1)
        if metadata.pop("model_id") != bytes.fromhex(self.fingerprint)[:16]:
            raise ValueError("bitstream was encoded with a different model bundle")
        if metadata != self.metadata:
            raise ValueError("bitstream shape/QP/CDF settings do not match this runtime")
        indexes, raw = ec.unwrap_meta_prior_stream(
            payload, self.z_shape, expected_bank_count=64,
            index_merge_probabilities=self.probabilities, adaptation_shift=self.adaptation_shift)
        if lanes not in self.decoders:
            decoder = self.ext.RansDecoder()
            decoder.set_entropy_coder_parallel(lanes)
            for slot, tables in enumerate(self.cdfs):
                decoder.set_cdf(*tables, slot)
            self.decoders[lanes] = decoder
        self.decoder = self.decoders[lanes]
        self.decoder.set_stream(np.frombuffer(raw, dtype=np.uint8))
        self.decoder.decode_z_meta_prior(z.size, indexes.reshape(-1), self.z_channels)
        decoded = np.asarray(self.decoder.get_decoded_tensor())
        np.copyto(z, decoded.reshape(*self.meta_indexes.shape, self.z_channels).transpose(0, 3, 1, 2))
        if on_decoded is not None:
            on_decoded(0)
        self.linear.packed_indexes_into(z, self.qp, self.indexes, self.threads)
        if self.part_offset:
            self.ext.offset_cdf_indexes(self.indexes[1], self.part_offset, self.cutoff)
        for part, output in enumerate((s0, s1)):
            cutoff = self.cutoff + (self.part_offset if part else 0)
            if self.cutoff < 0:
                decoded = self.decoder.decode_and_get_y_borrowed(self.indexes[part])
                self.ext.narrow_i16_to_i8_into(np.asarray(decoded), output.reshape(-1), self.threads)
            elif lanes == 1:
                decoded = self.decoder.decode_and_get_y_skip_borrowed(self.indexes[part], self.cutoff)
                self.ext.narrow_i16_to_i8_into(np.asarray(decoded), output.reshape(-1), self.threads)
            else:
                keep = self.indexes[part] > cutoff
                decoded = self.decoder.decode_and_get_y_borrowed(np.ascontiguousarray(self.indexes[part][keep]))
                output.fill(0)
                output.reshape(-1)[keep] = np.asarray(decoded).astype(np.int8)
            if on_decoded is not None:
                on_decoded(part + 1)
