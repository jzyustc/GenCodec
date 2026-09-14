"""PULSE hyperprior and two-step entropy model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from codec.models.entropy.entropy_coding import (
    DEFAULT_Y_SKIP_CONTEXT_MODE,
    DEFAULT_Y_SKIP_INDEX_CUTOFF,
    Y_SKIP_CONTEXT_MODES,
    default_meta_prior_index_merge_probabilities,
)
from codec.models.entropy.int8scale import ScaleDecoder
from codec.models.entropy.y_cdf import resolve_y_cdf, part1_packed, part1_indexes
from codec.models.entropy.entropy_models import (
    BitEstimator,
    GaussianEncoder,
    LowerBound,
    quantize_ste as ste_round,
)
from codec.models.modules.blocks import DCB, _normalize_dico_block_variant

__all__ = [
    "EMStep1",
    "EMStep2",
    "HyperDecoder",
    "HyperEncoder",
    "HyperTwoStepEM",
]


class HyperDownBlock(nn.Module):
    """Stride-2 down via pixel_unshuffle + 1x1, then wrapped DCB."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        self.down = nn.Conv2d(in_ch * 4, out_ch, 1)
        self.dcb = DCB(
            out_ch,
            out_ch,
            dico_block_variant=dico_block_variant,
            dico_activation=dico_activation,
        )

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.down(x)
        return self.dcb(x)


class HyperUpBlock(nn.Module):
    """Subpel-2x up via 1x1 + pixel_shuffle, then wrapped DCB."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pointwise_groups: int = 1,
        use_rmsnorm: bool = True,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        self.up = nn.Conv2d(in_ch, out_ch * 4, 1)
        self.dcb = DCB(
            out_ch,
            out_ch,
            pointwise_groups=pointwise_groups,
            use_rmsnorm=use_rmsnorm,
            dico_block_variant=dico_block_variant,
            dico_activation=dico_activation,
        )

    def forward(self, x):
        x = self.up(x)
        x = F.pixel_shuffle(x, 2)
        return self.dcb(x)


class HyperEncoder(nn.Module):
    """y(M) -> z(z_ch). 1x DCB + 2x stride-2 (z at y/4 each dim, 1/16 area).
    Per-qp `quant_step_z` (z_ch) multiplied after every block."""

    def __init__(
        self,
        M: int,
        z_ch: int,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        self.body = nn.Sequential(
            DCB(
                M,
                z_ch,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            ),
            HyperDownBlock(
                z_ch,
                z_ch,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            ),
            HyperDownBlock(
                z_ch,
                z_ch,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            ),
        )

    def forward(self, x, quant_step_z=None):
        for blk in self.body:
            x = blk(x)
            if quant_step_z is not None:
                x = x * quant_step_z.to(x.dtype)
        return x


class HyperDecoder(nn.Module):
    """z(z_ch) at y/4 -> y-resolution M-channel hyper feature."""

    def __init__(
        self,
        M: int,
        z_ch: int,
        pointwise_groups: int = 1,
        use_rmsnorm: bool = True,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        self.body = nn.Sequential(
            HyperUpBlock(
                z_ch,
                z_ch,
                pointwise_groups=pointwise_groups,
                use_rmsnorm=use_rmsnorm,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            ),
            HyperUpBlock(
                z_ch,
                z_ch,
                pointwise_groups=pointwise_groups,
                use_rmsnorm=use_rmsnorm,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            ),
            nn.Conv2d(z_ch, M, 1),
        )

    def forward(self, x, quant_step_z=None, quant_step_m=None):
        x = self.body[0](x)
        if quant_step_z is not None:
            x = x * quant_step_z.to(x.dtype)
        x = self.body[1](x)
        if quant_step_z is not None:
            x = x * quant_step_z.to(x.dtype)
        x = self.body[2](x)
        if quant_step_m is not None:
            x = x * quant_step_m.to(x.dtype)
        return x


# =============================================================================
# Two-step entropy-model modules
# =============================================================================


class EMStep1(nn.Module):
    """Fuse hyper_dec output + per-qp q_basic to (qstep || means). M -> 2M.
    Decoupled mu/sigma: this branch outputs only qstep and means; scales come
    from a separate `ScaleDecoder` directly off z_hat. `quant_step_2m` gates
    the inner DCB; the 2M output is raw entropy params and is NOT gated.

    `channel`: when set (and != M), route through a narrower internal width to
    keep entropy MAC constant while M widens. In→inner (1x1), DCB, inner→2M (1x1).
    """

    def __init__(
        self,
        M: int,
        channel: int = None,
        pointwise_groups: int = 1,
        use_rmsnorm: bool = True,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        inner = channel if channel is not None else M
        self._has_inner = inner != M
        if self._has_inner:
            self.in_proj = nn.Conv2d(M, inner, 1)
            self.dcb = DCB(
                inner,
                inner,
                pointwise_groups=pointwise_groups,
                use_rmsnorm=use_rmsnorm,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            )
            self.out_proj = nn.Conv2d(inner, 2 * M, 1)
        else:
            self.conv = nn.Sequential(
                DCB(
                    M,
                    M,
                    pointwise_groups=pointwise_groups,
                    dico_block_variant=dico_block_variant,
                    dico_activation=dico_activation,
                ),
                nn.Conv2d(M, 2 * M, 1),
            )

    def forward(self, params, q_basic, quant_step_2m=None):
        if self._has_inner:
            x = self.in_proj(params * q_basic)
            x = self.dcb(x)
            if quant_step_2m is not None:
                x = x * quant_step_2m.to(x.dtype)
            return self.out_proj(x)
        x = self.conv[0](params * q_basic)
        if quant_step_2m is not None:
            x = x * quant_step_2m.to(x.dtype)
        return self.conv[1](x)


class EMStep2(nn.Module):
    """Part-1 EM step 2. Consumes cat(y_hat_0, common); common is 2M
    (qstep || means_0_full). Outputs refined means_1 (scales come from
    ScaleDecoder). `quant_step_2m` gates intermediate DCBs; final M output
    is NOT gated (additively combined with means_0).

    `channel`: when set, narrower internal width. cat(y_hat_0, common) is 3M,
    projected to inner first.
    """

    def __init__(
        self,
        M: int,
        channel: int = None,
        pointwise_groups: int = 1,
        use_rmsnorm: bool = True,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        inner = channel if channel is not None else M
        self._has_inner = inner != M
        if self._has_inner:
            self.in_proj = nn.Conv2d(3 * M, inner, 1)
            self.dcb = DCB(
                inner,
                inner,
                pointwise_groups=pointwise_groups,
                use_rmsnorm=use_rmsnorm,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            )
            self.out_proj = nn.Conv2d(inner, M, 1)
        else:
            self.conv = nn.Sequential(
                DCB(
                    3 * M,
                    M,
                    pointwise_groups=pointwise_groups,
                    dico_block_variant=dico_block_variant,
                    dico_activation=dico_activation,
                ),
                nn.Conv2d(M, M, 1),
            )

    def forward(self, y_hat_0, common, quant_step_2m=None):
        cat = torch.cat([y_hat_0, common], dim=1)
        if self._has_inner:
            x = self.in_proj(cat)
            x = self.dcb(x)
            if quant_step_2m is not None:
                x = x * quant_step_2m.to(x.dtype)
            return self.out_proj(x)
        x = self.conv[0](cat)
        if quant_step_2m is not None:
            x = x * quant_step_2m.to(x.dtype)
        return self.conv[1](x)


# =============================================================================
class TargetRateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _calc_bits_per_batch(self, likelihoods: Tensor) -> Tensor:
        # fp32 cast + LowerBound STE clamps (prob >= 1e-6, bits >= 0) — mirror
        # DCVC's prob_to_bits; bf16 autocast loses precision in log/clamp.
        batch_size = likelihoods.shape[0]
        likelihoods = likelihoods.reshape(batch_size, -1)
        dtype = likelihoods.dtype
        prob = likelihoods.float()
        bits = torch.log(LowerBound.apply(prob, 1e-6)) / -math.log(2)
        bits = LowerBound.apply(bits, 0.0)
        return bits.sum(1).to(dtype)

    def forward(
        self,
        num_pixels,
        latent_likelihoods: Tensor,
        hyper_latent_likelihoods: Tensor,
        return_parts: bool = False,
    ):
        latent_bpp = self._calc_bits_per_batch(latent_likelihoods) / num_pixels
        hyper_bpp = self._calc_bits_per_batch(hyper_latent_likelihoods) / num_pixels
        total_bpp = latent_bpp + hyper_bpp
        if return_parts:
            return total_bpp, latent_bpp, hyper_bpp
        return total_bpp


class HyperTwoStepEM(nn.Module):
    """Map y to rate metadata/bitstreams and decode bitstreams back to y_hat."""

    def __init__(
        self,
        *,
        M: int,
        z_ch: int,
        entropy_channels: int,
        dico_block_variant: str,
    ):
        super().__init__()
        self.M, self.z_ch, self.qp_num = int(M), int(z_ch), 8
        self.q_scale_lower_bound = 0.0
        self.dico_block_variant = _normalize_dico_block_variant(dico_block_variant)
        self.em_step1_channel = self.em_step2_channel = int(entropy_channels)

        self.hyper_enc = HyperEncoder(
            self.M,
            self.z_ch,
            dico_block_variant=self.dico_block_variant,
            dico_activation="gelu",
        )
        self.hyper_dec = HyperDecoder(
            self.M,
            self.z_ch,
            pointwise_groups=4,
            use_rmsnorm=False,
            dico_block_variant=self.dico_block_variant,
            dico_activation="hard_gelu_030",
        )
        self.em_step1 = EMStep1(
            self.M,
            channel=self.em_step1_channel,
            pointwise_groups=4,
            use_rmsnorm=False,
            dico_block_variant=self.dico_block_variant,
            dico_activation="hard_gelu_030",
        )
        self.em_step2 = EMStep2(
            self.M,
            channel=self.em_step2_channel,
            pointwise_groups=4,
            use_rmsnorm=False,
            dico_block_variant=self.dico_block_variant,
            dico_activation="hard_gelu_030",
        )
        self.scale_dec = ScaleDecoder(
            self.M,
            self.z_ch,
            self.qp_num,
        )

        self.entropy_bottleneck = BitEstimator(
            qp_num=self.qp_num,
            channel=self.z_ch,
        )
        self.meta_prior = None
        self.gaussian_conditional = GaussianEncoder()
        self.q_scale_hyper_z = nn.Parameter(torch.ones(self.qp_num, 1))
        self.q_scale_hyper_m = nn.Parameter(torch.ones(self.qp_num, 1))
        self.q_scale_prior_2m = nn.Parameter(torch.ones(self.qp_num, 1))
        self.q_basic = nn.Parameter(torch.ones(self.qp_num, 1))

        self.masks = {}
        self.rate = TargetRateModule()

    @property
    def meta_prior_enabled(self) -> bool:
        return (
            self.meta_prior is not None
            and bool(self.meta_prior.ready.item())
        )

    def _z_likelihood(self, z_for_bit: Tensor, qp_idx):
        """Return z probability and optional Meta Prior bank indexes."""
        if self.training or not self.meta_prior_enabled:
            return self.entropy_bottleneck(z_for_bit, qp_idx), None
        return self.meta_prior(z_for_bit, qp_idx)

    def _meta_prior_cdf(self, qp_int: int):
        from codec.models.entropy import entropy_coding as ec

        if not self.meta_prior_enabled:
            raise RuntimeError("Meta Prior is not available")
        cached = getattr(self, "_cached_meta_prior_cdfs", None)
        if cached is not None:
            return cached[int(qp_int)]
        runtime = getattr(self, "_runtime_meta_prior_cdfs", None)
        if runtime is None:
            runtime = {}
            self._runtime_meta_prior_cdfs = runtime
        if int(qp_int) not in runtime:
            runtime[int(qp_int)] = ec.update_meta_prior_cdf(
                self.meta_prior,
                int(qp_int),
            )
        return runtime[int(qp_int)]

    def _meta_prior_cost_table(self, qp_int: int):
        from codec.models.entropy import entropy_coding as ec

        runtime = getattr(self, "_runtime_meta_prior_costs", None)
        if runtime is None:
            runtime = {}
            self._runtime_meta_prior_costs = runtime
        if int(qp_int) not in runtime:
            cdf, lengths = self._meta_prior_cdf(int(qp_int))
            runtime[int(qp_int)] = ec.factorized_cdf_cost_table(
                cdf,
                lengths,
            )
        return runtime[int(qp_int)]

    def _meta_prior_index_merge_parameters(self, qp_int: int):
        """Return the QP-specific Index Merge probability table and update."""
        cached = getattr(
            self,
            "_cached_meta_prior_index_merge",
            None,
        )
        if cached is not None:
            probabilities = cached["probabilities"]
            return (
                probabilities[int(qp_int)],
                int(cached["adaptation_shift"]),
            )
        meta_prior = getattr(self, "meta_prior", None)
        if (
            meta_prior is not None
            and bool(
                getattr(
                    meta_prior,
                    "index_merge_ready",
                    torch.tensor(False),
                ).item()
            )
        ):
            return (
                meta_prior.index_merge_probabilities[int(qp_int)]
                .detach()
                .cpu()
                .numpy(),
                int(meta_prior.index_merge_adaptation_shift.item()),
            )
        # A deterministic P=0.5 initialization remains decodable without a
        # fitted sidecar. Release artifacts install a clean-trained table.
        if self.meta_prior_enabled:
            return default_meta_prior_index_merge_probabilities(meta_prior.bank_count), 5
        return None, 5

    def _resolve_qp(self, qp, batch_size, device):
        """Normalize qp to either an int (single-slot) or a 1-D long index tensor."""
        if qp is None:
            return 0
        if isinstance(qp, int):
            return qp
        qp = torch.as_tensor(qp, device=device, dtype=torch.long)
        if qp.dim() == 0:
            return int(qp.item())
        assert qp.shape == (batch_size,), (
            f"qp tensor must be scalar or shape ({batch_size},), got {tuple(qp.shape)}"
        )
        return qp

    def _select_q_scale(self, param: Tensor, qp_idx) -> Tensor:
        """Index a (qp_num, 1) scalar gate, return (B_or_1, 1, 1, 1)."""
        if isinstance(qp_idx, int):
            return param[qp_idx : qp_idx + 1, :, None, None]
        return torch.index_select(param, 0, qp_idx)[:, :, None, None]

    # LowerBound STE keeps q_scale >= bound but lets gradient flow above it —
    # blocks the MSE+LPIPS degenerate attractor from sign-flipping the gate.

    def _select_q_scale_bounded(self, param: Tensor, qp_idx) -> Tensor:
        """Like _select_q_scale, but clamps to >= self.q_scale_lower_bound with STE grad."""
        return LowerBound.apply(
            self._select_q_scale(param, qp_idx), self.q_scale_lower_bound
        )

    def _q_basic_feature(self, qp_idx, batch_size, H, W) -> Tensor:
        """Broadcast per-qp scalar q_basic to (B, 1, H, W). LowerBound >=0 STE."""
        qb = LowerBound.apply(self._select_q_scale(self.q_basic, qp_idx), 0.0)
        return qb.expand(batch_size, qb.shape[1], H, W)

    @staticmethod
    def _validate_packed_scale_indexes(
        indexes,
        *,
        keep_mask: Tensor,
        part: int,
    ) -> Tensor:
        indexes = torch.as_tensor(indexes)
        if indexes.dtype != torch.uint8:
            raise TypeError(
                f"packed scale indexes for part {part} must be UINT8, "
                f"got {indexes.dtype}"
            )
        if indexes.ndim != 1:
            raise ValueError(
                f"packed scale indexes for part {part} must be flat, "
                f"got {tuple(indexes.shape)}"
            )
        expected = int(keep_mask.count_nonzero().item())
        if indexes.numel() != expected:
            raise ValueError(
                f"packed scale index size mismatch for part {part}: "
                f"expected {expected}, got {indexes.numel()}"
            )
        if indexes.numel() and int(indexes.max().item()) >= 64:
            raise ValueError(
                f"packed scale indexes for part {part} contain index >= 64"
            )
        return indexes.detach().to(device="cpu").contiguous()

    @staticmethod
    def _pack_symbols_with_indexes(
        symbols_i8: Tensor,
        indexes_u8: Tensor,
        keep_mask: Tensor,
    ):
        symbols = symbols_i8.permute(0, 2, 3, 1).reshape(-1)
        keep = keep_mask.permute(0, 2, 3, 1).reshape(-1).bool()
        symbols = symbols[keep].to(device="cpu", dtype=torch.int16)
        if symbols.numel() != indexes_u8.numel():
            raise ValueError(
                "packed symbol/index size mismatch: "
                f"{symbols.numel()} != {indexes_u8.numel()}"
            )
        return (
            (symbols << 8) + indexes_u8.to(torch.int16)
        ).contiguous().numpy()

    # -------------------------------------------------------------------------
    # 2-part prior helpers (forward / compress / decompress share the math)
    # -------------------------------------------------------------------------

    @staticmethod
    def _split_common(common):
        """Chunk common_params into (qstep, means); derive q_enc=1/qstep,
        q_dec=qstep with qstep clamped to >= 0.5. Scales come from scale_dec."""
        qstep, means = common.chunk(2, 1)
        qstep = LowerBound.apply(qstep, 0.5)
        q_enc = 1.0 / qstep
        q_dec = qstep
        return q_enc, q_dec, means

    # -------------------------------------------------------------------------
    # forward (training / inference)
    # -------------------------------------------------------------------------

    def forward(self, y, num_pixels, qp=None):
        batch = y.shape[0]
        qp_idx = self._resolve_qp(qp, batch, y.device)

        q_hyper_z = self._select_q_scale_bounded(self.q_scale_hyper_z, qp_idx)
        q_hyper_m = self._select_q_scale_bounded(self.q_scale_hyper_m, qp_idx)
        q_prior_2m = self._select_q_scale_bounded(self.q_scale_prior_2m, qp_idx)

        z = self.hyper_enc(y, q_hyper_z)

        # z prior: Ballé/Minnen two-path (factorized BitEstimator).
        z_hat = ste_round(z)
        if self.training:
            z_for_bit = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_for_bit = z_hat
        z_likelihoods, meta_prior_index = self._z_likelihood(
            z_for_bit,
            qp_idx,
        )

        params = self.hyper_dec(z_hat, q_hyper_z, q_hyper_m)  # (B, M,  H_y, W_y)
        B_p, _, H_p, W_p = params.shape
        q_basic_feat = self._q_basic_feature(qp_idx, B_p, H_p, W_p)
        common = self.em_step1(
            params, q_basic_feat, q_prior_2m
        )  # (B, 2M, H_y, W_y)
        q_enc, q_dec, means_0_full = self._split_common(common)
        # The continuous CDF coordinate is mapped to sigma only for the
        # differentiable likelihood used by Linear CDF Index Decoding.
        scales = self.scale_dec(z_hat, qp_idx)  # (B, M, H_y, W_y)
        y = y * q_enc

        B_y, C_y, H_y, W_y = y.shape
        mask_0, mask_1 = self.get_mask_two_parts(B_y, C_y, H_y, W_y, device=y.device)

        # Part 0: scales/means from common.
        means_0 = means_0_full * mask_0
        scales_0 = scales * mask_0
        y_hat_0 = ste_round(y * mask_0 - means_0) + means_0

        # Part 1: means refined by EM step 2; scales are reused.
        means_1_full = self.em_step2(y_hat_0, common, q_prior_2m)
        means_1 = means_1_full * mask_1
        scales_1 = scales * mask_1
        y_hat_1 = ste_round(y * mask_1 - means_1) + means_1

        means_all = means_0 + means_1
        scales_all = scales_0 + scales_1

        y_centered = y - means_all
        if self.training:
            y_for_bit = y_centered + torch.empty_like(y_centered).uniform_(-0.5, 0.5)
        else:
            y_for_bit = ste_round(y_centered)
        y_likelihoods = self.gaussian_conditional(y_for_bit, scales_all)

        y_hat = (y_hat_0 + y_hat_1) * q_dec

        total_bpp, y_bpp, z_bpp = self.rate(
            num_pixels=num_pixels,
            latent_likelihoods=y_likelihoods,
            hyper_latent_likelihoods=z_likelihoods,
            return_parts=True,
        )
        if meta_prior_index is None:
            meta_prior_side_bpp = torch.zeros_like(z_bpp).float()
        else:
            # Keep the differentiable/non-rANS estimate as a conservative
            # fixed-width charge. Actual streams use Index Merge Coding with
            # a fixed-width fallback and are measured by byte length.
            meta_prior_side_bpp = (
                torch.full(
                    (batch,),
                    float(self.meta_prior.selector_bits),
                    device=z_bpp.device,
                    dtype=torch.float32,
                )
                * meta_prior_index.shape[-2]
                * meta_prior_index.shape[-1]
                / num_pixels
            )
            z_bpp = z_bpp.float() + meta_prior_side_bpp
            total_bpp = total_bpp.float() + meta_prior_side_bpp

        return y_hat, {
            "sq_loss": total_bpp,
            "y_bpp": y_bpp,
            "z_bpp": z_bpp,
            "meta_prior_side_bpp": meta_prior_side_bpp,
        }

    def inference(self, y, num_pixels, qp=None):
        return self.forward(y, num_pixels=num_pixels, qp=qp)

    # -------------------------------------------------------------------------
    # Real entropy coding: rANS encode/decode (MLCodec_extensions_cpp)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def compress(
        self,
        y,
        qp=None,
        skip_index_cutoff=DEFAULT_Y_SKIP_INDEX_CUTOFF,
        skip_context_mode=DEFAULT_Y_SKIP_CONTEXT_MODE,
        y_cdf_mode="part_specific",
        collect_y_symbols=False,
    ):
        """Encode y to a real bitstream. Mirrors `forward` but routes z and y
        through the vendored rANS coder.

        Returns a dict with `stream` (bytes), `z_shape`, `y_shape`, `qp`, and
        `y_hat_local` — the y_hat computed from the *clamped* symbols that
        were actually encoded. `decompress(stream, ...)` is required to return
        the same tensor bit-exactly; if it doesn't the rANS round-trip is
        broken. `y_hat_local` may differ from `forward()`'s y_hat because the
        forward path doesn't clamp to the entropy coder's [-MAX, MAX] range.

        By default, CDF rows corresponding to scale <= 0.15 are omitted. The
        production ``entropy_only`` policy keeps the original part-1 residual
        at the sender and applies the deterministic skip only in entropy
        transport; ``exact`` recomputes that residual from the skipped part-0
        context.
        """
        from codec.models.entropy import entropy_coding as ec

        y = y.float()
        B = y.shape[0]
        assert B == 1, "compress() is single-image. Loop externally for batches."
        qp_idx = self._resolve_qp(qp, B, y.device)
        qp_int = qp_idx if isinstance(qp_idx, int) else int(qp_idx[0].item())
        if skip_index_cutoff is not None:
            skip_index_cutoff = int(skip_index_cutoff)
            if not 0 <= skip_index_cutoff < 64:
                raise ValueError("skip_index_cutoff must be in [0, 63]")
        skip_context_mode = str(skip_context_mode).strip().lower()
        if skip_context_mode not in Y_SKIP_CONTEXT_MODES:
            raise ValueError(
                "skip_context_mode must be one of "
                f"{Y_SKIP_CONTEXT_MODES}, got {skip_context_mode!r}"
            )

        q_hyper_z = self._select_q_scale_bounded(self.q_scale_hyper_z, qp_idx).float()
        q_hyper_m = self._select_q_scale_bounded(self.q_scale_hyper_m, qp_idx).float()
        q_prior_2m = self._select_q_scale_bounded(self.q_scale_prior_2m, qp_idx).float()
        z = self.hyper_enc(y, q_hyper_z)
        # Preserve z exactly. The rANS CDF has a compact modeled core, while
        # values outside that core are carried losslessly by bypass bits using
        # the int16 z-symbol transport. Clipping z here would also corrupt the
        # hyper-decoder input and causes large high-rate RD loss.
        z_hat = ste_round(z)

        coder = ec.EntropyCoder()
        meta_prior_indexes = None
        if self.meta_prior_enabled:
            z_cdf, z_cdf_len = self._meta_prior_cdf(qp_int)
            meta_prior_indexes = ec.select_meta_prior_indexes_from_cdf(
                z_hat,
                z_cdf,
                z_cdf_len,
                bank_count=self.meta_prior.bank_count,
            )
        elif hasattr(self, "_cached_z_cdfs"):
            z_cdf, z_cdf_len = self._cached_z_cdfs[qp_int]
        else:
            z_cdf, z_cdf_len = ec.update_z_cdf(self.entropy_bottleneck, qp_int)
        y_cdf_mode, y_cdf, y_cdf_len = resolve_y_cdf(self, y_cdf_mode)
        coder.set_z_cdf(z_cdf, z_cdf_len)
        coder.set_y_cdf(y_cdf, y_cdf_len)
        coder.reset()

        params = self.hyper_dec(z_hat, q_hyper_z, q_hyper_m)
        B_p, _, H_p, W_p = params.shape
        q_basic_feat = self._q_basic_feature(qp_int, B_p, H_p, W_p)
        common = self.em_step1(params, q_basic_feat, q_prior_2m)
        q_enc, q_dec, means_0_full = self._split_common(common)
        y = y * q_enc

        B_y, C_y, H_y, W_y = y.shape
        mask_0, mask_1 = self.get_mask_two_parts(B_y, C_y, H_y, W_y, device=y.device)
        packed_scale_indexes = None
        if hasattr(self.scale_dec, "packed_indexes"):
            raw_part0, raw_part1 = self.scale_dec.packed_indexes(
                z_hat,
                qp_int,
            )
            packed_scale_indexes = (
                self._validate_packed_scale_indexes(
                    raw_part0,
                    keep_mask=mask_0,
                    part=0,
                ),
                self._validate_packed_scale_indexes(
                    raw_part1,
                    keep_mask=mask_1,
                    part=1,
                ),
            )
            s_idx_full = None
        else:
            s_idx_full = self.scale_dec.scale_bucket_idx(z_hat, qp_int)
        if skip_index_cutoff is not None:
            if packed_scale_indexes is None:
                skip_source_indexes = (
                    ec.select_y_indexes(s_idx_full, keep_mask=mask_0),
                    ec.select_y_indexes(s_idx_full, keep_mask=mask_1),
                )
            else:
                skip_source_indexes = packed_scale_indexes
            skip_layouts = (
                ec.select_y_indexes_nchw_offsets_cpu(
                    skip_source_indexes[0],
                    skip_index_cutoff,
                    y.shape,
                    part=0,
                ),
                ec.select_y_indexes_nchw_offsets_cpu(
                    skip_source_indexes[1],
                    skip_index_cutoff,
                    y.shape,
                    part=1,
                ),
            )
        else:
            skip_layouts = None

        def pack_skipped(centered: Tensor, layout):
            selected_indexes, selected_offsets = layout
            if centered.device.type == "cpu":
                packed, _outside = ec.pack_y_symbols_nchw_offsets_cpu(
                    centered,
                    selected_indexes,
                    selected_offsets,
                )
                effective_numpy = torch.zeros(
                    tuple(centered.shape),
                    dtype=torch.float32,
                ).numpy()
                ec.restore_y_symbols_nchw_offsets_into_cpu(
                    effective_numpy,
                    (packed >> 8).astype("int16", copy=False),
                    selected_offsets,
                )
                effective = torch.from_numpy(effective_numpy)
            else:
                offsets_device = torch.from_numpy(
                    selected_offsets.astype("int64", copy=False)
                ).to(centered.device)
                selected = (
                    centered.reshape(-1)
                    .index_select(0, offsets_device)
                    .round()
                    .clamp(
                        -ec.MAX_ENTROPY_CODING_VALUE,
                        ec.MAX_ENTROPY_CODING_VALUE,
                    )
                    .to(torch.int8)
                )
                packed = (
                    (selected.to(device="cpu", dtype=torch.int16) << 8)
                    + torch.from_numpy(selected_indexes).to(torch.int16)
                ).contiguous().numpy()
                effective = torch.zeros_like(
                    centered,
                    dtype=torch.float32,
                )
                effective.reshape(-1).index_copy_(
                    0,
                    offsets_device,
                    selected.float(),
                )
            return packed, effective

        # Part 0. Entropy-only skip intentionally keeps the full sender-side
        # context for part 1 while transmitting only the selected symbols.
        m0 = means_0_full * mask_0
        y_centered_0 = (y - m0) * mask_0
        y_sym_0 = ec.quantize_y_centered(y_centered_0)
        if skip_layouts is not None:
            packed_0, y_sym_0_effective = pack_skipped(
                y_sym_0,
                skip_layouts[0],
            )
        else:
            if packed_scale_indexes is not None:
                if y_sym_0.device.type == "cpu":
                    packed_0 = ec.pack_y_symbols_checkerboard_cpu(
                        y_sym_0,
                        packed_scale_indexes[0],
                        part=0,
                    )
                else:
                    packed_0 = self._pack_symbols_with_indexes(
                        y_sym_0,
                        packed_scale_indexes[0],
                        mask_0,
                    )
            else:
                packed_0 = ec.pack_y_symbols(
                    y_sym_0,
                    s_idx_full,
                    keep_mask=mask_0,
                )
        if skip_layouts is None:
            y_sym_0_effective = y_sym_0.float()
        y_hat_0_decoder = (y_sym_0_effective + m0) * mask_0
        if skip_layouts is not None and skip_context_mode == "entropy_only":
            y_hat_0_encoder = (y_sym_0.float() + m0) * mask_0
        else:
            y_hat_0_encoder = y_hat_0_decoder

        # Part 1: means refined; scales reused.
        means_1_encoder = self.em_step2(
            y_hat_0_encoder,
            common,
            q_prior_2m,
        ) * mask_1
        y_centered_1 = (y - means_1_encoder) * mask_1
        y_sym_1 = ec.quantize_y_centered(y_centered_1)
        if skip_layouts is not None:
            packed_1, y_sym_1_effective = pack_skipped(
                y_sym_1,
                skip_layouts[1],
            )
        else:
            if packed_scale_indexes is not None:
                if y_sym_1.device.type == "cpu":
                    packed_1 = ec.pack_y_symbols_checkerboard_cpu(
                        y_sym_1,
                        packed_scale_indexes[1],
                        part=1,
                    )
                else:
                    packed_1 = self._pack_symbols_with_indexes(
                        y_sym_1,
                        packed_scale_indexes[1],
                        mask_1,
                    )
            else:
                packed_1 = ec.pack_y_symbols(
                    y_sym_1,
                    s_idx_full,
                    keep_mask=mask_1,
                )

        # Local y_hat for round-trip verification (decompress should reproduce
        # this bit-exactly). Uses the *encoded* (clamped) symbols, so it can
        # differ from forward() which doesn't clamp — see compress() docstring.
        if skip_layouts is None:
            y_sym_1_effective = y_sym_1.float()
        if skip_layouts is not None and skip_context_mode == "entropy_only":
            means_1_decoder = self.em_step2(
                y_hat_0_decoder,
                common,
                q_prior_2m,
            ) * mask_1
        else:
            means_1_decoder = means_1_encoder
        y_hat_1 = (y_sym_1_effective + means_1_decoder) * mask_1
        y_hat_local = (y_hat_0_decoder + y_hat_1) * q_dec

        # rANS is LIFO — emit in reverse so decode reads in forward order.
        # These indexes came from the validated/clamped scale decoder. Avoid
        # rescanning every transported symbol just to repeat its range check.
        coder.encode_y(
            part1_packed(packed_1, validate=False)
            if y_cdf_mode == "part_specific" else packed_1
        )
        coder.encode_y(packed_0)
        z_syms = ec.quantize_z(z_hat)
        if meta_prior_indexes is None:
            coder.encode_z(
                z_syms,
                qp_int * self.entropy_bottleneck.channel,
                self.entropy_bottleneck.channel,
            )
        else:
            coder.encode_z_meta_prior(
                z_syms,
                meta_prior_indexes.reshape(-1).cpu().numpy(),
                self.entropy_bottleneck.channel,
            )

        stream = coder.flush()
        selector_transport_bytes = 0
        selector_coding = "disabled"
        if meta_prior_indexes is not None:
            raw_rans_stream = stream
            (
                index_merge_probabilities,
                index_merge_adaptation_shift,
            ) = self._meta_prior_index_merge_parameters(qp_int)
            stream = ec.wrap_meta_prior_stream(
                stream,
                meta_prior_indexes.cpu().numpy(),
                bank_count=self.meta_prior.bank_count,
                spatial_shape=tuple(meta_prior_indexes.shape),
                index_merge_probabilities=index_merge_probabilities,
                adaptation_shift=index_merge_adaptation_shift,
                coding="index_merge",
            )
            selector_transport_bytes = len(stream) - len(raw_rans_stream)
            selector_coding = (
                "index_merge_v1"
                if stream[4] == ec.META_PRIOR_INDEX_MERGE_VERSION
                else "fixed_width_v1"
            )
        result = {
            "stream": stream,
            "z_shape": tuple(z.shape),
            "y_shape": tuple(y.shape),
            "qp": qp_int,
            "y_hat_local": y_hat_local,
            "skip_index_cutoff": (
                -1 if skip_index_cutoff is None else skip_index_cutoff
            ),
            "skip_context_mode": (
                "disabled"
                if skip_index_cutoff is None
                else skip_context_mode
            ),
            "y_symbols_full": int(C_y * H_y * W_y),
            "y_symbols_kept": (
                int(C_y * H_y * W_y)
                if skip_layouts is None
                else int(
                    skip_layouts[0][0].size
                    + skip_layouts[1][0].size
                )
            ),
            "meta_prior": meta_prior_indexes is not None,
            "meta_prior_selector_bits": (
                0
                if meta_prior_indexes is None
                else int(
                    meta_prior_indexes.numel()
                    * self.meta_prior.selector_bits
                )
            ),
            "meta_prior_selector_coded_bits": (
                int(selector_transport_bytes * 8)
            ),
            "meta_prior_selector_coding": (
                selector_coding
            ),
            "y_cdf_mode": y_cdf_mode,
        }
        if collect_y_symbols:
            result["y_cdf_samples"] = (packed_0, packed_1)
        return result

    @torch.no_grad()
    def decompress(
        self,
        stream: bytes,
        z_shape,
        y_shape,
        qp: int = 0,
        skip_index_cutoff=DEFAULT_Y_SKIP_INDEX_CUTOFF,
        y_cdf_mode=None,
        entropy_lanes=1,
        entropy_sections=None,
        entropy_dense=False,
    ):
        """Decode a bitstream produced by `compress`. Returns pre-decode latent y_hat."""
        from codec.models.entropy import entropy_coding as ec

        device = next(self.parameters()).device
        qp_int = int(qp)
        if skip_index_cutoff is not None:
            skip_index_cutoff = int(skip_index_cutoff)
            if not 0 <= skip_index_cutoff < 64:
                raise ValueError("skip_index_cutoff must be in [0, 63]")
        q_hyper_z = self._select_q_scale_bounded(self.q_scale_hyper_z, qp_int)
        q_hyper_m = self._select_q_scale_bounded(self.q_scale_hyper_m, qp_int)
        q_prior_2m = self._select_q_scale_bounded(self.q_scale_prior_2m, qp_int)
        (
            index_merge_probabilities,
            index_merge_adaptation_shift,
        ) = self._meta_prior_index_merge_parameters(qp_int)
        if (
            len(stream) >= ec.META_PRIOR_HEADER.size
            and stream[:4] == ec.META_PRIOR_MAGIC
            and stream[4] == ec.META_PRIOR_INDEX_MERGE_VERSION
            and index_merge_probabilities is None
        ):
            raise RuntimeError(
                "bitstream uses Meta-Prior Index Merge Coding, but the "
                "checkpoint/artifact has no matching context table"
            )
        meta_prior_indexes, rans_stream = ec.unwrap_meta_prior_stream(
            stream,
            z_shape,
            expected_bank_count=(
                self.meta_prior.bank_count
                if self.meta_prior_enabled
                else None
            ),
            index_merge_probabilities=index_merge_probabilities,
            adaptation_shift=index_merge_adaptation_shift,
        )
        if entropy_sections is None:
            coder = ec.EntropyCoder()
            coder.decoder.set_entropy_coder_parallel(entropy_lanes)
        else:
            coder = ec.SplitEntropyDecoder(entropy_sections, dense=entropy_dense, cutoff=skip_index_cutoff)
        if meta_prior_indexes is not None:
            z_cdf, z_cdf_len = self._meta_prior_cdf(qp_int)
        elif hasattr(self, "_cached_z_cdfs"):
            z_cdf, z_cdf_len = self._cached_z_cdfs[qp_int]
        else:
            z_cdf, z_cdf_len = ec.update_z_cdf(self.entropy_bottleneck, qp_int)
        y_cdf_mode, y_cdf, y_cdf_len = resolve_y_cdf(self, y_cdf_mode)
        coder.set_z_cdf(z_cdf, z_cdf_len)
        coder.set_y_cdf(y_cdf, y_cdf_len)
        coder.set_stream(rans_stream)

        # z
        Bz, Cz, Hz, Wz = z_shape
        if meta_prior_indexes is None:
            coder.decode_z(
                Bz * Hz * Wz * Cz,
                qp_int * self.entropy_bottleneck.channel,
                self.entropy_bottleneck.channel,
            )
        else:
            coder.decode_z_meta_prior(
                Bz * Hz * Wz * Cz,
                meta_prior_indexes.reshape(-1),
                self.entropy_bottleneck.channel,
            )
        z_flat = coder.get_decoded(device, torch.float32)
        z_hat = ec.dequantize_z(z_flat, z_shape).to(device)

        params = self.hyper_dec(z_hat, q_hyper_z, q_hyper_m)
        B_p, _, H_p, W_p = params.shape
        q_basic_feat = self._q_basic_feature(qp_int, B_p, H_p, W_p)
        common = self.em_step1(params, q_basic_feat, q_prior_2m)
        _q_enc, q_dec, means_0_full = self._split_common(common)
        By, Cy, Hy, Wy = y_shape
        mask_0, mask_1 = self.get_mask_two_parts(By, Cy, Hy, Wy, device=device)
        packed_scale_indexes = None
        if hasattr(self.scale_dec, "packed_indexes"):
            raw_part0, raw_part1 = self.scale_dec.packed_indexes(
                z_hat,
                qp_int,
            )
            packed_scale_indexes = (
                self._validate_packed_scale_indexes(
                    raw_part0,
                    keep_mask=mask_0,
                    part=0,
                ).numpy(),
                self._validate_packed_scale_indexes(
                    raw_part1,
                    keep_mask=mask_1,
                    part=1,
                ).numpy(),
            )
            s_idx_full = None
        else:
            s_idx_full = self.scale_dec.scale_bucket_idx(z_hat, qp_int)
        if skip_index_cutoff is not None and not entropy_dense:
            if packed_scale_indexes is None:
                skip_source_indexes = (
                    ec.select_y_indexes(s_idx_full, keep_mask=mask_0),
                    ec.select_y_indexes(s_idx_full, keep_mask=mask_1),
                )
            else:
                skip_source_indexes = packed_scale_indexes
            skip_layouts = (
                ec.select_y_indexes_nchw_offsets_cpu(
                    skip_source_indexes[0],
                    skip_index_cutoff,
                    y_shape,
                    part=0,
                ),
                ec.select_y_indexes_nchw_offsets_cpu(
                    skip_source_indexes[1],
                    skip_index_cutoff,
                    y_shape,
                    part=1,
                ),
            )
        else:
            skip_layouts = None

        def restore_skipped(decoded, selected_offsets):
            if device.type == "cpu":
                output_numpy = torch.zeros(
                    y_shape,
                    dtype=torch.float32,
                ).numpy()
                ec.restore_y_symbols_nchw_offsets_into_cpu(
                    output_numpy,
                    decoded,
                    selected_offsets,
                )
                return torch.from_numpy(output_numpy)
            values = torch.as_tensor(
                decoded.copy(),
                device=device,
                dtype=torch.float32,
            )
            offsets_device = torch.from_numpy(
                selected_offsets.astype("int64", copy=False)
            ).to(device)
            output = torch.zeros(
                y_shape,
                device=device,
                dtype=torch.float32,
            )
            output.reshape(-1).index_copy_(
                0,
                offsets_device,
                values,
            )
            return output

        # Part 0
        m0 = means_0_full * mask_0
        if skip_layouts is not None:
            idx_0, offset_0 = skip_layouts[0]
        elif packed_scale_indexes is not None:
            idx_0 = packed_scale_indexes[0]
        else:
            idx_0 = ec.select_y_indexes(s_idx_full, keep_mask=mask_0)
        if skip_layouts is not None:
            sym_0_flat = coder.decode_and_get_y(idx_0)
            y_sym_0 = restore_skipped(sym_0_flat, offset_0)
        elif packed_scale_indexes is not None and device.type == "cpu":
            sym_0_flat = coder.decode_and_get_y(idx_0)
            y_sym_0 = ec.restore_y_symbols_checkerboard_cpu(
                sym_0_flat,
                y_shape,
                part=0,
            )
        else:
            coder.decode_y(idx_0)
            sym_0_flat = coder.get_decoded(device, torch.float32)
            keep_0 = mask_0.permute(0, 2, 3, 1).reshape(-1).bool()
            full_0 = torch.zeros_like(keep_0, dtype=torch.float32, device=device)
            full_0[keep_0] = sym_0_flat
            y_sym_0 = (
                full_0.reshape(By, Hy, Wy, Cy)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        y_hat_0 = (y_sym_0 + m0) * mask_0

        # Part 1: means refined; scales reused.
        means_1_full = self.em_step2(y_hat_0, common, q_prior_2m)
        m1 = means_1_full * mask_1
        if skip_layouts is not None:
            idx_1, offset_1 = skip_layouts[1]
        elif packed_scale_indexes is not None:
            idx_1 = packed_scale_indexes[1]
        else:
            idx_1 = ec.select_y_indexes(s_idx_full, keep_mask=mask_1)
        if y_cdf_mode == "part_specific":
            idx_1 = part1_indexes(idx_1, validate=False)
        if skip_layouts is not None:
            sym_1_flat = coder.decode_and_get_y(idx_1)
            y_sym_1 = restore_skipped(sym_1_flat, offset_1)
        elif packed_scale_indexes is not None and device.type == "cpu":
            sym_1_flat = coder.decode_and_get_y(idx_1)
            y_sym_1 = ec.restore_y_symbols_checkerboard_cpu(
                sym_1_flat,
                y_shape,
                part=1,
            )
        else:
            coder.decode_y(idx_1)
            sym_1_flat = coder.get_decoded(device, torch.float32)
            keep_1 = mask_1.permute(0, 2, 3, 1).reshape(-1).bool()
            full_1 = torch.zeros_like(keep_1, dtype=torch.float32, device=device)
            full_1[keep_1] = sym_1_flat
            y_sym_1 = (
                full_1.reshape(By, Hy, Wy, Cy)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        y_hat_1 = (y_sym_1 + m1) * mask_1

        y_hat = (y_hat_0 + y_hat_1) * q_dec
        return y_hat

    # -------------------------------------------------------------------------
    # 2-part checkerboard masks (DCVC-style)
    # -------------------------------------------------------------------------

    def get_mask_two_parts(self, batch, channel, height, width, device="cuda"):
        """mask_0 = ((1,0),(0,1)) checkerboard; mask_1 = complementary."""
        curr_mask_str = f"{batch}_{channel}x{width}x{height}_2x"
        if curr_mask_str not in self.masks:
            assert channel % 2 == 0
            micro_m0 = torch.tensor(((1.0, 0), (0, 1.0)), device=device)
            m0 = micro_m0.repeat((height + 1) // 2, (width + 1) // 2)
            m0 = m0[:height, :width]
            m0 = torch.unsqueeze(m0, 0)
            m0 = torch.unsqueeze(m0, 0)

            micro_m1 = torch.tensor(((0, 1.0), (1.0, 0)), device=device)
            m1 = micro_m1.repeat((height + 1) // 2, (width + 1) // 2)
            m1 = m1[:height, :width]
            m1 = torch.unsqueeze(m1, 0)
            m1 = torch.unsqueeze(m1, 0)

            m = torch.ones((batch, channel // 2, height, width), device=device)
            mask_0 = torch.cat((m * m0, m * m1), dim=1)
            mask_1 = torch.cat((m * m1, m * m0), dim=1)
            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]
