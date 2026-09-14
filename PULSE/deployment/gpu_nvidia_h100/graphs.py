"""BF16 H100 sender/receiver graphs with compact integer-symbol interfaces."""
import torch
from torch import nn
from codec.models.entropy.entropy_coding import MAX_ENTROPY_CODING_VALUE as MAX_SYMBOL


def selected_scalar(parameter: torch.Tensor, qp: int, lower: float) -> torch.Tensor:
    return parameter[qp : qp + 1, :, None, None].detach().clone().clamp_min(lower)


def meta_core_cost(cdf, lengths, *, banks: int, channels: int):
    cdf = torch.as_tensor(cdf, dtype=torch.int64).reshape(
        banks,
        channels,
        -1,
    )
    lengths = torch.as_tensor(
        lengths,
        dtype=torch.int64,
    ).reshape(banks, channels)
    max_value = lengths - 2
    total = torch.gather(
        cdf,
        2,
        (lengths - 1)[..., None],
    ).squeeze(2).float()
    encoded_values = torch.arange(
        128,
        dtype=torch.int64,
    )[None, None]
    encoded_values = torch.minimum(
        encoded_values,
        max_value[..., None],
    )
    low = torch.gather(cdf, 2, encoded_values)
    high = torch.gather(cdf, 2, encoded_values + 1)
    core_cost = -torch.log2(
        ((high - low).float() / total[..., None]).clamp_min(1e-12)
    )
    return (
        core_cost.contiguous(),
        max_value.to(torch.int32).contiguous(),
    )


class ExactMetaSelector(nn.Module):
    """Exact current-bitstream bank selection from resident quantized CDFs."""

    def __init__(self, cdf: torch.Tensor, lengths: torch.Tensor, *, banks: int, channels: int):
        super().__init__()
        core_cost, max_value = meta_core_cost(
            cdf,
            lengths,
            banks=banks,
            channels=channels,
        )
        self.banks = int(banks)
        self.channels = int(channels)
        self.register_buffer("max_value", max_value)
        self.register_buffer("core_cost", core_cost)
        self.register_buffer(
            "bypass_thresholds",
            torch.tensor([1, 4, 16, 64, 256, 1024, 4096, 16384, 65536], dtype=torch.int64),
        )

    def forward(self, z_i16: torch.Tensor) -> torch.Tensor:
        positions = z_i16.permute(0, 2, 3, 1).reshape(-1, self.channels).to(torch.int64)
        values = positions.abs() * 2 - (positions > 0).to(torch.int64)
        values = values[None].expand(self.banks, -1, -1)
        maximum = self.max_value[:, None, :]
        encoded = torch.minimum(values, maximum)
        expanded_cost = self.core_cost[:, None].expand(-1, values.shape[1], -1, -1)
        cost = torch.gather(expanded_cost, 3, encoded[..., None]).squeeze(3)
        tail = values >= maximum
        raw = torch.where(tail, values - maximum, torch.zeros_like(values))
        n_bypass = (raw[..., None] >= self.bypass_thresholds).sum(dim=3)
        bypass_bins = torch.div(n_bypass, 3, rounding_mode="floor") + 1 + n_bypass
        cost = cost + torch.where(tail, bypass_bins.float() * 2.0, torch.zeros_like(cost))
        selected = cost.sum(dim=2).argmin(dim=0).to(torch.uint8)
        return selected.reshape(z_i16.shape[0], z_i16.shape[2], z_i16.shape[3])


class ProductionCommon(nn.Module):
    def __init__(self, net: nn.Module, *, qp: int, height: int, width: int):
        super().__init__()
        self.net = net
        self.qp = int(qp)
        self.height = int(height)
        self.width = int(width)
        em = net.entropy_model
        self.register_buffer("q_hyper_z", selected_scalar(em.q_scale_hyper_z, qp, em.q_scale_lower_bound))
        self.register_buffer("q_hyper_m", selected_scalar(em.q_scale_hyper_m, qp, em.q_scale_lower_bound))
        self.register_buffer("q_prior", selected_scalar(em.q_scale_prior_2m, qp, em.q_scale_lower_bound))
        self.register_buffer("q_basic", selected_scalar(em.q_basic, qp, 0.0))
        yh, yw = height // 16, width // 16
        rows = torch.arange(yh).reshape(1, 1, yh, 1)
        cols = torch.arange(yw).reshape(1, 1, 1, yw)
        even = ((rows + cols) % 2 == 0)
        self.register_buffer("even", even)

    def common_params(self, z_hat: torch.Tensor):
        em = self.net.entropy_model
        params = em.hyper_dec(z_hat, self.q_hyper_z, self.q_hyper_m)
        qb = self.q_basic.expand(params.shape[0], 1, params.shape[2], params.shape[3])
        common = em.em_step1(params, qb.to(params.dtype), self.q_prior)
        qstep, means0 = common.chunk(2, dim=1)
        qstep = qstep.clamp_min(0.5)
        return common, 1.0 / qstep, qstep, means0

    def full_part0(self, compact: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(compact)
        return torch.cat(
            (torch.where(self.even, compact, zero), torch.where(self.even, zero, compact)),
            dim=1,
        )

    def full_part1(self, compact: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(compact)
        return torch.cat(
            (torch.where(self.even, zero, compact), torch.where(self.even, compact, zero)),
            dim=1,
        )

    def mask_part0(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return torch.cat((torch.where(self.even, first, 0.0), torch.where(self.even, 0.0, second)), dim=1)

    def mask_part1(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return torch.cat((torch.where(self.even, 0.0, first), torch.where(self.even, second, 0.0)), dim=1)


class ProductionEncode(ProductionCommon):
    """GPU sender outputs only actual entropy payload inputs, compactly."""

    def __init__(
        self,
        net: nn.Module,
        *,
        qp: int,
        height: int,
        width: int,
        meta_cdf=None,
        meta_lengths=None,
    ):
        super().__init__(net, qp=qp, height=height, width=width)
        em = net.entropy_model
        self.register_buffer("q_encoder", selected_scalar(net.q_scale_enc, qp, em.q_scale_lower_bound))
        self.meta_selector = (
            None
            if meta_cdf is None
            else ExactMetaSelector(
                meta_cdf,
                meta_lengths,
                banks=64,
                channels=net.z_ch,
            )
        )

    @staticmethod
    def quantize_symbol(value: torch.Tensor) -> torch.Tensor:
        return torch.round(value).clamp(-MAX_SYMBOL, MAX_SYMBOL).to(torch.int8)

    def forward(self, image: torch.Tensor):
        net = self.net
        em = net.entropy_model
        y = net.encoder(image, self.q_encoder)
        z = em.hyper_enc(y, self.q_hyper_z)
        z_hat = torch.round(z)
        z_i16 = z_hat.to(torch.int16)
        common, q_enc, _q_dec, means0 = self.common_params(z_hat)
        yq = y * q_enc
        half = yq.shape[1] // 2
        y_first, y_second = yq[:, :half], yq[:, half:]
        m0_first, m0_second = means0[:, :half], means0[:, half:]
        s0_first = self.quantize_symbol(y_first - m0_first)
        s0_second = self.quantize_symbol(y_second - m0_second)
        compact0 = torch.where(self.even, s0_first, s0_second)
        symbols0 = self.full_part0(compact0).to(yq.dtype)
        yhat0 = symbols0 + self.mask_part0(means0)
        means1 = em.em_step2(yhat0, common, self.q_prior)
        m1_first, m1_second = means1[:, :half], means1[:, half:]
        s1_first = self.quantize_symbol(y_first - m1_first)
        s1_second = self.quantize_symbol(y_second - m1_second)
        compact1 = torch.where(self.even, s1_second, s1_first)
        outputs = (
            z_i16.contiguous(),
            compact0.permute(0, 2, 3, 1).contiguous(),
            compact1.permute(0, 2, 3, 1).contiguous(),
        )
        if self.meta_selector is None:
            return outputs
        return outputs + (
            self.meta_selector(z_i16).contiguous(),
        )


class ProductionDecode(ProductionCommon):
    """GPU receiver accepts compact decoded entropy symbols."""

    def forward(self, z_i16: torch.Tensor, compact0_nhwc: torch.Tensor, compact1_nhwc: torch.Tensor):
        em = self.net.entropy_model
        z_hat = z_i16.to(torch.bfloat16)
        compact0 = compact0_nhwc.permute(0, 3, 1, 2).to(torch.bfloat16)
        compact1 = compact1_nhwc.permute(0, 3, 1, 2).to(torch.bfloat16)
        common, _q_enc, q_dec, means0 = self.common_params(z_hat)
        yhat0 = self.full_part0(compact0) + self.mask_part0(means0)
        means1 = em.em_step2(yhat0, common, self.q_prior)
        yhat1 = self.full_part1(compact1) + self.mask_part1(means1)
        yhat = (yhat0 + yhat1) * q_dec
        return self.net.decoder(yhat, self.qp)
