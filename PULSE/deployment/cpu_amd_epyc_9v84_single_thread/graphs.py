"""Static compact-symbol graphs and AVX-512 receiver weight layout."""
import torch
from torch import nn
from codec.models.entropy.entropy_coding import MAX_ENTROPY_CODING_VALUE
MEMORY_FORMAT = torch.channels_last


def selected_scalar(
    parameter: torch.Tensor,
    qp: int,
    lower: float,
) -> torch.Tensor:
    return (
        parameter[qp : qp + 1, :, None, None]
        .detach()
        .clone()
        .clamp_min(lower)
    )


def selected_vector(parameter: torch.Tensor, qp: int) -> torch.Tensor:
    return parameter[qp : qp + 1, :, None, None].detach().clone()


def checkerboard_masks(
    batch: int,
    channels: int,
    height: int,
    width: int,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(height).reshape(1, 1, height, 1)
    columns = torch.arange(width).reshape(1, 1, 1, width)
    even = ((rows + columns) % 2 == 0).to(dtype)
    odd = 1.0 - even
    first_even = even.expand(batch, channels // 2, height, width)
    first_odd = odd.expand(batch, channels // 2, height, width)
    return (
        torch.cat((first_even, first_odd), dim=1).contiguous(
            memory_format=MEMORY_FORMAT
        ),
        torch.cat((first_odd, first_even), dim=1).contiguous(
            memory_format=MEMORY_FORMAT
        ),
    )


class DecoderCommon(nn.Module):
    """Baked-QP entropy reconstruction shared by sender and receiver."""

    def __init__(
        self,
        net: nn.Module,
        *,
        qp: int,
        image_height: int,
        image_width: int,
    ):
        super().__init__()
        self.net = net
        self.qp = int(qp)
        entropy = net.entropy_model
        self.register_buffer(
            "q_hyper_z",
            selected_scalar(
                entropy.q_scale_hyper_z,
                qp,
                entropy.q_scale_lower_bound,
            ),
        )
        self.register_buffer(
            "q_hyper_m",
            selected_scalar(
                entropy.q_scale_hyper_m,
                qp,
                entropy.q_scale_lower_bound,
            ),
        )
        self.register_buffer(
            "q_prior",
            selected_scalar(
                entropy.q_scale_prior_2m,
                qp,
                entropy.q_scale_lower_bound,
            ),
        )
        self.register_buffer(
            "q_basic",
            selected_scalar(entropy.q_basic, qp, 0.0),
        )
        latent_height = image_height // 16
        latent_width = image_width // 16
        mask0, mask1 = checkerboard_masks(
            1,
            net.M,
            latent_height,
            latent_width,
            dtype=next(net.parameters()).dtype,
        )
        self.register_buffer("mask0", mask0)
        self.register_buffer("mask1", mask1)

    def common_params(self, z_hat: torch.Tensor):
        entropy = self.net.entropy_model
        params = entropy.hyper_dec(
            z_hat,
            self.q_hyper_z,
            self.q_hyper_m,
        )
        q_basic = self.q_basic.expand(
            params.shape[0],
            1,
            params.shape[2],
            params.shape[3],
        )
        common = entropy.em_step1(
            params,
            q_basic.to(params.dtype),
            self.q_prior,
        )
        q_step, means0 = common.chunk(2, dim=1)
        q_step = q_step.clamp_min(0.5)
        return common, 1.0 / q_step, q_step, means0


class CompactSymbolEncode(DecoderCommon):
    """Production sender tensors consumed directly by the entropy runtime."""

    def __init__(
        self,
        net: nn.Module,
        *,
        qp: int,
        image_height: int,
        image_width: int,
    ):
        super().__init__(
            net,
            qp=qp,
            image_height=image_height,
            image_width=image_width,
        )
        self.register_buffer(
            "q_encoder",
            selected_scalar(
                net.q_scale_enc,
                qp,
                net.entropy_model.q_scale_lower_bound,
            ),
        )
        rows = torch.arange(image_height // 16).reshape(1, 1, -1, 1)
        columns = torch.arange(image_width // 16).reshape(1, 1, 1, -1)
        self.register_buffer("even", (rows + columns) % 2 == 0)

    @staticmethod
    def quantize_symbol(value: torch.Tensor) -> torch.Tensor:
        return (
            value.round()
            .clamp(
                -MAX_ENTROPY_CODING_VALUE,
                MAX_ENTROPY_CODING_VALUE,
            )
            .to(torch.int8)
        )

    def forward(self, image: torch.Tensor):
        entropy = self.net.entropy_model
        y = self.net.encoder(image, self.q_encoder)
        z_hat = entropy.hyper_enc(y, self.q_hyper_z).round()
        common, q_encode, _q_decode, means0 = self.common_params(z_hat)
        y_quantized = y * q_encode
        half = y_quantized.shape[1] // 2
        y_first, y_second = y_quantized[:, :half], y_quantized[:, half:]
        mean_first, mean_second = means0[:, :half], means0[:, half:]
        symbol0_first = self.quantize_symbol(y_first - mean_first)
        symbol0_second = self.quantize_symbol(y_second - mean_second)
        compact0 = torch.where(
            self.even,
            symbol0_first,
            symbol0_second,
        )

        zero = torch.zeros_like(compact0)
        symbols0 = torch.cat(
            (
                torch.where(self.even, compact0, zero),
                torch.where(self.even, zero, compact0),
            ),
            dim=1,
        ).to(y_quantized.dtype)
        means0_masked = torch.cat(
            (
                torch.where(self.even, mean_first, 0.0),
                torch.where(self.even, 0.0, mean_second),
            ),
            dim=1,
        )
        means1 = entropy.em_step2(
            symbols0 + means0_masked,
            common,
            self.q_prior,
        )
        mean1_first, mean1_second = means1[:, :half], means1[:, half:]
        symbol1_first = self.quantize_symbol(y_first - mean1_first)
        symbol1_second = self.quantize_symbol(y_second - mean1_second)
        compact1 = torch.where(
            self.even,
            symbol1_second,
            symbol1_first,
        )
        return (
            z_hat.to(torch.int16).contiguous(),
            compact0.permute(0, 2, 3, 1).contiguous(),
            compact1.permute(0, 2, 3, 1).contiguous(),
        )


class ReferenceCompactDecode(DecoderCommon):
    """PyTorch reference for the native int16/int8 complete-codec receiver."""

    def __init__(
        self,
        net: nn.Module,
        *,
        qp: int,
        image_height: int,
        image_width: int,
    ):
        super().__init__(
            net,
            qp=qp,
            image_height=image_height,
            image_width=image_width,
        )
        rows = torch.arange(image_height // 16).reshape(1, 1, -1, 1)
        columns = torch.arange(image_width // 16).reshape(1, 1, 1, -1)
        self.register_buffer("even", (rows + columns) % 2 == 0)

    def forward(
        self,
        z_i16: torch.Tensor,
        compact0_nhwc: torch.Tensor,
        compact1_nhwc: torch.Tensor,
    ):
        entropy = self.net.entropy_model
        z_hat = z_i16.to(torch.bfloat16)
        compact0 = compact0_nhwc.permute(0, 3, 1, 2).to(torch.bfloat16)
        compact1 = compact1_nhwc.permute(0, 3, 1, 2).to(torch.bfloat16)
        zero = torch.zeros_like(compact0)
        symbols0 = torch.cat(
            (
                torch.where(self.even, compact0, zero),
                torch.where(self.even, zero, compact0),
            ),
            dim=1,
        )
        symbols1 = torch.cat(
            (
                torch.where(self.even, zero, compact1),
                torch.where(self.even, compact1, zero),
            ),
            dim=1,
        )
        common, _q_encode, q_decode, means0 = self.common_params(z_hat)
        y_hat0 = symbols0 + means0 * self.mask0
        means1 = entropy.em_step2(y_hat0, common, self.q_prior)
        y_hat1 = symbols1 + means1 * self.mask1
        return self.net.decoder((y_hat0 + y_hat1) * q_decode, self.qp)


def extension_state_dict(net: nn.Module) -> dict[str, torch.Tensor]:
    """Make every floating weight BF16 and every 4-D tensor channels-last."""
    output = {}
    for key, value in net.state_dict().items():
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.to(torch.bfloat16)
        if value.ndim == 4:
            value = value.contiguous(memory_format=MEMORY_FORMAT)
        else:
            value = value.contiguous()
        output[key] = value
    return output
