"""Whole-image QNN graphs with eight runtime QP controls."""
import torch
from torch import nn
from codec.models.activations import HardGELU
from codec.models.modules.blocks import DCB
from codec.models.entropy.entropy_coding import MAX_ENTROPY_CODING_VALUE as MAX_Y_SYMBOL

FOLD_SCALE = 1.8


def baked_q_scale(entropy, parameter, qp):
    return parameter[qp:qp + 1, :, None, None].detach().clamp_min(entropy.q_scale_lower_bound)


def spatial_masks(height, width):
    even = ((torch.arange(height)[:, None] + torch.arange(width)[None, :]) % 2 == 0).float()[None, None]
    return even, 1.0 - even


def fold_hardgelu_for_qnn(model: nn.Module):
    """Replace HardGELU with an exactly weight-folded standard Hardswish.

    ``HardGELU(x) = Hardswish(1.8*x) / 1.8``.  Scaling the producer and
    consumer weights therefore changes only floating-point rounding, while
    allowing QAIRT to use its native HardSwish operator.
    """

    converted = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, DCB) or not hasattr(module, "conv1"):
                continue
            if not isinstance(module.dc_activation, HardGELU):
                continue
            if not isinstance(module.ffn_activation, HardGELU):
                raise TypeError(f"{name}: DCB activations do not match")

            module.conv2.weight.mul_(FOLD_SCALE)
            module.conv2.bias.mul_(FOLD_SCALE)
            module.conv3.weight.div_(FOLD_SCALE)
            if module.ca is not None:
                module.ca[1].weight.div_(FOLD_SCALE)
            module.dc_activation = nn.Hardswish()

            module.conv4.weight.mul_(FOLD_SCALE)
            module.conv4.bias.mul_(FOLD_SCALE)
            module.conv5.weight.div_(FOLD_SCALE)
            module.ffn_activation = nn.Hardswish()
            converted.append(name)

    remaining = [
        name
        for name, module in model.named_modules()
        if isinstance(module, HardGELU)
    ]
    if remaining:
        raise RuntimeError(f"unfolded HardGELU modules remain: {remaining}")
    if not converted:
        raise RuntimeError("no HardGELU DCB was converted")
    return converted

class RuntimeQpEncoder(nn.Module):
    def __init__(self, network: nn.Module, *, height: int, width: int):
        super().__init__()
        self.network = network
        even, odd = spatial_masks(height // 16, width // 16)
        self.register_buffer("even", even)
        self.register_buffer("odd", odd)

    @staticmethod
    def quantize_y(value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.round(value),
            -float(MAX_Y_SYMBOL),
            float(MAX_Y_SYMBOL),
        )

    def forward(
        self,
        image,
        q_image,
        q_hyper_z,
        q_hyper_m,
        q_prior,
        q_basic,
    ):
        network = self.network
        entropy = network.entropy_model
        y = network.encoder(image, q_image)
        z = entropy.hyper_enc(y, q_hyper_z)
        z_hat = torch.round(z)
        params = entropy.hyper_dec(z_hat, q_hyper_z, q_hyper_m)
        common = entropy.em_step1(params, q_basic, q_prior)
        q_enc, _q_dec, means0_full = entropy._split_common(common)
        y_scaled = y * q_enc
        y_a, y_b = torch.chunk(y_scaled, 2, dim=1)
        means0_a, means0_b = torch.chunk(means0_full, 2, dim=1)
        symbols0_a = self.quantize_y(y_a - means0_a)
        symbols0_b = self.quantize_y(y_b - means0_b)
        part0 = symbols0_a * self.even + symbols0_b * self.odd
        y_hat0 = torch.cat(
            (
                (part0 + means0_a) * self.even,
                (part0 + means0_b) * self.odd,
            ),
            dim=1,
        )
        means1_full = entropy.em_step2(y_hat0, common, q_prior)
        means1_a, means1_b = torch.chunk(means1_full, 2, dim=1)
        symbols1_a = self.quantize_y(y_a - means1_a)
        symbols1_b = self.quantize_y(y_b - means1_b)
        part1 = symbols1_a * self.odd + symbols1_b * self.even
        return (
            z_hat,
            part0.permute(0, 2, 3, 1).contiguous(),
            part1.permute(0, 2, 3, 1).contiguous(),
        )

class RuntimeQpDecoder(nn.Module):
    def __init__(self, network: nn.Module, *, height: int, width: int):
        super().__init__()
        self.network = network
        even, odd = spatial_masks(height // 16, width // 16)
        self.register_buffer("even", even)
        self.register_buffer("odd", odd)

    def forward(
        self,
        z_hat,
        symbols0_packed,
        symbols1_packed,
        q_hyper_z,
        q_hyper_m,
        q_prior,
        q_basic,
        q_codec_dec,
        q_dico,
        q_render,
    ):
        entropy = self.network.entropy_model
        part0 = symbols0_packed.float().permute(0, 3, 1, 2)
        part1 = symbols1_packed.float().permute(0, 3, 1, 2)
        params = entropy.hyper_dec(z_hat, q_hyper_z, q_hyper_m)
        common = entropy.em_step1(params, q_basic, q_prior)
        _q_enc, q_dec, means0_full = entropy._split_common(common)
        means0_a, means0_b = torch.chunk(means0_full, 2, dim=1)
        y_hat0 = torch.cat(
            (
                (part0 + means0_a) * self.even,
                (part0 + means0_b) * self.odd,
            ),
            dim=1,
        )
        means1_full = entropy.em_step2(y_hat0, common, q_prior)
        means1_a, means1_b = torch.chunk(means1_full, 2, dim=1)
        y_hat1 = torch.cat(
            (
                (part1 + means1_a) * self.odd,
                (part1 + means1_b) * self.even,
            ),
            dim=1,
        )
        y_hat = (y_hat0 + y_hat1) * q_dec

        decoder = self.network.decoder
        latent = y_hat * q_codec_dec
        body_feature = decoder.up0(latent)
        for block in decoder.blocks:
            body_feature = block(body_feature)
            body_feature = body_feature * q_dico
        return decoder.renderer(latent, body_feature, q_render)

def qp_controls(network, qp: int, *, height: int, width: int):
    entropy = network.entropy_model
    decoder = network.decoder
    with torch.no_grad():
        return (
            baked_q_scale(entropy, network.q_scale_enc, qp).clone(),
            baked_q_scale(
                entropy,
                entropy.q_scale_hyper_z,
                qp,
            ).clone(),
            baked_q_scale(
                entropy,
                entropy.q_scale_hyper_m,
                qp,
            ).clone(),
            baked_q_scale(
                entropy,
                entropy.q_scale_prior_2m,
                qp,
            ).clone(),
            entropy._q_basic_feature(
                qp,
                1,
                height // 16,
                width // 16,
            ).clone(),
            decoder.q_scale_codec_dec[
                qp : qp + 1, :, None, None
            ].clamp_min(0.0).clone(),
            decoder.q_scale_dico[
                qp : qp + 1, :, None, None
            ].clone(),
            decoder.renderer.q_scale[
                qp : qp + 1, :, None, None
            ].clone(),
        )
