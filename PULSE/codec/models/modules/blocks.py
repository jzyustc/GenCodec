"""Reusable convolution and normalization blocks for PULSE models."""

import torch
import torch.nn as nn

from codec.models.activations import (
    dico_preactivation_channels,
    make_dico_activation,
    normalize_dico_activation,
)

__all__ = ["DCB", "RMSNorm2d"]


class RMSNorm2d(nn.Module):
    """RMSNorm over the channel dim for NCHW tensors."""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(dim=1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        x = xf.to(input_dtype)
        if self.weight is not None:
            x = x * self.weight.view(1, -1, 1, 1)
        return x


def _channel_shuffle_2d(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle NCHW channels so consecutive grouped 1x1 layers can communicate."""
    if groups <= 1:
        return x
    batch, channels, height, width = x.shape
    assert channels % groups == 0
    x = x.view(batch, groups, channels // groups, height, width)
    return x.transpose(1, 2).contiguous().view(batch, channels, height, width)


def _normalize_dico_block_variant(value: str) -> str:
    if value not in {"plain", "ca"}:
        raise ValueError(
            f"unknown dico_block_variant={value!r}; expected 'plain' or 'ca'"
        )
    return value


class DCB(nn.Module):
    """Unified DiCo block, optional channel adaptor, and optional CA.

    Passing ``out_channels`` selects the channel-adaptor layout used by
    entropy modules (``adaptor`` + nested ``block``), preserving checkpoint
    keys. Omitting it creates the direct layout used by encoder/decoder blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        mlp_ratio=None,
        pointwise_groups: int = 1,
        norm_affine: bool | None = None,
        use_rmsnorm: bool = True,
        dico_block_variant: str = "plain",
        dico_activation: str = "gelu",
    ):
        super().__init__()
        if out_channels is not None:
            block_mlp_ratio = 2.0 if mlp_ratio is None else mlp_ratio
            block_norm_affine = True if norm_affine is None else norm_affine
            self.adaptor = (
                nn.Conv2d(in_channels, out_channels, 1)
                if in_channels != out_channels
                else None
            )
            self.block = DCB(
                out_channels,
                mlp_ratio=block_mlp_ratio,
                pointwise_groups=pointwise_groups,
                norm_affine=block_norm_affine,
                use_rmsnorm=use_rmsnorm,
                dico_block_variant=dico_block_variant,
                dico_activation=dico_activation,
            )
            return

        hidden_size = int(in_channels)
        mlp_ratio = 4.0 if mlp_ratio is None else mlp_ratio
        norm_affine = False if norm_affine is None else norm_affine
        groups = int(pointwise_groups)
        if hidden_size % groups:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by groups={groups}"
            )
        self.pointwise_groups = groups
        self.dico_activation = normalize_dico_activation(dico_activation)
        dc_pre_channels = dico_preactivation_channels(
            hidden_size,
            self.dico_activation,
        )
        if dc_pre_channels % groups:
            raise ValueError(
                f"dc_pre_channels={dc_pre_channels} is not divisible "
                f"by groups={groups}"
            )
        self.conv1 = nn.Conv2d(
            hidden_size,
            dc_pre_channels,
            1,
            groups=groups,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            dc_pre_channels,
            dc_pre_channels,
            3,
            padding=1,
            groups=dc_pre_channels,
            bias=True,
        )
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1, groups=groups, bias=True)
        self.dc_activation = make_dico_activation(
            self.dico_activation,
        )

        ffn_channel = int(mlp_ratio * hidden_size)
        if ffn_channel % groups:
            raise ValueError(
                f"ffn_channel={ffn_channel} is not divisible by groups={groups}"
            )
        ffn_pre_channels = dico_preactivation_channels(
            ffn_channel,
            self.dico_activation,
        )
        if ffn_pre_channels % groups:
            raise ValueError(
                f"ffn_pre_channels={ffn_pre_channels} is not divisible "
                f"by groups={groups}"
            )
        self.conv4 = nn.Conv2d(
            hidden_size,
            ffn_pre_channels,
            1,
            groups=groups,
            bias=True,
        )
        self.conv5 = nn.Conv2d(ffn_channel, hidden_size, 1, groups=groups, bias=True)
        self.ffn_activation = make_dico_activation(
            self.dico_activation,
        )
        self.use_rmsnorm = bool(use_rmsnorm)
        self.norm1 = (
            RMSNorm2d(hidden_size, eps=1e-6, affine=norm_affine)
            if self.use_rmsnorm
            else nn.Identity()
        )
        self.norm2 = (
            RMSNorm2d(hidden_size, eps=1e-6, affine=norm_affine)
            if self.use_rmsnorm
            else nn.Identity()
        )

        variant = _normalize_dico_block_variant(dico_block_variant)
        self.ca = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_size, hidden_size, 1, bias=True),
                nn.Sigmoid(),
            )
            if variant == "ca"
            else None
        )

    def forward(self, inputs):
        if hasattr(self, "block"):
            if self.adaptor is not None:
                inputs = self.adaptor(inputs)
            return self.block(inputs)

        x = self.norm1(inputs)
        x = self.dc_activation(self.conv2(self.conv1(x)))
        if self.ca is not None:
            x = x * self.ca(x)
        x = _channel_shuffle_2d(self.conv3(x), self.pointwise_groups)
        x = inputs + x
        ffn = self.conv5(
            self.ffn_activation(self.conv4(self.norm2(x)))
        )
        return x + _channel_shuffle_2d(ffn, self.pointwise_groups)


def _make_dcb(
    hidden_size,
    mlp_ratio=4.0,
    pointwise_groups: int = 1,
    dico_block_variant: str = "plain",
    norm_affine: bool = True,
    use_rmsnorm: bool = True,
    dico_activation: str = "gelu",
):
    return DCB(
        hidden_size,
        mlp_ratio=mlp_ratio,
        pointwise_groups=pointwise_groups,
        norm_affine=norm_affine,
        use_rmsnorm=use_rmsnorm,
        dico_block_variant=dico_block_variant,
        dico_activation=dico_activation,
    )
