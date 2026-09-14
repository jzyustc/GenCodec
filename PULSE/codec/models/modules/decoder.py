"""PULSE latent-to-RGB decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec.models.activations import LeakyHardSigmoid
from codec.models.modules.blocks import (
    DCB,
    _normalize_dico_block_variant,
)

__all__ = ["AdaLNRenderer", "Decoder"]


class AdaLNRenderer(nn.Module):
    """Render RGB from y_hat and decoder-body features at the /4 grid.

    Pipeline:
      y_hat -> 4x upsample -> content projection + position bias
      body feature -> condition projection -> AdaLN residual block
      -> QP render scale -> RGB output layer
    """

    def __init__(
        self,
        latent_channels,
        body_channels,
        render_channels,
        bottleneck_channels,
    ):
        super().__init__()
        self.condition_patch_size = 4
        self.latent_upsample = nn.Sequential(
            nn.Conv2d(
                latent_channels,
                bottleneck_channels * 16,
                kernel_size=1,
            ),
            nn.PixelShuffle(4),
        )
        self.input_proj = nn.Conv2d(
            bottleneck_channels,
            render_channels,
            1,
            bias=False,
        )
        self.position_bias = nn.Parameter(
            torch.zeros(
                1,
                render_channels,
                self.condition_patch_size,
                self.condition_patch_size,
            )
        )
        self.condition_proj = nn.Conv2d(
            body_channels,
            self.condition_patch_size**2 * render_channels,
            1,
        )
        self.norm = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Conv2d(render_channels, render_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(render_channels, render_channels, 1, bias=True),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(render_channels, 3 * render_channels, 1, bias=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(render_channels, 3 * 16, kernel_size=1),
            nn.PixelShuffle(4),
        )
        self.q_scale = nn.Parameter(torch.ones(8, render_channels))
        self.gate_activation = LeakyHardSigmoid()

    def forward(self, y_hat, body_feature, render_scale=None):
        x = self.input_proj(self.latent_upsample(y_hat))
        _, _, height, width = x.shape
        position_bias = self.position_bias.repeat(
            1,
            1,
            height // self.condition_patch_size,
            width // self.condition_patch_size,
        )
        x = x + position_bias

        condition = F.pixel_shuffle(
            self.condition_proj(body_feature),
            self.condition_patch_size,
        )
        shift, scale, gate = self.adaln(condition).chunk(3, dim=1)
        residual = self.norm(x) * self.gate_activation(scale) + shift
        x = x + self.gate_activation(gate) * self.mlp(residual)
        if render_scale is not None:
            x = x * render_scale.to(x.dtype)
        return self.output(x)


class Decoder(nn.Module):
    """Decode ``y_hat`` into RGB using a DCB body and AdaLN renderer."""

    def __init__(
        self,
        latent_channels,
        hidden_size=144,
        hidden_size_x=48,
        hx_bottleneck=48,
        mlp_ratio=272 / 144,
        num_cond_blocks=2,
        dico_block_variant: str = "plain",
    ):
        super().__init__()
        self.dico_block_variant = _normalize_dico_block_variant(dico_block_variant)
        self.pixel_patch_size = 4
        hx_bn = int(hx_bottleneck)

        self.up0 = nn.Sequential(
            nn.Conv2d(
                latent_channels,
                hidden_size,
                kernel_size=1,
            ),
        )
        self.q_scale_codec_dec = nn.Parameter(torch.ones(8, 1))
        self.blocks = nn.ModuleList(
            [
                DCB(
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    dico_block_variant=self.dico_block_variant,
                    dico_activation="hard_gelu_030",
                    use_rmsnorm=False,
                )
                for _ in range(num_cond_blocks)
            ]
        )
        self.renderer = AdaLNRenderer(
            latent_channels=latent_channels,
            body_channels=hidden_size,
            render_channels=hidden_size_x,
            bottleneck_channels=hx_bn,
        )
        self.q_scale_dico = nn.Parameter(torch.ones(8, hidden_size))

    @staticmethod
    def _select_qp_scale_4d(param, qp_idx, clamp=False):
        """Index a (qp_num, C) scale and return (B_or_1, C, 1, 1)."""
        if isinstance(qp_idx, int):
            value = param[qp_idx : qp_idx + 1, :, None, None]
        else:
            value = torch.index_select(param, 0, qp_idx)[:, :, None, None]
        return value.clamp_min(0.0) if clamp else value

    def forward(self, y_hat, qp_idx=None):
        dico_scale = (
            self._select_qp_scale_4d(self.q_scale_dico, qp_idx)
            if qp_idx is not None
            else None
        )
        render_scale = (
            self._select_qp_scale_4d(self.renderer.q_scale, qp_idx)
            if qp_idx is not None
            else None
        )
        codec_scale = (
            self._select_qp_scale_4d(
                self.q_scale_codec_dec,
                qp_idx,
                clamp=True,
            )
            if qp_idx is not None
            else None
        )

        latent = y_hat
        if codec_scale is not None:
            latent = latent * codec_scale.to(latent.dtype)

        body_feature = self.up0(latent)
        for block in self.blocks:
            body_feature = block(body_feature)
            if dico_scale is not None:
                body_feature = body_feature * dico_scale.to(body_feature.dtype)

        return self.renderer(latent, body_feature, render_scale)
