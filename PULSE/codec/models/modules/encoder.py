"""PULSE image encoder."""

import torch.nn as nn

from codec.models.modules.blocks import RMSNorm2d, _make_dcb

__all__ = ["Encoder"]


class Encoder(nn.Module):
    """Encoder: patch embedding + uniform-width DiCoBlocks at hidden_size.

    image -> patch_embed(hidden_size) -> blocks -> norm -> proj -> latent.
    Per-qp `quant_step` (1) multiplied after each block.
    """

    def __init__(
        self,
        z_ch=128,
        dico_block_variant: str = "plain",
    ):
        super().__init__()
        self.patch_size = 16
        self.hidden_size = 512

        self.patch_embed = nn.Conv2d(
            3, 512, kernel_size=16, stride=16, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                _make_dcb(
                    512,
                    mlp_ratio=4.0,
                    dico_block_variant=dico_block_variant,
                    dico_activation="gelu",
                )
                for _ in range(4)
            ]
        )

        self.norm_out = RMSNorm2d(512)
        self.proj_out = nn.Conv2d(512, z_ch, kernel_size=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, quant_step=None):
        h = self.patch_embed(x)
        for block in self.blocks:
            h = block(h)
            if quant_step is not None:
                h = h * quant_step.to(h.dtype)
        h = self.norm_out(h)
        h = self.proj_out(h)
        return h
