"""Top-level PULSE image codec."""

import torch
import torch.nn as nn

from codec.models.entropy.compression_model import HyperTwoStepEM
from codec.models.entropy.entropy_coding import (
    DEFAULT_Y_SKIP_CONTEXT_MODE,
    DEFAULT_Y_SKIP_INDEX_CUTOFF,
)
from codec.models.modules.blocks import _normalize_dico_block_variant
from codec.models.modules.decoder import Decoder
from codec.models.modules.encoder import Encoder

__all__ = ["PULSE"]


class PULSE(nn.Module):
    """Encoder + hyperprior two-step entropy model + RGB decoder."""

    def __init__(
        self,
        hidden_size=144,
        hidden_size_x=48,
        hx_bottleneck=48,
        mlp_ratio=272 / 144,
        num_cond_blocks=2,
        M: int = 320,
        z_ch: int = 96,
        entropy_channels: int = 128,
        dico_block_variant: str = "ca",
    ):
        super().__init__()
        self.M = int(M)
        self.z_ch = int(z_ch)
        self.qp_num = 8
        self.dico_block_variant = _normalize_dico_block_variant(dico_block_variant)

        self.encoder = Encoder(z_ch=self.M, dico_block_variant=self.dico_block_variant)
        self.q_scale_enc = nn.Parameter(torch.ones(self.qp_num, 1))

        self.entropy_model = HyperTwoStepEM(
            M=self.M, z_ch=self.z_ch, entropy_channels=entropy_channels,
            dico_block_variant=self.dico_block_variant,
        )
        self.decoder = Decoder(
            latent_channels=self.M,
            hidden_size=hidden_size,
            hidden_size_x=hidden_size_x,
            hx_bottleneck=hx_bottleneck,
            mlp_ratio=mlp_ratio,
            num_cond_blocks=num_cond_blocks,
            dico_block_variant=self.dico_block_variant,
        )

        self.initialize_weights()


    def initialize_weights(self):
        """Preserve the canonical joint encoder/entropy/decoder initialization."""

        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for block in self.decoder.blocks:
            nn.init.constant_(block.conv3.weight, 0)
            nn.init.constant_(block.conv3.bias, 0)
            nn.init.constant_(block.conv5.weight, 0)
            nn.init.constant_(block.conv5.bias, 0)
        nn.init.constant_(self.decoder.renderer.adaln[-1].weight, 0)
        nn.init.constant_(self.decoder.renderer.adaln[-1].bias, 0)
        nn.init.constant_(self.decoder.renderer.output[0].weight, 0)
        nn.init.constant_(self.decoder.renderer.output[0].bias, 0)

    def _qp_index(self, qp, batch_size, device):
        return self.entropy_model._resolve_qp(qp, batch_size, device)

    def _encoder_q_scale(self, qp_idx):
        return self.entropy_model._select_q_scale_bounded(
            self.q_scale_enc,
            qp_idx,
        )

    def encode(self, image, *, qp=None):
        """Encode an image to y_hat and differentiable rate metadata."""
        batch, _, height, width = image.shape
        qp_idx = self._qp_index(qp, batch, image.device)
        q_scale = self._encoder_q_scale(qp_idx)
        y = self.encoder(image, q_scale)
        return self.entropy_model(
            y,
            num_pixels=height * width,
            qp=qp_idx,
        )

    @torch.no_grad()
    def compress(
        self,
        image,
        qp=None,
        skip_index_cutoff=DEFAULT_Y_SKIP_INDEX_CUTOFF,
        skip_context_mode=DEFAULT_Y_SKIP_CONTEXT_MODE,
        y_cdf_mode="part_specific",
        collect_y_symbols=False,
    ):
        """Encode an image to a real entropy-coded bitstream."""
        batch = image.shape[0]
        if batch != 1:
            raise ValueError("compress() is single-image; loop externally")
        qp_idx = self._qp_index(qp, batch, image.device)
        q_scale = self._encoder_q_scale(qp_idx).float()
        y = self.encoder(image.float(), q_scale)
        return self.entropy_model.compress(
            y,
            qp=qp_idx,
            skip_index_cutoff=skip_index_cutoff,
            skip_context_mode=skip_context_mode,
            y_cdf_mode=y_cdf_mode,
            collect_y_symbols=collect_y_symbols,
        )

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
        """Decode an entropy bitstream into y_hat."""
        return self.entropy_model.decompress(
            stream,
            z_shape,
            y_shape,
            qp=qp,
            skip_index_cutoff=skip_index_cutoff,
            y_cdf_mode=y_cdf_mode,
            entropy_lanes=entropy_lanes,
            entropy_sections=entropy_sections,
            entropy_dense=entropy_dense,
        )

    def decode(self, y_hat, qp=None):
        """Render RGB from y_hat."""
        qp_idx = self._qp_index(qp, y_hat.shape[0], y_hat.device)
        return self.decoder(y_hat, qp_idx=qp_idx)

    def forward(self, image, y_hat=None, return_codec_res=False, qp=None):
        codec_res = None
        if y_hat is None:
            y_hat, codec_res = self.encode(image, qp=qp)
        output = self.decode(y_hat, qp=qp)
        if return_codec_res:
            return output, y_hat, codec_res
        return output, y_hat
