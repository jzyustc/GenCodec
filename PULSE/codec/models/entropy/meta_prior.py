"""Spatially adaptive factorized hyperlatent prior.

Meta Prior stores multiple factorized prior banks for every QP.  The encoder
selects one bank per z spatial position by minimum code length and transmits
the bank index.  One index is shared by every channel at that position.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from codec.models.entropy.entropy_models import bit_estimator_z_fwd


__all__ = ["MetaPrior"]


class MetaPrior(nn.Module):
    """Per-QP bank of factorized priors selected by minimum code length."""

    selection_method = "minimum_code_length"

    def __init__(self, qp_num: int, bank_count: int, channel: int):
        super().__init__()
        if (
            qp_num <= 0
            or bank_count <= 1
            or bank_count > 256
            or channel <= 0
        ):
            raise ValueError("invalid Meta Prior dimensions")
        self.qp_num = int(qp_num)
        self.bank_count = int(bank_count)
        self.channel = int(channel)
        self.selector_bits = int(math.ceil(math.log2(self.bank_count)))
        index_merge_probability_count = (
            4 + 4 * ((1 << self.selector_bits) - 1)
        )
        layers = 4
        self.h = nn.Parameter(
            torch.empty(self.qp_num, self.bank_count, self.channel, layers)
        )
        self.b = nn.Parameter(
            torch.empty(self.qp_num, self.bank_count, self.channel, layers)
        )
        self.a = nn.Parameter(
            torch.empty(
                self.qp_num,
                self.bank_count,
                self.channel,
                layers - 1,
            )
        )
        self.register_buffer(
            "ready",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "index_merge_probabilities",
            torch.full(
                (self.qp_num, index_merge_probability_count),
                1024,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "index_merge_ready",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "index_merge_adaptation_shift",
            torch.tensor(5, dtype=torch.int32),
        )
        nn.init.normal_(self.h, 0, 0.01)
        nn.init.normal_(self.b, 0, 0.01)
        nn.init.normal_(self.a, 0, 0.01)

    @torch.no_grad()
    def initialize_from_factorized_prior(self, prior) -> None:
        """Replicate an ordinary factorized prior into every Meta Prior bank."""
        if prior.qp_num != self.qp_num or prior.channel != self.channel:
            raise ValueError("factorized prior shape does not match Meta Prior")
        self.h.copy_(prior.h[:, None].expand_as(self.h))
        self.b.copy_(prior.b[:, None].expand_as(self.b))
        self.a.copy_(prior.a[:, None].expand_as(self.a))
        self.ready.fill_(False)
        self.index_merge_probabilities.fill_(1024)
        self.index_merge_ready.fill_(False)
        self.index_merge_adaptation_shift.fill_(5)

    @torch.no_grad()
    def load_fitted(self, payload: dict) -> None:
        """Load fitted bank parameters from a post-training artifact."""
        for name in ("h", "b", "a"):
            expected = getattr(self, name)
            value = torch.as_tensor(payload[name])
            if tuple(value.shape) != tuple(expected.shape):
                raise ValueError(
                    f"Meta Prior {name} shape mismatch: "
                    f"expected {tuple(expected.shape)}, got {tuple(value.shape)}"
                )
            expected.copy_(value.to(device=expected.device, dtype=expected.dtype))
        index_merge = payload.get("index_merge_model")
        if index_merge is None:
            self.index_merge_probabilities.fill_(1024)
            self.index_merge_ready.fill_(False)
            self.index_merge_adaptation_shift.fill_(5)
        else:
            probabilities = torch.as_tensor(
                index_merge["probabilities"],
                dtype=torch.int32,
            )
            if tuple(probabilities.shape) != tuple(
                self.index_merge_probabilities.shape
            ):
                raise ValueError(
                    "Meta Prior Index Merge probability shape mismatch: "
                    f"expected {tuple(self.index_merge_probabilities.shape)}, "
                    f"got {tuple(probabilities.shape)}"
                )
            if bool(
                (
                    (probabilities < 1)
                    | (probabilities >= 2048)
                ).any().item()
            ):
                raise ValueError(
                    "Meta Prior Index Merge probabilities must be in "
                    "[1, 2047]"
                )
            self.index_merge_probabilities.copy_(
                probabilities.to(self.index_merge_probabilities.device)
            )
            adaptation_shift = int(index_merge.get("adaptation_shift", 5))
            if not 0 <= adaptation_shift <= 15:
                raise ValueError(
                    "Meta Prior Index Merge adaptation_shift must be in "
                    "[0, 15]"
                )
            self.index_merge_adaptation_shift.fill_(adaptation_shift)
            self.index_merge_ready.fill_(True)
        self.ready.fill_(True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Accept checkpoints created before Index Merge Coding was added."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name in (
            "index_merge_probabilities",
            "index_merge_ready",
            "index_merge_adaptation_shift",
        ):
            full_name = prefix + name
            if full_name in missing_keys:
                missing_keys.remove(full_name)

    def _select_qp(self, qp, batch_size: int):
        if isinstance(qp, int):
            if not 0 <= qp < self.qp_num:
                raise ValueError(f"qp must be in [0, {self.qp_num - 1}]")
            return (
                self.h[qp : qp + 1],
                self.b[qp : qp + 1],
                self.a[qp : qp + 1],
            )
        qp = torch.as_tensor(qp, device=self.h.device, dtype=torch.long)
        if qp.dim() == 0:
            return self._select_qp(int(qp.item()), batch_size)
        if tuple(qp.shape) != (batch_size,):
            raise ValueError(
                f"qp tensor must have shape ({batch_size},), "
                f"got {tuple(qp.shape)}"
            )
        if bool(((qp < 0) | (qp >= self.qp_num)).any().item()):
            raise ValueError(f"qp values must be in [0, {self.qp_num - 1}]")
        return (
            torch.index_select(self.h, 0, qp),
            torch.index_select(self.b, 0, qp),
            torch.index_select(self.a, 0, qp),
        )

    def all_probability(self, x: Tensor, qp=0) -> Tensor:
        """Return probabilities for every bank as ``(B,K,C,H,W)``."""
        if x.ndim != 4 or x.shape[1] != self.channel:
            raise ValueError(
                f"expected BCHW with C={self.channel}, got {tuple(x.shape)}"
            )
        batch, channel, height, width = x.shape
        h, b, a = self._select_qp(qp, batch)
        if h.shape[0] == 1 and batch != 1:
            h = h.expand(batch, -1, -1, -1)
            b = b.expand(batch, -1, -1, -1)
            a = a.expand(batch, -1, -1, -1)
        if h.shape[0] != batch:
            raise ValueError(
                f"QP batch {h.shape[0]} does not match input batch {batch}"
            )
        values = (
            x[:, None]
            .expand(-1, self.bank_count, -1, -1, -1)
            .reshape(batch * self.bank_count, channel, height, width)
        )
        probability = bit_estimator_z_fwd(
            values,
            h.reshape(batch * self.bank_count, channel, 4),
            b.reshape(batch * self.bank_count, channel, 4),
            a.reshape(batch * self.bank_count, channel, 3),
        )
        return probability.reshape(
            batch,
            self.bank_count,
            channel,
            height,
            width,
        )

    @staticmethod
    def gather_probability(
        all_probability: Tensor,
        bank_index: Tensor,
    ) -> Tensor:
        """Gather one bank per spatial position, returning ``(B,C,H,W)``."""
        if all_probability.ndim != 5 or bank_index.ndim != 3:
            raise ValueError("invalid Meta Prior probability/index ranks")
        batch, _, channel, height, width = all_probability.shape
        if tuple(bank_index.shape) != (batch, height, width):
            raise ValueError(
                f"bank index shape {tuple(bank_index.shape)} does not match "
                f"{(batch, height, width)}"
            )
        gather_index = bank_index[:, None, None].expand(
            -1,
            1,
            channel,
            -1,
            -1,
        )
        return torch.gather(
            all_probability,
            dim=1,
            index=gather_index,
        ).squeeze(1)

    def forward(self, x: Tensor, qp=0) -> tuple[Tensor, Tensor]:
        """Select minimum continuous code length for estimated inference."""
        if not bool(self.ready.item()):
            raise RuntimeError("Meta Prior has not been fitted")
        all_probability = self.all_probability(x, qp)
        with torch.no_grad():
            cost = -torch.log2(
                all_probability.detach().float().clamp_min(1e-9)
            ).sum(dim=2)
            bank_index = cost.argmin(dim=1)
        return self.gather_probability(all_probability, bank_index), bank_index
