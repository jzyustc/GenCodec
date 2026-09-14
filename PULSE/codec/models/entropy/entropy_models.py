"""PyTorch probability models and quantization primitives adapted from DCVC.

Used by the training rate loss, Meta Prior fitting and integer CDF construction.
Actual rANS transport is implemented in entropy_coding and the C++ extension.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


__all__ = [
    "LowerBound",
    "quantize_ste",
    "BitEstimator",
    "GaussianEncoder",
    "bit_estimator_z_prob",
    "bit_estimator_z_fwd",
    "bit_estimator_y_fwd",
]


# -----------------------------------------------------------------------------
# Primitive ops
# -----------------------------------------------------------------------------

class LowerBound(Function):
    """clamp_min with straight-through gradient when input is below bound but
    grad would push it back into the feasible region."""

    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs)
        ctx.bound = bound
        return torch.clamp_min(inputs, bound)

    @staticmethod
    def backward(ctx, grad):
        (inputs,) = ctx.saved_tensors
        bound = ctx.bound
        pass_through = (inputs >= bound) | (grad < 0)
        return pass_through * grad, None


class _QuantSTE(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through round; matches compressai.ops.quantize_ste semantics."""
    return _QuantSTE.apply(x)


# -----------------------------------------------------------------------------
# Hyperprior factorized CDF (z)
# -----------------------------------------------------------------------------

def bit_estimator_z_prob(x: torch.Tensor, h: torch.Tensor, b: torch.Tensor,
                         a: torch.Tensor) -> torch.Tensor:
    """4-layer factorized CDF used by the BitEstimator.

    h, b: (B_qp, C, 4) — softplus weight + bias per layer.
    a:    (B_qp, C, 3) — tanh mix weight for the first 3 layers.
    """
    for i in range(4):
        x = x * F.softplus(h[:, :, i:i + 1, None]) + b[:, :, i:i + 1, None]
        if i != 3:
            x = x + torch.tanh(x) * torch.tanh(a[:, :, i:i + 1, None])
    return torch.sigmoid(x)


def bit_estimator_z_fwd(x: torch.Tensor, h: torch.Tensor, b: torch.Tensor,
                        a: torch.Tensor) -> torch.Tensor:
    """Probability mass P(x - 0.5 < X < x + 0.5) for the factorized prior.

    Computed in float32 for numeric stability, cast back to x.dtype.
    """
    dtype = x.dtype
    x32 = x.float()
    h32, b32, a32 = h.float(), b.float(), a.float()
    lower = bit_estimator_z_prob(x32 - 0.5, h32, b32, a32)
    upper = bit_estimator_z_prob(x32 + 0.5, h32, b32, a32)
    prob = upper - lower
    return prob.to(dtype)


# -----------------------------------------------------------------------------
# Conditional Gaussian pmf (y)
# -----------------------------------------------------------------------------

_SQRT_HALF_NEG = float(-(2 ** -0.5))


def bit_estimator_y_fwd(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """P(values - 0.5 < X < values + 0.5) under N(0, scales^2), via erfc.

    Mirrors DCVC's PyTorch implementation: scales floored at 0.11, 0.5x folding
    around |values|, and clamp_min(prob, 1e-9). Caller is responsible for
    subtracting the mean before passing `values`.
    """
    dtype = values.dtype
    values = values.float()
    scales = scales.float()
    scales = LowerBound.apply(scales, 0.11)
    values = torch.abs(values)
    upper = torch.erfc(_SQRT_HALF_NEG * (0.5 - values) / scales)
    lower = torch.erfc(_SQRT_HALF_NEG * (-0.5 - values) / scales)
    prob = upper - lower
    prob = torch.clamp_min(0.5 * prob, 1e-9)
    return prob.to(dtype)


# -----------------------------------------------------------------------------
# Variable-rate factorized prior for z
# -----------------------------------------------------------------------------

class BitEstimator(nn.Module):
    """Per-channel factorized prior with `qp_num` rate slots.

    Parameters are shaped (qp_num, channel, layers); select a slot via the
    `index` argument to forward/get_prob (an int or a 1-D long tensor of
    length B). Forward returns the per-element probability mass; integrate
    it into a bpp loss with `-log(prob).sum() / log(2) / num_pixels`.
    """

    def __init__(self, qp_num: int, channel: int):
        super().__init__()
        layers = 4
        self.qp_num = qp_num
        self.channel = channel
        self.h = nn.Parameter(torch.empty(qp_num, channel, layers))
        self.b = nn.Parameter(torch.empty(qp_num, channel, layers))
        self.a = nn.Parameter(torch.empty(qp_num, channel, layers - 1))
        nn.init.normal_(self.h, 0, 0.01)
        nn.init.normal_(self.b, 0, 0.01)
        nn.init.normal_(self.a, 0, 0.01)

    def _select(self, index, dtype=None):
        if isinstance(index, int):
            sl = slice(index, index + 1)
            h, b, a = self.h[sl], self.b[sl], self.a[sl]
        else:
            h = torch.index_select(self.h, 0, index)
            b = torch.index_select(self.b, 0, index)
            a = torch.index_select(self.a, 0, index)
        if dtype is not None:
            h, b, a = h.to(dtype), b.to(dtype), a.to(dtype)
        return h, b, a

    def forward(self, x: torch.Tensor, index=0) -> torch.Tensor:
        """Return P(x - 0.5 < X < x + 0.5) for the chosen rate slot(s)."""
        h, b, a = self._select(index)
        return bit_estimator_z_fwd(x, h, b, a)


# -----------------------------------------------------------------------------
# Gaussian conditional for y (training-time only)
# -----------------------------------------------------------------------------

# Match DCVC's GlobalSettings defaults so the scale table covers the
# same dynamic range. These are inert here (only used to initialize
# scale_table) but kept as attributes in case downstream code reads them.
_SCALE_MIN = 0.11
_SCALE_MAX = 256.0
_SCALE_LEVEL = 64


class GaussianEncoder(nn.Module):
    """Gaussian conditional shell for training-time bpp.

    Only the prob path is implemented:
        prob = bit_estimator_y_fwd(y - means, scales)

    `scale_table` and `skip_thres` are kept for API parity with DCVC but
    are not used at training time.
    """

    def __init__(self, scale_min: float = _SCALE_MIN, scale_max: float = _SCALE_MAX,
                 scale_level: int = _SCALE_LEVEL):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_level = scale_level
        self.register_buffer(
            "scale_table",
            torch.exp(torch.linspace(math.log(scale_min), math.log(scale_max), scale_level)),
            persistent=False,
        )
        self.skip_thres = 0.0

    @staticmethod
    def get_prob_train(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return bit_estimator_y_fwd(values, scales)

    def forward(self, values: torch.Tensor, scales: torch.Tensor,
                means: torch.Tensor | None = None) -> torch.Tensor:
        if means is not None:
            values = values - means
        return bit_estimator_y_fwd(values, scales)
