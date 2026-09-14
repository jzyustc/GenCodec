"""GELU and its deployment-friendly approximation used by PULSE."""

import torch
from torch import nn


class _LeakyHardSigmoidFn(torch.autograd.Function):
    """DCVC-style hard sigmoid with a leaky clipped-region gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, leak: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.leak = float(leak)
        return torch.clamp(x + 0.5, 0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        # Preserve the forward dtype's rounding at the clipping boundary.
        affine = x + 0.5
        in_linear_region = (affine > 0.0) & (affine < 1.0)
        slope = torch.where(
            in_linear_region,
            torch.ones((), dtype=grad_output.dtype, device=grad_output.device),
            torch.full(
                (), ctx.leak, dtype=grad_output.dtype,
                device=grad_output.device,
            ),
        )
        return grad_output * slope, None


class LeakyHardSigmoid(nn.Module):
    """``clamp(x + 0.5, 0, 1)`` with DCVC's leaky backward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LeakyHardSigmoidFn.apply(x, 0.001)


class _InputDomainHardSigmoidFn(_LeakyHardSigmoidFn):
    """The HR and perceptual training references use the input-domain surrogate."""

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        in_linear_region = (x > -0.5) & (x < 0.5)
        slope = torch.where(in_linear_region, torch.ones_like(grad_output),
                            torch.full_like(grad_output, 0.001))
        return grad_output * slope, None


class InputDomainHardSigmoid(nn.Module):
    """Same inference function; preserve the HR/perceptual training backward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _InputDomainHardSigmoidFn.apply(x, 0.001)


class _LeakyAffineHardSigmoidFn(torch.autograd.Function):
    """Affine hard sigmoid used by the retained HardGELU activation."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        alpha: float,
        beta: float,
        leak: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        ctx.beta = float(beta)
        ctx.leak = float(leak)
        return torch.clamp(ctx.alpha * x + ctx.beta, 0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        affine = ctx.alpha * x + ctx.beta
        in_linear_region = (affine > 0.0) & (affine < 1.0)
        slope = torch.where(
            in_linear_region,
            torch.full_like(grad_output, ctx.alpha),
            torch.full_like(grad_output, ctx.leak),
        )
        return grad_output * slope, None, None, None


class HardGELU(nn.Module):
    """``x * clamp(0.30*x + 0.5, 0, 1)`` with leaky clipped gradients."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _LeakyAffineHardSigmoidFn.apply(
            x,
            0.30,
            0.5,
            0.001,
        )
        return x * gate



def normalize_dico_activation(value: str) -> str:
    if value not in {"gelu", "hard_gelu_030"}:
        raise ValueError("PULSE supports gelu and hard_gelu_030")
    return value


def make_dico_activation(value: str) -> nn.Module:
    return nn.GELU() if normalize_dico_activation(value) == "gelu" else HardGELU()


def dico_preactivation_channels(output_channels: int, value: str) -> int:
    normalize_dico_activation(value)
    return int(output_channels)
