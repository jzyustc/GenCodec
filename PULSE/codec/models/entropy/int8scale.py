"""Linear CDF Index Decoding and its strict integer deployment runtime."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from codec.models.entropy.entropy_models import LowerBound


__all__ = [
    "ScaleDecoder",
    "IntegerLinearCDFIndexDecoder",
    "Int8ScaleDecoder",
    "attach_integer_linear_cdf_index_decoder",
    "attach_int8_scale_decoder",
    "require_cross_platform_entropy",
]


LOG_INDEX_FRAC_BITS = 8
LOG_INDEX_AFFINE_SHIFT = 10
LOG_INDEX_TOTAL_SHIFT = LOG_INDEX_FRAC_BITS + LOG_INDEX_AFFINE_SHIFT
LOG_INDEX_FACTOR = 1 << LOG_INDEX_TOTAL_SHIFT
LOG_INDEX_LEVELS = 64


class _UpperBound(Function):
    """Upper clamp that only passes gradients pointing into the valid set."""

    @staticmethod
    def forward(ctx, inputs: Tensor, bound: float) -> Tensor:
        ctx.save_for_backward(inputs)
        ctx.bound = bound
        return torch.clamp_max(inputs, bound)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (inputs,) = ctx.saved_tensors
        pass_through = (inputs <= ctx.bound) | (grad > 0)
        return pass_through * grad, None


class ScaleDecoder(nn.Module):
    """Predict a continuous coordinate in the 64-row Gaussian CDF table.

    The learned head remains one 1x1 projection followed by PixelShuffle. Its
    output is no longer a linear-domain sigma. Coordinate 0 corresponds to
    ``SCALE_MIN`` and coordinate 63 corresponds to ``SCALE_MAX``.

    Training maps the bounded continuous coordinate to sigma for a smooth
    likelihood. Real entropy coding truncates the same coordinate directly to
    a UINT8 CDF row, so inference needs no exp/log/bucket-search path.
    """

    def __init__(
        self,
        M: int,
        z_ch: int,
        qp_num: int,
        up_factor: int = 4,
    ):
        super().__init__()
        self.M = int(M)
        self.z_ch = int(z_ch)
        self.qp_num = int(qp_num)
        self.up_factor = int(up_factor)
        self.proj = nn.Conv2d(
            self.z_ch,
            self.M * self.up_factor * self.up_factor,
            1,
        )
        self.qp_bias = nn.Parameter(torch.zeros(self.qp_num, 1))

    def _select_qp_bias(self, qp, batch_size: int, device) -> Tensor:
        if qp is None:
            qp = 0
        if isinstance(qp, int):
            if not 0 <= qp < self.qp_num:
                raise ValueError(f"qp must be in [0, {self.qp_num - 1}]")
            return self.qp_bias[qp : qp + 1, :, None, None]
        qp = torch.as_tensor(qp, device=device, dtype=torch.long)
        if qp.dim() == 0:
            return self._select_qp_bias(int(qp.item()), batch_size, device)
        if tuple(qp.shape) != (batch_size,):
            raise ValueError(
                f"qp tensor must have shape ({batch_size},), "
                f"got {tuple(qp.shape)}"
            )
        if bool(((qp < 0) | (qp >= self.qp_num)).any().item()):
            raise ValueError(f"qp values must be in [0, {self.qp_num - 1}]")
        return torch.index_select(self.qp_bias, 0, qp)[:, :, None, None]

    def log_index(self, z_hat: Tensor, qp=None) -> Tensor:
        coordinate = self.proj(z_hat)
        coordinate = coordinate + self._select_qp_bias(
            qp,
            z_hat.shape[0],
            z_hat.device,
        ).to(coordinate.dtype)
        return F.pixel_shuffle(coordinate, self.up_factor)

    def _bounded_log_index(self, z_hat: Tensor, qp=None) -> Tensor:
        coordinate = self.log_index(z_hat, qp).float()
        coordinate = LowerBound.apply(coordinate, 0.0)
        return _UpperBound.apply(coordinate, float(LOG_INDEX_LEVELS - 1))

    def forward(self, z_hat: Tensor, qp=None) -> Tensor:
        from codec.models.entropy import entropy_coding as ec

        coordinate = self._bounded_log_index(z_hat, qp)
        log_scale = ec.LOG_SCALE_MIN + coordinate * ec.LOG_SCALE_STEP
        # Keep exp in FP32 under BF16 mixed training.
        return torch.exp(log_scale)

    @torch.no_grad()
    def scale_bucket_idx(self, z_hat: Tensor, qp_int: int) -> Tensor:
        coordinate = self.log_index(z_hat, int(qp_int)).float()
        return coordinate.clamp(
            0.0,
            float(LOG_INDEX_LEVELS - 1),
        ).to(torch.uint8)


def _quantize_weight_per_output_channel(weight: Tensor):
    weight = weight.detach().float()
    scale = (
        weight.abs()
        .amax(dim=tuple(range(1, weight.ndim)))
        .clamp_min(1e-12)
        / 127.0
    )
    view_shape = (scale.numel(),) + (1,) * (weight.ndim - 1)
    quantized = (
        (weight / scale.view(view_shape))
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return quantized, scale


class IntegerLinearCDFIndexDecoder(nn.Module):
    """Strict integer runtime for Linear CDF Index Decoding.

    Runtime:

    ``integer z_hat -> INT8 activation -> INT8 weight / INT32 MAC
    -> Q8 fixed-point affine -> clamp -> packed UINT8 CDF indexes``.

    The Q8/shift-10 format is fixed rather than version-selected. Old
    linear-sigma threshold artifacts are intentionally unsupported.
    """

    artifact_implementation = "linear_cdf_index_decoding_int8_v1"

    def __init__(
        self,
        *,
        M: int,
        z_ch: int,
        qp_num: int,
        up_factor: int,
        input_clip: int,
        input_divisor: Tensor,
        weight_q: Tensor,
        multiplier: Tensor,
        affine_bias: Tensor,
    ):
        super().__init__()
        self.M = int(M)
        self.z_ch = int(z_ch)
        self.qp_num = int(qp_num)
        self.up_factor = int(up_factor)
        self.input_clip = int(input_clip)
        self.out_ch = self.M * self.up_factor * self.up_factor
        self.entropy_coding_version = 0
        self.max_entropy_coding_value = -1
        self._cpp_backend = None

        if self.up_factor != 4:
            raise ValueError(
                "Linear CDF Index Decoding requires up_factor=4"
            )
        if self.M % 32:
            raise ValueError("the C++ packed kernel requires M divisible by 32")
        if self.z_ch % 4:
            raise ValueError("the C++ packed kernel requires z_ch divisible by 4")
        if not 0 < self.input_clip <= 127:
            raise ValueError("input_clip must be in [1, 127]")

        divisor = input_divisor.detach().to(torch.int32)
        if tuple(divisor.shape) != (self.z_ch,):
            raise ValueError(
                f"input_divisor must have shape ({self.z_ch},), "
                f"got {tuple(divisor.shape)}"
            )
        if bool((divisor <= 0).any().item()):
            raise ValueError("input_divisor values must be positive")

        expected_weight = (self.out_ch, self.z_ch, 1, 1)
        if tuple(weight_q.shape) != expected_weight:
            raise ValueError(
                f"weight_q must have shape {expected_weight}, "
                f"got {tuple(weight_q.shape)}"
            )
        if tuple(multiplier.shape) != (self.out_ch,):
            raise ValueError(
                f"multiplier must have shape ({self.out_ch},), "
                f"got {tuple(multiplier.shape)}"
            )
        if tuple(affine_bias.shape) != (self.qp_num, self.out_ch):
            raise ValueError(
                "affine_bias must have shape "
                f"({self.qp_num}, {self.out_ch}), "
                f"got {tuple(affine_bias.shape)}"
            )

        self.register_buffer("input_divisor", divisor.contiguous())
        self.register_buffer(
            "weight_q",
            weight_q.detach().to(torch.int8).contiguous(),
        )
        self.register_buffer(
            "multiplier",
            multiplier.detach().to(torch.int32).contiguous(),
        )
        self.register_buffer(
            "affine_bias",
            affine_bias.detach().to(torch.int32).contiguous(),
        )
        self.register_buffer(
            "last_input_saturation_count",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )

    @staticmethod
    def _round_divide_nearest_even(value: Tensor, divisor: Tensor) -> Tensor:
        value = value.to(torch.int64)
        divisor = divisor.to(device=value.device, dtype=torch.int64)
        magnitude = value.abs()
        quotient = torch.div(magnitude, divisor, rounding_mode="floor")
        remainder = magnitude - quotient * divisor
        increment = (remainder * 2 > divisor) | (
            (remainder * 2 == divisor) & ((quotient & 1) != 0)
        )
        quotient = quotient + increment.to(torch.int64)
        return torch.where(value < 0, -quotient, quotient)

    @staticmethod
    def _integer_conv2d(x_q: Tensor, weight_q: Tensor):
        # Production z widths keep exact integer sums below 2**24.
        return F.conv2d(
            x_q.float(),
            weight_q.float(),
            bias=None,
            stride=1,
            padding=0,
        ).round()

    def _quantize_input(self, z_hat: Tensor) -> Tensor:
        rounded = z_hat.float().round().to(torch.int32)
        quantized = self._round_divide_nearest_even(
            rounded,
            self.input_divisor.view(1, -1, 1, 1),
        )
        saturated = quantized.abs() > self.input_clip
        self.last_input_saturation_count.copy_(
            saturated.sum().to(
                device=self.last_input_saturation_count.device,
                dtype=torch.int64,
            )
        )
        return quantized.clamp(
            -self.input_clip,
            self.input_clip,
        ).to(torch.int8)

    def _pre_shuffle_accumulator(self, z_hat: Tensor) -> Tensor:
        return self._integer_conv2d(
            self._quantize_input(z_hat),
            self.weight_q,
        )

    def _bucket_from_accumulator(
        self,
        accumulator: Tensor,
        qp_int: int,
    ) -> Tensor:
        if not 0 <= int(qp_int) < self.qp_num:
            raise ValueError(f"qp must be in [0, {self.qp_num - 1}]")
        numerator = (
            accumulator.to(torch.int64)
            * self.multiplier.view(1, -1, 1, 1).to(torch.int64)
            + self.affine_bias[int(qp_int)]
            .view(1, -1, 1, 1)
            .to(torch.int64)
        )
        # For f>=1:
        # floor(round_even(n / 2**a) / 2**f)
        #   == floor((n + 2**(a-1)) / 2**(a+f)).
        bucket = torch.div(
            numerator + (1 << (LOG_INDEX_AFFINE_SHIFT - 1)),
            1 << LOG_INDEX_TOTAL_SHIFT,
            rounding_mode="floor",
        )
        return bucket.clamp(0, LOG_INDEX_LEVELS - 1)

    @torch.no_grad()
    def scale_bucket_idx(self, z_hat: Tensor, qp_int: int) -> Tensor:
        with torch.autocast(
            device_type=z_hat.device.type,
            enabled=False,
        ):
            bucket = self._bucket_from_accumulator(
                self._pre_shuffle_accumulator(z_hat),
                int(qp_int),
            )
            return F.pixel_shuffle(
                bucket,
                self.up_factor,
            ).to(torch.uint8)

    def _get_cpp_backend(self):
        if self._cpp_backend is not None:
            return self._cpp_backend
        try:
            from MLCodec_extensions_cpp import LogIndexDecoder
        except ImportError as error:
            raise RuntimeError(
                "MLCodec_extensions_cpp was built without LogIndexDecoder; "
                "rebuild with pip install -e ."
            ) from error

        self._cpp_backend = LogIndexDecoder(
            self.M,
            self.z_ch,
            self.qp_num,
            self.up_factor,
            self.input_clip,
            self.input_divisor.detach().cpu().numpy(),
            self.weight_q.detach()
            .cpu()
            .squeeze(-1)
            .squeeze(-1)
            .numpy(),
            self.multiplier.detach().cpu().numpy(),
            self.affine_bias.detach().cpu().numpy(),
        )
        return self._cpp_backend

    @torch.no_grad()
    def packed_indexes(self, z_hat: Tensor, qp_int: int):
        rounded = z_hat.detach().round()
        info = torch.iinfo(torch.int16)
        if bool(((rounded < info.min) | (rounded > info.max)).any().item()):
            raise OverflowError("z_hat exceeds the INT16 deployment range")
        z_int16 = rounded.to(
            device="cpu",
            dtype=torch.int16,
        ).contiguous()
        packed = self._get_cpp_backend().packed_indexes(
            z_int16.numpy(),
            int(qp_int),
        )
        packed = torch.from_numpy(packed)
        if tuple(packed.shape[:1]) != (2,):
            raise RuntimeError(
                f"C++ packed index shape is invalid: {tuple(packed.shape)}"
            )
        return packed[0].contiguous(), packed[1].contiguous()

    def forward(self, z_hat: Tensor, qp=0) -> Tensor:
        from codec.models.entropy import entropy_coding as ec

        if not isinstance(qp, int):
            qp_tensor = torch.as_tensor(qp, device=z_hat.device)
            if qp_tensor.dim() != 1 or qp_tensor.numel() != z_hat.shape[0]:
                raise ValueError("INT8 forward expects an int or one QP per sample")
            buckets = torch.cat(
                [
                    self.scale_bucket_idx(
                        z_hat[index : index + 1],
                        int(qp_tensor[index].item()),
                    )
                    for index in range(z_hat.shape[0])
                ],
                dim=0,
            )
        else:
            buckets = self.scale_bucket_idx(z_hat, qp)
        return torch.exp(
            ec.LOG_SCALE_MIN
            + buckets.float() * ec.LOG_SCALE_STEP
        )

    @classmethod
    def from_float(
        cls,
        module: ScaleDecoder,
        *,
        input_divisor: Tensor,
        input_clip: int = 127,
    ):
        if not isinstance(module, ScaleDecoder):
            raise TypeError(
                f"expected ScaleDecoder, got {type(module).__name__}"
            )
        divisor = input_divisor.detach().float()
        if tuple(divisor.shape) != (module.z_ch,):
            raise ValueError(
                f"input_divisor must have shape ({module.z_ch},)"
            )
        if bool((divisor != divisor.round()).any().item()):
            raise ValueError("input_divisor values must be integers")
        if bool((divisor <= 0).any().item()):
            raise ValueError("input_divisor values must be positive")
        divisor = divisor.to(torch.int32)

        folded_weight = module.proj.weight.detach().float() * divisor.to(
            module.proj.weight.device,
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        weight_q, weight_scale = _quantize_weight_per_output_channel(
            folded_weight
        )
        multiplier64 = torch.round(
            weight_scale.double() * LOG_INDEX_FACTOR
        ).to(torch.int64)
        base_bias = module.proj.bias.detach().double()
        qp_bias = module.qp_bias.detach().double().flatten()
        affine_bias64 = torch.stack(
            [
                torch.round(
                    (base_bias + qp_bias[qp]) * LOG_INDEX_FACTOR
                )
                for qp in range(module.qp_num)
            ]
        ).to(torch.int64)

        i32 = torch.iinfo(torch.int32)
        if (
            int(multiplier64.abs().max().item()) > i32.max
            or int(affine_bias64.abs().max().item()) > i32.max
        ):
            raise OverflowError(
                "Linear CDF Index Decoding affine parameters exceed INT32"
            )
        accumulator_bound = (
            weight_q.to(torch.int64)
            .abs()
            .sum(dim=(1, 2, 3))
            * int(input_clip)
        )
        max_numerator = (
            accumulator_bound[None, :]
            * multiplier64.abs()[None, :]
            + affine_bias64.abs()
        ).max()
        if int(max_numerator.item()) > i32.max:
            raise OverflowError(
                "Q8/shift-10 CDF-index numerator exceeds INT32; "
                "the optimized C++ kernel cannot represent this model"
            )

        return cls(
            M=module.M,
            z_ch=module.z_ch,
            qp_num=module.qp_num,
            up_factor=module.up_factor,
            input_clip=input_clip,
            input_divisor=divisor,
            weight_q=weight_q,
            multiplier=multiplier64.to(torch.int32),
            affine_bias=affine_bias64.to(torch.int32),
        )

    def int8_storage_bytes(self):
        tensors = (
            self.input_divisor,
            self.weight_q,
            self.multiplier,
            self.affine_bias,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )


def _attach_cached_cdf_tables(codec, artifact):
    codec._entropy_control_checkpoint_sha256 = artifact.get(
        "y_cdf_calibration", {}
    ).get("checkpoint_sha256")
    if "cdf_tables" not in artifact:
        codec._y_cdf_mode = "gaussian"
        if hasattr(codec, "_cached_y_part_cdf"):
            del codec._cached_y_part_cdf
        return codec
    import numpy as np

    tables = artifact["cdf_tables"]
    codec._cached_z_cdfs = {
        int(qp): (
            torch.as_tensor(cdf).cpu().numpy(),
            torch.as_tensor(lengths).cpu().numpy().astype(np.int32),
        )
        for qp, (cdf, lengths) in tables["z"].items()
    }
    y_cdf, y_lengths = tables["y"]
    codec._cached_y_cdf = (
        torch.as_tensor(y_cdf).cpu().numpy(),
        torch.as_tensor(y_lengths).cpu().numpy().astype(np.int32),
    )
    from codec.models.entropy.y_cdf import Y_CDF_MODES, validate_part_cdf
    mode = artifact.get("y_cdf_mode", "gaussian")
    if mode not in Y_CDF_MODES:
        raise ValueError(f"unknown artifact y_cdf_mode: {mode!r}")
    if hasattr(codec, "_cached_y_part_cdf"):
        del codec._cached_y_part_cdf
    if "y_part_specific" in tables:
        if artifact.get("y_cdf_format_version") != 1:
            raise ValueError("unsupported or missing part-specific y CDF format version")
        cdf, lengths = tables["y_part_specific"]
        codec._cached_y_part_cdf = validate_part_cdf(
            torch.as_tensor(cdf).cpu().numpy(),
            torch.as_tensor(lengths).cpu().numpy(),
            codec._cached_y_cdf[1],
        )
    if mode == "part_specific" and not hasattr(codec, "_cached_y_part_cdf"):
        raise ValueError("part_specific artifact is missing its calibrated y CDFs")
    if hasattr(codec, "_cached_y_part_cdf"):
        mode = "part_specific"
    codec._y_cdf_mode = mode
    if "meta_prior" in tables:
        if getattr(codec, "meta_prior", None) is None:
            raise ValueError(
                "integer entropy artifact contains Meta Prior tables, "
                "but the model has Meta Prior disabled"
            )
        codec._cached_meta_prior_cdfs = {
            int(qp): (
                torch.as_tensor(cdf).cpu().numpy(),
                torch.as_tensor(lengths).cpu().numpy().astype(np.int32),
            )
            for qp, (cdf, lengths) in tables["meta_prior"].items()
        }
    index_merge = artifact.get("meta_prior_index_merge")
    if index_merge is not None:
        probabilities = np.ascontiguousarray(
            torch.as_tensor(index_merge["probabilities"])
            .cpu()
            .numpy()
            .astype(np.uint16)
        )
        qp_num = int(getattr(codec, "qp_num", probabilities.shape[0]))
        bank_count = int(
            getattr(getattr(codec, "meta_prior", None), "bank_count", 0)
        )
        if bank_count <= 1:
            raise ValueError(
                "integer entropy artifact contains Meta-Prior Index Merge "
                "state, but the model has Meta Prior disabled"
            )
        from codec.models.entropy.entropy_coding import (
            META_PRIOR_PROBABILITY_TOTAL,
            meta_prior_index_merge_probability_count,
        )

        expected_shape = (
            qp_num,
            meta_prior_index_merge_probability_count(bank_count),
        )
        if tuple(probabilities.shape) != expected_shape:
            raise ValueError(
                "invalid Meta-Prior Index Merge probability shape: "
                f"{tuple(probabilities.shape)} != {expected_shape}"
            )
        if (
            int(probabilities.min()) < 1
            or int(probabilities.max()) >= META_PRIOR_PROBABILITY_TOTAL
        ):
            raise ValueError(
                "Meta-Prior Index Merge probabilities must be in [1, 2047]"
            )
        adaptation_shift = int(index_merge.get("adaptation_shift", 5))
        if not 0 <= adaptation_shift <= 15:
            raise ValueError(
                "Meta-Prior Index Merge adaptation_shift must be in [0, 15]"
            )
        codec._cached_meta_prior_index_merge = {
            "probabilities": probabilities,
            "adaptation_shift": adaptation_shift,
        }
    return codec


def attach_integer_linear_cdf_index_decoder(codec, artifact_path):
    """Install the integer artifact for Linear CDF Index Decoding."""
    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    implementation = artifact.get("implementation")
    supported = {
        IntegerLinearCDFIndexDecoder.artifact_implementation,
    }
    if implementation not in supported:
        raise ValueError(
            "unsupported scale artifact implementation: "
            f"{implementation!r}; rebuild it with the Linear CDF Index "
            "Decoding exporter"
        )
    config = dict(artifact["constructor"])
    expected = (
        int(getattr(codec, "M", codec.scale_dec.M)),
        int(getattr(codec, "z_ch", codec.scale_dec.z_ch)),
        int(getattr(codec, "qp_num", codec.scale_dec.qp_num)),
    )
    actual = (
        int(config["M"]),
        int(config["z_ch"]),
        int(config["qp_num"]),
    )
    if actual != expected:
        raise ValueError(
            "INT8 scale artifact architecture mismatch: "
            f"artifact M/z_ch/qp_num={actual}, model={expected}"
        )

    state = artifact["state_dict"]
    module = IntegerLinearCDFIndexDecoder(
        **config,
        input_divisor=state["input_divisor"],
        weight_q=state["weight_q"],
        multiplier=state["multiplier"],
        affine_bias=state["affine_bias"],
    )
    module.entropy_coding_version = int(
        artifact.get("entropy_coding_version", 0)
    )
    module.max_entropy_coding_value = int(
        artifact.get("max_entropy_coding_value", -1)
    )
    device = next(codec.parameters()).device
    codec.scale_dec = module.to(device).eval()
    _attach_cached_cdf_tables(codec, artifact)
    return codec


def attach_int8_scale_decoder(codec, artifact_path):
    """Backward-compatible alias for the strict integer runtime."""
    return attach_integer_linear_cdf_index_decoder(codec, artifact_path)


# Backward-compatible class alias. New code and release metadata use the
# algorithm name rather than the implementation precision.
Int8ScaleDecoder = IntegerLinearCDFIndexDecoder


def require_cross_platform_entropy(codec):
    """Validate the bit-exact integer path used across CPU/GPU platforms."""
    if not isinstance(codec.scale_dec, IntegerLinearCDFIndexDecoder):
        raise ValueError(
            "cross-platform mode requires the integer runtime for "
            "Linear CDF Index Decoding"
        )
    if not hasattr(codec, "_cached_z_cdfs") or not hasattr(
        codec,
        "_cached_y_cdf",
    ):
        raise ValueError(
            "INT8 scale artifact is missing precomputed rANS CDF tables"
        )
    if getattr(codec, "meta_prior_enabled", False) and not hasattr(
        codec,
        "_cached_meta_prior_cdfs",
    ):
        raise ValueError(
            "integer entropy artifact is missing Meta Prior CDF tables"
        )
    if (
        getattr(codec.scale_dec, "entropy_coding_version", 0) >= 3
        and getattr(codec, "meta_prior_enabled", False)
        and not hasattr(codec, "_cached_meta_prior_index_merge")
    ):
        raise ValueError(
            "integer entropy artifact is missing Meta-Prior Index Merge "
            "probabilities"
        )
    # Instantiate eagerly so missing/stale C++ extensions fail before coding.
    codec.scale_dec._get_cpp_backend()

    from codec.models.entropy.entropy_coding import (
        MAX_ENTROPY_CODING_VALUE,
    )

    if codec.scale_dec.entropy_coding_version not in {2, 3}:
        raise ValueError(
            "INT8 scale artifact predates the signed-rANS range fix"
        )
    if (
        codec.scale_dec.max_entropy_coding_value
        != MAX_ENTROPY_CODING_VALUE
    ):
        raise ValueError(
            "INT8 scale artifact entropy symbol range does not match runtime"
        )
    return codec
