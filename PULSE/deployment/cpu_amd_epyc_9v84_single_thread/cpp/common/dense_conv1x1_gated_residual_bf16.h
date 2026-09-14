#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_gated_residual_scale_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias,
    const at::Tensor& gate, const at::Tensor& residual, const at::Tensor& output_scale);