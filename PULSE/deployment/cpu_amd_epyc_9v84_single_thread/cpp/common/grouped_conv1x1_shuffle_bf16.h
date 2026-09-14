#pragma once

#include <torch/extension.h>

at::Tensor
grouped_conv1x1_shuffle_bias_avx512_bf16(const at::Tensor& feature, const at::Tensor& packed_weight,
                                         const at::optional<at::Tensor>& bias = at::nullopt,
                                         const at::optional<at::Tensor>& residual = at::nullopt,
                                         const at::optional<at::Tensor>& output_scale = at::nullopt);