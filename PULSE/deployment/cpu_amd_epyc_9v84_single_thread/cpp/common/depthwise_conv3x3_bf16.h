// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> depthwise_conv3x3_bias_hard_gelu_030_mean_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias);
