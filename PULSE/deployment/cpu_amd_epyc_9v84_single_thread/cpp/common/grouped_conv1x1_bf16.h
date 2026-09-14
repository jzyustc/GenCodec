// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

at::Tensor grouped_conv1x1_bias_avx512_bf16(const at::Tensor& feature, const at::Tensor& packed_weight,
                                            const at::optional<at::Tensor>& bias = at::nullopt,
                                            int64_t output_channels = -1,
                                            bool fuse_hard_gelu_030 = false);
