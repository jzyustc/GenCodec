// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_bias_avx512_bf16(const at::Tensor& feature, const at::Tensor& packed_weight,
                                          const at::optional<at::Tensor>& bias = at::nullopt,
                                          const at::optional<at::Tensor>& residual = at::nullopt,
                                          const at::optional<at::Tensor>& output_scale = at::nullopt,
                                          int64_t output_channels = -1, int64_t upscale_factor = 1,
                                          int64_t output_chunks = 1, bool fuse_silu = false,
                                          bool fuse_hard_gelu_030 = false,
                                          bool fuse_sigmoid = false, bool fuse_scale_decoder = false,
                                          float scale_decoder_qp_bias = 0.0f);
