// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

struct DispatchArgs {
    int64_t input_channels;
    int64_t output_channels;
    const at::BFloat16* feature;
    const at::BFloat16* packed_weight;
    const at::BFloat16* bias;
    const at::BFloat16* residual;
    const at::BFloat16* channel_scale;
    at::BFloat16* output;
    int64_t pixels;
    int64_t height;
    int64_t width;
    float output_scale;
};

bool dispatch_dense_plain(const DispatchArgs& args, bool has_bias, bool has_residual,
                          bool has_scalar_scale, bool has_channel_scale);
bool dispatch_dense_activation(const DispatchArgs& args, bool has_bias, bool has_scalar_scale,
                               bool has_silu, bool has_hard_gelu_030, bool has_sigmoid);
bool dispatch_dense_upsample(const DispatchArgs& args, bool has_bias, bool has_scalar_scale,
                             bool has_silu, bool has_hard_gelu_030, bool has_sigmoid,
                             bool has_scale_decoder, int64_t upscale_factor);
bool dispatch_dense_chunks(const DispatchArgs& args, bool has_bias, bool has_scalar_scale, bool has_silu,
                           bool has_hard_gelu_030, bool has_sigmoid, int64_t output_chunks);
