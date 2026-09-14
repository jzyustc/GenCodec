// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

class DenseConv1x1Bf16 {
public:
    void set_param(const at::Tensor& weight, const at::optional<at::Tensor>& bias = at::nullopt,
                   int64_t upscale_factor = 1, int64_t output_chunks = 1, bool fuse_silu = false,
                   bool fuse_hard_gelu_030 = false, bool fuse_sigmoid = false);
    // Ordering: fold input_scale into packed weights, convolve, add residual, apply output_scale.
    at::Tensor forward_dense_conv1x1(const at::Tensor& feature,
                                     const at::optional<at::Tensor>& input_scale = at::nullopt,
                                     const at::optional<at::Tensor>& residual = at::nullopt,
                                     const at::optional<at::Tensor>& output_scale = at::nullopt) const;
    at::Tensor forward_dense_conv1x1_scale_decoder(const at::Tensor& feature, float qp_bias) const;
    at::Tensor forward_dense_conv1x1_cat3(const at::Tensor& input0, const at::Tensor& input1,
                                          const at::Tensor& input2) const;
    at::Tensor forward_dense_conv1x1_gated_residual_scale(const at::Tensor& feature,
                                                          const at::Tensor& gate,
                                                          const at::Tensor& residual,
                                                          const at::Tensor& output_scale) const;
    at::Tensor forward_dense_conv1x1_periodic_bias4x4(const at::Tensor& feature,
                                                      const at::Tensor& periodic_bias) const;
    at::Tensor forward_dense_conv1x1_ada_hard_sigmoid_affine(const at::Tensor& feature,
                                                             const at::Tensor& residual) const;
    at::Tensor forward_dense_conv1x1_spatial_reconstruct(
        const at::Tensor& feature, const at::Tensor& y_hat0, const at::Tensor& symbols1,
        const at::Tensor& qstep, const at::Tensor& mask0, const at::Tensor& codec_scale) const;
    at::Tensor forward_dense_conv1x1_prior_masked_add(const at::Tensor& feature,
                                                      const at::Tensor& symbols0,
                                                      const at::Tensor& mask0) const;

private:
    int64_t output_channels_{ 0 };
    int64_t input_channels_{ 0 };
    int64_t upscale_factor_{ 1 };
    int64_t output_chunks_{ 1 };
    bool fuse_silu_{ false };
    bool fuse_hard_gelu_030_{ false };
    bool fuse_sigmoid_{ false };
    at::Tensor packed_weight_;
    at::optional<at::Tensor> bias_{ at::nullopt };
};

class GroupedConv1x1Bf16 {
public:
    void set_param(const at::Tensor& weight, const at::optional<at::Tensor>& bias = at::nullopt,
                   bool fuse_hard_gelu_030 = false, bool shuffle_output = false);
    // Ordering matches dense: input weight folding, convolution, residual, output scaling.
    at::Tensor forward_grouped_conv1x1(const at::Tensor& feature,
                                       const at::optional<at::Tensor>& input_scale = at::nullopt,
                                       const at::optional<at::Tensor>& residual = at::nullopt,
                                       const at::optional<at::Tensor>& output_scale = at::nullopt) const;

private:
    int64_t output_channels_{ 0 };
    int64_t input_channels_{ 0 };
    bool fuse_hard_gelu_030_{ false };
    bool shuffle_output_{ false };
    at::Tensor packed_weight_;
    at::optional<at::Tensor> bias_{ at::nullopt };
};

class DepthwiseConv3x3Bf16 {
public:
    void set_param(const at::Tensor& weight, const at::Tensor& bias);
    std::tuple<at::Tensor, at::Tensor>
    forward_depthwise_conv3x3_hard_gelu_030_mean(const at::Tensor& feature) const;

private:
    at::Tensor packed_weight_;
    at::Tensor bias_;
};
