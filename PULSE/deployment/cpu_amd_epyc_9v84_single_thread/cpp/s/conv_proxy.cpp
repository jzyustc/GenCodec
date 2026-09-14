// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <conv_proxy.h>

#include "dense_conv1x1_ada_bf16.h"
#include "dense_conv1x1_bf16.h"
#include "dense_conv1x1_cat3_bf16.h"
#include "dense_conv1x1_gated_residual_bf16.h"
#include "dense_conv1x1_periodic_bias_bf16.h"
#include "dense_conv1x1_prior_bf16.h"
#include "dense_conv1x1_reconstruct_bf16.h"
#include <depthwise_conv3x3_bf16.h>
#include "grouped_conv1x1_bf16.h"
#include "grouped_conv1x1_shuffle_bf16.h"

#include <immintrin.h>

static at::Tensor scale_packed_weights(const at::Tensor& packed_weight, const at::Tensor& input_scale,
                                       int64_t input_channels, int64_t output_channels, int64_t groups)
{
    const int64_t group_input_channels = input_channels / groups;
    const int64_t group_output_channels = output_channels / groups;
    const int64_t input_pairs = (group_input_channels + 1) / 2;
    const int64_t output_blocks = (group_output_channels + 15) / 16;
    const int64_t group_weight_elements = input_pairs * output_blocks * 32;
    thread_local at::Tensor scaled_weight;
    if (!scaled_weight.defined() || scaled_weight.sizes() != packed_weight.sizes()
        || scaled_weight.scalar_type() != packed_weight.scalar_type()
        || scaled_weight.device() != packed_weight.device()
        || scaled_weight.layout() != packed_weight.layout()) {
        scaled_weight = at::empty_like(packed_weight);
    }
    const auto* scale = input_scale.data_ptr<at::BFloat16>();
    const auto* source = packed_weight.data_ptr<at::BFloat16>();
    auto* destination = scaled_weight.data_ptr<at::BFloat16>();

    for (int64_t group = 0; group < groups; ++group) {
        const at::BFloat16* group_scale = scale + group * group_input_channels;
        const at::BFloat16* group_source = source + group * group_weight_elements;
        at::BFloat16* group_destination = destination + group * group_weight_elements;
        for (int64_t pair = 0; pair < input_pairs; ++pair) {
            const float scale0 = static_cast<float>(group_scale[pair * 2]);
            const float scale1 = pair * 2 + 1 < group_input_channels
                                     ? static_cast<float>(group_scale[pair * 2 + 1])
                                     : 0.0f;
            const __m512 scale_vector =
                _mm512_set_ps(scale1, scale0, scale1, scale0, scale1, scale0, scale1, scale0,
                              scale1, scale0, scale1, scale0, scale1, scale0, scale1, scale0);
            for (int64_t block = 0; block < output_blocks; ++block) {
                const int64_t offset = (pair * output_blocks + block) * 32;
                const __m256i weight0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_source + offset));
                const __m256i weight1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group_source + offset + 16));
                const __m256i scaled0 = (__m256i)_mm512_cvtneps_pbh(
                    _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)weight0), scale_vector));
                const __m256i scaled1 = (__m256i)_mm512_cvtneps_pbh(
                    _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)weight1), scale_vector));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(group_destination + offset), scaled0);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(group_destination + offset + 16),
                                    scaled1);
            }
        }
    }
    return scaled_weight;
}

static at::Tensor scale_shuffled_grouped_weights(const at::Tensor& packed_weight,
                                                 const at::Tensor& input_scale,
                                                 int64_t input_channels, int64_t output_channels)
{
    constexpr int64_t groups = 4;
    const int64_t group_input_channels = input_channels / groups;
    const int64_t input_pairs = (group_input_channels + 1) / 2;
    const int64_t output_blocks = (output_channels + 15) / 16;
    thread_local at::Tensor scaled_weight;
    if (!scaled_weight.defined() || scaled_weight.sizes() != packed_weight.sizes()
        || scaled_weight.scalar_type() != packed_weight.scalar_type()
        || scaled_weight.device() != packed_weight.device()
        || scaled_weight.layout() != packed_weight.layout()) {
        scaled_weight = at::empty_like(packed_weight);
    }
    const auto* scale = input_scale.data_ptr<at::BFloat16>();
    const auto* source = packed_weight.data_ptr<at::BFloat16>();
    auto* destination = scaled_weight.data_ptr<at::BFloat16>();

    for (int64_t pair = 0; pair < input_pairs; ++pair) {
        alignas(64) float scale_values[16];
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (int group = 0; group < groups; ++group) {
                const int64_t channel0 = pair * 2;
                scale_values[repeat * 8 + group * 2] =
                    static_cast<float>(scale[group * group_input_channels + channel0]);
                scale_values[repeat * 8 + group * 2 + 1] =
                    channel0 + 1 < group_input_channels
                        ? static_cast<float>(scale[group * group_input_channels + channel0 + 1])
                        : 0.0f;
            }
        }
        const __m512 scale_vector = _mm512_load_ps(scale_values);
        for (int64_t block = 0; block < output_blocks; ++block) {
            const int64_t offset = (pair * output_blocks + block) * 32;
            const __m256i weight0 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + offset));
            const __m256i weight1 =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + offset + 16));
            const __m256i scaled0 = (__m256i)_mm512_cvtneps_pbh(
                _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)weight0), scale_vector));
            const __m256i scaled1 = (__m256i)_mm512_cvtneps_pbh(
                _mm512_mul_ps(_mm512_cvtpbh_ps((__m256bh)weight1), scale_vector));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + offset), scaled0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + offset + 16), scaled1);
        }
    }
    return scaled_weight;
}

void DenseConv1x1Bf16::set_param(const at::Tensor& weight, const at::optional<at::Tensor>& bias,
                                 int64_t upscale_factor, int64_t output_chunks, bool fuse_silu,
                                 bool fuse_hard_gelu_030, bool fuse_sigmoid)
{
    TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kBFloat16,
                "weight must be a CPU BF16 tensor");
    TORCH_CHECK(weight.dim() == 4 && weight.size(2) == 1 && weight.size(3) == 1,
                "weight must have shape [Cout, Cin, 1, 1]");
    TORCH_CHECK(weight.is_contiguous(at::MemoryFormat::ChannelsLast),
                "weight must be channels-last contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cpu() && bias->scalar_type() == at::kBFloat16 && bias->dim() == 1
                        && bias->size(0) == weight.size(0) && bias->is_contiguous(),
                    "bias must be contiguous CPU BF16 [Cout]");
    }
    TORCH_CHECK(upscale_factor == 1 || upscale_factor == 2 || upscale_factor == 4,
                "upscale_factor must be 1, 2, or 4");
    const int64_t output_channels = weight.size(0);
    const int64_t upscale_area = upscale_factor * upscale_factor;
    TORCH_CHECK(output_channels % upscale_area == 0,
                "output channels must be divisible by upscale_factor squared");
    TORCH_CHECK(output_chunks == 1 || output_chunks == 2 || output_chunks == 3,
                "output_chunks must be 1, 2, or 3");
    TORCH_CHECK(upscale_factor == 1 || output_chunks == 1,
                "pixel shuffle and output chunking cannot be combined");
    TORCH_CHECK(output_channels % output_chunks == 0,
                "output channels must be divisible by output_chunks");
    output_channels_ = output_channels;
    upscale_factor_ = upscale_factor;
    output_chunks_ = output_chunks;
    fuse_silu_ = fuse_silu;
    fuse_hard_gelu_030_ = fuse_hard_gelu_030;
    fuse_sigmoid_ = fuse_sigmoid;
    const int64_t input_channels = weight.size(1);
    input_channels_ = input_channels;
    const int64_t input_pairs = (input_channels + 1) / 2;
    const int64_t output_blocks = (output_channels + 15) / 16;
    at::Tensor padded = at::zeros({ output_blocks * 16, input_pairs * 2 }, weight.options());
    at::Tensor matrix = weight.index({ at::indexing::Slice(), at::indexing::Slice(), 0, 0 });
    if (upscale_factor > 1) {
        const int64_t shuffled_channels = output_channels / upscale_area;
        matrix = matrix.view({ shuffled_channels, upscale_area, input_channels })
                     .permute({ 1, 0, 2 })
                     .reshape({ output_channels, input_channels });
    }
    padded.index_put_(
        { at::indexing::Slice(0, output_channels), at::indexing::Slice(0, input_channels) }, matrix);
    packed_weight_ = padded.view({ output_blocks, 16, input_pairs, 2 })
                         .permute({ 2, 0, 1, 3 })
                         .contiguous()
                         .view({ input_pairs, output_blocks, 32 });
    if (bias.has_value() && upscale_factor > 1) {
        const int64_t shuffled_channels = output_channels / upscale_area;
        bias_ = bias->view({ shuffled_channels, upscale_area })
                    .transpose(0, 1)
                    .reshape({ output_channels })
                    .contiguous();
    } else {
        bias_ = bias.has_value() ? at::optional<at::Tensor>(bias->contiguous()) : at::nullopt;
    }
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1(const at::Tensor& feature,
                                                   const at::optional<at::Tensor>& input_scale,
                                                   const at::optional<at::Tensor>& residual,
                                                   const at::optional<at::Tensor>& output_scale) const
{
    at::Tensor packed_weight = packed_weight_;
    if (input_scale.has_value()) {
        TORCH_CHECK(feature.size(0) == 1
                        && input_scale->sizes() == at::IntArrayRef({ 1, input_channels_, 1, 1 }),
                    "input_scale must have shape [1, Cin, 1, 1] for batch-1 gated convolution");
        TORCH_CHECK(input_scale->device().is_cpu() && input_scale->scalar_type() == at::kBFloat16,
                    "input_scale must be a CPU BF16 tensor");
        TORCH_CHECK(input_scale->is_contiguous(), "input_scale must be contiguous");
        packed_weight =
            scale_packed_weights(packed_weight_, *input_scale, input_channels_, output_channels_, 1);
    }
    at::Tensor output = dense_conv1x1_bias_avx512_bf16(
        feature, packed_weight, bias_, residual, output_scale, output_channels_, upscale_factor_,
        output_chunks_, fuse_silu_, fuse_hard_gelu_030_, fuse_sigmoid_);
    return output;
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_scale_decoder(const at::Tensor& feature,
                                                                 float qp_bias) const
{
    TORCH_CHECK(input_channels_ == 80 && output_channels_ == 4096 && upscale_factor_ == 4
                    && output_chunks_ == 1 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && bias_.has_value(),
                "scale decoder path requires a plain biased 80->4096 pixel-shuffle convolution");
    return dense_conv1x1_bias_avx512_bf16(feature, packed_weight_, bias_, at::nullopt, at::nullopt,
                                          output_channels_, upscale_factor_, output_chunks_, false,
                                          false, false, true, qp_bias);
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_cat3(const at::Tensor& input0,
                                                        const at::Tensor& input1,
                                                        const at::Tensor& input2) const
{
    TORCH_CHECK(input_channels_ == 768 && output_channels_ == 128 && upscale_factor_ == 1
                    && output_chunks_ == 1 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && bias_.has_value(),
                "three-input dense path requires plain biased 768->128 convolution");
    at::Tensor output =
        dense_conv1x1_cat3_bias_avx512_bf16(input0, input1, input2, packed_weight_, *bias_);
    return output;
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_gated_residual_scale(const at::Tensor& feature,
                                                                        const at::Tensor& gate,
                                                                        const at::Tensor& residual,
                                                                        const at::Tensor& output_scale) const
{
    TORCH_CHECK(input_channels_ == 32 && output_channels_ == 32 && upscale_factor_ == 1
                    && output_chunks_ == 1 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && bias_.has_value(),
                "gated residual dense path requires plain biased 32->32 convolution");
    at::Tensor output = dense_conv1x1_gated_residual_scale_avx512_bf16(
        feature, packed_weight_, *bias_, gate, residual, output_scale);
    return output;
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_periodic_bias4x4(const at::Tensor& feature,
                                                                    const at::Tensor& periodic_bias) const
{
    TORCH_CHECK(input_channels_ == 16 && output_channels_ == 32 && upscale_factor_ == 1
                    && output_chunks_ == 1 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && !bias_.has_value(),
                "periodic bias dense path requires plain unbiased 16->32 convolution");
    at::Tensor output =
        dense_conv1x1_periodic_bias4x4_avx512_bf16(feature, packed_weight_, periodic_bias);
    return output;
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_ada_hard_sigmoid_affine(const at::Tensor& feature,
                                                                           const at::Tensor& residual) const
{
    TORCH_CHECK(input_channels_ == 32 && output_channels_ == 96 && upscale_factor_ == 1
                    && output_chunks_ == 3 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && bias_.has_value(),
                "Ada affine path requires a plain biased 32->96 three-chunk convolution");
    return dense_conv1x1_ada_hard_sigmoid_affine_avx512_bf16(feature, residual, packed_weight_, *bias_);
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_spatial_reconstruct(
    const at::Tensor& feature, const at::Tensor& y_hat0, const at::Tensor& symbols1,
    const at::Tensor& qstep, const at::Tensor& mask0, const at::Tensor& codec_scale) const
{
    TORCH_CHECK(input_channels_ == 128 && output_channels_ == 256 && upscale_factor_ == 1
                    && output_chunks_ == 1 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && bias_.has_value(),
                "spatial reconstruction path requires a plain biased 128->256 convolution");
    return dense_conv1x1_spatial_reconstruct_avx512_bf16(feature, packed_weight_, *bias_, y_hat0,
                                                         symbols1, qstep, mask0, codec_scale);
}

at::Tensor DenseConv1x1Bf16::forward_dense_conv1x1_prior_masked_add(const at::Tensor& feature,
                                                                    const at::Tensor& symbols0,
                                                                    const at::Tensor& mask0) const
{
    TORCH_CHECK(input_channels_ == 128 && output_channels_ == 512 && upscale_factor_ == 1
                    && output_chunks_ == 2 && !fuse_silu_ && !fuse_hard_gelu_030_ && !fuse_sigmoid_
                    && bias_.has_value(),
                "prior masked-add path requires a plain biased 128->512 two-chunk convolution");
    return dense_conv1x1_prior_masked_add_avx512_bf16(feature, packed_weight_, *bias_, symbols0, mask0);
}

void GroupedConv1x1Bf16::set_param(const at::Tensor& weight, const at::optional<at::Tensor>& bias,
                                   bool fuse_hard_gelu_030, bool shuffle_output)
{
    constexpr int64_t groups = 4;
    TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == at::kBFloat16,
                "weight must be a CPU BF16 tensor");
    TORCH_CHECK(weight.dim() == 4 && weight.size(2) == 1 && weight.size(3) == 1,
                "weight must have shape [Cout, Cin/groups, 1, 1]");
    TORCH_CHECK(weight.size(0) % groups == 0, "Cout must be divisible by 4");
    TORCH_CHECK(weight.is_contiguous(at::MemoryFormat::ChannelsLast),
                "weight must be channels-last contiguous");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device().is_cpu() && bias->scalar_type() == at::kBFloat16 && bias->dim() == 1
                        && bias->size(0) == weight.size(0) && bias->is_contiguous(),
                    "bias must be contiguous CPU BF16 [Cout]");
    }
    const int64_t output_channels = weight.size(0);
    output_channels_ = output_channels;
    const int64_t group_input_channels = weight.size(1);
    input_channels_ = groups * group_input_channels;
    const int64_t group_output_channels = output_channels / groups;
    const int64_t input_pairs = (group_input_channels + 1) / 2;
    const int64_t output_blocks = (group_output_channels + 15) / 16;
    at::Tensor grouped_weight = weight.index({ at::indexing::Slice(), at::indexing::Slice(), 0, 0 })
                                    .view({ groups, group_output_channels, group_input_channels });
    if (shuffle_output) {
        const int64_t shuffled_output_blocks = (output_channels + 15) / 16;
        at::Tensor padded =
            at::zeros({ shuffled_output_blocks * 16, input_pairs * 2 }, weight.options());
        at::Tensor shuffled_weight =
            grouped_weight.permute({ 1, 0, 2 }).reshape({ output_channels, group_input_channels });
        padded.index_put_({ at::indexing::Slice(0, output_channels),
                            at::indexing::Slice(0, group_input_channels) },
                          shuffled_weight);
        packed_weight_ = padded.view({ shuffled_output_blocks, 16, input_pairs, 2 })
                             .permute({ 2, 0, 1, 3 })
                             .contiguous()
                             .view({ input_pairs, shuffled_output_blocks, 32 });
        bias_ = bias.has_value()
                    ? at::optional<at::Tensor>(bias->view({ groups, group_output_channels })
                                                   .transpose(0, 1)
                                                   .reshape({ output_channels })
                                                   .contiguous())
                    : at::nullopt;
    } else {
        at::Tensor padded =
            at::zeros({ groups, output_blocks * 16, input_pairs * 2 }, weight.options());
        padded.index_put_({ at::indexing::Slice(), at::indexing::Slice(0, group_output_channels),
                            at::indexing::Slice(0, group_input_channels) },
                          grouped_weight);
        packed_weight_ = padded.view({ groups, output_blocks, 16, input_pairs, 2 })
                             .permute({ 0, 3, 1, 2, 4 })
                             .contiguous()
                             .view({ groups, input_pairs, output_blocks, 32 });
        bias_ = bias.has_value() ? at::optional<at::Tensor>(bias->contiguous()) : at::nullopt;
    }
    fuse_hard_gelu_030_ = fuse_hard_gelu_030;
    shuffle_output_ = shuffle_output;
}

at::Tensor GroupedConv1x1Bf16::forward_grouped_conv1x1(const at::Tensor& feature,
                                                       const at::optional<at::Tensor>& input_scale,
                                                       const at::optional<at::Tensor>& residual,
                                                       const at::optional<at::Tensor>& output_scale) const
{
    at::Tensor packed_weight = packed_weight_;
    if (input_scale.has_value()) {
        constexpr int64_t groups = 4;
        TORCH_CHECK(feature.size(0) == 1
                        && input_scale->sizes() == at::IntArrayRef({ 1, input_channels_, 1, 1 }),
                    "input_scale must have shape [1, Cin, 1, 1] for batch-1 gated convolution");
        TORCH_CHECK(input_scale->device().is_cpu() && input_scale->scalar_type() == at::kBFloat16,
                    "input_scale must be a CPU BF16 tensor");
        TORCH_CHECK(input_scale->is_contiguous(), "input_scale must be contiguous");
        packed_weight = shuffle_output_
                            ? scale_shuffled_grouped_weights(packed_weight_, *input_scale,
                                                             input_channels_, output_channels_)
                            : scale_packed_weights(packed_weight_, *input_scale, input_channels_,
                                                   output_channels_, groups);
    }
    if (shuffle_output_) {
        TORCH_CHECK(!fuse_hard_gelu_030_, "shuffled grouped output does not support HardGELU030");
        at::Tensor output = grouped_conv1x1_shuffle_bias_avx512_bf16(feature, packed_weight, bias_,
                                                                     residual, output_scale);
        return output;
    }
    TORCH_CHECK(!residual.has_value(), "residual requires shuffled grouped output");
    TORCH_CHECK(!output_scale.has_value(), "output_scale requires shuffled grouped output");
    at::Tensor output = grouped_conv1x1_bias_avx512_bf16(feature, packed_weight, bias_,
                                                         output_channels_, fuse_hard_gelu_030_);
    return output;
}

void DepthwiseConv3x3Bf16::set_param(const at::Tensor& weight, const at::Tensor& bias)
{
    const int64_t channels = weight.size(0);
    const int64_t channel_blocks = (channels + 15) / 16;
    at::Tensor padded = at::zeros({ 9, channel_blocks * 16 }, weight.options().dtype(at::kFloat));
    at::Tensor flattened =
        weight.index({ at::indexing::Slice(), 0 }).permute({ 1, 2, 0 }).reshape({ 9, channels }).to(at::kFloat);
    padded.index_put_({ at::indexing::Slice(), at::indexing::Slice(0, channels) }, flattened);
    packed_weight_ = padded.view({ 9, channel_blocks, 16 });
    bias_ = bias.contiguous();
}

at::Tensor
DepthwiseConv3x3Bf16::forward_depthwise_conv3x3_hard_gelu_030(const at::Tensor& feature) const
{
    return depthwise_conv3x3_bias_hard_gelu_030_avx512_bf16(
        feature,
        packed_weight_,
        bias_);
}
