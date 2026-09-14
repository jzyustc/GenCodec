// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dense_conv1x1_bf16.h"

#include "dense_conv1x1_bf16_dispatch.h"

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

at::Tensor dense_conv1x1_bias_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::optional<at::Tensor>& bias,
    const at::optional<at::Tensor>& residual, const at::optional<at::Tensor>& output_scale,
    int64_t output_channels, int64_t upscale_factor, int64_t output_chunks, bool fuse_silu,
    bool fuse_hard_gelu_030, bool fuse_sigmoid, bool fuse_scale_decoder, float scale_decoder_qp_bias)
{
    TORCH_CHECK(feature.device().is_cpu(), "feature must be a CPU tensor");
    TORCH_CHECK(feature.scalar_type() == at::kBFloat16, "feature must be BF16");
    TORCH_CHECK(feature.dim() == 4, "feature must have shape [N, C, H, W]");
    TORCH_CHECK(feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feature must be channels-last contiguous");

    TORCH_CHECK(packed_weight.device().is_cpu(), "packed_weight must be a CPU tensor");
    TORCH_CHECK(packed_weight.scalar_type() == at::kBFloat16, "packed_weight must be BF16");
    TORCH_CHECK(packed_weight.dim() == 3 && packed_weight.size(2) == 32,
                "packed_weight must have shape [ceil(Cin/2), ceil(Cout/16), 32]");
    TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");

    const bool has_bias = bias.has_value();
    if (has_bias) {
        TORCH_CHECK(bias->device().is_cpu(), "bias must be a CPU tensor");
        TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be BF16");
        TORCH_CHECK(bias->dim() == 1, "bias must have shape [Cout]");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        if (output_channels < 0) {
            output_channels = bias->size(0);
        } else {
            TORCH_CHECK(output_channels == bias->size(0),
                        "output_channels does not match bias channels");
        }
    } else {
        TORCH_CHECK(output_channels > 0, "output_channels must be provided when bias is undefined");
    }

    const int64_t batch = feature.size(0);
    const int64_t height = feature.size(2);
    const int64_t width = feature.size(3);
    const int64_t input_channels = feature.size(1);
    const int64_t input_pairs = (input_channels + 1) / 2;
    const int64_t output_blocks = (output_channels + 15) / 16;

    TORCH_CHECK(upscale_factor == 1 || upscale_factor == 2 || upscale_factor == 4,
                "upscale_factor must be 1, 2, or 4");
    TORCH_CHECK(output_channels % (upscale_factor * upscale_factor) == 0,
                "output_channels must be divisible by upscale_factor squared");
    TORCH_CHECK(output_chunks == 1 || output_chunks == 2 || output_chunks == 3,
                "output_chunks must be 1, 2, or 3");
    TORCH_CHECK(upscale_factor == 1 || output_chunks == 1,
                "pixel shuffle and output chunking cannot be combined");
    TORCH_CHECK(output_channels % output_chunks == 0,
                "output_channels must be divisible by output_chunks");
    TORCH_CHECK(output_chunks == 1 || (output_channels / output_chunks) % 16 == 0,
                "chunk channels must be divisible by 16");
    TORCH_CHECK(static_cast<int>(fuse_silu) + static_cast<int>(fuse_hard_gelu_030)
                        + static_cast<int>(fuse_sigmoid) + static_cast<int>(fuse_scale_decoder)
                    <= 1,
                "at most one activation epilogue can be enabled");
    TORCH_CHECK(
        !fuse_scale_decoder
            || (has_bias && upscale_factor > 1 && output_chunks == 1 && !output_scale.has_value()),
        "scale decoder epilogue requires a biased pixel-shuffle convolution");

    const bool has_residual = residual.has_value();
    if (has_residual) {
        TORCH_CHECK(upscale_factor == 1 && output_chunks == 1 && !fuse_silu && !fuse_hard_gelu_030
                        && !fuse_sigmoid,
                    "residual requires a plain, unshuffled dense convolution");
        TORCH_CHECK(residual->sizes() == at::IntArrayRef({ batch, output_channels, height, width })
                        && residual->device().is_cpu() && residual->scalar_type() == at::kBFloat16
                        && residual->is_contiguous(at::MemoryFormat::ChannelsLast),
                    "residual must be channels-last CPU BF16 and match output shape");
    }
    const bool has_output_scale = output_scale.has_value();
    if (has_output_scale) {
        TORCH_CHECK(output_scale->device().is_cpu() && output_scale->scalar_type() == at::kBFloat16
                        && output_scale->is_contiguous(),
                    "output_scale must be a contiguous CPU BF16 tensor");
        TORCH_CHECK(output_scale->sizes() == at::IntArrayRef({ 1, 1, 1, 1 })
                        || output_scale->sizes() == at::IntArrayRef({ 1, output_channels, 1, 1 }),
                    "output_scale must have shape [1, 1, 1, 1] or [1, Cout, 1, 1]");
    }
    const bool has_scalar_scale = has_output_scale && output_scale->numel() == 1;
    const bool has_channel_scale = has_output_scale && output_scale->numel() == output_channels;
    TORCH_CHECK(!has_channel_scale || has_residual, "per-channel output_scale requires residual");

    TORCH_CHECK(packed_weight.size(0) == input_pairs,
                "packed_weight input-pair dimension does not match feature channels");
    TORCH_CHECK(packed_weight.size(1) == output_blocks,
                "packed_weight output-block dimension does not match bias channels");

    const int64_t shuffled_channels =
        output_channels / (upscale_factor * upscale_factor * output_chunks);
    at::Tensor output = at::empty({ batch * output_chunks, shuffled_channels,
                                    height * upscale_factor, width * upscale_factor },
                                  feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* packed_weight_ptr = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_ptr = has_bias ? bias->data_ptr<at::BFloat16>() : nullptr;
    const auto* residual_ptr = has_residual ? residual->data_ptr<at::BFloat16>() : nullptr;
    const auto* channel_scale_ptr = has_channel_scale ? output_scale->data_ptr<at::BFloat16>() : nullptr;
    auto* output_ptr = output.data_ptr<at::BFloat16>();

    const int64_t pixels = batch * height * width;
    const float scalar_output_scale = fuse_scale_decoder
                                          ? scale_decoder_qp_bias
                                          : (has_scalar_scale ? output_scale->item<float>() : 1.0f);
    const DispatchArgs args{ input_channels,
                             output_channels,
                             feature_ptr,
                             packed_weight_ptr,
                             bias_ptr,
                             residual_ptr,
                             channel_scale_ptr,
                             output_ptr,
                             pixels,
                             height,
                             width,
                             scalar_output_scale };

    bool dispatched;
    const char* dispatch_family;
    if (has_residual) {
        dispatch_family = "plain residual";
        dispatched = dispatch_dense_plain(args, has_bias, true, has_scalar_scale, has_channel_scale);
        TORCH_CHECK(dispatched, "unsupported dense 1x1 residual channel pair (Cin=", input_channels,
                    ", Cout=", output_channels, ")");
        return output;
    }
    if (output_chunks != 1) {
        dispatch_family = "output chunks";
        dispatched = dispatch_dense_chunks(args, has_bias, has_scalar_scale, fuse_silu,
                                           fuse_hard_gelu_030, fuse_sigmoid, output_chunks);
    } else if (upscale_factor != 1) {
        dispatch_family = "upsample";
        dispatched =
            dispatch_dense_upsample(args, has_bias, has_scalar_scale, fuse_silu, fuse_hard_gelu_030,
                                    fuse_sigmoid, fuse_scale_decoder, upscale_factor);
    } else if (fuse_silu || fuse_hard_gelu_030 || fuse_sigmoid) {
        dispatch_family = "activation";
        dispatched = dispatch_dense_activation(args, has_bias, has_scalar_scale, fuse_silu,
                                               fuse_hard_gelu_030, fuse_sigmoid);
    } else {
        dispatch_family = "plain";
        dispatched = dispatch_dense_plain(args, has_bias, false, has_scalar_scale, false);
    }
    TORCH_CHECK(dispatched, "unsupported dense 1x1 ", dispatch_family,
                " channel pair (Cin=", input_channels, ", Cout=", output_channels, ")");
    return output;
}
