// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "decoder_proxy.h"

#include <cstring>
#include <immintrin.h>

namespace {

inline void convert_i8_to_bf16(const int8_t* input, at::BFloat16* output, int64_t count)
{
    int64_t channel = 0;
    for (; channel + 16 <= count; channel += 16) {
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + channel));
        const __m256i values_i16 = _mm256_cvtepi8_epi16(packed);
        const __m512i values_i32 = _mm512_cvtepi16_epi32(values_i16);
        const __m512 values_f32 = _mm512_cvtepi32_ps(values_i32);
        const __m256i values_bf16 = (__m256i)_mm512_cvtneps_pbh(values_f32);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(output + channel),
            values_bf16);
    }
    for (; channel < count; ++channel) {
        output[channel] =
            at::BFloat16(static_cast<float>(input[channel]));
    }
}

}  // namespace

void DecodeWrapperProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t mask_height,
                                   int64_t mask_width)
{
    TORCH_CHECK(mask_height > 0 && mask_width > 0, "mask dimensions must be positive");
    const int64_t mask_channels = sd.at("entropy_model.em_step1.out_proj.bias").size(0) / 2;
    at::Tensor mask_reference = at::empty(
        { 1, mask_channels, mask_height, mask_width },
        sd.at("entropy_model.em_step1.out_proj.bias")
            .options()
            .memory_format(at::MemoryFormat::ChannelsLast));
    mask0_ = checkerboard_mask0(mask_reference);

    qp_num_ = sd.at("entropy_model.q_scale_hyper_z").size(0);
    TORCH_CHECK(qp_num_ > 0 && qp_num_ <= g_max_qp_num, "QP row count must be in [1, ",
                g_max_qp_num, "], got ", qp_num_);
    for (const char* key :
         { "entropy_model.q_scale_hyper_z", "entropy_model.q_scale_hyper_m",
           "entropy_model.q_basic", "entropy_model.q_scale_prior_2m", "decoder.q_scale_dico",
           "decoder.renderer.q_scale", "decoder.q_scale_codec_dec" }) {
        const at::Tensor& value = sd.at(key);
        TORCH_CHECK(value.device().is_cpu() && value.scalar_type() == at::kBFloat16
                        && value.dim() == 2 && value.size(0) == qp_num_,
                    key, " must be CPU BF16 [", qp_num_, ", C]");
    }
    for (int64_t i = 0; i < qp_num_; ++i) {
        auto row = [&](const std::string& key) {
            at::Tensor output =
                sd.at(key).index({ at::indexing::Slice(i, i + 1), at::indexing::Slice(),
                                   at::indexing::None, at::indexing::None });
            return output;
        };
        q_hyper_z_[i] = row("entropy_model.q_scale_hyper_z").clamp_min(0.0);
        q_hyper_m_basic_[i] = row("entropy_model.q_scale_hyper_m").clamp_min(0.0)
                              * row("entropy_model.q_basic").clamp_min(0.0);
        q_prior_[i] = row("entropy_model.q_scale_prior_2m").clamp_min(0.0);
        dico_scale_[i] = row("decoder.q_scale_dico");
        resb_scale_[i] = row("decoder.renderer.q_scale");
        codec_scale_[i] = row("decoder.q_scale_codec_dec").clamp_min(0.0);
    }

    hyper_dec_.set_param(get_submodule_state_dict(sd, "entropy_model.hyper_dec."), g_entropy_groups);
    y_prior_fusion_.set_param(get_submodule_state_dict(sd, "entropy_model.em_step1."),
                              g_entropy_groups);
    y_spatial_prior_.set_param(get_submodule_state_dict(sd, "entropy_model.em_step2."),
                               g_entropy_groups);
    scale_dec_.set_param(get_submodule_state_dict(sd, "entropy_model.scale_dec."), g_pixel_upsample);
    TORCH_CHECK(scale_dec_.qp_num() == qp_num_, "scale decoder QP row count must be ", qp_num_,
                ", got ", scale_dec_.qp_num());

    up0_.set_param(get_submodule_state_dict(sd, "decoder.up0."), g_body_upsample);
    for (int i = 0; i < g_num_cond_blocks; ++i) {
        blocks_[i].set_param(
            get_submodule_state_dict(sd, "decoder.blocks." + std::to_string(i) + "."));
    }
    dec_cond_embed_.set_param(sd.at("decoder.renderer.condition_proj.weight"),
                              sd.at("decoder.renderer.condition_proj.bias"), g_patch_size, 1, true);
    pixel_head_.set_param(sd, g_patch_size);
    res_block_.set_param(get_submodule_state_dict(sd, "decoder.renderer."));
    final_layer_.set_param(get_submodule_state_dict(sd, "decoder.renderer.output."),
                           g_pixel_upsample);
}

std::pair<at::Tensor, at::Tensor> DecodeWrapperProxy::forward(const at::Tensor& z_hat,
                                                              const at::Tensor& symbols0,
                                                              const at::Tensor& symbols1, int64_t qp) const
{
    at::Tensor rgb = decode_rgb(z_hat, symbols0, symbols1, qp);
    at::Tensor scales = scale_dec_.forward(z_hat, qp);
    return { rgb, scales };
}

at::Tensor DecodeWrapperProxy::forward_rgb(const at::Tensor& z_hat, const at::Tensor& symbols0,
                                           const at::Tensor& symbols1, int64_t qp) const
{
    return decode_rgb(z_hat, symbols0, symbols1, qp);
}

at::Tensor DecodeWrapperProxy::forward_compact(const at::Tensor& z_i16,
                                               const at::Tensor& compact0_nhwc,
                                               const at::Tensor& compact1_nhwc, int64_t qp) const
{
    TORCH_CHECK(z_i16.device().is_cpu() && z_i16.scalar_type() == at::kShort
                    && z_i16.dim() == 4 && z_i16.is_contiguous(),
                "z_i16 must be contiguous CPU int16 NCHW");
    at::Tensor z_hat =
        z_i16.to(at::kBFloat16).contiguous(at::MemoryFormat::ChannelsLast);
    at::Tensor symbols0 = expand_compact_checkerboard(compact0_nhwc, 0);
    at::Tensor symbols1 = expand_compact_checkerboard(compact1_nhwc, 1);
    return decode_rgb(z_hat, symbols0, symbols1, qp);
}

at::Tensor DecodeWrapperProxy::expand_compact_checkerboard(const at::Tensor& compact_nhwc,
                                                            int part) const
{
    TORCH_CHECK(part == 0 || part == 1, "checkerboard part must be 0 or 1");
    TORCH_CHECK(compact_nhwc.device().is_cpu() && compact_nhwc.scalar_type() == at::kChar
                    && compact_nhwc.dim() == 4 && compact_nhwc.is_contiguous(),
                "compact checkerboard symbols must be contiguous CPU int8 NHWC");
    const int64_t batch = compact_nhwc.size(0);
    const int64_t height = compact_nhwc.size(1);
    const int64_t width = compact_nhwc.size(2);
    const int64_t compact_channels = compact_nhwc.size(3);
    const int64_t full_channels = compact_channels * 2;
    TORCH_CHECK(full_channels == mask0_.size(1) && height == mask0_.size(2)
                    && width == mask0_.size(3),
                "compact checkerboard shape does not match immutable decoder masks");

    at::Tensor output = at::empty(
        { batch, full_channels, height, width },
        q_hyper_z_[0].options().memory_format(at::MemoryFormat::ChannelsLast));
    auto* destination = output.data_ptr<at::BFloat16>();
    const auto* source = compact_nhwc.data_ptr<int8_t>();
    std::memset(destination, 0, output.numel() * sizeof(at::BFloat16));

    const int64_t pixels = batch * height * width;
    for (int64_t pixel = 0; pixel < pixels; ++pixel) {
        const int64_t spatial = pixel % (height * width);
        const int64_t y = spatial / width;
        const int64_t x = spatial - y * width;
        const bool even = ((x + y) & 1) == 0;
        const bool second_half = (part == 0) ? !even : even;
        const int64_t destination_offset =
            pixel * full_channels + (second_half ? compact_channels : 0);
        convert_i8_to_bf16(
            source + pixel * compact_channels,
            destination + destination_offset,
            compact_channels);
    }
    return output;
}

at::Tensor DecodeWrapperProxy::decode_rgb(const at::Tensor& z_hat, const at::Tensor& symbols0,
                                          const at::Tensor& symbols1, int64_t qp) const
{
    TORCH_CHECK(qp >= 0 && qp < qp_num_, "qp must be in [0, ", qp_num_, "), got ", qp);
    TORCH_CHECK(symbols0.size(1) == mask0_.size(1) && symbols0.size(2) == mask0_.size(2)
                    && symbols0.size(3) == mask0_.size(3),
                "symbols0 shape does not match immutable checkerboard masks");

    // q_hyper_m and q_basic are pre-multiplied per QP to replace two scalar passes with one.
    at::Tensor params = hyper_dec_.forward(z_hat, q_hyper_z_[qp], q_hyper_m_basic_[qp]);

    at::Tensor common_parts = y_prior_fusion_.forward(params, q_prior_[qp], symbols0, mask0_);

    auto parts = common_parts.chunk(3, 0);
    at::Tensor qstep = parts[0];
    at::Tensor means0 = parts[1];
    at::Tensor y_hat0 = parts[2];

    // Fuses y_hat1 = (symbols1 + means1) * (1 - mask0_), then
    // z = ((y_hat0 + y_hat1) * qstep.clamp_min(0.5)) * codec_scale_[qp].
    at::Tensor z = y_spatial_prior_.forward(y_hat0, symbols1, qstep, means0, q_prior_[qp], mask0_,
                                            codec_scale_[qp]);

    at::Tensor s = up0_.forward(z);
    for (int i = 0; i < g_num_cond_blocks; ++i) {
        s = blocks_[i].forward(s, dico_scale_[qp]);
    }

    at::Tensor x = pixel_head_.forward(z);
    at::Tensor modulation = dec_cond_embed_.forward_dense_conv1x1(s);
    x = res_block_.forward(x, modulation, resb_scale_[qp]);

    at::Tensor rgb = final_layer_.forward(x);
    return rgb;
}
