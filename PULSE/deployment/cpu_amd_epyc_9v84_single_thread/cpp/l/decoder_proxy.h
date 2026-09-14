// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "common.h"
#include "submodule_proxy.h"

#include <array>

static constexpr int g_max_qp_num = 64;
static constexpr int g_entropy_groups = 4;
static constexpr int g_body_upsample = 1;
static constexpr int g_patch_size = 4;
static constexpr int g_pixel_upsample = 4;
static constexpr int g_num_cond_blocks = 4;

class DecodeWrapperProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t mask_height, int64_t mask_width);

    std::pair<at::Tensor, at::Tensor> forward(const at::Tensor& z_hat, const at::Tensor& symbols0,
                                              const at::Tensor& symbols1, int64_t qp) const;

    // Production receiver path. Linear CDF Index Decoding supplies entropy
    // indexes directly, so the legacy scale-decoder output is unnecessary.
    at::Tensor forward_rgb(const at::Tensor& z_hat, const at::Tensor& symbols0,
                           const at::Tensor& symbols1, int64_t qp) const;

    // Complete-codec path: consume the native entropy payload tensors without
    // materializing BF16 checkerboard inputs in Python.
    //   z_i16:         [N, 144, H/64, W/64], contiguous NCHW
    //   compact{0,1}:  [N, H/16, W/16, 160], contiguous NHWC
    at::Tensor forward_compact(const at::Tensor& z_i16, const at::Tensor& compact0_nhwc,
                               const at::Tensor& compact1_nhwc, int64_t qp) const;
    // Separate diagnostic entry: normal timed path has no clocks/markers.
    std::map<std::string, double> profile_compact(
        const at::Tensor& z_i16, const at::Tensor& compact0_nhwc,
        const at::Tensor& compact1_nhwc, int64_t qp) const;

private:
    at::Tensor decode_rgb(const at::Tensor& z_hat, const at::Tensor& symbols0,
                          const at::Tensor& symbols1, int64_t qp) const;
    at::Tensor expand_compact_checkerboard(const at::Tensor& compact_nhwc, int part) const;

    int64_t qp_num_{ 0 };
    std::array<at::Tensor, g_max_qp_num> q_hyper_z_;        // (1,1,1,1)
    std::array<at::Tensor, g_max_qp_num> q_hyper_m_basic_;  // (1,1,1,1)
    std::array<at::Tensor, g_max_qp_num> q_prior_;          // (1,1,1,1)
    std::array<at::Tensor, g_max_qp_num> dico_scale_;       // (1, hidden_size, 1, 1)
    std::array<at::Tensor, g_max_qp_num> resb_scale_;       // (1, hidden_size_x, 1, 1)
    std::array<at::Tensor, g_max_qp_num> codec_scale_;      // (1,1,1,1)

    HyperDecoderProxy hyper_dec_;
    YPriorFusionProxy y_prior_fusion_;
    SpatialPriorProxy y_spatial_prior_;
    ScaleDecoderProxy scale_dec_;

    Up0Proxy up0_;
    std::array<DiCoBlockCAProxy, g_num_cond_blocks> blocks_;
    DenseConv1x1Bf16 dec_cond_embed_;
    PixelHeadProxy pixel_head_;
    ResBlockProxy res_block_;
    FinalLayerProxy final_layer_;

    at::Tensor mask0_;
};
