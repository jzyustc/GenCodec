// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// PULSE-L renderer: 48->96 periodic embedding and 96->96 gated residual.
// Keep six output vectors live across a small spatial tile: no intermediate
// FP32 image, separate bias pass, or materialized positional embedding.
#include "dense_conv1x1_periodic_bias_bf16.h"
#include "dense_conv1x1_gated_residual_bf16.h"
#include "activations_avx512.h"
#include <cstdint>

#ifndef PULSE_L_RENDERER_TILE
#define PULSE_L_RENDERER_TILE 4
#endif
static_assert(PULSE_L_RENDERER_TILE >= 1 && PULSE_L_RENDERER_TILE <= 4,
              "renderer tile must be in [1,4]");

namespace {
using BF = at::BFloat16;
using U32 = uint32_t __attribute__((may_alias));
constexpr int C = 96;
constexpr int BLOCKS = C / 16;

inline __m512 load(const BF* p) {
    return _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256((const __m256i*)p));
}
inline void store(BF* p, __m512 v) {
    _mm256_storeu_si256((__m256i*)p, (__m256i)_mm512_cvtneps_pbh(v));
}
void check_feature(const at::Tensor& t, int c) {
    TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kBFloat16 &&
                t.dim() == 4 && t.size(1) == c &&
                t.is_contiguous(at::MemoryFormat::ChannelsLast),
                "PULSE-L kernel requires channels-last CPU BF16 with C=", c);
}

template<int CIN, int TILE, bool GATED>
inline void tile(const BF* input, const BF* weights, const BF* bias,
                 const BF* gate, const BF* residual, const BF* scale,
                 BF* output, int64_t pixel, int64_t plane, int64_t width) {
    __m512 acc[TILE][BLOCKS];
    #pragma GCC unroll 8
    for (int t = 0; t < TILE; ++t) {
        #pragma GCC unroll 8
        for (int b = 0; b < BLOCKS; ++b)
            acc[t][b] = GATED ? load(bias + b*16) : _mm512_setzero_ps();
    }
    for (int pair = 0; pair < CIN/2; ++pair) {
        #pragma GCC unroll 8
        for (int b = 0; b < BLOCKS; ++b) {
            const __m512bh w = (__m512bh)_mm512_loadu_si512(weights + (pair*BLOCKS+b)*32);
            #pragma GCC unroll 8
            for (int t = 0; t < TILE; ++t) {
                const auto* x = reinterpret_cast<const U32*>(input + (pixel+t)*CIN);
                const __m512bh v = (__m512bh)_mm512_set1_epi32(x[pair]);
                acc[t][b] = _mm512_dpbf16_ps(acc[t][b], v, w);
            }
        }
    }
    #pragma GCC unroll 8
    for (int t = 0; t < TILE; ++t) {
        const int64_t spatial = (pixel+t) % plane;
        const int64_t phase = ((spatial / width) & 3)*4 + (spatial % width & 3);
        #pragma GCC unroll 8
        for (int b = 0; b < BLOCKS; ++b) {
            const int64_t off = (pixel+t)*C + b*16;
            __m512 v = acc[t][b];
            if constexpr (GATED) {
                v = _mm512_fmadd_ps(hard_sigmoid(load(gate+off)), v, load(residual+off));
                v = _mm512_mul_ps(v, load(scale+b*16));
            } else {
                v = _mm512_add_ps(v, load(bias + phase*C+b*16));
            }
            store(output+off, v);
        }
    }
}

template<int CIN, bool GATED>
at::Tensor run(const at::Tensor& feature, const at::Tensor& weight,
               const at::Tensor& bias, const at::Tensor* gate=nullptr,
               const at::Tensor* residual=nullptr, const at::Tensor* scale=nullptr) {
    check_feature(feature, CIN);
    TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type()==at::kBFloat16 &&
                weight.is_contiguous() &&
                weight.sizes()==at::IntArrayRef({CIN/2,BLOCKS,32}), "invalid packed weights");
    TORCH_CHECK(bias.device().is_cpu() && bias.scalar_type()==at::kBFloat16 &&
                (GATED ? bias.is_contiguous() :
                    bias.is_contiguous(at::MemoryFormat::ChannelsLast)) &&
                bias.numel()==(GATED ? C : 16*C), "invalid bias");
    if constexpr(GATED) {
        check_feature(*gate, C);
        check_feature(*residual, C);
        TORCH_CHECK(gate->sizes()==feature.sizes() && residual->sizes()==feature.sizes(),
                    "gate and residual shapes must match");
        TORCH_CHECK(scale->device().is_cpu() && scale->scalar_type()==at::kBFloat16 &&
                    scale->is_contiguous() && scale->numel()==C, "invalid channel scale");
    }
    const int64_t h=feature.size(2), w=feature.size(3), plane=h*w;
    TORCH_CHECK(plane > 0, "empty spatial dimensions");
    const int64_t pixels=feature.size(0)*plane;
    auto result=at::empty({feature.size(0),C,h,w},
        feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* x=feature.data_ptr<BF>(); const auto* wt=weight.data_ptr<BF>();
    const auto* b=bias.data_ptr<BF>();
    const auto* g=GATED ? gate->data_ptr<BF>() : nullptr;
    const auto* r=GATED ? residual->data_ptr<BF>() : nullptr;
    const auto* s=GATED ? scale->data_ptr<BF>() : nullptr;
    auto* out=result.data_ptr<BF>();
    int64_t p=0;
    for (;p+PULSE_L_RENDERER_TILE<=pixels;p+=PULSE_L_RENDERER_TILE)
        tile<CIN,PULSE_L_RENDERER_TILE,GATED>(x,wt,b,g,r,s,out,p,plane,w);
    for (;p<pixels;++p)
        tile<CIN,1,GATED>(x,wt,b,g,r,s,out,p,plane,w);
    return result;
}
} // namespace

at::Tensor dense_conv1x1_periodic_bias4x4_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight,
    const at::Tensor& periodic_bias) {
    return run<48,false>(feature,packed_weight,periodic_bias);
}

at::Tensor dense_conv1x1_gated_residual_scale_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias,
    const at::Tensor& gate, const at::Tensor& residual, const at::Tensor& output_scale) {
    return run<96,true>(feature,packed_weight,bias,&gate,&residual,&output_scale);
}
