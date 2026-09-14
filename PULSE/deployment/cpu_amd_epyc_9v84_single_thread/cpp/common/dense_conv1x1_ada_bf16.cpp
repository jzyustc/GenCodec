#include "model_config.h"
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dense_conv1x1_ada_bf16.h"

#include "activations_avx512.h"

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_ada_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <cstdint>
#include <immintrin.h>

namespace {

constexpr int CHANNELS = PULSE_RENDERER_CHANNELS;
constexpr int OUTPUT_CHANNELS = 3 * CHANNELS;
constexpr int CHANNEL_BLOCK = 16;
constexpr int INPUT_PAIRS = CHANNELS / 2;
constexpr int OUTPUT_BLOCKS = OUTPUT_CHANNELS / CHANNEL_BLOCK;
constexpr int PART_BLOCKS = CHANNELS / CHANNEL_BLOCK;
constexpr int PIXEL_TILE = 5;

using AliasedUint32 = uint32_t __attribute__((may_alias));

inline __m512 load_bf16(const at::BFloat16* input)
{
    return _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input)));
}

inline void store_bf16(at::BFloat16* output, __m512 value)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), (__m256i)_mm512_cvtneps_pbh(value));
}

inline void store_affine_and_gate(const at::BFloat16* residual, at::BFloat16* affine_output,
                                  at::BFloat16* gate_output, __m512 shift_accumulator,
                                  __m512 scale_accumulator, __m512 gate_accumulator)
{
    const __m512 affine_gate = hard_sigmoid(scale_accumulator);
    store_bf16(affine_output, _mm512_fmadd_ps(load_bf16(residual), affine_gate, shift_accumulator));
    store_bf16(gate_output, gate_accumulator);
}

inline __m512 load_bias(const at::BFloat16* bias, int block)
{
    return load_bf16(bias + block * CHANNEL_BLOCK);
}

void ada_pixel_tile(const at::BFloat16* feature, const at::BFloat16* residual,
                    const at::BFloat16* packed_weight, const at::BFloat16* bias,
                    at::BFloat16* affine_output, at::BFloat16* gate_output, int64_t pixel, int block)
{
    const int shift_block = block;
    const int scale_block = PART_BLOCKS + block;
    const int gate_block = 2 * PART_BLOCKS + block;

    __m512 shift0 = load_bias(bias, shift_block);
    __m512 shift1 = shift0;
    __m512 shift2 = shift0;
    __m512 shift3 = shift0;
    __m512 shift4 = shift0;
    __m512 scale0 = load_bias(bias, scale_block);
    __m512 scale1 = scale0;
    __m512 scale2 = scale0;
    __m512 scale3 = scale0;
    __m512 scale4 = scale0;
    __m512 gate0 = load_bias(bias, gate_block);
    __m512 gate1 = gate0;
    __m512 gate2 = gate0;
    __m512 gate3 = gate0;
    __m512 gate4 = gate0;

    const auto* feature0 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 0) * CHANNELS);
    const auto* feature1 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 1) * CHANNELS);
    const auto* feature2 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 2) * CHANNELS);
    const auto* feature3 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 3) * CHANNELS);
    const auto* feature4 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 4) * CHANNELS);

    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input0 = (__m512bh)_mm512_set1_epi32(feature0[pair]);
        const __m512bh input1 = (__m512bh)_mm512_set1_epi32(feature1[pair]);
        const __m512bh input2 = (__m512bh)_mm512_set1_epi32(feature2[pair]);
        const __m512bh input3 = (__m512bh)_mm512_set1_epi32(feature3[pair]);
        const __m512bh input4 = (__m512bh)_mm512_set1_epi32(feature4[pair]);
        const at::BFloat16* pair_weight = packed_weight + pair * OUTPUT_BLOCKS * 32;
        const __m512bh shift_weight = (__m512bh)_mm512_loadu_si512(pair_weight + shift_block * 32);
        const __m512bh scale_weight = (__m512bh)_mm512_loadu_si512(pair_weight + scale_block * 32);
        const __m512bh gate_weight = (__m512bh)_mm512_loadu_si512(pair_weight + gate_block * 32);

#define ACCUMULATE(PIXEL)                                                      \
    shift##PIXEL = _mm512_dpbf16_ps(shift##PIXEL, input##PIXEL, shift_weight); \
    scale##PIXEL = _mm512_dpbf16_ps(scale##PIXEL, input##PIXEL, scale_weight); \
    gate##PIXEL = _mm512_dpbf16_ps(gate##PIXEL, input##PIXEL, gate_weight)
        ACCUMULATE(0);
        ACCUMULATE(1);
        ACCUMULATE(2);
        ACCUMULATE(3);
        ACCUMULATE(4);
#undef ACCUMULATE
    }

#define STORE(PIXEL)                                                                          \
    store_affine_and_gate(residual + (pixel + PIXEL) * CHANNELS + block * CHANNEL_BLOCK,      \
                          affine_output + (pixel + PIXEL) * CHANNELS + block * CHANNEL_BLOCK, \
                          gate_output + (pixel + PIXEL) * CHANNELS + block * CHANNEL_BLOCK,   \
                          shift##PIXEL, scale##PIXEL, gate##PIXEL)
    STORE(0);
    STORE(1);
    STORE(2);
    STORE(3);
    STORE(4);
#undef STORE
}

void ada_single_pixel(const at::BFloat16* feature, const at::BFloat16* residual,
                      const at::BFloat16* packed_weight, const at::BFloat16* bias,
                      at::BFloat16* affine_output, at::BFloat16* gate_output, int64_t pixel, int block)
{
    const int shift_block = block;
    const int scale_block = PART_BLOCKS + block;
    const int gate_block = 2 * PART_BLOCKS + block;
    __m512 shift = load_bias(bias, shift_block);
    __m512 scale = load_bias(bias, scale_block);
    __m512 gate = load_bias(bias, gate_block);
    const auto* input = reinterpret_cast<const AliasedUint32*>(feature + pixel * CHANNELS);

    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(input[pair]);
        const at::BFloat16* pair_weight = packed_weight + pair * OUTPUT_BLOCKS * 32;
        shift = _mm512_dpbf16_ps(shift, input_vector,
                                 (__m512bh)_mm512_loadu_si512(pair_weight + shift_block * 32));
        scale = _mm512_dpbf16_ps(scale, input_vector,
                                 (__m512bh)_mm512_loadu_si512(pair_weight + scale_block * 32));
        gate = _mm512_dpbf16_ps(gate, input_vector,
                                (__m512bh)_mm512_loadu_si512(pair_weight + gate_block * 32));
    }

    const int64_t offset = pixel * CHANNELS + block * CHANNEL_BLOCK;
    store_affine_and_gate(residual + offset, affine_output + offset, gate_output + offset, shift,
                          scale, gate);
}

}  // namespace

at::Tensor dense_conv1x1_ada_hard_sigmoid_affine_avx512_bf16(const at::Tensor& feature,
                                                             const at::Tensor& residual,
                                                             const at::Tensor& packed_weight,
                                                             const at::Tensor& bias)
{
    TORCH_CHECK(feature.device().is_cpu() && feature.scalar_type() == at::kBFloat16
                    && feature.dim() == 4 && feature.size(1) == CHANNELS
                    && feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "unexpected tensor shape, layout or dtype");
    TORCH_CHECK(residual.sizes() == feature.sizes() && residual.device().is_cpu()
                    && residual.scalar_type() == at::kBFloat16
                    && residual.is_contiguous(at::MemoryFormat::ChannelsLast),
                "residual must match feature and be channels-last CPU BF16");
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.sizes() == at::IntArrayRef({ INPUT_PAIRS, OUTPUT_BLOCKS, 32 })
                    && packed_weight.is_contiguous(),
                "unexpected tensor shape, layout or dtype");
    TORCH_CHECK(bias.device().is_cpu() && bias.scalar_type() == at::kBFloat16
                    && bias.sizes() == at::IntArrayRef({ OUTPUT_CHANNELS }) && bias.is_contiguous(),
                "unexpected tensor shape, layout or dtype");

    const int64_t batch = feature.size(0);
    const int64_t height = feature.size(2);
    const int64_t width = feature.size(3);
    const int64_t pixels = batch * height * width;
    at::Tensor output = at::empty({ batch * 2, CHANNELS, height, width },
                                  feature.options().memory_format(at::MemoryFormat::ChannelsLast));

    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* residual_ptr = residual.data_ptr<at::BFloat16>();
    const auto* weight_ptr = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_ptr = bias.data_ptr<at::BFloat16>();
    auto* affine_ptr = output.data_ptr<at::BFloat16>();
    auto* gate_ptr = affine_ptr + pixels * CHANNELS;

#pragma omp parallel for schedule(static) num_threads(1)
    for (int block = 0; block < PART_BLOCKS; ++block) {
        int64_t pixel = 0;
        for (; pixel + PIXEL_TILE <= pixels; pixel += PIXEL_TILE) {
            ada_pixel_tile(feature_ptr, residual_ptr, weight_ptr, bias_ptr, affine_ptr, gate_ptr,
                           pixel, block);
        }
        for (; pixel < pixels; ++pixel) {
            ada_single_pixel(feature_ptr, residual_ptr, weight_ptr, bias_ptr, affine_ptr, gate_ptr,
                             pixel, block);
        }
    }
    return output;
}
