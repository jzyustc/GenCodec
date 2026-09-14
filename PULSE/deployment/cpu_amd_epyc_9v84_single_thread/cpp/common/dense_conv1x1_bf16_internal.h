// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "dense_conv1x1_bf16_dispatch.h"
#include "dense_conv1x1_pairs.h"

#include "activations_avx512.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_bf16_internal.h requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int PIXEL_TILE = 5;
#ifndef DENSE_OUTPUT_BLOCK_TILE
    #define DENSE_OUTPUT_BLOCK_TILE 0
#endif
static_assert(DENSE_OUTPUT_BLOCK_TILE >= 0 && DENSE_OUTPUT_BLOCK_TILE <= 5);
static constexpr int WEIGHT_PREFETCH_DISTANCE = 4;

using AliasedUint32 =
    uint32_t __attribute__((may_alias));  // Load two BF16 values without violating strict aliasing.

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, bool HAS_SILU, bool HAS_HARD_GELU_030,
          bool HAS_SIGMOID, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static constexpr int dense_output_block_tile()
{
    if constexpr (DENSE_OUTPUT_BLOCK_TILE > 0) {
        return DENSE_OUTPUT_BLOCK_TILE;
    } else if constexpr (UPSCALE_FACTOR == 2) {
        return INPUT_CHANNELS == 72 ? 3 : 2;
    } else if constexpr (UPSCALE_FACTOR == 4) {
        return INPUT_CHANNELS == 100 ? 4 : 3;
    } else if constexpr (OUTPUT_CHUNKS == 2) {
        return 2;
    } else if constexpr (OUTPUT_CHUNKS == 3) {
        return 3;
    } else if constexpr (HAS_SILU || HAS_HARD_GELU_030 || HAS_SIGMOID) {
        if constexpr ((INPUT_CHANNELS == 48 && OUTPUT_CHANNELS == 48)
                      || (INPUT_CHANNELS == 100 && OUTPUT_CHANNELS == 100)) {
            return 4;
        } else if constexpr (INPUT_CHANNELS == 148 && OUTPUT_CHANNELS == 148) {
            return 2;
        } else {
            return 3;
        }
    } else if constexpr ((INPUT_CHANNELS == 40 && OUTPUT_CHANNELS == 48)
                         || (INPUT_CHANNELS == 48 && OUTPUT_CHANNELS == 48)
                         || (INPUT_CHANNELS == 48 && OUTPUT_CHANNELS == 144)
                         || (INPUT_CHANNELS == 72 && OUTPUT_CHANNELS == 72)
                         || (INPUT_CHANNELS == 72 && OUTPUT_CHANNELS == 136)
                         || (INPUT_CHANNELS == 136 && OUTPUT_CHANNELS == 72)) {
        return 3;
    } else if constexpr (INPUT_CHANNELS == 100 && OUTPUT_CHANNELS == 100) {
        return 4;
    } else if constexpr (INPUT_CHANNELS == 100 && OUTPUT_CHANNELS == 5120) {
        return 5;
    } else {
        return 2;
    }
}

#define FOR_EACH_BLOCK(OP) \
    OP(0)                  \
    OP(1)                  \
    OP(2)                  \
    OP(3)                  \
    OP(4)

#define DISPATCH_PAIR(CIN, COUT)                                                               \
    if (args.input_channels == CIN && args.output_channels == COUT) {                          \
        dense_conv1x1_specialized<CIN, COUT, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE,         \
                                  HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, \
                                  HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(           \
            args.feature, args.packed_weight, args.bias, args.residual, args.channel_scale,    \
            args.output, args.pixels, args.height, args.width, args.output_scale);             \
        return true;                                                                           \
    }

#define LOAD_BIAS(BLOCK)                                                               \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                             \
        const int64_t output0 = (block0 + BLOCK) * 16;                                 \
        const __m256i bits = load_bias_bits<OUTPUT_CHANNELS, HAS_BIAS>(bias, output0); \
        acc##BLOCK = _mm512_cvtpbh_ps((__m256bh)bits);                                 \
    }

#define LOAD_BIAS_PIXEL_TILE(BLOCK)                                                           \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                    \
        const int64_t output_channel = (block0 + BLOCK) * 16;                                 \
        const __m256i bits = load_bias_bits<OUTPUT_CHANNELS, HAS_BIAS>(bias, output_channel); \
        const __m512 bias_vector = _mm512_cvtpbh_ps((__m256bh)bits);                          \
        acc0##BLOCK = bias_vector;                                                            \
        acc1##BLOCK = bias_vector;                                                            \
        acc2##BLOCK = bias_vector;                                                            \
        acc3##BLOCK = bias_vector;                                                            \
        acc4##BLOCK = bias_vector;                                                            \
    }

#define STORE_OUTPUT(BLOCK)                                                                                    \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                                     \
        const int64_t output0 = (block0 + BLOCK) * 16;                                                         \
        const __m256i bits =                                                                                   \
            finalize_output_bits<OUTPUT_CHANNELS, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, \
                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(                 \
                acc##BLOCK, bias, residual, channel_scale, output0, scale_vector);                             \
        store_dense_output<OUTPUT_CHANNELS, UPSCALE_FACTOR, OUTPUT_CHUNKS>(                                    \
            output, output_pixel, output_width, output_plane_pixels, output0, bits);                           \
    }

#define STORE_OUTPUT_PIXEL_TILE(BLOCK)                                                                         \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                                     \
        const int64_t output_channel = (block0 + BLOCK) * 16;                                                  \
        const __m256i bits0 =                                                                                  \
            finalize_output_bits<OUTPUT_CHANNELS, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, \
                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(                 \
                acc0##BLOCK, bias, residual0, channel_scale, output_channel, scale_vector);                    \
        const __m256i bits1 =                                                                                  \
            finalize_output_bits<OUTPUT_CHANNELS, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, \
                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(                 \
                acc1##BLOCK, bias, residual1, channel_scale, output_channel, scale_vector);                    \
        const __m256i bits2 =                                                                                  \
            finalize_output_bits<OUTPUT_CHANNELS, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, \
                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(                 \
                acc2##BLOCK, bias, residual2, channel_scale, output_channel, scale_vector);                    \
        const __m256i bits3 =                                                                                  \
            finalize_output_bits<OUTPUT_CHANNELS, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, \
                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(                 \
                acc3##BLOCK, bias, residual3, channel_scale, output_channel, scale_vector);                    \
        const __m256i bits4 =                                                                                  \
            finalize_output_bits<OUTPUT_CHANNELS, HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, \
                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(                 \
                acc4##BLOCK, bias, residual4, channel_scale, output_channel, scale_vector);                    \
        store_dense_output<OUTPUT_CHANNELS, UPSCALE_FACTOR, OUTPUT_CHUNKS>(                                    \
            output, output_pixel0, output_width, output_plane_pixels, output_channel, bits0);                  \
        store_dense_output<OUTPUT_CHANNELS, UPSCALE_FACTOR, OUTPUT_CHUNKS>(                                    \
            output, output_pixel1, output_width, output_plane_pixels, output_channel, bits1);                  \
        store_dense_output<OUTPUT_CHANNELS, UPSCALE_FACTOR, OUTPUT_CHUNKS>(                                    \
            output, output_pixel2, output_width, output_plane_pixels, output_channel, bits2);                  \
        store_dense_output<OUTPUT_CHANNELS, UPSCALE_FACTOR, OUTPUT_CHUNKS>(                                    \
            output, output_pixel3, output_width, output_plane_pixels, output_channel, bits3);                  \
        store_dense_output<OUTPUT_CHANNELS, UPSCALE_FACTOR, OUTPUT_CHUNKS>(                                    \
            output, output_pixel4, output_width, output_plane_pixels, output_channel, bits4);                  \
    }

#define STORE_PHASE_OUTPUT(BLOCK)                                                                    \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                           \
        constexpr int64_t shuffled_channels = OUTPUT_CHANNELS / (UPSCALE_FACTOR * UPSCALE_FACTOR);   \
        const int64_t output_channel = (block0 - phase_block0 + BLOCK) * 16;                         \
        const int64_t bias_channel = (block0 + BLOCK) * 16;                                          \
        const __m512 scaled = finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, \
                                              HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(    \
            acc##BLOCK, bias, bias_channel, scale_vector);                                           \
        store_output_bits<shuffled_channels>(output + output_pixel * shuffled_channels,              \
                                             output_channel, (__m256i)_mm512_cvtneps_pbh(scaled));   \
    }

#define STORE_PHASE_OUTPUT_PIXEL_TILE(BLOCK)                                                          \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                            \
        constexpr int64_t shuffled_channels = OUTPUT_CHANNELS / (UPSCALE_FACTOR * UPSCALE_FACTOR);    \
        const int64_t output_channel = (block0 - phase_block0 + BLOCK) * 16;                          \
        const int64_t bias_channel = (block0 + BLOCK) * 16;                                           \
        const __m512 scaled0 = finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, \
                                               HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(    \
            acc0##BLOCK, bias, bias_channel, scale_vector);                                           \
        const __m512 scaled1 = finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, \
                                               HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(    \
            acc1##BLOCK, bias, bias_channel, scale_vector);                                           \
        const __m512 scaled2 = finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, \
                                               HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(    \
            acc2##BLOCK, bias, bias_channel, scale_vector);                                           \
        const __m512 scaled3 = finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, \
                                               HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(    \
            acc3##BLOCK, bias, bias_channel, scale_vector);                                           \
        const __m512 scaled4 = finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, \
                                               HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER>(    \
            acc4##BLOCK, bias, bias_channel, scale_vector);                                           \
        store_output_bits<shuffled_channels>(output + output_pixel0 * shuffled_channels,              \
                                             output_channel, (__m256i)_mm512_cvtneps_pbh(scaled0));   \
        store_output_bits<shuffled_channels>(output + output_pixel1 * shuffled_channels,              \
                                             output_channel, (__m256i)_mm512_cvtneps_pbh(scaled1));   \
        store_output_bits<shuffled_channels>(output + output_pixel2 * shuffled_channels,              \
                                             output_channel, (__m256i)_mm512_cvtneps_pbh(scaled2));   \
        store_output_bits<shuffled_channels>(output + output_pixel3 * shuffled_channels,              \
                                             output_channel, (__m256i)_mm512_cvtneps_pbh(scaled3));   \
        store_output_bits<shuffled_channels>(output + output_pixel4 * shuffled_channels,              \
                                             output_channel, (__m256i)_mm512_cvtneps_pbh(scaled4));   \
    }

#define UPDATE_BLOCK(BLOCK)                                                                \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                 \
        acc##BLOCK = _mm512_dpbf16_ps(acc##BLOCK, input_vector,                            \
                                      (__m512bh)_mm512_loadu_si512(weights + BLOCK * 32)); \
    }

#define UPDATE_BLOCK_PIXEL_TILE(BLOCK)                                                     \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                 \
        const __m512bh weight_vector = (__m512bh)_mm512_loadu_si512(weights + BLOCK * 32); \
        acc0##BLOCK = _mm512_dpbf16_ps(acc0##BLOCK, input_vector0, weight_vector);         \
        acc1##BLOCK = _mm512_dpbf16_ps(acc1##BLOCK, input_vector1, weight_vector);         \
        acc2##BLOCK = _mm512_dpbf16_ps(acc2##BLOCK, input_vector2, weight_vector);         \
        acc3##BLOCK = _mm512_dpbf16_ps(acc3##BLOCK, input_vector3, weight_vector);         \
        acc4##BLOCK = _mm512_dpbf16_ps(acc4##BLOCK, input_vector4, weight_vector);         \
    }

template <int OUTPUT_CHANNELS>
static inline __mmask16 output_mask(int64_t output0)
{
    const int64_t remaining = OUTPUT_CHANNELS - output0;
    return remaining >= 16 ? static_cast<__mmask16>(0xffff)
                           : static_cast<__mmask16>((1u << remaining) - 1u);
}

template <int OUTPUT_CHANNELS, bool HAS_BIAS>
static inline __m256i load_bias_bits(const at::BFloat16* bias, int64_t output0)
{
    if constexpr (!HAS_BIAS) {
        return _mm256_setzero_si256();
    }
    if constexpr (OUTPUT_CHANNELS % 16 == 0) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + output0));
    }
    return _mm256_maskz_loadu_epi16(output_mask<OUTPUT_CHANNELS>(output0), bias + output0);
}

template <int OUTPUT_CHANNELS>
static inline void store_output_bits(at::BFloat16* output, int64_t output0, __m256i bits)
{
    if constexpr (OUTPUT_CHANNELS % 16 == 0) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + output0), bits);
    } else {
        _mm256_mask_storeu_epi16(output + output0, output_mask<OUTPUT_CHANNELS>(output0), bits);
    }
}

template <int OUTPUT_CHANNELS, int UPSCALE_FACTOR>
static inline void store_shuffled_output(at::BFloat16* output, int64_t output_pixel,
                                         int64_t output_width, int64_t packed_output0, __m256i bits)
{
    static_assert(UPSCALE_FACTOR == 1 || UPSCALE_FACTOR == 2 || UPSCALE_FACTOR == 4);
    if constexpr (UPSCALE_FACTOR == 1) {
        store_output_bits<OUTPUT_CHANNELS>(output + output_pixel * OUTPUT_CHANNELS, packed_output0,
                                           bits);
        return;
    }

    constexpr int64_t upscale_area = UPSCALE_FACTOR * UPSCALE_FACTOR;
    constexpr int64_t shuffled_channels = OUTPUT_CHANNELS / upscale_area;
    const int64_t valid_lanes = std::min<int64_t>(16, OUTPUT_CHANNELS - packed_output0);
    const int64_t first_channel = packed_output0 % shuffled_channels;
    if (first_channel + valid_lanes <= shuffled_channels) {
        const int64_t offset = packed_output0 / shuffled_channels;
        at::BFloat16* destination =
            output
            + (output_pixel + (offset / UPSCALE_FACTOR) * output_width + offset % UPSCALE_FACTOR)
                  * shuffled_channels
            + first_channel;
        if (valid_lanes == 16) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), bits);
        } else {
            _mm256_mask_storeu_epi16(destination, static_cast<__mmask16>((1u << valid_lanes) - 1u),
                                     bits);
        }
        return;
    }

    alignas(32) at::BFloat16 values[16];
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), bits);
    int64_t lane = 0;
    int64_t packed_output = packed_output0;
    while (lane < valid_lanes) {
        const int64_t offset = packed_output / shuffled_channels;
        const int64_t channel = packed_output % shuffled_channels;
        const int64_t count = std::min(valid_lanes - lane, shuffled_channels - channel);
        at::BFloat16* destination =
            output
            + (output_pixel + (offset / UPSCALE_FACTOR) * output_width + offset % UPSCALE_FACTOR)
                  * shuffled_channels
            + channel;
        std::memcpy(destination, values + lane, count * sizeof(at::BFloat16));
        lane += count;
        packed_output += count;
    }
}

template <int OUTPUT_CHANNELS, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static inline void store_dense_output(at::BFloat16* output, int64_t output_pixel,
                                      int64_t output_width, int64_t output_plane_pixels,
                                      int64_t packed_output0, __m256i bits)
{
    static_assert(OUTPUT_CHUNKS == 1 || OUTPUT_CHUNKS == 2 || OUTPUT_CHUNKS == 3);
    if constexpr (OUTPUT_CHUNKS == 1) {
        store_shuffled_output<OUTPUT_CHANNELS, UPSCALE_FACTOR>(output, output_pixel, output_width,
                                                               packed_output0, bits);
    } else {
        static_assert(UPSCALE_FACTOR == 1);
        constexpr int64_t part_channels = OUTPUT_CHANNELS / OUTPUT_CHUNKS;
        static_assert(part_channels % 16 == 0);
        const int64_t part = packed_output0 / part_channels;
        const int64_t output_channel = packed_output0 % part_channels;
        at::BFloat16* destination = output + (part * output_plane_pixels + output_pixel) * part_channels;
        store_output_bits<part_channels>(destination, output_channel, bits);
    }
}

template <int OUTPUT_CHANNELS, int UPSCALE_FACTOR>
static inline int64_t shuffled_output_pixel(int64_t input_pixel, int64_t height, int64_t width)
{
    if constexpr (UPSCALE_FACTOR == 1) {
        return input_pixel;
    }
    const int64_t batch_index = input_pixel / (height * width);
    const int64_t spatial_pixel = input_pixel % (height * width);
    const int64_t y = spatial_pixel / width;
    const int64_t x = spatial_pixel % width;
    return batch_index * height * UPSCALE_FACTOR * width * UPSCALE_FACTOR
           + y * UPSCALE_FACTOR * width * UPSCALE_FACTOR + x * UPSCALE_FACTOR;
}

static inline void store_rgb4x4(at::BFloat16* output, int64_t output_pixel, int64_t output_width,
                                __m256i bits0, __m256i bits1, __m256i bits2)
{
    alignas(32) at::BFloat16 values[48];
    _mm256_store_si256(reinterpret_cast<__m256i*>(values), bits0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(values + 16), bits1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(values + 32), bits2);
    for (int64_t row = 0; row < 4; ++row) {
        std::memcpy(output + (output_pixel + row * output_width) * 3, values + row * 12,
                    12 * sizeof(at::BFloat16));
    }
}

template <bool HAS_SCALAR_SCALE>
static inline __m512 apply_output_scale(__m512 value, __m512 scale)
{
    if constexpr (HAS_SCALAR_SCALE) {
        return _mm512_mul_ps(value, scale);
    }
    return value;
}

static inline __m512 sigmoid_approx(__m512 x)
{
    return _mm512_div_ps(
        _mm512_set1_ps(1.0f),
        _mm512_add_ps(_mm512_set1_ps(1.0f), exp_approx(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

static inline __m512 silu(__m512 x)
{
    return _mm512_mul_ps(x, sigmoid_approx(x));
}

static constexpr float LOG_SCALE_MIN = -2.20727491f;
static constexpr float LOG_SCALE_STEP = 0.12305480f;

static inline __m512 scale_decoder(__m512 x, __m512 qp_bias)
{
    x = _mm512_add_ps(x, qp_bias);
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_setzero_ps()), _mm512_set1_ps(63.0f));
    return exp_approx_unclamped(
        _mm512_fmadd_ps(x, _mm512_set1_ps(LOG_SCALE_STEP), _mm512_set1_ps(LOG_SCALE_MIN)));
}

template <int OUTPUT_CHANNELS, bool HAS_BIAS, bool HAS_SCALAR_SCALE, bool HAS_SILU,
          bool HAS_HARD_GELU_030, bool HAS_SIGMOID, bool HAS_SCALE_DECODER>
static inline __m512 finalize_output(__m512 accumulator, const at::BFloat16* bias, int64_t output0,
                                     __m512 output_scale)
{
    static_assert(!HAS_SCALE_DECODER
                  || (HAS_BIAS && !HAS_SCALAR_SCALE && !HAS_SILU && !HAS_HARD_GELU_030 && !HAS_SIGMOID));
    __m512 result = accumulator;
    result = apply_output_scale<HAS_SCALAR_SCALE>(result, output_scale);
    if constexpr (HAS_SILU) {
        result = silu(result);
    }
    if constexpr (HAS_HARD_GELU_030) {
        result = hard_gelu_030(result);
    }
    if constexpr (HAS_SIGMOID) {
        result = sigmoid_approx(result);
    }
    if constexpr (HAS_SCALE_DECODER) {
        result = scale_decoder(result, output_scale);
    }
    return result;
}

template <int OUTPUT_CHANNELS, bool HAS_BIAS, bool HAS_RESIDUAL, bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE,
          bool HAS_SILU, bool HAS_HARD_GELU_030, bool HAS_SIGMOID, bool HAS_SCALE_DECODER>
static inline __m256i
finalize_output_bits(__m512 accumulator, const at::BFloat16* bias, const at::BFloat16* residual,
                     const at::BFloat16* channel_scale, int64_t output0, __m512 output_scale)
{
    static_assert(!(HAS_SCALAR_SCALE && HAS_CHANNEL_SCALE));
    if constexpr (!HAS_RESIDUAL) {
        static_assert(!HAS_CHANNEL_SCALE);
        const __m512 result =
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                            HAS_SIGMOID, HAS_SCALE_DECODER>(accumulator, bias, output0, output_scale);
        return (__m256i)_mm512_cvtneps_pbh(result);
    } else {
        static_assert(!HAS_SILU && !HAS_HARD_GELU_030 && !HAS_SIGMOID && !HAS_SCALE_DECODER);
        const __m256i residual_bits = load_bias_bits<OUTPUT_CHANNELS, true>(residual, output0);
        const __m512 residual_value = _mm512_cvtpbh_ps((__m256bh)residual_bits);
        __m512 result = _mm512_add_ps(accumulator, residual_value);
        if constexpr (HAS_SCALAR_SCALE || HAS_CHANNEL_SCALE) {
            __m512 scale = output_scale;
            if constexpr (HAS_CHANNEL_SCALE) {
                const __m256i scale_bits =
                    load_bias_bits<OUTPUT_CHANNELS, true>(channel_scale, output0);
                scale = _mm512_cvtpbh_ps((__m256bh)scale_bits);
            }
            result = _mm512_mul_ps(result, scale);
        }
        return (__m256i)_mm512_cvtneps_pbh(result);
    }
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS, bool HAS_RESIDUAL,
          bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, bool HAS_SILU, bool HAS_HARD_GELU_030,
          bool HAS_SIGMOID, bool HAS_SCALE_DECODER, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static inline void
dense_conv1x1_pixel_tile(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                         const at::BFloat16* bias, const at::BFloat16* residual,
                         const at::BFloat16* channel_scale, at::BFloat16* output,
                         int64_t output_pixel, int64_t output_width, int64_t output_plane_pixels,
                         int64_t block0, int64_t phase_block0, float output_scale)
{
    constexpr int INPUT_PAIRS = INPUT_CHANNELS / 2;
    constexpr int OUTPUT_BLOCKS = (OUTPUT_CHANNELS + 15) / 16;
    constexpr bool PHASE_MAJOR =
        UPSCALE_FACTOR > 1 && (OUTPUT_CHANNELS / (UPSCALE_FACTOR * UPSCALE_FACTOR)) % 16 == 0;

    // Allocate one FP32 accumulator register per output block in the tile.
    __m512 acc0;
    __m512 acc1;
    __m512 acc2;
    __m512 acc3;
    __m512 acc4;

    // Initialize active accumulators from the BF16 bias values.
    FOR_EACH_BLOCK(LOAD_BIAS)

    const auto* input_pairs = reinterpret_cast<const AliasedUint32*>(feature);

    // Accumulate four packed BF16 input pairs per iteration across all active blocks.
    int pair = 0;
    for (; pair + 3 < INPUT_PAIRS; pair += 4) {
        if (pair + WEIGHT_PREFETCH_DISTANCE < INPUT_PAIRS) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             packed_weight
                             + ((pair + WEIGHT_PREFETCH_DISTANCE) * OUTPUT_BLOCKS + block0) * 32),
                         _MM_HINT_T0);
        }
        for (int64_t unroll = 0; unroll < 4; ++unroll) {
            const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(input_pairs[pair + unroll]);
            const at::BFloat16* weights =
                packed_weight + ((pair + unroll) * OUTPUT_BLOCKS + block0) * 32;
            FOR_EACH_BLOCK(UPDATE_BLOCK)
        }
    }

    // Accumulate any input pairs left after the four-way unrolled loop.
    for (; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(input_pairs[pair]);
        const at::BFloat16* weights = packed_weight + (pair * OUTPUT_BLOCKS + block0) * 32;
        FOR_EACH_BLOCK(UPDATE_BLOCK)
    }

    const __m512 scale_vector = _mm512_set1_ps(output_scale);
    if constexpr (PHASE_MAJOR) {
        FOR_EACH_BLOCK(STORE_PHASE_OUTPUT)
    } else if constexpr (OUTPUT_CHANNELS == 48 && UPSCALE_FACTOR == 4) {
        const __m512 scaled0 =
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                            HAS_SIGMOID, HAS_SCALE_DECODER>(acc0, bias, 0, scale_vector);
        const __m512 scaled1 =
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                            HAS_SIGMOID, HAS_SCALE_DECODER>(acc1, bias, 16, scale_vector);
        const __m512 scaled2 =
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                            HAS_SIGMOID, HAS_SCALE_DECODER>(acc2, bias, 32, scale_vector);
        store_rgb4x4(output, output_pixel, output_width, (__m256i)_mm512_cvtneps_pbh(scaled0),
                     (__m256i)_mm512_cvtneps_pbh(scaled1), (__m256i)_mm512_cvtneps_pbh(scaled2));
    } else {
        FOR_EACH_BLOCK(STORE_OUTPUT)
    }
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS, bool HAS_RESIDUAL,
          bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, bool HAS_SILU, bool HAS_HARD_GELU_030,
          bool HAS_SIGMOID, bool HAS_SCALE_DECODER, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static inline void dense_conv1x1_multi_pixel_tile(
    const at::BFloat16* feature0, const at::BFloat16* feature1, const at::BFloat16* feature2,
    const at::BFloat16* feature3, const at::BFloat16* feature4, const at::BFloat16* packed_weight,
    const at::BFloat16* bias, const at::BFloat16* residual0, const at::BFloat16* residual1,
    const at::BFloat16* residual2, const at::BFloat16* residual3, const at::BFloat16* residual4,
    const at::BFloat16* channel_scale, at::BFloat16* output, int64_t output_pixel0,
    int64_t output_pixel1, int64_t output_pixel2, int64_t output_pixel3, int64_t output_pixel4,
    int64_t output_width, int64_t output_plane_pixels, int64_t block0, int64_t phase_block0,
    float output_scale)
{
    constexpr int INPUT_PAIRS = INPUT_CHANNELS / 2;
    constexpr int OUTPUT_BLOCKS = (OUTPUT_CHANNELS + 15) / 16;
    constexpr bool PHASE_MAJOR =
        UPSCALE_FACTOR > 1 && (OUTPUT_CHANNELS / (UPSCALE_FACTOR * UPSCALE_FACTOR)) % 16 == 0;

    // Keep the pixel-by-output-block tile in named FP32 accumulator registers.
    __m512 acc00;
    __m512 acc01;
    __m512 acc02;
    __m512 acc03;
    __m512 acc04;
    __m512 acc10;
    __m512 acc11;
    __m512 acc12;
    __m512 acc13;
    __m512 acc14;
    __m512 acc20;
    __m512 acc21;
    __m512 acc22;
    __m512 acc23;
    __m512 acc24;
    __m512 acc30;
    __m512 acc31;
    __m512 acc32;
    __m512 acc33;
    __m512 acc34;
    __m512 acc40;
    __m512 acc41;
    __m512 acc42;
    __m512 acc43;
    __m512 acc44;

    FOR_EACH_BLOCK(LOAD_BIAS_PIXEL_TILE)

    const auto* input_pairs0 = reinterpret_cast<const AliasedUint32*>(feature0);
    const auto* input_pairs1 = reinterpret_cast<const AliasedUint32*>(feature1);
    const auto* input_pairs2 = reinterpret_cast<const AliasedUint32*>(feature2);
    const auto* input_pairs3 = reinterpret_cast<const AliasedUint32*>(feature3);
    const auto* input_pairs4 = reinterpret_cast<const AliasedUint32*>(feature4);

    // Reuse each loaded weight vector across all pixels in the tile.
    int pair = 0;
    for (; pair + 3 < INPUT_PAIRS; pair += 4) {
        if (pair + WEIGHT_PREFETCH_DISTANCE < INPUT_PAIRS) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             packed_weight
                             + ((pair + WEIGHT_PREFETCH_DISTANCE) * OUTPUT_BLOCKS + block0) * 32),
                         _MM_HINT_T0);
        }
        for (int64_t unroll = 0; unroll < 4; ++unroll) {
            const __m512bh input_vector0 = (__m512bh)_mm512_set1_epi32(input_pairs0[pair + unroll]);
            const __m512bh input_vector1 = (__m512bh)_mm512_set1_epi32(input_pairs1[pair + unroll]);
            const __m512bh input_vector2 = (__m512bh)_mm512_set1_epi32(input_pairs2[pair + unroll]);
            const __m512bh input_vector3 = (__m512bh)_mm512_set1_epi32(input_pairs3[pair + unroll]);
            const __m512bh input_vector4 = (__m512bh)_mm512_set1_epi32(input_pairs4[pair + unroll]);
            const at::BFloat16* weights =
                packed_weight + ((pair + unroll) * OUTPUT_BLOCKS + block0) * 32;
            FOR_EACH_BLOCK(UPDATE_BLOCK_PIXEL_TILE)
        }
    }
    for (; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector0 = (__m512bh)_mm512_set1_epi32(input_pairs0[pair]);
        const __m512bh input_vector1 = (__m512bh)_mm512_set1_epi32(input_pairs1[pair]);
        const __m512bh input_vector2 = (__m512bh)_mm512_set1_epi32(input_pairs2[pair]);
        const __m512bh input_vector3 = (__m512bh)_mm512_set1_epi32(input_pairs3[pair]);
        const __m512bh input_vector4 = (__m512bh)_mm512_set1_epi32(input_pairs4[pair]);
        const at::BFloat16* weights = packed_weight + (pair * OUTPUT_BLOCKS + block0) * 32;
        FOR_EACH_BLOCK(UPDATE_BLOCK_PIXEL_TILE)
    }

    const __m512 scale_vector = _mm512_set1_ps(output_scale);
    if constexpr (PHASE_MAJOR) {
        FOR_EACH_BLOCK(STORE_PHASE_OUTPUT_PIXEL_TILE)
    } else if constexpr (OUTPUT_CHANNELS == 48 && UPSCALE_FACTOR == 4) {
#define STORE_RGB_PIXEL(PIXEL)                                                                        \
    store_rgb4x4(                                                                                     \
        output, output_pixel##PIXEL, output_width,                                                    \
        (__m256i)_mm512_cvtneps_pbh(                                                                  \
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030, \
                            HAS_SIGMOID, HAS_SCALE_DECODER>(acc##PIXEL##0, bias, 0, scale_vector)),   \
        (__m256i)_mm512_cvtneps_pbh(                                                                  \
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030, \
                            HAS_SIGMOID, HAS_SCALE_DECODER>(acc##PIXEL##1, bias, 16, scale_vector)),  \
        (__m256i)_mm512_cvtneps_pbh(                                                                  \
            finalize_output<OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU, HAS_HARD_GELU_030, \
                            HAS_SIGMOID, HAS_SCALE_DECODER>(acc##PIXEL##2, bias, 32, scale_vector)));
        STORE_RGB_PIXEL(0)
        STORE_RGB_PIXEL(1)
        STORE_RGB_PIXEL(2)
        STORE_RGB_PIXEL(3)
        STORE_RGB_PIXEL(4)
#undef STORE_RGB_PIXEL
    } else {
        FOR_EACH_BLOCK(STORE_OUTPUT_PIXEL_TILE)
    }
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS, bool HAS_RESIDUAL,
          bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, bool HAS_SILU, bool HAS_HARD_GELU_030,
          bool HAS_SIGMOID, bool HAS_SCALE_DECODER, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static inline void
dense_conv1x1_output_tile(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                          const at::BFloat16* bias, const at::BFloat16* residual,
                          const at::BFloat16* channel_scale, at::BFloat16* output, int64_t pixels,
                          int64_t height, int64_t width, int64_t block0, float output_scale)
{
    const int64_t pixel_tiles = pixels / PIXEL_TILE;
    const int64_t output_width = width * UPSCALE_FACTOR;
    const int64_t output_plane_pixels = pixels * UPSCALE_FACTOR * UPSCALE_FACTOR;

    for (int64_t pixel_tile = 0; pixel_tile < pixel_tiles; ++pixel_tile) {
        const int64_t pixel0 = pixel_tile * PIXEL_TILE;
        const int64_t pixel1 = pixel0 + 1;
        const int64_t pixel2 = pixel0 + 2;
        const int64_t pixel3 = pixel0 + 3;
        const int64_t pixel4 = pixel0 + 4;
        const at::BFloat16* input0 = feature + pixel0 * INPUT_CHANNELS;
        const at::BFloat16* input1 = feature + pixel1 * INPUT_CHANNELS;
        const at::BFloat16* input2 = feature + pixel2 * INPUT_CHANNELS;
        const at::BFloat16* input3 = feature + pixel3 * INPUT_CHANNELS;
        const at::BFloat16* input4 = feature + pixel4 * INPUT_CHANNELS;
        const at::BFloat16* residual0 = HAS_RESIDUAL ? residual + pixel0 * OUTPUT_CHANNELS : nullptr;
        const at::BFloat16* residual1 = HAS_RESIDUAL ? residual + pixel1 * OUTPUT_CHANNELS : nullptr;
        const at::BFloat16* residual2 = HAS_RESIDUAL ? residual + pixel2 * OUTPUT_CHANNELS : nullptr;
        const at::BFloat16* residual3 = HAS_RESIDUAL ? residual + pixel3 * OUTPUT_CHANNELS : nullptr;
        const at::BFloat16* residual4 = HAS_RESIDUAL ? residual + pixel4 * OUTPUT_CHANNELS : nullptr;
        const int64_t output_pixel0 =
            shuffled_output_pixel<OUTPUT_CHANNELS, UPSCALE_FACTOR>(pixel0, height, width);
        const int64_t output_pixel1 =
            shuffled_output_pixel<OUTPUT_CHANNELS, UPSCALE_FACTOR>(pixel1, height, width);
        const int64_t output_pixel2 =
            shuffled_output_pixel<OUTPUT_CHANNELS, UPSCALE_FACTOR>(pixel2, height, width);
        const int64_t output_pixel3 =
            shuffled_output_pixel<OUTPUT_CHANNELS, UPSCALE_FACTOR>(pixel3, height, width);
        const int64_t output_pixel4 =
            shuffled_output_pixel<OUTPUT_CHANNELS, UPSCALE_FACTOR>(pixel4, height, width);

        dense_conv1x1_multi_pixel_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, ACTIVE_BLOCKS, HAS_BIAS, HAS_RESIDUAL,
                                       HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                                       HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
            input0, input1, input2, input3, input4, packed_weight, bias, residual0, residual1,
            residual2, residual3, residual4, channel_scale, output, output_pixel0, output_pixel1,
            output_pixel2, output_pixel3, output_pixel4, output_width, output_plane_pixels, block0,
            0, output_scale);
    }

    for (int64_t pixel = pixel_tiles * PIXEL_TILE; pixel < pixels; ++pixel) {
        const at::BFloat16* input = feature + pixel * INPUT_CHANNELS;
        const at::BFloat16* pixel_residual = HAS_RESIDUAL ? residual + pixel * OUTPUT_CHANNELS : nullptr;
        const int64_t output_pixel =
            shuffled_output_pixel<OUTPUT_CHANNELS, UPSCALE_FACTOR>(pixel, height, width);
        dense_conv1x1_pixel_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, ACTIVE_BLOCKS, HAS_BIAS, HAS_RESIDUAL,
                                 HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                                 HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
            input, packed_weight, bias, pixel_residual, channel_scale, output, output_pixel,
            output_width, output_plane_pixels, block0, 0, output_scale);
    }
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, bool HAS_BIAS, bool HAS_SCALAR_SCALE, bool HAS_SILU,
          bool HAS_HARD_GELU_030, bool HAS_SIGMOID, bool HAS_SCALE_DECODER, int UPSCALE_FACTOR>
static void dense_conv1x1_phase_major(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                                      const at::BFloat16* bias, at::BFloat16* output, int64_t pixels,
                                      int64_t height, int64_t width, float output_scale)
{
    constexpr int OUTPUT_BLOCK_TILE =
        dense_output_block_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, HAS_SILU, HAS_HARD_GELU_030,
                                HAS_SIGMOID, UPSCALE_FACTOR, 1>();
    constexpr int SHUFFLED_CHANNELS = OUTPUT_CHANNELS / (UPSCALE_FACTOR * UPSCALE_FACTOR);
    constexpr int PHASE_BLOCKS = SHUFFLED_CHANNELS / 16;
    constexpr int FULL_TILES = PHASE_BLOCKS / OUTPUT_BLOCK_TILE;
    constexpr int TAIL_BLOCKS = PHASE_BLOCKS % OUTPUT_BLOCK_TILE;
    constexpr int OUTPUT_TILES = FULL_TILES + (TAIL_BLOCKS != 0);
    static_assert(SHUFFLED_CHANNELS % 16 == 0);

    const int64_t batch = pixels / (height * width);
    const int64_t output_width = width * UPSCALE_FACTOR;
    const int64_t output_plane_pixels = pixels * UPSCALE_FACTOR * UPSCALE_FACTOR;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t y = 0; y < height; ++y) {
            const at::BFloat16* input_row =
                feature + (batch_index * height + y) * width * INPUT_CHANNELS;
            for (int64_t phase_y = 0; phase_y < UPSCALE_FACTOR; ++phase_y) {
                const int64_t output_row_pixel =
                    (batch_index * height * UPSCALE_FACTOR + y * UPSCALE_FACTOR + phase_y) * output_width;
                for (int64_t phase_x = 0; phase_x < UPSCALE_FACTOR; ++phase_x) {
                    const int64_t phase = phase_y * UPSCALE_FACTOR + phase_x;
                    const int64_t phase_block0 = phase * PHASE_BLOCKS;
                    for (int64_t tile = 0; tile < OUTPUT_TILES; ++tile) {
                        const int64_t block0 = phase_block0 + tile * OUTPUT_BLOCK_TILE;
                        int64_t x = 0;
                        for (; x + PIXEL_TILE <= width; x += PIXEL_TILE) {
                            const at::BFloat16* input0 = input_row + x * INPUT_CHANNELS;
                            const at::BFloat16* input1 = input0 + INPUT_CHANNELS;
                            const at::BFloat16* input2 = input1 + INPUT_CHANNELS;
                            const at::BFloat16* input3 = input2 + INPUT_CHANNELS;
                            const at::BFloat16* input4 = input3 + INPUT_CHANNELS;
                            const int64_t output_pixel0 =
                                output_row_pixel + x * UPSCALE_FACTOR + phase_x;
                            const int64_t output_pixel1 = output_pixel0 + UPSCALE_FACTOR;
                            const int64_t output_pixel2 = output_pixel1 + UPSCALE_FACTOR;
                            const int64_t output_pixel3 = output_pixel2 + UPSCALE_FACTOR;
                            const int64_t output_pixel4 = output_pixel3 + UPSCALE_FACTOR;
                            if (tile < FULL_TILES) {
                                dense_conv1x1_multi_pixel_tile<
                                    INPUT_CHANNELS, OUTPUT_CHANNELS, OUTPUT_BLOCK_TILE, HAS_BIAS,
                                    false, HAS_SCALAR_SCALE, false, HAS_SILU, HAS_HARD_GELU_030,
                                    HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, 1>(
                                    input0, input1, input2, input3, input4, packed_weight, bias,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output,
                                    output_pixel0, output_pixel1, output_pixel2, output_pixel3,
                                    output_pixel4, output_width, output_plane_pixels, block0,
                                    phase_block0, output_scale);
                            } else if constexpr (TAIL_BLOCKS != 0) {
                                dense_conv1x1_multi_pixel_tile<
                                    INPUT_CHANNELS, OUTPUT_CHANNELS, TAIL_BLOCKS, HAS_BIAS, false,
                                    HAS_SCALAR_SCALE, false, HAS_SILU, HAS_HARD_GELU_030,
                                    HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, 1>(
                                    input0, input1, input2, input3, input4, packed_weight, bias,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, output,
                                    output_pixel0, output_pixel1, output_pixel2, output_pixel3,
                                    output_pixel4, output_width, output_plane_pixels, block0,
                                    phase_block0, output_scale);
                            }
                        }
                        for (; x < width; ++x) {
                            const at::BFloat16* input = input_row + x * INPUT_CHANNELS;
                            const int64_t output_pixel = output_row_pixel + x * UPSCALE_FACTOR + phase_x;
                            if (tile < FULL_TILES) {
                                dense_conv1x1_pixel_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, OUTPUT_BLOCK_TILE,
                                                         HAS_BIAS, false, HAS_SCALAR_SCALE, false,
                                                         HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID,
                                                         HAS_SCALE_DECODER, UPSCALE_FACTOR, 1>(
                                    input, packed_weight, bias, nullptr, nullptr, output,
                                    output_pixel, output_width, output_plane_pixels, block0,
                                    phase_block0, output_scale);
                            } else if constexpr (TAIL_BLOCKS != 0) {
                                dense_conv1x1_pixel_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, TAIL_BLOCKS,
                                                         HAS_BIAS, false, HAS_SCALAR_SCALE, false,
                                                         HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID,
                                                         HAS_SCALE_DECODER, UPSCALE_FACTOR, 1>(
                                    input, packed_weight, bias, nullptr, nullptr, output,
                                    output_pixel, output_width, output_plane_pixels, block0,
                                    phase_block0, output_scale);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, bool HAS_BIAS, bool HAS_RESIDUAL,
          bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, bool HAS_SILU, bool HAS_HARD_GELU_030,
          bool HAS_SIGMOID, bool HAS_SCALE_DECODER, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static void dense_conv1x1_specialized(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                                      const at::BFloat16* bias, const at::BFloat16* residual,
                                      const at::BFloat16* channel_scale, at::BFloat16* output,
                                      int64_t pixels, int64_t height, int64_t width, float output_scale)
{
    constexpr int OUTPUT_BLOCKS = (OUTPUT_CHANNELS + 15) / 16;
    constexpr int TILE_OUTPUT_BLOCKS =
        dense_output_block_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, HAS_SILU, HAS_HARD_GELU_030,
                                HAS_SIGMOID, UPSCALE_FACTOR, OUTPUT_CHUNKS>();
    constexpr int FULL_TILES = OUTPUT_BLOCKS / TILE_OUTPUT_BLOCKS;
    constexpr int TAIL_BLOCKS = OUTPUT_BLOCKS % TILE_OUTPUT_BLOCKS;
    constexpr int OUTPUT_TILES = FULL_TILES + (TAIL_BLOCKS != 0);
    static_assert(OUTPUT_CHUNKS == 1 || UPSCALE_FACTOR == 1);
    static_assert(!HAS_RESIDUAL || (UPSCALE_FACTOR == 1 && OUTPUT_CHUNKS == 1));
    static_assert(!HAS_CHANNEL_SCALE || HAS_RESIDUAL);

    if constexpr (UPSCALE_FACTOR > 1 && (OUTPUT_CHANNELS / (UPSCALE_FACTOR * UPSCALE_FACTOR)) % 16 == 0) {
        dense_conv1x1_phase_major<INPUT_CHANNELS, OUTPUT_CHANNELS, HAS_BIAS, HAS_SCALAR_SCALE, HAS_SILU,
                                  HAS_HARD_GELU_030, HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR>(
            feature, packed_weight, bias, output, pixels, height, width, output_scale);
        return;
    }

    if constexpr (UPSCALE_FACTOR == 1) {
#pragma omp parallel for schedule(static) num_threads(1)
        for (int64_t tile = 0; tile < OUTPUT_TILES; ++tile) {
            const int64_t block0 = tile * TILE_OUTPUT_BLOCKS;
            if (tile < FULL_TILES) {
                dense_conv1x1_output_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, TILE_OUTPUT_BLOCKS, HAS_BIAS,
                                          HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                                          HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
                    feature, packed_weight, bias, residual, channel_scale, output, pixels, height,
                    width, block0, output_scale);
            } else if constexpr (TAIL_BLOCKS != 0) {
                dense_conv1x1_output_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, TAIL_BLOCKS, HAS_BIAS, HAS_RESIDUAL,
                                          HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                                          HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
                    feature, packed_weight, bias, residual, channel_scale, output, pixels, height,
                    width, block0, output_scale);
            }
        }
        return;
    }

    const int64_t batch = pixels / (height * width);
    const int64_t output_width = width * UPSCALE_FACTOR;
    const int64_t output_plane_pixels = pixels * UPSCALE_FACTOR * UPSCALE_FACTOR;
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t y = 0; y < height; ++y) {
            const at::BFloat16* input_row =
                feature + (batch_index * height + y) * width * INPUT_CHANNELS;
            const int64_t output_row_pixel =
                (batch_index * height * UPSCALE_FACTOR + y * UPSCALE_FACTOR) * output_width;
            int64_t x = 0;
            for (; x + PIXEL_TILE <= width; x += PIXEL_TILE) {
                const at::BFloat16* input0 = input_row + x * INPUT_CHANNELS;
                const at::BFloat16* input1 = input0 + INPUT_CHANNELS;
                const at::BFloat16* input2 = input1 + INPUT_CHANNELS;
                const at::BFloat16* input3 = input2 + INPUT_CHANNELS;
                const at::BFloat16* input4 = input3 + INPUT_CHANNELS;
                const int64_t output_pixel0 = output_row_pixel + x * UPSCALE_FACTOR;
                const int64_t output_pixel1 = output_pixel0 + UPSCALE_FACTOR;
                const int64_t output_pixel2 = output_pixel1 + UPSCALE_FACTOR;
                const int64_t output_pixel3 = output_pixel2 + UPSCALE_FACTOR;
                const int64_t output_pixel4 = output_pixel3 + UPSCALE_FACTOR;
                for (int64_t tile = 0; tile < OUTPUT_TILES; ++tile) {
                    const int64_t block0 = tile * TILE_OUTPUT_BLOCKS;
                    if (tile < FULL_TILES) {
                        dense_conv1x1_multi_pixel_tile<
                            INPUT_CHANNELS, OUTPUT_CHANNELS, TILE_OUTPUT_BLOCKS, HAS_BIAS, HAS_RESIDUAL,
                            HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                            HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
                            input0, input1, input2, input3, input4, packed_weight, bias, nullptr,
                            nullptr, nullptr, nullptr, nullptr, channel_scale, output, output_pixel0,
                            output_pixel1, output_pixel2, output_pixel3, output_pixel4,
                            output_width, output_plane_pixels, block0, 0, output_scale);
                    } else if constexpr (TAIL_BLOCKS != 0) {
                        dense_conv1x1_multi_pixel_tile<
                            INPUT_CHANNELS, OUTPUT_CHANNELS, TAIL_BLOCKS, HAS_BIAS, HAS_RESIDUAL,
                            HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030,
                            HAS_SIGMOID, HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
                            input0, input1, input2, input3, input4, packed_weight, bias, nullptr,
                            nullptr, nullptr, nullptr, nullptr, channel_scale, output, output_pixel0,
                            output_pixel1, output_pixel2, output_pixel3, output_pixel4,
                            output_width, output_plane_pixels, block0, 0, output_scale);
                    }
                }
            }
            for (; x < width; ++x) {
                const at::BFloat16* input = input_row + x * INPUT_CHANNELS;
                const int64_t output_pixel = output_row_pixel + x * UPSCALE_FACTOR;
                for (int64_t tile = 0; tile < OUTPUT_TILES; ++tile) {
                    const int64_t block0 = tile * TILE_OUTPUT_BLOCKS;
                    if (tile < FULL_TILES) {
                        dense_conv1x1_pixel_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, TILE_OUTPUT_BLOCKS,
                                                 HAS_BIAS, HAS_RESIDUAL, HAS_SCALAR_SCALE,
                                                 HAS_CHANNEL_SCALE, HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID,
                                                 HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
                            input, packed_weight, bias, nullptr, channel_scale, output, output_pixel,
                            output_width, output_plane_pixels, block0, 0, output_scale);
                    } else if constexpr (TAIL_BLOCKS != 0) {
                        dense_conv1x1_pixel_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, TAIL_BLOCKS, HAS_BIAS,
                                                 HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE,
                                                 HAS_SILU, HAS_HARD_GELU_030, HAS_SIGMOID,
                                                 HAS_SCALE_DECODER, UPSCALE_FACTOR, OUTPUT_CHUNKS>(
                            input, packed_weight, bias, nullptr, channel_scale, output, output_pixel,
                            output_width, output_plane_pixels, block0, 0, output_scale);
                    }
                }
            }
        }
    }
}

#ifndef DENSE_STANDARD_PAIRS
    #define DENSE_STANDARD_PAIRS DENSE_PLAIN_PAIRS
#endif

#ifndef DENSE_UPSCALE2_PAIRS
    #define DENSE_UPSCALE2_PAIRS DENSE_UPSAMPLE2_REGISTRY
#endif

#ifndef DENSE_UPSCALE4_PAIRS
    #define DENSE_UPSCALE4_PAIRS DENSE_UPSAMPLE4_REGISTRY
#endif

#ifndef DENSE_CHUNKS2_PAIRS
    #define DENSE_CHUNKS2_PAIRS DENSE_CHUNKS2_REGISTRY
#endif

#ifndef DENSE_CHUNKS3_PAIRS
    #define DENSE_CHUNKS3_PAIRS DENSE_CHUNKS3_REGISTRY
#endif

template <bool HAS_BIAS, bool HAS_RESIDUAL, bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, bool HAS_SILU,
          bool HAS_HARD_GELU_030, bool HAS_SIGMOID, bool HAS_SCALE_DECODER, int UPSCALE_FACTOR, int OUTPUT_CHUNKS>
static bool dispatch_specialized(const DispatchArgs& args)
{
    if constexpr (OUTPUT_CHUNKS == 2) {
        DENSE_CHUNKS2_PAIRS
    } else if constexpr (OUTPUT_CHUNKS == 3) {
        DENSE_CHUNKS3_PAIRS
    } else if constexpr (UPSCALE_FACTOR == 1) {
        DENSE_STANDARD_PAIRS
    } else if constexpr (UPSCALE_FACTOR == 2) {
        DENSE_UPSCALE2_PAIRS
    } else {
        DENSE_UPSCALE4_PAIRS
    }
    return false;
}
