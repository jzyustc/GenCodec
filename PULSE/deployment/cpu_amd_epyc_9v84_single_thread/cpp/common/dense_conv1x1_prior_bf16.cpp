#include "model_config.h"
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dense_conv1x1_prior_bf16.h"

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_prior_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <cstdint>
#include <immintrin.h>

namespace {

constexpr int INPUT_CHANNELS = PULSE_ENTROPY_CHANNELS;
constexpr int PART_CHANNELS = PULSE_MAIN_CHANNELS;
constexpr int OUTPUT_CHANNELS = 2 * PART_CHANNELS;
constexpr int CHANNEL_BLOCK = 16;
constexpr int INPUT_PAIRS = INPUT_CHANNELS / 2;
constexpr int OUTPUT_BLOCKS = OUTPUT_CHANNELS / CHANNEL_BLOCK;
constexpr int PART_BLOCKS = PART_CHANNELS / CHANNEL_BLOCK;
constexpr int OUTPUT_BLOCK_TILE = 2;
constexpr int PIXEL_TILE = 5;

using AliasedUint32 = uint32_t __attribute__((may_alias));

inline __m512 load_bf16(const at::BFloat16* input)
{
    return _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input)));
}

inline void store_prior(__m512 accumulator, at::BFloat16* output)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), (__m256i)_mm512_cvtneps_pbh(accumulator));
}

inline void store_means_and_y_hat0(__m512 accumulator, const at::BFloat16* symbols0,
                                   const at::BFloat16* mask0, at::BFloat16* means0, at::BFloat16* y_hat0)
{
    const __m256i means_bits = (__m256i)_mm512_cvtneps_pbh(accumulator);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(means0), means_bits);
    // checkerboard_mask0 is constant within each aligned 16-channel block.
    if (static_cast<float>(*mask0) == 0.0f) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(y_hat0), _mm256_setzero_si256());
        return;
    }
    const __m512 means = _mm512_cvtpbh_ps((__m256bh)means_bits);
    const __m256i sum_bits = (__m256i)_mm512_cvtneps_pbh(_mm512_add_ps(load_bf16(symbols0), means));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y_hat0), sum_bits);
}

template <bool IS_MEANS>
inline void store_output(__m512 accumulator, const at::BFloat16* symbols0,
                         const at::BFloat16* mask0, at::BFloat16* part, at::BFloat16* y_hat0)
{
    if constexpr (IS_MEANS) {
        store_means_and_y_hat0(accumulator, symbols0, mask0, part, y_hat0);
    } else {
        store_prior(accumulator, part);
    }
}

template <bool IS_MEANS>
inline void pixel_tile(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                       const at::BFloat16* bias, const at::BFloat16* symbols0,
                       const at::BFloat16* mask0, at::BFloat16* part, at::BFloat16* y_hat0,
                       int64_t pixel, int weight_block0, int part_block0)
{
    const __m512 bias0 = load_bf16(bias + weight_block0 * CHANNEL_BLOCK);
    const __m512 bias1 = load_bf16(bias + (weight_block0 + 1) * CHANNEL_BLOCK);
    __m512 acc00 = bias0, acc01 = bias1;
    __m512 acc10 = bias0, acc11 = bias1;
    __m512 acc20 = bias0, acc21 = bias1;
    __m512 acc30 = bias0, acc31 = bias1;
    __m512 acc40 = bias0, acc41 = bias1;

    const auto* input0 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 0) * INPUT_CHANNELS);
    const auto* input1 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 1) * INPUT_CHANNELS);
    const auto* input2 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 2) * INPUT_CHANNELS);
    const auto* input3 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 3) * INPUT_CHANNELS);
    const auto* input4 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 4) * INPUT_CHANNELS);

    for (int pair = 0; pair < INPUT_PAIRS; pair += 4) {
        if (pair + 4 < INPUT_PAIRS) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             packed_weight + ((pair + 4) * OUTPUT_BLOCKS + weight_block0) * 32),
                         _MM_HINT_T0);
        }
#define ACCUMULATE_PAIR(PAIR)                                                                        \
    do {                                                                                             \
        const at::BFloat16* weights = packed_weight + ((PAIR) * OUTPUT_BLOCKS + weight_block0) * 32; \
        const __m512bh weight0 = (__m512bh)_mm512_loadu_si512(weights);                              \
        const __m512bh weight1 = (__m512bh)_mm512_loadu_si512(weights + 32);                         \
        const __m512bh value0 = (__m512bh)_mm512_set1_epi32(input0[PAIR]);                           \
        const __m512bh value1 = (__m512bh)_mm512_set1_epi32(input1[PAIR]);                           \
        const __m512bh value2 = (__m512bh)_mm512_set1_epi32(input2[PAIR]);                           \
        const __m512bh value3 = (__m512bh)_mm512_set1_epi32(input3[PAIR]);                           \
        const __m512bh value4 = (__m512bh)_mm512_set1_epi32(input4[PAIR]);                           \
        acc00 = _mm512_dpbf16_ps(acc00, value0, weight0);                                            \
        acc01 = _mm512_dpbf16_ps(acc01, value0, weight1);                                            \
        acc10 = _mm512_dpbf16_ps(acc10, value1, weight0);                                            \
        acc11 = _mm512_dpbf16_ps(acc11, value1, weight1);                                            \
        acc20 = _mm512_dpbf16_ps(acc20, value2, weight0);                                            \
        acc21 = _mm512_dpbf16_ps(acc21, value2, weight1);                                            \
        acc30 = _mm512_dpbf16_ps(acc30, value3, weight0);                                            \
        acc31 = _mm512_dpbf16_ps(acc31, value3, weight1);                                            \
        acc40 = _mm512_dpbf16_ps(acc40, value4, weight0);                                            \
        acc41 = _mm512_dpbf16_ps(acc41, value4, weight1);                                            \
    } while (false)
        ACCUMULATE_PAIR(pair + 0);
        ACCUMULATE_PAIR(pair + 1);
        ACCUMULATE_PAIR(pair + 2);
        ACCUMULATE_PAIR(pair + 3);
#undef ACCUMULATE_PAIR
    }

#define STORE(PIXEL, BLOCK)                                                                             \
    do {                                                                                                \
        const int64_t offset = (pixel + PIXEL) * PART_CHANNELS + (part_block0 + BLOCK) * CHANNEL_BLOCK; \
        store_output<IS_MEANS>(acc##PIXEL##BLOCK, symbols0 + offset, mask0 + offset,                    \
                               part + offset, y_hat0 + offset);                                         \
    } while (false)
    STORE(0, 0);
    STORE(0, 1);
    STORE(1, 0);
    STORE(1, 1);
    STORE(2, 0);
    STORE(2, 1);
    STORE(3, 0);
    STORE(3, 1);
    STORE(4, 0);
    STORE(4, 1);
#undef STORE
}

template <bool IS_MEANS>
inline void single_pixel(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                         const at::BFloat16* bias, const at::BFloat16* symbols0,
                         const at::BFloat16* mask0, at::BFloat16* part, at::BFloat16* y_hat0,
                         int64_t pixel, int weight_block0, int part_block0)
{
    __m512 acc0 = load_bf16(bias + weight_block0 * CHANNEL_BLOCK);
    __m512 acc1 = load_bf16(bias + (weight_block0 + 1) * CHANNEL_BLOCK);
    const auto* input = reinterpret_cast<const AliasedUint32*>(feature + pixel * INPUT_CHANNELS);
    for (int pair = 0; pair < INPUT_PAIRS; pair += 4) {
        if (pair + 4 < INPUT_PAIRS) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             packed_weight + ((pair + 4) * OUTPUT_BLOCKS + weight_block0) * 32),
                         _MM_HINT_T0);
        }
#define ACCUMULATE_PAIR(PAIR)                                                                        \
    do {                                                                                             \
        const __m512bh value = (__m512bh)_mm512_set1_epi32(input[PAIR]);                             \
        const at::BFloat16* weights = packed_weight + ((PAIR) * OUTPUT_BLOCKS + weight_block0) * 32; \
        acc0 = _mm512_dpbf16_ps(acc0, value, (__m512bh)_mm512_loadu_si512(weights));                 \
        acc1 = _mm512_dpbf16_ps(acc1, value, (__m512bh)_mm512_loadu_si512(weights + 32));            \
    } while (false)
        ACCUMULATE_PAIR(pair + 0);
        ACCUMULATE_PAIR(pair + 1);
        ACCUMULATE_PAIR(pair + 2);
        ACCUMULATE_PAIR(pair + 3);
#undef ACCUMULATE_PAIR
    }
    const int64_t offset = pixel * PART_CHANNELS + part_block0 * CHANNEL_BLOCK;
    store_output<IS_MEANS>(acc0, symbols0 + offset, mask0 + offset, part + offset, y_hat0 + offset);
    store_output<IS_MEANS>(acc1, symbols0 + offset + CHANNEL_BLOCK, mask0 + offset + CHANNEL_BLOCK,
                           part + offset + CHANNEL_BLOCK, y_hat0 + offset + CHANNEL_BLOCK);
}

template <bool IS_MEANS>
void output_part(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                 const at::BFloat16* bias, const at::BFloat16* symbols0, const at::BFloat16* mask0,
                 at::BFloat16* part, at::BFloat16* y_hat0, int64_t pixels, int tile)
{
    const int part_block0 = tile * OUTPUT_BLOCK_TILE;
    const int weight_block0 = (IS_MEANS ? PART_BLOCKS : 0) + part_block0;
    int64_t pixel = 0;
    for (; pixel + PIXEL_TILE <= pixels; pixel += PIXEL_TILE) {
        pixel_tile<IS_MEANS>(feature, packed_weight, bias, symbols0, mask0, part, y_hat0, pixel,
                             weight_block0, part_block0);
    }
    for (; pixel < pixels; ++pixel) {
        single_pixel<IS_MEANS>(feature, packed_weight, bias, symbols0, mask0, part, y_hat0, pixel,
                               weight_block0, part_block0);
    }
}

void check_feature(const at::Tensor& tensor, const char* name, int64_t batch, int64_t height,
                   int64_t width)
{
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == at::kBFloat16
                    && tensor.sizes() == at::IntArrayRef({ batch, PART_CHANNELS, height, width })
                    && tensor.is_contiguous(at::MemoryFormat::ChannelsLast),
                name, "unexpected tensor shape, layout or dtype", batch, ", 320, ", height, ", ", width,
                "]");
}

}  // namespace

at::Tensor dense_conv1x1_prior_masked_add_avx512_bf16(const at::Tensor& feature,
                                                      const at::Tensor& packed_weight,
                                                      const at::Tensor& bias, const at::Tensor& symbols0,
                                                      const at::Tensor& mask0)
{
    TORCH_CHECK(feature.device().is_cpu() && feature.scalar_type() == at::kBFloat16
                    && feature.dim() == 4 && feature.size(1) == INPUT_CHANNELS
                    && feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "unexpected tensor shape, layout or dtype");
    const int64_t batch = feature.size(0);
    const int64_t height = feature.size(2);
    const int64_t width = feature.size(3);
    check_feature(symbols0, "symbols0", batch, height, width);
    check_feature(mask0, "mask0", batch, height, width);
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.sizes() == at::IntArrayRef({ INPUT_PAIRS, OUTPUT_BLOCKS, 32 })
                    && packed_weight.is_contiguous(),
                "unexpected tensor shape, layout or dtype");
    TORCH_CHECK(bias.device().is_cpu() && bias.scalar_type() == at::kBFloat16
                    && bias.sizes() == at::IntArrayRef({ OUTPUT_CHANNELS }) && bias.is_contiguous(),
                "unexpected tensor shape, layout or dtype");

    at::Tensor output = at::empty({ batch * 3, PART_CHANNELS, height, width },
                                  feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* weight_ptr = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_ptr = bias.data_ptr<at::BFloat16>();
    const auto* symbols0_ptr = symbols0.data_ptr<at::BFloat16>();
    const auto* mask0_ptr = mask0.data_ptr<at::BFloat16>();
    auto* qstep_ptr = output.data_ptr<at::BFloat16>();
    const int64_t part_elements = batch * height * width * PART_CHANNELS;
    auto* means0_ptr = qstep_ptr + part_elements;
    auto* y_hat0_ptr = means0_ptr + part_elements;
    constexpr int PART_TILES = PART_BLOCKS / OUTPUT_BLOCK_TILE;
    const int64_t pixels = batch * height * width;

#pragma omp parallel for schedule(static) num_threads(1)
    for (int tile = 0; tile < 2 * PART_TILES; ++tile) {
        if (tile < PART_TILES) {
            output_part<false>(feature_ptr, weight_ptr, bias_ptr, symbols0_ptr, mask0_ptr,
                               qstep_ptr, y_hat0_ptr, pixels, tile);
        } else {
            output_part<true>(feature_ptr, weight_ptr, bias_ptr, symbols0_ptr, mask0_ptr,
                              means0_ptr, y_hat0_ptr, pixels, tile - PART_TILES);
        }
    }
    return output;
}
