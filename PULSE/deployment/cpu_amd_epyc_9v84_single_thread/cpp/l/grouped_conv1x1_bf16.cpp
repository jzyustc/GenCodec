// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "grouped_conv1x1_bf16.h"

#include "activations_avx512.h"

#include <cstdint>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "grouped_conv1x1_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int GROUPS = 4;
static constexpr int PIXEL_TILE = 5;
#ifndef GROUPED_OUTPUT_BLOCK_TILE
    #define GROUPED_OUTPUT_BLOCK_TILE 5
#endif
static constexpr int PIXEL_TILE_OUTPUT_BLOCKS = GROUPED_OUTPUT_BLOCK_TILE;
static_assert(PIXEL_TILE_OUTPUT_BLOCKS >= 1 && PIXEL_TILE_OUTPUT_BLOCKS <= 5);
static constexpr int WEIGHT_PREFETCH_DISTANCE = 4;

using AliasedUint16 = uint16_t __attribute__((may_alias));

#define FOR_EACH_BLOCK(OP) \
    OP(0)                  \
    OP(1)                  \
    OP(2)                  \
    OP(3)                  \
    OP(4)

#define DISPATCH_PAIR(CIN, COUT)                                                                    \
    if (input_channels == CIN && output_channels == COUT) {                                         \
        grouped_conv1x1_specialized<CIN, COUT, HAS_BIAS, HAS_HARD_GELU_030>(feature, packed_weight, \
                                                                            bias, output, pixels);  \
        return true;                                                                                \
    }

#define LOAD_BIAS(BLOCK)                                                                     \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                   \
        const int64_t output0 = (block0 + BLOCK) * 16;                                       \
        const __m256i bits = load_bias_bits<GROUP_OUTPUT_CHANNELS, HAS_BIAS>(bias, output0); \
        acc##BLOCK = _mm512_cvtpbh_ps((__m256bh)bits);                                       \
    }

#define LOAD_BIAS_PIXEL_TILE(BLOCK)                                                          \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                   \
        const int64_t output0 = (block0 + BLOCK) * 16;                                       \
        const __m256i bits = load_bias_bits<GROUP_OUTPUT_CHANNELS, HAS_BIAS>(bias, output0); \
        const __m512 bias_vector = _mm512_cvtpbh_ps((__m256bh)bits);                         \
        acc0##BLOCK = bias_vector;                                                           \
        acc1##BLOCK = bias_vector;                                                           \
        acc2##BLOCK = bias_vector;                                                           \
        acc3##BLOCK = bias_vector;                                                           \
        acc4##BLOCK = bias_vector;                                                           \
    }

#define STORE_OUTPUT(BLOCK)                                                                             \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                                              \
        const int64_t output0 = (block0 + BLOCK) * 16;                                                  \
        const __m512 result = finalize_output<HAS_HARD_GELU_030>(acc##BLOCK);                           \
        store_output_bits<GROUP_OUTPUT_CHANNELS>(output, output0, (__m256i)_mm512_cvtneps_pbh(result)); \
    }

#define STORE_OUTPUT_PIXEL_TILE(BLOCK)                                                  \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                              \
        const int64_t output_channel = (block0 + BLOCK) * 16;                           \
        const __m512 result0 = finalize_output<HAS_HARD_GELU_030>(acc0##BLOCK);         \
        const __m512 result1 = finalize_output<HAS_HARD_GELU_030>(acc1##BLOCK);         \
        const __m512 result2 = finalize_output<HAS_HARD_GELU_030>(acc2##BLOCK);         \
        const __m512 result3 = finalize_output<HAS_HARD_GELU_030>(acc3##BLOCK);         \
        const __m512 result4 = finalize_output<HAS_HARD_GELU_030>(acc4##BLOCK);         \
        store_output_bits<GROUP_OUTPUT_CHANNELS>(output0, output_channel,               \
                                                 (__m256i)_mm512_cvtneps_pbh(result0)); \
        store_output_bits<GROUP_OUTPUT_CHANNELS>(output1, output_channel,               \
                                                 (__m256i)_mm512_cvtneps_pbh(result1)); \
        store_output_bits<GROUP_OUTPUT_CHANNELS>(output2, output_channel,               \
                                                 (__m256i)_mm512_cvtneps_pbh(result2)); \
        store_output_bits<GROUP_OUTPUT_CHANNELS>(output3, output_channel,               \
                                                 (__m256i)_mm512_cvtneps_pbh(result3)); \
        store_output_bits<GROUP_OUTPUT_CHANNELS>(output4, output_channel,               \
                                                 (__m256i)_mm512_cvtneps_pbh(result4)); \
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

template <int GROUP_OUTPUT_CHANNELS>
static inline __mmask16 output_mask(int64_t output0)
{
    const int64_t remaining = GROUP_OUTPUT_CHANNELS - output0;
    return remaining >= 16 ? static_cast<__mmask16>(0xffff)
                           : static_cast<__mmask16>((1u << remaining) - 1u);
}

template <int GROUP_OUTPUT_CHANNELS, bool HAS_BIAS>
static inline __m256i load_bias_bits(const at::BFloat16* bias, int64_t output0)
{
    if constexpr (!HAS_BIAS) {
        return _mm256_setzero_si256();
    }
    if constexpr (GROUP_OUTPUT_CHANNELS % 16 == 0) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + output0));
    }
    return _mm256_maskz_loadu_epi16(output_mask<GROUP_OUTPUT_CHANNELS>(output0), bias + output0);
}

template <int GROUP_OUTPUT_CHANNELS>
static inline void store_output_bits(at::BFloat16* output, int64_t output0, __m256i bits)
{
    if constexpr (GROUP_OUTPUT_CHANNELS % 16 == 0) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + output0), bits);
    } else {
        _mm256_mask_storeu_epi16(output + output0, output_mask<GROUP_OUTPUT_CHANNELS>(output0), bits);
    }
}

template <bool HAS_HARD_GELU_030>
static inline __m512 finalize_output(__m512 accumulator)
{
    __m512 result = accumulator;
    if constexpr (HAS_HARD_GELU_030) {
        result = hard_gelu_030(result);
    }
    return result;
}

template <int GROUP_INPUT_CHANNELS>
static inline uint32_t load_input_pair(const at::BFloat16* feature, int pair)
{
    const int channel = pair * 2;
    if (channel + 1 < GROUP_INPUT_CHANNELS) {
        uint32_t bits;
        __builtin_memcpy(&bits, feature + channel, sizeof(bits));
        return bits;
    }
    return static_cast<uint32_t>(*reinterpret_cast<const AliasedUint16*>(feature + channel));
}

template <int TOTAL_INPUT_CHANNELS, int TOTAL_OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS, bool HAS_HARD_GELU_030>
static inline void
grouped_conv1x1_pixel_tile(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                           const at::BFloat16* bias, at::BFloat16* output, int64_t block0)
{
    constexpr int GROUP_INPUT_CHANNELS = TOTAL_INPUT_CHANNELS / GROUPS;
    constexpr int GROUP_OUTPUT_CHANNELS = TOTAL_OUTPUT_CHANNELS / GROUPS;
    constexpr int INPUT_PAIRS = (GROUP_INPUT_CHANNELS + 1) / 2;
    constexpr int OUTPUT_BLOCKS = (GROUP_OUTPUT_CHANNELS + 15) / 16;

    __m512 acc0;
    __m512 acc1;
    __m512 acc2;
    __m512 acc3;
    __m512 acc4;

    FOR_EACH_BLOCK(LOAD_BIAS)

    int pair = 0;
    for (; pair + 3 < INPUT_PAIRS; pair += 4) {
        if (pair + WEIGHT_PREFETCH_DISTANCE < INPUT_PAIRS) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             packed_weight
                             + ((pair + WEIGHT_PREFETCH_DISTANCE) * OUTPUT_BLOCKS + block0) * 32),
                         _MM_HINT_T0);
        }
        for (int unroll = 0; unroll < 4; ++unroll) {
            const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(
                load_input_pair<GROUP_INPUT_CHANNELS>(feature, pair + unroll));
            const at::BFloat16* weights =
                packed_weight + ((pair + unroll) * OUTPUT_BLOCKS + block0) * 32;
            FOR_EACH_BLOCK(UPDATE_BLOCK)
        }
    }
    for (; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector =
            (__m512bh)_mm512_set1_epi32(load_input_pair<GROUP_INPUT_CHANNELS>(feature, pair));
        const at::BFloat16* weights = packed_weight + (pair * OUTPUT_BLOCKS + block0) * 32;
        FOR_EACH_BLOCK(UPDATE_BLOCK)
    }

    FOR_EACH_BLOCK(STORE_OUTPUT)
}

template <int TOTAL_INPUT_CHANNELS, int TOTAL_OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS, bool HAS_HARD_GELU_030>
static inline void grouped_conv1x1_multi_pixel_tile(
    const at::BFloat16* feature0, const at::BFloat16* feature1, const at::BFloat16* feature2,
    const at::BFloat16* feature3, const at::BFloat16* feature4, const at::BFloat16* packed_weight,
    const at::BFloat16* bias, at::BFloat16* output0, at::BFloat16* output1, at::BFloat16* output2,
    at::BFloat16* output3, at::BFloat16* output4, int64_t block0)
{
    constexpr int GROUP_INPUT_CHANNELS = TOTAL_INPUT_CHANNELS / GROUPS;
    constexpr int GROUP_OUTPUT_CHANNELS = TOTAL_OUTPUT_CHANNELS / GROUPS;
    constexpr int INPUT_PAIRS = (GROUP_INPUT_CHANNELS + 1) / 2;
    constexpr int OUTPUT_BLOCKS = (GROUP_OUTPUT_CHANNELS + 15) / 16;

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

    int pair = 0;
    for (; pair + 3 < INPUT_PAIRS; pair += 4) {
        if (pair + WEIGHT_PREFETCH_DISTANCE < INPUT_PAIRS) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             packed_weight
                             + ((pair + WEIGHT_PREFETCH_DISTANCE) * OUTPUT_BLOCKS + block0) * 32),
                         _MM_HINT_T0);
        }
        for (int unroll = 0; unroll < 4; ++unroll) {
            const int input_pair = pair + unroll;
            const __m512bh input_vector0 = (__m512bh)_mm512_set1_epi32(
                load_input_pair<GROUP_INPUT_CHANNELS>(feature0, input_pair));
            const __m512bh input_vector1 = (__m512bh)_mm512_set1_epi32(
                load_input_pair<GROUP_INPUT_CHANNELS>(feature1, input_pair));
            const __m512bh input_vector2 = (__m512bh)_mm512_set1_epi32(
                load_input_pair<GROUP_INPUT_CHANNELS>(feature2, input_pair));
            const __m512bh input_vector3 = (__m512bh)_mm512_set1_epi32(
                load_input_pair<GROUP_INPUT_CHANNELS>(feature3, input_pair));
            const __m512bh input_vector4 = (__m512bh)_mm512_set1_epi32(
                load_input_pair<GROUP_INPUT_CHANNELS>(feature4, input_pair));
            const at::BFloat16* weights = packed_weight + (input_pair * OUTPUT_BLOCKS + block0) * 32;
            FOR_EACH_BLOCK(UPDATE_BLOCK_PIXEL_TILE)
        }
    }
    for (; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector0 =
            (__m512bh)_mm512_set1_epi32(load_input_pair<GROUP_INPUT_CHANNELS>(feature0, pair));
        const __m512bh input_vector1 =
            (__m512bh)_mm512_set1_epi32(load_input_pair<GROUP_INPUT_CHANNELS>(feature1, pair));
        const __m512bh input_vector2 =
            (__m512bh)_mm512_set1_epi32(load_input_pair<GROUP_INPUT_CHANNELS>(feature2, pair));
        const __m512bh input_vector3 =
            (__m512bh)_mm512_set1_epi32(load_input_pair<GROUP_INPUT_CHANNELS>(feature3, pair));
        const __m512bh input_vector4 =
            (__m512bh)_mm512_set1_epi32(load_input_pair<GROUP_INPUT_CHANNELS>(feature4, pair));
        const at::BFloat16* weights = packed_weight + (pair * OUTPUT_BLOCKS + block0) * 32;
        FOR_EACH_BLOCK(UPDATE_BLOCK_PIXEL_TILE)
    }

    FOR_EACH_BLOCK(STORE_OUTPUT_PIXEL_TILE)
}

template <int TOTAL_INPUT_CHANNELS, int TOTAL_OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS, bool HAS_HARD_GELU_030>
static inline void grouped_conv1x1_output_tile(const at::BFloat16* feature,
                                               const at::BFloat16* packed_weight,
                                               const at::BFloat16* bias, at::BFloat16* output,
                                               int64_t pixels, int group, int64_t block0)
{
    constexpr int GROUP_INPUT_CHANNELS = TOTAL_INPUT_CHANNELS / GROUPS;
    constexpr int GROUP_OUTPUT_CHANNELS = TOTAL_OUTPUT_CHANNELS / GROUPS;
    const int64_t pixel_tiles = pixels / PIXEL_TILE;
    const at::BFloat16* group_weight =
        packed_weight
        + group * ((GROUP_INPUT_CHANNELS + 1) / 2) * ((GROUP_OUTPUT_CHANNELS + 15) / 16) * 32;
    const at::BFloat16* group_bias = HAS_BIAS ? bias + group * GROUP_OUTPUT_CHANNELS : nullptr;

    for (int64_t pixel_tile = 0; pixel_tile < pixel_tiles; ++pixel_tile) {
        const int64_t pixel0 = pixel_tile * PIXEL_TILE;
        const int64_t pixel1 = pixel0 + 1;
        const int64_t pixel2 = pixel0 + 2;
        const int64_t pixel3 = pixel0 + 3;
        const int64_t pixel4 = pixel0 + 4;
        const at::BFloat16* input0 =
            feature + pixel0 * TOTAL_INPUT_CHANNELS + group * GROUP_INPUT_CHANNELS;
        const at::BFloat16* input1 =
            feature + pixel1 * TOTAL_INPUT_CHANNELS + group * GROUP_INPUT_CHANNELS;
        const at::BFloat16* input2 =
            feature + pixel2 * TOTAL_INPUT_CHANNELS + group * GROUP_INPUT_CHANNELS;
        const at::BFloat16* input3 =
            feature + pixel3 * TOTAL_INPUT_CHANNELS + group * GROUP_INPUT_CHANNELS;
        const at::BFloat16* input4 =
            feature + pixel4 * TOTAL_INPUT_CHANNELS + group * GROUP_INPUT_CHANNELS;
        at::BFloat16* output0 = output + pixel0 * TOTAL_OUTPUT_CHANNELS + group * GROUP_OUTPUT_CHANNELS;
        at::BFloat16* output1 = output + pixel1 * TOTAL_OUTPUT_CHANNELS + group * GROUP_OUTPUT_CHANNELS;
        at::BFloat16* output2 = output + pixel2 * TOTAL_OUTPUT_CHANNELS + group * GROUP_OUTPUT_CHANNELS;
        at::BFloat16* output3 = output + pixel3 * TOTAL_OUTPUT_CHANNELS + group * GROUP_OUTPUT_CHANNELS;
        at::BFloat16* output4 = output + pixel4 * TOTAL_OUTPUT_CHANNELS + group * GROUP_OUTPUT_CHANNELS;

        grouped_conv1x1_multi_pixel_tile<TOTAL_INPUT_CHANNELS, TOTAL_OUTPUT_CHANNELS, ACTIVE_BLOCKS,
                                         HAS_BIAS, HAS_HARD_GELU_030>(
            input0, input1, input2, input3, input4, group_weight, group_bias, output0, output1,
            output2, output3, output4, block0);
    }

    for (int64_t pixel = pixel_tiles * PIXEL_TILE; pixel < pixels; ++pixel) {
        const at::BFloat16* input =
            feature + pixel * TOTAL_INPUT_CHANNELS + group * GROUP_INPUT_CHANNELS;
        at::BFloat16* out = output + pixel * TOTAL_OUTPUT_CHANNELS + group * GROUP_OUTPUT_CHANNELS;
        grouped_conv1x1_pixel_tile<TOTAL_INPUT_CHANNELS, TOTAL_OUTPUT_CHANNELS, ACTIVE_BLOCKS, HAS_BIAS,
                                   HAS_HARD_GELU_030>(input, group_weight, group_bias, out, block0);
    }
}

template <int TOTAL_INPUT_CHANNELS, int TOTAL_OUTPUT_CHANNELS, bool HAS_BIAS, bool HAS_HARD_GELU_030>
static void grouped_conv1x1_specialized(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                                        const at::BFloat16* bias, at::BFloat16* output, int64_t pixels)
{
    constexpr int GROUP_OUTPUT_CHANNELS = TOTAL_OUTPUT_CHANNELS / GROUPS;
    constexpr int OUTPUT_BLOCKS = (GROUP_OUTPUT_CHANNELS + 15) / 16;
    constexpr int FULL_TILES = OUTPUT_BLOCKS / PIXEL_TILE_OUTPUT_BLOCKS;
    constexpr int TAIL_BLOCKS = OUTPUT_BLOCKS % PIXEL_TILE_OUTPUT_BLOCKS;
    constexpr int OUTPUT_TILES = FULL_TILES + (TAIL_BLOCKS != 0);

#pragma omp parallel for schedule(static) num_threads(1)
    for (int task = 0; task < GROUPS * OUTPUT_TILES; ++task) {
        const int group = task / OUTPUT_TILES;
        const int tile = task % OUTPUT_TILES;
        const int64_t block0 = tile * PIXEL_TILE_OUTPUT_BLOCKS;
        if (tile < FULL_TILES) {
            grouped_conv1x1_output_tile<TOTAL_INPUT_CHANNELS, TOTAL_OUTPUT_CHANNELS,
                                        PIXEL_TILE_OUTPUT_BLOCKS, HAS_BIAS, HAS_HARD_GELU_030>(
                feature, packed_weight, bias, output, pixels, group, block0);
        } else if constexpr (TAIL_BLOCKS != 0) {
            grouped_conv1x1_output_tile<TOTAL_INPUT_CHANNELS, TOTAL_OUTPUT_CHANNELS, TAIL_BLOCKS,
                                        HAS_BIAS, HAS_HARD_GELU_030>(feature, packed_weight, bias,
                                                                     output, pixels, group, block0);
        }
    }
}

template <bool HAS_BIAS, bool HAS_HARD_GELU_030>
static bool dispatch_specialized(int64_t input_channels, int64_t output_channels,
                                 const at::BFloat16* feature, const at::BFloat16* packed_weight,
                                 const at::BFloat16* bias, at::BFloat16* output, int64_t pixels)
{
    DISPATCH_PAIR(96, 96)
    DISPATCH_PAIR(96, 192)
    DISPATCH_PAIR(100, 100)
    DISPATCH_PAIR(100, 200)
    DISPATCH_PAIR(128, 128)
    DISPATCH_PAIR(128, 256)
    DISPATCH_PAIR(148, 148)
    DISPATCH_PAIR(148, 296)
    DISPATCH_PAIR(192, 96)
    DISPATCH_PAIR(200, 100)
    DISPATCH_PAIR(256, 128)
    DISPATCH_PAIR(296, 148)
    DISPATCH_PAIR(144, 144)
    DISPATCH_PAIR(144, 288)
    DISPATCH_PAIR(256, 256)
    DISPATCH_PAIR(256, 512)
    return false;
}

at::Tensor grouped_conv1x1_bias_avx512_bf16(const at::Tensor& feature, const at::Tensor& packed_weight,
                                            const at::optional<at::Tensor>& bias,
                                            int64_t output_channels, bool fuse_hard_gelu_030)
{
    TORCH_CHECK(feature.device().is_cpu(), "feature must be a CPU tensor");
    TORCH_CHECK(feature.scalar_type() == at::kBFloat16, "feature must be BF16");
    TORCH_CHECK(feature.dim() == 4, "feature must have shape [N, C, H, W]");
    TORCH_CHECK(feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feature must be channels-last contiguous");

    TORCH_CHECK(packed_weight.device().is_cpu(), "packed_weight must be a CPU tensor");
    TORCH_CHECK(packed_weight.scalar_type() == at::kBFloat16, "packed_weight must be BF16");
    TORCH_CHECK(packed_weight.dim() == 4 && packed_weight.size(0) == GROUPS
                    && packed_weight.size(3) == 32,
                "packed_weight must have shape [4, ceil(Cin/4/2), ceil(Cout/4/16), 32]");
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
    const int64_t input_channels = feature.size(1);
    const int64_t height = feature.size(2);
    const int64_t width = feature.size(3);
    TORCH_CHECK(input_channels % GROUPS == 0, "Cin must be divisible by 4");
    TORCH_CHECK(output_channels % GROUPS == 0, "Cout must be divisible by 4");
    const int64_t group_input_channels = input_channels / GROUPS;
    const int64_t group_output_channels = output_channels / GROUPS;
    const int64_t input_pairs = (group_input_channels + 1) / 2;
    const int64_t output_blocks = (group_output_channels + 15) / 16;

    TORCH_CHECK(packed_weight.size(1) == input_pairs,
                "packed_weight input-pair dimension does not match grouped feature channels");
    TORCH_CHECK(packed_weight.size(2) == output_blocks,
                "packed_weight output-block dimension does not match grouped bias channels");

    at::Tensor output = at::empty({ batch, output_channels, height, width },
                                  feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* packed_weight_ptr = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_ptr = has_bias ? bias->data_ptr<at::BFloat16>() : nullptr;
    auto* output_ptr = output.data_ptr<at::BFloat16>();
    const int64_t pixels = batch * height * width;

    auto dispatch = [&](auto has_bias_tag, auto has_hard_gelu_030_tag) {
        return dispatch_specialized<decltype(has_bias_tag)::value, decltype(has_hard_gelu_030_tag)::value>(
            input_channels, output_channels, feature_ptr, packed_weight_ptr, bias_ptr, output_ptr,
            pixels);
    };
    auto dispatch_hard_gelu_030 = [&](auto has_bias_tag) {
        return fuse_hard_gelu_030 ? dispatch(has_bias_tag, std::true_type{})
                                  : dispatch(has_bias_tag, std::false_type{});
    };
    const bool dispatched = has_bias ? dispatch_hard_gelu_030(std::true_type{})
                                     : dispatch_hard_gelu_030(std::false_type{});
    TORCH_CHECK(dispatched, "unsupported grouped 1x1 channel pair (Cin=", input_channels,
                ", Cout=", output_channels, ", groups=4)");
    return output;
}
