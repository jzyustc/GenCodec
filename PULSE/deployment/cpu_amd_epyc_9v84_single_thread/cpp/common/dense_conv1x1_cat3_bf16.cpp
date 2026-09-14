#include "model_config.h"
#include "dense_conv1x1_cat3_bf16.h"

#include <cstdint>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_cat3_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int INPUT_CHANNELS = PULSE_MAIN_CHANNELS;
static constexpr int OUTPUT_CHANNELS = PULSE_ENTROPY_CHANNELS;
static constexpr int INPUT_PAIRS = INPUT_CHANNELS / 2;
static constexpr int OUTPUT_BLOCKS = (OUTPUT_CHANNELS + 15) / 16;
static constexpr int PIXEL_TILE = 5;
#ifndef CAT3_OUTPUT_BLOCK_TILE
    #define CAT3_OUTPUT_BLOCK_TILE 4
#endif
static constexpr int OUTPUT_BLOCK_TILE = CAT3_OUTPUT_BLOCK_TILE;
static_assert(OUTPUT_BLOCK_TILE >= 1 && OUTPUT_BLOCK_TILE <= 5);
static_assert(OUTPUT_BLOCKS % OUTPUT_BLOCK_TILE == 0);

using AliasedUint32 = uint32_t __attribute__((may_alias));

#define FOR_EACH_BLOCK(OP) \
    OP(0)                  \
    OP(1)                  \
    OP(2)                  \
    OP(3)                  \
    OP(4)

#define LOAD_BIAS(BLOCK)                                                               \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                             \
        const int64_t output0 = (block0 + BLOCK) * 16;                                 \
        const __mmask16 mask = output_mask(output0);                                   \
        const __m256i bits =                                                           \
            mask == static_cast<__mmask16>(0xffff)                                     \
                ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + output0)) \
                : _mm256_maskz_loadu_epi16(mask, bias + output0);                      \
        acc##BLOCK = _mm512_cvtpbh_ps((__m256bh)bits);                                 \
    }

#define LOAD_BIAS_PIXEL_TILE(BLOCK)                                                    \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                             \
        const int64_t output0 = (block0 + BLOCK) * 16;                                 \
        const __mmask16 mask = output_mask(output0);                                   \
        const __m256i bits =                                                           \
            mask == static_cast<__mmask16>(0xffff)                                     \
                ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + output0)) \
                : _mm256_maskz_loadu_epi16(mask, bias + output0);                      \
        const __m512 bias_vector = _mm512_cvtpbh_ps((__m256bh)bits);                   \
        acc0##BLOCK = bias_vector;                                                     \
        acc1##BLOCK = bias_vector;                                                     \
        acc2##BLOCK = bias_vector;                                                     \
        acc3##BLOCK = bias_vector;                                                     \
        acc4##BLOCK = bias_vector;                                                     \
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

#define STORE_BLOCK(BLOCK)                                                          \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                          \
        const int64_t output0 = (block0 + BLOCK) * 16;                              \
        store_output(output + output_pixel * OUTPUT_CHANNELS, output0, acc##BLOCK); \
    }

#define STORE_BLOCK_PIXEL_TILE(BLOCK)                                                 \
    if constexpr (ACTIVE_BLOCKS > BLOCK) {                                            \
        const int64_t output0 = (block0 + BLOCK) * 16;                                \
        store_output(output + output_pixel0 * OUTPUT_CHANNELS, output0, acc0##BLOCK); \
        store_output(output + output_pixel1 * OUTPUT_CHANNELS, output0, acc1##BLOCK); \
        store_output(output + output_pixel2 * OUTPUT_CHANNELS, output0, acc2##BLOCK); \
        store_output(output + output_pixel3 * OUTPUT_CHANNELS, output0, acc3##BLOCK); \
        store_output(output + output_pixel4 * OUTPUT_CHANNELS, output0, acc4##BLOCK); \
    }

static inline __mmask16 output_mask(int64_t output0)
{
    const int64_t remaining = OUTPUT_CHANNELS - output0;
    return remaining >= 16 ? static_cast<__mmask16>(0xffff)
                           : static_cast<__mmask16>((1u << remaining) - 1u);
}

static inline void store_output(at::BFloat16* output, int64_t output0, __m512 value)
{
    const __m256i bits = (__m256i)_mm512_cvtneps_pbh(value);
    const __mmask16 mask = output_mask(output0);
    if (mask == static_cast<__mmask16>(0xffff)) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + output0), bits);
    } else {
        _mm256_mask_storeu_epi16(output + output0, mask, bits);
    }
}

template <int ACTIVE_BLOCKS>
static inline void cat3_pixel_tile(const at::BFloat16* input0, const at::BFloat16* input1,
                                   const at::BFloat16* input2, const at::BFloat16* packed_weight,
                                   const at::BFloat16* bias, at::BFloat16* output,
                                   int64_t output_pixel, int64_t block0)
{
    __m512 acc0;
    __m512 acc1;
    __m512 acc2;
    __m512 acc3;
    __m512 acc4;
    FOR_EACH_BLOCK(LOAD_BIAS)

    const at::BFloat16* inputs[3] = { input0, input1, input2 };
    for (int part = 0; part < 3; ++part) {
        const auto* input_pairs = reinterpret_cast<const AliasedUint32*>(inputs[part]);
        const int global_pair0 = part * INPUT_PAIRS;
        for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
            const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(input_pairs[pair]);
            const at::BFloat16* weights =
                packed_weight + ((global_pair0 + pair) * OUTPUT_BLOCKS + block0) * 32;
            FOR_EACH_BLOCK(UPDATE_BLOCK)
        }
    }
    FOR_EACH_BLOCK(STORE_BLOCK)
}

template <int ACTIVE_BLOCKS>
static inline void cat3_multi_pixel_tile(
    const at::BFloat16* input00, const at::BFloat16* input01, const at::BFloat16* input02,
    const at::BFloat16* input10, const at::BFloat16* input11, const at::BFloat16* input12,
    const at::BFloat16* input20, const at::BFloat16* input21, const at::BFloat16* input22,
    const at::BFloat16* input30, const at::BFloat16* input31, const at::BFloat16* input32,
    const at::BFloat16* input40, const at::BFloat16* input41, const at::BFloat16* input42,
    const at::BFloat16* packed_weight, const at::BFloat16* bias, at::BFloat16* output,
    int64_t output_pixel0, int64_t output_pixel1, int64_t output_pixel2, int64_t output_pixel3,
    int64_t output_pixel4, int64_t block0)
{
    __m512 acc00, acc01, acc02, acc03, acc04;
    __m512 acc10, acc11, acc12, acc13, acc14;
    __m512 acc20, acc21, acc22, acc23, acc24;
    __m512 acc30, acc31, acc32, acc33, acc34;
    __m512 acc40, acc41, acc42, acc43, acc44;
    FOR_EACH_BLOCK(LOAD_BIAS_PIXEL_TILE)

    const at::BFloat16* inputs0[3] = { input00, input01, input02 };
    const at::BFloat16* inputs1[3] = { input10, input11, input12 };
    const at::BFloat16* inputs2[3] = { input20, input21, input22 };
    const at::BFloat16* inputs3[3] = { input30, input31, input32 };
    const at::BFloat16* inputs4[3] = { input40, input41, input42 };
    for (int part = 0; part < 3; ++part) {
        const auto* input_pairs0 = reinterpret_cast<const AliasedUint32*>(inputs0[part]);
        const auto* input_pairs1 = reinterpret_cast<const AliasedUint32*>(inputs1[part]);
        const auto* input_pairs2 = reinterpret_cast<const AliasedUint32*>(inputs2[part]);
        const auto* input_pairs3 = reinterpret_cast<const AliasedUint32*>(inputs3[part]);
        const auto* input_pairs4 = reinterpret_cast<const AliasedUint32*>(inputs4[part]);
        const int global_pair0 = part * INPUT_PAIRS;
        for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
            const __m512bh input_vector0 = (__m512bh)_mm512_set1_epi32(input_pairs0[pair]);
            const __m512bh input_vector1 = (__m512bh)_mm512_set1_epi32(input_pairs1[pair]);
            const __m512bh input_vector2 = (__m512bh)_mm512_set1_epi32(input_pairs2[pair]);
            const __m512bh input_vector3 = (__m512bh)_mm512_set1_epi32(input_pairs3[pair]);
            const __m512bh input_vector4 = (__m512bh)_mm512_set1_epi32(input_pairs4[pair]);
            const at::BFloat16* weights =
                packed_weight + ((global_pair0 + pair) * OUTPUT_BLOCKS + block0) * 32;
            FOR_EACH_BLOCK(UPDATE_BLOCK_PIXEL_TILE)
        }
    }
    FOR_EACH_BLOCK(STORE_BLOCK_PIXEL_TILE)
}

at::Tensor dense_conv1x1_cat3_bias_avx512_bf16(const at::Tensor& input0, const at::Tensor& input1,
                                               const at::Tensor& input2,
                                               const at::Tensor& packed_weight, const at::Tensor& bias)
{
    const at::Tensor* inputs[3] = { &input0, &input1, &input2 };
    for (const at::Tensor* input : inputs) {
        TORCH_CHECK(input->device().is_cpu() && input->scalar_type() == at::kBFloat16
                        && input->dim() == 4 && input->size(1) == INPUT_CHANNELS
                        && input->is_contiguous(at::MemoryFormat::ChannelsLast),
                    "unexpected tensor shape, layout or dtype");
        TORCH_CHECK(input->sizes() == input0.sizes(), "unexpected tensor shape, layout or dtype");
    }
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.sizes() == at::IntArrayRef({ 3 * INPUT_PAIRS, OUTPUT_BLOCKS, 32 })
                    && packed_weight.is_contiguous(),
                "unexpected tensor shape, layout or dtype");
    TORCH_CHECK(bias.device().is_cpu() && bias.scalar_type() == at::kBFloat16
                    && bias.sizes() == at::IntArrayRef({ OUTPUT_CHANNELS }) && bias.is_contiguous(),
                "unexpected tensor shape, layout or dtype");

    at::Tensor output = at::empty({ input0.size(0), OUTPUT_CHANNELS, input0.size(2), input0.size(3) },
                                  input0.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* pointers0 = input0.data_ptr<at::BFloat16>();
    const auto* pointers1 = input1.data_ptr<at::BFloat16>();
    const auto* pointers2 = input2.data_ptr<at::BFloat16>();
    const auto* weight = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_pointer = bias.data_ptr<at::BFloat16>();
    auto* output_pointer = output.data_ptr<at::BFloat16>();
    const int64_t pixels = input0.size(0) * input0.size(2) * input0.size(3);

    for (int64_t block0 = 0; block0 < OUTPUT_BLOCKS; block0 += OUTPUT_BLOCK_TILE) {
        int64_t pixel = 0;
        for (; pixel + PIXEL_TILE <= pixels; pixel += PIXEL_TILE) {
            cat3_multi_pixel_tile<OUTPUT_BLOCK_TILE>(
                pointers0 + pixel * INPUT_CHANNELS, pointers1 + pixel * INPUT_CHANNELS,
                pointers2 + pixel * INPUT_CHANNELS, pointers0 + (pixel + 1) * INPUT_CHANNELS,
                pointers1 + (pixel + 1) * INPUT_CHANNELS, pointers2 + (pixel + 1) * INPUT_CHANNELS,
                pointers0 + (pixel + 2) * INPUT_CHANNELS, pointers1 + (pixel + 2) * INPUT_CHANNELS,
                pointers2 + (pixel + 2) * INPUT_CHANNELS, pointers0 + (pixel + 3) * INPUT_CHANNELS,
                pointers1 + (pixel + 3) * INPUT_CHANNELS, pointers2 + (pixel + 3) * INPUT_CHANNELS,
                pointers0 + (pixel + 4) * INPUT_CHANNELS, pointers1 + (pixel + 4) * INPUT_CHANNELS,
                pointers2 + (pixel + 4) * INPUT_CHANNELS, weight, bias_pointer, output_pointer,
                pixel, pixel + 1, pixel + 2, pixel + 3, pixel + 4, block0);
        }
        for (; pixel < pixels; ++pixel) {
            cat3_pixel_tile<OUTPUT_BLOCK_TILE>(pointers0 + pixel * INPUT_CHANNELS,
                                               pointers1 + pixel * INPUT_CHANNELS,
                                               pointers2 + pixel * INPUT_CHANNELS, weight,
                                               bias_pointer, output_pointer, pixel, block0);
        }
    }
    return output;
}
