#include "model_config.h"
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dense_conv1x1_reconstruct_bf16.h"

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_reconstruct_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <cstdint>
#include <immintrin.h>

namespace {

constexpr int INPUT_CHANNELS = PULSE_ENTROPY_CHANNELS;
constexpr int OUTPUT_CHANNELS = PULSE_MAIN_CHANNELS;
constexpr int CHANNEL_BLOCK = 16;
constexpr int INPUT_PAIRS = INPUT_CHANNELS / 2;
constexpr int OUTPUT_BLOCKS = OUTPUT_CHANNELS / CHANNEL_BLOCK;
constexpr int OUTPUT_BLOCK_TILE = 2;
constexpr int PIXEL_TILE = 5;
static_assert((OUTPUT_CHANNELS / 2) % (OUTPUT_BLOCK_TILE * CHANNEL_BLOCK) == 0);

using AliasedUint32 = uint32_t __attribute__((may_alias));

inline __m512 load_bf16(const at::BFloat16* input)
{
    return _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input)));
}

inline void store_scaled(__m512 selected, const at::BFloat16* qstep, __m512 codec_scale,
                         at::BFloat16* output)
{
    const __m512 q = _mm512_max_ps(load_bf16(qstep), _mm512_set1_ps(0.5f));
    const __m512 quantized = _mm512_mul_ps(selected, q);
    const __m256i bits = (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(quantized, codec_scale));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), bits);
}

inline void store_first(const at::BFloat16* y_hat0, const at::BFloat16* qstep, __m512 codec_scale,
                        at::BFloat16* output)
{
    store_scaled(load_bf16(y_hat0), qstep, codec_scale, output);
}

inline void store_second(__m512 means1_accumulator, const at::BFloat16* symbols1,
                         const at::BFloat16* qstep, __m512 codec_scale, at::BFloat16* output)
{
    store_scaled(_mm512_add_ps(load_bf16(symbols1), means1_accumulator), qstep, codec_scale, output);
}

template <bool COMPUTE_EVEN_PIXELS>
inline void pixel_tile(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                       const at::BFloat16* bias, const at::BFloat16* y_hat0,
                       const at::BFloat16* symbols1, const at::BFloat16* qstep, __m512 codec_scale,
                       at::BFloat16* output, int64_t pixel, int block0)
{
    const __m512 bias0 = load_bf16(bias + block0 * CHANNEL_BLOCK);
    const __m512 bias1 = load_bf16(bias + (block0 + 1) * CHANNEL_BLOCK);
    __m512 acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31, acc40, acc41;
#define INITIALIZE(PIXEL)                                      \
    if constexpr (COMPUTE_EVEN_PIXELS == ((PIXEL % 2) == 0)) { \
        acc##PIXEL##0 = bias0;                                 \
        acc##PIXEL##1 = bias1;                                 \
    }
    INITIALIZE(0);
    INITIALIZE(1);
    INITIALIZE(2);
    INITIALIZE(3);
    INITIALIZE(4);
#undef INITIALIZE

    const auto* input0 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 0) * INPUT_CHANNELS);
    const auto* input1 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 1) * INPUT_CHANNELS);
    const auto* input2 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 2) * INPUT_CHANNELS);
    const auto* input3 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 3) * INPUT_CHANNELS);
    const auto* input4 = reinterpret_cast<const AliasedUint32*>(feature + (pixel + 4) * INPUT_CHANNELS);

    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        const at::BFloat16* weights = packed_weight + (pair * OUTPUT_BLOCKS + block0) * 32;
        const __m512bh weight0 = (__m512bh)_mm512_loadu_si512(weights);
        const __m512bh weight1 = (__m512bh)_mm512_loadu_si512(weights + 32);
#define ACCUMULATE(PIXEL)                                                              \
    if constexpr (COMPUTE_EVEN_PIXELS == ((PIXEL % 2) == 0)) {                         \
        const __m512bh value##PIXEL = (__m512bh)_mm512_set1_epi32(input##PIXEL[pair]); \
        acc##PIXEL##0 = _mm512_dpbf16_ps(acc##PIXEL##0, value##PIXEL, weight0);        \
        acc##PIXEL##1 = _mm512_dpbf16_ps(acc##PIXEL##1, value##PIXEL, weight1);        \
    }
        ACCUMULATE(0);
        ACCUMULATE(1);
        ACCUMULATE(2);
        ACCUMULATE(3);
        ACCUMULATE(4);
#undef ACCUMULATE
    }

#define STORE(PIXEL, BLOCK)                                                                          \
    do {                                                                                             \
        const int64_t offset = (pixel + PIXEL) * OUTPUT_CHANNELS + (block0 + BLOCK) * CHANNEL_BLOCK; \
        if constexpr (COMPUTE_EVEN_PIXELS == ((PIXEL % 2) == 0)) {                                   \
            store_second(acc##PIXEL##BLOCK, symbols1 + offset, qstep + offset, codec_scale,          \
                         output + offset);                                                           \
        } else {                                                                                     \
            store_first(y_hat0 + offset, qstep + offset, codec_scale, output + offset);              \
        }                                                                                            \
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

template <bool COMPUTE_MEANS>
inline void single_pixel(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                         const at::BFloat16* bias, const at::BFloat16* y_hat0,
                         const at::BFloat16* symbols1, const at::BFloat16* qstep,
                         __m512 codec_scale, at::BFloat16* output, int64_t pixel, int block0)
{
    const int64_t offset = pixel * OUTPUT_CHANNELS + block0 * CHANNEL_BLOCK;
    if constexpr (COMPUTE_MEANS) {
        __m512 acc0 = load_bf16(bias + block0 * CHANNEL_BLOCK);
        __m512 acc1 = load_bf16(bias + (block0 + 1) * CHANNEL_BLOCK);
        const auto* input = reinterpret_cast<const AliasedUint32*>(feature + pixel * INPUT_CHANNELS);
        for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
            const __m512bh value = (__m512bh)_mm512_set1_epi32(input[pair]);
            const at::BFloat16* weights = packed_weight + (pair * OUTPUT_BLOCKS + block0) * 32;
            acc0 = _mm512_dpbf16_ps(acc0, value, (__m512bh)_mm512_loadu_si512(weights));
            acc1 = _mm512_dpbf16_ps(acc1, value, (__m512bh)_mm512_loadu_si512(weights + 32));
        }
        store_second(acc0, symbols1 + offset, qstep + offset, codec_scale, output + offset);
        store_second(acc1, symbols1 + offset + CHANNEL_BLOCK, qstep + offset + CHANNEL_BLOCK,
                     codec_scale, output + offset + CHANNEL_BLOCK);
    } else {
        store_first(y_hat0 + offset, qstep + offset, codec_scale, output + offset);
        store_first(y_hat0 + offset + CHANNEL_BLOCK, qstep + offset + CHANNEL_BLOCK, codec_scale,
                    output + offset + CHANNEL_BLOCK);
    }
}

void check_feature(const at::Tensor& tensor, const char* name, int64_t channels, int64_t batch,
                   int64_t height, int64_t width)
{
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == at::kBFloat16
                    && tensor.sizes() == at::IntArrayRef({ batch, channels, height, width })
                    && tensor.is_contiguous(at::MemoryFormat::ChannelsLast),
                name, "unexpected tensor shape, layout or dtype", batch, ", ", channels, ", ", height,
                ", ", width, "]");
}

}  // namespace

at::Tensor dense_conv1x1_spatial_reconstruct_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias,
    const at::Tensor& y_hat0, const at::Tensor& symbols1, const at::Tensor& qstep,
    const at::Tensor& mask0, const at::Tensor& codec_scale)
{
    TORCH_CHECK(feature.device().is_cpu() && feature.scalar_type() == at::kBFloat16
                    && feature.dim() == 4 && feature.size(1) == INPUT_CHANNELS
                    && feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "unexpected tensor shape, layout or dtype");
    const int64_t batch = feature.size(0);
    const int64_t height = feature.size(2);
    const int64_t width = feature.size(3);
    check_feature(y_hat0, "y_hat0", OUTPUT_CHANNELS, batch, height, width);
    check_feature(symbols1, "symbols1", OUTPUT_CHANNELS, batch, height, width);
    check_feature(qstep, "qstep", OUTPUT_CHANNELS, batch, height, width);
    check_feature(mask0, "mask0", OUTPUT_CHANNELS, batch, height, width);
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.sizes() == at::IntArrayRef({ INPUT_PAIRS, OUTPUT_BLOCKS, 32 })
                    && packed_weight.is_contiguous(),
                "unexpected tensor shape, layout or dtype");
    TORCH_CHECK(bias.device().is_cpu() && bias.scalar_type() == at::kBFloat16
                    && bias.sizes() == at::IntArrayRef({ OUTPUT_CHANNELS }) && bias.is_contiguous(),
                "unexpected tensor shape, layout or dtype");
    TORCH_CHECK(codec_scale.device().is_cpu() && codec_scale.scalar_type() == at::kBFloat16
                    && codec_scale.numel() == 1 && codec_scale.is_contiguous(),
                "unexpected tensor shape, layout or dtype");

    at::Tensor output = at::empty_like(y_hat0, y_hat0.options(), at::MemoryFormat::ChannelsLast);
    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* weight_ptr = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_ptr = bias.data_ptr<at::BFloat16>();
    const auto* y_hat0_ptr = y_hat0.data_ptr<at::BFloat16>();
    const auto* symbols1_ptr = symbols1.data_ptr<at::BFloat16>();
    const auto* qstep_ptr = qstep.data_ptr<at::BFloat16>();
    const auto* mask0_ptr = mask0.data_ptr<at::BFloat16>();
    auto* output_ptr = output.data_ptr<at::BFloat16>();
    const __m512 scale = _mm512_set1_ps(static_cast<float>(*codec_scale.data_ptr<at::BFloat16>()));
#pragma omp parallel for schedule(static) num_threads(1)
    for (int tile = 0; tile < OUTPUT_BLOCKS / OUTPUT_BLOCK_TILE; ++tile) {
        const int block0 = tile * OUTPUT_BLOCK_TILE;
        for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            for (int64_t y = 0; y < height; ++y) {
                const int64_t row_pixel = (batch_index * height + y) * width;
                int64_t x = 0;
                for (; x + PIXEL_TILE <= width; x += PIXEL_TILE) {
                    const int64_t pixel = row_pixel + x;
                    const int64_t mask_offset = pixel * OUTPUT_CHANNELS + block0 * CHANNEL_BLOCK;
                    if (mask0_ptr[mask_offset].x == 0) {
                        pixel_tile<true>(feature_ptr, weight_ptr, bias_ptr, y_hat0_ptr,
                                         symbols1_ptr, qstep_ptr, scale, output_ptr, pixel, block0);
                    } else {
                        pixel_tile<false>(feature_ptr, weight_ptr, bias_ptr, y_hat0_ptr,
                                          symbols1_ptr, qstep_ptr, scale, output_ptr, pixel, block0);
                    }
                }
                for (; x < width; ++x) {
                    const int64_t pixel = row_pixel + x;
                    const int64_t mask_offset = pixel * OUTPUT_CHANNELS + block0 * CHANNEL_BLOCK;
                    if (mask0_ptr[mask_offset].x == 0) {
                        single_pixel<true>(feature_ptr, weight_ptr, bias_ptr, y_hat0_ptr,
                                           symbols1_ptr, qstep_ptr, scale, output_ptr, pixel, block0);
                    } else {
                        single_pixel<false>(feature_ptr, weight_ptr, bias_ptr, y_hat0_ptr,
                                            symbols1_ptr, qstep_ptr, scale, output_ptr, pixel, block0);
                    }
                }
            }
        }
    }
    return output;
}
