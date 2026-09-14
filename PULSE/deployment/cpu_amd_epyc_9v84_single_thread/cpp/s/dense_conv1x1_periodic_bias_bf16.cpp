#include "dense_conv1x1_periodic_bias_bf16.h"

#include <cstdint>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_periodic_bias_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int INPUT_CHANNELS = 16;
static constexpr int OUTPUT_CHANNELS = 32;
static constexpr int INPUT_PAIRS = INPUT_CHANNELS / 2;
static constexpr int OUTPUT_BLOCKS = OUTPUT_CHANNELS / 16;
static constexpr int PIXEL_TILE = 5;

using AliasedUint32 = uint32_t __attribute__((may_alias));

static inline void store_output(__m512 convolution, const at::BFloat16* bias, at::BFloat16* output)
{
    const __m512 bias_value =
        _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output),
                        (__m256i)_mm512_cvtneps_pbh(_mm512_add_ps(convolution, bias_value)));
}

static inline const at::BFloat16* phase_bias(const at::BFloat16* bias, int64_t pixel,
                                             int64_t plane_pixels, int64_t width)
{
    const int64_t spatial_pixel = pixel % plane_pixels;
    const int64_t phase = ((spatial_pixel / width) & 3) * 4 + (spatial_pixel % width & 3);
    return bias + phase * OUTPUT_CHANNELS;
}

static inline void pixel_kernel(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                                const at::BFloat16* bias, at::BFloat16* output)
{
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    const auto* input_pairs = reinterpret_cast<const AliasedUint32*>(feature);
    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(input_pairs[pair]);
        const at::BFloat16* weights = packed_weight + pair * OUTPUT_BLOCKS * 32;
        acc0 = _mm512_dpbf16_ps(acc0, input_vector, (__m512bh)_mm512_loadu_si512(weights));
        acc1 = _mm512_dpbf16_ps(acc1, input_vector, (__m512bh)_mm512_loadu_si512(weights + 32));
    }
    store_output(acc0, bias, output);
    store_output(acc1, bias + 16, output + 16);
}

static inline void pixel_tile_kernel(const at::BFloat16* feature0, const at::BFloat16* feature1,
                                     const at::BFloat16* feature2, const at::BFloat16* feature3,
                                     const at::BFloat16* feature4, const at::BFloat16* packed_weight,
                                     const at::BFloat16* bias0, const at::BFloat16* bias1,
                                     const at::BFloat16* bias2, const at::BFloat16* bias3,
                                     const at::BFloat16* bias4, at::BFloat16* output0,
                                     at::BFloat16* output1, at::BFloat16* output2,
                                     at::BFloat16* output3, at::BFloat16* output4)
{
    __m512 acc00 = _mm512_setzero_ps(), acc01 = _mm512_setzero_ps();
    __m512 acc10 = _mm512_setzero_ps(), acc11 = _mm512_setzero_ps();
    __m512 acc20 = _mm512_setzero_ps(), acc21 = _mm512_setzero_ps();
    __m512 acc30 = _mm512_setzero_ps(), acc31 = _mm512_setzero_ps();
    __m512 acc40 = _mm512_setzero_ps(), acc41 = _mm512_setzero_ps();
    const auto* pairs0 = reinterpret_cast<const AliasedUint32*>(feature0);
    const auto* pairs1 = reinterpret_cast<const AliasedUint32*>(feature1);
    const auto* pairs2 = reinterpret_cast<const AliasedUint32*>(feature2);
    const auto* pairs3 = reinterpret_cast<const AliasedUint32*>(feature3);
    const auto* pairs4 = reinterpret_cast<const AliasedUint32*>(feature4);

    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input0 = (__m512bh)_mm512_set1_epi32(pairs0[pair]);
        const __m512bh input1 = (__m512bh)_mm512_set1_epi32(pairs1[pair]);
        const __m512bh input2 = (__m512bh)_mm512_set1_epi32(pairs2[pair]);
        const __m512bh input3 = (__m512bh)_mm512_set1_epi32(pairs3[pair]);
        const __m512bh input4 = (__m512bh)_mm512_set1_epi32(pairs4[pair]);
        const at::BFloat16* weights = packed_weight + pair * OUTPUT_BLOCKS * 32;
        const __m512bh weight0 = (__m512bh)_mm512_loadu_si512(weights);
        const __m512bh weight1 = (__m512bh)_mm512_loadu_si512(weights + 32);
#define UPDATE_PIXEL(PIXEL)                                                 \
    acc##PIXEL##0 = _mm512_dpbf16_ps(acc##PIXEL##0, input##PIXEL, weight0); \
    acc##PIXEL##1 = _mm512_dpbf16_ps(acc##PIXEL##1, input##PIXEL, weight1)
        UPDATE_PIXEL(0);
        UPDATE_PIXEL(1);
        UPDATE_PIXEL(2);
        UPDATE_PIXEL(3);
        UPDATE_PIXEL(4);
#undef UPDATE_PIXEL
    }

#define STORE_PIXEL(PIXEL)                                             \
    store_output(acc##PIXEL##0, bias##PIXEL, output##PIXEL);           \
    store_output(acc##PIXEL##1, bias##PIXEL + 16, output##PIXEL + 16)
    STORE_PIXEL(0);
    STORE_PIXEL(1);
    STORE_PIXEL(2);
    STORE_PIXEL(3);
    STORE_PIXEL(4);
#undef STORE_PIXEL
}

at::Tensor dense_conv1x1_periodic_bias4x4_avx512_bf16(const at::Tensor& feature,
                                                      const at::Tensor& packed_weight,
                                                      const at::Tensor& periodic_bias)
{
    TORCH_CHECK(feature.device().is_cpu() && feature.scalar_type() == at::kBFloat16
                    && feature.dim() == 4 && feature.size(1) == INPUT_CHANNELS
                    && feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feature must be channels-last CPU BF16 [N, 16, H, W]");
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.sizes() == at::IntArrayRef({ INPUT_PAIRS, OUTPUT_BLOCKS, 32 })
                    && packed_weight.is_contiguous(),
                "packed_weight must be contiguous BF16 [8, 2, 32]");
    TORCH_CHECK(periodic_bias.device().is_cpu() && periodic_bias.scalar_type() == at::kBFloat16
                    && periodic_bias.sizes() == at::IntArrayRef({ 1, OUTPUT_CHANNELS, 4, 4 })
                    && periodic_bias.is_contiguous(at::MemoryFormat::ChannelsLast),
                "periodic_bias must be channels-last CPU BF16 [1, 32, 4, 4]");

    at::Tensor output =
        at::empty({ feature.size(0), OUTPUT_CHANNELS, feature.size(2), feature.size(3) },
                  feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* input = feature.data_ptr<at::BFloat16>();
    const auto* weight = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias = periodic_bias.data_ptr<at::BFloat16>();
    auto* output_pointer = output.data_ptr<at::BFloat16>();
    const int64_t plane_pixels = feature.size(2) * feature.size(3);
    const int64_t pixels = feature.size(0) * plane_pixels;
    const int64_t width = feature.size(3);

    int64_t pixel = 0;
    for (; pixel + PIXEL_TILE <= pixels; pixel += PIXEL_TILE) {
        pixel_tile_kernel(input + pixel * INPUT_CHANNELS, input + (pixel + 1) * INPUT_CHANNELS,
                          input + (pixel + 2) * INPUT_CHANNELS, input + (pixel + 3) * INPUT_CHANNELS,
                          input + (pixel + 4) * INPUT_CHANNELS, weight,
                          phase_bias(bias, pixel, plane_pixels, width),
                          phase_bias(bias, pixel + 1, plane_pixels, width),
                          phase_bias(bias, pixel + 2, plane_pixels, width),
                          phase_bias(bias, pixel + 3, plane_pixels, width),
                          phase_bias(bias, pixel + 4, plane_pixels, width),
                          output_pointer + pixel * OUTPUT_CHANNELS,
                          output_pointer + (pixel + 1) * OUTPUT_CHANNELS,
                          output_pointer + (pixel + 2) * OUTPUT_CHANNELS,
                          output_pointer + (pixel + 3) * OUTPUT_CHANNELS,
                          output_pointer + (pixel + 4) * OUTPUT_CHANNELS);
    }
    for (; pixel < pixels; ++pixel) {
        pixel_kernel(input + pixel * INPUT_CHANNELS, weight,
                     phase_bias(bias, pixel, plane_pixels, width),
                     output_pointer + pixel * OUTPUT_CHANNELS);
    }
    return output;
}
