// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <depthwise_conv3x3_bf16.h>

#include "activations_avx512.h"

#include <cstdint>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "depthwise_conv3x3_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int CHANNEL_BLOCK = 16;
static constexpr int PIXEL_TILE = 12;

#define FOR_EACH_PIXEL(OP) \
    OP(0)                  \
    OP(1)                  \
    OP(2)                  \
    OP(3)                  \
    OP(4)                  \
    OP(5)                  \
    OP(6)                  \
    OP(7)                  \
    OP(8)                  \
    OP(9)                  \
    OP(10)                 \
    OP(11)

#define DISPATCH_CHANNELS(CHANNELS)                                                            \
    if (channels == CHANNELS) {                                                                \
        depthwise_conv3x3_specialized<CHANNELS>(feature_ptr, weight_ptr, bias_ptr, output_ptr, \
                                                mean_ptr, batch, height, width);               \
        return { output, mean };                                                               \
    }

#define INIT_PIXEL(PIXEL) __m512 acc##PIXEL = bias_vector;

#define LOAD_INPUT(INPUT)                                                                \
    const at::BFloat16* input_ptr##INPUT = row + (x0 + INPUT - 1) * CHANNELS + channel0; \
    const __m256i input_bits##INPUT = load_bf16<MASKED>(input_ptr##INPUT, mask);         \
    const __m512 input##INPUT = _mm512_cvtpbh_ps((__m256bh)input_bits##INPUT);

#define FMA(OUTPUT, INPUT, WEIGHT) \
    acc##OUTPUT = _mm512_fmadd_ps(input##INPUT, weight##WEIGHT, acc##OUTPUT);

#define STORE_PIXEL(PIXEL)                                                                         \
    const __m512 output_value##PIXEL = hard_gelu_030(acc##PIXEL);                                  \
    const __m256i output_bits##PIXEL = (__m256i)_mm512_cvtneps_pbh(output_value##PIXEL);           \
    store_bf16<MASKED>(output_row + (x0 + PIXEL) * CHANNELS + channel0, mask, output_bits##PIXEL); \
    channel_sum = _mm512_add_ps(channel_sum, output_value##PIXEL);

template <int CHANNELS>
static inline __mmask16 channel_mask(int channel0)
{
    constexpr int tail = CHANNELS % CHANNEL_BLOCK;
    if constexpr (tail == 0) {
        return static_cast<__mmask16>(0xffff);
    }
    return channel0 + CHANNEL_BLOCK <= CHANNELS ? static_cast<__mmask16>(0xffff)
                                                : static_cast<__mmask16>((1u << tail) - 1u);
}

template <bool MASKED>
static inline __m256i load_bf16(const at::BFloat16* input, __mmask16 mask)
{
    if constexpr (MASKED) {
        return _mm256_maskz_loadu_epi16(mask, input);
    }
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
}

template <bool MASKED>
static inline void store_bf16(at::BFloat16* output, __mmask16 mask, __m256i bits)
{
    if constexpr (MASKED) {
        _mm256_mask_storeu_epi16(output, mask, bits);
    } else {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), bits);
    }
}

template <bool MASKED>
static inline __m512 load_bias(const at::BFloat16* bias, int channel0, __mmask16 mask)
{
    const __m256i bits = load_bf16<MASKED>(bias + channel0, mask);
    return _mm512_cvtpbh_ps((__m256bh)bits);
}

template <int CHANNELS, bool MASKED>
static inline void depthwise_conv3x3_pixel(const at::BFloat16* feature, const float* packed_weight,
                                           const at::BFloat16* bias, at::BFloat16* output,
                                           int64_t batch_index, int64_t y, int64_t x, int64_t height,
                                           int64_t width, int channel0, __m512& channel_sum)
{
    constexpr int channel_blocks = (CHANNELS + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    const __mmask16 mask = channel_mask<CHANNELS>(channel0);
    __m512 acc = load_bias<MASKED>(bias, channel0, mask);

    for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
        const int64_t input_y = y + kernel_y - 1;
        if (input_y < 0 || input_y >= height) {
            continue;
        }
        for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
            const int64_t input_x = x + kernel_x - 1;
            if (input_x < 0 || input_x >= width) {
                continue;
            }
            const at::BFloat16* input =
                feature + ((batch_index * height + input_y) * width + input_x) * CHANNELS + channel0;
            const float* weight =
                packed_weight + ((kernel_y * 3 + kernel_x) * channel_blocks) * CHANNEL_BLOCK + channel0;
            const __m256i input_bits = load_bf16<MASKED>(input, mask);
            acc = _mm512_fmadd_ps(_mm512_cvtpbh_ps((__m256bh)input_bits), _mm512_loadu_ps(weight), acc);
        }
    }

    at::BFloat16* out = output + ((batch_index * height + y) * width + x) * CHANNELS + channel0;
    const __m512 output_value = hard_gelu_030(acc);
    const __m256i output_bits = (__m256i)_mm512_cvtneps_pbh(output_value);
    store_bf16<MASKED>(out, mask, output_bits);
    channel_sum = _mm512_add_ps(channel_sum, output_value);
}

template <int CHANNELS, bool MASKED>
static inline void depthwise_conv3x3_pixel_tile(const at::BFloat16* feature,
                                                const float* packed_weight, const at::BFloat16* bias,
                                                at::BFloat16* output, int64_t batch_index,
                                                int64_t y, int64_t x0, int64_t height,
                                                int64_t width, int channel0, __m512& channel_sum)
{
    constexpr int channel_blocks = (CHANNELS + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    const __mmask16 mask = channel_mask<CHANNELS>(channel0);
    const __m512 bias_vector = load_bias<MASKED>(bias, channel0, mask);
    FOR_EACH_PIXEL(INIT_PIXEL)

    for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
        const int64_t input_y = y + kernel_y - 1;
        if (input_y < 0 || input_y >= height) {
            continue;
        }
        const at::BFloat16* row = feature + ((batch_index * height + input_y) * width) * CHANNELS;
        const float* weight_ptr0 =
            packed_weight + (kernel_y * 3 * channel_blocks) * CHANNEL_BLOCK + channel0;
        const float* weight_ptr1 = weight_ptr0 + channel_blocks * CHANNEL_BLOCK;
        const float* weight_ptr2 = weight_ptr1 + channel_blocks * CHANNEL_BLOCK;
        const __m512 weight0 = _mm512_loadu_ps(weight_ptr0);
        const __m512 weight1 = _mm512_loadu_ps(weight_ptr1);
        const __m512 weight2 = _mm512_loadu_ps(weight_ptr2);

        LOAD_INPUT(0)
        LOAD_INPUT(1)
        LOAD_INPUT(2)
        LOAD_INPUT(3)
        LOAD_INPUT(4)
        LOAD_INPUT(5)
        LOAD_INPUT(6)
        LOAD_INPUT(7)
        LOAD_INPUT(8)
        LOAD_INPUT(9)
        LOAD_INPUT(10)
        LOAD_INPUT(11)
        LOAD_INPUT(12)
        LOAD_INPUT(13)

        FMA(0, 0, 0)
        FMA(0, 1, 1)
        FMA(0, 2, 2)
        FMA(1, 1, 0)
        FMA(1, 2, 1)
        FMA(1, 3, 2)
        FMA(2, 2, 0)
        FMA(2, 3, 1)
        FMA(2, 4, 2)
        FMA(3, 3, 0)
        FMA(3, 4, 1)
        FMA(3, 5, 2)
        FMA(4, 4, 0)
        FMA(4, 5, 1)
        FMA(4, 6, 2)
        FMA(5, 5, 0)
        FMA(5, 6, 1)
        FMA(5, 7, 2)
        FMA(6, 6, 0)
        FMA(6, 7, 1)
        FMA(6, 8, 2)
        FMA(7, 7, 0)
        FMA(7, 8, 1)
        FMA(7, 9, 2)
        FMA(8, 8, 0)
        FMA(8, 9, 1)
        FMA(8, 10, 2)
        FMA(9, 9, 0)
        FMA(9, 10, 1)
        FMA(9, 11, 2)
        FMA(10, 10, 0)
        FMA(10, 11, 1)
        FMA(10, 12, 2)
        FMA(11, 11, 0)
        FMA(11, 12, 1)
        FMA(11, 13, 2)
    }

    at::BFloat16* output_row = output + ((batch_index * height + y) * width) * CHANNELS;
    FOR_EACH_PIXEL(STORE_PIXEL)
}

template <int CHANNELS, bool MASKED>
static inline void depthwise_conv3x3_channel_block(const at::BFloat16* feature,
                                                   const float* packed_weight,
                                                   const at::BFloat16* bias, at::BFloat16* output,
                                                   int64_t batch_index, int64_t y, int64_t height,
                                                   int64_t width, int channel0, __m512& channel_sum)
{
    int64_t x = 0;
    if (width > 0) {
        depthwise_conv3x3_pixel<CHANNELS, MASKED>(feature, packed_weight, bias, output, batch_index,
                                                  y, 0, height, width, channel0, channel_sum);
        x = 1;
    }
    for (; x + PIXEL_TILE <= width - 1; x += PIXEL_TILE) {
        depthwise_conv3x3_pixel_tile<CHANNELS, MASKED>(feature, packed_weight, bias, output, batch_index,
                                                       y, x, height, width, channel0, channel_sum);
    }
    for (; x < width; ++x) {
        depthwise_conv3x3_pixel<CHANNELS, MASKED>(feature, packed_weight, bias, output, batch_index,
                                                  y, x, height, width, channel0, channel_sum);
    }
}

template <int CHANNELS>
static void depthwise_conv3x3_specialized(const at::BFloat16* feature, const float* packed_weight,
                                          const at::BFloat16* bias, at::BFloat16* output,
                                          at::BFloat16* mean, int64_t batch, int64_t height,
                                          int64_t width)
{
    constexpr int full_blocks = CHANNELS / CHANNEL_BLOCK;
    constexpr int tail = CHANNELS % CHANNEL_BLOCK;
    constexpr int channel_blocks = full_blocks + (tail != 0);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        __m512 channel_sums[channel_blocks];
        for (int block = 0; block < channel_blocks; ++block) {
            channel_sums[block] = _mm512_setzero_ps();
        }
        for (int64_t y = 0; y < height; ++y) {
            for (int block = 0; block < full_blocks; ++block) {
                depthwise_conv3x3_channel_block<CHANNELS, false>(
                    feature, packed_weight, bias, output, batch_index, y, height, width,
                    block * CHANNEL_BLOCK, channel_sums[block]);
            }
            if constexpr (tail != 0) {
                depthwise_conv3x3_channel_block<CHANNELS, true>(
                    feature, packed_weight, bias, output, batch_index, y, height, width,
                    full_blocks * CHANNEL_BLOCK, channel_sums[full_blocks]);
            }
        }
        const __m512 reciprocal_pixels = _mm512_set1_ps(1.0f / static_cast<float>(height * width));
        for (int block = 0; block < full_blocks; ++block) {
            store_bf16<false>(
                mean + batch_index * CHANNELS + block * CHANNEL_BLOCK, static_cast<__mmask16>(0xffff),
                (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(channel_sums[block], reciprocal_pixels)));
        }
        if constexpr (tail != 0) {
            store_bf16<true>(mean + batch_index * CHANNELS + full_blocks * CHANNEL_BLOCK,
                             channel_mask<CHANNELS>(full_blocks * CHANNEL_BLOCK),
                             (__m256i)_mm512_cvtneps_pbh(
                                 _mm512_mul_ps(channel_sums[full_blocks], reciprocal_pixels)));
        }
    }
}

static std::tuple<at::Tensor, at::Tensor>
depthwise_conv3x3_hard_gelu_030_mean_impl(const at::Tensor& feature,
                                          const at::Tensor& packed_weight, const at::Tensor& bias)
{
    TORCH_CHECK(feature.device().is_cpu(), "feature must be a CPU tensor");
    TORCH_CHECK(feature.scalar_type() == at::kBFloat16, "feature must be BF16");
    TORCH_CHECK(feature.dim() == 4, "feature must have shape [N, C, H, W]");
    TORCH_CHECK(feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feature must be channels-last contiguous");

    TORCH_CHECK(packed_weight.device().is_cpu(), "packed_weight must be a CPU tensor");
    TORCH_CHECK(packed_weight.scalar_type() == at::kFloat, "packed_weight must be FP32");
    TORCH_CHECK(packed_weight.dim() == 3 && packed_weight.size(0) == 9
                    && packed_weight.size(2) == CHANNEL_BLOCK,
                "packed_weight must have shape [9, ceil(C/16), 16]");
    TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");

    TORCH_CHECK(bias.device().is_cpu(), "bias must be a CPU tensor");
    TORCH_CHECK(bias.scalar_type() == at::kBFloat16, "bias must be BF16");
    TORCH_CHECK(bias.dim() == 1, "bias must have shape [C]");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    const int64_t batch = feature.size(0);
    const int64_t channels = feature.size(1);
    const int64_t height = feature.size(2);
    const int64_t width = feature.size(3);
    const int64_t channel_blocks = (channels + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    TORCH_CHECK(packed_weight.size(1) == channel_blocks,
                "packed_weight channel-block dimension does not match feature channels");
    TORCH_CHECK(bias.size(0) == channels, "bias channels do not match feature channels");

    at::Tensor output =
        at::empty(feature.sizes(), feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    at::Tensor mean = at::empty({ batch, channels, 1, 1 },
                                feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* weight_ptr = packed_weight.data_ptr<float>();
    const auto* bias_ptr = bias.data_ptr<at::BFloat16>();
    auto* output_ptr = output.data_ptr<at::BFloat16>();
    auto* mean_ptr = mean.data_ptr<at::BFloat16>();

    DISPATCH_CHANNELS(72)
    DISPATCH_CHANNELS(96)
    DISPATCH_CHANNELS(100)
    DISPATCH_CHANNELS(128)
    DISPATCH_CHANNELS(144)
    DISPATCH_CHANNELS(148)
    DISPATCH_CHANNELS(256)
    DISPATCH_CHANNELS(288)
    TORCH_CHECK(false, "unsupported depthwise 3x3 channels: ", channels);
}

std::tuple<at::Tensor, at::Tensor> depthwise_conv3x3_bias_hard_gelu_030_mean_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias)
{
    std::tuple<at::Tensor, at::Tensor> result =
        depthwise_conv3x3_hard_gelu_030_mean_impl(feature, packed_weight, bias);
    return result;
}
