#include "dense_conv1x1_gated_residual_bf16.h"

#include "activations_avx512.h"

#include <cstdint>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "dense_conv1x1_gated_residual_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int CHANNELS = 48;
static constexpr int CHANNEL_BLOCK = 16;
static constexpr int INPUT_PAIRS = CHANNELS / 2;
static constexpr int OUTPUT_BLOCKS = CHANNELS / CHANNEL_BLOCK;
static constexpr int PIXEL_TILE = 5;

using AliasedUint32 = uint32_t __attribute__((may_alias));

// BF16((residual + hard_sigmoid(gate) * convolution) * output_scale).
static inline void store_output(__m512 convolution, const at::BFloat16* gate,
                                const at::BFloat16* residual, const at::BFloat16* output_scale,
                                at::BFloat16* output)
{
    const __m512 gate_value =
        _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(gate)));
    const __m512 residual_value =
        _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(residual)));
    const __m512 scale =
        _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(output_scale)));
    const __m512 gated_residual =
        _mm512_fmadd_ps(hard_sigmoid(gate_value), convolution, residual_value);
    convolution = _mm512_mul_ps(gated_residual, scale);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output), (__m256i)_mm512_cvtneps_pbh(convolution));
}

static inline void pixel_kernel(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                                const at::BFloat16* bias, const at::BFloat16* gate,
                                const at::BFloat16* residual, const at::BFloat16* output_scale,
                                at::BFloat16* output)
{
    __m512 acc0 =
        _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias)));
    __m512 acc1 = _mm512_cvtpbh_ps(
        (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + CHANNEL_BLOCK)));
    __m512 acc2 = _mm512_cvtpbh_ps(
        (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + 2 * CHANNEL_BLOCK)));
    const auto* input_pairs = reinterpret_cast<const AliasedUint32*>(feature);
    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        const __m512bh input_vector = (__m512bh)_mm512_set1_epi32(input_pairs[pair]);
        const at::BFloat16* weights = packed_weight + pair * OUTPUT_BLOCKS * 32;
        acc0 = _mm512_dpbf16_ps(acc0, input_vector, (__m512bh)_mm512_loadu_si512(weights));
        acc1 = _mm512_dpbf16_ps(acc1, input_vector, (__m512bh)_mm512_loadu_si512(weights + 32));
        acc2 = _mm512_dpbf16_ps(acc2, input_vector, (__m512bh)_mm512_loadu_si512(weights + 64));
    }
    store_output(acc0, gate, residual, output_scale, output);
    store_output(acc1, gate + CHANNEL_BLOCK, residual + CHANNEL_BLOCK, output_scale + CHANNEL_BLOCK,
                 output + CHANNEL_BLOCK);
    store_output(acc2, gate + 2 * CHANNEL_BLOCK, residual + 2 * CHANNEL_BLOCK,
                 output_scale + 2 * CHANNEL_BLOCK, output + 2 * CHANNEL_BLOCK);
}

static inline void pixel_tile_kernel(
    const at::BFloat16* feature0, const at::BFloat16* feature1, const at::BFloat16* feature2,
    const at::BFloat16* feature3, const at::BFloat16* feature4, const at::BFloat16* packed_weight,
    const at::BFloat16* bias, const at::BFloat16* gate0, const at::BFloat16* gate1,
    const at::BFloat16* gate2, const at::BFloat16* gate3, const at::BFloat16* gate4,
    const at::BFloat16* residual0, const at::BFloat16* residual1, const at::BFloat16* residual2,
    const at::BFloat16* residual3, const at::BFloat16* residual4, const at::BFloat16* output_scale,
    at::BFloat16* output0, at::BFloat16* output1, at::BFloat16* output2, at::BFloat16* output3,
    at::BFloat16* output4)
{
    const __m512 bias0 =
        _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias)));
    const __m512 bias1 = _mm512_cvtpbh_ps(
        (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + CHANNEL_BLOCK)));
    const __m512 bias2 = _mm512_cvtpbh_ps(
        (__m256bh)_mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias + 2 * CHANNEL_BLOCK)));
    __m512 acc00 = bias0, acc01 = bias1, acc02 = bias2;
    __m512 acc10 = bias0, acc11 = bias1, acc12 = bias2;
    __m512 acc20 = bias0, acc21 = bias1, acc22 = bias2;
    __m512 acc30 = bias0, acc31 = bias1, acc32 = bias2;
    __m512 acc40 = bias0, acc41 = bias1, acc42 = bias2;
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
        const __m512bh weight2 = (__m512bh)_mm512_loadu_si512(weights + 64);
#define UPDATE_PIXEL(PIXEL)                                                 \
    acc##PIXEL##0 = _mm512_dpbf16_ps(acc##PIXEL##0, input##PIXEL, weight0); \
    acc##PIXEL##1 = _mm512_dpbf16_ps(acc##PIXEL##1, input##PIXEL, weight1); \
    acc##PIXEL##2 = _mm512_dpbf16_ps(acc##PIXEL##2, input##PIXEL, weight2)
        UPDATE_PIXEL(0);
        UPDATE_PIXEL(1);
        UPDATE_PIXEL(2);
        UPDATE_PIXEL(3);
        UPDATE_PIXEL(4);
#undef UPDATE_PIXEL
    }

#define STORE_PIXEL(PIXEL)                                                                            \
    store_output(acc##PIXEL##0, gate##PIXEL, residual##PIXEL, output_scale, output##PIXEL);           \
    store_output(acc##PIXEL##1, gate##PIXEL + CHANNEL_BLOCK, residual##PIXEL + CHANNEL_BLOCK,         \
                 output_scale + CHANNEL_BLOCK, output##PIXEL + CHANNEL_BLOCK);                        \
    store_output(acc##PIXEL##2, gate##PIXEL + 2 * CHANNEL_BLOCK, residual##PIXEL + 2 * CHANNEL_BLOCK, \
                 output_scale + 2 * CHANNEL_BLOCK, output##PIXEL + 2 * CHANNEL_BLOCK)
    STORE_PIXEL(0);
    STORE_PIXEL(1);
    STORE_PIXEL(2);
    STORE_PIXEL(3);
    STORE_PIXEL(4);
#undef STORE_PIXEL
}

at::Tensor dense_conv1x1_gated_residual_scale_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias,
    const at::Tensor& gate, const at::Tensor& residual, const at::Tensor& output_scale)
{
    TORCH_CHECK(feature.device().is_cpu() && feature.scalar_type() == at::kBFloat16
                    && feature.dim() == 4 && feature.size(1) == CHANNELS
                    && feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feature must be channels-last CPU BF16 [N, 48, H, W]");
    TORCH_CHECK(gate.sizes() == feature.sizes() && residual.sizes() == feature.sizes()
                    && gate.device().is_cpu() && residual.device().is_cpu()
                    && gate.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16
                    && gate.is_contiguous(at::MemoryFormat::ChannelsLast)
                    && residual.is_contiguous(at::MemoryFormat::ChannelsLast),
                "gate and residual must be channels-last CPU BF16 matching feature");
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.sizes() == at::IntArrayRef({ INPUT_PAIRS, OUTPUT_BLOCKS, 32 })
                    && packed_weight.is_contiguous(),
                "packed_weight must be contiguous BF16 [24, 3, 32]");
    TORCH_CHECK(bias.device().is_cpu() && bias.scalar_type() == at::kBFloat16
                    && bias.sizes() == at::IntArrayRef({ CHANNELS }) && bias.is_contiguous(),
                "bias must be contiguous CPU BF16 [48]");
    TORCH_CHECK(output_scale.device().is_cpu() && output_scale.scalar_type() == at::kBFloat16
                    && output_scale.sizes() == at::IntArrayRef({ 1, CHANNELS, 1, 1 })
                    && output_scale.is_contiguous(),
                "output_scale must be contiguous CPU BF16 [1, 48, 1, 1]");

    at::Tensor output = at::empty_like(feature, feature.options(), at::MemoryFormat::ChannelsLast);
    const auto* input = feature.data_ptr<at::BFloat16>();
    const auto* weight = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_pointer = bias.data_ptr<at::BFloat16>();
    const auto* gate_pointer = gate.data_ptr<at::BFloat16>();
    const auto* residual_pointer = residual.data_ptr<at::BFloat16>();
    const auto* output_scale_pointer = output_scale.data_ptr<at::BFloat16>();
    auto* output_pointer = output.data_ptr<at::BFloat16>();
    const int64_t pixels = feature.numel() / CHANNELS;

    int64_t pixel = 0;
    for (; pixel + PIXEL_TILE <= pixels; pixel += PIXEL_TILE) {
        pixel_tile_kernel(
            input + pixel * CHANNELS, input + (pixel + 1) * CHANNELS, input + (pixel + 2) * CHANNELS,
            input + (pixel + 3) * CHANNELS, input + (pixel + 4) * CHANNELS, weight, bias_pointer,
            gate_pointer + pixel * CHANNELS, gate_pointer + (pixel + 1) * CHANNELS,
            gate_pointer + (pixel + 2) * CHANNELS, gate_pointer + (pixel + 3) * CHANNELS,
            gate_pointer + (pixel + 4) * CHANNELS, residual_pointer + pixel * CHANNELS,
            residual_pointer + (pixel + 1) * CHANNELS, residual_pointer + (pixel + 2) * CHANNELS,
            residual_pointer + (pixel + 3) * CHANNELS, residual_pointer + (pixel + 4) * CHANNELS,
            output_scale_pointer, output_pointer + pixel * CHANNELS,
            output_pointer + (pixel + 1) * CHANNELS, output_pointer + (pixel + 2) * CHANNELS,
            output_pointer + (pixel + 3) * CHANNELS, output_pointer + (pixel + 4) * CHANNELS);
    }
    for (; pixel < pixels; ++pixel) {
        pixel_kernel(input + pixel * CHANNELS, weight, bias_pointer,
                     gate_pointer + pixel * CHANNELS, residual_pointer + pixel * CHANNELS,
                     output_scale_pointer, output_pointer + pixel * CHANNELS);
    }
    return output;
}
