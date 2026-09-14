#include "grouped_conv1x1_shuffle_bf16.h"

#include <cstdint>

#if !defined(__AVX512F__) || !defined(__AVX512BF16__)
    #error "grouped_conv1x1_shuffle_bf16.cpp requires AVX-512F and AVX-512 BF16"
#endif

#include <immintrin.h>

static constexpr int GROUPS = 4;
static constexpr int CHANNEL_BLOCK = 16;
static constexpr int PIXEL_TILE = 5;
static constexpr int OUTPUT_BLOCK_TILE = 5;

#ifndef SHUFFLED_OUTPUT_BLOCK_TILE
    #define SHUFFLED_OUTPUT_BLOCK_TILE 0
#endif
static_assert(SHUFFLED_OUTPUT_BLOCK_TILE >= 0 && SHUFFLED_OUTPUT_BLOCK_TILE <= 5);

using AliasedUint16 = uint16_t __attribute__((may_alias));

template <int GROUP_INPUT_CHANNELS>
static inline uint32_t load_input_pair(const at::BFloat16* input, int pair)
{
    const int channel = pair * 2;
    if (channel + 1 < GROUP_INPUT_CHANNELS) {
        uint32_t bits;
        __builtin_memcpy(&bits, input + channel, sizeof(bits));
        return bits;
    }
    return static_cast<uint32_t>(*reinterpret_cast<const AliasedUint16*>(input + channel));
}

template <int GROUP_INPUT_CHANNELS>
static inline __m512bh load_group_input_pairs(const at::BFloat16* input, int pair)
{
    const uint32_t pair0 = load_input_pair<GROUP_INPUT_CHANNELS>(input, pair);
    const uint32_t pair1 = load_input_pair<GROUP_INPUT_CHANNELS>(input + GROUP_INPUT_CHANNELS, pair);
    const uint32_t pair2 =
        load_input_pair<GROUP_INPUT_CHANNELS>(input + 2 * GROUP_INPUT_CHANNELS, pair);
    const uint32_t pair3 =
        load_input_pair<GROUP_INPUT_CHANNELS>(input + 3 * GROUP_INPUT_CHANNELS, pair);
    return (__m512bh)_mm512_set_epi32(pair3, pair2, pair1, pair0, pair3, pair2, pair1, pair0, pair3,
                                      pair2, pair1, pair0, pair3, pair2, pair1, pair0);
}

template <int OUTPUT_CHANNELS>
static inline __mmask16 output_mask(int64_t output0)
{
    const int64_t remaining = OUTPUT_CHANNELS - output0;
    return remaining >= CHANNEL_BLOCK ? static_cast<__mmask16>(0xffff)
                                      : static_cast<__mmask16>((1u << remaining) - 1u);
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

template <int OUTPUT_CHANNELS, bool HAS_BIAS, bool MASKED>
static inline __m512 initialize_accumulator(const at::BFloat16* bias, int64_t output0, __mmask16 mask)
{
    if constexpr (!HAS_BIAS) {
        return _mm512_setzero_ps();
    }
    return _mm512_cvtpbh_ps((__m256bh)load_bf16<MASKED>(bias + output0, mask));
}

template <int OUTPUT_CHANNELS, bool HAS_RESIDUAL, bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, bool MASKED>
static inline void finalize_and_store(__m512 accumulator, const at::BFloat16* residual,
                                      const at::BFloat16* output_scale, at::BFloat16* output,
                                      int64_t output0, __mmask16 mask)
{
    static_assert(!(HAS_SCALAR_SCALE && HAS_CHANNEL_SCALE));
    __m512 result = accumulator;
    if constexpr (HAS_RESIDUAL) {
        const __m512 residual_value =
            _mm512_cvtpbh_ps((__m256bh)load_bf16<MASKED>(residual + output0, mask));
        result = _mm512_add_ps(result, residual_value);
    }
    if constexpr (HAS_SCALAR_SCALE || HAS_CHANNEL_SCALE) {
        __m512 scale;
        if constexpr (HAS_SCALAR_SCALE) {
            scale = _mm512_set1_ps(static_cast<float>(output_scale[0]));
        } else {
            scale = _mm512_cvtpbh_ps((__m256bh)load_bf16<MASKED>(output_scale + output0, mask));
        }
        result = _mm512_mul_ps(result, scale);
    }
    store_bf16<MASKED>(output + output0, mask, (__m256i)_mm512_cvtneps_pbh(result));
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, int ACTIVE_BLOCKS, bool HAS_BIAS,
          bool HAS_RESIDUAL, bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE, int ACTIVE_PIXELS>
static inline void
grouped_conv1x1_shuffle_tile(const at::BFloat16* feature, const at::BFloat16* packed_weight,
                             const at::BFloat16* bias, const at::BFloat16* residual,
                             const at::BFloat16* output_scale, at::BFloat16* output, int64_t pixel0,
                             int64_t output_block0)
{
    constexpr int GROUP_INPUT_CHANNELS = INPUT_CHANNELS / GROUPS;
    constexpr int INPUT_PAIRS = (GROUP_INPUT_CHANNELS + 1) / 2;
    constexpr int OUTPUT_BLOCKS = (OUTPUT_CHANNELS + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;

    __m512 accumulators[PIXEL_TILE][OUTPUT_BLOCK_TILE];
    for (int pixel = 0; pixel < ACTIVE_PIXELS; ++pixel) {
        for (int block = 0; block < ACTIVE_BLOCKS; ++block) {
            const int64_t output0 = (output_block0 + block) * CHANNEL_BLOCK;
            const __mmask16 mask = output_mask<OUTPUT_CHANNELS>(output0);
            if (mask == static_cast<__mmask16>(0xffff)) {
                accumulators[pixel][block] =
                    initialize_accumulator<OUTPUT_CHANNELS, HAS_BIAS, false>(bias, output0, mask);
            } else {
                accumulators[pixel][block] =
                    initialize_accumulator<OUTPUT_CHANNELS, HAS_BIAS, true>(bias, output0, mask);
            }
        }
    }

    for (int pair = 0; pair < INPUT_PAIRS; ++pair) {
        __m512bh input_vectors[PIXEL_TILE];
        for (int pixel = 0; pixel < ACTIVE_PIXELS; ++pixel) {
            input_vectors[pixel] = load_group_input_pairs<GROUP_INPUT_CHANNELS>(
                feature + (pixel0 + pixel) * INPUT_CHANNELS, pair);
        }
        for (int block = 0; block < ACTIVE_BLOCKS; ++block) {
            const __m512bh weight_vector = (__m512bh)_mm512_loadu_si512(
                packed_weight + (pair * OUTPUT_BLOCKS + output_block0 + block) * 32);
            for (int pixel = 0; pixel < ACTIVE_PIXELS; ++pixel) {
                accumulators[pixel][block] =
                    _mm512_dpbf16_ps(accumulators[pixel][block], input_vectors[pixel], weight_vector);
            }
        }
    }

    for (int pixel = 0; pixel < ACTIVE_PIXELS; ++pixel) {
        at::BFloat16* pixel_output = output + (pixel0 + pixel) * OUTPUT_CHANNELS;
        const at::BFloat16* pixel_residual =
            HAS_RESIDUAL ? residual + (pixel0 + pixel) * OUTPUT_CHANNELS : nullptr;
        for (int block = 0; block < ACTIVE_BLOCKS; ++block) {
            const int64_t output0 = (output_block0 + block) * CHANNEL_BLOCK;
            const __mmask16 mask = output_mask<OUTPUT_CHANNELS>(output0);
            if (mask == static_cast<__mmask16>(0xffff)) {
                finalize_and_store<OUTPUT_CHANNELS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, false>(
                    accumulators[pixel][block], pixel_residual, output_scale, pixel_output, output0,
                    mask);
            } else {
                finalize_and_store<OUTPUT_CHANNELS, HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, true>(
                    accumulators[pixel][block], pixel_residual, output_scale, pixel_output, output0,
                    mask);
            }
        }
    }
}

template <int INPUT_CHANNELS, int OUTPUT_CHANNELS, bool HAS_BIAS, bool HAS_RESIDUAL,
          bool HAS_SCALAR_SCALE, bool HAS_CHANNEL_SCALE>
static void grouped_conv1x1_shuffle_specialized(const at::BFloat16* feature,
                                                const at::BFloat16* packed_weight,
                                                const at::BFloat16* bias, const at::BFloat16* residual,
                                                const at::BFloat16* output_scale,
                                                at::BFloat16* output, int64_t pixels)
{
    constexpr int OUTPUT_BLOCKS = (OUTPUT_CHANNELS + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    constexpr int BLOCK_TILE =
        SHUFFLED_OUTPUT_BLOCK_TILE > 0
            ? SHUFFLED_OUTPUT_BLOCK_TILE
            : (OUTPUT_CHANNELS == 96 || (INPUT_CHANNELS == 100 && OUTPUT_CHANNELS == 100) ? 4 : 5);
    for (int64_t output_block0 = 0; output_block0 < OUTPUT_BLOCKS; output_block0 += BLOCK_TILE) {
        const int active_blocks = std::min<int64_t>(BLOCK_TILE, OUTPUT_BLOCKS - output_block0);
        int64_t pixel = 0;
        for (; pixel + PIXEL_TILE <= pixels; pixel += PIXEL_TILE) {
            switch (active_blocks) {
            case 5:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 5, HAS_BIAS, HAS_RESIDUAL,
                                             HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, PIXEL_TILE>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            case 4:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 4, HAS_BIAS, HAS_RESIDUAL,
                                             HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, PIXEL_TILE>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            case 3:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 3, HAS_BIAS, HAS_RESIDUAL,
                                             HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, PIXEL_TILE>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            case 2:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 2, HAS_BIAS, HAS_RESIDUAL,
                                             HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, PIXEL_TILE>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            default:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 1, HAS_BIAS, HAS_RESIDUAL,
                                             HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, PIXEL_TILE>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            }
        }
        for (; pixel < pixels; ++pixel) {
            switch (active_blocks) {
            case 5:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 5, HAS_BIAS,
                                             HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, 1>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            case 4:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 4, HAS_BIAS,
                                             HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, 1>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            case 3:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 3, HAS_BIAS,
                                             HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, 1>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            case 2:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 2, HAS_BIAS,
                                             HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, 1>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            default:
                grouped_conv1x1_shuffle_tile<INPUT_CHANNELS, OUTPUT_CHANNELS, 1, HAS_BIAS,
                                             HAS_RESIDUAL, HAS_SCALAR_SCALE, HAS_CHANNEL_SCALE, 1>(
                    feature, packed_weight, bias, residual, output_scale, output, pixel, output_block0);
                break;
            }
        }
    }
}

#define DISPATCH_SCALE(CIN, COUT, HAS_BIAS, HAS_RESIDUAL)                                           \
    if (has_scalar_scale) {                                                                         \
        grouped_conv1x1_shuffle_specialized<CIN, COUT, HAS_BIAS, HAS_RESIDUAL, true, false>(        \
            feature_ptr, weight_ptr, bias_ptr, residual_ptr, output_scale_ptr, output_ptr, pixels); \
    } else if (has_channel_scale) {                                                                 \
        grouped_conv1x1_shuffle_specialized<CIN, COUT, HAS_BIAS, HAS_RESIDUAL, false, true>(        \
            feature_ptr, weight_ptr, bias_ptr, residual_ptr, output_scale_ptr, output_ptr, pixels); \
    } else {                                                                                        \
        grouped_conv1x1_shuffle_specialized<CIN, COUT, HAS_BIAS, HAS_RESIDUAL, false, false>(       \
            feature_ptr, weight_ptr, bias_ptr, residual_ptr, output_scale_ptr, output_ptr, pixels); \
    }

#define DISPATCH_PAIR(CIN, COUT)                            \
    if (input_channels == CIN && output_channels == COUT) { \
        if (has_bias) {                                     \
            if (has_residual) {                             \
                DISPATCH_SCALE(CIN, COUT, true, true)       \
            } else {                                        \
                DISPATCH_SCALE(CIN, COUT, true, false)      \
            }                                               \
        } else if (has_residual) {                          \
            DISPATCH_SCALE(CIN, COUT, false, true)          \
        } else {                                            \
            DISPATCH_SCALE(CIN, COUT, false, false)         \
        }                                                   \
        return output;                                      \
    }

at::Tensor grouped_conv1x1_shuffle_bias_avx512_bf16(const at::Tensor& feature,
                                                    const at::Tensor& packed_weight,
                                                    const at::optional<at::Tensor>& bias,
                                                    const at::optional<at::Tensor>& residual,
                                                    const at::optional<at::Tensor>& output_scale)
{
    TORCH_CHECK(feature.device().is_cpu() && feature.scalar_type() == at::kBFloat16,
                "feature must be a CPU BF16 tensor");
    TORCH_CHECK(feature.dim() == 4 && feature.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feature must be channels-last [N, C, H, W]");
    TORCH_CHECK(packed_weight.device().is_cpu() && packed_weight.scalar_type() == at::kBFloat16
                    && packed_weight.dim() == 3 && packed_weight.size(2) == 32
                    && packed_weight.is_contiguous(),
                "packed_weight must be contiguous BF16 [ceil(Cin/4/2), ceil(Cout/16), 32]");

    const int64_t input_channels = feature.size(1);
    TORCH_CHECK(bias.has_value() || residual.has_value(),
                "bias or residual is required to infer output channels");
    const int64_t output_channels = bias.has_value() ? bias->size(0) : residual->size(1);
    const int64_t group_input_channels = input_channels / GROUPS;
    const int64_t input_pairs = (group_input_channels + 1) / 2;
    const int64_t output_blocks = (output_channels + CHANNEL_BLOCK - 1) / CHANNEL_BLOCK;
    TORCH_CHECK(input_channels % GROUPS == 0 && output_channels % GROUPS == 0,
                "input and output channels must be divisible by 4");
    TORCH_CHECK(packed_weight.size(0) == input_pairs && packed_weight.size(1) == output_blocks,
                "packed_weight dimensions do not match channels");

    const bool has_bias = bias.has_value();
    if (has_bias) {
        TORCH_CHECK(bias->device().is_cpu() && bias->scalar_type() == at::kBFloat16
                        && bias->dim() == 1 && bias->is_contiguous(),
                    "bias must be contiguous CPU BF16 [Cout]");
    }
    const bool has_residual = residual.has_value();
    if (has_residual) {
        TORCH_CHECK(residual->sizes()
                            == at::IntArrayRef({ feature.size(0), output_channels, feature.size(2),
                                                 feature.size(3) })
                        && residual->device().is_cpu() && residual->scalar_type() == at::kBFloat16
                        && residual->is_contiguous(at::MemoryFormat::ChannelsLast),
                    "residual must be channels-last CPU BF16 and match output shape");
    }
    const bool has_output_scale = output_scale.has_value();
    if (has_output_scale) {
        TORCH_CHECK(output_scale->device().is_cpu() && output_scale->scalar_type() == at::kBFloat16
                        && output_scale->is_contiguous(),
                    "output_scale must be a contiguous CPU BF16 tensor");
        TORCH_CHECK(output_scale->sizes() == at::IntArrayRef({ 1, 1, 1, 1 })
                        || output_scale->sizes() == at::IntArrayRef({ 1, output_channels, 1, 1 }),
                    "output_scale must have shape [1, 1, 1, 1] or [1, Cout, 1, 1]");
    }
    const bool has_scalar_scale = has_output_scale && output_scale->numel() == 1;
    const bool has_channel_scale = has_output_scale && output_scale->numel() == output_channels;

    at::Tensor output =
        at::empty({ feature.size(0), output_channels, feature.size(2), feature.size(3) },
                  feature.options().memory_format(at::MemoryFormat::ChannelsLast));
    const auto* feature_ptr = feature.data_ptr<at::BFloat16>();
    const auto* weight_ptr = packed_weight.data_ptr<at::BFloat16>();
    const auto* bias_ptr = has_bias ? bias->data_ptr<at::BFloat16>() : nullptr;
    const auto* residual_ptr = has_residual ? residual->data_ptr<at::BFloat16>() : nullptr;
    const auto* output_scale_ptr = has_output_scale ? output_scale->data_ptr<at::BFloat16>() : nullptr;
    auto* output_ptr = output.data_ptr<at::BFloat16>();
    const int64_t pixels = feature.size(0) * feature.size(2) * feature.size(3);

    DISPATCH_PAIR(96, 96)
    DISPATCH_PAIR(100, 100)
    DISPATCH_PAIR(128, 128)
    DISPATCH_PAIR(148, 148)
    DISPATCH_PAIR(192, 96)
    DISPATCH_PAIR(200, 100)
    DISPATCH_PAIR(256, 128)
    DISPATCH_PAIR(296, 148)
    TORCH_CHECK(false, "unsupported shuffled grouped 1x1 channel pair (Cin=", input_channels,
                ", Cout=", output_channels, ")");
}
