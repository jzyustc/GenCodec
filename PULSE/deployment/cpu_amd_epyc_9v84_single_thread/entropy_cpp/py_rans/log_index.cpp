// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "log_index.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PULSE_X86_64 1
#else
#define PULSE_X86_64 0
#endif

#if PULSE_X86_64 && (defined(__GNUC__) || defined(__clang__))
#define PULSE_AVX512_VNNI_TARGET                                                \
    __attribute__((target(                                                     \
        "avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))
#else
#define PULSE_AVX512_VNNI_TARGET
#endif

namespace {

constexpr int kFracBits = 8;
constexpr int kAffineShift = 10;
constexpr int kTotalShift = kFracBits + kAffineShift;
constexpr int kRounding = 1 << (kAffineShift - 1);
constexpr int kScaleLevels = 64;
constexpr int kInputBlock = 4;
constexpr int kOutputBlock = 8;
constexpr int kOutputVectors = 4;
constexpr int kTileChannels = kOutputBlock * kOutputVectors;
constexpr int kPackedVectorBytes = kOutputBlock * kInputBlock;

template <typename T>
std::vector<T> copy_array(
    const py::array_t<T>& array,
    const std::vector<py::ssize_t>& expected,
    const char* name)
{
    const py::buffer_info info = array.request();
    if (info.ndim != static_cast<py::ssize_t>(expected.size())) {
        throw std::runtime_error(
            std::string(name) + " has the wrong rank");
    }
    size_t count = 1;
    for (size_t axis = 0; axis < expected.size(); ++axis) {
        if (info.shape[axis] != expected[axis]) {
            throw std::runtime_error(
                std::string(name) + " has the wrong shape");
        }
        count *= static_cast<size_t>(expected[axis]);
    }
    const T* source = static_cast<const T*>(info.ptr);
    return std::vector<T>(source, source + count);
}

int64_t round_divide_nearest_even(const int64_t value, const int64_t divisor)
{
    if (divisor <= 0) {
        throw std::runtime_error("integer divisor must be positive");
    }
    const uint64_t magnitude = value < 0
        ? static_cast<uint64_t>(-(value + 1)) + 1U
        : static_cast<uint64_t>(value);
    uint64_t quotient = magnitude / static_cast<uint64_t>(divisor);
    const uint64_t remainder =
        magnitude - quotient * static_cast<uint64_t>(divisor);
    const uint64_t twice = remainder * 2U;
    if (twice > static_cast<uint64_t>(divisor)
        || (twice == static_cast<uint64_t>(divisor)
            && (quotient & 1U) != 0U)) {
        ++quotient;
    }
    const int64_t rounded = static_cast<int64_t>(quotient);
    return value < 0 ? -rounded : rounded;
}

int floor_shift(const int64_t value, const int shift)
{
    const int64_t denominator = int64_t{ 1 } << shift;
    if (value >= 0) {
        return static_cast<int>(value / denominator);
    }
    return static_cast<int>(
        -(((-value) + denominator - 1) / denominator));
}

bool has_avx512_vnni()
{
#if PULSE_X86_64 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl")
        && __builtin_cpu_supports("avx512dq")
        && __builtin_cpu_supports("avx512vnni");
#else
    return false;
#endif
}

#if PULSE_X86_64
template <int M, int ZChannels>
PULSE_AVX512_VNNI_TARGET
void packed_indexes_vnni_fixed(
    const uint8_t* biased_nhwc,
    const int8_t* packed_weight,
    const int32_t* correction,
    const int32_t* multiplier,
    const int32_t* affine_bias,
    const int height,
    const int width,
    uint8_t* output,
    const int threads)
{
    static_assert(M % kTileChannels == 0);
    static_assert(ZChannels % kInputBlock == 0);
    constexpr int up_factor = 4;
    constexpr int pre_channels = M * up_factor * up_factor;
    constexpr int input_blocks = ZChannels / kInputBlock;
    constexpr int tiles = pre_channels / kTileChannels;
    constexpr int input_unroll = 4;
    constexpr int packed_tile_stride =
        input_blocks * kOutputVectors * kPackedVectorBytes;

    const int sites = height * width;
    const int output_height = height * up_factor;
    const int output_width = width * up_factor;
    const int channel_half = M / 2;
    const size_t packed_length = static_cast<size_t>(
        M * output_height * output_width / 2);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i maximum =
        _mm512_set1_epi32(kScaleLevels - 1);
    const __m512i rounding = _mm512_set1_epi32(kRounding);

#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int tile = 0; tile < tiles; ++tile) {
        const int8_t* tile_weight =
            packed_weight
            + static_cast<size_t>(tile * packed_tile_stride);
        __m256i correction_vectors[kOutputVectors];
#pragma GCC unroll 4
        for (int vector = 0;
             vector < kOutputVectors;
             ++vector) {
            correction_vectors[vector] = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    correction
                    + tile * kTileChannels
                    + vector * kOutputBlock));
        }

        const int tile_row = tile * kTileChannels;
        size_t output_fixed_offset[2] = {};
#pragma GCC unroll 2
        for (int pair = 0; pair < 2; ++pair) {
            const int packed_row =
                tile_row + pair * 2 * kOutputBlock;
            const int subpixel = packed_row / M;
            const int channel = packed_row % M;
            const int sub_y = subpixel / up_factor;
            const int sub_x = subpixel % up_factor;
            const bool spatial_even =
                ((sub_y + sub_x) & 1) == 0;
            const bool first_channel_half =
                channel < channel_half;
            const int part =
                first_channel_half == spatial_even ? 0 : 1;
            output_fixed_offset[pair] =
                static_cast<size_t>(part) * packed_length
                + static_cast<size_t>(
                    (sub_y * output_width + sub_x) * channel_half
                    + channel % channel_half);
        }

        for (int site_base = 0;
             site_base < sites;
             site_base += 2) {
            const int valid_sites =
                std::min(2, sites - site_base);
            __m256i accumulators[2][kOutputVectors];
#pragma GCC unroll 2
            for (int lane = 0; lane < 2; ++lane) {
#pragma GCC unroll 4
                for (int vector = 0;
                     vector < kOutputVectors;
                     ++vector) {
                    accumulators[lane][vector] =
                        _mm256_setzero_si256();
                }
            }

            int block = 0;
            for (;
                 block + input_unroll <= input_blocks;
                 block += input_unroll) {
#pragma GCC unroll 4
                for (int input_lane = 0;
                     input_lane < input_unroll;
                     ++input_lane) {
                    const int current_block = block + input_lane;
                    __m256i weights[kOutputVectors];
#pragma GCC unroll 4
                    for (int vector = 0;
                         vector < kOutputVectors;
                         ++vector) {
                        weights[vector] = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(
                                tile_weight
                                + static_cast<size_t>(
                                    (current_block * kOutputVectors
                                     + vector)
                                    * kPackedVectorBytes)));
                    }
#pragma GCC unroll 2
                    for (int lane = 0;
                         lane < valid_sites;
                         ++lane) {
                        uint32_t input_word = 0;
                        std::memcpy(
                            &input_word,
                            biased_nhwc
                                + static_cast<size_t>(
                                    (site_base + lane) * ZChannels
                                    + current_block * kInputBlock),
                            sizeof(input_word));
                        const __m256i broadcast =
                            _mm256_set1_epi32(
                                static_cast<int32_t>(input_word));
#pragma GCC unroll 4
                        for (int vector = 0;
                             vector < kOutputVectors;
                             ++vector) {
                            accumulators[lane][vector] =
                                _mm256_dpbusd_epi32(
                                    accumulators[lane][vector],
                                    broadcast,
                                    weights[vector]);
                        }
                    }
                }
            }
            for (; block < input_blocks; ++block) {
                __m256i weights[kOutputVectors];
#pragma GCC unroll 4
                for (int vector = 0;
                     vector < kOutputVectors;
                     ++vector) {
                    weights[vector] = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            tile_weight
                            + static_cast<size_t>(
                                (block * kOutputVectors + vector)
                                * kPackedVectorBytes)));
                }
#pragma GCC unroll 2
                for (int lane = 0;
                     lane < valid_sites;
                     ++lane) {
                    uint32_t input_word = 0;
                    std::memcpy(
                        &input_word,
                        biased_nhwc
                            + static_cast<size_t>(
                                (site_base + lane) * ZChannels
                                + block * kInputBlock),
                        sizeof(input_word));
                    const __m256i broadcast =
                        _mm256_set1_epi32(
                            static_cast<int32_t>(input_word));
#pragma GCC unroll 4
                    for (int vector = 0;
                         vector < kOutputVectors;
                         ++vector) {
                        accumulators[lane][vector] =
                            _mm256_dpbusd_epi32(
                                accumulators[lane][vector],
                                broadcast,
                                weights[vector]);
                    }
                }
            }

#pragma GCC unroll 2
            for (int lane = 0; lane < valid_sites; ++lane) {
                const int site = site_base + lane;
                const int z_y = site / width;
                const int z_x = site % width;
                const size_t site_offset = static_cast<size_t>(
                    (z_y * up_factor * output_width
                     + z_x * up_factor)
                    * channel_half);
#pragma GCC unroll 2
                for (int pair = 0; pair < 2; ++pair) {
                    const int vector = pair * 2;
                    const __m256i low = _mm256_add_epi32(
                        accumulators[lane][vector],
                        correction_vectors[vector]);
                    const __m256i high = _mm256_add_epi32(
                        accumulators[lane][vector + 1],
                        correction_vectors[vector + 1]);
                    __m512i merged = _mm512_castsi256_si512(low);
                    merged = _mm512_inserti64x4(
                        merged,
                        high,
                        1);
                    const int packed_row =
                        tile_row + pair * 2 * kOutputBlock;
                    __m512i numerator = _mm512_add_epi32(
                        _mm512_add_epi32(
                            _mm512_mullo_epi32(
                                merged,
                                _mm512_loadu_si512(
                                    static_cast<const void*>(
                                        multiplier + packed_row))),
                            _mm512_loadu_si512(
                                static_cast<const void*>(
                                    affine_bias + packed_row))),
                        rounding);
                    __m512i bucket = _mm512_srai_epi32(
                        numerator,
                        kTotalShift);
                    bucket = _mm512_max_epi32(bucket, zero);
                    bucket = _mm512_min_epi32(bucket, maximum);
                    const __m128i packed =
                        _mm512_cvtepi32_epi8(bucket);
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(
                            output
                            + site_offset
                            + output_fixed_offset[pair]),
                        packed);
                }
            }
        }
    }
}
#endif

size_t source_row(
    const size_t packed_row,
    const size_t output_channels,
    const size_t subpixels)
{
    const size_t subpixel = packed_row / output_channels;
    const size_t channel = packed_row % output_channels;
    return channel * subpixels + subpixel;
}

}  // namespace

LogIndexDecoder::LogIndexDecoder(
    const int m,
    const int z_channels,
    const int qp_num,
    const int up_factor,
    const int input_clip,
    const py::array_t<int32_t>& input_divisor,
    const py::array_t<int8_t>& weight,
    const py::array_t<int32_t>& multiplier,
    const py::array_t<int32_t>& affine_bias)
    : m_m(m),
      m_z_channels(z_channels),
      m_qp_num(qp_num),
      m_up_factor(up_factor),
      m_input_clip(input_clip),
      m_pre_channels(m * up_factor * up_factor),
      m_use_vnni(has_avx512_vnni())
{
    if (m_m <= 0 || m_m % kTileChannels != 0) {
        throw std::runtime_error(
            "M must be positive and divisible by 32");
    }
    if (m_z_channels <= 0 || m_z_channels % kInputBlock != 0) {
        throw std::runtime_error(
            "z_channels must be positive and divisible by 4");
    }
    if (m_qp_num <= 0) {
        throw std::runtime_error("qp_num must be positive");
    }
    if (m_up_factor != 4) {
        throw std::runtime_error("up_factor must be 4");
    }
    if (m_input_clip <= 0 || m_input_clip > 127) {
        throw std::runtime_error("input_clip must be in [1, 127]");
    }

    m_input_divisor = copy_array<int32_t>(
        input_divisor,
        { m_z_channels },
        "input_divisor");
    for (const int32_t divisor : m_input_divisor) {
        if (divisor <= 0) {
            throw std::runtime_error(
                "input_divisor values must be positive");
        }
    }
    const std::vector<int8_t> original_weight = copy_array<int8_t>(
        weight,
        { m_pre_channels, m_z_channels },
        "weight");
    const std::vector<int32_t> original_multiplier =
        copy_array<int32_t>(
            multiplier,
            { m_pre_channels },
            "multiplier");
    const std::vector<int32_t> original_bias = copy_array<int32_t>(
        affine_bias,
        { m_qp_num, m_pre_channels },
        "affine_bias");

    const size_t subpixels =
        static_cast<size_t>(m_up_factor * m_up_factor);
    m_reordered_weight.resize(
        static_cast<size_t>(m_pre_channels * m_z_channels));
    m_correction.resize(static_cast<size_t>(m_pre_channels));
    m_multiplier.resize(static_cast<size_t>(m_pre_channels));
    m_affine_bias.resize(
        static_cast<size_t>(m_qp_num * m_pre_channels));

    for (int packed_row = 0; packed_row < m_pre_channels; ++packed_row) {
        const size_t original = source_row(
            static_cast<size_t>(packed_row),
            static_cast<size_t>(m_m),
            subpixels);
        int32_t sum = 0;
        for (int channel = 0; channel < m_z_channels; ++channel) {
            const int8_t value = original_weight[
                original * static_cast<size_t>(m_z_channels)
                + static_cast<size_t>(channel)];
            m_reordered_weight[
                static_cast<size_t>(packed_row * m_z_channels + channel)] =
                value;
            sum += static_cast<int32_t>(value);
        }
        m_correction[static_cast<size_t>(packed_row)] = -128 * sum;
        m_multiplier[static_cast<size_t>(packed_row)] =
            original_multiplier[original];
        for (int qp = 0; qp < m_qp_num; ++qp) {
            m_affine_bias[
                static_cast<size_t>(qp * m_pre_channels + packed_row)] =
                original_bias[
                    static_cast<size_t>(qp * m_pre_channels)
                    + original];
        }
    }

    const int input_blocks = m_z_channels / kInputBlock;
    const int tiles = m_pre_channels / kTileChannels;
    m_packed_weight.resize(
        static_cast<size_t>(m_pre_channels * m_z_channels));
    for (int tile = 0; tile < tiles; ++tile) {
        for (int block = 0; block < input_blocks; ++block) {
            for (int vector = 0; vector < kOutputVectors; ++vector) {
                int8_t* destination =
                    m_packed_weight.data()
                    + static_cast<size_t>(
                        ((tile * input_blocks + block)
                             * kOutputVectors
                         + vector)
                        * kPackedVectorBytes);
                for (int lane = 0; lane < kOutputBlock; ++lane) {
                    const int packed_row =
                        tile * kTileChannels
                        + vector * kOutputBlock + lane;
                    std::memcpy(
                        destination + lane * kInputBlock,
                        m_reordered_weight.data()
                            + static_cast<size_t>(
                                packed_row * m_z_channels
                                + block * kInputBlock),
                        kInputBlock);
                }
            }
        }
    }
}

void LogIndexDecoder::quantize_input(
    const int16_t* z_hat,
    const int height,
    const int width,
    std::vector<uint8_t>& biased_nhwc) const
{
    const int sites = height * width;
    biased_nhwc.resize(
        static_cast<size_t>(sites * m_z_channels));
    for (int channel = 0; channel < m_z_channels; ++channel) {
        const int32_t divisor =
            m_input_divisor[static_cast<size_t>(channel)];
        for (int site = 0; site < sites; ++site) {
            const int64_t rounded = round_divide_nearest_even(
                z_hat[channel * sites + site],
                divisor);
            const int8_t quantized = static_cast<int8_t>(
                std::clamp<int64_t>(
                    rounded,
                    -m_input_clip,
                    m_input_clip));
            biased_nhwc[
                static_cast<size_t>(site * m_z_channels + channel)] =
                static_cast<uint8_t>(quantized) ^ 0x80U;
        }
    }
}

void LogIndexDecoder::packed_indexes_scalar(
    const uint8_t* biased_nhwc,
    const int height,
    const int width,
    const int qp,
    uint8_t* output) const
{
    const int sites = height * width;
    const int output_height = height * m_up_factor;
    const int output_width = width * m_up_factor;
    const int channel_half = m_m / 2;
    const size_t packed_length = static_cast<size_t>(
        m_m * output_height * output_width / 2);
    for (int site = 0; site < sites; ++site) {
        const int z_y = site / width;
        const int z_x = site % width;
        for (int packed_row = 0;
             packed_row < m_pre_channels;
             ++packed_row) {
            int64_t accumulator =
                m_correction[static_cast<size_t>(packed_row)];
            const int8_t* row =
                m_reordered_weight.data()
                + static_cast<size_t>(packed_row * m_z_channels);
            const uint8_t* input =
                biased_nhwc
                + static_cast<size_t>(site * m_z_channels);
            for (int channel = 0;
                 channel < m_z_channels;
                 ++channel) {
                accumulator +=
                    static_cast<int64_t>(input[channel])
                    * static_cast<int64_t>(row[channel]);
            }
            const int64_t numerator =
                accumulator
                    * m_multiplier[static_cast<size_t>(packed_row)]
                + m_affine_bias[
                    static_cast<size_t>(
                        qp * m_pre_channels + packed_row)]
                + kRounding;
            const uint8_t bucket = static_cast<uint8_t>(
                std::clamp(
                    floor_shift(numerator, kTotalShift),
                    0,
                    kScaleLevels - 1));

            const int subpixel = packed_row / m_m;
            const int channel = packed_row % m_m;
            const int sub_y = subpixel / m_up_factor;
            const int sub_x = subpixel % m_up_factor;
            const int out_y = z_y * m_up_factor + sub_y;
            const int out_x = z_x * m_up_factor + sub_x;
            const bool spatial_even =
                ((out_y + out_x) & 1) == 0;
            const bool first_channel_half =
                channel < channel_half;
            const int part =
                first_channel_half == spatial_even ? 0 : 1;
            const size_t offset =
                static_cast<size_t>(part) * packed_length
                + static_cast<size_t>(
                    (out_y * output_width + out_x) * channel_half
                    + channel % channel_half);
            output[offset] = bucket;
        }
    }
}

#if PULSE_X86_64
PULSE_AVX512_VNNI_TARGET
void LogIndexDecoder::packed_indexes_vnni(
    const uint8_t* biased_nhwc,
    const int height,
    const int width,
    const int qp,
    uint8_t* output,
    const int threads) const
{
#if PULSE_X86_64
    const int32_t* qp_bias =
        m_affine_bias.data() + qp * m_pre_channels;
    if (m_m == 256 && m_z_channels == 80) {
        packed_indexes_vnni_fixed<256, 80>(
            biased_nhwc,
            m_packed_weight.data(),
            m_correction.data(),
            m_multiplier.data(),
            qp_bias,
            height,
            width,
            output,
            threads);
        return;
    }
    if (m_m == 320 && m_z_channels == 96) {
        packed_indexes_vnni_fixed<320, 96>(
            biased_nhwc,
            m_packed_weight.data(),
            m_correction.data(),
            m_multiplier.data(),
            qp_bias,
            height,
            width,
            output,
            threads);
        return;
    }
    if (m_m == 320 && m_z_channels == 136) {
        packed_indexes_vnni_fixed<320, 136>(
            biased_nhwc,
            m_packed_weight.data(),
            m_correction.data(),
            m_multiplier.data(),
            qp_bias,
            height,
            width,
            output,
            threads);
        return;
    }
#endif

    const int sites = height * width;
    const int output_height = height * m_up_factor;
    const int output_width = width * m_up_factor;
    const int channel_half = m_m / 2;
    const size_t packed_length = static_cast<size_t>(
        m_m * output_height * output_width / 2);
    const int input_blocks = m_z_channels / kInputBlock;
    const int tiles = m_pre_channels / kTileChannels;
    const size_t packed_tile_stride = static_cast<size_t>(
        input_blocks * kOutputVectors * kPackedVectorBytes);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i maximum =
        _mm512_set1_epi32(kScaleLevels - 1);
    const __m512i rounding = _mm512_set1_epi32(kRounding);

#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int tile = 0; tile < tiles; ++tile) {
        const int8_t* tile_weight =
            m_packed_weight.data()
            + static_cast<size_t>(tile) * packed_tile_stride;
        __m256i correction[kOutputVectors];
        for (int vector = 0;
             vector < kOutputVectors;
             ++vector) {
            correction[vector] = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    m_correction.data()
                    + tile * kTileChannels
                    + vector * kOutputBlock));
        }

        size_t output_fixed_offset[2] = {};
        const int tile_row = tile * kTileChannels;
        for (int pair = 0; pair < 2; ++pair) {
            const int packed_row =
                tile_row + pair * 2 * kOutputBlock;
            const int subpixel = packed_row / m_m;
            const int channel = packed_row % m_m;
            const int sub_y = subpixel / m_up_factor;
            const int sub_x = subpixel % m_up_factor;
            const bool spatial_even =
                ((sub_y + sub_x) & 1) == 0;
            const bool first_channel_half =
                channel < channel_half;
            const int part =
                first_channel_half == spatial_even ? 0 : 1;
            output_fixed_offset[pair] =
                static_cast<size_t>(part) * packed_length
                + static_cast<size_t>(
                    (sub_y * output_width + sub_x) * channel_half
                    + channel % channel_half);
        }

        for (int site_base = 0;
             site_base < sites;
             site_base += 2) {
            const int valid_sites =
                std::min(2, sites - site_base);
            __m256i accumulator[2][kOutputVectors];
            for (int lane = 0; lane < valid_sites; ++lane) {
                for (int vector = 0;
                     vector < kOutputVectors;
                     ++vector) {
                    accumulator[lane][vector] =
                        _mm256_setzero_si256();
                }
            }

            for (int block = 0;
                 block < input_blocks;
                 ++block) {
                __m256i weights[kOutputVectors];
                for (int vector = 0;
                     vector < kOutputVectors;
                     ++vector) {
                    weights[vector] = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(
                            tile_weight
                            + static_cast<size_t>(
                                (block * kOutputVectors + vector)
                                * kPackedVectorBytes)));
                }
                for (int lane = 0; lane < valid_sites; ++lane) {
                    uint32_t input_word = 0;
                    std::memcpy(
                        &input_word,
                        biased_nhwc
                            + static_cast<size_t>(
                                (site_base + lane) * m_z_channels
                                + block * kInputBlock),
                        sizeof(input_word));
                    const __m256i broadcast =
                        _mm256_set1_epi32(
                            static_cast<int32_t>(input_word));
                    for (int vector = 0;
                         vector < kOutputVectors;
                         ++vector) {
                        accumulator[lane][vector] =
                            _mm256_dpbusd_epi32(
                                accumulator[lane][vector],
                                broadcast,
                                weights[vector]);
                    }
                }
            }

            for (int lane = 0; lane < valid_sites; ++lane) {
                const int site = site_base + lane;
                const int z_y = site / width;
                const int z_x = site % width;
                const size_t site_offset = static_cast<size_t>(
                    (z_y * m_up_factor * output_width
                     + z_x * m_up_factor)
                    * channel_half);
                for (int pair = 0; pair < 2; ++pair) {
                    const int vector = pair * 2;
                    const __m256i low = _mm256_add_epi32(
                        accumulator[lane][vector],
                        correction[vector]);
                    const __m256i high = _mm256_add_epi32(
                        accumulator[lane][vector + 1],
                        correction[vector + 1]);
                    __m512i merged = _mm512_castsi256_si512(low);
                    merged = _mm512_inserti64x4(
                        merged,
                        high,
                        1);
                    const int packed_row =
                        tile_row + pair * 2 * kOutputBlock;
                    __m512i numerator = _mm512_add_epi32(
                        _mm512_add_epi32(
                            _mm512_mullo_epi32(
                                merged,
                                _mm512_loadu_si512(
                                    static_cast<const void*>(
                                        m_multiplier.data()
                                        + packed_row))),
                            _mm512_loadu_si512(
                                static_cast<const void*>(
                                    m_affine_bias.data()
                                    + qp * m_pre_channels
                                    + packed_row))),
                        rounding);
                    __m512i bucket = _mm512_srai_epi32(
                        numerator,
                        kTotalShift);
                    bucket = _mm512_max_epi32(bucket, zero);
                    bucket = _mm512_min_epi32(bucket, maximum);
                    const __m128i packed =
                        _mm512_cvtepi32_epi8(bucket);
                    _mm_storeu_si128(
                        reinterpret_cast<__m128i*>(
                            output
                            + site_offset
                            + output_fixed_offset[pair]),
                        packed);
                }
            }
        }
    }
}
#else
void LogIndexDecoder::packed_indexes_vnni(
    const uint8_t*,
    int,
    int,
    int,
    uint8_t*,
    int) const
{
    throw std::runtime_error(
        "AVX-512 VNNI is unavailable on this architecture");
}
#endif

py::array_t<uint8_t> LogIndexDecoder::packed_indexes_impl(
    const py::array_t<int16_t>& z_hat,
    const int qp,
    const bool allow_vnni,
    const int threads) const
{
    if (threads < 1 || threads > 80) {
        throw std::runtime_error("threads must be in [1, 80]");
    }
    if (qp < 0 || qp >= m_qp_num) {
        throw std::runtime_error("qp is outside the trained range");
    }
    const py::buffer_info z_info = z_hat.request();
    if (z_info.ndim != 4
        || z_info.shape[0] != 1
        || z_info.shape[1] != m_z_channels
        || z_info.shape[2] <= 0
        || z_info.shape[3] <= 0) {
        throw std::runtime_error(
            "z_hat must have shape [1, z_channels, H, W]");
    }
    const int height = static_cast<int>(z_info.shape[2]);
    const int width = static_cast<int>(z_info.shape[3]);
    const int output_height = height * m_up_factor;
    const int output_width = width * m_up_factor;
    const py::ssize_t packed_length = static_cast<py::ssize_t>(
        m_m * output_height * output_width / 2);
    py::array_t<uint8_t> result(
        py::array::ShapeContainer{
            py::ssize_t{ 2 },
            packed_length,
        });
    packed_indexes_into_impl(
        z_hat,
        qp,
        result,
        allow_vnni,
        threads);
    return result;
}

void LogIndexDecoder::packed_indexes_into_impl(
    const py::array_t<int16_t>& z_hat,
    const int qp,
    const py::array_t<uint8_t>& result,
    const bool allow_vnni,
    const int threads) const
{
    if (threads < 1 || threads > 80) {
        throw std::runtime_error("threads must be in [1, 80]");
    }
    if (qp < 0 || qp >= m_qp_num) {
        throw std::runtime_error("qp is outside the trained range");
    }
    const py::buffer_info z_info = z_hat.request();
    if (z_info.ndim != 4
        || z_info.shape[0] != 1
        || z_info.shape[1] != m_z_channels
        || z_info.shape[2] <= 0
        || z_info.shape[3] <= 0) {
        throw std::runtime_error(
            "z_hat must have shape [1, z_channels, H, W]");
    }
    const int height = static_cast<int>(z_info.shape[2]);
    const int width = static_cast<int>(z_info.shape[3]);
    const int output_height = height * m_up_factor;
    const int output_width = width * m_up_factor;
    const py::ssize_t packed_length = static_cast<py::ssize_t>(
        m_m * output_height * output_width / 2);
    const py::buffer_info result_info = result.request();
    if (result_info.ndim != 2
        || result_info.shape[0] != 2
        || result_info.shape[1] != packed_length) {
        throw std::runtime_error(
            "output must have shape [2, packed_length]");
    }
    uint8_t* output = static_cast<uint8_t*>(result_info.ptr);
    std::vector<uint8_t> biased_nhwc;
    {
        py::gil_scoped_release release;
        quantize_input(
            static_cast<const int16_t*>(z_info.ptr),
            height,
            width,
            biased_nhwc);
        if (allow_vnni && m_use_vnni) {
            packed_indexes_vnni(
                biased_nhwc.data(),
                height,
                width,
                qp,
                output,
                threads);
        } else {
            packed_indexes_scalar(
                biased_nhwc.data(),
                height,
                width,
                qp,
                output);
        }
    }
}

py::array_t<uint8_t> LogIndexDecoder::packed_indexes(
    const py::array_t<int16_t>& z_hat,
    const int qp) const
{
    return packed_indexes_impl(z_hat, qp, true, 1);
}

py::array_t<uint8_t> LogIndexDecoder::packed_indexes_mt(
    const py::array_t<int16_t>& z_hat,
    const int qp,
    const int threads) const
{
    return packed_indexes_impl(z_hat, qp, true, threads);
}

void LogIndexDecoder::packed_indexes_into(
    const py::array_t<int16_t>& z_hat,
    const int qp,
    const py::array_t<uint8_t>& output,
    const int threads) const
{
    packed_indexes_into_impl(z_hat, qp, output, true, threads);
}

void LogIndexDecoder::packed_symbols_into(
    const py::array_t<int16_t>& z_hat,
    const py::array_t<int8_t>& symbols0,
    const py::array_t<int8_t>& symbols1,
    const int qp,
    const py::array_t<int16_t>& output0,
    const py::array_t<int16_t>& output1,
    const int threads)
{
    if (threads < 1 || threads > 80) {
        throw std::runtime_error("threads must be in [1, 80]");
    }
    if (qp < 0 || qp >= m_qp_num) {
        throw std::runtime_error("qp is outside the trained range");
    }
    const py::buffer_info z_info = z_hat.request();
    if (z_info.ndim != 4
        || z_info.shape[0] != 1
        || z_info.shape[1] != m_z_channels
        || z_info.shape[2] <= 0
        || z_info.shape[3] <= 0) {
        throw std::runtime_error(
            "z_hat must have shape [1, z_channels, H, W]");
    }
    const int height = static_cast<int>(z_info.shape[2]);
    const int width = static_cast<int>(z_info.shape[3]);
    const py::ssize_t packed_length = static_cast<py::ssize_t>(
        m_m * height * m_up_factor * width * m_up_factor / 2);
    if (symbols0.size() != packed_length
        || symbols1.size() != packed_length
        || output0.size() != packed_length
        || output1.size() != packed_length) {
        throw std::runtime_error(
            "compact symbol/output sizes do not match z shape");
    }
    const py::buffer_info symbols0_info = symbols0.request();
    const py::buffer_info symbols1_info = symbols1.request();
    const py::buffer_info output0_info = output0.request();
    const py::buffer_info output1_info = output1.request();
    const auto* symbol0_ptr =
        static_cast<const int8_t*>(symbols0_info.ptr);
    const auto* symbol1_ptr =
        static_cast<const int8_t*>(symbols1_info.ptr);
    auto* output0_ptr = static_cast<int16_t*>(output0_info.ptr);
    auto* output1_ptr = static_cast<int16_t*>(output1_info.ptr);
    m_index_workspace.resize(
        static_cast<size_t>(2 * packed_length));
    std::vector<uint8_t> biased_nhwc;
    {
        py::gil_scoped_release release;
        quantize_input(
            static_cast<const int16_t*>(z_info.ptr),
            height,
            width,
            biased_nhwc);
        if (m_use_vnni) {
            packed_indexes_vnni(
                biased_nhwc.data(),
                height,
                width,
                qp,
                m_index_workspace.data(),
                threads);
        } else {
            packed_indexes_scalar(
                biased_nhwc.data(),
                height,
                width,
                qp,
                m_index_workspace.data());
        }
        const auto* index0_ptr = m_index_workspace.data();
        const auto* index1_ptr =
            m_index_workspace.data() + packed_length;
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
        for (py::ssize_t i = 0; i < packed_length; ++i) {
            output0_ptr[i] = static_cast<int16_t>(
                (static_cast<uint16_t>(
                    static_cast<uint8_t>(symbol0_ptr[i]))
                 << 8)
                | static_cast<uint16_t>(index0_ptr[i]));
            output1_ptr[i] = static_cast<int16_t>(
                (static_cast<uint16_t>(
                    static_cast<uint8_t>(symbol1_ptr[i]))
                 << 8)
                | static_cast<uint16_t>(index1_ptr[i]));
        }
    }
}

py::array_t<uint8_t> LogIndexDecoder::packed_indexes_reference(
    const py::array_t<int16_t>& z_hat,
    const int qp) const
{
    return packed_indexes_impl(z_hat, qp, false, 1);
}

bool LogIndexDecoder::uses_vnni() const
{
    return m_use_vnni;
}
