// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

namespace pulse::log_index_scalar {

constexpr int kFracBits = 8;
constexpr int kAffineShift = 10;
constexpr int kTotalShift = kFracBits + kAffineShift;
constexpr int kRounding = 1 << (kAffineShift - 1);
constexpr int kScaleLevels = 64;
#ifndef PULSE_I8MM_ROWS
#define PULSE_I8MM_ROWS 16
#endif
constexpr int kI8mmRows = PULSE_I8MM_ROWS;

inline int64_t round_divide_nearest_even(
    const int64_t value,
    const int64_t divisor)
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

inline int floor_shift(const int64_t value, const int shift)
{
    const int64_t denominator = int64_t{1} << shift;
    if (value >= 0) {
        return static_cast<int>(value / denominator);
    }
    return static_cast<int>(
        -(((-value) + denominator - 1) / denominator));
}

inline size_t source_row(
    const size_t packed_row,
    const size_t output_channels,
    const size_t subpixels)
{
    const size_t subpixel = packed_row / output_channels;
    const size_t channel = packed_row % output_channels;
    return channel * subpixels + subpixel;
}

struct PreparedLinearDecoder {
    int m{};
    int z_channels{};
    int up_factor{};
    int input_clip{};
    int pre_channels{};
    std::vector<int32_t> input_divisor;
    std::vector<int8_t> reordered_weight;
    // Packed as [output-row pair, input-channel block, 2, 8] for ARM I8MM.
    // One SMMLA then computes two input sites by two output rows.
    std::vector<int8_t> reordered_weight_i8mm;
    std::vector<int32_t> correction;
    std::vector<int32_t> multiplier;
    std::vector<int32_t> affine_bias;
};

inline PreparedLinearDecoder prepare(
    const int m,
    const int z_channels,
    const int up_factor,
    const int input_clip,
    const std::vector<int32_t>& input_divisor,
    const std::vector<int8_t>& original_weight,
    const std::vector<int32_t>& original_multiplier,
    const std::vector<int32_t>& original_bias)
{
    if (m <= 0 || (m & 1) != 0) {
        throw std::runtime_error("M must be positive and even");
    }
    if (z_channels <= 0) {
        throw std::runtime_error("z_channels must be positive");
    }
    if (up_factor != 4) {
        throw std::runtime_error("up_factor must be 4");
    }
    if (input_clip <= 0 || input_clip > 127) {
        throw std::runtime_error("input_clip must be in [1, 127]");
    }
    const int pre_channels = m * up_factor * up_factor;
    if (input_divisor.size() != static_cast<size_t>(z_channels)
        || original_weight.size()
            != static_cast<size_t>(pre_channels * z_channels)
        || original_multiplier.size() != static_cast<size_t>(pre_channels)
        || original_bias.size() != static_cast<size_t>(pre_channels)) {
        throw std::runtime_error("integer linear decoder array shape mismatch");
    }
    for (const int32_t divisor : input_divisor) {
        if (divisor <= 0) {
            throw std::runtime_error("input divisors must be positive");
        }
    }

    PreparedLinearDecoder result;
    result.m = m;
    result.z_channels = z_channels;
    result.up_factor = up_factor;
    result.input_clip = input_clip;
    result.pre_channels = pre_channels;
    result.input_divisor = input_divisor;
    result.reordered_weight.resize(original_weight.size());
    result.correction.resize(static_cast<size_t>(pre_channels));
    result.multiplier.resize(static_cast<size_t>(pre_channels));
    result.affine_bias.resize(static_cast<size_t>(pre_channels));
    const size_t subpixels =
        static_cast<size_t>(up_factor * up_factor);
    for (int packed_row = 0; packed_row < pre_channels; ++packed_row) {
        const size_t original = source_row(
            static_cast<size_t>(packed_row),
            static_cast<size_t>(m),
            subpixels);
        int32_t sum = 0;
        for (int channel = 0; channel < z_channels; ++channel) {
            const int8_t value = original_weight[
                original * static_cast<size_t>(z_channels)
                + static_cast<size_t>(channel)];
            result.reordered_weight[
                static_cast<size_t>(packed_row * z_channels + channel)] =
                value;
            sum += static_cast<int32_t>(value);
        }
        result.correction[static_cast<size_t>(packed_row)] = -128 * sum;
        result.multiplier[static_cast<size_t>(packed_row)] =
            original_multiplier[original];
        result.affine_bias[static_cast<size_t>(packed_row)] =
            original_bias[original];
    }
    if (kI8mmRows > 0
        && (kI8mmRows & 1) == 0
        && pre_channels % kI8mmRows == 0
        && (z_channels & 7) == 0) {
        const int channel_blocks = z_channels / 8;
        const int row_pairs = kI8mmRows / 2;
        result.reordered_weight_i8mm.resize(original_weight.size());
        for (int row_base = 0;
             row_base < pre_channels;
             row_base += kI8mmRows) {
            for (int block = 0; block < channel_blocks; ++block) {
                for (int row_pair = 0;
                     row_pair < row_pairs;
                     ++row_pair) {
                    int8_t* packed =
                        result.reordered_weight_i8mm.data()
                        + static_cast<size_t>(
                            (((row_base / kI8mmRows) * channel_blocks
                               + block)
                                  * row_pairs
                              + row_pair)
                            * 16);
                    const int channel = block * 8;
                    const int row0 = row_base + row_pair * 2;
                    const int row1 = row0 + 1;
                    std::copy_n(
                        result.reordered_weight.data()
                            + static_cast<size_t>(
                                row0 * z_channels + channel),
                        8,
                        packed);
                    std::copy_n(
                        result.reordered_weight.data()
                            + static_cast<size_t>(
                                row1 * z_channels + channel),
                        8,
                        packed + 8);
                }
            }
        }
    }
    return result;
}

inline size_t packed_length(
    const PreparedLinearDecoder& decoder,
    const int height,
    const int width)
{
    return static_cast<size_t>(
        decoder.m
        * height * decoder.up_factor
        * width * decoder.up_factor
        / 2);
}

inline size_t infer_packed(
    const PreparedLinearDecoder& decoder,
    const int16_t* z_hat_nchw,
    const int height,
    const int width,
    uint8_t* output,
    size_t* saturation_count = nullptr)
{
    if (height <= 0 || width <= 0 || z_hat_nchw == nullptr
        || output == nullptr) {
        throw std::runtime_error("invalid integer linear decoder input");
    }
    const int sites = height * width;
    std::vector<uint8_t> biased_nhwc(
        static_cast<size_t>(sites * decoder.z_channels));
    size_t saturated = 0;
    for (int channel = 0; channel < decoder.z_channels; ++channel) {
        const int32_t divisor =
            decoder.input_divisor[static_cast<size_t>(channel)];
        for (int site = 0; site < sites; ++site) {
            const int64_t rounded = round_divide_nearest_even(
                z_hat_nchw[channel * sites + site],
                divisor);
            saturated += static_cast<size_t>(
                rounded < -decoder.input_clip
                || rounded > decoder.input_clip);
            const int8_t quantized = static_cast<int8_t>(
                std::clamp<int64_t>(
                    rounded,
                    -decoder.input_clip,
                    decoder.input_clip));
            biased_nhwc[
                static_cast<size_t>(
                    site * decoder.z_channels + channel)] =
                static_cast<uint8_t>(quantized) ^ 0x80U;
        }
    }

    const int output_width = width * decoder.up_factor;
    const int channel_half = decoder.m / 2;
    const size_t one_part = packed_length(decoder, height, width);
    for (int site = 0; site < sites; ++site) {
        const int z_y = site / width;
        const int z_x = site % width;
        for (int packed_row = 0;
             packed_row < decoder.pre_channels;
             ++packed_row) {
            int64_t accumulator =
                decoder.correction[static_cast<size_t>(packed_row)];
            const int8_t* row =
                decoder.reordered_weight.data()
                + static_cast<size_t>(
                    packed_row * decoder.z_channels);
            const uint8_t* input =
                biased_nhwc.data()
                + static_cast<size_t>(
                    site * decoder.z_channels);
            for (int channel = 0;
                 channel < decoder.z_channels;
                 ++channel) {
                accumulator +=
                    static_cast<int64_t>(input[channel])
                    * static_cast<int64_t>(row[channel]);
            }
            const int64_t numerator =
                accumulator
                    * decoder.multiplier[
                        static_cast<size_t>(packed_row)]
                + decoder.affine_bias[
                    static_cast<size_t>(packed_row)]
                + kRounding;
            const uint8_t bucket = static_cast<uint8_t>(
                std::clamp(
                    floor_shift(numerator, kTotalShift),
                    0,
                    kScaleLevels - 1));

            const int subpixel = packed_row / decoder.m;
            const int channel = packed_row % decoder.m;
            const int sub_y = subpixel / decoder.up_factor;
            const int sub_x = subpixel % decoder.up_factor;
            const int out_y = z_y * decoder.up_factor + sub_y;
            const int out_x = z_x * decoder.up_factor + sub_x;
            const bool spatial_even =
                ((out_y + out_x) & 1) == 0;
            const bool first_channel_half =
                channel < channel_half;
            const int part =
                first_channel_half == spatial_even ? 0 : 1;
            const size_t offset =
                static_cast<size_t>(part) * one_part
                + static_cast<size_t>(
                    (out_y * output_width + out_x) * channel_half
                    + channel % channel_half);
            output[offset] = bucket;
        }
    }
    if (saturation_count != nullptr) {
        *saturation_count = saturated;
    }
    return one_part;
}

inline void quantize_signed_nhwc(
    const PreparedLinearDecoder& decoder,
    const int16_t* z_hat_nchw,
    const int height,
    const int width,
    std::vector<int8_t>& signed_nhwc,
    size_t& saturation_count)
{
    const int sites = height * width;
    signed_nhwc.resize(
        static_cast<size_t>(sites * decoder.z_channels));
    saturation_count = 0;
    for (int channel = 0; channel < decoder.z_channels; ++channel) {
        const int32_t divisor =
            decoder.input_divisor[static_cast<size_t>(channel)];
        for (int site = 0; site < sites; ++site) {
            const int64_t rounded = round_divide_nearest_even(
                z_hat_nchw[channel * sites + site],
                divisor);
            saturation_count += static_cast<size_t>(
                rounded < -decoder.input_clip
                || rounded > decoder.input_clip);
            signed_nhwc[
                static_cast<size_t>(
                    site * decoder.z_channels + channel)] =
                static_cast<int8_t>(
                    std::clamp<int64_t>(
                        rounded,
                        -decoder.input_clip,
                        decoder.input_clip));
        }
    }
}

inline void store_bucket(
    const PreparedLinearDecoder& decoder,
    const int site,
    const int width,
    const int packed_row,
    const int32_t accumulator,
    const size_t one_part,
    uint8_t* output)
{
    const int64_t numerator =
        static_cast<int64_t>(accumulator)
            * decoder.multiplier[static_cast<size_t>(packed_row)]
        + decoder.affine_bias[static_cast<size_t>(packed_row)]
        + kRounding;
    const uint8_t bucket = static_cast<uint8_t>(
        std::clamp(
            floor_shift(numerator, kTotalShift),
            0,
            kScaleLevels - 1));
    const int z_y = site / width;
    const int z_x = site % width;
    const int output_width = width * decoder.up_factor;
    const int channel_half = decoder.m / 2;
    const int subpixel = packed_row / decoder.m;
    const int channel = packed_row % decoder.m;
    const int sub_y = subpixel / decoder.up_factor;
    const int sub_x = subpixel % decoder.up_factor;
    const int out_y = z_y * decoder.up_factor + sub_y;
    const int out_x = z_x * decoder.up_factor + sub_x;
    const bool spatial_even = ((out_y + out_x) & 1) == 0;
    const bool first_channel_half = channel < channel_half;
    const int part = first_channel_half == spatial_even ? 0 : 1;
    const size_t offset =
        static_cast<size_t>(part) * one_part
        + static_cast<size_t>(
            (out_y * output_width + out_x) * channel_half
            + channel % channel_half);
    output[offset] = bucket;
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
inline void infer_packed_neon_worker(
    const PreparedLinearDecoder& decoder,
    const int8_t* signed_nhwc,
    const int height,
    const int width,
    const int site_begin,
    const int site_end,
    const size_t one_part,
    uint8_t* output)
{
    (void)height;
#ifndef PULSE_DOT_ROWS
#define PULSE_DOT_ROWS 8
#endif
    constexpr int kRows = PULSE_DOT_ROWS;
    static_assert(kRows > 0, "PULSE_DOT_ROWS must be positive");
    for (int site = site_begin; site < site_end; ++site) {
        const int8_t* input =
            signed_nhwc
            + static_cast<size_t>(site * decoder.z_channels);
        for (int row_base = 0;
             row_base < decoder.pre_channels;
             row_base += kRows) {
            const int valid_rows = std::min(
                kRows,
                decoder.pre_channels - row_base);
            int32x4_t accumulators[kRows];
            for (int row = 0; row < valid_rows; ++row) {
                accumulators[row] = vdupq_n_s32(0);
            }
            int channel = 0;
            for (; channel + 16 <= decoder.z_channels; channel += 16) {
                const int8x16_t inputs =
                    vld1q_s8(input + channel);
                for (int row = 0; row < valid_rows; ++row) {
                    const int8_t* weights =
                        decoder.reordered_weight.data()
                        + static_cast<size_t>(
                            (row_base + row) * decoder.z_channels
                            + channel);
                    accumulators[row] = vdotq_s32(
                        accumulators[row],
                        inputs,
                        vld1q_s8(weights));
                }
            }
            for (int row = 0; row < valid_rows; ++row) {
                int32_t accumulator = vaddvq_s32(accumulators[row]);
                const int8_t* weights =
                    decoder.reordered_weight.data()
                    + static_cast<size_t>(
                        (row_base + row) * decoder.z_channels);
                for (int tail = channel;
                     tail < decoder.z_channels;
                     ++tail) {
                    accumulator +=
                        static_cast<int32_t>(input[tail])
                        * static_cast<int32_t>(weights[tail]);
                }
                store_bucket(
                    decoder,
                    site,
                    width,
                    row_base + row,
                    accumulator,
                    one_part,
                    output);
            }
        }
    }
}
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
inline int32x4_t requantize_i8mm_4(
    const int32x4_t values,
    const int32_t* multiplier,
    const int32_t* affine_bias)
{
    const int32x4_t multipliers = vld1q_s32(multiplier);
    const int32x4_t biases = vld1q_s32(affine_bias);
    int64x2_t low = vmull_s32(
        vget_low_s32(values),
        vget_low_s32(multipliers));
    int64x2_t high = vmull_high_s32(values, multipliers);
    low = vaddq_s64(
        low,
        vmovl_s32(vget_low_s32(biases)));
    high = vaddq_s64(
        high,
        vmovl_high_s32(biases));
    low = vaddq_s64(low, vdupq_n_s64(kRounding));
    high = vaddq_s64(high, vdupq_n_s64(kRounding));
    const int32x4_t shifted = vcombine_s32(
        vmovn_s64(vshrq_n_s64(low, kTotalShift)),
        vmovn_s64(vshrq_n_s64(high, kTotalShift)));
    return vmaxq_s32(
        vdupq_n_s32(0),
        vminq_s32(vdupq_n_s32(kScaleLevels - 1), shifted));
}

inline size_t i8mm_output_block_offset(
    const PreparedLinearDecoder& decoder,
    const int site,
    const int width,
    const int packed_row,
    const size_t one_part)
{
    const int z_y = site / width;
    const int z_x = site - z_y * width;
    const int subpixel = packed_row / decoder.m;
    const int channel = packed_row - subpixel * decoder.m;
    const int sub_y = subpixel / decoder.up_factor;
    const int sub_x = subpixel - sub_y * decoder.up_factor;
    const int out_y = z_y * decoder.up_factor + sub_y;
    const int out_x = z_x * decoder.up_factor + sub_x;
    const int channel_half = decoder.m / 2;
    const bool first_channel_half = channel < channel_half;
    const bool spatial_even = ((out_y + out_x) & 1) == 0;
    const int part = first_channel_half == spatial_even ? 0 : 1;
    const int output_width = width * decoder.up_factor;
    return static_cast<size_t>(part) * one_part
        + static_cast<size_t>(
            (out_y * output_width + out_x) * channel_half
            + channel % channel_half);
}

inline void store_i8mm_8(
    const PreparedLinearDecoder& decoder,
    const int32x4_t values0,
    const int32x4_t values1,
    const int site,
    const int width,
    const int packed_row,
    const size_t one_part,
    uint8_t* output)
{
    const int32x4_t buckets0 = requantize_i8mm_4(
        values0,
        decoder.multiplier.data() + packed_row,
        decoder.affine_bias.data() + packed_row);
    const int32x4_t buckets1 = requantize_i8mm_4(
        values1,
        decoder.multiplier.data() + packed_row + 4,
        decoder.affine_bias.data() + packed_row + 4);
    const uint16x8_t narrowed16 = vcombine_u16(
        vmovn_u32(vreinterpretq_u32_s32(buckets0)),
        vmovn_u32(vreinterpretq_u32_s32(buckets1)));
    const uint8x8_t narrowed8 = vmovn_u16(narrowed16);
    vst1_u8(
        output
            + i8mm_output_block_offset(
                decoder,
                site,
                width,
                packed_row,
                one_part),
        narrowed8);
}

inline void infer_packed_i8mm_worker(
    const PreparedLinearDecoder& decoder,
    const int8_t* signed_nhwc,
    const int width,
    const int pair_begin,
    const int pair_end,
    const int sites,
    const size_t one_part,
    uint8_t* output)
{
    constexpr int kRows = kI8mmRows;
    static_assert(kRows >= 8 && (kRows & 7) == 0,
                  "I8MM row tile must be a multiple of eight");
    constexpr int kRowPairs = kRows / 2;
    const int channel_blocks = decoder.z_channels / 8;
    for (int site_pair = pair_begin;
         site_pair < pair_end;
         ++site_pair) {
        const int site0 = site_pair * 2;
        const int site1 = site0 + 1;
        const int8_t* input0 =
            signed_nhwc
            + static_cast<size_t>(site0 * decoder.z_channels);
        const int8_t* input1 = site1 < sites
            ? signed_nhwc
                + static_cast<size_t>(site1 * decoder.z_channels)
            : input0;
        for (int row_base = 0;
             row_base < decoder.pre_channels;
             row_base += kRows) {
            int32x4_t accumulators[kRowPairs];
            for (auto& accumulator : accumulators) {
                accumulator = vdupq_n_s32(0);
            }
            for (int block = 0; block < channel_blocks; ++block) {
                const int channel = block * 8;
                const int8x16_t inputs = vcombine_s8(
                    vld1_s8(input0 + channel),
                    vld1_s8(input1 + channel));
                for (int row_pair = 0;
                     row_pair < kRowPairs;
                     ++row_pair) {
                    const int8_t* weights =
                        decoder.reordered_weight_i8mm.data()
                        + static_cast<size_t>(
                            ((((row_base / kRows) * channel_blocks
                               + block)
                                  * kRowPairs)
                              + row_pair)
                            * 16);
                    accumulators[row_pair] = vmmlaq_s32(
                        accumulators[row_pair],
                        inputs,
                        vld1q_s8(weights));
                }
            }
            for (int row = 0; row < kRows; row += 8) {
                const int pair = row / 2;
                const int32x4_t site0_rows0 = vcombine_s32(
                    vget_low_s32(accumulators[pair]),
                    vget_low_s32(accumulators[pair + 1]));
                const int32x4_t site0_rows1 = vcombine_s32(
                    vget_low_s32(accumulators[pair + 2]),
                    vget_low_s32(accumulators[pair + 3]));
                store_i8mm_8(
                    decoder,
                    site0_rows0,
                    site0_rows1,
                    site0,
                    width,
                    row_base + row,
                    one_part,
                    output);
                if (site1 < sites) {
                    const int32x4_t site1_rows0 = vcombine_s32(
                        vget_high_s32(accumulators[pair]),
                        vget_high_s32(accumulators[pair + 1]));
                    const int32x4_t site1_rows1 = vcombine_s32(
                        vget_high_s32(accumulators[pair + 2]),
                        vget_high_s32(accumulators[pair + 3]));
                    store_i8mm_8(
                        decoder,
                        site1_rows0,
                        site1_rows1,
                        site1,
                        width,
                        row_base + row,
                        one_part,
                        output);
                }
            }
        }
    }
}
#endif

inline size_t infer_packed_i8mm(
    const PreparedLinearDecoder& decoder,
    const int16_t* z_hat_nchw,
    const int height,
    const int width,
    uint8_t* output,
    int thread_count,
    size_t* saturation_count = nullptr)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (height <= 0 || width <= 0 || z_hat_nchw == nullptr
        || output == nullptr) {
        throw std::runtime_error("invalid integer linear decoder input");
    }
    if (decoder.pre_channels % kI8mmRows != 0
        || decoder.m % kI8mmRows != 0
        || (decoder.m / 2) % kI8mmRows != 0
        || (decoder.z_channels & 7) != 0
        || decoder.reordered_weight_i8mm.empty()) {
        throw std::runtime_error(
            "I8MM kernel requires dimensions divisible by eight");
    }
    std::vector<int8_t> signed_nhwc;
    size_t saturated = 0;
    quantize_signed_nhwc(
        decoder,
        z_hat_nchw,
        height,
        width,
        signed_nhwc,
        saturated);
    const int sites = height * width;
    const int site_pairs = (sites + 1) / 2;
    const size_t one_part = packed_length(decoder, height, width);
    thread_count = std::clamp(thread_count, 1, site_pairs);
    if (thread_count == 1) {
        infer_packed_i8mm_worker(
            decoder,
            signed_nhwc.data(),
            width,
            0,
            site_pairs,
            sites,
            one_part,
            output);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(thread_count));
        for (int thread = 0; thread < thread_count; ++thread) {
            const int begin = thread * site_pairs / thread_count;
            const int end = (thread + 1) * site_pairs / thread_count;
            workers.emplace_back(
                infer_packed_i8mm_worker,
                std::cref(decoder),
                signed_nhwc.data(),
                width,
                begin,
                end,
                sites,
                one_part,
                output);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }
    if (saturation_count != nullptr) {
        *saturation_count = saturated;
    }
    return one_part;
#else
    (void)decoder;
    (void)z_hat_nchw;
    (void)height;
    (void)width;
    (void)output;
    (void)thread_count;
    (void)saturation_count;
    throw std::runtime_error(
        "ARM I8MM kernel was not compiled for this target");
#endif
}

inline size_t infer_packed_neon_dotprod(
    const PreparedLinearDecoder& decoder,
    const int16_t* z_hat_nchw,
    const int height,
    const int width,
    uint8_t* output,
    int thread_count,
    size_t* saturation_count = nullptr)
{
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
    if (height <= 0 || width <= 0 || z_hat_nchw == nullptr
        || output == nullptr) {
        throw std::runtime_error("invalid integer linear decoder input");
    }
    std::vector<int8_t> signed_nhwc;
    size_t saturated = 0;
    quantize_signed_nhwc(
        decoder,
        z_hat_nchw,
        height,
        width,
        signed_nhwc,
        saturated);
    const int sites = height * width;
    const size_t one_part = packed_length(decoder, height, width);
    thread_count = std::clamp(thread_count, 1, sites);
    if (thread_count == 1) {
        infer_packed_neon_worker(
            decoder,
            signed_nhwc.data(),
            height,
            width,
            0,
            sites,
            one_part,
            output);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(thread_count));
        for (int thread = 0; thread < thread_count; ++thread) {
            const int begin = thread * sites / thread_count;
            const int end = (thread + 1) * sites / thread_count;
            workers.emplace_back(
                infer_packed_neon_worker,
                std::cref(decoder),
                signed_nhwc.data(),
                height,
                width,
                begin,
                end,
                one_part,
                output);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }
    if (saturation_count != nullptr) {
        *saturation_count = saturated;
    }
    return one_part;
#else
    (void)decoder;
    (void)z_hat_nchw;
    (void)height;
    (void)width;
    (void)output;
    (void)thread_count;
    (void)saturation_count;
    throw std::runtime_error(
        "ARM NEON dot-product kernel was not compiled for this target");
#endif
}

}  // namespace pulse::log_index_scalar
