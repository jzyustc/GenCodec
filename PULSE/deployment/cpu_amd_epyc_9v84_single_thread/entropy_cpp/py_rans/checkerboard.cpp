// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "checkerboard.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PULSE_CHECKERBOARD_X86_64 1
#else
#define PULSE_CHECKERBOARD_X86_64 0
#endif

#if PULSE_CHECKERBOARD_X86_64 && (defined(__GNUC__) || defined(__clang__))
#define PULSE_AVX512_COMPRESS_TARGET                                           \
    __attribute__((target("avx512f,avx512bw,avx512vl")))
#else
#define PULSE_AVX512_COMPRESS_TARGET
#endif

namespace {

void validate_skip_cutoff(const int cutoff)
{
    if (cutoff < 0 || cutoff >= 64) {
        throw std::runtime_error("checkerboard skip cutoff must be in [0, 63]");
    }
}

void validate_part(const int part)
{
    if (part != 0 && part != 1) {
        throw std::runtime_error("checkerboard part must be 0 or 1");
    }
}

template<typename T>
const T* require_contiguous_1d(const py::array_t<T>& array, const char* name)
{
    const py::buffer_info info = array.request();
    if (info.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be one-dimensional");
    }
    if (info.strides[0] != static_cast<py::ssize_t>(sizeof(T))) {
        throw std::runtime_error(std::string(name) + " must be contiguous");
    }
    return static_cast<const T*>(info.ptr);
}

int64_t checked_compact_size(
    const int batch,
    const int channels,
    const int height,
    const int width)
{
    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error("checkerboard dimensions must be positive");
    }
    if ((channels & 1) != 0) {
        throw std::runtime_error("checkerboard channel count must be even");
    }
    const int64_t size = static_cast<int64_t>(batch) * (channels / 2)
        * height * width;
    if (size > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("checkerboard compact tensor exceeds UINT32 positions");
    }
    return size;
}

int64_t nchw_offset_from_compact_position(
    const uint32_t position,
    const int channels,
    const int height,
    const int width,
    const int part)
{
    const int half = channels / 2;
    uint32_t cursor = position;
    const int compact_channel = static_cast<int>(cursor % half);
    cursor /= static_cast<uint32_t>(half);
    const int x = static_cast<int>(cursor % width);
    cursor /= static_cast<uint32_t>(width);
    const int y = static_cast<int>(cursor % height);
    const int batch_index = static_cast<int>(cursor / static_cast<uint32_t>(height));

    const bool even = ((x + y) & 1) == 0;
    const bool first_half = (part == 0) ? even : !even;
    const int channel = compact_channel + (first_half ? 0 : half);
    return ((static_cast<int64_t>(batch_index) * channels + channel) * height + y)
        * width + x;
}

bool has_avx512_compress()
{
#if PULSE_CHECKERBOARD_X86_64 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}

#if PULSE_CHECKERBOARD_X86_64
PULSE_AVX512_COMPRESS_TARGET
py::ssize_t count_indexes_above_avx512(
    const uint8_t* input,
    const py::ssize_t size,
    const int cutoff)
{
    const __m512i threshold = _mm512_set1_epi8(static_cast<char>(cutoff));
    py::ssize_t kept = 0;
    py::ssize_t offset = 0;
    for (; offset + 64 <= size; offset += 64) {
        const __m512i values = _mm512_loadu_si512(
            static_cast<const void*>(input + offset));
        const __mmask64 mask =
            _mm512_cmp_epu8_mask(values, threshold, _MM_CMPINT_GT);
        kept += static_cast<py::ssize_t>(
            __builtin_popcountll(static_cast<unsigned long long>(mask)));
    }
    for (; offset < size; ++offset) {
        kept += input[offset] > cutoff;
    }
    return kept;
}

PULSE_AVX512_COMPRESS_TARGET
void select_indexes_above_avx512(
    const uint8_t* input,
    const py::ssize_t size,
    const int cutoff,
    uint8_t* output_indexes,
    uint32_t* output_positions)
{
    const __m512i threshold = _mm512_set1_epi8(static_cast<char>(cutoff));
    py::ssize_t output = 0;
    py::ssize_t offset = 0;
    for (; offset + 64 <= size; offset += 64) {
        const __m512i values = _mm512_loadu_si512(
            static_cast<const void*>(input + offset));
        uint64_t mask = static_cast<uint64_t>(
            _mm512_cmp_epu8_mask(values, threshold, _MM_CMPINT_GT));
        while (mask != 0) {
            const unsigned int lane =
                static_cast<unsigned int>(__builtin_ctzll(mask));
            const py::ssize_t position = offset + lane;
            output_indexes[output] = input[position];
            output_positions[output] = static_cast<uint32_t>(position);
            ++output;
            mask &= mask - 1;
        }
    }
    for (; offset < size; ++offset) {
        if (input[offset] > cutoff) {
            output_indexes[output] = input[offset];
            output_positions[output] = static_cast<uint32_t>(offset);
            ++output;
        }
    }
}

PULSE_AVX512_COMPRESS_TARGET
void select_checkerboard_offsets_avx512(
    const uint8_t* input,
    const int cutoff,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int part,
    uint8_t* output_indexes,
    uint32_t* output_offsets)
{
    const __m512i threshold = _mm512_set1_epi8(static_cast<char>(cutoff));
    const int half = channels / 2;
    const uint32_t plane = static_cast<uint32_t>(height * width);
    py::ssize_t output = 0;
    py::ssize_t compact_offset = 0;
    for (int batch_index = 0; batch_index < batch; ++batch_index) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const bool even = ((x + y) & 1) == 0;
                const bool first_half = (part == 0) ? even : !even;
                const int channel_base = first_half ? 0 : half;
                const uint32_t spatial_offset =
                    static_cast<uint32_t>(
                        ((batch_index * channels + channel_base) * height + y)
                        * width + x);
                int channel = 0;
                for (; channel + 64 <= half; channel += 64) {
                    const __m512i values = _mm512_loadu_si512(
                        static_cast<const void*>(
                            input + compact_offset + channel));
                    uint64_t mask = static_cast<uint64_t>(
                        _mm512_cmp_epu8_mask(
                            values, threshold, _MM_CMPINT_GT));
                    while (mask != 0) {
                        const unsigned int lane =
                            static_cast<unsigned int>(__builtin_ctzll(mask));
                        const int selected_channel =
                            channel + static_cast<int>(lane);
                        output_indexes[output] =
                            input[compact_offset + selected_channel];
                        output_offsets[output] =
                            spatial_offset
                            + static_cast<uint32_t>(selected_channel) * plane;
                        ++output;
                        mask &= mask - 1;
                    }
                }
                for (; channel < half; ++channel) {
                    const uint8_t index = input[compact_offset + channel];
                    if (index > cutoff) {
                        output_indexes[output] = index;
                        output_offsets[output] =
                            spatial_offset
                            + static_cast<uint32_t>(channel) * plane;
                        ++output;
                    }
                }
                compact_offset += half;
            }
        }
    }
}

#endif

} // namespace

py::tuple select_indexes_above(const py::array_t<uint8_t>& indexes,
                               const int skip_index_cutoff)
{
    validate_skip_cutoff(skip_index_cutoff);
    const uint8_t* input = require_contiguous_1d(indexes, "indexes");
    const py::ssize_t size = indexes.size();
    py::ssize_t kept = 0;
#if PULSE_CHECKERBOARD_X86_64
    const bool use_avx512 = has_avx512_compress();
    if (use_avx512) {
        kept = count_indexes_above_avx512(
            input, size, skip_index_cutoff);
    } else
#endif
    {
        for (py::ssize_t i = 0; i < size; ++i) {
            kept += input[i] > skip_index_cutoff;
        }
    }

    py::array_t<uint8_t> selected_indexes({ kept }, { sizeof(uint8_t) });
    py::array_t<uint32_t> selected_positions({ kept }, { sizeof(uint32_t) });
    uint8_t* output_indexes =
        static_cast<uint8_t*>(selected_indexes.request().ptr);
    uint32_t* output_positions =
        static_cast<uint32_t*>(selected_positions.request().ptr);

#if PULSE_CHECKERBOARD_X86_64
    if (use_avx512) {
        select_indexes_above_avx512(
            input,
            size,
            skip_index_cutoff,
            output_indexes,
            output_positions);
    } else
#endif
    {
        py::ssize_t output = 0;
        for (py::ssize_t i = 0; i < size; ++i) {
            if (input[i] > skip_index_cutoff) {
                output_indexes[output] = input[i];
                output_positions[output] = static_cast<uint32_t>(i);
                ++output;
            }
        }
    }
    return py::make_tuple(selected_indexes, selected_positions);
}

py::tuple select_checkerboard_indexes_above(
    const py::array_t<uint8_t>& indexes,
    const int skip_index_cutoff,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int part)
{
    validate_skip_cutoff(skip_index_cutoff);
    validate_part(part);
    const int64_t compact_size =
        checked_compact_size(batch, channels, height, width);
    if (indexes.size() != compact_size) {
        throw std::runtime_error("packed checkerboard index size mismatch");
    }
    const uint8_t* input = require_contiguous_1d(indexes, "indexes");
    py::ssize_t kept = 0;
#if PULSE_CHECKERBOARD_X86_64
    if (has_avx512_compress()) {
        kept = count_indexes_above_avx512(
            input, indexes.size(), skip_index_cutoff);
    } else
#endif
    {
        for (py::ssize_t i = 0; i < indexes.size(); ++i) {
            kept += input[i] > skip_index_cutoff;
        }
    }

    py::array_t<uint8_t> selected_indexes({ kept }, { sizeof(uint8_t) });
    py::array_t<uint32_t> selected_offsets({ kept }, { sizeof(uint32_t) });
    uint8_t* output_indexes =
        static_cast<uint8_t*>(selected_indexes.request().ptr);
    uint32_t* output_offsets =
        static_cast<uint32_t*>(selected_offsets.request().ptr);

#if PULSE_CHECKERBOARD_X86_64
    if (has_avx512_compress()) {
        select_checkerboard_offsets_avx512(
            input,
            skip_index_cutoff,
            batch,
            channels,
            height,
            width,
            part,
            output_indexes,
            output_offsets);
    } else
#endif
    {
        const int half = channels / 2;
        const uint32_t plane = static_cast<uint32_t>(height * width);
        py::ssize_t output = 0;
        py::ssize_t compact_offset = 0;
        for (int batch_index = 0; batch_index < batch; ++batch_index) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const bool even = ((x + y) & 1) == 0;
                    const bool first_half = (part == 0) ? even : !even;
                    const int channel_base = first_half ? 0 : half;
                    const uint32_t spatial_offset =
                        static_cast<uint32_t>(
                            ((batch_index * channels + channel_base) * height + y)
                            * width + x);
                    for (int channel = 0; channel < half; ++channel) {
                        const uint8_t index = input[compact_offset + channel];
                        if (index > skip_index_cutoff) {
                            output_indexes[output] = index;
                            output_offsets[output] =
                                spatial_offset
                                + static_cast<uint32_t>(channel) * plane;
                            ++output;
                        }
                    }
                    compact_offset += half;
                }
            }
        }
    }
    return py::make_tuple(selected_indexes, selected_offsets);
}

py::tuple pack_checkerboard_positions(
    const py::array_t<float>& symbols_nchw,
    const py::array_t<uint8_t>& selected_indexes,
    const py::array_t<uint32_t>& selected_positions,
    const int part,
    const int max_symbol)
{
    validate_part(part);
    if (max_symbol <= 0 || max_symbol > 127) {
        throw std::runtime_error("max_symbol must be in [1, 127]");
    }

    const py::buffer_info symbols_info = symbols_nchw.request();
    if (symbols_info.ndim != 4) {
        throw std::runtime_error("symbols_nchw must be four-dimensional NCHW");
    }
    if (symbols_info.strides[3] != static_cast<py::ssize_t>(sizeof(float))) {
        throw std::runtime_error("symbols_nchw must be C-contiguous");
    }
    const int batch = static_cast<int>(symbols_info.shape[0]);
    const int channels = static_cast<int>(symbols_info.shape[1]);
    const int height = static_cast<int>(symbols_info.shape[2]);
    const int width = static_cast<int>(symbols_info.shape[3]);
    const int64_t compact_size =
        checked_compact_size(batch, channels, height, width);

    const uint8_t* indexes =
        require_contiguous_1d(selected_indexes, "selected_indexes");
    const uint32_t* positions =
        require_contiguous_1d(selected_positions, "selected_positions");
    if (selected_indexes.size() != selected_positions.size()) {
        throw std::runtime_error("selected index/position size mismatch");
    }

    const float* symbols = static_cast<const float*>(symbols_info.ptr);
    const py::ssize_t selected_size = selected_indexes.size();
    py::array_t<int16_t> packed({ selected_size }, { sizeof(int16_t) });
    int16_t* output = static_cast<int16_t*>(packed.request().ptr);
    int outside = 0;
    uint32_t previous = 0;
    for (py::ssize_t i = 0; i < selected_size; ++i) {
        const uint32_t position = positions[i];
        if (position >= static_cast<uint64_t>(compact_size)) {
            throw std::runtime_error("selected checkerboard position is out of range");
        }
        if (i > 0 && position <= previous) {
            throw std::runtime_error("selected checkerboard positions must be increasing");
        }
        previous = position;

        const int64_t source_offset = nchw_offset_from_compact_position(
            position, channels, height, width, part);
        long value = std::lrint(symbols[source_offset]);
        outside += value < -max_symbol || value > max_symbol;
        value = std::clamp(value, static_cast<long>(-max_symbol),
                           static_cast<long>(max_symbol));
        const uint16_t high =
            static_cast<uint16_t>(static_cast<uint8_t>(static_cast<int8_t>(value))) << 8;
        const uint16_t combined = high | indexes[i];
        output[i] = static_cast<int16_t>(combined);
    }
    return py::make_tuple(packed, outside);
}

py::tuple pack_nchw_offsets(
    const py::array_t<float>& symbols_nchw,
    const py::array_t<uint8_t>& selected_indexes,
    const py::array_t<uint32_t>& selected_offsets,
    const int max_symbol)
{
    if (max_symbol <= 0 || max_symbol > 127) {
        throw std::runtime_error("max_symbol must be in [1, 127]");
    }
    const py::buffer_info symbols_info = symbols_nchw.request();
    if (symbols_info.ndim != 4) {
        throw std::runtime_error("symbols_nchw must be four-dimensional NCHW");
    }
    const int64_t full_size = static_cast<int64_t>(symbols_info.shape[0])
        * symbols_info.shape[1] * symbols_info.shape[2] * symbols_info.shape[3];
    const py::ssize_t expected_last_stride =
        static_cast<py::ssize_t>(sizeof(float));
    if (symbols_info.strides[3] != expected_last_stride) {
        throw std::runtime_error("symbols_nchw must be C-contiguous");
    }
    const uint8_t* indexes =
        require_contiguous_1d(selected_indexes, "selected_indexes");
    const uint32_t* offsets =
        require_contiguous_1d(selected_offsets, "selected_offsets");
    if (selected_indexes.size() != selected_offsets.size()) {
        throw std::runtime_error("selected index/offset size mismatch");
    }
    const float* symbols = static_cast<const float*>(symbols_info.ptr);
    py::array_t<int16_t> packed(
        { selected_indexes.size() },
        { sizeof(int16_t) });
    int16_t* output = static_cast<int16_t*>(packed.request().ptr);
    int outside = 0;
    for (py::ssize_t i = 0; i < selected_indexes.size(); ++i) {
        if (offsets[i] >= static_cast<uint64_t>(full_size)) {
            throw std::runtime_error("selected NCHW offset is out of range");
        }
        long value = std::lrint(symbols[offsets[i]]);
        outside += value < -max_symbol || value > max_symbol;
        value = std::clamp(value, static_cast<long>(-max_symbol),
                           static_cast<long>(max_symbol));
        const uint16_t high =
            static_cast<uint16_t>(static_cast<uint8_t>(static_cast<int8_t>(value))) << 8;
        output[i] = static_cast<int16_t>(high | indexes[i]);
    }
    return py::make_tuple(packed, outside);
}

py::array_t<float> restore_checkerboard_positions(
    const py::array_t<int16_t>& decoded_symbols,
    const py::array_t<uint32_t>& selected_positions,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int part)
{
    validate_part(part);
    const int64_t compact_size =
        checked_compact_size(batch, channels, height, width);
    const int16_t* decoded =
        require_contiguous_1d(decoded_symbols, "decoded_symbols");
    const uint32_t* positions =
        require_contiguous_1d(selected_positions, "selected_positions");
    if (decoded_symbols.size() != selected_positions.size()) {
        throw std::runtime_error("decoded symbol/position size mismatch");
    }

    py::array_t<float> output(
        { batch, channels, height, width },
        {
            static_cast<py::ssize_t>(channels) * height * width * sizeof(float),
            static_cast<py::ssize_t>(height) * width * sizeof(float),
            static_cast<py::ssize_t>(width) * sizeof(float),
            sizeof(float),
        });
    float* restored = static_cast<float*>(output.request().ptr);
    std::fill(
        restored,
        restored + static_cast<int64_t>(batch) * channels * height * width,
        0.0F);

    uint32_t previous = 0;
    for (py::ssize_t i = 0; i < decoded_symbols.size(); ++i) {
        const uint32_t position = positions[i];
        if (position >= static_cast<uint64_t>(compact_size)) {
            throw std::runtime_error("selected checkerboard position is out of range");
        }
        if (i > 0 && position <= previous) {
            throw std::runtime_error("selected checkerboard positions must be increasing");
        }
        previous = position;
        const int64_t destination_offset = nchw_offset_from_compact_position(
            position, channels, height, width, part);
        restored[destination_offset] = static_cast<float>(decoded[i]);
    }
    return output;
}

void restore_checkerboard_positions_into(
    const py::array_t<float>& output_nchw,
    const py::array_t<int16_t>& decoded_symbols,
    const py::array_t<uint32_t>& selected_positions,
    const int part)
{
    validate_part(part);
    const py::buffer_info output_info = output_nchw.request();
    if (output_info.ndim != 4) {
        throw std::runtime_error("output_nchw must be four-dimensional NCHW");
    }
    const int batch = static_cast<int>(output_info.shape[0]);
    const int channels = static_cast<int>(output_info.shape[1]);
    const int height = static_cast<int>(output_info.shape[2]);
    const int width = static_cast<int>(output_info.shape[3]);
    const int64_t compact_size =
        checked_compact_size(batch, channels, height, width);
    const py::ssize_t expected_channel_stride =
        static_cast<py::ssize_t>(height) * width * sizeof(float);
    const py::ssize_t expected_batch_stride =
        static_cast<py::ssize_t>(channels) * expected_channel_stride;
    if (output_info.strides[0] != expected_batch_stride
        || output_info.strides[1] != expected_channel_stride
        || output_info.strides[2]
            != static_cast<py::ssize_t>(width)
                * static_cast<py::ssize_t>(sizeof(float))
        || output_info.strides[3] != static_cast<py::ssize_t>(sizeof(float))) {
        throw std::runtime_error("output_nchw must be C-contiguous");
    }

    const int16_t* decoded =
        require_contiguous_1d(decoded_symbols, "decoded_symbols");
    const uint32_t* positions =
        require_contiguous_1d(selected_positions, "selected_positions");
    if (decoded_symbols.size() != selected_positions.size()) {
        throw std::runtime_error("decoded symbol/position size mismatch");
    }
    float* output = static_cast<float*>(output_info.ptr);
    uint32_t previous = 0;
    for (py::ssize_t i = 0; i < decoded_symbols.size(); ++i) {
        const uint32_t position = positions[i];
        if (position >= static_cast<uint64_t>(compact_size)) {
            throw std::runtime_error("selected checkerboard position is out of range");
        }
        if (i > 0 && position <= previous) {
            throw std::runtime_error("selected checkerboard positions must be increasing");
        }
        previous = position;
        const int64_t destination_offset = nchw_offset_from_compact_position(
            position, channels, height, width, part);
        output[destination_offset] = static_cast<float>(decoded[i]);
    }
}

void restore_nchw_offsets_into(
    const py::array_t<float>& output_nchw,
    const py::array_t<int16_t>& decoded_symbols,
    const py::array_t<uint32_t>& selected_offsets)
{
    const py::buffer_info output_info = output_nchw.request();
    if (output_info.ndim != 4) {
        throw std::runtime_error("output_nchw must be four-dimensional NCHW");
    }
    const int64_t full_size = static_cast<int64_t>(output_info.shape[0])
        * output_info.shape[1] * output_info.shape[2] * output_info.shape[3];
    if (output_info.strides[3]
        != static_cast<py::ssize_t>(sizeof(float))) {
        throw std::runtime_error("output_nchw must be C-contiguous");
    }
    const int16_t* decoded =
        require_contiguous_1d(decoded_symbols, "decoded_symbols");
    const uint32_t* offsets =
        require_contiguous_1d(selected_offsets, "selected_offsets");
    if (decoded_symbols.size() != selected_offsets.size()) {
        throw std::runtime_error("decoded symbol/offset size mismatch");
    }
    float* output = static_cast<float*>(output_info.ptr);
    for (py::ssize_t i = 0; i < decoded_symbols.size(); ++i) {
        if (offsets[i] >= static_cast<uint64_t>(full_size)) {
            throw std::runtime_error("selected NCHW offset is out of range");
        }
        output[offsets[i]] = static_cast<float>(decoded[i]);
    }
}
