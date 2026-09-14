// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "pinned_host_buffer.h"
#include "rans.h"
#include <memory>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr int MAX_EC_PARALLEL = 32;

// the classes in this file only perform the type conversion
// from python type (numpy) to C++ type (vector)
class RansEncoder {
public:
    RansEncoder();

    RansEncoder(const RansEncoder&) = delete;
    RansEncoder(RansEncoder&&) = delete;
    RansEncoder& operator=(const RansEncoder&) = delete;
    RansEncoder& operator=(RansEncoder&&) = delete;

    // symbols may contain more elements than symbolSize
    void encode_y(const int16_t* symbols, const int symbolSize,
                  const std::shared_ptr<void>& owner = nullptr);
    void encode_y(const py::array_t<int16_t>& symbols);
    void encode_y_borrowed(const py::array_t<int16_t>& symbols);
    void encode_y_skip(const int16_t* symbols, const int symbolSize,
                       const int skipIndexCutoff,
                       const std::shared_ptr<void>& owner = nullptr);
    void encode_y_skip(const py::array_t<int16_t>& symbols, const int skipIndexCutoff);
    void encode_y_skip_borrowed(
        const py::array_t<int16_t>& symbols,
        int skipIndexCutoff);
    void encode_z(const int16_t* symbols, const int symbolSize, const int cdf_offset, const int ch,
                  const std::shared_ptr<void>& owner = nullptr);
    void encode_z(const py::array_t<int16_t>& symbols, const int cdf_offset, const int ch);
    void encode_z_meta_prior(
        const int16_t* symbols, const uint8_t* meta_prior_indexes,
        const int symbolSize, const int positionCount, const int ch,
        const std::shared_ptr<void>& owner = nullptr);
    void encode_z_meta_prior(
        const py::array_t<int16_t>& symbols,
        const py::array_t<uint8_t>& meta_prior_indexes,
        const int ch);
    void encode_z_meta_prior_borrowed(
        const py::array_t<int16_t>& symbols,
        const py::array_t<uint8_t>& meta_prior_indexes,
        int ch);
    void flush();
    py::array_t<uint8_t> get_encoded_stream();
    void reset();
    void set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                 const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index);
    void set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                 const int index);
    void set_entropy_coder_parallel(int n);
    int get_entropy_coder_parallel();

private:
    std::vector<std::shared_ptr<RansEncoderLib>> m_encoders;
    int m_entropy_coder_parallel{ 1 };
    bool m_has_cdf{ false };
};

class RansDecoder {
public:
    RansDecoder();

    RansDecoder(const RansDecoder&) = delete;
    RansDecoder(RansDecoder&&) = delete;
    RansDecoder& operator=(const RansDecoder&) = delete;
    RansDecoder& operator=(RansDecoder&&) = delete;

    void set_stream(const py::array_t<uint8_t>&);
    void set_stream(const uint8_t* ptr, const int size);

    void decode_y(const uint8_t* indexes, const int indexSize,
                  const std::shared_ptr<void>& owner = nullptr);
    void decode_y(const py::array_t<uint8_t>& indexes);
    void decode_y_borrowed(const py::array_t<uint8_t>& indexes);
    void decode_y_skip(const uint8_t* indexes, const int indexSize,
                       const int skipIndexCutoff,
                       const std::shared_ptr<void>& owner = nullptr);
    void decode_y_skip(const py::array_t<uint8_t>& indexes, const int skipIndexCutoff);
    void decode_y_skip_borrowed(
        const py::array_t<uint8_t>& indexes,
        int skipIndexCutoff);
    std::shared_ptr<PinnedHostBuffer<int16_t>>
    decode_and_get_y(const uint8_t* indexes, const int indexSize,
                     const std::shared_ptr<void>& owner = nullptr);
    py::array_t<int16_t> decode_and_get_y(const py::array_t<uint8_t>& indexes);
    py::array_t<int16_t> decode_and_get_y_borrowed(
        const py::array_t<uint8_t>& indexes);
    std::shared_ptr<PinnedHostBuffer<int16_t>>
    decode_and_get_y_skip(const uint8_t* indexes, const int indexSize,
                          const int skipIndexCutoff,
                          const std::shared_ptr<void>& owner = nullptr);
    py::array_t<int16_t> decode_and_get_y_skip(
        const py::array_t<uint8_t>& indexes, const int skipIndexCutoff);
    py::array_t<int16_t> decode_and_get_y_skip_borrowed(
        const py::array_t<uint8_t>& indexes,
        int skipIndexCutoff);
    // if is_stream_nhmw, ch is channel number
    // otherwise, ch is per channel element number
    void decode_z(const int total_size, const int cdf_offset, const int ch);
    void decode_z_meta_prior(
        const int total_size,
        const py::array_t<uint8_t>& meta_prior_indexes,
        const int ch);
    std::shared_ptr<PinnedHostBuffer<int16_t>> get_decoded_tensor_cpp();
    py::array_t<int16_t> get_decoded_tensor();
    void set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                 const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index);
    void set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                 const int index);
    void set_entropy_coder_parallel(int n);
    int get_entropy_coder_parallel();

private:
    std::shared_ptr<PinnedHostBuffer<int16_t>> m_decoded_tensor;
    int m_current_decoded_tensor_size{ 0 };
    std::vector<std::shared_ptr<RansDecoderLib>> m_decoders;
    int m_entropy_coder_parallel{ 1 };
    bool m_has_cdf{ false };
};

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float>& pmf);
