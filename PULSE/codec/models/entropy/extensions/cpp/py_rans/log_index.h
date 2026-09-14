// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>

namespace py = pybind11;

class LogIndexDecoder {
public:
    LogIndexDecoder(
        int m,
        int z_channels,
        int qp_num,
        int up_factor,
        int input_clip,
        const py::array_t<int32_t>& input_divisor,
        const py::array_t<int8_t>& weight,
        const py::array_t<int32_t>& multiplier,
        const py::array_t<int32_t>& affine_bias);

    py::array_t<uint8_t> packed_indexes(
        const py::array_t<int16_t>& z_hat,
        int qp) const;
    py::array_t<uint8_t> packed_indexes_reference(
        const py::array_t<int16_t>& z_hat,
        int qp) const;

    bool uses_vnni() const;

private:
    void quantize_input(
        const int16_t* z_hat,
        int height,
        int width,
        std::vector<uint8_t>& biased_nhwc) const;

    void packed_indexes_scalar(
        const uint8_t* biased_nhwc,
        int height,
        int width,
        int qp,
        uint8_t* output) const;

    void packed_indexes_vnni(
        const uint8_t* biased_nhwc,
        int height,
        int width,
        int qp,
        uint8_t* output) const;

    py::array_t<uint8_t> packed_indexes_impl(
        const py::array_t<int16_t>& z_hat,
        int qp,
        bool allow_vnni) const;

    int m_m;
    int m_z_channels;
    int m_qp_num;
    int m_up_factor;
    int m_input_clip;
    int m_pre_channels;
    bool m_use_vnni;

    std::vector<int32_t> m_input_divisor;
    std::vector<int8_t> m_reordered_weight;
    std::vector<int8_t> m_packed_weight;
    std::vector<int32_t> m_correction;
    std::vector<int32_t> m_multiplier;
    std::vector<int32_t> m_affine_bias;
};
