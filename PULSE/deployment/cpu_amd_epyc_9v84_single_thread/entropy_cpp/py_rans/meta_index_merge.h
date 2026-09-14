// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

// Spatial merge coding for a Meta Prior index map.  The probability table is
// laid out as:
//
//   [same-left contexts: 2]
//   [same-up contexts:   2]
//   [escape index binary-tree nodes x left/up bit contexts: (2^bits-1) x 4]
//
// Probabilities are P(bin == 0) in Q11, constrained to [1, 2047].  The table
// is copied for every call and then adapted locally, so every image is an
// independently decodable entropy unit.
py::bytes encode_meta_prior_index_merge(
    const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>&
        indexes,
    int batch,
    int height,
    int width,
    int index_bits,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        initial_probabilities,
    int adaptation_shift);

py::array_t<uint8_t> decode_meta_prior_index_merge(
    const py::bytes& payload,
    int batch,
    int height,
    int width,
    int index_bits,
    int bank_count,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        initial_probabilities,
    int adaptation_shift);
