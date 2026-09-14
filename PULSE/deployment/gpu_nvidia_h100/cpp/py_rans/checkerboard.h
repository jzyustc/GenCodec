// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Select CDF rows above the deterministic skip cutoff. The returned UINT32
// positions refer to the original compact checkerboard array.
py::tuple select_indexes_above(const py::array_t<uint8_t>& indexes,
                               const int skip_index_cutoff);

// Shape-aware selector that returns direct flat NCHW offsets, avoiding
// repeated compact-position division/modulo in both gather and restore.
py::tuple select_checkerboard_indexes_above(
    const py::array_t<uint8_t>& indexes,
    const int skip_index_cutoff,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int part);

// Gather selected symbols from a full NCHW tensor, round/clamp them, and pack
// each symbol with its CDF row in the format consumed by RansEncoder::encode_y.
py::tuple pack_checkerboard_positions(
    const py::array_t<float>& symbols_nchw,
    const py::array_t<uint8_t>& selected_indexes,
    const py::array_t<uint32_t>& selected_positions,
    const int part,
    const int max_symbol);

py::tuple pack_nchw_offsets(
    const py::array_t<float>& symbols_nchw,
    const py::array_t<uint8_t>& selected_indexes,
    const py::array_t<uint32_t>& selected_offsets,
    const int max_symbol);

// Restore compact decoded symbols directly into a zero-initialized full NCHW
// tensor. Skipped entries remain zero.
py::array_t<float> restore_checkerboard_positions(
    const py::array_t<int16_t>& decoded_symbols,
    const py::array_t<uint32_t>& selected_positions,
    const int batch,
    const int channels,
    const int height,
    const int width,
    const int part);

// Sparse in-place variant used when both checkerboard parts share one full
// decoder input buffer. The caller owns initialization of the output tensor.
void restore_checkerboard_positions_into(
    const py::array_t<float>& output_nchw,
    const py::array_t<int16_t>& decoded_symbols,
    const py::array_t<uint32_t>& selected_positions,
    const int part);

void restore_nchw_offsets_into(
    const py::array_t<float>& output_nchw,
    const py::array_t<int16_t>& decoded_symbols,
    const py::array_t<uint32_t>& selected_offsets);
