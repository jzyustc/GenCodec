// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once
#include <pybind11/numpy.h>
#include <cstdint>

// Keep skipped indexes in 0..63 for the single-lane skip scanner. A native
// vectorizable loop avoids NumPy's masked iteration over an entire frame.
inline void offset_cdf_symbols(
    pybind11::array_t<int16_t, pybind11::array::c_style> values, int offset, int cutoff)
{
    auto* data=values.mutable_data();
    const auto count=values.size();
    pybind11::gil_scoped_release release;
    for (pybind11::ssize_t i=0;i<count;++i)
        data[i]=int16_t(data[i]+(((data[i]&255)>cutoff)?offset:0));
}
inline void offset_cdf_indexes(
    pybind11::array_t<uint8_t, pybind11::array::c_style> values, int offset, int cutoff)
{
    auto* data=values.mutable_data();
    const auto count=values.size();
    pybind11::gil_scoped_release release;
    for (pybind11::ssize_t i=0;i<count;++i)
        data[i]=uint8_t(data[i]+((data[i]>cutoff)?offset:0));
}
