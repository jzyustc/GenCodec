#pragma once

#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>

namespace py = pybind11;

class MetaPriorSelector {
public:
    MetaPriorSelector(
        int banks,
        int channels,
        const py::array_t<float>& core_cost,
        const py::array_t<int32_t>& max_value);

    void select_into(
        const py::array_t<int16_t>& z_hat,
        const py::array_t<uint8_t>& output,
        int threads) const;

private:
    int m_banks;
    int m_channels;
    std::vector<float> m_core_cost;
    std::vector<int32_t> m_max_value;
    std::vector<float> m_symbol_cost;
};
