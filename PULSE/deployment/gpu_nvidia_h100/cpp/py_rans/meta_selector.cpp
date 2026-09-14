#include "meta_selector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PULSE_META_X86_64 1
#else
#define PULSE_META_X86_64 0
#endif

#if PULSE_META_X86_64 && (defined(__GNUC__) || defined(__clang__))
#define PULSE_META_AVX512_TARGET \
    __attribute__((target("avx512f,avx512bw,avx512vl")))
#else
#define PULSE_META_AVX512_TARGET
#endif

namespace {

template <typename T>
std::vector<T> copy_array(
    const py::array_t<T>& array,
    const std::vector<py::ssize_t>& expected,
    const char* name)
{
    const py::buffer_info info = array.request();
    if (info.ndim != static_cast<py::ssize_t>(expected.size())) {
        throw std::runtime_error(std::string(name) + " has wrong rank");
    }
    size_t count = 1;
    for (size_t axis = 0; axis < expected.size(); ++axis) {
        if (info.shape[axis] != expected[axis]) {
            throw std::runtime_error(std::string(name) + " has wrong shape");
        }
        count *= static_cast<size_t>(expected[axis]);
    }
    const auto* source = static_cast<const T*>(info.ptr);
    return std::vector<T>(source, source + count);
}

inline uint32_t symbol_value(const int16_t symbol)
{
    const int32_t value = static_cast<int32_t>(symbol);
    const uint32_t magnitude = value < 0
        ? static_cast<uint32_t>(-value)
        : static_cast<uint32_t>(value);
    return magnitude * 2U - static_cast<uint32_t>(value > 0);
}

inline float bypass_cost(const uint32_t raw)
{
    uint32_t remaining = raw;
    int groups = 0;
    while (remaining > 0) {
        ++groups;
        remaining >>= 2;
    }
    return static_cast<float>(2 * (groups / 3 + 1 + groups));
}

bool has_avx512()
{
#if PULSE_META_X86_64 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl");
#else
    return false;
#endif
}

#if PULSE_META_X86_64
PULSE_META_AVX512_TARGET
void select_avx512(
    const int16_t* z_ptr,
    const float* symbol_cost,
    const int sites,
    const int channels,
    uint8_t* output_ptr,
    const int threads)
{
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int position = 0; position < sites; ++position) {
        __m512 accumulators[4] = {
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        };
        bool fast = true;
        for (int channel = 0; channel < channels; ++channel) {
            const int symbol = static_cast<int>(
                z_ptr[channel * sites + position]);
            if (symbol < -128 || symbol > 127) {
                fast = false;
                break;
            }
            const float* costs = symbol_cost
                + static_cast<size_t>(
                    (channel * 256 + symbol + 128) * 64);
            accumulators[0] = _mm512_add_ps(
                accumulators[0],
                _mm512_loadu_ps(costs));
            accumulators[1] = _mm512_add_ps(
                accumulators[1],
                _mm512_loadu_ps(costs + 16));
            accumulators[2] = _mm512_add_ps(
                accumulators[2],
                _mm512_loadu_ps(costs + 32));
            accumulators[3] = _mm512_add_ps(
                accumulators[3],
                _mm512_loadu_ps(costs + 48));
        }
        if (!fast) {
            output_ptr[position] = 255;
            continue;
        }
        alignas(64) float totals[64];
        _mm512_store_ps(totals, accumulators[0]);
        _mm512_store_ps(totals + 16, accumulators[1]);
        _mm512_store_ps(totals + 32, accumulators[2]);
        _mm512_store_ps(totals + 48, accumulators[3]);
        int best_bank = 0;
        float best_cost = totals[0];
        for (int bank = 1; bank < 64; ++bank) {
            if (totals[bank] < best_cost) {
                best_cost = totals[bank];
                best_bank = bank;
            }
        }
        output_ptr[position] = static_cast<uint8_t>(best_bank);
    }
}
#endif

}  // namespace

MetaPriorSelector::MetaPriorSelector(
    const int banks,
    const int channels,
    const py::array_t<float>& core_cost,
    const py::array_t<int32_t>& max_value)
    : m_banks(banks),
      m_channels(channels)
{
    if (m_banks < 2 || m_banks > 256 || m_channels <= 0) {
        throw std::runtime_error("invalid Meta Prior dimensions");
    }
    m_core_cost = copy_array<float>(
        core_cost,
        {m_banks, m_channels, 128},
        "core_cost");
    m_max_value = copy_array<int32_t>(
        max_value,
        {m_banks, m_channels},
        "max_value");
    if (m_banks == 64) {
        m_symbol_cost.resize(
            static_cast<size_t>(m_channels * 256 * m_banks));
        for (int channel = 0; channel < m_channels; ++channel) {
            for (int symbol = -128; symbol <= 127; ++symbol) {
                const uint32_t value = symbol_value(
                    static_cast<int16_t>(symbol));
                for (int bank = 0; bank < m_banks; ++bank) {
                    const size_t bank_channel = static_cast<size_t>(
                        bank * m_channels + channel);
                    const int32_t maximum =
                        m_max_value[bank_channel];
                    const uint32_t encoded =
                        std::min<uint32_t>(
                            value,
                            static_cast<uint32_t>(maximum));
                    float cost = m_core_cost[
                        bank_channel * 128U + encoded];
                    if (value >= static_cast<uint32_t>(maximum)) {
                        cost += bypass_cost(
                            value - static_cast<uint32_t>(maximum));
                    }
                    m_symbol_cost[static_cast<size_t>(
                        (channel * 256 + symbol + 128) * m_banks
                        + bank)] = cost;
                }
            }
        }
    }
}

void MetaPriorSelector::select_into(
    const py::array_t<int16_t>& z_hat,
    const py::array_t<uint8_t>& output,
    const int threads) const
{
    if (threads < 1 || threads > 80) {
        throw std::runtime_error("threads must be in [1, 80]");
    }
    const py::buffer_info z_info = z_hat.request();
    if (z_info.ndim != 4
        || z_info.shape[0] != 1
        || z_info.shape[1] != m_channels
        || z_info.shape[2] <= 0
        || z_info.shape[3] <= 0) {
        throw std::runtime_error(
            "z_hat must have shape [1, channels, H, W]");
    }
    const int sites = static_cast<int>(z_info.shape[2] * z_info.shape[3]);
    const py::buffer_info output_info = output.request();
    if (output.size() != sites) {
        throw std::runtime_error("Meta Prior output has wrong size");
    }
    const auto* z_ptr = static_cast<const int16_t*>(z_info.ptr);
    auto* output_ptr = static_cast<uint8_t*>(output_info.ptr);
    std::vector<uint32_t> values(
        static_cast<size_t>(sites * m_channels));
    {
        py::gil_scoped_release release;
#if PULSE_META_X86_64
        if (m_banks == 64 && has_avx512()) {
            select_avx512(
                z_ptr,
                m_symbol_cost.data(),
                sites,
                m_channels,
                output_ptr,
                threads);
            bool all_fast = true;
            for (int position = 0; position < sites; ++position) {
                all_fast = all_fast && output_ptr[position] != 255;
            }
            if (all_fast) {
                return;
            }
        }
#endif
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
        for (int position = 0; position < sites; ++position) {
            uint32_t* position_values = values.data()
                + static_cast<size_t>(position * m_channels);
            for (int channel = 0; channel < m_channels; ++channel) {
                position_values[channel] = symbol_value(
                    z_ptr[channel * sites + position]);
            }
        }
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
        for (int position = 0; position < sites; ++position) {
            const uint32_t* position_values = values.data()
                + static_cast<size_t>(position * m_channels);
            float best_cost = std::numeric_limits<float>::infinity();
            int best_bank = 0;
            for (int bank = 0; bank < m_banks; ++bank) {
                float total = 0.0F;
                const size_t bank_offset = static_cast<size_t>(
                    bank * m_channels);
                for (int channel = 0; channel < m_channels; ++channel) {
                    const uint32_t value = position_values[channel];
                    const int32_t maximum = m_max_value[
                        bank_offset + static_cast<size_t>(channel)];
                    const uint32_t encoded = std::min<uint32_t>(
                        value,
                        static_cast<uint32_t>(maximum));
                    const size_t cost_offset =
                        (bank_offset + static_cast<size_t>(channel)) * 128U
                        + encoded;
                    total += m_core_cost[cost_offset];
                    if (value >= static_cast<uint32_t>(maximum)) {
                        total += bypass_cost(
                            value - static_cast<uint32_t>(maximum));
                    }
                }
                if (total < best_cost) {
                    best_cost = total;
                    best_bank = bank;
                }
            }
            output_ptr[position] = static_cast<uint8_t>(best_bank);
        }
    }
}
