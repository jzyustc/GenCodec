// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"
#include "checkerboard.h"
#include "log_index.h"
#include "meta_index_merge.h"
#include "meta_selector.h"
#include "../../../cdf_offsets.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

void pack_symbols_indexes_into(
    const py::array_t<int8_t>& symbols,
    const py::array_t<uint8_t>& indexes,
    const py::array_t<int16_t>& output,
    const int threads)
{
    if (threads < 1 || threads > 80) {
        throw std::runtime_error("threads must be in [1, 80]");
    }
    const py::buffer_info symbols_info = symbols.request();
    const py::buffer_info indexes_info = indexes.request();
    const py::buffer_info output_info = output.request();
    if (symbols.size() != indexes.size()
        || symbols.size() != output.size()) {
        throw std::runtime_error(
            "symbols, indexes, and output must have equal sizes");
    }
    const auto* symbol_ptr =
        static_cast<const int8_t*>(symbols_info.ptr);
    const auto* index_ptr =
        static_cast<const uint8_t*>(indexes_info.ptr);
    auto* output_ptr = static_cast<int16_t*>(output_info.ptr);
    const py::ssize_t size = symbols.size();
    {
        py::gil_scoped_release release;
        if (threads == 1) {
            for (py::ssize_t i = 0; i < size; ++i) {
                output_ptr[i] = static_cast<int16_t>(
                    (static_cast<uint16_t>(
                        static_cast<uint8_t>(symbol_ptr[i]))
                     << 8)
                    | static_cast<uint16_t>(index_ptr[i]));
            }
        } else {
#pragma omp parallel for schedule(static) num_threads(threads)
            for (py::ssize_t i = 0; i < size; ++i) {
                output_ptr[i] = static_cast<int16_t>(
                    (static_cast<uint16_t>(
                        static_cast<uint8_t>(symbol_ptr[i]))
                     << 8)
                    | static_cast<uint16_t>(index_ptr[i]));
            }
        }
    }
}

void narrow_i16_to_i8_into(
    const py::array_t<int16_t>& input,
    const py::array_t<int8_t>& output,
    const int threads)
{
    if (threads < 1 || threads > 80) {
        throw std::runtime_error("threads must be in [1, 80]");
    }
    const py::buffer_info input_info = input.request();
    const py::buffer_info output_info = output.request();
    if (input.size() != output.size()) {
        throw std::runtime_error(
            "input and output must have equal sizes");
    }
    const auto* input_ptr =
        static_cast<const int16_t*>(input_info.ptr);
    auto* output_ptr = static_cast<int8_t*>(output_info.ptr);
    const py::ssize_t size = input.size();
    {
        py::gil_scoped_release release;
        if (threads == 1) {
            for (py::ssize_t i = 0; i < size; ++i) {
                output_ptr[i] = static_cast<int8_t>(input_ptr[i]);
            }
        } else {
#pragma omp parallel for schedule(static) num_threads(threads)
            for (py::ssize_t i = 0; i < size; ++i) {
                output_ptr[i] = static_cast<int8_t>(input_ptr[i]);
            }
        }
    }
}

py::bytes pack_fixed_width_u8(
    const py::array_t<uint8_t>& values,
    const int bits)
{
    if (bits < 1 || bits > 8) {
        throw std::runtime_error("bits must be in [1, 8]");
    }
    const py::buffer_info info = values.request();
    const auto* input = static_cast<const uint8_t*>(info.ptr);
    const size_t count = static_cast<size_t>(values.size());
    const uint32_t limit = 1U << bits;
    std::string output((count * static_cast<size_t>(bits) + 7U) / 8U, '\0');
    uint64_t accumulator = 0;
    int available = 0;
    size_t output_index = 0;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = input[i];
        if (value >= limit) {
            throw std::runtime_error(
                "value exceeds fixed-width field");
        }
        accumulator |= static_cast<uint64_t>(value) << available;
        available += bits;
        while (available >= 8) {
            output[output_index++] = static_cast<char>(
                accumulator & 0xffU);
            accumulator >>= 8;
            available -= 8;
        }
    }
    if (available != 0) {
        output[output_index] = static_cast<char>(
            accumulator & 0xffU);
    }
    return py::bytes(output);
}

py::array_t<uint8_t> unpack_fixed_width_u8(
    const py::bytes& payload,
    const py::ssize_t count,
    const int bits)
{
    if (count < 0 || bits < 1 || bits > 8) {
        throw std::runtime_error(
            "invalid fixed-width unpack arguments");
    }
    const std::string input = payload;
    const size_t expected = (
        static_cast<size_t>(count) * static_cast<size_t>(bits) + 7U
    ) / 8U;
    if (input.size() != expected) {
        throw std::runtime_error(
            "fixed-width payload size mismatch");
    }
    py::array_t<uint8_t> output(count);
    py::buffer_info output_info = output.request();
    auto* values = static_cast<uint8_t*>(output_info.ptr);
    uint64_t accumulator = 0;
    int available = 0;
    size_t input_index = 0;
    const uint64_t mask = (uint64_t{1} << bits) - 1U;
    for (py::ssize_t i = 0; i < count; ++i) {
        while (available < bits) {
            accumulator |= static_cast<uint64_t>(
                static_cast<uint8_t>(input[input_index++])
            ) << available;
            available += 8;
        }
        values[i] = static_cast<uint8_t>(
            accumulator & mask);
        accumulator >>= bits;
        available -= bits;
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(pulse_cpu_amd_entropy_ext, m)
{
    m.def("offset_cdf_symbols", &offset_cdf_symbols);
    m.def("offset_cdf_indexes", &offset_cdf_indexes);
    py::class_<MetaPriorSelector>(
        m,
        "MetaPriorSelector",
        py::module_local())
        .def(py::init<
             int,
             int,
             const py::array_t<float>&,
             const py::array_t<int32_t>&>())
        .def("select_into", &MetaPriorSelector::select_into);

    py::class_<LogIndexDecoder>(
        m,
        "LogIndexDecoder",
        py::module_local())
        .def(py::init<
             int,
             int,
             int,
             int,
             int,
             const py::array_t<int32_t>&,
             const py::array_t<int8_t>&,
             const py::array_t<int32_t>&,
             const py::array_t<int32_t>&>())
        .def("packed_indexes", &LogIndexDecoder::packed_indexes)
        .def("packed_indexes_mt", &LogIndexDecoder::packed_indexes_mt)
        .def("packed_indexes_into", &LogIndexDecoder::packed_indexes_into)
        .def("packed_symbols_into", &LogIndexDecoder::packed_symbols_into)
        .def(
            "packed_indexes_reference",
            &LogIndexDecoder::packed_indexes_reference)
        .def_property_readonly("uses_vnni", &LogIndexDecoder::uses_vnni);

    py::class_<RansEncoder>(
        m,
        "RansEncoder",
        py::module_local())
        .def(py::init<>())
        .def("encode_y", py::overload_cast<const py::array_t<int16_t>&>(&RansEncoder::encode_y))
        .def("encode_y_borrowed", &RansEncoder::encode_y_borrowed)
        .def("encode_y_skip",
             py::overload_cast<const py::array_t<int16_t>&, const int>(
                 &RansEncoder::encode_y_skip))
        .def(
            "encode_y_skip_borrowed",
            &RansEncoder::encode_y_skip_borrowed)
        .def("encode_z", py::overload_cast<const py::array_t<int16_t>&, const int, const int>(
                             &RansEncoder::encode_z))
        .def(
            "encode_z_meta_prior",
            py::overload_cast<
                const py::array_t<int16_t>&,
                const py::array_t<uint8_t>&,
                const int>(&RansEncoder::encode_z_meta_prior))
        .def(
            "encode_z_meta_prior_borrowed",
            &RansEncoder::encode_z_meta_prior_borrowed)
        .def("flush", &RansEncoder::flush)
        .def("get_encoded_stream", &RansEncoder::get_encoded_stream)
        .def("reset", &RansEncoder::reset)
        .def("set_cdf",
             py::overload_cast<const py::array_t<int32_t>&, const py::array_t<int32_t>&, const int>(
                 &RansEncoder::set_cdf))
        .def("set_entropy_coder_parallel", &RansEncoder::set_entropy_coder_parallel)
        .def("get_entropy_coder_parallel", &RansEncoder::get_entropy_coder_parallel);

    py::class_<RansDecoder>(
        m,
        "RansDecoder",
        py::module_local())
        .def(py::init<>())
        .def("set_stream", py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::set_stream))
        .def("decode_y", py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::decode_y))
        .def("decode_y_borrowed", &RansDecoder::decode_y_borrowed)
        .def("decode_y_skip",
             py::overload_cast<const py::array_t<uint8_t>&, const int>(
                 &RansDecoder::decode_y_skip))
        .def(
            "decode_y_skip_borrowed",
            &RansDecoder::decode_y_skip_borrowed)
        .def("decode_and_get_y",
             py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::decode_and_get_y))
        .def(
            "decode_and_get_y_borrowed",
            &RansDecoder::decode_and_get_y_borrowed)
        .def("decode_and_get_y_skip",
             py::overload_cast<const py::array_t<uint8_t>&, const int>(
                 &RansDecoder::decode_and_get_y_skip))
        .def(
            "decode_and_get_y_skip_borrowed",
            &RansDecoder::decode_and_get_y_skip_borrowed)
        .def("decode_z", &RansDecoder::decode_z)
        .def("decode_z_meta_prior", &RansDecoder::decode_z_meta_prior)
        .def("get_decoded_tensor", &RansDecoder::get_decoded_tensor)
        .def("set_cdf",
             py::overload_cast<const py::array_t<int32_t>&, const py::array_t<int32_t>&, const int>(
                 &RansDecoder::set_cdf))
        .def("set_entropy_coder_parallel", &RansDecoder::set_entropy_coder_parallel)
        .def("get_entropy_coder_parallel", &RansDecoder::get_entropy_coder_parallel);

    m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf, "Return quantized CDF for a given PMF");
    m.def("select_indexes_above", &select_indexes_above);
    m.def("select_checkerboard_indexes_above",
          &select_checkerboard_indexes_above);
    m.def("pack_checkerboard_positions", &pack_checkerboard_positions);
    m.def("pack_nchw_offsets", &pack_nchw_offsets);
    m.def("restore_checkerboard_positions", &restore_checkerboard_positions);
    m.def("restore_checkerboard_positions_into",
          &restore_checkerboard_positions_into);
    m.def("restore_nchw_offsets_into", &restore_nchw_offsets_into);
    m.def("pack_symbols_indexes_into", &pack_symbols_indexes_into);
    m.def("narrow_i16_to_i8_into", &narrow_i16_to_i8_into);
    m.def("pack_fixed_width_u8", &pack_fixed_width_u8);
    m.def("unpack_fixed_width_u8", &unpack_fixed_width_u8);
    m.def(
        "encode_meta_prior_index_merge",
        &encode_meta_prior_index_merge);
    m.def(
        "decode_meta_prior_index_merge",
        &decode_meta_prior_index_merge);
}
