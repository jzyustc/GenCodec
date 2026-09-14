// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"
#include "checkerboard.h"
#include "log_index.h"
#include "meta_index_merge.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(MLCodec_extensions_cpp, m)
{
    py::class_<LogIndexDecoder>(m, "LogIndexDecoder")
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
        .def(
            "packed_indexes_reference",
            &LogIndexDecoder::packed_indexes_reference)
        .def_property_readonly("uses_vnni", &LogIndexDecoder::uses_vnni);

    py::class_<RansEncoder>(m, "RansEncoder")
        .def(py::init<>())
        .def("encode_y", py::overload_cast<const py::array_t<int16_t>&>(&RansEncoder::encode_y))
        .def("encode_y_skip",
             py::overload_cast<const py::array_t<int16_t>&, const int>(
                 &RansEncoder::encode_y_skip))
        .def("encode_z", py::overload_cast<const py::array_t<int16_t>&, const int, const int>(
                             &RansEncoder::encode_z))
        .def(
            "encode_z_meta_prior",
            py::overload_cast<
                const py::array_t<int16_t>&,
                const py::array_t<uint8_t>&,
                const int>(&RansEncoder::encode_z_meta_prior))
        .def("flush", &RansEncoder::flush)
        .def("get_encoded_stream", &RansEncoder::get_encoded_stream)
        .def("reset", &RansEncoder::reset)
        .def("set_cdf",
             py::overload_cast<const py::array_t<int32_t>&, const py::array_t<int32_t>&, const int>(
                 &RansEncoder::set_cdf))
        .def("set_entropy_coder_parallel", &RansEncoder::set_entropy_coder_parallel)
        .def("get_entropy_coder_parallel", &RansEncoder::get_entropy_coder_parallel);

    py::class_<RansDecoder>(m, "RansDecoder")
        .def(py::init<>())
        .def("set_stream", py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::set_stream))
        .def("decode_y", py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::decode_y))
        .def("decode_y_skip",
             py::overload_cast<const py::array_t<uint8_t>&, const int>(
                 &RansDecoder::decode_y_skip))
        .def("decode_and_get_y",
             py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::decode_and_get_y))
        .def("decode_and_get_y_skip",
             py::overload_cast<const py::array_t<uint8_t>&, const int>(
                 &RansDecoder::decode_and_get_y_skip))
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
    m.def(
        "encode_meta_prior_index_merge",
        &encode_meta_prior_index_merge);
    m.def(
        "decode_meta_prior_index_merge",
        &decode_meta_prior_index_merge);
}
