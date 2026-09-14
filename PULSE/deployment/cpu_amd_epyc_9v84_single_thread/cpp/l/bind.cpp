// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>

#include "decoder_proxy.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<DecodeWrapperProxy>(m, "DecodeWrapperProxy", py::module_local())
        .def(py::init<>())
        .def("set_param", &DecodeWrapperProxy::set_param)
        .def("forward", &DecodeWrapperProxy::forward)
        .def("forward_rgb", &DecodeWrapperProxy::forward_rgb)
        .def("forward_compact", &DecodeWrapperProxy::forward_compact)
        .def("profile_compact", &DecodeWrapperProxy::profile_compact);
}
