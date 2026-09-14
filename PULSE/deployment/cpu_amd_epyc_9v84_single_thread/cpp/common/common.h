// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <string>
#include <tuple>

#include <torch/extension.h>

#include <conv_proxy.h>

std::map<std::string, at::Tensor> get_submodule_state_dict(const std::map<std::string, at::Tensor>& state_dict,
                                                           const std::string& prefix);

at::Tensor checkerboard_mask0(const at::Tensor& ref);
