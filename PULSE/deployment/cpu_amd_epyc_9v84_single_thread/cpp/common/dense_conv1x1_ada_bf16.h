// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_ada_hard_sigmoid_affine_avx512_bf16(const at::Tensor& feature,
                                                             const at::Tensor& residual,
                                                             const at::Tensor& packed_weight,
                                                             const at::Tensor& bias);
