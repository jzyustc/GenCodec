#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_cat3_bias_avx512_bf16(const at::Tensor& input0, const at::Tensor& input1,
                                               const at::Tensor& input2,
                                               const at::Tensor& packed_weight, const at::Tensor& bias);
