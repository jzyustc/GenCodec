#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_periodic_bias4x4_avx512_bf16(const at::Tensor& feature,
                                                      const at::Tensor& packed_weight,
                                                      const at::Tensor& periodic_bias);
