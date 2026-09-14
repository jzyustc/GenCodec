#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_prior_masked_add_avx512_bf16(const at::Tensor& feature,
                                                      const at::Tensor& packed_weight,
                                                      const at::Tensor& bias, const at::Tensor& symbols0,
                                                      const at::Tensor& mask0);
