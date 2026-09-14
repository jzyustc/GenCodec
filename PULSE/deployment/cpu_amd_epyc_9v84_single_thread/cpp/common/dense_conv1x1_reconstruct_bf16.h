#pragma once

#include <torch/extension.h>

at::Tensor dense_conv1x1_spatial_reconstruct_avx512_bf16(
    const at::Tensor& feature, const at::Tensor& packed_weight, const at::Tensor& bias,
    const at::Tensor& y_hat0, const at::Tensor& symbols1, const at::Tensor& qstep,
    const at::Tensor& mask0, const at::Tensor& codec_scale);
