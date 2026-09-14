// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "common.h"

class DiCoBlockCAProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd);
    at::Tensor forward(const at::Tensor& inp, const at::optional<at::Tensor>& scale = at::nullopt) const;

private:
    DenseConv1x1Bf16 conv1_, conv3_, conv4_, conv5_, ca_;
    DepthwiseConv3x3Bf16 conv2_;
};

class EncoderDiCoBlockCAProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t pointwise_groups);
    at::Tensor forward(const at::Tensor& inp, const at::optional<at::Tensor>& scale = at::nullopt) const;

private:
    GroupedConv1x1Bf16 conv1_, conv3_, conv4_, conv5_;
    DepthwiseConv3x3Bf16 conv2_;
    DenseConv1x1Bf16 ca_;
};

class DepthConvBlockProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t pointwise_groups);
    at::Tensor forward(const at::Tensor& x, const at::optional<at::Tensor>& scale = at::nullopt) const;

private:
    bool has_adaptor_{ false };
    DenseConv1x1Bf16 adaptor_;
    EncoderDiCoBlockCAProxy block_;
};
