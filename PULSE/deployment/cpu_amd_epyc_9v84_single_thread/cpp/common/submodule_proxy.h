// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <blocks_proxy.h>
#include "common.h"

class HyperDecoderProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t pointwise_groups);
    at::Tensor forward(const at::Tensor& z_hat, const at::Tensor& q_hyper_z,
                       const at::Tensor& q_hyper_m_basic) const;

private:
    DenseConv1x1Bf16 up0_, up1_, final_;
    DepthConvBlockProxy dcb0_, dcb1_;
};

class YPriorFusionProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t pointwise_groups);
    at::Tensor forward(const at::Tensor& params, const at::Tensor& q_prior,
                       const at::Tensor& symbols0, const at::Tensor& mask0) const;

private:
    DenseConv1x1Bf16 in_, out_;
    DepthConvBlockProxy dcb_;
};

class SpatialPriorProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t pointwise_groups);
    at::Tensor forward(const at::Tensor& y_hat0, const at::Tensor& symbols1,
                       const at::Tensor& qstep, const at::Tensor& means0, const at::Tensor& q_prior,
                       const at::Tensor& mask0, const at::Tensor& codec_scale) const;

private:
    DenseConv1x1Bf16 in_, out_;
    DepthConvBlockProxy dcb_;
};

class ScaleDecoderProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t up_factor = 4);
    at::Tensor forward(const at::Tensor& z_hat, int64_t qp) const;
    int64_t qp_num() const { return qp_num_; }

private:
    DenseConv1x1Bf16 proj_;
    at::Tensor qp_bias_;
    int64_t qp_num_{ 0 };
};

class ResBlockProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd);
    at::Tensor forward(const at::Tensor& x, const at::Tensor& modulation, const at::Tensor& scale) const;

private:
    DenseConv1x1Bf16 mlp0_, mlp2_, ada_;
};

class Up0Proxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t body_upsample);
    at::Tensor forward(const at::Tensor& z) const;

private:
    DenseConv1x1Bf16 conv_;
};

class PixelHeadProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t patch_size);
    at::Tensor forward(const at::Tensor& z) const;

private:
    DenseConv1x1Bf16 upx_, embedding_;
    at::Tensor pos_bias_;  // x_embedder.pos_bias (1,C,ps,ps)
};

class FinalLayerProxy {
public:
    void set_param(const std::map<std::string, at::Tensor>& sd, int64_t pixel_upsample);
    at::Tensor forward(const at::Tensor& x) const;

private:
    DenseConv1x1Bf16 conv_;
};
