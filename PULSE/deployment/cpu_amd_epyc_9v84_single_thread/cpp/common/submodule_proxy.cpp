// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "submodule_proxy.h"

void HyperDecoderProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t groups)
{
    up0_.set_param(sd.at("body.0.up.weight"), sd.at("body.0.up.bias"), 2);
    up1_.set_param(sd.at("body.1.up.weight"), sd.at("body.1.up.bias"), 2);
    dcb0_.set_param(get_submodule_state_dict(sd, "body.0.dcb."), groups);
    dcb1_.set_param(get_submodule_state_dict(sd, "body.1.dcb."), groups);
    final_.set_param(sd.at("body.2.weight"), sd.at("body.2.bias"));
}

at::Tensor HyperDecoderProxy::forward(const at::Tensor& z_hat, const at::Tensor& q_hyper_z,
                                      const at::Tensor& q_hyper_m_basic) const
{
    at::Tensor x = up0_.forward_dense_conv1x1(z_hat);
    x = dcb0_.forward(x, q_hyper_z);
    x = up1_.forward_dense_conv1x1(x);
    x = dcb1_.forward(x, q_hyper_z);
    x = final_.forward_dense_conv1x1(x, at::nullopt, at::nullopt, q_hyper_m_basic);
    return x;
}

void YPriorFusionProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t groups)
{
    in_.set_param(sd.at("in_proj.weight"), sd.at("in_proj.bias"));
    out_.set_param(sd.at("out_proj.weight"), sd.at("out_proj.bias"), 1, 2);
    dcb_.set_param(get_submodule_state_dict(sd, "dcb."), groups);
}

at::Tensor YPriorFusionProxy::forward(const at::Tensor& params, const at::Tensor& q_prior,
                                      const at::Tensor& symbols0, const at::Tensor& mask0) const
{
    at::Tensor x = in_.forward_dense_conv1x1(params);
    x = dcb_.forward(x, q_prior);
    x = out_.forward_dense_conv1x1_prior_masked_add(x, symbols0, mask0);
    return x;
}

void SpatialPriorProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t groups)
{
    in_.set_param(sd.at("in_proj.weight"), sd.at("in_proj.bias"));
    out_.set_param(sd.at("out_proj.weight"), sd.at("out_proj.bias"));
    dcb_.set_param(get_submodule_state_dict(sd, "dcb."), groups);
}

at::Tensor SpatialPriorProxy::forward(const at::Tensor& y_hat0, const at::Tensor& symbols1,
                                      const at::Tensor& qstep, const at::Tensor& means0,
                                      const at::Tensor& q_prior, const at::Tensor& mask0,
                                      const at::Tensor& codec_scale) const
{
    at::Tensor x = in_.forward_dense_conv1x1_cat3(y_hat0, qstep, means0);
    x = dcb_.forward(x, q_prior);
    x = out_.forward_dense_conv1x1_spatial_reconstruct(x, y_hat0, symbols1, qstep, mask0, codec_scale);
    return x;
}

void ScaleDecoderProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t up_factor)
{
    const at::Tensor& qp_bias = sd.at("qp_bias");
    TORCH_CHECK(qp_bias.device().is_cpu() && qp_bias.scalar_type() == at::kBFloat16
                    && qp_bias.dim() == 2 && qp_bias.size(0) > 0 && qp_bias.size(1) == 1,
                "qp_bias must be CPU BF16 [Q, 1]");
    qp_bias_ = qp_bias.contiguous();
    qp_num_ = qp_bias_.size(0);
    proj_.set_param(sd.at("proj.weight"), sd.at("proj.bias"), up_factor);
}

at::Tensor ScaleDecoderProxy::forward(const at::Tensor& z_hat, int64_t qp) const
{
    TORCH_CHECK(qp >= 0 && qp < qp_num_, "qp must be in [0, ", qp_num_, "), got ", qp);
    const float qp_bias = static_cast<float>(qp_bias_.data_ptr<at::BFloat16>()[qp]);
    return proj_.forward_dense_conv1x1_scale_decoder(z_hat, qp_bias);
}

void Up0Proxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t body_upsample)
{
    conv_.set_param(sd.at("0.weight"), sd.at("0.bias"), body_upsample);
}

at::Tensor Up0Proxy::forward(const at::Tensor& z) const
{
    at::Tensor output = conv_.forward_dense_conv1x1(z);
    return output;
}

void PixelHeadProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t patch_size)
{
    TORCH_CHECK(patch_size == 4, "periodic positional bias requires patch_size=4");
    upx_.set_param(sd.at("decoder.renderer.latent_upsample.0.weight"),
                   sd.at("decoder.renderer.latent_upsample.0.bias"), 4);
    embedding_.set_param(sd.at("decoder.renderer.input_proj.weight"));
    pos_bias_ = sd.at("decoder.renderer.position_bias");  // (1, C, ps, ps)
}

at::Tensor PixelHeadProxy::forward(const at::Tensor& z) const
{
    at::Tensor up = upx_.forward_dense_conv1x1(z);
    at::Tensor out = embedding_.forward_dense_conv1x1_periodic_bias4x4(up, pos_bias_);
    return out;
}

void FinalLayerProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t pixel_upsample)
{
    conv_.set_param(sd.at("0.weight"), sd.at("0.bias"), pixel_upsample);
}

at::Tensor FinalLayerProxy::forward(const at::Tensor& x) const
{
    at::Tensor output = conv_.forward_dense_conv1x1(x);
    return output;
}

void ResBlockProxy::set_param(const std::map<std::string, at::Tensor>& sd)
{
    mlp0_.set_param(sd.at("mlp.0.weight"), sd.at("mlp.0.bias"), 1, 1, true);
    mlp2_.set_param(sd.at("mlp.2.weight"), sd.at("mlp.2.bias"));
    ada_.set_param(sd.at("adaln.1.weight"), sd.at("adaln.1.bias"), 1, 3);
}

at::Tensor ResBlockProxy::forward(const at::Tensor& x, const at::Tensor& modulation,
                                  const at::Tensor& scale) const
{
    at::Tensor affine_and_gate = ada_.forward_dense_conv1x1_ada_hard_sigmoid_affine(modulation, x);
    auto parts = affine_and_gate.chunk(2, 0);
    at::Tensor h = parts[0];
    at::Tensor gate_mlp = parts[1];
    h = mlp0_.forward_dense_conv1x1(h);
    at::Tensor output = mlp2_.forward_dense_conv1x1_gated_residual_scale(h, gate_mlp, x, scale);
    return output;
}
