// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <blocks_proxy.h>

void DiCoBlockProxy::set_param(const std::map<std::string, at::Tensor>& sd)
{
    conv1_.set_param(sd.at("conv1.weight"), sd.at("conv1.bias"));
    conv2_.set_param(sd.at("conv2.weight"), sd.at("conv2.bias"));
    conv3_.set_param(sd.at("conv3.weight"), sd.at("conv3.bias"));
    conv4_.set_param(sd.at("conv4.weight"), sd.at("conv4.bias"), 1, 1, false, true);
    conv5_.set_param(sd.at("conv5.weight"), sd.at("conv5.bias"));
}

at::Tensor DiCoBlockProxy::forward(const at::Tensor& inp, const at::optional<at::Tensor>& scale) const
{
    at::Tensor x = conv1_.forward_dense_conv1x1(inp);
    x = conv2_.forward_depthwise_conv3x3_hard_gelu_030(x);
    x = conv3_.forward_dense_conv1x1(x, at::nullopt, inp);
    at::Tensor t = conv4_.forward_dense_conv1x1(x);
    at::Tensor out = conv5_.forward_dense_conv1x1(t, at::nullopt, x, scale);
    return out;
}

void EncoderDiCoBlockProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t groups)
{
    TORCH_CHECK(groups == 4, "grouped pointwise convolutions require groups=4");
    conv1_.set_param(sd.at("conv1.weight"), sd.at("conv1.bias"));
    conv2_.set_param(sd.at("conv2.weight"), sd.at("conv2.bias"));
    conv3_.set_param(sd.at("conv3.weight"), sd.at("conv3.bias"), false, true);
    conv4_.set_param(sd.at("conv4.weight"), sd.at("conv4.bias"), true);
    conv5_.set_param(sd.at("conv5.weight"), sd.at("conv5.bias"), false, true);
}

at::Tensor EncoderDiCoBlockProxy::forward(const at::Tensor& inp,
                                          const at::optional<at::Tensor>& scale) const
{
    at::Tensor x = conv1_.forward_grouped_conv1x1(inp);
    x = conv2_.forward_depthwise_conv3x3_hard_gelu_030(x);
    x = conv3_.forward_grouped_conv1x1(x, at::nullopt, inp);
    at::Tensor ffn = conv4_.forward_grouped_conv1x1(x);
    x = conv5_.forward_grouped_conv1x1(ffn, at::nullopt, x, scale);
    return x;
}

void DepthConvBlockProxy::set_param(const std::map<std::string, at::Tensor>& sd, int64_t groups)
{
    auto it = sd.find("adaptor.weight");
    has_adaptor_ = it != sd.end();
    if (has_adaptor_) {
        TORCH_CHECK(sd.find("adaptor.bias") != sd.end(),
                    "adaptor.bias is required with adaptor.weight");
        adaptor_.set_param(sd.at("adaptor.weight"), sd.at("adaptor.bias"));
    }
    block_.set_param(get_submodule_state_dict(sd, "block."), groups);
}

at::Tensor DepthConvBlockProxy::forward(const at::Tensor& x, const at::optional<at::Tensor>& scale) const
{
    at::Tensor h = x;
    if (has_adaptor_) {
        h = adaptor_.forward_dense_conv1x1(h);
    }
    h = block_.forward(h, scale);
    return h;
}
