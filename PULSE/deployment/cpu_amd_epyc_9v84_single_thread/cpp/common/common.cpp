// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.h"

std::map<std::string, at::Tensor>
get_submodule_state_dict(const std::map<std::string, at::Tensor>& state_dict, const std::string& prefix)
{
    std::map<std::string, at::Tensor> out;
    for (const auto& kv : state_dict) {
        if (kv.first.rfind(prefix, 0) == 0) {
            out[kv.first.substr(prefix.size())] = kv.second;
        }
    }
    return out;
}

static at::Tensor get_one_mask(const at::Tensor& micro_mask, int64_t H, int64_t W)
{
    at::Tensor mask = micro_mask.repeat({ (H + 1) / 2, (W + 1) / 2 });
    mask = mask.slice(0, 0, H).slice(1, 0, W);
    mask = mask.unsqueeze(0).unsqueeze(0);
    return mask;
}

at::Tensor checkerboard_mask0(const at::Tensor& ref)
{
    const int64_t B = ref.size(0), C = ref.size(1), H = ref.size(2), W = ref.size(3);
    auto opts = at::TensorOptions().dtype(ref.scalar_type()).device(ref.device());
    auto mask_opts = at::TensorOptions().dtype(at::kFloat).device(ref.device());
    at::Tensor m = at::ones({ B, C / 2, H, W }, opts);
    at::Tensor m0 = get_one_mask(
        at::tensor({ 1.0, 0.0, 0.0, 1.0 }, mask_opts).to(ref.scalar_type()).view({ 2, 2 }), H, W);
    at::Tensor m1 = get_one_mask(
        at::tensor({ 0.0, 1.0, 1.0, 0.0 }, mask_opts).to(ref.scalar_type()).view({ 2, 2 }), H, W);
    at::Tensor even = m * m0;
    at::Tensor odd = m * m1;
    at::Tensor mask0 = at::cat({ even, odd }, 1).contiguous(at::MemoryFormat::ChannelsLast);
    return mask0;
}
