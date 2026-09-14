// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dense_conv1x1_pairs.h"

#define DENSE_STANDARD_PAIRS DENSE_ACTIVATION_PAIRS

#include "dense_conv1x1_bf16_internal.h"

template <bool HAS_BIAS, bool HAS_SCALAR_SCALE>
static bool dispatch_activation_epilogue(const DispatchArgs& args, bool has_silu,
                                         bool has_hard_gelu_030, bool has_sigmoid)
{
    if (has_silu) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, true, false, false, false, 1, 1>(
            args);
    }
    if (has_hard_gelu_030) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, true, false, false, 1, 1>(
            args);
    }
    if (has_sigmoid) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, false, true, false, 1, 1>(
            args);
    }
    return false;
}

template <bool HAS_BIAS>
static bool dispatch_activation_scale(const DispatchArgs& args, bool has_scalar_scale,
                                      bool has_silu, bool has_hard_gelu_030, bool has_sigmoid)
{
    return has_scalar_scale
               ? dispatch_activation_epilogue<HAS_BIAS, true>(args, has_silu, has_hard_gelu_030,
                                                              has_sigmoid)
               : dispatch_activation_epilogue<HAS_BIAS, false>(args, has_silu, has_hard_gelu_030,
                                                               has_sigmoid);
}

bool dispatch_dense_activation(const DispatchArgs& args, bool has_bias, bool has_scalar_scale,
                               bool has_silu, bool has_hard_gelu_030, bool has_sigmoid)
{
    return has_bias ? dispatch_activation_scale<true>(args, has_scalar_scale, has_silu,
                                                      has_hard_gelu_030, has_sigmoid)
                    : dispatch_activation_scale<false>(args, has_scalar_scale, has_silu,
                                                       has_hard_gelu_030, has_sigmoid);
}
