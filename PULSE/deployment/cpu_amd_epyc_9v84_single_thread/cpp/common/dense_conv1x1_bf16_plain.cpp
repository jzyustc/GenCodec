#include "model_config.h"
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#define DENSE_OUTPUT_BLOCK_TILE PULSE_PLAIN_OUTPUT_TILE
#include "dense_conv1x1_bf16_internal.h"

template <bool HAS_BIAS>
static bool dispatch_plain_bias(const DispatchArgs& args, bool has_residual, bool has_scalar_scale,
                                bool has_channel_scale)
{
    if (has_residual) {
        if (has_scalar_scale) {
            return dispatch_specialized<HAS_BIAS, true, true, false, false, false, false, false, 1, 1>(
                args);
        }
        if (has_channel_scale) {
            return dispatch_specialized<HAS_BIAS, true, false, true, false, false, false, false, 1, 1>(
                args);
        }
        return dispatch_specialized<HAS_BIAS, true, false, false, false, false, false, false, 1, 1>(args);
    }
    if (has_scalar_scale) {
        return dispatch_specialized<HAS_BIAS, false, true, false, false, false, false, false, 1, 1>(args);
    }
    return dispatch_specialized<HAS_BIAS, false, false, false, false, false, false, false, 1, 1>(args);
}

bool dispatch_dense_plain(const DispatchArgs& args, bool has_bias, bool has_residual,
                          bool has_scalar_scale, bool has_channel_scale)
{
    return has_bias
               ? dispatch_plain_bias<true>(args, has_residual, has_scalar_scale, has_channel_scale)
               : dispatch_plain_bias<false>(args, has_residual, has_scalar_scale, has_channel_scale);
}
