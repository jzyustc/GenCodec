#include "model_config.h"
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Only pixel-shuffle pairs are instantiated in this translation unit.
#define DENSE_STANDARD_PAIRS
#define DENSE_CHUNKS2_PAIRS
#define DENSE_CHUNKS3_PAIRS

#include "dense_conv1x1_bf16_internal.h"

template <bool HAS_BIAS, bool HAS_SCALAR_SCALE, int UPSCALE_FACTOR>
static bool dispatch_upsample_epilogue(const DispatchArgs& args, bool has_silu,
                                       bool has_hard_gelu_030, bool has_sigmoid)
{
    if (has_silu) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, true, false, false,
                                    false, UPSCALE_FACTOR, 1>(args);
    }
    if (has_hard_gelu_030) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, true, false,
                                    false, UPSCALE_FACTOR, 1>(args);
    }
    if (has_sigmoid) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, false, true,
                                    false, UPSCALE_FACTOR, 1>(args);
    }
    return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, false, false,
                                false, UPSCALE_FACTOR, 1>(args);
}

template <int UPSCALE_FACTOR, bool HAS_BIAS>
static bool dispatch_upsample_scale(const DispatchArgs& args, bool has_scalar_scale, bool has_silu,
                                    bool has_hard_gelu_030, bool has_sigmoid)
{
    return has_scalar_scale ? dispatch_upsample_epilogue<HAS_BIAS, true, UPSCALE_FACTOR>(
                                  args, has_silu, has_hard_gelu_030, has_sigmoid)
                            : dispatch_upsample_epilogue<HAS_BIAS, false, UPSCALE_FACTOR>(
                                  args, has_silu, has_hard_gelu_030, has_sigmoid);
}

template <int UPSCALE_FACTOR>
static bool dispatch_upsample_bias(const DispatchArgs& args, bool has_bias, bool has_scalar_scale,
                                   bool has_silu, bool has_hard_gelu_030, bool has_sigmoid)
{
    return has_bias ? dispatch_upsample_scale<UPSCALE_FACTOR, true>(args, has_scalar_scale, has_silu,
                                                                    has_hard_gelu_030, has_sigmoid)
                    : dispatch_upsample_scale<UPSCALE_FACTOR, false>(args, has_scalar_scale, has_silu,
                                                                     has_hard_gelu_030, has_sigmoid);
}

bool dispatch_dense_upsample(const DispatchArgs& args, bool has_bias, bool has_scalar_scale,
                             bool has_silu, bool has_hard_gelu_030, bool has_sigmoid,
                             bool has_scale_decoder, int64_t upscale_factor)
{
    if (has_scale_decoder) {
        if (!has_bias || has_scalar_scale || upscale_factor != 4 || args.input_channels != PULSE_Z_CHANNELS
            || args.output_channels != PULSE_MAIN_CHANNELS * 16) {
            return false;
        }
        dense_conv1x1_specialized<PULSE_Z_CHANNELS, PULSE_MAIN_CHANNELS * 16, true, false, false, false, false, false, false, true, 4, 1>(
            args.feature, args.packed_weight, args.bias, nullptr, nullptr, args.output, args.pixels,
            args.height, args.width, args.output_scale);
        return true;
    }
    if (upscale_factor == 2) {
        return dispatch_upsample_bias<2>(args, has_bias, has_scalar_scale, has_silu,
                                         has_hard_gelu_030, has_sigmoid);
    }
    return dispatch_upsample_bias<4>(args, has_bias, has_scalar_scale, has_silu, has_hard_gelu_030,
                                     has_sigmoid);
}
