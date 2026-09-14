// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Only output-chunk pairs are instantiated in this translation unit.
#define DENSE_STANDARD_PAIRS
#define DENSE_UPSCALE2_PAIRS
#define DENSE_UPSCALE4_PAIRS

#include "dense_conv1x1_bf16_internal.h"

template <bool HAS_BIAS, bool HAS_SCALAR_SCALE, int OUTPUT_CHUNKS>
static bool dispatch_chunks_epilogue(const DispatchArgs& args, bool has_silu,
                                     bool has_hard_gelu_030, bool has_sigmoid)
{
    if (has_silu) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, true, false, false,
                                    false, 1, OUTPUT_CHUNKS>(args);
    }
    if (has_hard_gelu_030) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, true, false,
                                    false, 1, OUTPUT_CHUNKS>(args);
    }
    if (has_sigmoid) {
        return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, false, true,
                                    false, 1, OUTPUT_CHUNKS>(args);
    }
    return dispatch_specialized<HAS_BIAS, false, HAS_SCALAR_SCALE, false, false, false, false,
                                false, 1, OUTPUT_CHUNKS>(args);
}

template <int OUTPUT_CHUNKS, bool HAS_BIAS>
static bool dispatch_chunks_scale(const DispatchArgs& args, bool has_scalar_scale, bool has_silu,
                                  bool has_hard_gelu_030, bool has_sigmoid)
{
    return has_scalar_scale ? dispatch_chunks_epilogue<HAS_BIAS, true, OUTPUT_CHUNKS>(
                                  args, has_silu, has_hard_gelu_030, has_sigmoid)
                            : dispatch_chunks_epilogue<HAS_BIAS, false, OUTPUT_CHUNKS>(
                                  args, has_silu, has_hard_gelu_030, has_sigmoid);
}

template <int OUTPUT_CHUNKS>
static bool dispatch_chunks_bias(const DispatchArgs& args, bool has_bias, bool has_scalar_scale,
                                 bool has_silu, bool has_hard_gelu_030, bool has_sigmoid)
{
    return has_bias ? dispatch_chunks_scale<OUTPUT_CHUNKS, true>(args, has_scalar_scale, has_silu,
                                                                 has_hard_gelu_030, has_sigmoid)
                    : dispatch_chunks_scale<OUTPUT_CHUNKS, false>(args, has_scalar_scale, has_silu,
                                                                  has_hard_gelu_030, has_sigmoid);
}

bool dispatch_dense_chunks(const DispatchArgs& args, bool has_bias, bool has_scalar_scale, bool has_silu,
                           bool has_hard_gelu_030, bool has_sigmoid, int64_t output_chunks)
{
    if (output_chunks == 2) {
        return dispatch_chunks_bias<2>(args, has_bias, has_scalar_scale, has_silu,
                                       has_hard_gelu_030, has_sigmoid);
    }
    return dispatch_chunks_bias<3>(args, has_bias, has_scalar_scale, has_silu, has_hard_gelu_030,
                                   has_sigmoid);
}
