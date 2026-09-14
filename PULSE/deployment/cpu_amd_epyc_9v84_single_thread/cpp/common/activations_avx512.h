#pragma once

#include <immintrin.h>

static inline __m512 exp_approx_unclamped(__m512 x)
{
    __m512 fx = _mm512_floor_ps(
        _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f), _mm512_set1_ps(0.5f)));
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    x = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), x);
    __m512 y = _mm512_set1_ps(1.9875691500e-4f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
    y = _mm512_fmadd_ps(y, _mm512_mul_ps(x, x), x);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(y, fx);
}

static inline __m512 exp_approx(__m512 x)
{
    x = _mm512_max_ps(_mm512_set1_ps(-88.3762626647949f),
                      _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f)));
    return exp_approx_unclamped(x);
}

static constexpr float HARD_GELU_030_SLOPE = 0.30f;
static constexpr float HARD_GELU_OFFSET = 0.5f;

static inline __m512 hard_sigmoid(__m512 x)
{
    return _mm512_min_ps(
        _mm512_set1_ps(1.0f),
        _mm512_max_ps(_mm512_setzero_ps(), _mm512_add_ps(x, _mm512_set1_ps(HARD_GELU_OFFSET))));
}

static inline __m512 hard_gelu_030(__m512 x)
{
    const __m512 gate = _mm512_min_ps(
        _mm512_set1_ps(1.0f),
        _mm512_max_ps(_mm512_setzero_ps(), _mm512_fmadd_ps(_mm512_set1_ps(HARD_GELU_030_SLOPE), x,
                                                           _mm512_set1_ps(HARD_GELU_OFFSET))));
    return _mm512_mul_ps(x, gate);
}
