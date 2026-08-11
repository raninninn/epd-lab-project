#pragma once
#include <stdint.h>

#define QMAX 127
#define QMIN (-QMAX - 1)

// Divide by 2^shift, rounding half to even (banker's rounding) -- must match torch.round
static inline int32_t round_shift(int32_t acc, int shift)
{
    int32_t half = 1 << (shift - 1);
    int32_t q = (acc + half) >> shift;          // round half up
    if ((acc & ((1 << shift) - 1)) == half && (q & 1))
        q -= 1;                                 // tie and odd: round to even instead
    return q;
}

static inline int8_t clamp_int8(int32_t v)
{
    if (v > QMAX) return QMAX;
    if (v < QMIN) return QMIN;
    return (int8_t)v;
}

// Accumulator at scale 2^shift_in -> int8 activation at scale 2^shift_out
static inline int8_t requantize(int32_t acc, int shift_in, int shift_out)
{
    return clamp_int8(round_shift(acc, shift_in - shift_out));
}