#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "scaling.h"
#include "utils.h"

param_t clip(int32_t input, param_t min, param_t max)
{
    if (input > max)
        return max;
    if (input < min)
        return min;
    return input;
}

param_t rescale(int32_t input,
                size_p act_scale_a,
                size_p act_scale_b,
                size_p weight_scale)
{
      int32_t tmp = div_bankers_rounding(input * act_scale_b, act_scale_a * weight_scale);
      return clip(tmp, MIN_PARAM, MAX_PARAM);
}
