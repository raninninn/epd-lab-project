#ifndef SCALING_H
#define SCALING_H

#include <stdint.h>
#include "parameters.h"

/*
 * Rescales the given input value with the given quantization scales and clips it to
 * be in range [MIN_PARAM, MAX_PARAM]
 *
 * @param input: An unscaled, unclipped output of a network layer
 * @param act_scale_a: The quantization scale factor for the layer's input
 * @param act_scale_b: The quantization scale factor for the layer's output
 * @param weight_scale: The quantization scale factor for the layer's weights
 *
 * @return The clipped and rescaled input value
 */
param_t rescale(
    int32_t input,
    size_p act_scale_a,
    size_p act_scale_b,
    size_p weight_scale);

/*
 * Clips the given input value to the range of [min, max]
 */
param_t clip(
    int32_t input,
    param_t min,
    param_t max);
#endif
