#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "fc.h"
#include "scaling.h"

void fully_connected(
    const param_t *weights,
    const param_t *input, size_p input_size,
    param_t *output, size_p output_size,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale)
{
    for (size_t o = 0; o < output_size; ++o)
    {
        int32_t sum = 0;
        for (size_t i = 0; i < input_size; ++i)
        {
            size_t weight_idx = (size_t) o * input_size + i;
            sum += input[i] * weights[weight_idx];
        }
        output[o] = rescale(sum, act_scale_a, act_scale_b, weight_scale);
    }
}
