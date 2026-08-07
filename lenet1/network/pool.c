#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "pool.h"
#include "utils.h"
#define POOL_SIZE 2

void avgpool(
    const param_t *input, size_p input_size, size_p input_channels,
    param_t *output, size_p output_size)
{
    assert(output_size * POOL_SIZE == input_size);
    const int32_t scaler = POOL_SIZE * POOL_SIZE;
    for (size_t ic = 0; ic < input_channels; ++ic)
    {
        for (size_t input_y = 0; input_y < input_size; input_y += POOL_SIZE) // stride equals pooling size
        {
            for (size_t input_x = 0; input_x < input_size; input_x += POOL_SIZE)
            {
                int32_t sum = 0;
                for (size_t offset_x = 0; offset_x < POOL_SIZE; ++offset_x)
                {
                    for (size_t offset_y = 0; offset_y < POOL_SIZE; ++offset_y)
                    {
                        sum += input[(ic * input_size + (input_y + offset_y)) * input_size + (input_x + offset_x)];
                    }
                }

                sum = div_bankers_rounding(sum, scaler);
                output[(ic * output_size + (input_y / POOL_SIZE)) * output_size
                    + (input_x / POOL_SIZE)] = sum;
            }
        }
    }
}
