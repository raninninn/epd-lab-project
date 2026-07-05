#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "network.h"
#define KERNEL_SIZE 5
#define POOL_SIZE 2
#define THRESHOLD 0

param_t clip(int32_t input, param_t min, param_t max)
{
    if (input > max)
        return max;
    if (input < min)
        return min;
    return input;
}

void convolve(
    const param_t *weights, size_p in_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p output_channel,
    size_p stride,
    size_p padding,
    size_p act_scale, size_p weight_scale)
{
    for (size_t oy = 0; oy < out_size; ++oy)
    {
        for (size_t ox = 0; ox < out_size; ++ox)
        {
            int32_t sum = 0;
            for (size_t ic = 0; ic < in_channels; ++ic)
            {
                for (size_t ky = 0; ky < KERNEL_SIZE; ++ky)
                {
                    for (size_t kx = 0; kx < KERNEL_SIZE; ++kx)
                    {
                        int in_y = oy * stride + ky - padding; 
                        int in_x = ox * stride + kx - padding;

                        // Zero padding
                        if (in_y < 0 || in_y >= in_size ||
                            in_x < 0 || in_x >= in_size)
                        {
                            continue;
                        }

                        size_t input_idx = ((size_t) ic * in_size + in_y) * in_size + in_x;

                        size_t weight_idx = (((size_t) output_channel * in_channels + ic)
                             * KERNEL_SIZE + ky)
                             * KERNEL_SIZE + kx;

                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
            size_t output_idx = ((size_t) output_channel * out_size + oy) * out_size + ox;
            output[output_idx] = rescale(sum, act_scale, weight_scale);
            //output[output_idx] = sum;
        }
    }
}

void conv_layer(
    const param_t *weights, size_p in_channels, size_p out_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p stride,
    size_p padding,
    size_p act_scale, size_p weight_scale)
{
    assert(out_size == (in_size + 2 * padding - KERNEL_SIZE) / stride + 1);
    for (size_t oc = 0; oc < out_channels; ++oc)
    {
        convolve(weights, in_channels, input, in_size, output, out_size, oc, stride, padding, act_scale, weight_scale);
    }
}

void avgpool(
    const param_t *input, size_p input_size, size_p input_channels,
    param_t *output, size_p output_size)
{
    assert(output_size * POOL_SIZE == input_size);
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
                sum /= POOL_SIZE * POOL_SIZE;
                output[(ic * output_size + (input_y / POOL_SIZE)) * output_size
                    + (input_x / POOL_SIZE)] = sum;
            }
        }
    }
}

void fully_connected(
    const param_t *weights,
    const param_t *input, size_p input_size,
    param_t *output, size_p output_size,
    size_p act_scale, size_p weight_scale)
{
    for (size_t o = 0; o < output_size; ++o)
    {
        int32_t sum = 0;
        for (size_t i = 0; i < input_size; ++i)
        {
            size_t weight_idx = (size_t) o * input_size + i;
            sum += input[i] * weights[weight_idx];
        }
        output[o] = rescale(sum, act_scale, weight_scale);
    }
}

void relu(
    param_t *input, size_p input_size)
{
    for (size_t i = 0; i < input_size; ++i)
    {
        input[i] = input[i] < 0 ? 0 : input[i];
    }
}

param_t argmax(
    const param_t *input, size_p input_size)
{
    int max_index = 0;
    for (size_t i = 0; i < input_size; ++i)
    {
        if (input[i] > input[max_index])
        {
            max_index = i;
        }
    }
    return max_index;
}

param_t rescale(
    int32_t input,
    size_p act_scale,
    size_p weight_scale)
{
      int32_t tmp = llroundf(input * act_scale / weight_scale);
      return clip(tmp, -128, 127);
}
