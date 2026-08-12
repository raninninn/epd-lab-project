#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include "scaling.h"
#include "conv.h"
#define KERNEL_SIZE 5

// There are many more efficient ways to do convolution, but that would be out
// of scope.
void convolve(
    const param_t *weights, size_p in_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p output_channel,
    size_p stride,
    size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale)
{
    for (size_t oy = 0; oy < out_size; ++oy)
        for (size_t ox = 0; ox < out_size; ++ox) {
            int32_t sum = 0;
            for (size_t ic = 0; ic < in_channels; ++ic) {
                for (size_t ky = 0; ky < KERNEL_SIZE; ++ky) {
                    for (size_t kx = 0; kx < KERNEL_SIZE; ++kx) {
                        int in_y = oy * stride + ky - padding; 
                        int in_x = ox * stride + kx - padding;

                        // Zero padding
                        if (in_y < 0 || in_y >= in_size ||
                            in_x < 0 || in_x >= in_size)
                                continue;

                        size_t input_idx = ((size_t) ic * in_size + in_y) * in_size + in_x;

                        size_t weight_idx = (((size_t) output_channel * in_channels + ic)
                             * KERNEL_SIZE + ky)
                             * KERNEL_SIZE + kx;

                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
            size_t output_idx = ((size_t) output_channel * out_size + oy) * out_size + ox;
            output[output_idx] = rescale(sum, act_scale_a, act_scale_b, weight_scale);
        }
}

void conv_layer(const param_t *weights, size_p in_channels, size_p out_channels,
                const param_t *input, size_p in_size,
                param_t *output, size_p out_size,
                size_p stride,
                size_p padding,
                size_p act_scale_a, size_p act_scale_b,
                size_p weight_scale)
{
    assert(out_size == (in_size + 2 * padding - KERNEL_SIZE) / stride + 1);
    for (size_t oc = 0; oc < out_channels; ++oc)
    {
        convolve(weights, in_channels,
            input, in_size,
            output, out_size, oc,
            stride, padding,
            act_scale_a, act_scale_b, weight_scale);
    }
}
