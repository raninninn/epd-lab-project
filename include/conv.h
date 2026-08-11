#ifndef CONV_H
#define CONV_H

#include <stdint.h>
#include <stddef.h>

typedef uint16_t size_p;
typedef int8_t param_t;

void convolute(
    const param_t *weights, size_p input_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p output_channel,
    size_p stride,
    size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale
);

void convolutional_layer(
    const param_t *weights, size_p in_channels, size_p out_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p stride, size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale
);

#endif // CONV_H