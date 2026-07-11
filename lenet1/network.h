#include <stdint.h>

typedef uint16_t size_p;
typedef int8_t param_t;

void convolve(
    const param_t *weights, size_p input_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p output_channel,
    size_p stride,
    size_p padding,
    size_p act_scale,
    size_p weight_scale
    );

void avgpool(
    const param_t *input, size_p input_size, size_p input_channels,
    param_t *output, size_p output_size
    );

void fully_connected(
    const param_t *weights,
    const param_t *input, size_p input_size,
    param_t *output, size_p output_size,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale
    );

void conv_layer(
    const param_t *weights, size_p in_channels, size_p out_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p stride,
    size_p padding,
    size_p act_scale,
    size_p weight_scale
    );

// Warning: This is done in-place
void relu(
    param_t *input, size_p input_size
    );

param_t argmax(
    const param_t *input, size_p input_size
    );

param_t rescale(
    int32_t input,
    size_p act_scale_a,
    size_p act_scale_b,
    size_p weight_scale);
