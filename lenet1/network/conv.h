#include <stdint.h>
#include "parameters.h"

/*
 * Implements a single filter map. This assumes that a filter map accesses all
 * input channels and that input and output are squares. You probably want to
 * use conv_layer() instead.
 *
 * @param weights: The weights of the whole convolutional layer, as a flat array
 * @param input: The input, as a flat array
 * @param input_channels: The number of input channels for this layer
 * @param in_size: The size of one dimension of the input
 * @param output: The output location
 * @param out_size: The size of one dimension of the output
 * @param output_channel: The index of the output channel this should write to
 * @param stride: Convolution stride
 * @param padding: Convolution padding
 * @param act_scale_a: Quantization scale for the input
 * @param act_scale_b: Quantization scale for the output
 * @param weight_scale: Quantization scale for the weights
 */
void convolve(const param_t *weights, size_p input_channels,
              const param_t *input, size_p in_size,
              param_t *output, size_p out_size,
              size_p output_channel,
              size_p stride,
              size_p padding,
              size_p act_scale_a, size_p act_scale_b,
              size_p weight_scale);

/*
 * Implements a convolutional layer. Inputs and outputs are square-shaped.
 *
 * @param weights: The quantized weights, as a flat array
 * @param in_channels: The number of input channels
 * @param out_channels: The number of output channels
 * @param input: The quantized input, as a flat array
 * @param in_size: The size of the input in one dimension
 * @param output: The output location
 * @param out_size: The size of the output in one location
 * @param stride: The convolution stride
 * @param padding: The convolution padding
 * @param act_scale_a: The input quantization scale factor
 * @param act_scale_b: The output quantization scale factor
 * @param weight_scale: The weight quantization scale factor
 */
void conv_layer(const param_t *weights, size_p in_channels, size_p out_channels,
                const param_t *input, size_p in_size,
                param_t *output, size_p out_size,
                size_p stride,
                size_p padding,
                size_p act_scale_a, size_p act_scale_b,
                size_p weight_scale);

