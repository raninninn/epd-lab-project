#include <stdint.h>
#include <xil_types.h>

#define MAX_PARAM 127
#define MIN_PARAM -128

typedef uint16_t size_p;
typedef int8_t param_t;

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
void convolve(
    const param_t *weights, size_p input_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p output_channel,
    size_p stride,
    size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale
    );

/*
 * Implements an average pooling layer. Note that the averaging uses banker's
 * rounding, to mimic the default PyTorch behaviour. This implementation
 * assumes square-shaped inputs and outputs, as well as a stride equal to the
 * kernel size.
 *
 * @param input: The input, as a flat array
 * @param input_size: The size of one dimension of the input
 * @param input_channels: The number of input channels
 * @param output: The output location
 * @param output_size: The size of the output in one dimension
 */
void avgpool(
    const param_t *input, size_p input_size, size_p input_channels,
    param_t *output, size_p output_size
    );

/*
 * This implements a fully connected layer.
 * inputs and outputs
 *
 * @param weights: The weights to the layer, as a flat array
 * @param input: The input to the layer, as a flat array
 * @param input_size: The total size of the input
 * @param output: The output location
 * @param output_size: The total size of the output
 * @param act_scale_a: The quantization scaling factor for the input
 * @param act_scale_b: The quantization scaling factor for the output
 * @param weight_scale: The quantization scaling factor for the weights
 */
void fully_connected(
    const UINTPTR weights,
    const param_t *input, size_p input_size,
    param_t *output, size_p output_size,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale
    );

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
void conv_layer(
    const UINTPTR weights, size_p in_channels, size_p out_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p stride,
    size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_scale
    );

/*
 * An in-place ReLU implementation.
 *
 * @param input: The input, as a flat array
 * @param input_size: The total size of the input
 */
void relu(
    param_t *input, size_p input_size
    );

/*
 * An argmax implementation.
 *
 * @param input: The input, as a flat array
 * @param input_size: The total size of the input
 *
 * @return The index i, such that input[i] is maximum in the input
 */
param_t argmax(
    const param_t *input, size_p input_size
    );

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
