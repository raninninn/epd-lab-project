#include <stdint.h>
#include "parameters.h"

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
void fully_connected(const param_t *weights,
                     const param_t *input, size_p input_size,
                     param_t *output, size_p output_size,
                     size_p act_scale_a, size_p act_scale_b,
                     size_p weight_scale);
