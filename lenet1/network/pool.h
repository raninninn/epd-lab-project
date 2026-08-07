#include <stdint.h>
#include "parameters.h"

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
