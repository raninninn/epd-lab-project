#pragma once
#include <stdint.h>

// 5x5 convolution, stride 1, no padding: out_size = in_size - 4.
// Flat channels-first layout (PyTorch memory order):
//   input  [in_ch][in_size][in_size]
//   weight [out_ch][in_ch][5][5]
//   bias   [out_ch]
//   output [out_ch][out_size][out_size]
void conv_layer(const int8_t *input, int in_ch, int in_size,
                const int8_t *weight, const int8_t *bias, int out_ch,
                int shift_in, int shift_weight, int shift_bias, int shift_out,
                int8_t *output);