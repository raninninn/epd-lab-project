#pragma once
#include <stdint.h>

// Fully connected: output[o] = sum_i input[i] * weight[o][i] + bias[o]
// weight layout [out_len][in_len] (PyTorch nn.Linear memory order)
void fc_layer(const int8_t *input, int in_len,
              const int8_t *weight, const int8_t *bias, int out_len,
              int shift_in, int shift_weight, int shift_bias, int shift_out,
              int8_t *output);