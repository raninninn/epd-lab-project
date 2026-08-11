#include "fc.h"
#include "quant.h"

void fc_layer(const int8_t *input, int in_len,
              const int8_t *weight, const int8_t *bias, int out_len,
              int shift_in, int shift_weight, int shift_bias, int shift_out,
              int8_t *output)
{
    for (int o = 0; o < out_len; ++o) {
        // Align bias to the accumulator scale 2^(shift_in + shift_weight)
        int32_t acc = (int32_t)bias[o] << (shift_in + shift_weight - shift_bias);
        for (int i = 0; i < in_len; ++i)
            acc += (int32_t)input[i] * weight[o * in_len + i];
        output[o] = requantize(acc, shift_in + shift_weight, shift_out);
    }
}