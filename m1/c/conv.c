#include "conv.h"
#include "quant.h"

#define K 5   // kernel size, fixed for LeNet-1

void conv_layer(const int8_t *input, int in_ch, int in_size,
                const int8_t *weight, const int8_t *bias, int out_ch,
                int shift_in, int shift_weight, int shift_bias, int shift_out,
                int8_t *output)
{
    int out_size = in_size - (K - 1);

    for (int oc = 0; oc < out_ch; ++oc) {
        // Bias was quantized with its own shift; align it to the
        // accumulator scale 2^(shift_in + shift_weight)
        int32_t b = (int32_t)bias[oc] << (shift_in + shift_weight - shift_bias);

        for (int oy = 0; oy < out_size; ++oy) {
            for (int ox = 0; ox < out_size; ++ox) {
                int32_t acc = b;
                for (int ic = 0; ic < in_ch; ++ic)
                    for (int ky = 0; ky < K; ++ky)
                        for (int kx = 0; kx < K; ++kx)
                            acc += (int32_t)input[(ic * in_size + oy + ky) * in_size + ox + kx]
                                 * weight[((oc * in_ch + ic) * K + ky) * K + kx];
                output[(oc * out_size + oy) * out_size + ox] =
                    requantize(acc, shift_in + shift_weight, shift_out);
            }
        }
    }
}