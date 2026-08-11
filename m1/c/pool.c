#include "pool.h"
#include "quant.h"

void avgpool(const int8_t *input, int channels, int in_size, int8_t *output)
{
    int out_size = in_size / 2;

    for (int c = 0; c < channels; ++c)
        for (int oy = 0; oy < out_size; ++oy)
            for (int ox = 0; ox < out_size; ++ox) {
                const int8_t *row0 = &input[(c * in_size + 2 * oy) * in_size + 2 * ox];
                const int8_t *row1 = row0 + in_size;
                int32_t sum = row0[0] + row0[1] + row1[0] + row1[1];
                // Mean of four int8 stays in int8 range: no clamp needed
                output[(c * out_size + oy) * out_size + ox] = (int8_t)round_shift(sum, 2);
            }
}