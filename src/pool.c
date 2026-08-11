#include "pool.h"

static inline int32_t round_half_even_shift(int32_t val, int32_t shift) {
    if (shift <= 0) {
        return val;
    }

    // Arithmetic right shift of a signed integer floors toward -infinity
    // on every toolchain used in this project (GCC / arm-none-eabi-gcc),
    // so this gives floor(val / 2^shift).
    int32_t floor_val = val >> shift;
    int32_t mask       = (1 << shift) - 1;
    int32_t remainder  = val & mask;       // always in [0, 2^shift - 1]
    int32_t half       = 1 << (shift - 1);

    if (remainder > half || (remainder == half && (floor_val & 1))) {
        floor_val += 1;
    }
    return floor_val;
}

void average_pooling_layer(
    const param_t *input, size_p in_size, size_p channels,
    param_t *output, size_p out_size
) {
    // LeNet-1 uses 2x2 average pooling with a stride of 2
    const size_p POOL_SIZE = 2;
    const size_p STRIDE = 2;

    // Iterate over all feature maps (channels)
    for (size_p ch = 0; ch < channels; ++ch) {

        // Iterate through every row and column of the output feature map
        for (size_p out_y = 0; out_y < out_size; ++out_y) {
            for (size_p out_x = 0; out_x < out_size; ++out_x) {

                int32_t sum = 0;

                // Loop over the 2x2 pooling window
                for (size_p py = 0; py < POOL_SIZE; ++py) {
                    for (size_p px = 0; px < POOL_SIZE; ++px) {

                        // Calculate corresponding spatial coordinates on the input map
                        size_p in_y = (out_y * STRIDE) + py;
                        size_p in_x = (out_x * STRIDE) + px;

                        // Flat indexing for 3D input tensor: [channel][in_y][in_x]
                        size_t input_idx = ((size_t)ch * in_size * in_size) +
                                           ((size_t)in_y * in_size) +
                                           (size_t)in_x;

                        sum += (int32_t)input[input_idx];
                    }
                }

                // Calculate the average (divide by 4)
                int32_t average = round_half_even_shift(sum, 2);

                // Flat indexing for 3D output tensor: [channel][out_y][out_x]
                size_t output_idx = ((size_t)ch * out_size * out_size) +
                                    ((size_t)out_y * out_size) +
                                    (size_t)out_x;

                // Cast safely back to 8-bit parameter domain and store
                output[output_idx] = (param_t)average;
            }
        }
    }
}
