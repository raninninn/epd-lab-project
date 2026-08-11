#include "conv.h"

// Clamps a 32-bit accumulated value into the signed 8-bit range [-128, 127]
static inline param_t clip_param(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (param_t)val;
}

// Rescales a 32-bit accumulator by an arithmetic right-shift using
// round-half-to-even ("banker's rounding"). This matches the rounding
// mode PyTorch's torch.round() uses (and therefore Brevitas's default
// Int8WeightPerTensorPoT / Int8ActPerTensorPoT quantizers, which the
// golden Python model is built from). Plain truncating division
// (accumulator / (1 << shift)) is NOT equivalent: it rounds every
// non-zero remainder toward zero, not just exact ties, which biases
// every output toward zero rather than only breaking ties.
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

void convolute(
    const param_t *weights, size_p input_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p output_channel,
    size_p stride,
    size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_shift
) {
    (void)act_scale_a;
    (void)act_scale_b;

    // LeNet-1 kernel dimension (5x5 filter window)
    const size_p KERNEL_SIZE = 5;

    // Iterate through every row and column of the output feature map
    for (size_p out_y = 0; out_y < out_size; ++out_y) {
        for (size_p out_x = 0; out_x < out_size; ++out_x) {

            int32_t accumulator = 0;

            // Loop over all input channels
            for (size_p in_ch = 0; in_ch < input_channels; ++in_ch) {

                // Loop over the 5x5 spatial kernel grid
                for (size_p ky = 0; ky < KERNEL_SIZE; ++ky) {
                    for (size_p kx = 0; kx < KERNEL_SIZE; ++kx) {

                        // Calculate matching spatial indices on the input image
                        int32_t in_y = (int32_t)(out_y * stride + ky) - (int32_t)padding;
                        int32_t in_x = (int32_t)(out_x * stride + kx) - (int32_t)padding;

                        // Check padding boundary conditions
                        if (in_y >= 0 && in_y < (int32_t)in_size && in_x >= 0 && in_x < (int32_t)in_size) {

                            // Flat indexing for input tensor: [in_ch][in_y][in_x]
                            size_t input_idx = ((size_t)in_ch * in_size * in_size) +
                                               ((size_t)in_y * in_size) +
                                               (size_t)in_x;

                            // Flat indexing for 4D weight tensor: [output_channel][in_ch][ky][kx]
                            size_t weight_idx = ((size_t)output_channel * input_channels * KERNEL_SIZE * KERNEL_SIZE) +
                                                ((size_t)in_ch * KERNEL_SIZE * KERNEL_SIZE) +
                                                ((size_t)ky * KERNEL_SIZE) +
                                                (size_t)kx;

                            int32_t pixel_val  = (int32_t)input[input_idx];
                            int32_t weight_val = (int32_t)weights[weight_idx];

                            accumulator += pixel_val * weight_val;
                        }
                    }
                }
            }

            // Fixed-Point Scaling: round-half-to-even right shift (see above)
            int32_t scaled_acc = round_half_even_shift(accumulator, (int32_t)weight_shift);

            // Flat indexing for 3D output tensor: [output_channel][out_y][out_x]
            size_t output_idx = ((size_t)output_channel * out_size * out_size) +
                                ((size_t)out_y * out_size) +
                                (size_t)out_x;

            // Clamp down to 8-bit parameter domain and store
            output[output_idx] = clip_param(scaled_acc);
        }
    }
}

void convolutional_layer(
    const param_t *weights, size_p in_channels, size_p out_channels,
    const param_t *input, size_p in_size,
    param_t *output, size_p out_size,
    size_p stride, size_p padding,
    size_p act_scale_a, size_p act_scale_b,
    size_p weight_shift
) {
    // Iterate over all output feature maps and compute the convolution for each
    for (size_p oc = 0; oc < out_channels; ++oc) {
        convolute(
            weights, in_channels,
            input, in_size,
            output, out_size,
            oc,
            stride, padding,
            act_scale_a, act_scale_b,
            weight_shift
        );
    }
}
