#include "fc.h"

static inline param_t clip_param(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (param_t)val;
}

// same round-half-to-even shift as applied to conv.c
static inline int32_t round_half_even_shift(int32_t val, int32_t shift) {
    if (shift <= 0) {
        return val;
    }
    int32_t floor_val = val >> shift;
    int32_t mask       = (1 << shift) - 1;
    int32_t remainder  = val & mask;
    int32_t half       = 1 << (shift - 1);
    if (remainder > half || (remainder == half && (floor_val & 1))) {
        floor_val += 1;
    }
    return floor_val;
}

void fully_connected_layer(
    const param_t *weights,
    const param_t *input,
    size_p num_inputs,
    param_t *output,
    size_p num_outputs,
    size_p input_shift,
    size_p output_shift,
    size_p weight_scale   // this is now effectively a count for bits to be shifted because of PoT quantization
) {
    (void)input_shift;
    (void)output_shift;

    for (size_p out = 0; out < num_outputs; ++out) {
        int32_t accumulator = 0;
        for (size_p in = 0; in < num_inputs; ++in) {
            int32_t weight_val = (int32_t)weights[out * num_inputs + in];
            int32_t input_val  = (int32_t)input[in];
            accumulator += input_val * weight_val;
        }
        int32_t scaled_acc = round_half_even_shift(accumulator, (int32_t)weight_scale);
        output[out] = clip_param(scaled_acc);
    }
}

param_t argmax(const param_t *output, size_p num_classes) {
    if (num_classes == 0) return -1;
    param_t max_idx = 0;
    param_t max_val = output[0];
    for (size_p i = 1; i < num_classes; ++i) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = (param_t)i;
        }
    }
    return max_idx;
}