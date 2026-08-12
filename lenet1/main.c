#include <stdio.h>
#include <time.h>

#include "network.h"
#include "../weight_export/weights_c1.h" // contains weights
#include "../weight_export/weights_c3.h" // contains weights
#include "../weight_export/weights_out.h" // contains weights
#include "../weight_export/act_scales.h" // contains quantization scales for activations
#include "../data_export/all_tests.h"

// These includes contain comparison outputs for each layer. They can be gene-
// rated for a specific input by network.py.
#include "../weight_export/c1_out.h"
#include "../weight_export/s2_out.h"
#include "../weight_export/c3_out.h"
#include "../weight_export/s4_out.h"
#include "../weight_export/fc_out.h"

#define NUM_RUNS 2000 // Change to 10000 to test all images

size_p inference(param_t *input)
{
    param_t c1_out[24 * 24 * 4];
    param_t s2_out[12 * 12 * 4];
    param_t c3_out[8 * 8 * 12];
    param_t s4_out[4 * 4 * 12];
    param_t output[10];

    for (int i = 0; i < 784; ++i) {
        int32_t scaled = clip((int32_t)input[i] * FIRST_ACT_SCALE, MIN_PARAM, MAX_PARAM);
        input[i] = (param_t)scaled;
    }
    conv_layer(WEIGHTS_C1, 1, 4, input, 28, c1_out, 24, 1, 0, FIRST_ACT_SCALE, SECOND_ACT_SCALE, 1 << WEIGHTS_C1_SCALE);
    relu(c1_out, 24 * 24 * 4);
#ifdef DEBUG_C1
    for (size_t i = 0; i < 24 * 24 * 4; ++i)
        printf("%d vs %d, %s\n", c1_out[i], C1_OUT[i], (c1_out[i] == C1_OUT[i]) ? "true" : "FALSE");
#endif
    avgpool(c1_out, 24, 4, s2_out, 12);
#ifdef DEBUG_S2
    for (size_t i = 0; i < 12 * 12 * 4; ++i)
        printf("%d vs %d, %s\n", s2_out[i], S2_OUT[i], (s2_out[i] == S2_OUT[i]) ? "true" : "FALSE");
#endif
    conv_layer(WEIGHTS_C3, 4, 12, s2_out, 12, c3_out, 8, 1, 0, SECOND_ACT_SCALE, THIRD_ACT_SCALE, 1 << WEIGHTS_C3_SCALE);
    relu(c3_out, 8 * 8 * 12);
#ifdef DEBUG_C3
    for (size_t i = 0; i < 8 * 8 * 12; ++i)
        printf("%d vs %d, %s\n", c3_out[i], C3_OUT[i], (c3_out[i] == C3_OUT[i]) ? "true" : "FALSE");
#endif
    avgpool(c3_out, 8, 12, s4_out, 4);
#ifdef DEBUG_S4
    for (size_t i = 0; i < 4 * 4 * 12; ++i)
        printf("%d vs %d, %s\n", s4_out[i], S4_OUT[i], (s4_out[i] == S4_OUT[i]) ? "true" : "FALSE");
#endif
    fully_connected(WEIGHTS_OUT, s4_out, 4 * 4 * 12, output, 10, THIRD_ACT_SCALE, FOURTH_ACT_SCALE, 1 << WEIGHTS_OUT_SCALE);
#ifdef DEBUG_FC
    for (size_t i = 0; i < 10; ++i)
        printf("%d vs %d, %s\n", output[i], FC_OUT[i], (output[i] == FC_OUT[i]) ? "true" : "FALSE");
#endif
    return argmax(output, 10);
}

int main(int argc, char argv[])
{
    int sum = 0;
    clock_t start = clock();
    for (int i = 0; i < NUM_RUNS; ++i) {
        size_p result = inference(TEST_IMAGES[i]);
        sum += result == TEST_LABELS[i];
#ifdef PRINT_LABELS
        printf("Output: expected %d, got %d\n", TEST_LABELS[i], result);
#endif
    }
    clock_t end = clock();
    double total = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Avg. runtime per image: %.3f\n", total * 1000 / NUM_RUNS);
#ifndef PRINT_LABELS
    printf("Accuracy: %f\n", (float)sum / NUM_RUNS);
#endif
}
