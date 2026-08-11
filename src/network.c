#include "network.h"
#include <stdio.h>
#include <time.h>

#include "weights_export/weights_c1.h"
#include "weights_export/weights_c3.h"
#include "weights_export/weights_out.h"
#include "weights_export/act_scales.h"

#define INPUT_SIZE       (28 * 28)
#define C1_OUT_SIZE      (24 * 24 * 4)
#define S2_OUT_SIZE      (12 * 12 * 4)
#define C3_OUT_SIZE      (8 * 8 * 12)
#define S4_OUT_SIZE      (4 * 4 * 12)
#define FC_OUT_SIZE      10

// global accumulators for layer profiling (in milliseconds)
double g_time_c1 = 0.0;
double g_time_s2 = 0.0;
double g_time_c3 = 0.0;
double g_time_s4 = 0.0;
double g_time_fc = 0.0;

// helper function to calculate time difference in milliseconds
static inline double diff_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

// reset timers before starting a profiling loop
void reset_profiling_timers(void) {
    g_time_c1 = 0.0;
    g_time_s2 = 0.0;
    g_time_c3 = 0.0;
    g_time_s4 = 0.0;
    g_time_fc = 0.0;
}

// print the accumulated layer breakdown
void print_profiling_report(int num_runs) {
    double total = g_time_c1 + g_time_s2 + g_time_c3 + g_time_s4 + g_time_fc;
    
    printf("\n================ Layer Breakdown (Avg over %d runs) ================\n", num_runs);
    printf("Layer C1 (Conv 5x5, 4 maps)  : %8.4f ms  (%5.1f%%)\n", g_time_c1 / num_runs, (g_time_c1 / total) * 100.0);
    printf("Layer S2 (AvgPool 2x2)       : %8.4f ms  (%5.1f%%)\n", g_time_s2 / num_runs, (g_time_s2 / total) * 100.0);
    printf("Layer C3 (Conv 5x5, 12 maps) : %8.4f ms  (%5.1f%%)\n", g_time_c3 / num_runs, (g_time_c3 / total) * 100.0);
    printf("Layer S4 (AvgPool 2x2)       : %8.4f ms  (%5.1f%%)\n", g_time_s4 / num_runs, (g_time_s4 / total) * 100.0);
    printf("Layer FC (Dense 192->10)     : %8.4f ms  (%5.1f%%)\n", g_time_fc / num_runs, (g_time_fc / total) * 100.0);
    printf("-------------------------------------------------------------------\n");
    printf("Total Pipeline Latency       : %8.4f ms  (100.0%%)\n", total / num_runs);
    printf("===================================================================\n\n");
}

param_t inference(param_t *input) {
    struct timespec t_start, t_end;

    param_t c1_out[C1_OUT_SIZE];
    param_t s2_out[S2_OUT_SIZE];
    param_t c3_out[C3_OUT_SIZE];
    param_t s4_out[S4_OUT_SIZE];
    param_t output[FC_OUT_SIZE];

    // Input Quantization / Scaling
    for (int i = 0; i < INPUT_SIZE; ++i) {
        int32_t scaled = (int32_t)input[i] >> INPUT_ACT_SHIFT;
        input[i] = clip(scaled, MIN_PARAM, MAX_PARAM);
    }

    // Layer C1
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    convolutional_layer(WEIGHTS_C1, 1, 4, input, 28, c1_out, 24, 1, 0, INPUT_ACT_SHIFT, FIRST_ACT_SHIFT, INPUT_ACT_SHIFT + WEIGHTS_C1_SCALE - FIRST_ACT_SHIFT);
    relu(c1_out, C1_OUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_time_c1 += diff_ms(t_start, t_end);

    // Layer S2
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    average_pooling_layer(c1_out, 24, 4, s2_out, 12);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_time_s2 += diff_ms(t_start, t_end);

    // Layer C3
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    convolutional_layer(WEIGHTS_C3, 4, 12, s2_out, 12, c3_out, 8, 1, 0, FIRST_ACT_SHIFT, SECOND_ACT_SHIFT, FIRST_ACT_SHIFT + WEIGHTS_C3_SCALE - SECOND_ACT_SHIFT);
    relu(c3_out, C3_OUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_time_c3 += diff_ms(t_start, t_end);

    // Layer S4
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    average_pooling_layer(c3_out, 8, 12, s4_out, 4);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_time_s4 += diff_ms(t_start, t_end);

    // Layer FC
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    fully_connected_layer(WEIGHTS_OUT, s4_out, S4_OUT_SIZE, output, FC_OUT_SIZE, SECOND_ACT_SHIFT, OUTPUT_ACT_SHIFT, SECOND_ACT_SHIFT + WEIGHTS_OUT_SCALE - OUTPUT_ACT_SHIFT);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    g_time_fc += diff_ms(t_start, t_end);

    return argmax(output, FC_OUT_SIZE);
}