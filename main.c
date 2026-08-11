#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>   // need to be adapted to xilinx-specific timer header for M2

#include "include/network.h"
#include "weights_export/test_input_0.h"
#include "weights_export/test_label_0.h"

#define NUM_RUNS 1000

// Helper function to calculate time difference in milliseconds
double get_time_diff_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

int main(void) {
    printf("--- LeNet-1 C Software Model (Milestone 1) ---\n\n");

    // The input buffer must be mutable because network.c modifies it 
    // in-place during the initial input quantization step.
    param_t input_buffer[784];
    param_t predicted_label;

    printf("[1/2] Verifying Functional Accuracy...\n");
    
    // copy input to memory buffer (in RAM) for faster execution
    memcpy(input_buffer, TEST_INPUT_0, sizeof(TEST_INPUT_0));

    predicted_label = inference(input_buffer);

    printf("Expected Label:  %d\n", EXPECTED_LABEL);
    printf("Predicted Label: %d\n", predicted_label);

    if (predicted_label == EXPECTED_LABEL) {
        printf("Status: SUCCESS (Match)\n\n");
    } else {
        printf("Status: FAILED (Mismatch)\n");
        return -1;
    }

    // Profiling
    printf("[2/2] Profiling Execution Time over %d runs...\n", NUM_RUNS);

    reset_profiling_timers();

    struct timespec start, end;
    double total_time_ms = 0.0;

    for (int i = 0; i < NUM_RUNS; ++i) {
        // reset the buffer for each run (since inference modifies it)
        memcpy(input_buffer, TEST_INPUT_0, sizeof(TEST_INPUT_0));

        clock_gettime(CLOCK_MONOTONIC, &start);
        
        inference(input_buffer);
        
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_time_ms += get_time_diff_ms(&start, &end);
    }

    print_profiling_report(NUM_RUNS); // Print breakdown

    double avg_time_ms = total_time_ms / NUM_RUNS;
    
    printf("Total Profiling Time: %.3f ms\n", total_time_ms);
    printf("Average Inference Time: %.4f ms\n", avg_time_ms);
    printf("----------------------------------------------\n");

    return 0;
}