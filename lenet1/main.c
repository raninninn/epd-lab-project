#include <stdio.h>

#include "network.h"
#include "../weight_export/weights_c1.h" // contains weights
#include "../weight_export/weights_c3.h" // contains weights
#include "../weight_export/weights_out.h" // contains weights
//#include "../data_export/inputs/test_0.h" // contains inputs
#include "test.h"

#define FIRST_LAYER_ACT_SCALE 1 << 3
#define OTHER_LAYER_ACT_SCALE 1 << 2

param_t inference(param_t *input)
{
    param_t c1_out[24 * 24 * 4];
    param_t s2_out[12 * 12 * 4];
    param_t c3_out[8 * 8 * 12];
    param_t s4_out[4 * 4 * 12];
    param_t output[10];
    conv_layer(WEIGHTS_C1, 1, 4, input, 28, c1_out, 24, 1, 0, FIRST_LAYER_ACT_SCALE / OTHER_LAYER_ACT_SCALE, 1 << WEIGHTS_C1_SCALE);
    /*for (size_t i = 0; i < 24 * 24 * 4; ++i)
      printf("%d vs %d\n", c1_out[i], C1_OUT[i]);
    return 1;*/
    relu(c1_out, 24 * 24 * 4);
    avgpool(c1_out, 24, 4, s2_out, 12);
    conv_layer(WEIGHTS_C3, 4, 12, s2_out, 12, c3_out, 8, 1, 0, 1, 1 << WEIGHTS_C3_SCALE);
    relu(c3_out, 8 * 8 * 12);
    avgpool(c3_out, 8, 12, s4_out, 4);
    fully_connected(WEIGHTS_OUT, s4_out, 4 * 4 * 12, output, 10, 1, 1 << WEIGHTS_OUT_SCALE);
    return argmax(output, 10);
}

int main(int argc, char argv[])
{
    //printf("Output: expected %d, got %d", LABEL_TEST_0, inference(INPUT_TEST_0));
    printf("Output: expected %d, got %d\n", LABEL, inference(INPUT));
}
