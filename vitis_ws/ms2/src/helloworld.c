#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xil_io.h"
#include "xtime_l.h"

#include "network/network.h"

#include "parameters/weights_c1.h"
#include "parameters/weights_c3.h"
#include "parameters/weights_out.h"
#include "parameters/act_scales.h"
#include "inputs/all_tests.h"

#include "preds.h"

#define C1_BASE XPAR_BRAM_0_BASEADDR
#define C3_BASE XPAR_BRAM_1_BASEADDR
#define FC_BASE XPAR_BRAM_2_BASEADDR

// taken from https://stackoverflow.com/questions/2437283/c-c-packing-signed-char-into-int#2437305
#define PACK(c0, c1, c2, c3) \
    (((uint32_t)(uint8_t)(c0) << 24) | \
    ((uint32_t)(uint8_t)(c1) << 16) | \
    ((uint32_t)(uint8_t)(c2) << 8) | \
    ((uint32_t)(uint8_t)(c3)))

void setup_weights(UINTPTR addr, const int8_t *weights, int size)
{
	int data;
	for (int i = 0; i < size; i += 4) {
		data = PACK(weights[i], weights[i + 1], weights[i + 2], weights[i + 3]);
		Xil_Out32(addr + i, data);
	}
}

param_t inference(param_t *input)
{
    param_t c1_out[24 * 24 * 4];
    param_t s2_out[12 * 12 * 4];
    param_t c3_out[8 * 8 * 12];
    param_t s4_out[4 * 4 * 12];
    param_t output[10];

    for (int i = 0; i < 784; ++i)
    {
        int32_t scaled = clip((int32_t)input[i] * FIRST_ACT_SCALE, MIN_PARAM, MAX_PARAM);
        input[i] = (param_t)scaled;
    }
    conv_layer(C1_BASE, 1, 4, input, 28, c1_out, 24, 1, 0, FIRST_ACT_SCALE, SECOND_ACT_SCALE, 1 << WEIGHTS_C1_SCALE);
    relu(c1_out, 24 * 24 * 4);
    avgpool(c1_out, 24, 4, s2_out, 12);
    conv_layer(C3_BASE, 4, 12, s2_out, 12, c3_out, 8, 1, 0, SECOND_ACT_SCALE, THIRD_ACT_SCALE, 1 << WEIGHTS_C3_SCALE);
    relu(c3_out, 8 * 8 * 12);
    avgpool(c3_out, 8, 12, s4_out, 4);
    fully_connected(FC_BASE, s4_out, 4 * 4 * 12, output, 10, THIRD_ACT_SCALE, FOURTH_ACT_SCALE, 1 << WEIGHTS_OUT_SCALE);
    return argmax(output, 10);
}
int main() {
    init_platform();

    setup_weights(C1_BASE, WEIGHTS_C1, 100);
    setup_weights(C3_BASE, WEIGHTS_C3, 1200);
    setup_weights(FC_BASE, WEIGHTS_OUT, 1920);

    XTime tStart, tEnd;
    XTime_GetTime(&tStart);
    for (int i = 0; i < 1000; i++)
    {
        param_t result = inference(TEST_IMAGES[i]);
        if (result != PREDS[i])
        	xil_printf("%d :   %d vs %d\r\n", i, result, PREDS[i]);
    }
    XTime_GetTime(&tEnd);
    printf("Output took %llu clock cycles.\r\n", 2*(tEnd - tStart));
    printf("Output took %.2f ms.\r\n",
    		1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000));
    cleanup_platform();
    return 0;
}
