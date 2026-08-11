#include <stdio.h>
#include <time.h>
#include "lenet1.h"
#include "testdata.h"

#define IMG_PIXELS (IMG_SIZE * IMG_SIZE)
#define TIMING_RUNS 100

static int argmax(const int8_t *v, int len)
{
    int best = 0;
    for (int i = 1; i < len; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

int main(void)
{
    int8_t logits[NUM_CLASSES];
    int exact = 0, correct = 0;

    // Correctness: bit-exact against Python golden logits, plus label accuracy
    for (int i = 0; i < NUM_IMAGES; ++i) {
        predict(&TEST_IMAGES[i * IMG_PIXELS], logits);

        int match = 1;
        for (int c = 0; c < NUM_CLASSES; ++c)
            if (logits[c] != GOLDEN_LOGITS[i * NUM_CLASSES + c]) match = 0;
        exact += match;
        correct += argmax(logits, NUM_CLASSES) == TEST_LABELS[i];
        if (!match)
            printf("Image %d: logits differ from golden\n", i);
    }
    printf("Golden match: %d/%d  accuracy: %d/%d\n", exact, NUM_IMAGES, correct, NUM_IMAGES);

    // Timing: average over many runs (single run is below timer resolution)
    clock_t start = clock();
    for (int r = 0; r < TIMING_RUNS; ++r)
        for (int i = 0; i < NUM_IMAGES; ++i)
            predict(&TEST_IMAGES[i * IMG_PIXELS], logits);
    double total = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Avg inference: %.3f ms over %d runs\n",
           total * 1000.0 / (TIMING_RUNS * NUM_IMAGES), TIMING_RUNS * NUM_IMAGES);

    double layer_ms[NUM_LAYERS] = {0};
    for (int r = 0; r < TIMING_RUNS; ++r)
        for (int i = 0; i < NUM_IMAGES; ++i)
            predict_profiled(&TEST_IMAGES[i * IMG_PIXELS], logits, layer_ms);
    printf("Per-layer average:\n");
    for (int l = 0; l < NUM_LAYERS; ++l)
        printf("  %-6s %8.4f ms\n", LAYER_NAMES[l], layer_ms[l] / (TIMING_RUNS * NUM_IMAGES));

    return exact == NUM_IMAGES ? 0 : 1;
}