#pragma once
#include <stdint.h>

#define IMG_SIZE 28
#define NUM_CLASSES 10
#define NUM_LAYERS 7   // conv1, relu1, pool1, conv2, relu2, pool2, fc

// Full int8 forward pass: 28x28 int8 image (scale 2^SHIFT_INPUT) -> 10 logits
void predict(const int8_t *image, int8_t *logits);

extern const char *LAYER_NAMES[NUM_LAYERS];

// Like predict, but adds each layer's runtime to ms[NUM_LAYERS]
void predict_profiled(const int8_t *image, int8_t *logits, double *ms);

// Repoint the weight/bias arrays used by predict (default: the compiled-in
// arrays from weights.h). M2 calls this after the one-time copy into BRAM.
void lenet1_use_weights(const int8_t *conv1_w, const int8_t *conv1_b,
                        const int8_t *conv2_w, const int8_t *conv2_b,
                        const int8_t *fc_w,    const int8_t *fc_b);