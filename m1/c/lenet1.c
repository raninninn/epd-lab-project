#include "timer.h"
#include "lenet1.h"
#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "weights.h"
#include "relu.h"

const char *LAYER_NAMES[NUM_LAYERS] =
{"conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "fc"};

// Weight storage indirection: defaults to the compiled-in arrays; M2 repoints
// these into BRAM after the one-time copy. Kernels are unaffected.
static const int8_t *w_conv1 = CONV1_WEIGHT, *b_conv1 = CONV1_BIAS;
static const int8_t *w_conv2 = CONV2_WEIGHT, *b_conv2 = CONV2_BIAS;
static const int8_t *w_fc    = FC_WEIGHT,    *b_fc    = FC_BIAS;

void lenet1_use_weights(const int8_t *conv1_w, const int8_t *conv1_b,
                        const int8_t *conv2_w, const int8_t *conv2_b,
                        const int8_t *fc_w,    const int8_t *fc_b)
{
    w_conv1 = conv1_w; b_conv1 = conv1_b;
    w_conv2 = conv2_w; b_conv2 = conv2_b;
    w_fc    = fc_w;    b_fc    = fc_b;
}

// Record a lap time for layer i (no-op when ms is NULL)
#define LAP(i) if (ms) { double t2 = now_ms(); ms[i] += t2 - t; t = t2; }

void predict_profiled(const int8_t *image, int8_t *logits, double *ms)
{
    int8_t conv1_out[4 * 24 * 24];
    int8_t pool1_out[4 * 12 * 12];
    int8_t conv2_out[12 * 8 * 8];
    int8_t pool2_out[12 * 4 * 4];

    double t = ms ? now_ms() : 0;

    conv_layer(image, 1, 28, w_conv1, b_conv1, 4,
               SHIFT_INPUT, CONV1_WEIGHT_SHIFT, CONV1_BIAS_SHIFT, SHIFT_CONV1,
               conv1_out);
    LAP(0)
    relu(conv1_out, 4 * 24 * 24);
    LAP(1)
    avgpool(conv1_out, 4, 24, pool1_out);
    LAP(2)

    conv_layer(pool1_out, 4, 12, w_conv2, b_conv2, 12,
               SHIFT_CONV1, CONV2_WEIGHT_SHIFT, CONV2_BIAS_SHIFT, SHIFT_CONV2,
               conv2_out);
    LAP(3)
    relu(conv2_out, 12 * 8 * 8);
    LAP(4)
    avgpool(conv2_out, 12, 8, pool2_out);
    LAP(5)

    // flatten is a no-op: pool2_out is already in PyTorch flatten order
    fc_layer(pool2_out, 12 * 4 * 4, w_fc, b_fc, 10,
             SHIFT_CONV2, FC_WEIGHT_SHIFT, FC_BIAS_SHIFT, SHIFT_FC,
             logits);
    LAP(6)
}

void predict(const int8_t *image, int8_t *logits)
{
    predict_profiled(image, logits, NULL);
}