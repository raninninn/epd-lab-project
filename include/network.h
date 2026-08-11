#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>
#include <stddef.h>
#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"

#define MIN_PARAM -128
#define MAX_PARAM 127

static inline param_t clip(int32_t val, int32_t min, int32_t max) {
    if (val > max) return (param_t)max;
    if (val < min) return (param_t)min;
    return (param_t)val;
}

param_t inference(param_t *input);

// Profiling prototypes
void reset_profiling_timers(void);
void print_profiling_report(int num_runs);

#endif // NETWORK_H