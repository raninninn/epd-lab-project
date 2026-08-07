#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include "parameters.h"

// Pytorch uses banker's rounding when it divided integers. This differs from
// C's default behaviour of always rounding towards 0.
int32_t div_bankers_rounding(int32_t dividend, int32_t divisor);

/*
 * An argmax implementation
 *
 * @param input: the input, as a flat array
 * @param input_size: the total size of the input
 *
 * @return The index i, such that input[i] is maximum in the input
 */
size_p argmax(const param_t *input, size_p input_size);
#endif
