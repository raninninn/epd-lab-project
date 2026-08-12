#include <stdint.h>
#include <stddef.h>
#include "utils.h"

int32_t div_bankers_rounding(int32_t dividend, int32_t divisor)
{
    int32_t quotient = dividend / divisor;
    int32_t remainder = dividend % divisor;
    if (remainder * 2 > divisor || (remainder * 2 == divisor && (quotient % 2 != 0))) {
        quotient++;
    }
    return quotient;
}

size_p argmax(
    const param_t *input, size_p input_size)
{
    param_t max_index = 0;
    for (size_t i = 0; i < input_size; ++i) {
        if (input[i] > input[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}
