#include <stddef.h>
#include "relu.h"

void relu(param_t *input, size_p input_size)
{
    for (size_t i = 0; i < input_size; ++i)
        input[i] = input[i] < 0 ? 0 : input[i];
}
