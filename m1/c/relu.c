#include "relu.h"

void relu(int8_t *x, int len)
{
    for (int i = 0; i < len; ++i)
        if (x[i] < 0) x[i] = 0;
}