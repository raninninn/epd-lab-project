#ifndef POOL_H
#define POOL_H

#include <stdint.h>
#include <stddef.h>

typedef uint16_t size_p;
typedef int8_t param_t;

void average_pooling_layer(
    const param_t *input, size_p in_size, size_p channels, 
    param_t *output, size_p out_size
);

#endif // POOL_H