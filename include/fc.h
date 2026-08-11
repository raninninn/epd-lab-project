#ifndef FC_H
#define FC_H

#include <stdint.h>
#include <stddef.h>

typedef uint16_t size_p;
typedef int8_t param_t;

void fully_connected_layer(
    const param_t *weights,
    const param_t *input,
    size_p num_inputs,
    param_t *output,
    size_p num_outputs,
    size_p input_shift,
    size_p output_shift,
    size_p weight_scale
);


param_t argmax(const param_t *output, size_p num_classes);

#endif // FC_H