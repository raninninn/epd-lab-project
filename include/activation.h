#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdint.h>
#include <stddef.h>

typedef uint16_t size_p;
typedef int8_t param_t;

void relu(param_t *data, size_t length);

#endif // ACTIVATION_H