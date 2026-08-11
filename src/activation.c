#include "activation.h"

void relu(param_t *data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}