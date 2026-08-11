#pragma once
#include <stdint.h>

// Elementwise max(0, x), in place
void relu(int8_t *x, int len);