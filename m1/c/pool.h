#pragma once
#include <stdint.h>

// 2x2 average pool, stride 2: out_size = in_size / 2.
// Flat channels-first layout, same as conv.
void avgpool(const int8_t *input, int channels, int in_size, int8_t *output);