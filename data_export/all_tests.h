#ifndef ALL_TESTS_H
#define ALL_TESTS_H

#include <stdint.h>
#include "network/parameters.h"   // for param_t

#define NUM_TESTS 10000
#define IMAGE_SIZE 784

extern param_t TEST_IMAGES[NUM_TESTS][IMAGE_SIZE];
extern size_p TEST_LABELS[NUM_TESTS];

#endif
