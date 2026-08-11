#pragma once
// Millisecond timestamp for per-layer profiling.
// The ONLY platform-specific code shared between M1 (host) and M2 (board).

#if defined(_WIN32)

// Native high-res counter; avoids MinGW's libwinpthread DLL dependency
#include <windows.h>
static inline double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return c.QuadPart * 1000.0 / f.QuadPart;
}

#elif defined(__linux__)

#include <time.h>
static inline double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#else  /* bare-metal Zynq (Vitis standalone BSP): 64-bit global timer at CPU/2 */

#include "xtime_l.h"
static inline double now_ms(void)
{
    XTime t;
    XTime_GetTime(&t);
    return (double)t * 1000.0 / COUNTS_PER_SECOND;
}

#endif