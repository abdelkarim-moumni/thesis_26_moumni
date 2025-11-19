#pragma once

#include <cstdio>

#define CUDA_CHECK_EXIT(a)                                                                                             \
    {                                                                                                                  \
        cudaError_t ok = a;                                                                                            \
        if (ok != cudaSuccess) {                                                                                       \
            fprintf(stderr, "cuhear: CUDA error @ %s:%d (%s)\n", __FILE__, __LINE__, cudaGetErrorString(ok));          \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

#define CUDA_LAST_EXIT()                                                                                               \
    {                                                                                                                  \
        cudaError_t ok = cudaGetLastError();                                                                           \
        if (ok != cudaSuccess) {                                                                                       \
            fprintf(stderr, "cuhear: CUDA kernel error @ %s:%d (%s)\n", __FILE__, __LINE__, cudaGetErrorString(ok));   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }
    