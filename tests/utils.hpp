#pragma once

#define CHECK_CUDA_CALL(a)                                                                                             \
    {                                                                                                                  \
        cudaError_t ok = a;                                                                                            \
        if (ok != cudaSuccess)                                                                                         \
            fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString(ok));                  \
    }
#define CHECK_CUDA_LAST()                                                                                              \
    {                                                                                                                  \
        cudaError_t ok = cudaGetLastError();                                                                           \
        if (ok != cudaSuccess)                                                                                         \
            fprintf(stderr, "-- Error CUDA %d, last in line %d: %s\n", ok, __LINE__, cudaGetErrorString(ok));                  \
    }
    