#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define C10_CUDA_CHECK(EXPR)                                                   \
    do {                                                                       \
        cudaError_t ferrum_cuda_status_ = (EXPR);                              \
        if (ferrum_cuda_status_ != cudaSuccess) {                              \
            std::fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(ferrum_cuda_status_));   \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaPeekAtLastError())
