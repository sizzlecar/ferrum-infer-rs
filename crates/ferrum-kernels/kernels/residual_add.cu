// Element-wise residual add: output[i] = a[i] + b[i]

#include <cuda_fp16.h>

extern "C" __global__ void residual_add_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

// In-place variant: a += b (a is in/out).
// Exists to avoid Rust borrow-checker conflicts when aliasing `a` as
// both read-only and writable argument.
extern "C" __global__ void residual_add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
    }
}

extern "C" __global__ void residual_add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] + b[idx];
    }
}
