// Element-wise scaled add: dst[i] += scale * src[i]

#include <cuda_fp16.h>

extern "C" __global__ void scaled_add_inplace_f16(
    __half* __restrict__ dst,
    const __half* __restrict__ src,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(__half2float(dst[idx]) + scale * __half2float(src[idx]));
    }
}

extern "C" __global__ void scaled_add_inplace_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const float scale,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += scale * src[idx];
    }
}
