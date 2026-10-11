// Ferrum MarkerV2. Ordinary finite math stays in the pinned upstream kernels.
// Device flags describe propagation, never host-side rejection or silent fallback.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
static __device__ __forceinline__ bool marker_finite(float x) {
    return (__float_as_uint(x)&0x7fffffffu)<0x7f800000u;
}
static __device__ __forceinline__ bool marker_half_finite(half x) {
    return (__half_as_ushort(x)&0x7fffu)<0x7c00u;
}
static __device__ __forceinline__ float marker_nan() { return __uint_as_float(0x7fc00000u); }
