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
static __device__ __forceinline__ float marker_half_at(const unsigned char * b,int i) {
    return __half2float(__ushort_as_half(uint16_t(b[i])|(uint16_t(b[i+1])<<8)));
}
// One once-per-preparation scan of every physical block, including owned zero
// row padding. The retained flag belongs to the entire physical weight leaf.
static __global__ void marker_check_weights(const unsigned char * weights,uint64_t blocks,
        uint32_t format,bool mmq,uint32_t * flag) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=blocks) return;
    const int bytes=ferrum_linear_bytes(format);
    const auto b=weights+index*bytes;
    const float d=marker_half_at(b,format==11?108:0);
    bool bad=!marker_finite(d);
    // Non-affine weights. Each consumed scale is a finite half times a bounded
    // integer (Q3 signed6-bit, IQ3 odd1..31, IQ4NL signed LUT). No half group
    // coefficient round or minimum is consumed by either pinned implementation.
    if (bad) atomicOr(flag,1u);

}
static __global__ void marker_cast(const float * input,half * output,uint32_t rows,
        uint32_t columns,uint32_t stride,const uint32_t * row_flags,const uint32_t * weight_flag) {
    const uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=rows*columns) return;
    const float value=input[index]; const half rounded=__float2half_rn(value);
    const bool bad=*weight_flag || row_flags[index/columns] || !marker_finite(value) || !marker_half_finite(rounded);
    output[(index/columns)*stride+index%columns]=bad?__ushort_as_half(0x7e00):rounded;
}
