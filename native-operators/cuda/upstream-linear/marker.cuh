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
    const int bytes=format==12?144:format==13?176:136;
    const auto b=weights+index*bytes;
    const float d=marker_half_at(b,0);
    bool bad=!marker_finite(d);
    if(format==23) {
        const uint16_t high=uint16_t(b[2])|(uint16_t(b[3])<<8);
        for(int g=0;g<8;++g) {
            const int low=(b[4+g/2]>>(4*(g%2)))&15;
            const int scale=(low|(((high>>(2*g))&3)<<4))-32;
            bad|=!marker_finite(d*scale);
        }
    } else {
        const float minimum=marker_half_at(b,2);
        bad|=!marker_finite(minimum);
        for(int g=0;g<8;++g) {
            const int scale=g<4 ? b[4+g]&63 : (b[8+g]&15)|((b[g]>>6)<<4);
            const int min=g<4 ? b[8+g]&63 : (b[8+g]>>4)|((b[4+g]>>6)<<4);
            const float a=d*scale, m=-minimum*min;
            bad|=!marker_finite(a)||!marker_finite(m);
            if(mmq) bad|=!marker_half_finite(__float2half_rn(a))||!marker_half_finite(__float2half_rn(m));
        }
    }
    if(bad) atomicOr(flag,1u);
}
static __global__ void marker_cast(const float * input,half * output,uint32_t rows,
        uint32_t columns,uint32_t stride,const uint32_t * row_flags,const uint32_t * weight_flag) {
    const uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=rows*columns) return;
    const float value=input[index]; const half rounded=__float2half_rn(value);
    const bool bad=*weight_flag || row_flags[index/columns] || !marker_finite(value) || !marker_half_finite(rounded);
    output[(index/columns)*stride+index%columns]=bad?__ushort_as_half(0x7e00):rounded;
}
