#pragma once
#include <stdint.h>
#include <math.h>

extern "C" __global__ void ferrum_vnext_scale_rank4_f32(
    const float* input, float* output, uint64_t elements, float scale) {
    for (uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x; i<elements; i+=uint64_t(gridDim.x)*blockDim.x)
        output[i]=__fmul_rn(input[i],scale);
}

// Exactly 256 threads, one or more complete rows per CTA. No payload scratch.
// -Inf is an explicit masked value. NaN/+Inf/all -Inf produce whole-row NaN.
extern "C" __global__ void ferrum_vnext_row_softmax_rank4_f32(
    const float* input, float* output, uint64_t rows, uint64_t columns) {
    __shared__ float partial[256];
    __shared__ unsigned bad[256];
    const unsigned tid=threadIdx.x;
    for (uint64_t row=blockIdx.x; row<rows; row+=gridDim.x) {
        const uint64_t base=row*columns;
        float maximum=-INFINITY;
        unsigned invalid=0;
        for (uint64_t n=tid;n<columns;n+=256) {
            const float x=input[base+n];
            invalid|=isnan(x)||x==INFINITY;
            maximum=fmaxf(maximum,x);
        }
        partial[tid]=maximum; bad[tid]=invalid; __syncthreads();
        for (unsigned offset=128;offset;offset>>=1) {
            if(tid<offset) { partial[tid]=fmaxf(partial[tid],partial[tid+offset]); bad[tid]|=bad[tid+offset]; }
            __syncthreads();
        }
        maximum=partial[0]; invalid=bad[0]||maximum==-INFINITY;
        __syncthreads();
        float sum=0.0f;
        if(!invalid) for(uint64_t n=tid;n<columns;n+=256) sum=__fadd_rn(sum,expf(__fsub_rn(input[base+n],maximum)));
        partial[tid]=sum; __syncthreads();
        for(unsigned offset=128;offset;offset>>=1) {
            if(tid<offset) partial[tid]=__fadd_rn(partial[tid],partial[tid+offset]);
            __syncthreads();
        }
        sum=partial[0]; __syncthreads();
        for(uint64_t n=tid;n<columns;n+=256) output[base+n]=invalid ? __int_as_float(0x7fc00000) : __fdiv_rn(expf(__fsub_rn(input[base+n],maximum)),sum);
        __syncthreads();
    }
}
