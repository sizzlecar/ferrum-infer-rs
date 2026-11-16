#pragma once
// Match the instruction family in the fixed Inductor prefix kernel: both
// divisions are div.full without FTZ. Other arithmetic is explicitly RN.
__device__ __forceinline__ float ferrum_head_fp8_div(float a,float b){float r;asm("div.full.f32 %0, %1, %2;":"=f"(r):"f"(a),"f"(b));return r;}
__device__ __forceinline__ float ferrum_head_fp8_max(float a,float b){return isnan(a)?a:(isnan(b)?b:fmaxf(a,b));}
// Exact E4M3FN round/expand implemented with integer significand rounding, so
// this shared bundle does not impose a new minimum GPU architecture. This is
// equivalent to cvt.rn.satfinite.e4m3x2 followed by exact F32 expansion for
// clamped finite values; NaN payload is intentionally unspecified.
__device__ __forceinline__ float ferrum_head_fp8_round(float x){
    if(isnan(x))return x;
    const uint32_t bits=__float_as_uint(x),sign=bits&0x80000000u,mag=bits&0x7fffffffu;
    float value;
    if(mag<0x3c800000u)value=__fmul_rn(float(__float2int_rn(__fmul_rn(fabsf(x),512.0f))),0.001953125f);
    else value=__uint_as_float((mag+0x7ffffu+((mag>>20)&1u))&0x7ff00000u);
    return __uint_as_float(__float_as_uint(value)|sign);
}
// Return a source head start, or an invalid-row sentinel. The map is shared
// across batches; reduction lanes and arithmetic within a head are unchanged.
__device__ __forceinline__ uint64_t ferrum_head_fp8_source_start(
    uint64_t start,const int32_t* indices,uint64_t source_rows,uint64_t output_rows,uint64_t columns){
    if(!indices)return start;
    const uint64_t row=start/columns,batch=row/output_rows;
    const int32_t source=indices[row%output_rows];
    if(source<0 || uint64_t(source)>=source_rows)return (~uint64_t(0));
    return (batch*source_rows+uint64_t(source))*columns+start%columns;
}
extern "C" __global__ void ferrum_vnext_head_fp8_roundtrip_bf16(
    const uint16_t* input,uint16_t* output,uint64_t groups,uint64_t head_dim,
    const int32_t* indices,uint64_t source_rows,uint64_t output_rows,uint64_t columns){
    // Short heads need only one warp. Four independent heads share the CTA;
    // max is exact for finite inputs and propagates NaNs as in the scalar
    // contract. Keep the same div.full and rounding instructions below.
    if (head_dim <= 256) {
        const unsigned lane = threadIdx.x % 32;
        const unsigned warp = threadIdx.x / 32;
        for (uint64_t group = uint64_t(blockIdx.x) * 4 + warp;
             group < groups; group += uint64_t(gridDim.x) * 4) {
            const uint64_t start = group * head_dim;
            const uint64_t source_start = ferrum_head_fp8_source_start(start,indices,source_rows,output_rows,columns);
            float amax = 0.0f;
            for (uint64_t d = lane; d < head_dim; d += 32) {
                amax = ferrum_head_fp8_max(amax, fabsf(__uint_as_float(uint32_t(source_start==(~uint64_t(0)) ? 0 : input[source_start+d]) << 16)));
            }
            for (unsigned offset = 16; offset; offset >>= 1) {
                amax = ferrum_head_fp8_max(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
            }
            const float scale = ferrum_head_fp8_div(amax, 448.0f);
            const float inv = scale == 0.0f ? 0.0f : ferrum_head_fp8_div(1.0f, scale);
            for (uint64_t d = lane; d < head_dim; d += 32) {
                const float x = __uint_as_float(uint32_t(source_start==(~uint64_t(0)) ? 0 : input[source_start+d]) << 16);
                float v = __fmul_rn(x, inv);
                if (!isnan(v)) v = fminf(fmaxf(v, -448.0f), 448.0f);
                output[start+d] = ferrum_bf16_rne(__fmul_rn(ferrum_head_fp8_round(v), scale));
            }
        }
        return;
    }
    __shared__ float maxima[128];
    for(uint64_t group=blockIdx.x;group<groups;group+=gridDim.x){
        const uint64_t start=group*head_dim;
        const uint64_t source_start=ferrum_head_fp8_source_start(start,indices,source_rows,output_rows,columns);
        float amax=0.0f;
        for(uint64_t d=threadIdx.x;d<head_dim;d+=128)amax=ferrum_head_fp8_max(amax,fabsf(__uint_as_float(uint32_t(source_start==(~uint64_t(0)) ? 0 : input[source_start+d])<<16)));
        maxima[threadIdx.x]=amax;__syncthreads();
        for(unsigned stride=64;stride;stride>>=1){if(threadIdx.x<stride)maxima[threadIdx.x]=ferrum_head_fp8_max(maxima[threadIdx.x],maxima[threadIdx.x+stride]);__syncthreads();}
        const float scale=ferrum_head_fp8_div(maxima[0],448.0f);
        const float inv=scale==0.0f?0.0f:ferrum_head_fp8_div(1.0f,scale);
        for(uint64_t d=threadIdx.x;d<head_dim;d+=128){
            const float x=__uint_as_float(uint32_t(source_start==(~uint64_t(0)) ? 0 : input[source_start+d])<<16);
            float v=__fmul_rn(x,inv);
            if(!isnan(v))v=fminf(fmaxf(v,-448.0f),448.0f);
            output[start+d]=ferrum_bf16_rne(__fmul_rn(ferrum_head_fp8_round(v),scale));
        }
        __syncthreads();
    }
}
