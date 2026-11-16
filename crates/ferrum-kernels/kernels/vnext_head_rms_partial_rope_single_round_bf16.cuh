#pragma once
#include <cuda_bf16.h>
// Inspected Triton3.7/sm120 PTX SHA256:
// ec8e80527c21c2bf95c2048e687281c94302485de1b88c02c8c6cee69f7d0878.
// One 128-lane head, original F32 norm/partner, final rotary BF16 store only.
extern "C" __global__ void ferrum_vnext_head_rms_partial_rope_single_round_bf16(
    const __nv_bfloat16* x,const __nv_bfloat16* gamma,const __nv_bfloat16* cosine,
    const __nv_bfloat16* sine,__nv_bfloat16* output,uint64_t head_rows,
    uint64_t columns,uint64_t head_dim,uint64_t rotary_dim,float epsilon) {
    __shared__ float warp_sums[4];
    const unsigned lane=threadIdx.x;
    for(uint64_t h=blockIdx.x;h<head_rows;h+=gridDim.x) {
        const uint64_t start=h*head_dim;
        const uint64_t row=h/(columns/head_dim);
        const float v=lane<head_dim?__bfloat162float(x[start+lane]):0.0f;
        const float square=__fmul_rn(v,v);
        float sum=__fmaf_rn(v,v,__shfl_xor_sync(0xffffffff,square,16));
        #pragma unroll
        for(unsigned offset=8;offset>0;offset>>=1) sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
        if(lane%32==0) warp_sums[lane/32]=sum;
        __syncthreads();
        if(lane<32) {
            float cross=lane<4?warp_sums[lane]:0.0f;
            cross=__fadd_rn(cross,__shfl_xor_sync(0xffffffff,cross,2));
            cross=__fadd_rn(cross,__shfl_xor_sync(0xffffffff,cross,1));
            if(lane==0) warp_sums[0]=cross;
        }
        __syncthreads();
        float mean,inverse;
        const float divisor=float(head_dim);
        asm volatile("div.full.f32 %0, %1, %2;":"=f"(mean):"f"(warp_sums[0]),"f"(divisor));
        const float adjusted=__fadd_rn(epsilon,mean);
        asm volatile("rsqrt.approx.ftz.f32 %0, %1;":"=f"(inverse):"f"(adjusted));
        if(lane<head_dim) {
            const float normalized=__fmul_rn(__fmul_rn(inverse,v),__bfloat162float(gamma[lane]));
            float result=normalized;
            if(lane<rotary_dim) {
                const bool first=lane<rotary_dim/2;
                const uint64_t partner=first?lane+rotary_dim/2:lane-rotary_dim/2;
                const float p=__fmul_rn(inverse,__bfloat162float(x[start+partner]));
                const float g=__bfloat162float(gamma[partner]);
                const float rotated=first?__fmaf_rn(-p,g,0.0f):__fmul_rn(p,g);
                const float cos=__bfloat162float(cosine[row*rotary_dim+lane]);
                const float sin=__bfloat162float(sine[row*rotary_dim+lane]);
                result=__fmaf_rn(normalized,cos,__fmul_rn(rotated,sin));
            }
            output[start+lane]=__float2bfloat16_rn(result);
        }
        __syncthreads();
    }
}
