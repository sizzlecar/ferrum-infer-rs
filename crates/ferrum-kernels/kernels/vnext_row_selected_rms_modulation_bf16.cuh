#pragma once
#include <cuda_bf16.h>
// Fixed numerical profile matching the inspected 8192-column, vec8, 16-warp
// Triton3.7 kernels. PTX SHA256: norm 917c65fb17aa3bc923dccaa5b75be613b
// c2e3baeff15e1dcca9d06b246427513; residual d560a27bf8206b9310f3fe0d2926f341
// 314a178ac9da05ad25acd8f01ad1fe9b. Only rsqrt flushes subnormals.
struct FerrumRmsModulationRouting {
    uint64_t ends[32];
    uint64_t rows[32];
    uint64_t count;
};
extern "C" __global__ void ferrum_vnext_row_selected_rms_modulation_bf16(
    const __nv_bfloat16* input,const __nv_bfloat16* table,const __nv_bfloat16* gamma,
    const __nv_bfloat16* branch,__nv_bfloat16* hidden_output,__nv_bfloat16* output,
    uint64_t rows,uint64_t width,uint64_t table_width,uint64_t shift_column,
    uint64_t scale_column,uint64_t gate_column,float epsilon,unsigned residual,
    FerrumRmsModulationRouting routing) {
    __shared__ float warp_sums[16];
    const unsigned lane=threadIdx.x;
    for(uint64_t row=blockIdx.x;row<rows;row+=gridDim.x) {
        unsigned run=0;
        while(row>=routing.ends[run]) ++run;
        const uint64_t p=routing.rows[run]*table_width;
        float h[16];
        #pragma unroll
        for(unsigned j=0;j<16;++j) {
            const uint64_t col=uint64_t(lane)*8+(j%8)+(j/8)*4096;
            float value=0.0f;
            if(col<width) {
                value=__bfloat162float(input[row*width+col]);
                if(residual) {
                    const float gate=__bfloat162float(table[p+gate_column+col]);
                    value=__fmaf_rn(gate,__bfloat162float(branch[row*width+col]),value);
                    hidden_output[row*width+col]=__float2bfloat16_rn(value);
                }
            }
            h[j]=value;
        }
        float sum=__fmaf_rn(h[0],h[0],__fmul_rn(h[1],h[1]));
        #pragma unroll
        for(unsigned j=2;j<16;++j) {
            if(residual || (j%2)==0) sum=__fmaf_rn(h[j],h[j],sum);
            else sum=__fadd_rn(sum,__fmul_rn(h[j],h[j]));
        }
        #pragma unroll
        for(unsigned offset=16;offset>0;offset>>=1) sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,offset));
        if((lane%32)==0) warp_sums[lane/32]=sum;
        __syncthreads();
        if(lane<32) {
            float cross=lane<16?warp_sums[lane]:0.0f;
            #pragma unroll
            for(unsigned offset=8;offset>0;offset>>=1) cross=__fadd_rn(cross,__shfl_xor_sync(0xffffffff,cross,offset));
            if(lane==0) warp_sums[0]=cross;
        }
        __syncthreads();
        float mean,inverse;
        const float denominator=float(width);
        asm volatile("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(warp_sums[0]),"f"(denominator));
        const float adjusted=__fadd_rn(epsilon,mean);
        asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(adjusted));
        #pragma unroll
        for(unsigned j=0;j<16;++j) {
            const uint64_t col=uint64_t(lane)*8+(j%8)+(j/8)*4096;
            if(col<width) {
                const float norm=__fmul_rn(h[j],inverse);
                const float weighted=__fmul_rn(norm,__bfloat162float(gamma[col]));
                const float factor=__fadd_rn(1.0f,__bfloat162float(table[p+scale_column+col]));
                const float shift=__bfloat162float(table[p+shift_column+col]);
                output[row*width+col]=__float2bfloat16_rn(__fmaf_rn(weighted,factor,shift));
            }
        }
        __syncthreads();
    }
}
