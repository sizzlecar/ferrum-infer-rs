#pragma once
#include <cuda_bf16.h>
// Matches the expression and scalar rounding of FastVideo's cached
// _minimax_h3_swiglu_kernel (Triton3.7 / SM120). Its SIMD mul.f32x2 lanes
// are ordinary F32 products. ex2/div deliberately have NO .ftz modifier.
// Observed cached PTX SHA256:
// efcdd0873113efaa594810426ddfdd396acc0c08504e7f84090434582a585609
extern "C" __global__ void ferrum_vnext_packed_swiglu_bf16(
    const __nv_bfloat16* input,__nv_bfloat16* output,uint64_t elements,uint64_t width,uint32_t staged) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        const uint64_t base=(i/width)*width*2+i%width;
        const float value=__bfloat162float(input[base]);
        const float gate=__bfloat162float(input[base+width]);
        if (staged) {
            const uint16_t activated = ferrum_silu_bf16_value(gate);
            output[i] = __float2bfloat16_rn(__fmul_rn(ferrum_bf16_decode(activated), value));
            continue;
        }
        const float negative=__fsub_rn(0.0f,gate);
        const float exponent=__fmul_rn(negative,__int_as_float(0x3fb8aa3b));
        float e,sigmoid;
        asm volatile("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(exponent));
        const float denominator=__fadd_rn(1.0f,e);
        asm volatile("div.full.f32 %0, %1, %2;" : "=f"(sigmoid) : "f"(1.0f),"f"(denominator));
        const float silu=__fmul_rn(gate,sigmoid);
        output[i]=__float2bfloat16_rn(__fmul_rn(value,silu));
    }
}
