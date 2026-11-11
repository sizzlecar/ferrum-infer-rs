#pragma once
// Compiled in the precise TU, with gradual underflow, no fast math and no FMA
// contraction. This source identity does not prove libm/Torch binary parity.
extern "C" __global__ void ferrum_vnext_gelu_f32(
    const float* input,float* output,uint64_t elements,int variant) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        const float x=input[i];
        float t;
        if(variant==0) {
            const float square=__fmul_rn(x,x);
            const float cube=__fmul_rn(square,x);
            const float cubic_term=__fmul_rn(__uint_as_float(0x3d372713u),cube);
            const float polynomial=__fadd_rn(x,cubic_term);
            t=tanhf(__fmul_rn(__uint_as_float(0x3f4c422au),polynomial));
        } else {
            t=erff(__fmul_rn(x,__uint_as_float(0x3f3504f3u)));
        }
        output[i]=__fmul_rn(__fmul_rn(0.5f,x),__fadd_rn(1.0f,t));
    }
}
