#pragma once
// Compiled in the precise shared TU: gradual underflow, no FMA/fast math.
extern "C" __global__ void ferrum_vnext_pointwise_f32(
    const float* input,const float* update,float* output,uint64_t elements,int kind) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        const float x=input[i],u=update[i];
        if(kind==0) output[i]=__fadd_rn(x,u);
        else {
            const float denominator=__fadd_rn(1.0f,expf(-x));
            output[i]=__fmul_rn(__fdiv_rn(x,denominator),u);
        }
    }
}
