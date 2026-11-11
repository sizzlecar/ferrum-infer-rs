// Independent single-round operations in the existing precise native module.
#pragma once
extern "C" __global__ void ferrum_vnext_row_rms_norm_single_round_bf16(
    const uint16_t* input,const uint16_t* gamma,uint16_t* output,
    uint64_t rows,uint64_t width,float epsilon) {
    const unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(uint64_t row=uint64_t(blockIdx.x)*4+warp;row<rows;row+=uint64_t(gridDim.x)*4){
        const uint64_t start=row*width;float sum=0.0f;
        for(uint64_t d=lane;d<width;d+=32){const float x=ferrum_bf16_decode(input[start+d]);sum=__fmaf_rn(x,x,sum);}
        for(unsigned off=16;off;off>>=1)sum=__fadd_rn(sum,__shfl_down_sync(0xffffffffu,sum,off));
        sum=__shfl_sync(0xffffffffu,sum,0);
        const float mean=__fdiv_rn(sum,float(width));
        const float inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(mean,epsilon)));
        for(uint64_t d=lane;d<width;d+=32){const float x=ferrum_bf16_decode(input[start+d]);output[start+d]=ferrum_bf16_rne(__fmul_rn(__fmul_rn(x,inverse),ferrum_bf16_decode(gamma[d])));}
    }
}
extern "C" __global__ void ferrum_vnext_addcmul_bf16(
    const uint16_t* input,const uint16_t* multiplier,const uint16_t* update,
    uint16_t* output,uint64_t elements,uint64_t width,uint32_t broadcast_row){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x){
        const float x=ferrum_bf16_decode(input[i]);
        const float g=ferrum_bf16_decode(multiplier[broadcast_row?i%width:i]);
        const float u=ferrum_bf16_decode(update[i]);
        output[i]=ferrum_bf16_rne(__fadd_rn(x,__fmul_rn(g,u)));
    }
}
