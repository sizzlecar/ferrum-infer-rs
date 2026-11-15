// Narrowing rounds RNE; widening preserves every BF16 bit without FP arithmetic.
#pragma once
extern "C" __global__ void ferrum_vnext_float_cast(
    const void* input,void* output,uint64_t elements,int direction) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        if(direction == 0) {
            static_cast<uint16_t*>(output)[i]=ferrum_bf16_rne(static_cast<const float*>(input)[i]);
        } else {
            static_cast<uint32_t*>(output)[i]=uint32_t(static_cast<const uint16_t*>(input)[i]) << 16;
        }
    }
}
