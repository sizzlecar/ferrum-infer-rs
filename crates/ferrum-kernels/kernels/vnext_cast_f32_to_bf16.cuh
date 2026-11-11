// Shared precise TU's bit-defined RNE conversion preserves subnormal outputs.
#pragma once
extern "C" __global__ void ferrum_vnext_cast_f32_to_bf16(
    const float* input,uint16_t* output,uint64_t elements) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        output[i]=ferrum_bf16_rne(input[i]);
    }
}
