#pragma once
// Integer transport preserves every F32 bit, including signaling-NaN payload,
// signed zero and subnormal. This kernel performs no floating-point operation.
extern "C" __global__ void ferrum_vnext_reshape_copy_f32(
    const uint32_t* input,uint32_t* output,uint64_t elements) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<elements;i+=uint64_t(gridDim.x)*blockDim.x) output[i]=input[i];
}
