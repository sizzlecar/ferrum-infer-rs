#pragma once
// uint32_t loads/stores preserve F32 payload bits, including signaling NaNs.
extern "C" __global__ void ferrum_vnext_batched_suffix_append_f32(
    const uint32_t* input,const uint32_t* suffix,uint32_t* output,
    uint64_t elements,uint64_t columns,uint64_t patch_rows,
    uint64_t suffix_rows,uint64_t zero_rows){
    const uint64_t rows=patch_rows+suffix_rows+zero_rows;
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride){
        const uint64_t row=i/columns;const uint64_t local=row%rows;
        const uint64_t column=i%columns;const uint64_t batch=row/rows;
        output[i]=local<patch_rows?input[(batch*patch_rows+local)*columns+column]:
            local<patch_rows+suffix_rows?suffix[(local-patch_rows)*columns+column]:0u;
    }
}
