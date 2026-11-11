// Selected-row inverse ConvRot. Shared precise TU, no whole-table decode.
#pragma once
extern "C" __global__ void ferrum_comfy_convrot_int8_embedding_f32(
    const uint32_t* ids,const int8_t* table,const float* scales,float* output,
    uint64_t groups,uint64_t width,uint64_t vocabulary,uint32_t bf16_then_f32) {
    __shared__ float left[256];
    __shared__ float right[256];
    const unsigned i=threadIdx.x;
    const uint64_t groups_per_row=width/256;
    for(uint64_t group=blockIdx.x;group<groups;group+=gridDim.x) {
        const uint64_t token=group/groups_per_row;
        const uint64_t row=ids[token];
        const uint64_t column=(group%groups_per_row)*256+i;
        if(row>=vocabulary) {output[token*width+column]=0.0f;continue;}
        left[i]=__fmul_rn(static_cast<float>(table[row*width+column]),scales[row]);
        __syncthreads();
        float* source=left;float* destination=right;
        for(unsigned stride=1;stride<256;stride*=4) {
            const unsigned digit=(i/stride)&3;
            const unsigned base=i-digit*stride;
            destination[i]=__fmul_rn(0.5f,ferrum_comfy_h4(digit,source[base],source[base+stride],source[base+2*stride],source[base+3*stride]));
            __syncthreads();
            float* swap=source;source=destination;destination=swap;
        }
        const float value=source[i];
        output[token*width+column]=bf16_then_f32?ferrum_bf16_decode(ferrum_bf16_rne(value)):value;
        __syncthreads();
    }
}
