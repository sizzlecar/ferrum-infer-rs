#pragma once
// Exact [B,H,R,D] -> [B,R,H*D] index transport. Optional narrowing is applied
// once after address selection, using the shared gradual IEEE BF16 converter.
extern "C" __global__ void ferrum_vnext_head_axis_pack(
    const void* input,uint16_t* output,uint64_t elements,uint64_t heads,
    uint64_t rows,uint64_t head_dim,int narrow) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x){
        const uint64_t d=i%head_dim;
        const uint64_t h=(i/head_dim)%heads;
        const uint64_t r=(i/(head_dim*heads))%rows;
        const uint64_t b=i/(head_dim*heads*rows);
        const uint64_t source=((b*heads+h)*rows+r)*head_dim+d;
        output[i]=narrow ? ferrum_bf16_rne(static_cast<const float*>(input)[source]) : static_cast<const uint16_t*>(input)[source];
    }
}
