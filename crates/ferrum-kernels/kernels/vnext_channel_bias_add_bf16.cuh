// Two independent scalar RN32 additions with the original BF16 RNE boundary.
__device__ __forceinline__ uint32_t ferrum_channel_bias_pair(uint32_t x,uint32_t b){
    const float x0=__uint_as_float((x&0xffffu)<<16),x1=__uint_as_float(x&0xffff0000u);
    const float b0=__uint_as_float((b&0xffffu)<<16),b1=__uint_as_float(b&0xffff0000u);
    float s0,s1;
    asm("add.rn.f32 %0, %1, %2;":"=f"(s0):"f"(x0),"f"(b0));
    asm("add.rn.f32 %0, %1, %2;":"=f"(s1):"f"(x1),"f"(b1));
    return uint32_t(ferrum_bf16_rne(s0))|(uint32_t(ferrum_bf16_rne(s1))<<16);
}

// Explicit rank-one channel vector. No broadcast buffer or hidden allocation.
extern "C" __global__ void ferrum_vnext_channel_bias_add_bf16(
    const uint16_t* x,const uint16_t* bias,uint16_t* y,
    unsigned long long elements,unsigned long long columns){
    // All ports have checked 16-byte alignment. A whole channel group must
    // fit in the row; odd/unaligned channel widths retain scalar indexing.
    if((columns&7ull)==0){
        const unsigned long long groups=elements/8,channels=columns/8;
        const uint4* xv=reinterpret_cast<const uint4*>(x);
        const uint4* bv=reinterpret_cast<const uint4*>(bias);
        uint4* yv=reinterpret_cast<uint4*>(y);
        for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
            i<groups;i+=(unsigned long long)blockDim.x*gridDim.x){
            const uint4 a=xv[i],b=bv[i%channels];
            yv[i]=make_uint4(ferrum_channel_bias_pair(a.x,b.x),
                            ferrum_channel_bias_pair(a.y,b.y),
                            ferrum_channel_bias_pair(a.z,b.z),
                            ferrum_channel_bias_pair(a.w,b.w));
        }
        return;
    }
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        i<elements;i+=(unsigned long long)blockDim.x*gridDim.x){
        const float a=__uint_as_float(uint32_t(x[i])<<16);
        const float b=__uint_as_float(uint32_t(bias[i%columns])<<16);
        float sum;asm("add.rn.f32 %0, %1, %2;":"=f"(sum):"f"(a),"f"(b));
        y[i]=ferrum_bf16_rne(sum);
    }
}
