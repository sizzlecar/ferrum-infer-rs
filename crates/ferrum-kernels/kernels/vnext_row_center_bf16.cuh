// Both outputs are admitted tensors. The BF16 mean is a visible numerical
// boundary and is consumed by the second launch on the same owner stream.
extern "C" __global__ void ferrum_vnext_row_center_mean_bf16(
    const uint16_t* x,uint16_t* mean,unsigned long long mean_elements,
    unsigned long long rows,unsigned long long columns,float factor) {
    __shared__ float partial[8][32];
    const unsigned int lane=threadIdx.y;
    for(unsigned long long tile=blockIdx.x;tile<(mean_elements+31)/32;tile+=gridDim.x){
        const unsigned long long i=tile*32+threadIdx.x;
        float sum=0.0f;
        if(i<mean_elements){
            const unsigned long long b=i/columns,c=i%columns;
            for(unsigned long long r=lane;r<rows;r+=8){
                const float v=__uint_as_float(uint32_t(x[(b*rows+r)*columns+c])<<16);
                asm("add.rn.f32 %0, %1, %2;":"=f"(sum):"f"(sum),"f"(v));
            }
        }
        partial[lane][threadIdx.x]=sum;
        __syncthreads();
        for(unsigned int stride=4;stride;stride/=2){
            if(lane<stride){
                const float v=partial[lane+stride][threadIdx.x];
                asm("add.rn.f32 %0, %1, %2;":"=f"(sum):"f"(sum),"f"(v));
                partial[lane][threadIdx.x]=sum;
            }
            __syncthreads();
        }
        if(lane==0 && i<mean_elements){
            float value;asm("mul.rn.f32 %0, %1, %2;":"=f"(value):"f"(sum),"f"(factor));
            mean[i]=ferrum_bf16_rne(value);
        }
        __syncthreads();
    }
}
extern "C" __global__ void ferrum_vnext_row_center_subtract_bf16(
    const uint16_t* x,const uint16_t* mean,uint16_t* y,
    unsigned long long elements,unsigned long long rows,unsigned long long columns){
    for(unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
        i<elements;i+=(unsigned long long)blockDim.x*gridDim.x){
        const unsigned long long b=i/(rows*columns),c=i%columns;
        const float v=__uint_as_float(uint32_t(x[i])<<16);
        const float m=__uint_as_float(uint32_t(mean[b*columns+c])<<16);
        float result;asm("sub.rn.f32 %0, %1, %2;":"=f"(result):"f"(v),"f"(m));
        y[i]=ferrum_bf16_rne(result);
    }
}
