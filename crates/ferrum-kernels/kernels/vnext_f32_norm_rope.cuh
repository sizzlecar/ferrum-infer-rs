// Independent precise F32 operations; the BF16 kernels retain their old ABI.
#pragma once
extern "C" __global__ void ferrum_vnext_unit_head_rms_partial_rope_f32(
    const float* input,const float* cosine,const float* sine,float* output,
    uint64_t groups,uint64_t columns,uint32_t width,uint32_t rotary,float epsilon){
    __shared__ float sums[256];__shared__ float inverse;
    const unsigned lane=threadIdx.x;
    for(uint64_t group=blockIdx.x;group<groups;group+=gridDim.x){
        const uint64_t start=group*width;float sum=0.0f;
        for(uint32_t c=lane;c<width;c+=256){const float x=input[start+c];sum=__fadd_rn(sum,__fmul_rn(x,x));}
        sums[lane]=sum;__syncthreads();
        for(unsigned stride=128;stride!=0;stride/=2){if(lane<stride)sums[lane]=__fadd_rn(sums[lane],sums[lane+stride]);__syncthreads();}
        if(lane==0){const float mean=__fdiv_rn(sums[0],static_cast<float>(width));inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(mean,epsilon)));}
        __syncthreads();
        for(uint32_t c=lane;c<width;c+=256){
            const float normalized=__fmul_rn(input[start+c],inverse);
            if(c<rotary){
                const uint64_t frequency=(start/columns)*rotary+c;
                const float partner=c<rotary/2?-__fmul_rn(input[start+c+rotary/2],inverse):__fmul_rn(input[start+c-rotary/2],inverse);
                output[start+c]=__fadd_rn(__fmul_rn(normalized,cosine[frequency]),__fmul_rn(partner,sine[frequency]));
            }else output[start+c]=normalized;
        }
        __syncthreads();
    }
}
extern "C" __global__ void ferrum_vnext_head_rms_norm_f32(
    const float* input,const float* gamma,float* output,uint64_t groups,uint32_t width,float epsilon){
    __shared__ float sums[256];__shared__ float inverse;
    const unsigned lane=threadIdx.x;
    for(uint64_t group=blockIdx.x;group<groups;group+=gridDim.x){
        const uint64_t start=group*width;float sum=0.0f;
        for(uint32_t c=lane;c<width;c+=256){const float x=input[start+c];sum=__fadd_rn(sum,__fmul_rn(x,x));}
        sums[lane]=sum;__syncthreads();
        for(unsigned stride=128;stride!=0;stride/=2){if(lane<stride)sums[lane]=__fadd_rn(sums[lane],sums[lane+stride]);__syncthreads();}
        if(lane==0){const float mean=__fdiv_rn(sums[0],static_cast<float>(width));inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(mean,epsilon)));}
        __syncthreads();
        for(uint32_t c=lane;c<width;c+=256)output[start+c]=__fmul_rn(__fmul_rn(input[start+c],inverse),gamma[c]);
        __syncthreads();
    }
}
extern "C" __global__ void ferrum_vnext_half_split_rope_f32(
    const float* input,const float* cosine,const float* sine,float* output,uint64_t elements,uint64_t columns,uint64_t head_dim){
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride){
        const uint64_t local=i%columns%head_dim;
        const uint64_t frequency=(i/columns)*head_dim+local;
        const float partner=local<head_dim/2?-input[i+head_dim/2]:input[i-head_dim/2];
        const float p=__fmul_rn(input[i],cosine[frequency]);
        const float q=__fmul_rn(partner,sine[frequency]);
        output[i]=__fadd_rn(p,q);
    }
}
