// Independent precise full-F32 LayerNorm. Included only in the managed TU
// compiled with fmad=false, ftz=false, prec-div=true, prec-sqrt=true.
#pragma once
extern "C" __global__ void ferrum_vnext_layer_norm_f32(
    const float* input,const float* gamma,const float* beta,float* output,
    uint64_t rows,uint32_t width,float epsilon) {
    __shared__ float sums[256];
    __shared__ float mean,inverse;
    const unsigned lane=threadIdx.x;
    for(uint64_t row=blockIdx.x;row<rows;row+=gridDim.x) {
        const uint64_t start=row*width;
        float sum=0.0f;
        for(uint32_t c=lane;c<width;c+=256) sum=__fadd_rn(sum,input[start+c]);
        sums[lane]=sum;__syncthreads();
        for(unsigned stride=128;stride!=0;stride/=2) { if(lane<stride)sums[lane]=__fadd_rn(sums[lane],sums[lane+stride]);__syncthreads(); }
        if(lane==0)mean=__fdiv_rn(sums[0],static_cast<float>(width));
        __syncthreads();sum=0.0f;
        for(uint32_t c=lane;c<width;c+=256) { const float centered=__fsub_rn(input[start+c],mean);sum=__fadd_rn(sum,__fmul_rn(centered,centered)); }
        sums[lane]=sum;__syncthreads();
        for(unsigned stride=128;stride!=0;stride/=2) { if(lane<stride)sums[lane]=__fadd_rn(sums[lane],sums[lane+stride]);__syncthreads(); }
        if(lane==0) { const float variance=__fdiv_rn(sums[0],static_cast<float>(width));inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(variance,epsilon))); }
        __syncthreads();
        for(uint32_t c=lane;c<width;c+=256) { const float centered=__fsub_rn(input[start+c],mean);const float normalized=__fmul_rn(centered,inverse);const float scaled=__fmul_rn(normalized,gamma[c]);output[start+c]=__fadd_rn(scaled,beta[c]); }
        __syncthreads();
    }
}
