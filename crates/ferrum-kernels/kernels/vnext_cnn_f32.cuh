#pragma once
extern "C" __global__ void ferrum_vnext_nchw_pad_f32(
    const uint32_t* input,uint32_t* output,uint64_t elements,
    uint64_t height,uint64_t width,uint64_t padded_height,uint64_t padded_width,
    uint64_t top,uint64_t left,uint32_t reflect){
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride){
        const uint64_t plane=i/(padded_height*padded_width);
        int64_t y=int64_t(i/padded_width%padded_height)-int64_t(top);
        int64_t x=int64_t(i%padded_width)-int64_t(left);
        if(reflect){if(y<0)y=-y;else if(uint64_t(y)>=height)y=int64_t(2*height-2)-y;
            if(x<0)x=-x;else if(uint64_t(x)>=width)x=int64_t(2*width-2)-x;}
        output[i]=(y>=0&&x>=0&&uint64_t(y)<height&&uint64_t(x)<width)?
            input[(plane*height+uint64_t(y))*width+uint64_t(x)]:0u;
    }
}
extern "C" __global__ void ferrum_vnext_nchw_bias_f32(
    float* output,const float* bias,uint64_t elements,uint64_t channels,uint64_t spatial){
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride)
        output[i]=__fadd_rn(output[i],bias[i/spatial%channels]);
}
extern "C" __global__ void ferrum_vnext_plain_residual_add_nchw_f32(
    const float* input,const float* branch,float* output,uint64_t elements){
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride)
        output[i]=__fadd_rn(input[i],branch[i]);
}
extern "C" __global__ void ferrum_vnext_group_norm_silu_nchw_f32(
    const float* input,const float* gamma,const float* beta,float* output,
    uint64_t batch_groups,uint64_t group_elements,uint64_t channels,uint64_t spatial,float epsilon){
    __shared__ float sums[256];__shared__ float mean,inverse;const unsigned lane=threadIdx.x;
    for(uint64_t group=blockIdx.x;group<batch_groups;group+=gridDim.x){
        const uint64_t start=group*group_elements;float sum=0.0f;
        for(uint64_t i=lane;i<group_elements;i+=256)sum=__fadd_rn(sum,input[start+i]);
        sums[lane]=sum;__syncthreads();
        for(unsigned step=128;step;step/=2){if(lane<step)sums[lane]=__fadd_rn(sums[lane],sums[lane+step]);__syncthreads();}
        if(lane==0)mean=__fdiv_rn(sums[0],float(group_elements));__syncthreads();sum=0.0f;
        for(uint64_t i=lane;i<group_elements;i+=256){const float centered=__fsub_rn(input[start+i],mean);sum=__fadd_rn(sum,__fmul_rn(centered,centered));}
        sums[lane]=sum;__syncthreads();
        for(unsigned step=128;step;step/=2){if(lane<step)sums[lane]=__fadd_rn(sums[lane],sums[lane+step]);__syncthreads();}
        if(lane==0)inverse=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(__fdiv_rn(sums[0],float(group_elements)),epsilon)));__syncthreads();
        for(uint64_t i=lane;i<group_elements;i+=256){const uint64_t channel=(start+i)/spatial%channels;
            const float normalized=__fmul_rn(__fsub_rn(input[start+i],mean),inverse);
            const float affine=__fadd_rn(__fmul_rn(normalized,gamma[channel]),beta[channel]);
            output[start+i]=__fdiv_rn(affine,__fadd_rn(1.0f,expf(-affine)));}
        __syncthreads();
    }
}
