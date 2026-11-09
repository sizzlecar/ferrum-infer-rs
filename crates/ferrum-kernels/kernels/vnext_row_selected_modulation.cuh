#pragma once
struct FerrumRowSelectedModulationRouting{uint64_t ends[32];uint64_t rows[32];uint64_t count;};
static_assert(sizeof(FerrumRowSelectedModulationRouting)==520,"row-selected routing ABI");
// Precise shared TU, --ftz=false --fmad=false. The RNE product is committed
// before the final add; neither the source BF16 nor F32 parameters are recast.
extern "C" __global__ void ferrum_vnext_row_selected_modulation_bf16_f32(
    const uint16_t* input,const float* parameters,float* output,uint64_t elements,uint64_t columns,
    const FerrumRowSelectedModulationRouting routing){
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride){
        const uint64_t row=i/columns,column=i%columns;uint64_t run=0;
        while(run+1<routing.count&&row>=routing.ends[run])++run;
        const uint64_t parameter=routing.rows[run]*(2*columns)+column;
        const float x=ferrum_bf16_decode(input[i]);
        const float factor=__fadd_rn(1.0f,parameters[parameter+columns]);
        const float product=__fmul_rn(x,factor);
        output[i]=__fadd_rn(product,parameters[parameter]);
    }
}
