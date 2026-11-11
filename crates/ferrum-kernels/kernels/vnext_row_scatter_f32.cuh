#pragma once
struct FerrumRowScatterF32Shape {
    uint64_t rows,columns;
    uint64_t destination[32],source[32],length[32];
    uint32_t runs,padding;
};
static_assert(sizeof(FerrumRowScatterF32Shape)==792,"scatter launch ABI");
extern "C" __global__ void ferrum_vnext_row_scatter_f32(
    const uint32_t* base,const uint32_t* updates,uint32_t* output,
    const FerrumRowScatterF32Shape shape,int add) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<shape.rows*shape.columns;i+=uint64_t(gridDim.x)*blockDim.x) {
        const uint64_t row=i/shape.columns,column=i%shape.columns;
        bool selected=false;
        for(uint32_t run=0;run<shape.runs;++run) {
            if(row>=shape.destination[run]&&row-shape.destination[run]<shape.length[run]) {
                const uint32_t update=updates[(shape.source[run]+row-shape.destination[run])*shape.columns+column];
                output[i]=add?__float_as_uint(__fadd_rn(__uint_as_float(base[i]),__uint_as_float(update))):update;
                selected=true;break;
            }
        }
        if(!selected) output[i]=base[i];
    }
}
