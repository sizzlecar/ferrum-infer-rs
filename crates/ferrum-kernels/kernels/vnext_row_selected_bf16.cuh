#pragma once
struct FerrumRowSelectedBf16Routing { uint64_t ends[32]; uint64_t rows[32]; uint64_t count; };
static_assert(sizeof(FerrumRowSelectedBf16Routing)==520,"compact BF16 routing ABI");
static_assert(alignof(FerrumRowSelectedBf16Routing)==8,"compact BF16 routing alignment");
__device__ __forceinline__ uint64_t ferrum_selected_parameter_bf16(uint64_t row, const FerrumRowSelectedBf16Routing& routing) {
    uint64_t run=0;
    while(run+1<routing.count && row>=routing.ends[run]) ++run;
    return routing.rows[run];
}
// Precise TU: all three explicit BF16 conversions match expanded modulation
// then residual add. No table expansion, contraction, global scratch or FTZ.
extern "C" __global__ void ferrum_vnext_row_selected_modulation_bf16(
    const uint16_t* input,const uint16_t* table,uint16_t* output,
    uint64_t elements,uint64_t columns,uint64_t parameter_columns,
    uint64_t shift_column,uint64_t scale_column,FerrumRowSelectedBf16Routing routing) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        const uint64_t parameter=ferrum_selected_parameter_bf16(i/columns,routing)*parameter_columns+i%columns;
        const float scale=ferrum_bf16_decode(table[parameter+scale_column]);
        const float factor=ferrum_bf16_decode(ferrum_bf16_rne(__fadd_rn(1.0f,scale)));
        const float product=ferrum_bf16_decode(ferrum_bf16_rne(__fmul_rn(ferrum_bf16_decode(input[i]),factor)));
        output[i]=ferrum_bf16_rne(__fadd_rn(product,ferrum_bf16_decode(table[parameter+shift_column])));
    }
}
// Match expanded addcmul: the product stays F32, with separate RN32 multiply
// and add followed by one BF16 conversion. This must never become a single FMA.
extern "C" __global__ void ferrum_vnext_row_selected_addcmul_bf16(
    const uint16_t* input,const uint16_t* table,const uint16_t* update,uint16_t* output,
    uint64_t elements,uint64_t columns,uint64_t parameter_columns,
    uint64_t gate_column,FerrumRowSelectedBf16Routing routing) {
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=uint64_t(gridDim.x)*blockDim.x) {
        const uint64_t parameter=ferrum_selected_parameter_bf16(i/columns,routing)*parameter_columns+gate_column+i%columns;
        const float product=__fmul_rn(ferrum_bf16_decode(table[parameter]),ferrum_bf16_decode(update[i]));
        output[i]=ferrum_bf16_rne(__fadd_rn(ferrum_bf16_decode(input[i]),product));
    }
}
