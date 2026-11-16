#pragma once
// The shared TU uses --ftz=false --fmad=false and precise division/sqrt.
// Explicit __fmaf_rn is intentional; no compiler contraction is assumed.
extern "C" __global__ void ferrum_vnext_channelwise_f32(
    const float* input,const float* parameter1,const float* parameter2,float* output,
    uint64_t elements,uint64_t source_columns,uint64_t columns,
    uint64_t group_width,uint64_t column_start,uint64_t column_count,uint32_t kind){
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=stride){
        const uint64_t column=i%columns;
        if(kind==0){
            const uint64_t source=(i/columns)*source_columns+(column/column_count)*group_width+column_start+column%column_count;
            reinterpret_cast<uint32_t*>(output)[i]=reinterpret_cast<const uint32_t*>(input)[source];
        }else if(kind==1){
            output[i]=__fadd_rn(input[i],parameter1[column]);
        }else if(kind==2){
            output[i]=__fmaf_rn(parameter1[i],parameter2[column],input[i]);
        }else if(kind==4){
            output[i]=__fadd_rn(input[i],__fmul_rn(parameter1[i],parameter2[column]));
        }else if(kind==5){
            const uint64_t base=(i/columns)*source_columns, groups=source_columns/group_width;
            float sum=0.0f;
            for(uint64_t g=0;g<groups;g++) sum=__fadd_rn(sum,input[base+g*group_width+column]);
            output[i]=__fdiv_rn(sum,float(groups));
        }else if(kind==6){
            const uint64_t base=(i/columns)*source_columns;
            const uint64_t start=column*source_columns/columns, numerator=(column+1)*source_columns;
            const uint64_t end=numerator/columns+(numerator%columns!=0);
            float sum=0.0f;
            for(uint64_t j=start;j<end;j++) sum=__fadd_rn(sum,input[base+j]);
            output[i]=__fdiv_rn(sum,float(end-start));
        }else{
            const uint32_t bits=reinterpret_cast<const uint32_t*>(input)[i];
            const uint32_t magnitude=bits&0x7fffffffU;
            reinterpret_cast<uint32_t*>(output)[i]=magnitude>0x7f800000U?0U:
                (magnitude==0x7f800000U?((bits&0x80000000U)|0x7f7fffffU):bits);
        }
    }
}
