// Explicit gate/up F32-scale Q8 MMQ; opt-in geometry policy, strict down.
// Packed Q4 staging, N128/J8/K256 mapping and continuous Stream-K work division
// adapted from ggml CUDA mmq-load-tiles.cuh, mmq-vec-dot.cuh and mmq.cuh,
// llama.cpp ce8caa6e60a03093351d6016a818720e0d46f0fb. See adjacent license.
// Numeric differences are deliberate: F32 a/b/delta, quantized integer sum,
// RN division/ties-away pack, separate F32 products/adds. No llama half metadata.
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
using byte=unsigned char;
constexpr unsigned NC=128,MR=8,PITCH=68;
struct Shared {
    unsigned codes[NC][PITCH];
    float a[NC][8],b[NC][8];
    unsigned q[MR][PITCH];
    float d[MR][8];
    int sum[MR][8];
};
static_assert(sizeof(Shared)==45696,"Rust launch and CUDA shared ABI");
__device__ __forceinline__ float read_half(const byte* p) {
    return __half2float(__ushort_as_half(unsigned(p[0])|(unsigned(p[1])<<8)));
}
__device__ __forceinline__ unsigned read_word(const byte* p) {
    if ((reinterpret_cast<uintptr_t>(p)&3)==0) return *reinterpret_cast<const unsigned*>(p);
    return unsigned(p[0])|(unsigned(p[1])<<8)|(unsigned(p[2])<<16)|(unsigned(p[3])<<24);
}
__device__ __forceinline__ void coefficients(const byte* b,unsigned g,float& a,float& z) {
    const byte* s=b+4;
    const unsigned scale=g<4?s[g]&63:(s[g+4]&15)|((s[g-4]>>6)<<4);
    const unsigned minimum=g<4?s[g+4]&63:(s[g+4]>>4)|((s[g]>>6)<<4);
    a=__fmul_rn(read_half(b),float(scale));z=__fmul_rn(read_half(b+2),float(minimum));
}
__device__ __forceinline__ unsigned code_word(const byte* b,unsigned group,unsigned word) {
    return (read_word(b+16+(group/2)*32+word*4)>>(4*(group%2)))&0x0f0f0f0fu;
}
// Probe consumes precisely the decoder used for the actual MMQ tile.
extern "C" __global__ void vnext_q4_stream_metadata(const byte* w,unsigned* q,float* a,float* b,unsigned groups) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<groups) {
        const byte* block=w+(i/8)*144;
        coefficients(block,i%8,a[i],b[i]);
        for(unsigned word=0;word<8;++word)q[i*8+word]=code_word(block,i%8,word);
    }
}
// One warp per K32. Transposed K256-major physical pack, no lossy metadata.
extern "C" __global__ void vnext_q4_stream_pack(const half* x,unsigned* q,float* d,int* sums,unsigned rows,unsigned inputs) {
    const unsigned lane=threadIdx.x%32;
    const size_t gid=size_t(blockIdx.x)*8+threadIdx.x/32;
    const unsigned groups=inputs/32;
    if(gid>=size_t(rows)*groups)return;
    const unsigned row=gid/groups,group=gid%groups;
    const float value=__half2float(x[gid*32+lane]);
    const bool invalid=__ballot_sync(0xffffffff,!isfinite(value))!=0;
    float mx=isfinite(value)?fabsf(value):0;
    for(unsigned step=16;step;step/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,step));
    const float delta=invalid?NAN:mx==0?0:__fdiv_rn(mx,127.0f);
    int qi=0;
    if(!invalid&&mx!=0)qi=int(fmaxf(-127.0f,fminf(127.0f,roundf(__fdiv_rn(value,delta)))));
    int sum=qi;
    for(unsigned step=16;step;step/=2)sum+=__shfl_down_sync(0xffffffff,sum,step);
    const size_t pg=(size_t(group/8)*rows+row)*8+group%8;
    if(lane==0){d[pg]=delta;sums[pg]=sum;}
    const unsigned q0=unsigned(qi)&255;
    const unsigned q1=__shfl_down_sync(0xffffffff,q0,1,4);
    const unsigned q2=__shfl_down_sync(0xffffffff,q0,2,4);
    const unsigned q3=__shfl_down_sync(0xffffffff,q0,3,4);
    if(lane%4==0)q[pg*8+lane/4]=q0|(q1<<8)|(q2<<16)|(q3<<24);
}
// Each CTA owns a continuous interval in tile*K256. A tile's completing CTA
// writes one F32 destination; a CTA's final incomplete tile writes one partial.
// No atomics, no F16 rounding before the final ordered fixup.
extern "C" __global__ void vnext_q4_stream_mmq(const unsigned* q,const float* d,const int* sums,
    const byte* w,float* scratch,unsigned rows,unsigned inputs,unsigned outputs,unsigned ctas) {
    extern __shared__ __align__(32) byte storage[];
    Shared& sh=*reinterpret_cast<Shared*>(storage);
    const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane/4,t=lane%4;
    const unsigned blocks=inputs/256,ntx=(outputs+127)/128,nty=(rows+7)/8;
    const unsigned tiles=ntx*nty;
    const uint64_t total=uint64_t(tiles)*blocks;
    uint64_t pos=uint64_t(blockIdx.x)*total/ctas;
    const uint64_t stop=uint64_t(blockIdx.x+1)*total/ctas;
    while(pos<stop) {
        const unsigned tile=pos/blocks,start=pos%blocks;
        const unsigned end=unsigned(min(uint64_t(blocks),stop-uint64_t(tile)*blocks));
        const unsigned first_col=(tile%ntx)*128,first_row=(tile/ntx)*8;
        float acc[4]={0,0,0,0};
        for(unsigned bi=start;bi<end;++bi) {
            for(unsigned col=tid/32;col<128;col+=8) {
                const unsigned word=tid%32;
                const bool valid=first_col+col<outputs;
                const byte* block=valid?w+(size_t(first_col+col)*blocks+bi)*144:w;
                const unsigned group=(word/8)*2,kword=word%8;
                sh.codes[col][group*8+kword]=valid?code_word(block,group,kword):0;
                sh.codes[col][(group+1)*8+kword]=valid?code_word(block,group+1,kword):0;
                if(word<8){float a=0,b=0;if(valid)coefficients(block,word,a,b);sh.a[col][word]=a;sh.b[col][word]=b;}
            }
            for(unsigned j=tid;j<8*64;j+=256) {
                const unsigned row=j/64,k=j%64;
                const size_t pg=(size_t(bi)*rows+first_row+row)*8+k/8;
                sh.q[row][k]=first_row+row<rows?q[pg*8+k%8]:0;
                if(k%8==0){sh.d[row][k/8]=first_row+row<rows?d[pg]:0;sh.sum[row][k/8]=first_row+row<rows?sums[pg]:0;}
            }
            __syncthreads();
            // Register-prefetch fragments and metadata, as in the pinned MMQ
            // implementation. Four warps of absent token rows are never created.
            unsigned a0[8],a1[8],a2[8],a3[8];
            float al[8],ah[8],bl[8],bh[8];
            #pragma unroll
            for(unsigned group=0;group<8;++group) {
                const unsigned col=warp*16+g,k=group*8;
                a0[group]=sh.codes[col][k+t];a1[group]=sh.codes[col+8][k+t];
                a2[group]=sh.codes[col][k+t+4];a3[group]=sh.codes[col+8][k+t+4];
                al[group]=sh.a[col][group];ah[group]=sh.a[col+8][group];
                bl[group]=sh.b[col][group];bh[group]=sh.b[col+8][group];
            }
            #pragma unroll
            for(unsigned group=0;group<8;++group) {
                const unsigned b0=sh.q[g][group*8+t],b1=sh.q[g][group*8+t+4];
                const unsigned r0=2*t,r1=2*t+1;
                const float d0=sh.d[r0][group],d1=sh.d[r1][group];
                const int s0=sh.sum[r0][group],s1=sh.sum[r1][group];
                int c0=0,c1=0,c2=0,c3=0;
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
                    : "r"(a0[group]),"r"(a1[group]),"r"(a2[group]),"r"(a3[group]),"r"(b0),"r"(b1));
                acc[0]=__fadd_rn(acc[0],__fmul_rn(d0,__fsub_rn(__fmul_rn(al[group],float(c0)),__fmul_rn(bl[group],float(s0)))));
                acc[1]=__fadd_rn(acc[1],__fmul_rn(d1,__fsub_rn(__fmul_rn(al[group],float(c1)),__fmul_rn(bl[group],float(s1)))));
                acc[2]=__fadd_rn(acc[2],__fmul_rn(d0,__fsub_rn(__fmul_rn(ah[group],float(c2)),__fmul_rn(bh[group],float(s0)))));
                acc[3]=__fadd_rn(acc[3],__fmul_rn(d1,__fsub_rn(__fmul_rn(ah[group],float(c3)),__fmul_rn(bh[group],float(s1)))));
            }
            __syncthreads();
        }
        const size_t target=end==blocks?size_t(tile)*1024:size_t(tiles+blockIdx.x)*1024;
        #pragma unroll
        for(unsigned n=0;n<4;++n){unsigned row=2*t+n%2,col=warp*16+g+(n/2)*8;scratch[target+row*128+col]=acc[n];}
        pos=uint64_t(tile)*blocks+end;
        __syncthreads();
    }
}
extern "C" __global__ void vnext_q4_stream_fixup(const float* scratch,half* y,
    unsigned rows,unsigned inputs,unsigned outputs,unsigned stride,unsigned offset,unsigned ctas) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=rows*outputs)return;
    const unsigned row=i/outputs,col=i%outputs,ntx=(outputs+127)/128,blocks=inputs/256;
    const unsigned tile=(row/8)*ntx+col/128,tiles=ntx*((rows+7)/8),local=(row%8)*128+col%128;
    const uint64_t total=uint64_t(tiles)*blocks,begin=uint64_t(tile)*blocks,end=begin+blocks;
    const unsigned first=((begin+1)*ctas-1)/total,last=(end*ctas-1)/total;
    float sum=0;
    for(unsigned c=first;c<=last;++c) {
        const uint64_t stop=uint64_t(c+1)*total/ctas;
        if(stop<end)sum=__fadd_rn(sum,scratch[size_t(tiles+c)*1024+local]);
        else sum=__fadd_rn(sum,scratch[size_t(tile)*1024+local]);
    }
    y[size_t(row)*stride+offset+col]=__float2half_rn(sum);
}
