// Artifact adapter of llama.cpp d81235049384534c167caea52b85a694f6103d14.
// The MMQ headers remain in the separately pinned external source directory.
// Copied quantizer/launch policy: upstream ggml/src/ggml-cuda/{quantize.cu,mmq.cuh}.
// Test adaptation: caller-owned scratch, Q6_K-only geometry, explicit F32 boundaries.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Production Q6-only F32 boundary. Numerical expressions match the qualified prototype.
#include "boundary.h"
#include "marker.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include <limits>
#include <climits>

struct Q6MmqGeometry {
    uint32_t abi, format, rows, inputs, outputs, padded_inputs;
    uint32_t j, i, nthreads, shared_bytes, blocks, tiles_y, fixup;
    uint64_t converted_bytes, packed_bytes, output_bytes, fixup_bytes;
};

// Pinned V1 pack math: quantize.cu:456-555; explicit MarkerV2 branches below.
template <mmq_q8_1_ds_layout ds_layout, bool scatter, bool Marker = false>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used, uint32_t * row_flags=nullptr) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    uint32_t invalid=0;
    if constexpr(Marker) {
        invalid=!marker_finite(xi.x)||!marker_finite(xi.y)||!marker_finite(xi.z)||!marker_finite(xi.w);
        for(int offset=vals_per_scale/8;offset>0;offset>>=1) invalid|=__shfl_xor_sync(0xffffffff,invalid,offset,32);
        if constexpr(ds_layout==MMQ_Q8_1_DS_LAYOUT_DS4) invalid|=!marker_half_finite(__float2half_rn(sum));
    }
    const bool zero=Marker && amax==0.0f;
    const float d_inv = 127.0f / amax;
    if constexpr(Marker) {
        // F32 extends beyond the previously qualified F16 input domain.
        // All-subnormal groups may be flushed by upstream fast math: poison
        // rather than silently treating a nonzero logical group as +zero.
        uint32_t nonzero=(__float_as_uint(xi.x)&0x7fffffffU)!=0 ||
            (__float_as_uint(xi.y)&0x7fffffffU)!=0 ||
            (__float_as_uint(xi.z)&0x7fffffffU)!=0 ||
            (__float_as_uint(xi.w)&0x7fffffffU)!=0;
        for(int offset=vals_per_scale/8;offset>0;offset>>=1)
            nonzero|=__shfl_xor_sync(0xffffffff,nonzero,offset,32);
        invalid|=(zero && nonzero) || (!zero && (!marker_finite(d_inv) || d_inv==0.0f));
    }
    char4 q;
    q.x = Marker && (zero||invalid) ? 0 : roundf(xi.x*d_inv);
    q.y = Marker && (zero||invalid) ? 0 : roundf(xi.y*d_inv);
    q.z = Marker && (zero||invalid) ? 0 : roundf(xi.z*d_inv);
    q.w = Marker && (zero||invalid) ? 0 : roundf(xi.w*d_inv);
    float d = 1.0f / d_inv;
    if constexpr(Marker) {
        if constexpr(ds_layout==MMQ_Q8_1_DS_LAYOUT_DS4) invalid|=!marker_half_finite(__float2half_rn(d));
        else invalid|=!marker_finite(d);
        if(zero && !invalid) {d=0.0f;sum=0.0f;}
        if(invalid) {d=marker_nan();sum=marker_nan();atomicOr(row_flags+blockIdx.x,1u);}
        if(zero||invalid) q=make_char4(0,0,0,0);
    }

    // write the block once (normal) or to each of the token's compact rows (scatter)
    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
            ib = ib0 + k_block*ne1 + blockIdx.x;
        }

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = Marker && invalid
                    ? __halves2half2(__ushort_as_half(0x7e00),__ushort_as_half(0x7e00))
                    : make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}

static int q6_mmq_prepare_geometry(uint32_t format, uint32_t rows, uint32_t inputs,
        uint32_t outputs, uint32_t cc, uint32_t nsm, uint64_t shared_limit, Q6MmqGeometry * out) {
    // Dense Columns only; preserve the pinned selector and D4 arithmetic.
    if (!out || format != GGML_TYPE_Q6_K || !rows || rows > 32 || !inputs || inputs%256 ||
            !outputs || !nsm || cc < GGML_CUDA_CC_AMPERE ||
            uint64_t(rows)*inputs > INT_MAX || uint64_t(rows)*outputs > INT_MAX ||
            uint64_t(outputs)*(inputs/256) > INT_MAX ||
            uint64_t(inputs) + MATRIX_ROW_PADDING > INT_MAX) return -1;
    const auto type = static_cast<ggml_type>(format);
    const bool fallback = outputs%128 != 0;
    int best_j = 0, best_tiles = INT_MAX;
    {
        // Same upstream selection loop, including device shared-memory eligibility.
        for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
            const auto config = ggml_cuda_mmq_get_config(type, j, fallback, cc);
            if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > shared_limit) continue;
            const int tiles = (rows + config.J - 1)/config.J;
            if (tiles < best_tiles) { best_j = j; best_tiles = tiles; }
        }
    }
    if (!best_j || best_j > 32) return -2;
    const auto config = ggml_cuda_mmq_get_config(type, best_j, fallback, cc);
    const uint64_t tiles_y = (uint64_t(outputs) + config.I - 1)/config.I;
    const uint64_t tiles_x = (uint64_t(rows) + config.J - 1)/config.J;
    const uint64_t total_tiles = tiles_x*tiles_y;
    if (total_tiles*(inputs/256) >= (1u << 30)) return -3;
    const uint64_t waves = (total_tiles + nsm - 1)/nsm;
    const uint64_t efficiency = 100*total_tiles/(nsm*waves);
    const uint64_t blocks = config.stream_k ? (efficiency >= 90 ? total_tiles : nsm) : total_tiles;
    const bool fixup = config.stream_k && total_tiles%blocks != 0;
    const uint64_t padded = GGML_PAD(uint64_t(inputs), MATRIX_ROW_PADDING);
    if (uint64_t(rows)*padded*9/8 > INT_MAX) return -3;
    // The pinned process_tile loader reads whole nthreads-sized batches of
    // int32, including inactive J columns. Bound the final K128-block load;
    // write-back row guards alone do not make those reads safe. K padding is
    // already part of the packed body. Keep the upstream guard as a floor.
    const uint64_t packed_body = uint64_t(rows)*padded*sizeof(block_q8_1_mmq)/QK8_1_MMQ;
    const uint64_t last_column_tile = (tiles_x - 1)*config.J;
    const uint64_t load_ints = GGML_PAD(uint64_t(config.J)*MMQ_TILE_Y_K, uint64_t(config.nthreads));
    const uint64_t read_end = ((uint64_t(inputs)/QK8_1_MMQ - 1)*rows + last_column_tile)*
        sizeof(block_q8_1_mmq) + load_ints*sizeof(int);
    const uint64_t read_tail = read_end > packed_body ? read_end - packed_body : 0;
    const uint64_t read_guard_blocks = (read_tail + sizeof(block_q8_1_mmq) - 1)/sizeof(block_q8_1_mmq);
    const uint64_t upstream_guard_blocks = ggml_cuda_mmq_get_J_max(type, fallback, cc, rows);
    const uint64_t guard_blocks = std::max(upstream_guard_blocks, read_guard_blocks);
    if (guard_blocks > 512) return -3;
    const uint64_t packed = packed_body + guard_blocks*sizeof(block_q8_1_mmq);
    *out = {1, format, rows, inputs, outputs, uint32_t(padded), uint32_t(config.J),
        uint32_t(config.I), uint32_t(config.nthreads), uint32_t(mmq_get_nbytes_shared(config, cc)),
        uint32_t(blocks), uint32_t(tiles_y), uint32_t(fixup), uint64_t(rows)*padded*sizeof(float),
        packed, uint64_t(rows)*outputs*sizeof(float), fixup ? blocks*config.J*config.I*sizeof(float) : 0};
    return 0;
}

template<ggml_type Type, int J, bool Fallback>
static int q6_mmq_dispatch(const Q6MmqGeometry & p, const void * weights, const void * packed,
        float * output, float * fixup, cudaStream_t stream, bool setup) {
    if (setup) {
        return cudaFuncSetAttribute(mul_mat_q<Type, J, Fallback>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, p.shared_bytes);
    }
    const auto one = init_fastdiv_values(1);
    const auto blocks_per_k = init_fastdiv_values(p.inputs/256);
    const auto ntx = init_fastdiv_values((p.rows + J - 1)/J);
    const int weight_stride = p.inputs/256;
    const int packed_channel_stride = p.rows*p.padded_inputs*sizeof(block_q8_1_mmq)/(128*sizeof(int));
    constexpr bool stream_k = ggml_cuda_mmq_get_config_ampere(Type, J, Fallback).stream_k;
    const dim3 block(32, p.nthreads/32, 1);
    const dim3 grid = stream_k ? dim3(p.blocks, 1, 1) : dim3(p.tiles_y, ntx.z, 1);
    mul_mat_q<Type, J, Fallback><<<grid, block, p.shared_bytes, stream>>>(
        static_cast<const char *>(weights), static_cast<const int *>(packed), nullptr, nullptr,
        output, fixup, nullptr, blocks_per_k, p.outputs, p.rows, weight_stride, p.rows, p.outputs,
        one, one, p.outputs*weight_stride, packed_channel_stride, p.rows*p.outputs,
        one, one, p.outputs*weight_stride, packed_channel_stride, p.rows*p.outputs, ntx);
    auto error = cudaGetLastError();
    if (error != cudaSuccess || !p.fixup) return error;
    mul_mat_q_stream_k_fixup<Type, J, Fallback>
        <<<dim3(p.blocks, p.i/32, 1), dim3(32, p.nthreads/64, 1), 0, stream>>>(
            nullptr, nullptr, output, fixup, blocks_per_k, p.outputs, p.rows, p.outputs,
            one, p.rows*p.outputs, one, p.rows*p.outputs, ntx);
    return cudaGetLastError();
}

template<ggml_type Type, bool Fallback>
static int q6_mmq_j(const Q6MmqGeometry & p, const void * w, const void * q,
        float * out, float * fixup, cudaStream_t stream, bool setup) {
    switch (p.j) {
        case 8: return q6_mmq_dispatch<Type, 8, Fallback>(p,w,q,out,fixup,stream,setup);
        case 16: return q6_mmq_dispatch<Type,16,Fallback>(p,w,q,out,fixup,stream,setup);
        case 24: return q6_mmq_dispatch<Type,24,Fallback>(p,w,q,out,fixup,stream,setup);
        case 32: return q6_mmq_dispatch<Type,32,Fallback>(p,w,q,out,fixup,stream,setup);
        default: return -4;
    }
}
template<ggml_type Type>
static int q6_mmq_type(const Q6MmqGeometry & p, const void * w, const void * q,
        float * out, float * fixup, cudaStream_t stream, bool setup) {
    return p.outputs%128 ? q6_mmq_j<Type,true>(p,w,q,out,fixup,stream,setup)
                        : q6_mmq_j<Type,false>(p,w,q,out,fixup,stream,setup);
}
// Independent 48/168-byte ABI. Every launch rebuilds and compares geometry.
// Allocation, live ranges and retained leases are owned by the typed provider.
static int q6_plan(const FerrumUpstreamQ6F32RequestV1 * r,
        FerrumUpstreamQ6F32PlanV1 & out, Q6MmqGeometry & n) {
    if(!r || r->abi!=1 || r->size!=sizeof(*r) || r->reserved || r->layout ||
        r->format!=GGML_TYPE_Q6_K || !r->shared_limit) return -1;
    const int status=q6_mmq_prepare_geometry(r->format,r->rows,r->inputs,r->outputs,
        r->cc,r->sm_count,r->shared_limit,&n);
    if(status) return status;
    out={}; out.request=*r; out.abi=1; out.size=sizeof(out); out.algorithm=1;
    out.pack_abi=1; out.padded_inputs=n.padded_inputs; out.padded_outputs=n.outputs;
    out.guard_blocks=(n.packed_bytes-uint64_t(n.rows)*n.padded_inputs*9/8)/144;
    out.j=n.j; out.i=n.i; out.nthreads=n.nthreads; out.shared_bytes=n.shared_bytes;
    out.blocks=n.blocks; out.tiles_y=n.tiles_y; out.fixup=n.fixup;
    out.weight_bytes=uint64_t(n.outputs)*(n.inputs/256)*210;
    out.converted_bytes=n.converted_bytes; out.packed_bytes=n.packed_bytes;
    out.output_bytes=n.output_bytes; out.fixup_bytes=n.fixup_bytes;
    return 0;
}
static bool q6_valid(const FerrumUpstreamQ6F32PlanV1 * p, Q6MmqGeometry & n) {
    FerrumUpstreamQ6F32PlanV1 expected{};
    return p && !q6_plan(&p->request,expected,n) && ferrum_upstream_q6_f32_same(*p,expected);
}
static __global__ void q6_copy_f32(const float * x,float * padded,int rows,int k,int stride,int pk) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*pk) padded[i]=i%pk<k ? x[(i/pk)*stride+i%pk] : 0.0f;
}
static __global__ void q6_weights(const unsigned char * w,uint64_t blocks,uint32_t * flag) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=blocks) return;
    const auto * b=w+i*210;
    const float d=__half2float(__ushort_as_half(uint16_t(b[208])|(uint16_t(b[209])<<8)));
    bool bad=!marker_finite(d);
    for(int j=0;j<16;++j) bad|=!marker_finite(d*float(static_cast<int8_t>(b[192+j])));
    if(bad) atomicOr(flag,1u);
}
static __global__ void q6_publish(const float * raw,float * y,int rows,int n,int stride,
        const uint32_t * row_flags,const uint32_t * weight_flag) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*n) {
        const float v=raw[i];
        y[(i/n)*stride+i%n]=(row_flags[i/n] || *weight_flag || !marker_finite(v))
            ? marker_nan() : v; // F32 boundary: no F16 rounding/overflow rule.
    }
}
extern "C" int ferrum_upstream_q6_f32_plan_v1(const FerrumUpstreamQ6F32RequestV1 * r,
        FerrumUpstreamQ6F32PlanV1 * out) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        if(!out) return -1;
        FerrumUpstreamQ6F32PlanV1 p{}; Q6MmqGeometry n{};
        int status=q6_plan(r,p,n); if(status) return status;
        status=ferrum_upstream_q6_f32_verify_device(*r); if(status) return status;
        status=q6_mmq_type<GGML_TYPE_Q6_K>(n,nullptr,nullptr,nullptr,nullptr,nullptr,true);
        if(!status) *out=p;
        return status;
    });
}
extern "C" int ferrum_upstream_q6_f32_pack_v1(const FerrumUpstreamQ6F32PlanV1 * p,
        const void * input,uint32_t stride,void * padded,void * packed,void * rows,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if(!q6_valid(p,n)) return -1;
        if(!input||!padded||!packed||!rows||stride<n.inputs||uint64_t(stride)*n.rows>INT_MAX||
            uintptr_t(input)%4||uintptr_t(padded)%16||uintptr_t(packed)%16||uintptr_t(rows)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(rows,0,n.rows*4,stream); if(status!=cudaSuccess) return status;
        if(p->guard_blocks) {
            status=cudaMemsetAsync(static_cast<char *>(packed)+uint64_t(n.rows)*n.padded_inputs*9/8,
                0,p->guard_blocks*144,stream); if(status!=cudaSuccess) return status;
        }
        q6_copy_f32<<<(n.rows*n.padded_inputs+255)/256,256,0,stream>>>(static_cast<const float *>(input),
            static_cast<float *>(padded),n.rows,n.inputs,stride,n.padded_inputs);
        status=cudaGetLastError(); if(status!=cudaSuccess) return status;
        const dim3 grid(n.rows,n.padded_inputs/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ),1);
        quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4,false,true><<<grid,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,stream>>>(
            static_cast<const float *>(padded),nullptr,packed,n.inputs,n.padded_inputs,
            n.rows*n.padded_inputs,n.rows*n.padded_inputs,n.padded_inputs,n.rows,1,0,
            static_cast<uint32_t *>(rows));
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_q6_f32_dot_v1(const FerrumUpstreamQ6F32PlanV1 * p,
        const void * w,const void * packed,void * raw,void * fixup,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if(!q6_valid(p,n)) return -1;
        if(!w||!packed||!raw||(n.fixup&&!fixup)||uintptr_t(w)%4||uintptr_t(packed)%16||
            uintptr_t(raw)%4||uintptr_t(fixup)%4) return -5;
        return q6_mmq_type<GGML_TYPE_Q6_K>(n,w,packed,static_cast<float *>(raw),
            static_cast<float *>(fixup),static_cast<cudaStream_t>(s),false);
    });
}
extern "C" int ferrum_upstream_q6_f32_check_weights_v1(const FerrumUpstreamQ6F32PlanV1 * p,
        const void * w,void * flag,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if(!q6_valid(p,n)) return -1;
        if(!w||!flag||uintptr_t(w)%4||uintptr_t(flag)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(flag,0,4,stream); if(status!=cudaSuccess) return status;
        const uint64_t count=uint64_t(n.outputs)*(n.inputs/256);
        q6_weights<<<(count+255)/256,256,0,stream>>>(static_cast<const unsigned char *>(w),count,
            static_cast<uint32_t *>(flag)); return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_q6_f32_publish_v1(const FerrumUpstreamQ6F32PlanV1 * p,
        const void * raw,void * output,uint32_t stride,const void * rows,const void * weight,void * s) noexcept {
    return ferrum_upstream_q6_f32_boundary([&]() -> int {
        Q6MmqGeometry n{}; if(!q6_valid(p,n)) return -1;
        if(!raw||!output||!rows||!weight||stride<n.outputs||uint64_t(stride)*n.rows>INT_MAX||
            uintptr_t(raw)%4||uintptr_t(output)%4||uintptr_t(rows)%4||uintptr_t(weight)%4) return -5;
        q6_publish<<<(n.rows*n.outputs+255)/256,256,0,static_cast<cudaStream_t>(s)>>>(
            static_cast<const float *>(raw),static_cast<float *>(output),n.rows,n.outputs,stride,
            static_cast<const uint32_t *>(rows),static_cast<const uint32_t *>(weight));
        return cudaGetLastError();
    });
}

#include "f16_adapter.cuh"
