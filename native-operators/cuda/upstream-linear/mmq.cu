// Artifact adapter of llama.cpp d81235049384534c167caea52b85a694f6103d14.
// The MMQ headers remain in the separately pinned external source directory.
// Copied quantizer/launch policy: upstream ggml/src/ggml-cuda/{quantize.cu,mmq.cuh}.
// Local changes: caller-owned scratch, closed dense three-format ABI, F16 boundaries.
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

#include "boundary.h"
#include "marker.cuh"

#include "mmq.cuh"
#include "quantize.cuh"
#include <limits>
#include <cstdarg>

// Only the upstream fatal assertion helper is needed from GGML's host runtime.
// An exception boundary translates host invariants to typed errors; no GGML
// tensor/backend/pool implementation is linked. Device errors return CUDA status.
// Private versioned C ABI; no allocations, device queries or synchronization in launch.
struct TestMmqPlan {
    uint32_t abi, format, rows, inputs, outputs, padded_inputs;
    uint32_t j, i, nthreads, shared_bytes, blocks, tiles_y, fixup;
    uint64_t converted_bytes, packed_bytes, output_bytes, fixup_bytes;
};

// F16->F32 is exact. These boundary kernels are counted in inclusive timing.
static __global__ void test_mmq_convert(const half * input, float * converted,
        int rows, int inputs, int input_stride, int padded_inputs) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < rows*padded_inputs) {
        const int row = index/padded_inputs, col = index%padded_inputs;
        converted[index] = col < inputs ? __half2float(input[row*input_stride + col]) : 0.0f;
    }
}
static __global__ void test_mmq_cast(const float * output, half * result,
        int rows, int outputs, int output_stride) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < rows*outputs) {
        result[(index/outputs)*output_stride + index%outputs] = __float2half_rn(output[index]);
    }
}

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
// End pinned pack with versioned marker extension.

static bool test_mmq_format(uint32_t format) {
    return format == GGML_TYPE_IQ4_XS || format == GGML_TYPE_Q4_K || format == GGML_TYPE_Q5_K;
}

static int ferrum_test_mmq_plan(uint32_t format, uint32_t rows, uint32_t inputs,
        uint32_t outputs, uint32_t cc, uint32_t nsm, uint64_t shared_limit, TestMmqPlan * out) {
    // Dense Columns only. The existing small-row selector is unchanged; larger
    // admitted rows use the already instantiated J32 kernel over multiple tiles.
    if (!out || !test_mmq_format(format) || !rows || rows > 2048 || !inputs || inputs%256 ||
            !outputs || !nsm || cc < GGML_CUDA_CC_AMPERE ||
            uint64_t(rows)*inputs > INT_MAX || uint64_t(rows)*outputs > INT_MAX ||
            uint64_t(outputs)*(inputs/256) > INT_MAX ||
            uint64_t(inputs) + MATRIX_ROW_PADDING > INT_MAX) return -1;
    const auto type = static_cast<ggml_type>(format);
    const bool fallback = outputs%128 != 0;
    int best_j = 0, best_tiles = INT_MAX;
    if (rows > 32) {
        const auto config = ggml_cuda_mmq_get_config(type, 32, fallback, cc);
        if (config.type == GGML_TYPE_COUNT || config.J != 32 || config.I != 128 ||
                mmq_get_nbytes_shared(config, cc) > shared_limit) return -2;
        best_j = 32;
    } else {
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
static int test_mmq_dispatch(const TestMmqPlan & p, const void * weights, const void * packed,
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
static int test_mmq_j(const TestMmqPlan & p, const void * w, const void * q,
        float * out, float * fixup, cudaStream_t stream, bool setup) {
    switch (p.j) {
        case 8: return test_mmq_dispatch<Type, 8, Fallback>(p,w,q,out,fixup,stream,setup);
        case 16: return test_mmq_dispatch<Type,16,Fallback>(p,w,q,out,fixup,stream,setup);
        case 24: return test_mmq_dispatch<Type,24,Fallback>(p,w,q,out,fixup,stream,setup);
        case 32: return test_mmq_dispatch<Type,32,Fallback>(p,w,q,out,fixup,stream,setup);
        default: return -4;
    }
}
template<ggml_type Type>
static int test_mmq_type(const TestMmqPlan & p, const void * w, const void * q,
        float * out, float * fixup, cudaStream_t stream, bool setup) {
    return p.outputs%128 ? test_mmq_j<Type,true>(p,w,q,out,fixup,stream,setup)
                        : test_mmq_j<Type,false>(p,w,q,out,fixup,stream,setup);
}
static int ferrum_test_mmq_dot(const TestMmqPlan * p, const void * weights,
        const void * packed, float * output, float * fixup, cudaStream_t stream, int setup) {
    if (!p || p->abi != 1) return -1;
    if (!setup && (!weights || !packed || !output || (p->fixup && !fixup) ||
            reinterpret_cast<uintptr_t>(weights)%4 || reinterpret_cast<uintptr_t>(packed)%16 ||
            reinterpret_cast<uintptr_t>(output)%4 || reinterpret_cast<uintptr_t>(fixup)%4)) return -5;
    switch (p->format) {
        case GGML_TYPE_IQ4_XS: return test_mmq_type<GGML_TYPE_IQ4_XS>(*p,weights,packed,output,fixup,stream,setup);
        case GGML_TYPE_Q4_K: return test_mmq_type<GGML_TYPE_Q4_K>(*p,weights,packed,output,fixup,stream,setup);
        case GGML_TYPE_Q5_K: return test_mmq_type<GGML_TYPE_Q5_K>(*p,weights,packed,output,fixup,stream,setup);
        default: return -1;
    }
}
static int ferrum_test_mmq_pack(const TestMmqPlan * p, const half * input,
        uint32_t input_stride, float * converted, void * packed, cudaStream_t stream) {
    if (!p || p->abi != 1 || !input || !converted || !packed || input_stride < p->inputs ||
            uint64_t(input_stride)*p->rows > INT_MAX ||
            reinterpret_cast<uintptr_t>(input)%2 || reinterpret_cast<uintptr_t>(converted)%16 ||
            reinterpret_cast<uintptr_t>(packed)%16 || !test_mmq_format(p->format)) return -5;
    test_mmq_convert<<<(p->rows*p->padded_inputs+255)/256,256,0,stream>>>(
        input, converted, p->rows, p->inputs, input_stride, p->padded_inputs);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    const dim3 grid(p->rows,p->padded_inputs/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ),1);
    if (p->format == GGML_TYPE_IQ4_XS) {
        quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4,false><<<grid,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,stream>>>(
            converted,nullptr,packed,p->inputs,p->padded_inputs,p->rows*p->padded_inputs,
            p->rows*p->padded_inputs,p->padded_inputs,p->rows,1,0);
    } else {
        quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4,false><<<grid,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,stream>>>(
            converted,nullptr,packed,p->inputs,p->padded_inputs,p->rows*p->padded_inputs,
            p->rows*p->padded_inputs,p->padded_inputs,p->rows,1,0);
    }
    return cudaGetLastError();
}
static int ferrum_test_mmq_cast(const TestMmqPlan * p, const float * output,
        half * result, uint32_t stride, cudaStream_t stream) {
    if (!p || p->abi != 1 || !output || !result || stride < p->outputs ||
            uint64_t(stride)*p->rows > INT_MAX ||
            reinterpret_cast<uintptr_t>(output)%4 || reinterpret_cast<uintptr_t>(result)%2) return -5;
    test_mmq_cast<<<(p->rows*p->outputs+255)/256,256,0,stream>>>(output,result,p->rows,p->outputs,stride);
    return cudaGetLastError();
}

static int upstream_mmq_plan(const FerrumUpstreamLinearRequestV1 * r,
        FerrumUpstreamLinearPlanV1 & out, TestMmqPlan & p) {
    if (!ferrum_upstream_request(r) || r->layout!=0) return -1;
    const int status=ferrum_test_mmq_plan(r->format,r->rows,r->inputs,r->outputs,
        r->cc,r->sm_count,r->shared_limit,&p);
    if (status) return status;
    out={}; out.request=*r; out.abi=1; out.size=sizeof(out); out.algorithm=1;
    out.pack_abi=r->format==23 ? 1 : 2; out.padded_inputs=p.padded_inputs;
    out.padded_outputs=p.outputs;
    out.guard_blocks=(p.packed_bytes-uint64_t(p.rows)*p.padded_inputs*9/8)/144;
    out.j=p.j; out.i=p.i; out.nthreads=p.nthreads; out.shared_bytes=p.shared_bytes;
    out.blocks=p.blocks; out.tiles_y=p.tiles_y; out.fixup=p.fixup;
    const uint64_t block_bytes=r->format==12 ? 144 : r->format==13 ? 176 : 136;
    out.weight_bytes=uint64_t(p.outputs)*(p.inputs/256)*block_bytes;
    out.converted_bytes=p.converted_bytes; out.packed_bytes=p.packed_bytes;
    out.output_bytes=p.output_bytes; out.fixup_bytes=p.fixup_bytes;
    return 0;
}
static bool upstream_mmq_valid(const FerrumUpstreamLinearPlanV1 * p, TestMmqPlan & native) {
    FerrumUpstreamLinearPlanV1 expected{};
    return p && !upstream_mmq_plan(&p->request,expected,native) && ferrum_upstream_same(*p,expected);
}
extern "C" int ferrum_upstream_mmq_plan_v1(const FerrumUpstreamLinearRequestV1 * r,
        FerrumUpstreamLinearPlanV1 * out) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        if (!out || !r || r->rows > 32) return -1;
        FerrumUpstreamLinearPlanV1 candidate{}; TestMmqPlan native{};
        int status=upstream_mmq_plan(r,candidate,native);
        if (status) return status;
        status=ferrum_upstream_verify_device(*r); if (status) return status;
        status=ferrum_test_mmq_dot(&native,nullptr,nullptr,nullptr,nullptr,nullptr,1);
        if (!status) *out=candidate;
        return status;
    });
}
// Separate capability: repackaging an old archive cannot admit large prefill.
// This extends geometry, not the V1/MarkerV2 pack or accumulation semantics.
extern "C" int ferrum_upstream_mmq_prefill_plan_v1(const FerrumUpstreamLinearRequestV1 * r,
        FerrumUpstreamLinearPlanV1 * out) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        if (!out || !r || r->rows <= 32 || r->rows > 2048 || r->layout != 0) return -1;
        FerrumUpstreamLinearPlanV1 candidate{}; TestMmqPlan native{};
        int status=upstream_mmq_plan(r,candidate,native);
        if (status) return status;
        status=ferrum_upstream_verify_device(*r); if (status) return status;
        status=ferrum_test_mmq_dot(&native,nullptr,nullptr,nullptr,nullptr,nullptr,1);
        if (!status) *out=candidate;
        return status;
    });
}
extern "C" int ferrum_upstream_mmq_pack_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,uint32_t stride,void * converted,void * packed,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan native{}; if (!upstream_mmq_valid(p,native)) return -1;
        return ferrum_test_mmq_pack(&native,static_cast<const half *>(input),stride,
            static_cast<float *>(converted),packed,static_cast<cudaStream_t>(stream));
    });
}
extern "C" int ferrum_upstream_mmq_dot_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights,const void * packed,void * output,void * fixup,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan native{}; if (!upstream_mmq_valid(p,native)) return -1;
        return ferrum_test_mmq_dot(&native,weights,packed,static_cast<float *>(output),
            static_cast<float *>(fixup),static_cast<cudaStream_t>(stream),0);
    });
}
extern "C" int ferrum_upstream_mmq_cast_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,void * output,uint32_t stride,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan native{}; if (!upstream_mmq_valid(p,native)) return -1;
        return ferrum_test_mmq_cast(&native,static_cast<const float *>(input),
            static_cast<half *>(output),stride,static_cast<cudaStream_t>(stream));
    });
}

extern "C" int ferrum_upstream_mmq_pack_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,uint32_t stride,void * converted,void * packed,void * flags,void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{}; if(!upstream_mmq_valid(p,n)) return -1;
        if(!input||!converted||!packed||!flags||stride<n.inputs||uint64_t(stride)*n.rows>INT_MAX||
            uintptr_t(input)%2||uintptr_t(converted)%16||uintptr_t(packed)%16||uintptr_t(flags)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(flags,0,n.rows*sizeof(uint32_t),stream);
        if(status!=cudaSuccess) return status;
        if(p->guard_blocks) {
            const uint64_t logical=uint64_t(n.rows)*n.padded_inputs*9/8;
            status=cudaMemsetAsync(static_cast<char *>(packed)+logical,0,p->guard_blocks*144,stream);
            if(status!=cudaSuccess) return status;
        }
        test_mmq_convert<<<(n.rows*n.padded_inputs+255)/256,256,0,stream>>>(
            static_cast<const half *>(input),static_cast<float *>(converted),n.rows,n.inputs,stride,n.padded_inputs);
        status=cudaGetLastError(); if(status!=cudaSuccess) return status;
        const dim3 grid(n.rows,n.padded_inputs/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ),1);
        if(n.format==GGML_TYPE_IQ4_XS) {
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4,false,true><<<grid,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,stream>>>(
                static_cast<const float *>(converted),nullptr,packed,n.inputs,n.padded_inputs,
                n.rows*n.padded_inputs,n.rows*n.padded_inputs,n.padded_inputs,n.rows,1,0,static_cast<uint32_t *>(flags));
        } else {
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4,false,true><<<grid,CUDA_QUANTIZE_BLOCK_SIZE_MMQ,0,stream>>>(
                static_cast<const float *>(converted),nullptr,packed,n.inputs,n.padded_inputs,
                n.rows*n.padded_inputs,n.rows*n.padded_inputs,n.padded_inputs,n.rows,1,0,static_cast<uint32_t *>(flags));
        }
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_mmq_check_weights_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights,void * flag,void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{}; if(!upstream_mmq_valid(p,n)) return -1;
        if(!weights||!flag||uintptr_t(weights)%4||uintptr_t(flag)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(flag,0,4,stream); if(status!=cudaSuccess) return status;
        const uint64_t blocks=uint64_t(p->padded_outputs)*(n.inputs/256);
        marker_check_weights<<<(blocks+255)/256,256,0,stream>>>(static_cast<const unsigned char *>(weights),blocks,n.format,true,static_cast<uint32_t *>(flag));
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_mmq_cast_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,void * output,uint32_t stride,const void * rows,const void * weights,void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{}; if(!upstream_mmq_valid(p,n)) return -1;
        if(!input||!output||!rows||!weights||stride<n.outputs||uint64_t(stride)*n.rows>INT_MAX||
            uintptr_t(input)%4||uintptr_t(output)%2||uintptr_t(rows)%4||uintptr_t(weights)%4) return -5;
        marker_cast<<<(n.rows*n.outputs+255)/256,256,0,static_cast<cudaStream_t>(s)>>>(
            static_cast<const float *>(input),static_cast<half *>(output),n.rows,n.outputs,stride,
            static_cast<const uint32_t *>(rows),static_cast<const uint32_t *>(weights));
        return cudaGetLastError();
    });
}
