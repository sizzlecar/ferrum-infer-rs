// Isolated next-K L2-prefetch pilot of llama.cpp
// d81235049384534c167caea52b85a694f6103d14, MIT license (llama-MIT.txt).
// Device process_tile/main are pinned copies below with one transport-only
// insertion; the baseline planner/loaders/vector math/fixup remain unchanged.
// Included only after the original MMQ adapter definitions in mmq.cu.
#pragma once

namespace ferrum_mmq_next_k {

template<ggml_type Type, int I>
static __device__ __forceinline__ void prefetch_next_weight_tile(
        const char * x, int offset_x, int stride_row_x, int tile_x_max_i,
        int next_kb, int kb0_stop) {
    static_assert(Type == GGML_TYPE_Q4_K || Type == GGML_TYPE_Q5_K ||
        Type == GGML_TYPE_IQ4_XS, "next-K prefetch has a closed block ABI");
    constexpr size_t block_bytes = Type == GGML_TYPE_Q4_K ? sizeof(block_q4_K)
        : Type == GGML_TYPE_Q5_K ? sizeof(block_q5_K) : sizeof(block_iq4_xs);
    static_assert(block_bytes > 128, "hint addresses must be inside a block");
    // Never hint across this CTA's actual Stream-K segment or a partial I tile.
    if (next_kb >= kb0_stop) return;
    const int lane = int(threadIdx.y)*int(blockDim.x) + int(threadIdx.x);
    const int lanes = int(blockDim.x)*int(blockDim.y);
    const int valid_rows = min(I, tile_x_max_i + 1);
    for (int i = lane; i < valid_rows; i += lanes) {
        const size_t block_index = size_t(offset_x) + size_t(i)*size_t(stride_row_x)
            + size_t(next_kb);
        const char * block = x + block_index*block_bytes;
        // The first, +128 and last-byte hints cover possible three-line blocks
        // without alignment predicates. Duplicate hints are intentional here.
        // They name bytes of the original validated block, grant no authority,
        // and neither load numerical values nor retain a new owner.
        asm volatile("prefetch.global.L2 [%0];" :: "l"(block));
        asm volatile("prefetch.global.L2 [%0];" :: "l"(block + 128));
        asm volatile("prefetch.global.L2 [%0];" :: "l"(block + block_bytes - 1));
    }
}
template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1 = GGML_PREC_Q8>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int              warp_size  = ggml_cuda_get_physical_warp_size();
    constexpr int              nwarps     = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
    constexpr int              qk         = ggml_cuda_type_traits<type>::qk;
    constexpr int              I          = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
    constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback, prec_src1>();
    constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback, prec_src1>();
    constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback, prec_src1>();

    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + J;
    int * tile_x = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);

#if defined(BLACKWELL_MMA_AVAILABLE)
    // FP4 tile stores 8 blocks. src1 above Q4 uses the generic
    // Q8_1 tile layout instead of the packed FP4 tile.
    constexpr int ne_block = ((type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) && prec_src1 == GGML_PREC_Q4) ?
        QK_FP4_MMQ : QK8_1_MMQ;
#else
    constexpr int ne_block = QK8_1_MMQ;
#endif  // defined(BLACKWELL_MMA_AVAILABLE)

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J*I / (nwarps*warp_size)] = {0.0f};

    constexpr int sz = sizeof(block_q8_1_mmq) / sizeof(int);

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
        {
            const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        prefetch_next_weight_tile<type, I>(x, offset_x, stride_row_x,
            tile_x_max_i, kb0 + blocks_per_iter, kb0_stop);

        vec_dot(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
#pragma unroll
            for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += nwarps * warp_size) {
                int l = l0 + threadIdx.y*warp_size + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);

        __syncthreads();
    }

    if (fixup) {
        write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J);
    } else {
        write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}


// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const float * __restrict__ y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx) {

    // Skip unused template specializations for faster compilation:
    if (ggml_cuda_mmq_get_config(type, J, fallback, prec_src1).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / warp_size;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);

    const uint32_t nty = (nrows_x + I - 1) / I; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

        if (j0 + nwarps*warp_size > J && j >= J) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    if constexpr (!ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1)) {
        const uint2 tmp2 = fast_div_modulo(blockIdx.z, nchannels_y);
        const int wt = tmp2.x;
        const int zt = tmp2.y;
        const int jt = blockIdx.y;
        const int it = blockIdx.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                return;
            }

            // __syncthreads(); // There is no previous tile that could cause a race condition.
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt*J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

        constexpr bool fixup = false;
        mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, 0, blocks_per_ne00.z);
        return;
    }

    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc      -= fastmodulo(kbc,      blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= fastmodulo(kbc_stop, blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = fastmodulo(kbc, blocks_per_ne00);
    int kb0_stop  = min(blocks_per_ne00.z, uint32_t(kb0_start + kbc_stop - kbc));
    while (kbc < kbc_stop && kb0_stop == int(blocks_per_ne00.z)) {
        int tmp = fastdiv(kbc, blocks_per_ne00);
        uint2 tmp2 = fast_div_modulo(tmp, ntx);
        const int jt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nchannels_y);
        const int zt = tmp2.y;
        tmp = tmp2.x;
        tmp2 = fast_div_modulo(tmp, nsamples_y);
        const int wt = tmp2.y;
        const int it = tmp2.x;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
        int offset_y_scale;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
        } else {
            GGML_UNUSED(offset_y_scale);
        }

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;
            if constexpr (type == GGML_TYPE_NVFP4) {
                offset_y_scale = 0;
            }

            if (jt*J >= col_diff) {
                kbc += blocks_per_ne00.z;
                kbc -= fastmodulo(kbc, blocks_per_ne00);

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
                const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

                if (j0 + nwarps*warp_size > J && j >= J) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*J + j];
            }
            __syncthreads();
        }

        offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
        offset_dst += it*I;
        const float * y_scale_tile = nullptr;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale += col_low + jt * J;
            y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
             stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00.z;
        kbc -= fastmodulo(kbc, blocks_per_ne00);

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00.z, uint32_t(kbc_stop - kbc));
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = fastdiv(kbc, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y       = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst     = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst;
    int offset_y_scale;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale = wt*nchannels_y.z*ncols_y + zt*ncols_y;
    } else {
        GGML_UNUSED(offset_y_scale);
    }

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;
        if constexpr (type == GGML_TYPE_NVFP4) {
            offset_y_scale = 0;
        }

        if (jt*J >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps*warp_size) {
            const int j = j0 + threadIdx.y*warp_size + threadIdx.x;

            if (j0 + nwarps*warp_size > J && j >= J) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y += (col_low + jt * J) * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it*I;
    const float * y_scale_tile = nullptr;
    if constexpr (type == GGML_TYPE_NVFP4) {
        offset_y_scale += col_low + jt * J;
        y_scale_tile = y_scale ? y_scale + offset_y_scale : nullptr;
    }

    const int tile_x_max_i = nrows_x  - it*I - 1;
    const int tile_y_max_j = col_diff - jt*J - 1;

    const int offset_x = fastdiv(wt, sample_ratio)*stride_sample_x + fastdiv(zt, channel_ratio)*stride_channel_x + it*I*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<type, J, fallback, fixup, prec_src1>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
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
    ::mul_mat_q_stream_k_fixup<Type, J, Fallback>
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
} // namespace ferrum_mmq_next_k

// Additive test-only exports use the unchanged 48-byte request/168-byte plan.
// pack/check/cast call the original exports; no inference route selects this.
extern "C" int ferrum_upstream_mmq_next_k_prefetch_plan_v1(
        const FerrumUpstreamLinearRequestV1 * r,
        FerrumUpstreamLinearPlanV1 * out) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        if (!out || !r || r->rows > 32) return -1;
        FerrumUpstreamLinearPlanV1 candidate{}; TestMmqPlan native{};
        int status = upstream_mmq_plan(r, candidate, native);
        if (status) return status;
        status = ferrum_upstream_verify_device(*r); if (status) return status;
        status = ferrum_mmq_next_k::ferrum_test_mmq_dot(
            &native, nullptr, nullptr, nullptr, nullptr, nullptr, 1);
        if (!status) *out = candidate;
        return status;
    });
}

extern "C" int ferrum_upstream_mmq_next_k_prefetch_dot_v1(
        const FerrumUpstreamLinearPlanV1 * p, const void * weights,
        const void * packed, void * output, void * fixup, void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan native{};
        if (!p || p->request.rows > 32 || !upstream_mmq_valid(p, native)) return -1;
        return ferrum_mmq_next_k::ferrum_test_mmq_dot(
            &native, weights, packed, static_cast<float *>(output),
            static_cast<float *>(fixup), static_cast<cudaStream_t>(stream), 0);
    });
}
