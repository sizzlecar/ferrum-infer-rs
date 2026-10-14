// Independent TEST TU: original dense split-MMQ plan, quantization and math.
// No product launcher, pinned header, tile, stream-K partition or reduction changes.
#include "mmq.cu"
#include <vector>

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

// Adapted from pinned llama.cpp d81235049384534c167caea52b85a694f6103d14:
// mmq.cuh process_tile/main/fixup, and this adapter's direct F16 quantizer.
// Only dense (ids_dst == nullptr) branches are retained. The process shell
// retains the pinned load_tiles/vec_dot/write_back and every __syncthreads.
// Async-dependent pointers carry no restrict qualifier in either compiler pass.

template <mmq_q8_1_ds_layout ds_layout, bool scatter, bool Marker, typename Input>
static __device__ __forceinline__ void test_mmq_pdl_quantize_body(
        const Input * x, const int32_t * ids, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used, uint32_t * row_flags) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    // The private global wrapper synchronizes before this body or its tail writes.

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? mmq_pack_load4(x, base_idx + i00) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1 = GGML_PREC_Q8>
static __device__ __forceinline__ void test_mmq_pdl_process_tile(
        const char * x, const int offset_x, const int * y,
        const int * ids_dst, float * dst, float * tmp_fixup,
        const float * y_scale,
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

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1), ggml_cuda_mmq_get_occupancy(type, J, fallback, prec_src1))
static __global__ void test_mmq_pdl_main(
        const char * x, const int * y, const int32_t * ids_dst,
        const int32_t * expert_bounds, float * dst, float * tmp_fixup,
        const float * y_scale,
        const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const uint3 channel_ratio, const uint3 nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const uint3 nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const uint3 ntx) {
    ggml_cuda_pdl_lc(); // Including skipped/no-work CTAs.


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
    // Independent scalar/shared setup above; packed reads and output writes below.
    ggml_cuda_pdl_sync();

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
        test_mmq_pdl_process_tile<type, J, fallback, fixup, prec_src1>
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
        test_mmq_pdl_process_tile<type, J, fallback, fixup, prec_src1>
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
    test_mmq_pdl_process_tile<type, J, fallback, fixup, prec_src1>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,
         stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

template <ggml_type type, int J, bool fallback, ggml_prec prec_src1 = GGML_PREC_Q8>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1)/2, 1)
static __global__ void test_mmq_pdl_fixup(
        const int32_t * ids_dst, const int32_t * expert_bounds, float * dst,
        float * tmp_last_tile, const uint3 blocks_per_ne00, const int nrows_x, const int ncols_dst,
        const int stride_col_dst, const uint3 nchannels_y, const int stride_channel_dst, const uint3 nsamples_y,
        const int stride_sample_dst, const uint3 ntx) {
    ggml_cuda_pdl_lc(); // No-work CTAs still permit the next launch.

    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = (ggml_cuda_mmq_get_nthreads(type, J, fallback, prec_src1) / 2) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback, prec_src1);
    constexpr int qk              = ggml_cuda_type_traits<type>::qk;
    constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback, prec_src1);
    constexpr int blocks_per_iter = ITER_K / qk;

    float sum[J / nwarps] = {0.0f};
    const int i = blockIdx.y*warp_size + threadIdx.x;

    const int nty = (nrows_x + I - 1) / I;

    const int bidx0 = blockIdx.x;

    // kbc == k block continuous, current index in continuous ijk space.
    int kbc0      = int64_t(blockIdx.x)    *(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
    int kbc0_stop = int64_t(blockIdx.x + 1)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;

    kbc0      -= fastmodulo(kbc0,      blocks_per_ne00) % blocks_per_iter;
    kbc0_stop -= fastmodulo(kbc0_stop, blocks_per_ne00) % blocks_per_iter;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, blocks_per_ne00) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, blocks_per_ne00) == fastdiv(kbc0_stop, blocks_per_ne00) && fastmodulo(kbc0_stop, blocks_per_ne00) != 0;
    // Even if every fixup CTA has no work, its completion must transitively
    // wait for main before the following cast can consume main's output.
    // Only independent kbc arithmetic above is allowed to overlap main.
    ggml_cuda_pdl_sync();
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    bool any_fixup = false;

    // Iterate over previous blocks and sum up partial sums written to fixup buffer.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        int kbc = int64_t(bidx)*(nsamples_y.z*nchannels_y.z*ntx.z*nty*blocks_per_ne00.z) / gridDim.x;
        kbc -= fastmodulo(kbc, blocks_per_ne00) % blocks_per_iter;

        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        any_fixup = true;


#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            sum[j0/nwarps] += tmp_last_tile[bidx*(J*I) + j*I + i];
        }

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, blocks_per_ne00) == 0 || fastdiv(kbc, blocks_per_ne00) < fastdiv(kbc0, blocks_per_ne00)) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    if (!any_fixup) {
        return;
    }

    int tmp = fastdiv(kbc0, blocks_per_ne00);
    uint2 tmp2 = fast_div_modulo(tmp, ntx);
    const int jt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nchannels_y);
    const int zt = tmp2.y;
    tmp = tmp2.x;
    tmp2 = fast_div_modulo(tmp, nsamples_y);
    const int wt = tmp2.y;
    const int it = tmp2.x;


        const int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*J*stride_col_dst + it*I;
        dst += offset_dst;

        const int i_max = nrows_x   - it*I - 1;
        const int j_max = ncols_dst - jt*J - 1;
        if (fallback && i > i_max) {
            return;
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j > j_max) {
                return;
            }

            dst[j*stride_col_dst + i] += sum[j0/nwarps];
        }
        return;
    
}

static __global__ void test_mmq_pdl_clear(uint32_t * flags, uint32_t rows) {
    ggml_cuda_pdl_lc();
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync(); // Previous projection's cast must finish reading flags.
    if (i < rows) flags[i] = 0;
}

template<mmq_q8_1_ds_layout Layout>
static __global__ void test_mmq_pdl_pack_kernel(
        const half * input, void * packed,
        int64_t inputs, int64_t input_stride, int64_t padded_inputs,
        uint32_t rows, uint32_t guard_blocks, uint32_t * flags) {
    ggml_cuda_pdl_lc();
    // Before any input/flag read, packed overwrite, or guard-tail write.
    // Thus a body-level early return cannot bypass the scratch reuse barrier.
    ggml_cuda_pdl_sync();
    test_mmq_pdl_quantize_body<Layout,false,true>(input,nullptr,packed,
        inputs,input_stride,int64_t(rows)*input_stride,int64_t(rows)*input_stride,
        padded_inputs,rows,1,0,flags);
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        const uint64_t logical_bytes = uint64_t(rows)*padded_inputs*9/8;
        auto * guard = reinterpret_cast<uint32_t *>(static_cast<char *>(packed)+logical_bytes);
        const uint32_t words = guard_blocks*(144/sizeof(uint32_t));
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) guard[i] = 0;
    }
}

static __global__ void test_mmq_pdl_cast_kernel(
        const float * input, half * output,
        uint32_t rows, uint32_t columns, uint32_t stride,
        const uint32_t * row_flags,
        const uint32_t * weight_flag) {
    ggml_cuda_pdl_lc();
    const uint32_t index = blockIdx.x*blockDim.x + threadIdx.x;
    ggml_cuda_pdl_sync(); // Read raw output and both flags only after predecessor completion.
    if (index >= rows*columns) return;
    const float value = input[index];
    const half rounded = __float2half_rn(value);
    const bool bad = *weight_flag || row_flags[index/columns] ||
        !marker_finite(value) || !marker_half_finite(rounded);
    output[(index/columns)*stride+index%columns] = bad ? __ushort_as_half(0x7e00) : rounded;
}

static bool test_mmq_pdl_valid(const FerrumUpstreamLinearPlanV1 * p, TestMmqPlan & n) {
    return upstream_mmq_valid(p,n) && p->request.layout == 0 && n.rows <= 32;
}

struct TestMmqPdlKernels {
    const void * pack;
    const void * main;
    const void * fixup;
};

template<ggml_type Type, int J, bool Fallback>
static TestMmqPdlKernels test_mmq_pdl_kernels() {
    constexpr auto layout = Type == GGML_TYPE_IQ4_XS ? MMQ_Q8_1_DS_LAYOUT_D4 : MMQ_Q8_1_DS_LAYOUT_DS4;
    return {reinterpret_cast<const void *>(test_mmq_pdl_pack_kernel<layout>),
        reinterpret_cast<const void *>(test_mmq_pdl_main<Type,J,Fallback>),
        reinterpret_cast<const void *>(test_mmq_pdl_fixup<Type,J,Fallback>)};
}

template<ggml_type Type, bool Fallback>
static TestMmqPdlKernels test_mmq_pdl_kernels(const TestMmqPlan & n) {
    switch (n.j) {
        case 8: return test_mmq_pdl_kernels<Type,8,Fallback>();
        case 16: return test_mmq_pdl_kernels<Type,16,Fallback>();
        case 24: return test_mmq_pdl_kernels<Type,24,Fallback>();
        case 32: return test_mmq_pdl_kernels<Type,32,Fallback>();
        default: return {};
    }
}

template<ggml_type Type>
static TestMmqPdlKernels test_mmq_pdl_kernels(const TestMmqPlan & n) {
    return n.outputs%128 ? test_mmq_pdl_kernels<Type,true>(n) : test_mmq_pdl_kernels<Type,false>(n);
}

static TestMmqPdlKernels test_mmq_pdl_kernels(const TestMmqPlan & n) {
    switch (n.format) {
        case GGML_TYPE_IQ4_XS: return test_mmq_pdl_kernels<GGML_TYPE_IQ4_XS>(n);
        case GGML_TYPE_Q4_K: return test_mmq_pdl_kernels<GGML_TYPE_Q4_K>(n);
        case GGML_TYPE_Q5_K: return test_mmq_pdl_kernels<GGML_TYPE_Q5_K>(n);
        default: return {};
    }
}

#if defined(GGML_CUDA_USE_PDL)
template<typename Kernel, typename... Args>
static int test_mmq_launch_pdl(Kernel kernel, dim3 grid, dim3 block,
        uint32_t shared_bytes, cudaStream_t stream, Args... args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config,kernel,args...);
}

template<ggml_type Type, int J, bool Fallback>
static int test_mmq_pdl_dispatch(const TestMmqPlan & p, const void * weights,
        const void * packed, float * output, float * fixup, cudaStream_t stream) {
    // Exactly test_mmq_dispatch's scalar arguments and launch dimensions.
    const auto one = init_fastdiv_values(1);
    const auto blocks_per_k = init_fastdiv_values(p.inputs/256);
    const auto ntx = init_fastdiv_values((p.rows+J-1)/J);
    const int weight_stride = p.inputs/256;
    const int packed_channel_stride = p.rows*p.padded_inputs*sizeof(block_q8_1_mmq)/(128*sizeof(int));
    constexpr bool stream_k = ggml_cuda_mmq_get_config_ampere(Type,J,Fallback).stream_k;
    const dim3 block(32,p.nthreads/32,1);
    const dim3 grid = stream_k ? dim3(p.blocks,1,1) : dim3(p.tiles_y,ntx.z,1);
    auto status = test_mmq_launch_pdl(test_mmq_pdl_main<Type,J,Fallback>,grid,block,p.shared_bytes,stream,
        static_cast<const char *>(weights),static_cast<const int *>(packed),
        static_cast<const int32_t *>(nullptr),static_cast<const int32_t *>(nullptr),output,fixup,
        static_cast<const float *>(nullptr),blocks_per_k,p.outputs,p.rows,weight_stride,p.rows,p.outputs,
        one,one,p.outputs*weight_stride,packed_channel_stride,p.rows*p.outputs,
        one,one,p.outputs*weight_stride,packed_channel_stride,p.rows*p.outputs,ntx);
    if (status != cudaSuccess || !p.fixup) return status;
    return test_mmq_launch_pdl(test_mmq_pdl_fixup<Type,J,Fallback>,
        dim3(p.blocks,p.i/32,1),dim3(32,p.nthreads/64,1),0,stream,
        static_cast<const int32_t *>(nullptr),static_cast<const int32_t *>(nullptr),output,fixup,
        blocks_per_k,p.outputs,p.rows,p.outputs,one,p.rows*p.outputs,one,p.rows*p.outputs,ntx);
}

template<ggml_type Type, bool Fallback>
static int test_mmq_pdl_j(const TestMmqPlan & n, const void * w, const void * q,
        float * out, float * fixup, cudaStream_t stream) {
    switch (n.j) {
        case 8: return test_mmq_pdl_dispatch<Type,8,Fallback>(n,w,q,out,fixup,stream);
        case 16: return test_mmq_pdl_dispatch<Type,16,Fallback>(n,w,q,out,fixup,stream);
        case 24: return test_mmq_pdl_dispatch<Type,24,Fallback>(n,w,q,out,fixup,stream);
        case 32: return test_mmq_pdl_dispatch<Type,32,Fallback>(n,w,q,out,fixup,stream);
        default: return -2;
    }
}

template<ggml_type Type>
static int test_mmq_pdl_type(const TestMmqPlan & n, const void * w, const void * q,
        float * out, float * fixup, cudaStream_t stream) {
    return n.outputs%128 ? test_mmq_pdl_j<Type,true>(n,w,q,out,fixup,stream)
                        : test_mmq_pdl_j<Type,false>(n,w,q,out,fixup,stream);
}
#endif

// Cold only: revalidate the actual GPU, select every actual kernel specialization,
// set this main's dynamic shared memory, and seal support only if all loaded PTX
// can implement grid dependencies. No attribute query or setup occurs in hot ABI.
extern "C" int ferrum_test_upstream_mmq_pdl_support_v1(
        const FerrumUpstreamLinearPlanV1 * p, uint32_t * supported) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{};
        if (!supported || !test_mmq_pdl_valid(p,n)) return -1;
        *supported = 0;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
        auto status = ferrum_upstream_verify_device(p->request);
        if (status != cudaSuccess) return status;
        if (p->request.cc < GGML_CUDA_CC_HOPPER) return 0;
        const auto kernels = test_mmq_pdl_kernels(n);
        if (!kernels.pack || !kernels.main || !kernels.fixup) return -2;
        status = cudaFuncSetAttribute(kernels.main,cudaFuncAttributeMaxDynamicSharedMemorySize,n.shared_bytes);
        if (status != cudaSuccess) return status;
        const void * selected[] = {reinterpret_cast<const void *>(test_mmq_pdl_clear),kernels.pack,
            kernels.main,reinterpret_cast<const void *>(test_mmq_pdl_cast_kernel)};
        for (const auto kernel : selected) {
            cudaFuncAttributes attributes{};
            status = cudaFuncGetAttributes(&attributes,kernel);
            if (status != cudaSuccess) return status;
            if (attributes.ptxVersion < 90) return 0;
        }
        if (n.fixup) {
            cudaFuncAttributes attributes{};
            status = cudaFuncGetAttributes(&attributes,kernels.fixup);
            if (status != cudaSuccess) return status;
            if (attributes.ptxVersion < 90) return 0;
        }
        *supported = 1;
#endif
        return 0;
    });
}

extern "C" int ferrum_test_upstream_mmq_pdl_pack_f16_small_v2(
        const FerrumUpstreamLinearPlanV1 * p, const void * input, uint32_t stride,
        void * converted, void * packed, void * flags, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{};
        if (!test_mmq_pdl_valid(p,n)) return -1;
        // Preserve direct pack's full contract; converted is validated but untouched.
        if (!input || !converted || !packed || !flags || stride < n.inputs ||
                uint64_t(stride)*n.rows > INT_MAX || uintptr_t(input)%2 ||
                uintptr_t(converted)%16 || uintptr_t(packed)%16 || uintptr_t(flags)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (p->request.cc < GGML_CUDA_CC_HOPPER) return -2;
        const auto stream = static_cast<cudaStream_t>(s);
        auto status = test_mmq_launch_pdl(test_mmq_pdl_clear,
            dim3((uint64_t(n.rows)+127)/128,1,1),dim3(128,1,1),0,stream,
            static_cast<uint32_t *>(flags),n.rows);
        if (status != cudaSuccess) return status;
        const dim3 grid(n.rows,n.padded_inputs/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ),1);
        if (n.format == GGML_TYPE_IQ4_XS) {
            return test_mmq_launch_pdl(test_mmq_pdl_pack_kernel<MMQ_Q8_1_DS_LAYOUT_D4>,grid,
                dim3(CUDA_QUANTIZE_BLOCK_SIZE_MMQ,1,1),0,stream,
                static_cast<const half *>(input),packed,int64_t(n.inputs),int64_t(stride),
                int64_t(n.padded_inputs),n.rows,p->guard_blocks,static_cast<uint32_t *>(flags));
        }
        return test_mmq_launch_pdl(test_mmq_pdl_pack_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>,grid,
            dim3(CUDA_QUANTIZE_BLOCK_SIZE_MMQ,1,1),0,stream,
            static_cast<const half *>(input),packed,int64_t(n.inputs),int64_t(stride),
            int64_t(n.padded_inputs),n.rows,p->guard_blocks,static_cast<uint32_t *>(flags));
#else
        return -2;
#endif
    });
}

extern "C" int ferrum_test_upstream_mmq_pdl_dot_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights, const void * packed, void * output, void * fixup, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{};
        if (!test_mmq_pdl_valid(p,n)) return -1;
        if (!weights || !packed || !output || (n.fixup && !fixup) || uintptr_t(weights)%4 ||
                uintptr_t(packed)%16 || uintptr_t(output)%4 || uintptr_t(fixup)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (p->request.cc < GGML_CUDA_CC_HOPPER) return -2;
        switch (n.format) {
            case GGML_TYPE_IQ4_XS: return test_mmq_pdl_type<GGML_TYPE_IQ4_XS>(n,weights,packed,
                static_cast<float *>(output),static_cast<float *>(fixup),static_cast<cudaStream_t>(s));
            case GGML_TYPE_Q4_K: return test_mmq_pdl_type<GGML_TYPE_Q4_K>(n,weights,packed,
                static_cast<float *>(output),static_cast<float *>(fixup),static_cast<cudaStream_t>(s));
            case GGML_TYPE_Q5_K: return test_mmq_pdl_type<GGML_TYPE_Q5_K>(n,weights,packed,
                static_cast<float *>(output),static_cast<float *>(fixup),static_cast<cudaStream_t>(s));
            default: return -1;
        }
#else
        return -2;
#endif
    });
}

extern "C" int ferrum_test_upstream_mmq_pdl_cast_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * input, void * output, uint32_t stride,
        const void * rows, const void * weights, void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{};
        if (!test_mmq_pdl_valid(p,n)) return -1;
        if (!input || !output || !rows || !weights || stride < n.outputs || uint64_t(stride)*n.rows > INT_MAX ||
                uintptr_t(input)%4 || uintptr_t(output)%2 || uintptr_t(rows)%4 || uintptr_t(weights)%4) return -5;
#if defined(GGML_CUDA_USE_PDL)
        if (p->request.cc < GGML_CUDA_CC_HOPPER) return -2;
        return test_mmq_launch_pdl(test_mmq_pdl_cast_kernel,
            dim3((uint64_t(n.rows)*n.outputs+255)/256,1,1),dim3(256,1,1),0,static_cast<cudaStream_t>(s),
            static_cast<const float *>(input),static_cast<half *>(output),n.rows,n.outputs,stride,
            static_cast<const uint32_t *>(rows),static_cast<const uint32_t *>(weights));
#else
        return -2;
#endif
    });
}

// counts: all, programmatic, clear->pack, pack->main, main->fixup,
// fixup->cast, main->cast (no fixup), cast->next-clear. Match actual functions.
extern "C" int ferrum_test_upstream_mmq_pdl_graph_edges_v1(
        const FerrumUpstreamLinearPlanV1 * p, void * graph_ptr,
        uint64_t * counts, size_t counts_len) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmqPlan n{};
        if (!test_mmq_pdl_valid(p,n) || !graph_ptr || !counts || counts_len < 8) return -1;
        for (size_t i = 0; i < 8; ++i) counts[i] = 0;
#if defined(GGML_CUDA_USE_PDL) && CUDART_VERSION >= 12030
        const auto kernels = test_mmq_pdl_kernels(n);
        const void * clear = reinterpret_cast<const void *>(test_mmq_pdl_clear);
        const void * cast = reinterpret_cast<const void *>(test_mmq_pdl_cast_kernel);
        const void * from_kernels[] = {clear,kernels.pack,kernels.main,kernels.fixup,kernels.main,cast};
        const void * to_kernels[] = {kernels.pack,kernels.main,kernels.fixup,cast,cast,clear};
        const auto graph = static_cast<cudaGraph_t>(graph_ptr);
        size_t count = 0;
        auto status = cudaGraphGetEdges_v2(graph,nullptr,nullptr,nullptr,&count);
        if (status != cudaSuccess) return status;
        std::vector<cudaGraphNode_t> from(count), to(count);
        std::vector<cudaGraphEdgeData> data(count);
        status = cudaGraphGetEdges_v2(graph,from.data(),to.data(),data.data(),&count);
        if (status != cudaSuccess) return status;
        counts[0] = count;
        for (size_t i = 0; i < count; ++i) {
            if (data[i].type != cudaGraphDependencyTypeProgrammatic) continue;
            ++counts[1];
            if (data[i].from_port != cudaGraphKernelNodePortProgrammatic || data[i].to_port != 0) continue;
            bool reserved_zero = true;
            for (const auto byte : data[i].reserved) reserved_zero &= byte == 0;
            if (!reserved_zero) continue;
            cudaGraphNodeType a_type{}, b_type{};
            status = cudaGraphNodeGetType(from[i],&a_type); if (status != cudaSuccess) return status;
            status = cudaGraphNodeGetType(to[i],&b_type); if (status != cudaSuccess) return status;
            if (a_type != cudaGraphNodeTypeKernel || b_type != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeParams a{}, b{};
            status = cudaGraphKernelNodeGetParams(from[i],&a); if (status != cudaSuccess) return status;
            status = cudaGraphKernelNodeGetParams(to[i],&b); if (status != cudaSuccess) return status;
            for (size_t stage = 0; stage < 6; ++stage) {
                if (a.func == from_kernels[stage] && b.func == to_kernels[stage]) ++counts[2+stage];
            }
        }
        return 0;
#else
        return -2;
#endif
    });
}
