// Artifact adapter of llama.cpp d81235049384534c167caea52b85a694f6103d14.
// The MMVQ headers remain in the separately pinned external source directory.
// Copied quantizer/launch policy: upstream ggml/src/ggml-cuda/{quantize.cu,mmvq.cu}.
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

// Exact pinned device/table bodies; no GGML backend allocation or product routing.
#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int test_output_rows = 0>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int original_rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    static_assert(test_output_rows == 0 || (test_output_rows == 1 && ncols_dst == 8 && !has_fusion && !small_k && !halve_iters));
    constexpr int rows_per_cuda_block = test_output_rows ? test_output_rows : original_rows_per_cuda_block;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            // A shuffle reduction may have a lane-specific floating-point tree.
            // Keep the original row2 writer lane, including odd output rows.
            const int writer_lane = test_output_rows ? row0 % original_rows_per_cuda_block : i;
            if (threadIdx.x == writer_lane && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}


// Exact ordinary row-major Q8_1 pack; differs from MMQ D4/DS4.
template<bool Marker = false>
static __global__ void quantize_q8_1(
        const float * x_ptr, void * vy_ptr,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2, uint32_t * row_flags=nullptr) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    void        * GGML_CUDA_RESTRICT vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    uint32_t invalid=0;
    if constexpr(Marker) {
        invalid=!marker_finite(xi);
        for(int offset=16;offset>0;offset>>=1) invalid|=__shfl_xor_sync(0xffffffff,invalid,offset,32);
    }
    float d = amax / 127.0f;
    if constexpr(Marker) invalid|=!marker_half_finite(__float2half_rn(d));
    const int8_t q = (amax==0.0f || (Marker && invalid)) ? 0 : roundf(xi / d);
    if constexpr(Marker) {
        if(amax==0.0f && !invalid) {d=0.0f;sum=0.0f;}
        if(invalid) {d=marker_nan();sum=marker_nan();atomicOr(row_flags+i1,1u);}
    }

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = Marker && invalid
        ? __halves2half2(__ushort_as_half(0x7e00),__ushort_as_half(0x7e00))
        : make_half2(d, sum);
}

// Private ABI1. layout0=Columns(M columns, one channel), layout1=Channels
// (one column, M channels sharing W). Both compute the same dense logical GEMM.
struct TestMmvqPlan {
    uint32_t abi, format, rows, inputs, outputs, padded_inputs, layout, cc;
    uint32_t ncols, channels, nwarps, rows_per_block, small_k, padded_outputs;
    uint64_t weight_bytes, converted_bytes, packed_bytes, output_bytes;
};

template<ggml_type T>
static int test_mmvq_plan_type(uint32_t rows, uint32_t k, uint32_t n,
        uint32_t layout, uint32_t cc, TestMmvqPlan * out) {
    const int columns = layout ? 1 : rows;
    const auto table = get_device_table_id(cc);
    const int baseline_warps = calc_nwarps(T, columns, table);
    constexpr int blocks_per_warp = get_vdr_mmvq(T)*32/ggml_cuda_type_traits<T>::qi;
    const bool small = columns == 1 && baseline_warps > 1 &&
        k/256 < uint32_t(baseline_warps*blocks_per_warp);
    const int warps = calc_nwarps(T, columns, table, small, false);
    const int rpb = calc_rows_per_block(columns, table, small, warps);
    const uint64_t pk = GGML_PAD(uint64_t(k), MATRIX_ROW_PADDING);
    const uint64_t pn = GGML_PAD(uint64_t(n), uint64_t(rpb));
    if (pk > INT_MAX || pn > INT_MAX || uint64_t(rows)*pk > INT_MAX ||
            uint64_t(rows)*n > INT_MAX || pn*(k/256) > INT_MAX) return -3;
    *out = {1,T,rows,k,n,uint32_t(pk),layout,cc,uint32_t(columns),layout?rows:1u,
        uint32_t(warps),uint32_t(rpb),uint32_t(small),uint32_t(pn),
        pn*(k/256)*ggml_cuda_type_traits<T>::bs, uint64_t(rows)*pk*4,
        uint64_t(rows)*(pk/32)*sizeof(block_q8_1),uint64_t(rows)*n*4};
    return 0;
}

static int ferrum_test_mmvq_plan(uint32_t format, uint32_t rows, uint32_t k,
        uint32_t n, uint32_t layout, uint32_t cc, TestMmvqPlan * out) {
    // Architecture family validation, not a host/machine identity gate. This
    // minimal adapter preserves the original generic NVIDIA table (SM80+,
    // excluding GB10's independent halve-iters policy).
    if (!out || !rows || !k || k%256 || !n || layout>1 || cc<800 ||
            get_device_table_id(cc)!=MMVQ_PARAMETERS_GENERIC ||
            (layout==0 && rows!=1 && rows!=4 && rows!=8) ||
            (layout==1 && rows>32)) return -1;
    switch(format) {
        case GGML_TYPE_IQ4_XS: return test_mmvq_plan_type<GGML_TYPE_IQ4_XS>(rows,k,n,layout,cc,out);
        case GGML_TYPE_Q4_K: return test_mmvq_plan_type<GGML_TYPE_Q4_K>(rows,k,n,layout,cc,out);
        case GGML_TYPE_Q5_K: return test_mmvq_plan_type<GGML_TYPE_Q5_K>(rows,k,n,layout,cc,out);
        default: return -1;
    }
}

static bool test_mmvq_valid(const TestMmvqPlan * p) {
    if (!p || p->abi!=1) return false;
    TestMmvqPlan expected{};
    if (ferrum_test_mmvq_plan(p->format,p->rows,p->inputs,p->outputs,p->layout,p->cc,&expected)) return false;
    return p->padded_inputs==expected.padded_inputs && p->ncols==expected.ncols &&
        p->channels==expected.channels && p->nwarps==expected.nwarps &&
        p->rows_per_block==expected.rows_per_block && p->small_k==expected.small_k &&
        p->padded_outputs==expected.padded_outputs && p->weight_bytes==expected.weight_bytes &&
        p->converted_bytes==expected.converted_bytes && p->packed_bytes==expected.packed_bytes &&
        p->output_bytes==expected.output_bytes;
}

static __global__ void test_mmvq_convert(const half * input, float * converted,
        int rows, int inputs, int stride, int padded) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<rows*padded) converted[i]=i%padded<inputs ? __half2float(input[(i/padded)*stride+i%padded]) : 0.0f;
}
static __global__ void test_mmvq_cast_kernel(const float * input, half * output,
        int rows, int outputs, int stride) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<rows*outputs) output[(i/outputs)*stride+i%outputs]=__float2half_rn(input[i]);
}

static int ferrum_test_mmvq_pack(const TestMmvqPlan * p, const half * input,
        uint32_t stride, float * converted, void * packed, cudaStream_t stream) {
    if (!test_mmvq_valid(p) || !input || !converted || !packed || stride<p->inputs ||
        uint64_t(stride)*p->rows>INT_MAX || uintptr_t(input)%2 || uintptr_t(converted)%16 || uintptr_t(packed)%4) return -5;
    test_mmvq_convert<<<(p->rows*p->padded_inputs+255)/256,256,0,stream>>>(input,converted,p->rows,p->inputs,stride,p->padded_inputs);
    auto error=cudaGetLastError(); if(error!=cudaSuccess) return error;
    const dim3 grid((p->padded_inputs+CUDA_QUANTIZE_BLOCK_SIZE-1)/CUDA_QUANTIZE_BLOCK_SIZE,p->rows,1);
    quantize_q8_1<false><<<grid,dim3(CUDA_QUANTIZE_BLOCK_SIZE,1,1),0,stream>>>(converted,packed,int64_t(p->inputs),
        int64_t(p->padded_inputs),int64_t(p->rows)*p->padded_inputs,
        int64_t(p->rows)*p->padded_inputs,int64_t(p->padded_inputs),p->rows,init_fastdiv_values(1));
    return cudaGetLastError();
}

template<ggml_type T,int C,bool Small,int TestOutputRows = 0>
static int test_mmvq_launch(const TestMmvqPlan & p,const void * w,const void * q,float * o,cudaStream_t stream) {
    const dim3 grid(TestOutputRows ? p.outputs : p.padded_outputs/p.rows_per_block,p.channels,1), block(32,p.nwarps,1);
    const auto one=init_fastdiv_values(1);
    const uint32_t qstride=p.padded_inputs/32;
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q<T,C,false,Small,false,TestOutputRows><<<grid,block,0,stream>>>(w,q,
        static_cast<const int32_t *>(nullptr),fusion,o,p.inputs,make_uint3(0,0,0),
        p.inputs/256,qstride,p.outputs,init_fastdiv_values(p.channels),
        p.padded_outputs*(p.inputs/256),p.layout?qstride:p.rows*qstride,
        p.layout?p.outputs:p.rows*p.outputs,one,
        p.padded_outputs*(p.inputs/256),p.rows*qstride,p.rows*p.outputs,uint32_t(0));
    return cudaGetLastError();
}
template<ggml_type T>
static int test_mmvq_dispatch(const TestMmvqPlan & p,const void * w,const void * q,float * o,cudaStream_t s) {
    if(p.ncols==1) return p.small_k ? test_mmvq_launch<T,1,true>(p,w,q,o,s) : test_mmvq_launch<T,1,false>(p,w,q,o,s);
    if(p.ncols==4) return test_mmvq_launch<T,4,false>(p,w,q,o,s);
    if(p.ncols==8) return test_mmvq_launch<T,8,false>(p,w,q,o,s);
    return -1;
}
static int ferrum_test_mmvq_dot(const TestMmvqPlan * p,const void * w,const void * q,float * o,cudaStream_t stream) {
    if(!test_mmvq_valid(p) || !w || !q || !o || uintptr_t(w)%4 || uintptr_t(q)%4 || uintptr_t(o)%4) return -5;
    switch(p->format) {
        case GGML_TYPE_IQ4_XS: return test_mmvq_dispatch<GGML_TYPE_IQ4_XS>(*p,w,q,o,stream);
        case GGML_TYPE_Q4_K: return test_mmvq_dispatch<GGML_TYPE_Q4_K>(*p,w,q,o,stream);
        case GGML_TYPE_Q5_K: return test_mmvq_dispatch<GGML_TYPE_Q5_K>(*p,w,q,o,stream);
        default:return -1;
    }
}
static int ferrum_test_mmvq_cast(const TestMmvqPlan * p,const float * input,
        half * output,uint32_t stride,cudaStream_t stream) {
    if(!test_mmvq_valid(p)||!input||!output||stride<p->outputs||uint64_t(stride)*p->rows>INT_MAX||uintptr_t(input)%4||uintptr_t(output)%2) return -5;
    test_mmvq_cast_kernel<<<(p->rows*p->outputs+255)/256,256,0,stream>>>(input,output,p->rows,p->outputs,stride);
    return cudaGetLastError();
}

static int upstream_mmvq_plan(const FerrumUpstreamLinearRequestV1 * r,
        FerrumUpstreamLinearPlanV1 & out, TestMmvqPlan & p) {
    if (!ferrum_upstream_request(r)) return -1;
    const int status=ferrum_test_mmvq_plan(r->format,r->rows,r->inputs,r->outputs,r->layout,r->cc,&p);
    if (status) return status;
    out={}; out.request=*r; out.abi=1; out.size=sizeof(out); out.algorithm=2; out.pack_abi=3;
    out.padded_inputs=p.padded_inputs; out.padded_outputs=p.padded_outputs;
    out.ncols=p.ncols; out.channels=p.channels; out.nwarps=p.nwarps;
    out.rows_per_block=p.rows_per_block; out.small_k=p.small_k;
    out.weight_bytes=p.weight_bytes; out.converted_bytes=p.converted_bytes;
    out.packed_bytes=p.packed_bytes; out.output_bytes=p.output_bytes;
    return 0;
}
static bool upstream_mmvq_valid(const FerrumUpstreamLinearPlanV1 * p, TestMmvqPlan & native) {
    FerrumUpstreamLinearPlanV1 expected{};
    return p && !upstream_mmvq_plan(&p->request,expected,native) && ferrum_upstream_same(*p,expected);
}
extern "C" int ferrum_upstream_mmvq_plan_v1(const FerrumUpstreamLinearRequestV1 * r,
        FerrumUpstreamLinearPlanV1 * out) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        if (!out) return -1;
        FerrumUpstreamLinearPlanV1 candidate{}; TestMmvqPlan native{};
        int status=upstream_mmvq_plan(r,candidate,native);
        if (status) return status;
        status=ferrum_upstream_verify_device(*r);
        if (!status) *out=candidate;
        return status;
    });
}
extern "C" int ferrum_upstream_mmvq_pack_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,uint32_t stride,void * converted,void * packed,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan native{}; if (!upstream_mmvq_valid(p,native)) return -1;
        return ferrum_test_mmvq_pack(&native,static_cast<const half *>(input),stride,
            static_cast<float *>(converted),packed,static_cast<cudaStream_t>(stream));
    });
}
extern "C" int ferrum_upstream_mmvq_dot_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights,const void * packed,void * output,void * fixup,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan native{}; if (!upstream_mmvq_valid(p,native) || fixup) return -1;
        return ferrum_test_mmvq_dot(&native,weights,packed,static_cast<float *>(output),
            static_cast<cudaStream_t>(stream));
    });
}
// Experimental test ABI only: no provider registration or production caller.
// Borrow an unchanged, exact-validated baseline plan for logical extents and
// scratch. The original ABI's immutable launch geometry is never rewritten.
extern "C" int ferrum_test_mmvq_m8_row1_dot_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights,const void * packed,void * output,void * fixup,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{};
        if (!upstream_mmvq_valid(p,n) || fixup) return -1;
        if (n.layout != 0 || n.rows != 8 || n.ncols != 8 || n.channels != 1 ||
                n.nwarps != 2 || n.rows_per_block != 2 || n.small_k ||
                (n.format != GGML_TYPE_Q4_K && n.format != GGML_TYPE_Q5_K)) return -2;
        if (!weights || !packed || !output || uintptr_t(weights)%4 ||
                uintptr_t(packed)%4 || uintptr_t(output)%4) return -5;
        const auto s = static_cast<cudaStream_t>(stream);
        if (n.format == GGML_TYPE_Q4_K)
            return test_mmvq_launch<GGML_TYPE_Q4_K,8,false,1>(n,weights,packed,static_cast<float *>(output),s);
        return test_mmvq_launch<GGML_TYPE_Q5_K,8,false,1>(n,weights,packed,static_cast<float *>(output),s);
    });
}

extern "C" int ferrum_upstream_mmvq_cast_v1(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,void * output,uint32_t stride,void * stream) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan native{}; if (!upstream_mmvq_valid(p,native)) return -1;
        return ferrum_test_mmvq_cast(&native,static_cast<const float *>(input),
            static_cast<half *>(output),stride,static_cast<cudaStream_t>(stream));
    });
}

extern "C" int ferrum_upstream_mmvq_pack_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,uint32_t stride,void * converted,void * packed,void * flags,void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{}; if(!upstream_mmvq_valid(p,n)) return -1;
        if(!input||!converted||!packed||!flags||stride<n.inputs||uint64_t(stride)*n.rows>INT_MAX||
            uintptr_t(input)%2||uintptr_t(converted)%16||uintptr_t(packed)%4||uintptr_t(flags)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(flags,0,n.rows*sizeof(uint32_t),stream);
        if(status!=cudaSuccess) return status;
        test_mmvq_convert<<<(n.rows*n.padded_inputs+255)/256,256,0,stream>>>(
            static_cast<const half *>(input),static_cast<float *>(converted),n.rows,n.inputs,stride,n.padded_inputs);
        status=cudaGetLastError(); if(status!=cudaSuccess) return status;
        const dim3 grid((n.padded_inputs+CUDA_QUANTIZE_BLOCK_SIZE-1)/CUDA_QUANTIZE_BLOCK_SIZE,n.rows,1);
        quantize_q8_1<true><<<grid,dim3(CUDA_QUANTIZE_BLOCK_SIZE,1,1),0,stream>>>(
            static_cast<const float *>(converted),packed,int64_t(n.inputs),int64_t(n.padded_inputs),
            int64_t(n.rows)*n.padded_inputs,int64_t(n.rows)*n.padded_inputs,int64_t(n.padded_inputs),
            n.rows,init_fastdiv_values(1),static_cast<uint32_t *>(flags));
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_mmvq_check_weights_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * weights,void * flag,void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{}; if(!upstream_mmvq_valid(p,n)) return -1;
        if(!weights||!flag||uintptr_t(weights)%4||uintptr_t(flag)%4) return -5;
        const auto stream=static_cast<cudaStream_t>(s);
        auto status=cudaMemsetAsync(flag,0,4,stream); if(status!=cudaSuccess) return status;
        const uint64_t blocks=uint64_t(n.padded_outputs)*(n.inputs/256);
        marker_check_weights<<<(blocks+255)/256,256,0,stream>>>(static_cast<const unsigned char *>(weights),blocks,n.format,false,static_cast<uint32_t *>(flag));
        return cudaGetLastError();
    });
}
extern "C" int ferrum_upstream_mmvq_cast_v2(const FerrumUpstreamLinearPlanV1 * p,
        const void * input,void * output,uint32_t stride,const void * rows,const void * weights,void * s) noexcept {
    return ferrum_upstream_boundary([&]() -> int {
        TestMmvqPlan n{}; if(!upstream_mmvq_valid(p,n)) return -1;
        if(!input||!output||!rows||!weights||stride<n.outputs||uint64_t(stride)*n.rows>INT_MAX||
            uintptr_t(input)%4||uintptr_t(output)%2||uintptr_t(rows)%4||uintptr_t(weights)%4) return -5;
        marker_cast<<<(n.rows*n.outputs+255)/256,256,0,static_cast<cudaStream_t>(s)>>>(
            static_cast<const float *>(input),static_cast<half *>(output),n.rows,n.outputs,stride,
            static_cast<const uint32_t *>(rows),static_cast<const uint32_t *>(weights));
        return cudaGetLastError();
    });
}
