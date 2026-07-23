// Paged variable-length attention over vLLM paged_attention_v2 KV layout.
//
// This is the q_len>1 companion for Ferrum's vLLM-layout decode path:
// `split_qkv_norm_rope_into_paged_cache_vllm_*` writes K/V as
//   K: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
//   V: [num_blocks, num_kv_heads, head_dim, block_size]
// while the existing `paged_varlen_attention.cu` reads Ferrum's legacy
// [num_blocks, block_size, num_kv_heads, head_dim] pool.
//
// The kernel is intentionally simple and non split-K. It is a correctness
// bridge for prefill/chunk prefill when `FERRUM_USE_VLLM_PAGED_ATTN=1`; the
// hot decode path still uses vLLM's `paged_attention_v2` implementation.

#include "common.cuh"
#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_SIZE 32
#define VLLM_X 8  // 16 / sizeof(half)
#define VNEXT_VLLM_BLOCK_SIZE 16

__device__ __forceinline__ const __half* k_vllm_ptr(
    const __half* pool,
    int physical_block,
    int kv_head,
    int dim,
    int slot,
    int head_dim,
    int num_kv_heads,
    int block_size)
{
    const int x_chunk = dim / VLLM_X;
    const int x_off = dim - x_chunk * VLLM_X;
    const long long per_block = (long long)num_kv_heads * head_dim * block_size;
    const long long per_head = (long long)head_dim * block_size;
    const long long off =
        (long long)physical_block * per_block +
        (long long)kv_head * per_head +
        (long long)x_chunk * (block_size * VLLM_X) +
        (long long)slot * VLLM_X +
        x_off;
    return pool + off;
}

__device__ __forceinline__ const __half* v_vllm_ptr(
    const __half* pool,
    int physical_block,
    int kv_head,
    int dim,
    int slot,
    int head_dim,
    int num_kv_heads,
    int block_size)
{
    const long long per_block = (long long)num_kv_heads * head_dim * block_size;
    const long long per_head = (long long)head_dim * block_size;
    const long long off =
        (long long)physical_block * per_block +
        (long long)kv_head * per_head +
        (long long)dim * block_size +
        slot;
    return pool + off;
}

struct VllmPoolAccessor {
    const __half* k_block_pool;
    const __half* v_block_pool;
    const int* block_table;
    int head_dim;
    int num_kv_heads;
    int block_size;

    __device__ __forceinline__ bool contains(int logical_block) const
    {
        // The legacy ABI relies on the host-validated block-table length.
        return logical_block >= 0;
    }

    __device__ __forceinline__ float load_k(
        int logical_block,
        int kv_head,
        int dim,
        int slot) const
    {
        const int physical_block = block_table[logical_block];
        return __half2float(*k_vllm_ptr(
            k_block_pool, physical_block, kv_head, dim, slot, head_dim,
            num_kv_heads, block_size));
    }

    __device__ __forceinline__ float load_v(
        int logical_block,
        int kv_head,
        int dim,
        int slot) const
    {
        const int physical_block = block_table[logical_block];
        return __half2float(*v_vllm_ptr(
            v_block_pool, physical_block, kv_head, dim, slot, head_dim,
            num_kv_heads, block_size));
    }
};

struct VllmAddressAccessor {
    const uint64_t* block_addresses;
    int block_count;
    int head_dim;
    int num_kv_heads;

    __device__ __forceinline__ bool contains(int logical_block) const
    {
        return logical_block >= 0 && logical_block < block_count;
    }

    __device__ __forceinline__ const __half* k_block(int logical_block) const
    {
        if (!contains(logical_block)) return nullptr;
        return reinterpret_cast<const __half*>(
            static_cast<uintptr_t>(block_addresses[logical_block]));
    }

    __device__ __forceinline__ const __half* v_block(int logical_block) const
    {
        const __half* block = k_block(logical_block);
        if (block == nullptr) return nullptr;
        const long long block_elements =
            (long long)num_kv_heads * head_dim * VNEXT_VLLM_BLOCK_SIZE;
        return block + block_elements;
    }

    __device__ __forceinline__ float load_k(
        int logical_block,
        int kv_head,
        int dim,
        int slot) const
    {
        const __half* block = k_block(logical_block);
        return block == nullptr
            ? 0.0f
            : __half2float(*k_vllm_ptr(
                  block, 0, kv_head, dim, slot, head_dim, num_kv_heads,
                  VNEXT_VLLM_BLOCK_SIZE));
    }

    __device__ __forceinline__ float load_v(
        int logical_block,
        int kv_head,
        int dim,
        int slot) const
    {
        const __half* block = v_block(logical_block);
        return block == nullptr
            ? 0.0f
            : __half2float(*v_vllm_ptr(
                  block, 0, kv_head, dim, slot, head_dim, num_kv_heads,
                  VNEXT_VLLM_BLOCK_SIZE));
    }
};

template <typename KvAccessor>
__device__ __forceinline__ void paged_varlen_attn_vllm_f16_impl(
    const __half* __restrict__ q,
    const KvAccessor kv,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const float scale,
    const int token_global,
    const int valid_kv_len,
    float* s_scores,
    float* s_global_max,
    float* s_global_sum)
{
    const int q_head = blockIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    if (valid_kv_len <= 0) return;

    const __half* q_ptr =
        q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
    __half* out_ptr =
        output + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = (threadIdx.x % WARP_SIZE) + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim)
            ? __half2float(q_ptr[d]) : 0.0f;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        const int logical_block = kv_pos / block_size;
        const int slot = kv_pos % block_size;
        if (!kv.contains(logical_block)) continue;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) {
                dot += q_reg[i] * kv.load_k(logical_block, kv_head, d, slot);
            }
        }
        const float score = warp_reduce_sum(dot) * scale;
        if (lane_id == 0) s_scores[kv_pos] = score;
    }
    __syncthreads();

    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, s_scores[i]);
    }
    const float bmax = block_reduce_max(thread_max);
    if (threadIdx.x == 0) *s_global_max = bmax;
    __syncthreads();
    const float global_max = *s_global_max;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        const float val = expf(s_scores[i] - global_max);
        s_scores[i] = val;
        thread_sum += val;
    }
    __syncthreads();

    const float bsum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) *s_global_sum = bsum;
    __syncthreads();
    const float inv_sum = 1.0f / *s_global_sum;

    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            const int logical_block = kv_pos / block_size;
            const int slot = kv_pos % block_size;
            if (!kv.contains(logical_block)) continue;
            acc += s_scores[kv_pos] *
                   kv.load_v(logical_block, kv_head, d, slot);
        }
        out_ptr[d] = __float2half(acc);
    }
}

extern "C" __global__ void paged_varlen_attn_vllm_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    __half* __restrict__ output,
    const int num_seqs,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const float scale)
{
    const int token_global = blockIdx.y;
    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs &&
           cu_seqlens_q[seq_idx + 1] <= token_global) {
        seq_idx++;
    }
    const int seq_q_start = cu_seqlens_q[seq_idx];
    const int local_idx = token_global - seq_q_start;
    const int valid_kv_len = pos_offsets[seq_idx] + local_idx + 1;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const VllmPoolAccessor kv{
        k_block_pool,
        v_block_pool,
        my_block_table,
        head_dim,
        num_kv_heads,
        block_size,
    };

    extern __shared__ float s_scores[];
    __shared__ float s_global_max;
    __shared__ float s_global_sum;
    paged_varlen_attn_vllm_f16_impl(
        q, kv, output, num_q_heads, num_kv_heads, head_dim, block_size, scale,
        token_global, valid_kv_len, s_scores, &s_global_max, &s_global_sum);
}

extern "C" __global__ void vnext_paged_varlen_attn_vllm_addressed_f16(
    const __half* __restrict__ q,
    const int* __restrict__ control,
    const uint64_t* __restrict__ block_addresses,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale)
{
    const int block_count = control[0];
    const int position_start = control[1];
    const int active_tokens = control[2];
    const int final_seq_len = control[3];
    const int token = blockIdx.y;
    const long long active_end =
        (long long)position_start + (long long)active_tokens;
    const long long required_blocks =
        final_seq_len > 0
            ? ((long long)final_seq_len + VNEXT_VLLM_BLOCK_SIZE - 1) /
                  VNEXT_VLLM_BLOCK_SIZE
            : 0;
    if (block_count <= 0 || position_start < 0 || active_tokens <= 0 ||
        final_seq_len <= 0 || active_end > final_seq_len ||
        required_blocks > block_count || token >= active_tokens ||
        num_q_heads <= 0 || num_kv_heads <= 0 ||
        num_q_heads % num_kv_heads != 0 || head_dim <= 0 ||
        head_dim > 8 * WARP_SIZE || head_dim % VLLM_X != 0) {
        return;
    }

    const int valid_kv_len = position_start + token + 1;
    const int last_logical_block =
        (valid_kv_len - 1) / VNEXT_VLLM_BLOCK_SIZE;
    if (last_logical_block < 0 || last_logical_block >= block_count) return;
    const VllmAddressAccessor kv{
        block_addresses,
        block_count,
        head_dim,
        num_kv_heads,
    };

    extern __shared__ float s_scores[];
    __shared__ float s_global_max;
    __shared__ float s_global_sum;
    paged_varlen_attn_vllm_f16_impl(
        q, kv, output, num_q_heads, num_kv_heads, head_dim,
        VNEXT_VLLM_BLOCK_SIZE, scale, token, valid_kv_len, s_scores,
        &s_global_max, &s_global_sum);
}

template <typename KvAccessor>
__device__ __forceinline__ void paged_varlen_attn_vllm_tiled_q4_f16_impl(
    const __half* __restrict__ q,
    const KvAccessor kv,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int score_stride,
    const float scale,
    const int seq_q_start,
    const int tile_start,
    const int actual_tile,
    const int pos0,
    float* s_scores,
    float* s_global_max,
    float* s_global_sum)
{
    constexpr int TILE_Q = 4;
    const int q_head = blockIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int max_valid = pos0 + actual_tile;

    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[TILE_Q][8];

#pragma unroll
    for (int qi = 0; qi < TILE_Q; ++qi) {
        const int token_global = seq_q_start + tile_start + qi;
        const __half* q_ptr =
            q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane_id + i * WARP_SIZE;
            q_reg[qi][i] = (qi < actual_tile && i < elems_per_thread && d < head_dim)
                               ? __half2float(q_ptr[d])
                               : 0.0f;
        }
    }

    for (int kv_pos = warp_id; kv_pos < max_valid; kv_pos += num_warps) {
        const int logical_block = kv_pos / block_size;
        const int slot = kv_pos % block_size;
        if (!kv.contains(logical_block)) continue;
        float k_reg[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = lane_id + i * WARP_SIZE;
            k_reg[i] = (i < elems_per_thread && d < head_dim)
                           ? kv.load_k(logical_block, kv_head, d, slot)
                           : 0.0f;
        }

#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) {
            const int valid = pos0 + qi + 1;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot += q_reg[qi][i] * k_reg[i];
            }
            const float score = warp_reduce_sum(dot) * scale;
            if (lane_id == 0 && qi < actual_tile && kv_pos < valid) {
                s_scores[qi * score_stride + kv_pos] = score;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int qi = 0; qi < TILE_Q; ++qi) {
        if (qi >= actual_tile) continue;
        const int valid = pos0 + qi + 1;
        float thread_max = -1e20f;
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            thread_max = fmaxf(thread_max, s_scores[qi * score_stride + i]);
        }
        const float bmax = block_reduce_max(thread_max);
        if (threadIdx.x == 0) s_global_max[qi] = bmax;
        __syncthreads();

        float thread_sum = 0.0f;
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            const float val = expf(s_scores[qi * score_stride + i] - s_global_max[qi]);
            s_scores[qi * score_stride + i] = val;
            thread_sum += val;
        }
        __syncthreads();
        const float bsum = block_reduce_sum(thread_sum);
        if (threadIdx.x == 0) s_global_sum[qi] = bsum;
        __syncthreads();

        const float inv_sum = 1.0f / s_global_sum[qi];
        for (int i = threadIdx.x; i < valid; i += blockDim.x) {
            s_scores[qi * score_stride + i] *= inv_sum;
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[TILE_Q];
#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) acc[qi] = 0.0f;

        for (int kv_pos = 0; kv_pos < max_valid; ++kv_pos) {
            const int logical_block = kv_pos / block_size;
            const int slot = kv_pos % block_size;
            if (!kv.contains(logical_block)) continue;
            const float v = kv.load_v(logical_block, kv_head, d, slot);
#pragma unroll
            for (int qi = 0; qi < TILE_Q; ++qi) {
                const int valid = pos0 + qi + 1;
                if (qi < actual_tile && kv_pos < valid) {
                    acc[qi] += s_scores[qi * score_stride + kv_pos] * v;
                }
            }
        }

#pragma unroll
        for (int qi = 0; qi < TILE_Q; ++qi) {
            if (qi < actual_tile) {
                const int token_global = seq_q_start + tile_start + qi;
                __half* out_ptr =
                    output + ((size_t)token_global * num_q_heads + q_head) * head_dim;
                out_ptr[d] = __float2half(acc[qi]);
            }
        }
    }
}

extern "C" __global__ void paged_varlen_attn_vllm_tiled_q4_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_block_pool,
    const __half* __restrict__ v_block_pool,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    const int* __restrict__ tile_seqs,
    const int* __restrict__ tile_starts,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_blocks_per_seq,
    const int block_size,
    const int score_stride,
    const float scale)
{
    constexpr int TILE_Q = 4;
    const int tile_id = blockIdx.y;
    const int seq_idx = tile_seqs[tile_id];
    const int tile_start = tile_starts[tile_id];
    const int seq_q_start = cu_seqlens_q[seq_idx];
    const int q_len = cu_seqlens_q[seq_idx + 1] - seq_q_start;
    const int actual_tile = min(TILE_Q, q_len - tile_start);
    const int pos0 = pos_offsets[seq_idx] + tile_start;
    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const VllmPoolAccessor kv{
        k_block_pool,
        v_block_pool,
        my_block_table,
        head_dim,
        num_kv_heads,
        block_size,
    };

    extern __shared__ float s_scores[];
    __shared__ float s_global_max[TILE_Q];
    __shared__ float s_global_sum[TILE_Q];
    paged_varlen_attn_vllm_tiled_q4_f16_impl(
        q, kv, output, num_q_heads, num_kv_heads, head_dim, block_size,
        score_stride, scale, seq_q_start, tile_start, actual_tile, pos0,
        s_scores, s_global_max, s_global_sum);
}

extern "C" __global__ void vnext_paged_varlen_attn_vllm_tiled_q4_addressed_f16(
    const __half* __restrict__ q,
    const int* __restrict__ control,
    const uint64_t* __restrict__ block_addresses,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int score_stride,
    const float scale)
{
    constexpr int TILE_Q = 4;
    const int block_count = control[0];
    const int position_start = control[1];
    const int active_tokens = control[2];
    const int final_seq_len = control[3];
    const int tile_start = blockIdx.y * TILE_Q;
    const int actual_tile = min(TILE_Q, active_tokens - tile_start);
    const long long active_end =
        (long long)position_start + (long long)active_tokens;
    const long long max_valid_wide =
        (long long)position_start + tile_start + actual_tile;
    const long long required_blocks =
        final_seq_len > 0
            ? ((long long)final_seq_len + VNEXT_VLLM_BLOCK_SIZE - 1) /
                  VNEXT_VLLM_BLOCK_SIZE
            : 0;
    if (block_count <= 0 || position_start < 0 || active_tokens <= 0 ||
        final_seq_len <= 0 || active_end > final_seq_len ||
        required_blocks > block_count || tile_start >= active_tokens ||
        actual_tile <= 0 || max_valid_wide > 2147483647LL ||
        score_stride < max_valid_wide ||
        num_q_heads <= 0 || num_kv_heads <= 0 ||
        num_q_heads % num_kv_heads != 0 || head_dim <= 0 ||
        head_dim > 8 * WARP_SIZE || head_dim % VLLM_X != 0) {
        return;
    }

    const int max_valid = (int)max_valid_wide;
    const int last_logical_block =
        (max_valid - 1) / VNEXT_VLLM_BLOCK_SIZE;
    if (last_logical_block < 0 || last_logical_block >= block_count) return;
    const VllmAddressAccessor kv{
        block_addresses,
        block_count,
        head_dim,
        num_kv_heads,
    };

    extern __shared__ float s_scores[];
    __shared__ float s_global_max[TILE_Q];
    __shared__ float s_global_sum[TILE_Q];
    paged_varlen_attn_vllm_tiled_q4_f16_impl(
        q, kv, output, num_q_heads, num_kv_heads, head_dim,
        VNEXT_VLLM_BLOCK_SIZE, score_stride, scale, 0, tile_start,
        actual_tile, position_start + tile_start, s_scores, s_global_max,
        s_global_sum);
}
