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

#define WARP_SIZE 32
#define VLLM_X 8  // 16 / sizeof(half)

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
    const int q_head = blockIdx.x;
    const int token_global = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);

    int seq_idx = 0;
    while (seq_idx + 1 < num_seqs &&
           cu_seqlens_q[seq_idx + 1] <= token_global) {
        seq_idx++;
    }
    const int local_idx = token_global - cu_seqlens_q[seq_idx];
    const int abs_kv_pos = pos_offsets[seq_idx] + local_idx;
    const int valid_kv_len = abs_kv_pos + 1;
    if (valid_kv_len <= 0) return;

    const int* my_block_table = block_tables + seq_idx * max_blocks_per_seq;
    const __half* q_ptr =
        q + ((size_t)token_global * num_q_heads + q_head) * head_dim;
    __half* out_ptr =
        output + ((size_t)token_global * num_q_heads + q_head) * head_dim;

    extern __shared__ float s_scores[];

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
        const int physical_block = my_block_table[logical_block];
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim) {
                dot += q_reg[i] *
                       __half2float(*k_vllm_ptr(k_block_pool, physical_block,
                                                kv_head, d, slot, head_dim,
                                                num_kv_heads, block_size));
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
    __shared__ float s_global_max;
    const float bmax = block_reduce_max(thread_max);
    if (threadIdx.x == 0) s_global_max = bmax;
    __syncthreads();
    const float global_max = s_global_max;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        const float val = expf(s_scores[i] - global_max);
        s_scores[i] = val;
        thread_sum += val;
    }
    __syncthreads();

    __shared__ float s_global_sum;
    const float bsum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) s_global_sum = bsum;
    __syncthreads();
    const float inv_sum = 1.0f / s_global_sum;

    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            const int logical_block = kv_pos / block_size;
            const int slot = kv_pos % block_size;
            const int physical_block = my_block_table[logical_block];
            acc += s_scores[kv_pos] *
                   __half2float(*v_vllm_ptr(v_block_pool, physical_block,
                                            kv_head, d, slot, head_dim,
                                            num_kv_heads, block_size));
        }
        out_ptr[d] = __float2half(acc);
    }
}
