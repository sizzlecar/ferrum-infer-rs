// Batched single-query decode attention with GQA.
// Same warp-cooperative algorithm as decode_attention.cu, but processes
// multiple batch items in a single kernel launch.
//
// Grid:  (num_q_heads, batch_size)
// Block: (256,)
//
// Each block handles one (q_head, batch_item) pair.
// K/V caches are passed as device pointer arrays (one per batch item).

#include "common.cuh"

#define WARP_SIZE 32

// q_all:         [B, num_q_heads, head_dim] — all batch items' Q vectors
// k_cache_ptrs:  [B] — device pointer array, each -> [max_kv_len, num_kv_heads, head_dim]
// v_cache_ptrs:  [B] — device pointer array, same layout
// output:        [B, num_q_heads, head_dim]
// valid_kv_lens: [B] — per-item valid KV length
extern "C" __global__ void batched_decode_attention_f16(
    const __half* __restrict__ q_all,
    const __half* const* __restrict__ k_cache_ptrs,
    const __half* const* __restrict__ v_cache_ptrs,
    __half* __restrict__ output,
    const int* __restrict__ valid_kv_lens,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int num_kv_groups = num_q_heads / num_kv_heads;
    const int kv_head = q_head / num_kv_groups;

    const int valid_kv_len = valid_kv_lens[batch_idx];

    // Q pointer for this (batch, head)
    const int q_dim = num_q_heads * head_dim;
    const __half* q_ptr = q_all + batch_idx * q_dim + q_head * head_dim;
    __half* out_ptr = output + batch_idx * q_dim + q_head * head_dim;

    // K/V cache for this batch item
    const __half* k_cache = k_cache_ptrs[batch_idx];
    const __half* v_cache = v_cache_ptrs[batch_idx];

    // K/V cache layout: [seq_len, num_kv_heads, head_dim]
    const int kv_stride = num_kv_heads * head_dim;

    // Shared memory for attention scores
    extern __shared__ float s_scores[];

    // ====== Step 1: Q·K^T with warp-cooperative dot product ======
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[4]; // max 128/32 = 4
    for (int i = 0; i < elems_per_thread; i++) {
        int d = threadIdx.x % WARP_SIZE + i * WARP_SIZE;
        q_reg[i] = (d < head_dim) ? __half2float(q_ptr[d]) : 0.0f;
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    float local_max = -1e20f;
    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += num_warps) {
        const __half* k_row = k_cache + kv_pos * kv_stride + kv_head * head_dim;
        float partial = 0.0f;
        for (int i = 0; i < elems_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (d < head_dim)
                partial += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum(partial) * scale;
        if (lane_id == 0) {
            s_scores[kv_pos] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    __syncthreads();

    // ====== Step 2: Softmax ======
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, s_scores[i]);

    __shared__ float s_global_max;
    float bmax = block_reduce_max(thread_max);
    if (threadIdx.x == 0) s_global_max = bmax;
    __syncthreads();
    float global_max = s_global_max;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x) {
        float val = expf(s_scores[i] - global_max);
        s_scores[i] = val;
        thread_sum += val;
    }
    __syncthreads();

    __shared__ float s_global_sum;
    float bsum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) s_global_sum = bsum;
    __syncthreads();
    float inv_sum = 1.0f / s_global_sum;

    for (int i = threadIdx.x; i < valid_kv_len; i += blockDim.x)
        s_scores[i] *= inv_sum;
    __syncthreads();

    // ====== Step 3: Weighted sum of V ======
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            acc += s_scores[kv_pos] * __half2float(
                v_cache[kv_pos * kv_stride + kv_head * head_dim + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}
