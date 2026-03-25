// Single-query decode attention kernel with GQA (Grouped Query Attention).
//
// For decode phase: query has seq_len=1, K/V cache has seq_len=kv_len.
// Each block handles one query head:
//   1. Compute Q·K^T scores for all kv positions
//   2. Apply causal mask + softmax
//   3. Compute scores·V to produce output
//
// GQA: multiple query heads share the same KV head.
// kv_head_idx = q_head_idx / num_kv_groups
//
// This is more efficient than FlashAttention for single-query decode because
// FlashAttention's tile-based approach has overhead for tiny query lengths.

#include <cuda_fp16.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Grid:  (num_q_heads,)
// Block: (BLOCK_SIZE,)   — threads cooperate over kv_len and head_dim
//
// q:      [num_q_heads, head_dim] fp16           — single query token
// k_cache: [num_kv_heads, max_kv_len, head_dim] fp16 — full KV buffer
// v_cache: [num_kv_heads, max_kv_len, head_dim] fp16
// output: [num_q_heads, head_dim] fp16
// valid_kv_len: number of valid KV positions (for masking)
// scale:  1/sqrt(head_dim)
extern "C" __global__ void decode_attention_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ output,
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int max_kv_len,
    const int valid_kv_len,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int num_kv_groups = num_q_heads / num_kv_heads;
    const int kv_head = q_head / num_kv_groups;

    // Pointers for this head
    // Q layout: [num_q_heads, head_dim] (flat, contiguous per head)
    const __half* q_ptr = q + q_head * head_dim;
    __half* out_ptr = output + q_head * head_dim;

    // K/V cache layout: [seq_len, num_kv_heads, head_dim] (candle's layout)
    // To access head h at position p: offset = p * num_kv_heads * head_dim + h * head_dim
    const int kv_stride = num_kv_heads * head_dim;  // stride per sequence position

    // Shared memory for attention scores (one per kv position)
    extern __shared__ float s_scores[];

    // Step 1: Compute Q·K^T scores for all valid kv positions
    float local_max = -1e20f;
    for (int kv_pos = threadIdx.x; kv_pos < valid_kv_len; kv_pos += blockDim.x) {
        float score = 0.0f;
        // K at [kv_pos, kv_head, :] in [seq, heads, dim] layout
        const __half* k_row = k_cache + kv_pos * kv_stride + kv_head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(q_ptr[d]) * __half2float(k_row[d]);
        }
        score *= scale;
        s_scores[kv_pos] = score;
        local_max = fmaxf(local_max, score);
    }
    // Fill invalid positions with -inf
    for (int kv_pos = valid_kv_len + threadIdx.x; kv_pos < max_kv_len; kv_pos += blockDim.x) {
        s_scores[kv_pos] = -1e20f;
    }
    __syncthreads();

    // Step 2: Softmax — find global max
    // Reduce max across all threads in the block
    float block_max = local_max;
    // Use shared memory for cross-warp max reduction
    __shared__ float s_max[32];
    {
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        block_max = warp_reduce_max(block_max);
        if (lane == 0) s_max[wid] = block_max;
        __syncthreads();
        block_max = (threadIdx.x < (blockDim.x >> 5)) ? s_max[threadIdx.x & 31] : -1e20f;
        if (wid == 0) block_max = warp_reduce_max(block_max);
    }
    // Broadcast max to all threads
    __shared__ float s_global_max;
    if (threadIdx.x == 0) s_global_max = block_max;
    __syncthreads();
    float global_max = s_global_max;

    // Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kv_pos = threadIdx.x; kv_pos < valid_kv_len; kv_pos += blockDim.x) {
        float val = expf(s_scores[kv_pos] - global_max);
        s_scores[kv_pos] = val;
        local_sum += val;
    }
    __syncthreads();

    // Reduce sum
    __shared__ float s_sum[32];
    {
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) s_sum[wid] = local_sum;
        __syncthreads();
        local_sum = (threadIdx.x < (blockDim.x >> 5)) ? s_sum[threadIdx.x & 31] : 0.0f;
        if (wid == 0) local_sum = warp_reduce_sum(local_sum);
    }
    __shared__ float s_global_sum;
    if (threadIdx.x == 0) s_global_sum = local_sum;
    __syncthreads();
    float inv_sum = 1.0f / s_global_sum;

    // Normalize scores
    for (int kv_pos = threadIdx.x; kv_pos < valid_kv_len; kv_pos += blockDim.x) {
        s_scores[kv_pos] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Compute weighted sum of V: output = sum(scores[i] * V[i])
    // V cache layout: [seq_len, num_kv_heads, head_dim] (same as K)
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < valid_kv_len; kv_pos++) {
            // V at [kv_pos, kv_head, d] in [seq, heads, dim] layout
            acc += s_scores[kv_pos] * __half2float(
                v_cache[kv_pos * kv_stride + kv_head * head_dim + d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}
