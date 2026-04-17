// Single-token decode attention (head-major KV cache layout).
//
// The canonical `decode_attention.cu` expects SEQ-MAJOR cache
// `[seq, nkv, hd]` (candle's original). The new LlamaFamilyModel<B>
// path stores cache HEAD-MAJOR via `kv_cache_append_head_major_f16`,
// layout `[nkv, capacity, hd]`. Feeding that into the seq-major kernel
// produces garbage output.
//
// This variant reworks the kernel for head-major cache. Launch:
//   grid  = (num_q_heads,)
//   block = (256,)
// Shared mem = valid_kv_len * 4 bytes (one f32 per KV position).
//
// Q layout: [num_q_heads, head_dim] (single token, head-major flat)
// Output:   [num_q_heads, head_dim] (same)
// K/V cache: [num_kv_heads, capacity, head_dim]  (capacity >= valid_kv_len)
//
// cache_stride = capacity * head_dim (per KV head)

#include "common.cuh"

#define WARP_SIZE 32

extern "C" __global__ void decode_attention_head_major_f16(
    const __half* __restrict__ q,             // [num_q_heads * head_dim]
    const __half* __restrict__ k_cache,       // [num_kv_heads, capacity, head_dim]
    const __half* __restrict__ v_cache,       // [num_kv_heads, capacity, head_dim]
    __half* __restrict__ output,              // [num_q_heads * head_dim]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int capacity,
    const int valid_kv_len,
    const float scale
) {
    const int q_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int nq_per_kv = num_q_heads / num_kv_heads;
    const int kv_head = q_head / nq_per_kv;

    const __half* q_ptr = q + q_head * head_dim;
    __half* out_ptr = output + q_head * head_dim;

    // Base pointer for this KV head's cache slab.
    const __half* k_head_base = k_cache + (size_t)kv_head * capacity * head_dim;
    const __half* v_head_base = v_cache + (size_t)kv_head * capacity * head_dim;

    extern __shared__ float s_scores[]; // valid_kv_len entries

    // ── Step 1: scores[p] = dot(Q, K[kv_head, p, :]) * scale ──
    // Each thread walks a subset of KV positions.
    for (int p = tid; p < valid_kv_len; p += blockDim.x) {
        const __half* k_ptr = k_head_base + (size_t)p * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        s_scores[p] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: online softmax (single-pass max-stable) ──
    // Each thread computes a partial max over its slice, then reduce across block.
    float local_max = -1e30f;
    for (int p = tid; p < valid_kv_len; p += blockDim.x) {
        if (s_scores[p] > local_max) local_max = s_scores[p];
    }
    float block_max = block_reduce_max(local_max);
    __shared__ float s_block_max;
    if (tid == 0) s_block_max = block_max;
    __syncthreads();
    const float row_max = s_block_max;

    // Convert scores to exp(score - max), sum.
    float local_sum = 0.0f;
    for (int p = tid; p < valid_kv_len; p += blockDim.x) {
        float e = expf(s_scores[p] - row_max);
        s_scores[p] = e;
        local_sum += e;
    }
    float block_sum = block_reduce_sum(local_sum);
    __shared__ float s_block_sum;
    if (tid == 0) s_block_sum = block_sum;
    __syncthreads();
    const float inv_sum = 1.0f / s_block_sum;

    // ── Step 3: out[d] = sum_p s_scores[p] * V[kv_head, p, d] / block_sum ──
    // Each thread handles a subset of head_dim elements.
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < valid_kv_len; p++) {
            const __half* v_ptr = v_head_base + (size_t)p * head_dim;
            acc += s_scores[p] * __half2float(v_ptr[d]);
        }
        out_ptr[d] = __float2half(acc * inv_sum);
    }
}
