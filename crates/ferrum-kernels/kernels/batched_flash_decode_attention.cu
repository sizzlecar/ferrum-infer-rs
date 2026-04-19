// Batched Flash Decoding for multi-request decode.
//
// Same split-K algorithm as flash_decode_attention.cu, but processes
// multiple batch items in a single kernel launch.
//
// Phase 1 (batched_flash_decode_attn_f16):
//   Grid:  (num_q_heads, num_splits, batch_size)
//   Block: (256,)
//   Each block handles one (q_head, split, batch_item) triple.
//   K/V accessed via device pointer arrays (one per batch item).
//
// Phase 2 (batched_flash_decode_reduce_f16):
//   Grid:  (num_q_heads, batch_size)
//   Block: (min(head_dim, 256),)
//   Combines partial results per (q_head, batch_item).

#include "common.cuh"

#define WARP_SIZE 32
#define MAX_SPLITS 32

// ====================== Phase 1: Batched Split-K ======================

extern "C" __global__ void batched_flash_decode_attn_f16(
    const __half* __restrict__ q_all,         // [batch, num_q_heads, head_dim]
    const __half** __restrict__ k_cache_ptrs,  // [batch] → each [max_kv, nkv, hd]
    const __half** __restrict__ v_cache_ptrs,  // [batch] → each [max_kv, nkv, hd]
    const int* __restrict__ valid_kv_lens,     // [batch]
    float* __restrict__ partial_out,           // [batch, num_q_heads, num_splits, head_dim]
    float* __restrict__ partial_m,             // [batch, num_q_heads, num_splits]
    float* __restrict__ partial_l,             // [batch, num_q_heads, num_splits]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int split_id = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_stride = num_kv_heads * head_dim;

    const int valid_kv_len = valid_kv_lens[batch_idx];
    const int chunk_size = (valid_kv_len + num_splits - 1) / num_splits;
    const int split_start = split_id * chunk_size;
    const int split_end = min(split_start + chunk_size, valid_kv_len);
    const int my_len = split_end - split_start;

    // Output index: [batch, q_head, split]
    const int heads_splits = num_q_heads * num_splits;
    const int out_idx = batch_idx * heads_splits + q_head * num_splits + split_id;

    if (my_len <= 0) {
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -1e20f;
            partial_l[out_idx] = 0.0f;
        }
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            partial_out[out_idx * head_dim + d] = 0.0f;
        return;
    }

    // Q for this batch item
    const __half* q_ptr = q_all + batch_idx * num_q_heads * head_dim + q_head * head_dim;

    // K/V for this batch item
    const __half* k_cache = k_cache_ptrs[batch_idx];
    const __half* v_cache = v_cache_ptrs[batch_idx];

    // Load Q into registers
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

    extern __shared__ float smem[];

    // Step 1: Q·K^T for this chunk
    for (int pos = warp_id; pos < my_len; pos += num_warps) {
        const int kv_pos = split_start + pos;
        const __half* k_row = k_cache + kv_pos * kv_stride + kv_head * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int d = lane_id + i * WARP_SIZE;
            if (i < elems_per_thread && d < head_dim)
                dot += q_reg[i] * __half2float(k_row[d]);
        }
        float score = warp_reduce_sum(dot) * scale;
        if (lane_id == 0) smem[pos] = score;
    }
    __syncthreads();

    // Step 2: local max + softmax
    float thread_max = -1e20f;
    for (int pos = threadIdx.x; pos < my_len; pos += blockDim.x)
        thread_max = fmaxf(thread_max, smem[pos]);
    float local_max = block_reduce_max(thread_max);
    if (threadIdx.x == 0) smem[my_len] = local_max;
    __syncthreads();
    local_max = smem[my_len];

    float thread_sum = 0.0f;
    for (int pos = threadIdx.x; pos < my_len; pos += blockDim.x) {
        float w = expf(smem[pos] - local_max);
        smem[pos] = w;
        thread_sum += w;
    }
    float local_sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) {
        partial_m[out_idx] = local_max;
        partial_l[out_idx] = local_sum;
    }
    __syncthreads();

    // Step 3: weighted V accumulation
    float* po = partial_out + out_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < my_len; pos++) {
            const int kv_pos = split_start + pos;
            const __half* v_row = v_cache + kv_pos * kv_stride + kv_head * head_dim;
            acc += smem[pos] * __half2float(v_row[d]);
        }
        po[d] = acc;
    }
}

// ====================== Phase 2: Batched Reduce ======================

extern "C" __global__ void batched_flash_decode_reduce_f16(
    const float* __restrict__ partial_out,  // [batch, nq, num_splits, hd]
    const float* __restrict__ partial_m,    // [batch, nq, num_splits]
    const float* __restrict__ partial_l,    // [batch, nq, num_splits]
    __half* __restrict__ output,            // [batch, nq, hd]
    const int num_q_heads,
    const int head_dim,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= head_dim) return;

    const int heads_splits = num_q_heads * num_splits;
    const int base = batch_idx * heads_splits + q_head * num_splits;

    // Find global max across splits
    float global_max = -1e20f;
    for (int s = 0; s < num_splits; s++)
        global_max = fmaxf(global_max, partial_m[base + s]);

    // Combine using log-sum-exp rescaling
    float global_sum = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float m_s = partial_m[base + s];
        float l_s = partial_l[base + s];
        float w = expf(m_s - global_max) * l_s;
        global_sum += w;
        acc += w * partial_out[(base + s) * head_dim + d] / fmaxf(l_s, 1e-10f);
    }
    float result = acc / fmaxf(global_sum, 1e-10f);
    output[batch_idx * num_q_heads * head_dim + q_head * head_dim + d] = __float2half(result);
}
