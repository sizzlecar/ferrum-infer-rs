// Flash Decoding for single-query attention with GQA support.
//
// Splits KV sequence across multiple thread blocks per Q head for better
// SM utilization on long contexts. Two-phase algorithm:
//
// Phase 1 (flash_decode_attn_f16):
//   Grid:  (num_q_heads, num_splits)
//   Block: (256,)
//   Each block handles KV range [split_start, split_end) for one Q head.
//   Outputs: partial V accumulation, local max score, local exp sum.
//
// Phase 2 (flash_decode_reduce_f16):
//   Grid:  (num_q_heads,)
//   Block: (min(head_dim, 256),)
//   Combines partial results using log-sum-exp rescaling.
//
// Reference: Flash-Decoding (Stanford CRFM, 2023)
// K/V cache layout: [seq_len, num_kv_heads, head_dim]

#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_SPLITS 32

// ====================== Warp/Block Reductions ======================

__inline__ __device__ float warp_reduce_sum_fd(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_max_fd(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__inline__ __device__ float block_reduce_sum_fd(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum_fd(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum_fd(val);
    return val;
}

// ====================== Phase 1: Split-K Attention ======================
//
// Each block computes attention for one (q_head, split) pair over a chunk
// of the KV sequence. Outputs unnormalized weighted V sum plus log-sum-exp
// components for Phase 2 to combine.
//
// Shared memory: chunk_size * sizeof(float) for attention scores.

extern "C" __global__ void flash_decode_attn_f16(
    const __half* __restrict__ q,          // [num_q_heads, head_dim]
    const __half* __restrict__ k_cache,    // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ v_cache,    // [seq_len, num_kv_heads, head_dim]
    float* __restrict__ partial_out,       // [num_q_heads, num_splits, head_dim]
    float* __restrict__ partial_m,         // [num_q_heads, num_splits]
    float* __restrict__ partial_l,         // [num_q_heads, num_splits]
    const int num_q_heads,
    const int num_kv_heads,
    const int head_dim,
    const int valid_kv_len,
    const float scale,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int split_id = blockIdx.y;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int kv_stride = num_kv_heads * head_dim;

    // KV range for this split
    const int chunk_size = (valid_kv_len + num_splits - 1) / num_splits;
    const int split_start = split_id * chunk_size;
    const int split_end = min(split_start + chunk_size, valid_kv_len);
    const int my_len = split_end - split_start;

    const int out_idx = q_head * num_splits + split_id;

    if (my_len <= 0) {
        // Empty split — write sentinels so Phase 2 ignores this split
        if (threadIdx.x == 0) {
            partial_m[out_idx] = -1e20f;
            partial_l[out_idx] = 0.0f;
        }
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            partial_out[out_idx * head_dim + d] = 0.0f;
        return;
    }

    const __half* q_ptr = q + q_head * head_dim;

    // Load Q into registers (warp-cooperative: each lane holds a stripe)
    const int elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[8]; // supports head_dim up to 256
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int d = (threadIdx.x % WARP_SIZE) + i * WARP_SIZE;
        q_reg[i] = (i < elems_per_thread && d < head_dim)
            ? __half2float(q_ptr[d]) : 0.0f;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Shared memory for local attention scores
    extern __shared__ float smem[];

    // ---- Step 1: Q·K^T for this chunk ----
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
        float score = warp_reduce_sum_fd(dot) * scale;
        if (lane_id == 0)
            smem[pos] = score;
    }
    __syncthreads();

    // ---- Step 2: Local softmax ----
    float thread_max = -1e20f;
    for (int i = threadIdx.x; i < my_len; i += blockDim.x)
        thread_max = fmaxf(thread_max, smem[i]);
    float local_max = block_reduce_max_fd(thread_max);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = local_max;
    __syncthreads();
    local_max = s_max;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < my_len; i += blockDim.x) {
        float v = expf(smem[i] - local_max);
        smem[i] = v;
        thread_sum += v;
    }
    __syncthreads();

    float local_sum = block_reduce_sum_fd(thread_sum);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = local_sum;
    __syncthreads();

    // Write local max and exp-sum for Phase 2
    if (threadIdx.x == 0) {
        partial_m[out_idx] = local_max;
        partial_l[out_idx] = s_sum;
    }

    // ---- Step 3: Weighted V accumulation (unnormalized) ----
    float* out_ptr = partial_out + out_idx * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < my_len; i++) {
            acc += smem[i] * __half2float(
                v_cache[(split_start + i) * kv_stride + kv_head * head_dim + d]);
        }
        out_ptr[d] = acc;
    }
}

// ====================== Phase 2: Reduce Across Splits ======================
//
// Combines partial results from all splits using log-sum-exp rescaling.
// Each block handles one Q head.
//
// No shared memory needed (num_splits is small, fits in registers).

extern "C" __global__ void flash_decode_reduce_f16(
    const float* __restrict__ partial_out,  // [num_q_heads, num_splits, head_dim]
    const float* __restrict__ partial_m,    // [num_q_heads, num_splits]
    const float* __restrict__ partial_l,    // [num_q_heads, num_splits]
    __half* __restrict__ output,            // [num_q_heads, head_dim]
    const int head_dim,
    const int num_splits
) {
    const int q_head = blockIdx.x;
    const int base = q_head * num_splits;

    // All threads compute rescale factors (num_splits <= MAX_SPLITS = 32)
    float global_max = -1e20f;
    for (int s = 0; s < num_splits; s++)
        global_max = fmaxf(global_max, partial_m[base + s]);

    float rescale[MAX_SPLITS];
    float global_sum = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        rescale[s] = expf(partial_m[base + s] - global_max);
        global_sum += partial_l[base + s] * rescale[s];
    }
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // Combine partial V outputs (parallelized across head_dim)
    __half* out_ptr = output + q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < num_splits; s++)
            acc += partial_out[(base + s) * head_dim + d] * rescale[s];
        out_ptr[d] = __float2half(acc * inv_sum);
    }
}
