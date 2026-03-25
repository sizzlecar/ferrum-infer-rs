// Warp-cooperative single-query decode attention with GQA and online softmax.
//
// Each warp (32 threads) cooperates on one Q·K dot product:
//   - 32 threads each compute head_dim/32 partial products
//   - warp_reduce_sum combines them → one complete score
//   - Online softmax: running max + sum, no need to store all scores
//
// Grid:  (num_q_heads,)
// Block: (NUM_WARPS * 32,)  — multiple warps process KV positions in parallel
//
// K/V cache layout: [seq_len, num_kv_heads, head_dim] (candle's layout)

#include <cuda_fp16.h>

// Number of warps per block. Each warp handles one KV position at a time.
// More warps = more KV positions processed in parallel.
#define NUM_WARPS 8
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)

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

    const __half* q_ptr = q + q_head * head_dim;
    __half* out_ptr = output + q_head * head_dim;

    // K/V cache layout: [seq_len, num_kv_heads, head_dim]
    const int kv_stride = num_kv_heads * head_dim;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Each thread loads its portion of Q into registers (head_dim / 32 elements)
    // For head_dim=128: each thread holds 4 Q values
    const int q_elems_per_thread = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
    float q_reg[4];  // max head_dim=128 → 128/32=4
    #pragma unroll
    for (int i = 0; i < q_elems_per_thread && (lane_id + i * WARP_SIZE) < head_dim; i++) {
        q_reg[i] = __half2float(q_ptr[lane_id + i * WARP_SIZE]);
    }

    // ========== Phase 1: Q·K^T scores with online softmax ==========
    //
    // Online softmax: maintain running (max, sum) per warp, then reduce.
    // No need to store all scores to shared memory.
    //
    // Per-warp accumulators for V weighted sum (online softmax output)
    float v_acc[4];
    #pragma unroll
    for (int i = 0; i < q_elems_per_thread; i++) v_acc[i] = 0.0f;

    float warp_max = -1e20f;
    float warp_sum = 0.0f;

    // Each warp processes KV positions strided by NUM_WARPS
    for (int kv_pos = warp_id; kv_pos < valid_kv_len; kv_pos += NUM_WARPS) {
        // Warp-cooperative Q·K dot product
        const __half* k_row = k_cache + kv_pos * kv_stride + kv_head * head_dim;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < q_elems_per_thread && (lane_id + i * WARP_SIZE) < head_dim; i++) {
            partial += q_reg[i] * __half2float(k_row[lane_id + i * WARP_SIZE]);
        }
        // Reduce across warp → complete dot product (only lane 0 needs it,
        // but all lanes get the result from __shfl)
        float score = warp_reduce_sum(partial) * scale;

        // Online softmax update
        float old_max = warp_max;
        warp_max = fmaxf(warp_max, score);
        float exp_diff = expf(old_max - warp_max);  // correction factor

        // Rescale previous accumulator
        warp_sum = warp_sum * exp_diff;
        #pragma unroll
        for (int i = 0; i < q_elems_per_thread; i++) v_acc[i] *= exp_diff;

        // Add current position's contribution
        float exp_score = expf(score - warp_max);
        warp_sum += exp_score;

        // Accumulate V weighted by attention score
        const __half* v_row = v_cache + kv_pos * kv_stride + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < q_elems_per_thread && (lane_id + i * WARP_SIZE) < head_dim; i++) {
            v_acc[i] += exp_score * __half2float(v_row[lane_id + i * WARP_SIZE]);
        }
    }

    // ========== Phase 2: Cross-warp reduction ==========
    //
    // Each warp has its own (max, sum, v_acc). Merge them using the
    // online softmax correction formula.

    // Store per-warp results to shared memory
    __shared__ float s_max[NUM_WARPS];
    __shared__ float s_sum[NUM_WARPS];
    // v_acc per warp: [NUM_WARPS, head_dim] — each lane stores its elements
    extern __shared__ float s_vacc[];  // size = NUM_WARPS * head_dim

    if (lane_id == 0) {
        s_max[warp_id] = warp_max;
        s_sum[warp_id] = warp_sum;
    }
    // Store v_acc to shared memory
    #pragma unroll
    for (int i = 0; i < q_elems_per_thread && (lane_id + i * WARP_SIZE) < head_dim; i++) {
        s_vacc[warp_id * head_dim + lane_id + i * WARP_SIZE] = v_acc[i];
    }
    __syncthreads();

    // Warp 0 does the final reduction
    if (warp_id == 0) {
        // Find global max across all warps
        float global_max = -1e20f;
        for (int w = 0; w < NUM_WARPS; w++) {
            global_max = fmaxf(global_max, s_max[w]);
        }

        // Compute corrected sum and merge v_acc
        float global_sum = 0.0f;
        // Reset v_acc for final output
        #pragma unroll
        for (int i = 0; i < q_elems_per_thread; i++) v_acc[i] = 0.0f;

        for (int w = 0; w < NUM_WARPS; w++) {
            float correction = expf(s_max[w] - global_max);
            global_sum += s_sum[w] * correction;
            #pragma unroll
            for (int i = 0; i < q_elems_per_thread && (lane_id + i * WARP_SIZE) < head_dim; i++) {
                v_acc[i] += s_vacc[w * head_dim + lane_id + i * WARP_SIZE] * correction;
            }
        }

        // Normalize and write output
        float inv_sum = 1.0f / global_sum;
        #pragma unroll
        for (int i = 0; i < q_elems_per_thread && (lane_id + i * WARP_SIZE) < head_dim; i++) {
            out_ptr[lane_id + i * WARP_SIZE] = __float2half(v_acc[i] * inv_sum);
        }
    }
}
