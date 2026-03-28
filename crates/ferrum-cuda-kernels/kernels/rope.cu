// Fused RoPE (Rotary Position Embedding) kernel for Q and K.
//
// Applies rotary embedding to both query and key tensors in a single launch.
// Uses pre-computed cos/sin tables indexed by position.
//
// RoPE rotation for pair (x0, x1):
//   x0_out = x0 * cos - x1 * sin
//   x1_out = x1 * cos + x0 * sin

#include <cuda_fp16.h>

// Grid:  (num_q_heads + num_k_heads,)  — one block per head
// Block: (head_dim / 2,)               — one thread per rotation pair
//
// q:     [num_q_heads, head_dim] fp16 (read-only)
// k:     [num_k_heads, head_dim] fp16 (read-only)
// cos:   [head_dim / 2] fp16          — cos table row for current position
// sin:   [head_dim / 2] fp16          — sin table row for current position
// q_out: [num_q_heads, head_dim] fp16
// k_out: [num_k_heads, head_dim] fp16
extern "C" __global__ void rope_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    const int num_q_heads,
    const int num_k_heads,
    const int head_dim
) {
    const int head_idx = blockIdx.x;
    const int pair_idx = threadIdx.x;  // which rotation pair [0, head_dim/2)
    const int half_dim = head_dim / 2;

    if (pair_idx >= half_dim) return;

    float c = __half2float(cos_table[pair_idx]);
    float s = __half2float(sin_table[pair_idx]);

    if (head_idx < num_q_heads) {
        // Process Q head
        int base = head_idx * head_dim;
        float x0 = __half2float(q[base + pair_idx]);
        float x1 = __half2float(q[base + pair_idx + half_dim]);
        q_out[base + pair_idx]            = __float2half(x0 * c - x1 * s);
        q_out[base + pair_idx + half_dim] = __float2half(x1 * c + x0 * s);
    } else {
        // Process K head
        int ki = head_idx - num_q_heads;
        int base = ki * head_dim;
        float x0 = __half2float(k[base + pair_idx]);
        float x1 = __half2float(k[base + pair_idx + half_dim]);
        k_out[base + pair_idx]            = __float2half(x0 * c - x1 * s);
        k_out[base + pair_idx + half_dim] = __float2half(x1 * c + x0 * s);
    }
}

// Batched RoPE: processes B items in a single launch.
// Each batch item has its own position (different cos/sin row).
//
// Grid:  ((num_q_heads + num_k_heads) * batch_size,)
// Block: (head_dim / 2,)
//
// q:         [B, num_q_heads, head_dim]
// k:         [B, num_k_heads, head_dim]
// cos_table: [max_seq_len, head_dim/2] — full table (not pre-sliced)
// sin_table: same layout
// positions: [B] — per-item position index
// q_out:     [B, num_q_heads, head_dim]
// k_out:     [B, num_k_heads, head_dim]
extern "C" __global__ void batched_rope_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ cos_table,
    const __half* __restrict__ sin_table,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    const int* __restrict__ positions,
    const int num_q_heads,
    const int num_k_heads,
    const int head_dim,
    const int batch_size
) {
    const int pair_idx = threadIdx.x;
    const int half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

    const int total_heads = num_q_heads + num_k_heads;
    const int batch_idx = blockIdx.x / total_heads;
    const int head_idx = blockIdx.x % total_heads;

    if (batch_idx >= batch_size) return;

    const int pos = positions[batch_idx];
    const int cos_off = pos * half_dim;
    float c = __half2float(cos_table[cos_off + pair_idx]);
    float s = __half2float(sin_table[cos_off + pair_idx]);

    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_k_heads * head_dim;

    if (head_idx < num_q_heads) {
        int base = batch_idx * q_dim + head_idx * head_dim;
        float x0 = __half2float(q[base + pair_idx]);
        float x1 = __half2float(q[base + pair_idx + half_dim]);
        q_out[base + pair_idx]            = __float2half(x0 * c - x1 * s);
        q_out[base + pair_idx + half_dim] = __float2half(x1 * c + x0 * s);
    } else {
        int ki = head_idx - num_q_heads;
        int base = batch_idx * kv_dim + ki * head_dim;
        float x0 = __half2float(k[base + pair_idx]);
        float x1 = __half2float(k[base + pair_idx + half_dim]);
        k_out[base + pair_idx]            = __float2half(x0 * c - x1 * s);
        k_out[base + pair_idx + half_dim] = __float2half(x1 * c + x0 * s);
    }
}

extern "C" __global__ void rope_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    const int num_q_heads,
    const int num_k_heads,
    const int head_dim
) {
    const int head_idx = blockIdx.x;
    const int pair_idx = threadIdx.x;
    const int half_dim = head_dim / 2;

    if (pair_idx >= half_dim) return;

    float c = cos_table[pair_idx];
    float s = sin_table[pair_idx];

    if (head_idx < num_q_heads) {
        int base = head_idx * head_dim;
        float x0 = q[base + pair_idx];
        float x1 = q[base + pair_idx + half_dim];
        q_out[base + pair_idx]            = x0 * c - x1 * s;
        q_out[base + pair_idx + half_dim] = x1 * c + x0 * s;
    } else {
        int ki = head_idx - num_q_heads;
        int base = ki * head_dim;
        float x0 = k[base + pair_idx];
        float x1 = k[base + pair_idx + half_dim];
        k_out[base + pair_idx]            = x0 * c - x1 * s;
        k_out[base + pair_idx + half_dim] = x1 * c + x0 * s;
    }
}
