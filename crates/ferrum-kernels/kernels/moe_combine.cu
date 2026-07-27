// MoE combine: per-token weighted sum across top_k expert outputs.
//
// After the bucketed dispatch:
//   * `packed_down` holds `[total_pairs, hidden]` row-major — one row per
//     (token, k_slot) pair, in expert-bucketed order. So row `r` was
//     produced by some expert e for some original (token_b, k_slot).
//   * `pairs_by_token[batch * top_k]` is the inverse map: for each
//     (b, k_slot), which row of `packed_down` carries that pair's
//     contribution.
//   * `pair_weights[batch * top_k]` is the combine weight (router
//     softmax × renorm).
//
// This kernel computes:
//
//     out[b, h] = sum_{k in 0..top_k}
//                     pair_weights[b * top_k + k] *
//                     packed_down[pairs_by_token[b * top_k + k] * hidden + h]
//
// Grid: (ceil(hidden/256), batch, 1). Each thread handles one (b, h).
// One pass over top_k inside the thread — no atomics, no shared memory.
// For Qwen3-30B-A3B (hidden=2048, top_k=8, batch=512 prefill):
// 2048/256 × 512 = 4096 blocks × 256 threads = 1M threads, each loops 8.
// Latency-bound; bandwidth ≈ batch × top_k × hidden × 2 bytes read +
// batch × hidden × 2 bytes write ≈ ~17 MB at this size. 1 TB/s → ~17 us.

#include <cstdint>
#include <cuda_fp16.h>

extern "C" __global__ void moe_combine_f16(
    const __half* __restrict__ packed_down,    // [total_pairs, hidden]
    const int32_t* __restrict__ pairs_by_token, // [batch, top_k]
    const float* __restrict__ pair_weights,    // [batch, top_k] — fp32 softmax
    __half* __restrict__ out,                  // [batch, hidden]
    int batch,
    int hidden,
    int top_k
) {
    int b = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || h >= hidden) return;

    float acc = 0.0f;
    int row_base = b * top_k;
    #pragma unroll 4
    for (int k = 0; k < top_k; k++) {
        int pair_row = pairs_by_token[row_base + k];
        // pair_row < 0 means "this slot is unused" (e.g. tokens with
        // < top_k actually-used experts; not currently emitted by router
        // but kept as a safety branch). Skip silently.
        if (pair_row < 0) continue;
        float w = pair_weights[row_base + k];
        float v = __half2float(packed_down[pair_row * hidden + h]);
        acc += w * v;
    }
    out[b * hidden + h] = __float2half(acc);
}

// Fast path when rows are already in original [batch, top_k] order.
// Used by the vLLM pair-id MoE path: marlin_moe consumes sorted pair ids
// but writes phase-3 output back to row `pair_id`, so no inverse
// `pairs_by_token` lookup is needed for the final top-k reduction.
extern "C" __global__ void weighted_sum_batched_f16(
    const __half* __restrict__ slots,       // [batch, top_k, hidden]
    const float* __restrict__ weights,      // [batch, top_k]
    __half* __restrict__ out,               // [batch, hidden]
    int batch,
    int top_k,
    int hidden
) {
    int b = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || h >= hidden) return;

    float acc = 0.0f;
    int row_base = b * top_k;
    #pragma unroll 4
    for (int k = 0; k < top_k; k++) {
        int pair_row = row_base + k;
        float w = weights[row_base + k];
        float v = __half2float(slots[pair_row * hidden + h]);
        acc += w * v;
    }
    out[b * hidden + h] = __float2half(acc);
}

// Shared-expert merge used by the backend-neutral routed/shared SwiGLU MoE
// contract: dst += values * sigmoid(token_gate). The gate has one scalar per
// token and is broadcast across the hidden dimension.
extern "C" __global__ void apply_token_gate_and_add_inplace_f16(
    __half* __restrict__ dst,              // [batch, hidden]
    const __half* __restrict__ values,     // [batch, hidden]
    const __half* __restrict__ token_gate, // [batch]
    int batch,
    int hidden
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden;
    if (idx >= total) return;

    int token = idx / hidden;
    float gate = __half2float(token_gate[token]);
    float scale = 1.0f / (1.0f + expf(-gate));
    float merged = __half2float(dst[idx]) + __half2float(values[idx]) * scale;
    dst[idx] = __float2half(merged);
}

// Same logic for f32 outputs (CPU parity testing convenience).
extern "C" __global__ void moe_combine_f32(
    const float* __restrict__ packed_down,
    const int32_t* __restrict__ pairs_by_token,
    const float* __restrict__ pair_weights,
    float* __restrict__ out,
    int batch,
    int hidden,
    int top_k
) {
    int b = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || h >= hidden) return;

    float acc = 0.0f;
    int row_base = b * top_k;
    #pragma unroll 4
    for (int k = 0; k < top_k; k++) {
        int pair_row = pairs_by_token[row_base + k];
        if (pair_row < 0) continue;
        acc += pair_weights[row_base + k] * packed_down[pair_row * hidden + h];
    }
    out[b * hidden + h] = acc;
}
