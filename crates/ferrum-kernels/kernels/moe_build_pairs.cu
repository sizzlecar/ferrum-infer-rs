// MoE device-side pairs_by_token builder.
//
// Inverts the bucket-sort permutation so the moe_combine kernel can
// read the device-side routing output without a host round-trip.
//
// After `B::route_topk_softmax` produces `expert_ids[batch * top_k]`
// (each entry = the expert id that token b chose at slot k), this
// kernel builds:
//   * `pairs_by_token[batch * top_k]` — for each (b, k), the row index
//     into `packed_down` (= the sorted-by-expert position of (b, k)).
//   * `expert_offsets[num_experts + 1]` — exclusive prefix-sum of
//     tokens-per-expert (used by downstream bucketed dispatch).
//
// pairs_by_token is the INVERSE of sorted_token_ids produced by
// `moe_align_block_size`. Same algorithm (counting sort), exposed
// directly so callers don't have to invert the alignment output.
//
// Single-block launch — `batch * top_k ≤ ~1024` for Qwen3-MoE c=32 so
// one CTA suffices. Shared mem holds `num_experts` counts (≤ 256 × 4B
// = 1 KB) — well under the 48 KB/SM limit.

#include <cstdint>

extern "C" __global__ void moe_build_pairs_by_token(
    const int32_t* __restrict__ expert_ids,   // [batch * top_k]
    int32_t*       __restrict__ pairs_by_token,// [batch * top_k]
    int32_t*       __restrict__ expert_offsets,// [num_experts + 1]
    int batch_x_topk,
    int num_experts
) {
    // counts[e] used as both per-expert count (pass 1) and per-expert
    // write cursor (pass 3) — see comments below.
    extern __shared__ int32_t counts[];

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // ── Pass 1: zero counts ──────────────────────────────────────────
    for (int e = tid; e < num_experts; e += block_size) {
        counts[e] = 0;
    }
    __syncthreads();

    // ── Pass 1b: count tokens per expert (atomic for parallelism) ────
    for (int p = tid; p < batch_x_topk; p += block_size) {
        int e = expert_ids[p];
        if (e >= 0 && e < num_experts) {
            atomicAdd(&counts[e], 1);
        }
    }
    __syncthreads();

    // ── Pass 2: prefix sum on thread 0 (num_experts ≤ 256, tiny) ─────
    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < num_experts; e++) {
            expert_offsets[e] = acc;
            acc += counts[e];
            counts[e] = 0; // reset for use as write cursor in pass 3
        }
        expert_offsets[num_experts] = acc;
    }
    __syncthreads();

    // ── Pass 3: scatter (b, k) to its sorted slot ────────────────────
    // pairs_by_token[(b, k)] = expert_offsets[e] + per-expert position.
    // The per-expert position comes from atomicAdd into counts[e]
    // (now reused as a write cursor — starts at 0 from pass 2 reset).
    for (int p = tid; p < batch_x_topk; p += block_size) {
        int e = expert_ids[p];
        if (e < 0 || e >= num_experts) {
            pairs_by_token[p] = -1; // sentinel (skipped by moe_combine)
            continue;
        }
        int slot = atomicAdd(&counts[e], 1);
        pairs_by_token[p] = expert_offsets[e] + slot;
    }
}
