// MoE router: per-row softmax + top-K + optional renormalize.
//
// Replaces the host-side `B::sync(ctx) + B::to_vec(router_logits) +
// crate::moe::router::route_into(...)` round trip. Doing it on the GPU
// removes the per-layer pipeline drain entirely on the routing leg —
// the kernel writes ids/weights into device buffers; the caller D2H's
// only the small ids array (batch * top_k * 4 B ≈ 1 KB at c=32 / k=8)
// for the host-side bucket plan, instead of the 16 KB router_logits.
//
// Algorithm: one block per token row.
//   1. Stable softmax (max-subtract → exp → sum-reduce → divide).
//   2. Repeated argmax-mask top-K (k passes; K ≤ 32 in known MoE configs).
//      Picked entries get overwritten with -INFINITY in shared mem so
//      the next pass sees the next-best.
//   3. Optional renormalize: if `norm_topk_prob`, divide selected
//      weights by their sum so they total 1.0 (Qwen3-MoE default).
//
// Block: 32 threads (one warp). Each thread covers ceil(num_experts/32)
// logits during the softmax/argmax reductions. Shared memory holds
// the per-row probability vector — `num_experts` fp32 elements.
// Tie-break on argmax: smaller index wins (matches host `route_into`
// for bit-exact reproducibility).
//
// Input is f16 (the router gemv writes f16 via cuBLAS); softmax math
// runs in f32 to preserve precision. Output ids are i32, weights f32.

#include <cstdint>
#include <cuda_fp16.h>

#define WARP_SIZE 32

template <bool emit_single_token_marlin_blocks>
__device__ __forceinline__ void moe_router_topk_softmax_f16_impl(
    const __half* __restrict__ logits,      // [batch, num_experts]
    int32_t*      __restrict__ out_ids,      // [batch, top_k]
    float*        __restrict__ out_weights,  // [batch, top_k]
    int32_t*      __restrict__ sorted_token_ids,
    int32_t*      __restrict__ expert_block_ids,
    int32_t*      __restrict__ total_tokens_post_pad,
    int batch,
    int num_experts,
    int top_k,
    int norm_topk_prob,  // 0 / 1
    int moe_block_size
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    if constexpr (emit_single_token_marlin_blocks) {
        if (batch != 1 || moe_block_size <= 0) return;
    }

    int tid = threadIdx.x;  // 0..31

    extern __shared__ float smem[];
    float* probs = smem;  // [num_experts]

    // ── Pass 1: cooperative load f16→f32 + per-thread row max. ───────────
    float thread_max = -INFINITY;
    for (int i = tid; i < num_experts; i += WARP_SIZE) {
        float v = __half2float(logits[row * num_experts + i]);
        probs[i] = v;
        if (v > thread_max) thread_max = v;
    }
    // Warp-wide max reduction.
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, thread_max, offset);
        if (other > thread_max) thread_max = other;
    }
    float row_max = thread_max;

    // ── Pass 2: exp(logit - max) + warp-wide sum reduction. ──────────────
    float thread_sum = 0.0f;
    for (int i = tid; i < num_experts; i += WARP_SIZE) {
        float e = __expf(probs[i] - row_max);
        probs[i] = e;
        thread_sum += e;
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_xor_sync(0xffffffff, thread_sum, offset);
    }
    float inv_sum = 1.0f / thread_sum;

    // ── Pass 3: normalize in shared mem. ─────────────────────────────────
    for (int i = tid; i < num_experts; i += WARP_SIZE) {
        probs[i] *= inv_sum;
    }
    // Warp-only block, no __syncthreads needed for warp-coherent shmem,
    // but we follow it with a barrier to be safe — the cost is negligible
    // (~1 cycle on a single-warp block).
    __syncwarp(0xffffffff);

    // ── Top-K via argmax-mask. ───────────────────────────────────────────
    // K passes: each picks the largest remaining prob, writes (id, weight)
    // to global, then masks the picked slot with -INFINITY for the next
    // pass. Ties are broken by smaller index winning (matches host route).
    float sel_sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        // Find this thread's local best.
        float thread_best = -INFINITY;
        int   thread_idx  = 0;
        for (int i = tid; i < num_experts; i += WARP_SIZE) {
            float v = probs[i];
            if (v > thread_best) {
                thread_best = v;
                thread_idx = i;
            }
        }
        // Warp-wide max reduction → all lanes hold `warp_best`.
        float warp_best = thread_best;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, warp_best, offset);
            if (other > warp_best) warp_best = other;
        }
        // Smallest index whose thread holds `warp_best` wins. Lanes
        // not holding it report INT_MAX so they lose the min reduction.
        int my_idx_for_min = (thread_best == warp_best) ? thread_idx : 0x7fffffff;
        int win_idx = my_idx_for_min;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            int other = __shfl_xor_sync(0xffffffff, win_idx, offset);
            if (other < win_idx) win_idx = other;
        }

        if (tid == 0) {
            out_ids[row * top_k + k] = win_idx;
            out_weights[row * top_k + k] = warp_best;
            probs[win_idx] = -INFINITY;  // mask for next pass
        }
        sel_sum += warp_best;  // every lane now tracks the running sum
        __syncwarp(0xffffffff);
    }

    // ── Optional renorm of the K picked weights. ─────────────────────────
    if (norm_topk_prob != 0) {
        float scale;
        if (sel_sum > 0.0f) {
            scale = 1.0f / sel_sum;
        } else {
            scale = 1.0f / (float)top_k;
        }
        if (tid < top_k) {
            if (sel_sum > 0.0f) {
                out_weights[row * top_k + tid] *= scale;
            } else {
                out_weights[row * top_k + tid] = scale;
            }
        }
    }

    if constexpr (emit_single_token_marlin_blocks) {
        // A single token's top-K expert ids are unique, so every selected
        // expert owns exactly one Marlin block. Rank the selected experts by
        // id so the effective metadata matches generic align byte-for-byte,
        // then pad only the rows Marlin will read. This avoids the generic
        // expert histogram, prefix scan, and full-capacity clear.
        int padded_pair_count = top_k * moe_block_size;
        if (tid < top_k) {
            int expert = out_ids[tid];
            int expert_rank = 0;
            for (int k = 0; k < top_k; ++k) {
                int other = out_ids[k];
                expert_rank += other < expert || (other == expert && k < tid);
            }
            expert_block_ids[expert_rank] = expert;
            int block_start = expert_rank * moe_block_size;
            sorted_token_ids[block_start] = tid;
            for (int offset = 1; offset < moe_block_size; ++offset) {
                sorted_token_ids[block_start + offset] = top_k;
            }
        }
        if (tid == 0) {
            total_tokens_post_pad[0] = padded_pair_count;
        }
    }
}

extern "C" __global__ void moe_router_topk_softmax_f16(
    const __half* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    int batch,
    int num_experts,
    int top_k,
    int norm_topk_prob
) {
    moe_router_topk_softmax_f16_impl<false>(
        logits,
        out_ids,
        out_weights,
        nullptr,
        nullptr,
        nullptr,
        batch,
        num_experts,
        top_k,
        norm_topk_prob,
        0
    );
}

extern "C" __global__ void moe_router_topk_softmax_f16_single_token_marlin(
    const __half* __restrict__ logits,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_block_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int batch,
    int num_experts,
    int top_k,
    int norm_topk_prob,
    int moe_block_size
) {
    moe_router_topk_softmax_f16_impl<true>(
        logits,
        out_ids,
        out_weights,
        sorted_token_ids,
        expert_block_ids,
        total_tokens_post_pad,
        batch,
        num_experts,
        top_k,
        norm_topk_prob,
        moe_block_size
    );
}
