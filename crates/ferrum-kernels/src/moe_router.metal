// MoE router: per-row softmax + top-K + optional renormalize.
//
// Replaces the host-side `B::sync(ctx) + B::to_vec(router_logits) +
// crate::moe::router::route(...)` sequence used in the existing MoE
// path. Each per-layer call previously paid one full Metal pipeline
// drain (~1 ms on M1 Max) plus a host softmax/sort. Doing it on the
// GPU removes the sync entirely (the kernel writes ids/weights into
// device buffers that the next mul_mm_id pass reads directly).
//
// Algorithm: one threadgroup per token row.
//   1. Stable softmax (max-subtract → exp → sum-reduce → divide).
//   2. Repeated argmax to extract top-K — k ≤ 32 expected, so a
//      simple loop that masks each picked entry to -INFINITY suffices
//      (vs a partial sort, which would over-engineer for tiny K).
//   3. Optional renormalize: if `norm_topk_prob`, divide selected
//      weights by their sum so they total 1.0.
//
// Threadgroup: 32 threads. Each thread covers `ceil(num_experts/32)`
// logits during the softmax and argmax reductions. Shared memory
// holds the post-softmax probability vector for the row.

#include <metal_stdlib>
using namespace metal;

struct RouterParams {
    int num_experts;
    int top_k;
    int norm_topk_prob;  // 0 / 1
};

kernel void moe_router_topk_softmax_f32(
    device const float * logits        [[buffer(0)]],   // [batch, num_experts]
    device       int   * out_ids       [[buffer(1)]],   // [batch, top_k]
    device       float * out_weights   [[buffer(2)]],   // [batch, top_k]
    constant RouterParams & p          [[buffer(3)]],
    threadgroup float  * shmem         [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const int row = tgpig.x;
    const int n_exp = p.num_experts;
    const int top_k = p.top_k;

    threadgroup float * probs = shmem;       // [num_experts]
    // Cooperative load of logits into shmem, finding row max along the way.
    float thread_max = -INFINITY;
    for (int i = tiisg; i < n_exp; i += 32) {
        const float v = logits[row * n_exp + i];
        probs[i] = v;
        thread_max = max(thread_max, v);
    }
    // Reduce max across simdgroup.
    float row_max = simd_max(thread_max);

    // exp(logit - max) and partial sum.
    float thread_sum = 0.0f;
    for (int i = tiisg; i < n_exp; i += 32) {
        const float e = exp(probs[i] - row_max);
        probs[i] = e;
        thread_sum += e;
    }
    float row_sum = simd_sum(thread_sum);
    float inv_sum = 1.0f / row_sum;

    // Normalise.
    for (int i = tiisg; i < n_exp; i += 32) {
        probs[i] *= inv_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Repeated argmax for top-K. k ≤ 32 in all known MoE configs;
    // this is a 32-thread cooperative scan per pick (k passes total).
    // The picked entry is overwritten with -INFINITY so the next pass
    // sees the next-best value.
    //
    // Tie-breaking: simd_max returns the largest float. We follow
    // ferrum's host-side `route` convention of "smaller index wins"
    // by storing `(prob, -index)` pairs and reducing on prob first,
    // index-as-tiebreaker. Here we encode that as: when the max prob
    // is observed, threads with that prob race-write their index into
    // a shmem slot, and only `min(index)` survives (one extra reduce).
    threadgroup float * sel_weights  = (threadgroup float *)(probs + n_exp);
    threadgroup int   * sel_idxs     = (threadgroup int   *)(sel_weights + top_k);
    // One slot for the running renorm sum so it's visible to every
    // thread when the final write phase computes `scale = 1/sum`.
    threadgroup float * renorm_slot  = (threadgroup float *)(sel_idxs + top_k);
    if (tiisg == 0) {
        renorm_slot[0] = 0.0f;
    }

    for (int k = 0; k < top_k; k++) {
        // Find max prob this round.
        float thread_best = -INFINITY;
        int   thread_idx  = -1;
        for (int i = tiisg; i < n_exp; i += 32) {
            const float v = probs[i];
            if (v > thread_best) {
                thread_best = v;
                thread_idx = i;
            }
        }
        const float best = simd_max(thread_best);

        // Race: each thread that holds `best` reports its index; we
        // pick the smallest one as the winner. simd_min over threads
        // that don't match writes INT_MAX so they lose the race.
        int my_idx_for_min = (thread_best == best) ? thread_idx : 0x7fffffff;
        const int win_idx = simd_min(my_idx_for_min);

        if (tiisg == 0) {
            sel_weights[k] = best;
            sel_idxs[k] = win_idx;
            renorm_slot[0] += best;
            // Mask the picked entry from future passes.
            probs[win_idx] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Optional renormalise so the K weights sum to 1.0.
    float scale = 1.0f;
    if (p.norm_topk_prob != 0) {
        // Clamp like llama.cpp's `ggml_clamp` to avoid div-by-zero on
        // degenerate inputs (all logits -INFINITY).
        scale = 1.0f / max(renorm_slot[0], 6.103515625e-5f);
    }

    if (tiisg < top_k) {
        out_ids[row * top_k + tiisg] = sel_idxs[tiisg];
        out_weights[row * top_k + tiisg] = sel_weights[tiisg] * scale;
    }
}
