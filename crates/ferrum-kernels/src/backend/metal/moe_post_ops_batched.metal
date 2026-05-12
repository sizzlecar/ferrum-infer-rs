// MoE batched post-ops — variants of the decode-mode `silu_mul_stacked`
// and `weighted_sum_stacked` kernels that handle batch > 1 (prefill).
//
// In the decode path, `silu_mul_stacked` collapses [top_k, ffn] into
// one launch and `weighted_sum_stacked` reduces [top_k, hidden] →
// [hidden] for a single token. Prefill needs the same primitives over
// `[batch, top_k, ffn]` and `[batch, top_k, hidden]` respectively.

#include <metal_stdlib>
using namespace metal;

// ── Stacked SiLU·gate over [batch, top_k, ffn] ──────────────────────
// Output[b, k, i] = silu(gate[b, k, i]) * up[b, k, i].
// Layout matches what mul_mm_id writes: [batch * top_k, ffn] flat.

struct SiluMulBatchedParams {
    int total_pairs;   // = batch * top_k
    int ffn;
};

kernel void silu_mul_batched_f32(
    device const float* gate     [[buffer(0)]],
    device const float* up       [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant SiluMulBatchedParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpig  [[thread_position_in_threadgroup]])
{
    const uint pair = tgpig.y;
    if (pair >= uint(p.total_pairs)) return;

    const uint i = tgpig.x * 256 + tpig.x;
    if (i >= uint(p.ffn)) return;

    const uint off = pair * uint(p.ffn) + i;
    const float g = gate[off];
    const float u = up[off];
    const float s = g / (1.0f + exp(-g));
    out[off] = s * u;
}

// ── Weighted sum over top_k for each batch element ──────────────────
// out[b, h] = Σ_k weights[b, k] * slots[b, k, h]
// for b ∈ [0, batch), h ∈ [0, hidden). Single dispatch covers the whole
// batch, replacing the per-token weighted_sum_stacked that decode used.

struct WeightedSumBatchedParams {
    int batch;
    int top_k;
    int hidden;
};

kernel void weighted_sum_batched_f32(
    device const float* slots    [[buffer(0)]],   // [batch, top_k, hidden]
    device const float* weights  [[buffer(1)]],   // [batch, top_k]
    device       float* out      [[buffer(2)]],   // [batch, hidden]
    constant WeightedSumBatchedParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpig  [[thread_position_in_threadgroup]])
{
    const uint b = tgpig.y;
    if (b >= uint(p.batch)) return;

    const uint h = tgpig.x * 256 + tpig.x;
    if (h >= uint(p.hidden)) return;

    float sum = 0.0f;
    const uint slot_base = b * uint(p.top_k) * uint(p.hidden);
    const uint weight_base = b * uint(p.top_k);
    for (int k = 0; k < p.top_k; k++) {
        sum += weights[weight_base + uint(k)] * slots[slot_base + uint(k) * uint(p.hidden) + h];
    }
    out[b * uint(p.hidden) + h] = sum;
}
