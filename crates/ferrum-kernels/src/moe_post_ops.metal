// MoE post-ops — fused kernels that collapse the per-(token, expert)
// loop's silu staging and weighted-sum into single dispatches.
//
// Without these kernels, a Qwen3-30B-A3B layer's stacked MoE FFN does:
//   * 8 × (3 copy_slice + 1 silu_mul_split) = 32 dispatches just to
//     stage gate / up into silu output
//   * 8 × (1 copy_slice + 1 scaled_add) = 16 dispatches for the
//     weighted sum
//
// 48 dispatches per layer × 48 layers = 2304 launches per token of
// pure plumbing — enough to dominate decode latency on M1 Max even
// though each individual op is tiny. The kernels below cover the
// same work in 2 dispatches per layer total (one silu, one weighted
// sum), unlocking the win that batching the gemvs alone couldn't.

#include <metal_stdlib>
using namespace metal;

// ── Stacked SiLU·gate for top_k experts ──────────────────────────────
// Inputs:
//   gate : [n_slots, ffn] — gate gemv outputs, one row per selected expert
//   up   : [n_slots, ffn] — up gemv outputs, same layout
//   out  : [n_slots, ffn] — silu(gate[s, i]) * up[s, i] for each slot s, i
// Grid: (ceil(ffn/256), n_slots). Threadgroup: 256 threads.
//
// Single dispatch handles all `n_slots * ffn` elements without any
// per-row staging — replaces the 32 dispatches the per-slot loop
// emitted before.

struct SiluMulStackedParams {
    int ffn;
    int n_slots;
};

kernel void silu_mul_stacked_f32(
    device const float* gate     [[buffer(0)]],
    device const float* up       [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant SiluMulStackedParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpig  [[thread_position_in_threadgroup]])
{
    const uint slot = tgpig.y;
    if (slot >= uint(p.n_slots)) return;

    const uint i = tgpig.x * 256 + tpig.x;
    if (i >= uint(p.ffn)) return;

    const uint off = slot * uint(p.ffn) + i;
    const float g = gate[off];
    const float u = up[off];
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const float s = g / (1.0f + exp(-g));
    out[off] = s * u;
}

// ── Weighted sum across top_k slots ──────────────────────────────────
// Inputs:
//   slots   : [n_slots, hidden] — per-slot down outputs
//   weights : [n_slots]         — router-derived combine weights
//   out     : [hidden]          — out[i] = Σ_s weights[s] * slots[s, i]
// Grid: (ceil(hidden/256), 1). Threadgroup: 256 threads.
//
// One dispatch replaces the 16 (copy + scaled_add) dispatches the
// per-slot loop emitted before.

struct WeightedSumStackedParams {
    int hidden;
    int n_slots;
};

// ── Weighted sum + residual add (fused) ──────────────────────────────
// Inputs:
//   slots    : [n_slots, hidden]
//   weights  : [n_slots]
//   residual : [hidden] — read AND written: residual[i] += Σ_s w[s] * slots[s,i]
// One dispatch replaces (weighted_sum_stacked → moe_out) + (add_inplace
// residual += moe_out): saves 1 dispatch per decode token-layer plus
// the moe_out scratch traffic.

kernel void weighted_sum_residual_stacked_f32(
    device const float* slots    [[buffer(0)]],
    device const float* weights  [[buffer(1)]],
    device       float* residual [[buffer(2)]],
    constant WeightedSumStackedParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpig  [[thread_position_in_threadgroup]])
{
    const uint i = tgpig.x * 256 + tpig.x;
    if (i >= uint(p.hidden)) return;

    float sum = 0.0f;
    for (int s = 0; s < p.n_slots; s++) {
        sum += weights[s] * slots[s * uint(p.hidden) + i];
    }
    residual[i] += sum;
}

kernel void weighted_sum_stacked_f32(
    device const float* slots    [[buffer(0)]],
    device const float* weights  [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant WeightedSumStackedParams& p [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpig  [[thread_position_in_threadgroup]])
{
    const uint i = tgpig.x * 256 + tpig.x;
    if (i >= uint(p.hidden)) return;

    float sum = 0.0f;
    // top_k is small (typically 2-8); unroll-friendly without branching.
    for (int s = 0; s < p.n_slots; s++) {
        sum += weights[s] * slots[s * uint(p.hidden) + i];
    }
    out[i] = sum;
}
