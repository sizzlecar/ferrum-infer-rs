#include <metal_stdlib>
using namespace metal;

// ── Fused QK-Norm + RoPE + Transpose ────────────────────────────────────
// Input:  [tokens, heads, head_dim] (flat GEMM output)
// Output: [heads, tokens, head_dim] (transposed, normed, RoPE-applied)
//
// For V: set apply_norm=0 to skip norm+RoPE (just transpose).
// One threadgroup per (head, token) pair, 32 threads (1 simdgroup).

struct NormRopeParams {
    int tokens;
    int heads;
    int head_dim;
    int half_dim;     // head_dim / 2
    int pos_offset;
    float eps;
    int apply_norm;   // 1 = norm + RoPE, 0 = transpose only
};

kernel void qk_norm_rope_transpose_f32(
    device const float* input    [[buffer(0)]],   // [tokens, heads, hd]
    device const float* weight   [[buffer(1)]],   // [hd] norm weights
    device const float* cos_tab  [[buffer(2)]],   // [max_seq, half_dim]
    device const float* sin_tab  [[buffer(3)]],   // [max_seq, half_dim]
    device       float* output   [[buffer(4)]],   // [heads, tokens, hd]
    constant NormRopeParams& p   [[buffer(5)]],
    uint2  tgpig [[threadgroup_position_in_grid]],  // (token, head)
    uint   tiisg [[thread_index_in_simdgroup]])
{
    const int tok = tgpig.x;
    const int head = tgpig.y;
    if (tok >= p.tokens || head >= p.heads) return;

    const int hd = p.head_dim;
    const int half_d = p.half_dim;

    // Input: [tok * heads * hd + head * hd]
    device const float* src = input + tok * p.heads * hd + head * hd;
    // Output: [head * tokens * hd + tok * hd]
    device float* dst = output + head * p.tokens * hd + tok * hd;

    if (p.apply_norm == 0) {
        // Mode 0: Transpose only (for V)
        for (int i = tiisg; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    if (p.apply_norm == 2) {
        // Mode 2: Transpose + RoPE only, NO norm (vocoder Q/K without QK-norm)
        const int pos = p.pos_offset + tok;
        device const float* cos_row = cos_tab + pos * half_d;
        device const float* sin_row = sin_tab + pos * half_d;
        for (int i = tiisg; i < half_d; i += 32) {
            float x0 = src[i];
            float x1 = src[i + half_d];
            float c = cos_row[i];
            float s = sin_row[i];
            dst[i]          = x0 * c - x1 * s;
            dst[i + half_d] = x1 * c + x0 * s;
        }
        return;
    }

    // Mode 1: Full norm + RoPE
    threadgroup float shared_sum[1];

    // Step 1: Compute RMS norm scale
    float sum_sq = 0.0f;
    for (int i = tiisg; i < hd; i += 32) {
        float v = src[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float scale = 1.0f / sqrt(sum_sq / float(hd) + p.eps);

    // Step 2: Apply norm + RoPE and write output
    const int pos = p.pos_offset + tok;
    device const float* cos_row = cos_tab + pos * half_d;
    device const float* sin_row = sin_tab + pos * half_d;

    for (int i = tiisg; i < half_d; i += 32) {
        float x0 = src[i]        * scale * weight[i];
        float x1 = src[i + half_d] * scale * weight[i + half_d];
        float c = cos_row[i];
        float s = sin_row[i];
        dst[i]        = x0 * c - x1 * s;
        dst[i + half_d] = x1 * c + x0 * s;
    }
}

// ── Untranspose: [heads, tokens, hd] -> [tokens, heads * hd] ───────────
// One thread per element.

struct TransposeOutParams {
    int tokens;
    int heads;
    int head_dim;
};

kernel void transpose_out_f32(
    device const float* input    [[buffer(0)]],   // [heads, tokens, hd]
    device       float* output   [[buffer(1)]],   // [tokens, heads * hd]
    constant TransposeOutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const int total = p.tokens * p.heads * p.head_dim;
    if (tid >= uint(total)) return;

    const int hd = p.head_dim;
    const int nh_hd = p.heads * hd;

    // Decode flat index -> (head, tok, d)
    const int d = tid % hd;
    const int tok = (tid / hd) % p.tokens;
    const int head = tid / (p.tokens * hd);

    // Input: [head * tokens * hd + tok * hd + d]
    // Output: [tok * heads * hd + head * hd + d]
    output[tok * nh_hd + head * hd + d] = input[head * p.tokens * hd + tok * hd + d];
}

// ── KV Cache Append ─────────────────────────────────────────────────────
// Append new K/V data to pre-allocated GPU cache.
// Cache layout: [heads, max_len, hd]
// New data layout: [heads, new_len, hd]

struct KvAppendParams {
    int heads;
    int head_dim;
    int old_len;     // existing seq in cache
    int new_len;     // tokens being appended
    int max_len;     // cache capacity
};

kernel void kv_cache_append_f32(
    device const float* new_data  [[buffer(0)]],  // [heads, new_len, hd]
    device       float* cache     [[buffer(1)]],  // [heads, max_len, hd]
    constant KvAppendParams& p    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const int total = p.heads * p.new_len * p.head_dim;
    if (tid >= uint(total)) return;

    const int hd = p.head_dim;
    const int d = tid % hd;
    const int tok = (tid / hd) % p.new_len;
    const int head = tid / (p.new_len * hd);

    // Source: [head * new_len * hd + tok * hd + d]
    // Dest:   [head * max_len * hd + (old_len + tok) * hd + d]
    cache[head * p.max_len * hd + (p.old_len + tok) * hd + d] = new_data[tid];
}
