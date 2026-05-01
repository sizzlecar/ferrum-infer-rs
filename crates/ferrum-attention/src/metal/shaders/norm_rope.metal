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

// ── Fused Split-QKV + QK-Norm + RoPE + Transpose ─────────────────────────
// Replaces the (split_qkv → 3× qk_norm_rope_transpose) chain with one
// dispatch. Reads the fused-QKV linear output once, applies RMSNorm
// (Q/K) + RoPE (Q/K) + transpose (Q/K/V) in a single pass, and writes
// directly into the head-major Q/K/V scratch buffers used by attention.
//
// Layout:
//   qkv  : [tokens, q_heads*hd + 2*kv_heads*hd] flat
//   q_out: [q_heads,  tokens, hd]
//   k_out: [kv_heads, tokens, hd]
//   v_out: [kv_heads, tokens, hd]
//
// Grid: (tokens, q_heads + 2*kv_heads). Threadgroup: 32 threads (1 simd).
// head_global ∈ [0, q_heads)                              → Q (norm+RoPE)
// head_global ∈ [q_heads, q_heads+kv_heads)               → K (norm+RoPE, k_norm_w)
// head_global ∈ [q_heads+kv_heads, q_heads+2*kv_heads)    → V (transpose only)
//
// qk_mode: 1 = full QK-norm + RoPE (Qwen3); 2 = RoPE only no norm
// (vocoder Q/K). V always passes apply_norm=0.

struct SplitQkvNormRopeParams {
    int tokens;
    int q_heads;
    int kv_heads;
    int head_dim;
    int half_dim;
    int pos_offset;
    float eps;
    int qk_mode;       // 1 = norm+RoPE for Q/K, 2 = RoPE only for Q/K
};

kernel void split_qkv_norm_rope_f32(
    device const float* qkv      [[buffer(0)]],   // [tokens, q_dim+2*kv_dim]
    device const float* q_norm_w [[buffer(1)]],   // [head_dim] (unused if qk_mode==2)
    device const float* k_norm_w [[buffer(2)]],   // [head_dim] (unused if qk_mode==2)
    device const float* cos_tab  [[buffer(3)]],
    device const float* sin_tab  [[buffer(4)]],
    device       float* q_out    [[buffer(5)]],   // [q_heads,  tokens, hd]
    device       float* k_out    [[buffer(6)]],   // [kv_heads, tokens, hd]
    device       float* v_out    [[buffer(7)]],   // [kv_heads, tokens, hd]
    constant SplitQkvNormRopeParams& p [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],   // (token, head_global)
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const int tok = tgpig.x;
    const int head_g = tgpig.y;
    if (tok >= p.tokens) return;

    const int hd = p.head_dim;
    const int half_d = p.half_dim;
    const int q_dim = p.q_heads * hd;
    const int kv_dim = p.kv_heads * hd;
    const int qkv_stride = q_dim + 2 * kv_dim;

    // Region selection: 0 = Q, 1 = K, 2 = V.
    int region;
    int local_head;
    int src_off;
    if (head_g < uint(p.q_heads)) {
        region = 0;
        local_head = head_g;
        src_off = tok * qkv_stride + local_head * hd;
    } else if (head_g < uint(p.q_heads + p.kv_heads)) {
        region = 1;
        local_head = head_g - p.q_heads;
        src_off = tok * qkv_stride + q_dim + local_head * hd;
    } else {
        region = 2;
        local_head = head_g - p.q_heads - p.kv_heads;
        src_off = tok * qkv_stride + q_dim + kv_dim + local_head * hd;
    }

    device const float* src = qkv + src_off;
    // Pick destination buffer + per-region tokens stride (head_major).
    device float* dst_base = (region == 0) ? q_out
                            : (region == 1) ? k_out : v_out;
    device float* dst = dst_base + local_head * p.tokens * hd + tok * hd;

    if (region == 2) {
        // V: transpose only. Each thread handles HD / 32 elements.
        for (int i = tiisg; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    // Q or K: optional norm + RoPE.
    const bool apply_norm = (p.qk_mode == 1);
    float scale = 1.0f;
    device const float* norm_w = (region == 0) ? q_norm_w : k_norm_w;
    if (apply_norm) {
        float sum_sq = 0.0f;
        for (int i = tiisg; i < hd; i += 32) {
            float v = src[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        scale = 1.0f / sqrt(sum_sq / float(hd) + p.eps);
    }

    const int pos = p.pos_offset + tok;
    device const float* cos_row = cos_tab + pos * half_d;
    device const float* sin_row = sin_tab + pos * half_d;

    for (int i = tiisg; i < half_d; i += 32) {
        float w0 = apply_norm ? (scale * norm_w[i])          : 1.0f;
        float w1 = apply_norm ? (scale * norm_w[i + half_d]) : 1.0f;
        float x0 = src[i]          * w0;
        float x1 = src[i + half_d] * w1;
        float c = cos_row[i];
        float s = sin_row[i];
        dst[i]          = x0 * c - x1 * s;
        dst[i + half_d] = x1 * c + x0 * s;
    }
}

// ── Variant: write K/V straight into the KV cache ────────────────────────
// Same fused split-QKV + QK-Norm + RoPE + transpose, but K and V land
// directly in the pre-allocated head-major KV cache at position
// (cache_len + tok) instead of in a separate per-token scratch buffer.
// Eliminates the trailing `kv_cache_append_head_major` dispatch on the
// decode path (one extra dispatch saved per layer × 48 layers).
//
// q_out stays the per-token head-major scratch since flash_attention
// reads it as the query.
//
// Cache layout: [kv_heads, cache_capacity, hd]; only the slice
// [kv_heads, cache_len .. cache_len + tokens, hd] is written.

struct SplitQkvNormRopeKvcParams {
    int tokens;
    int q_heads;
    int kv_heads;
    int head_dim;
    int half_dim;
    int pos_offset;
    float eps;
    int qk_mode;
    int cache_len;       // existing seq length in cache (write offset)
    int cache_capacity;  // cache stride along token axis
};

kernel void split_qkv_norm_rope_kvc_f32(
    device const float* qkv      [[buffer(0)]],
    device const float* q_norm_w [[buffer(1)]],
    device const float* k_norm_w [[buffer(2)]],
    device const float* cos_tab  [[buffer(3)]],
    device const float* sin_tab  [[buffer(4)]],
    device       float* q_out    [[buffer(5)]],   // [q_heads, tokens, hd]
    device       float* cache_k  [[buffer(6)]],   // [kv_heads, cache_capacity, hd]
    device       float* cache_v  [[buffer(7)]],   // [kv_heads, cache_capacity, hd]
    constant SplitQkvNormRopeKvcParams& p [[buffer(8)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const int tok = tgpig.x;
    const int head_g = tgpig.y;
    if (tok >= p.tokens) return;

    const int hd = p.head_dim;
    const int half_d = p.half_dim;
    const int q_dim = p.q_heads * hd;
    const int kv_dim = p.kv_heads * hd;
    const int qkv_stride = q_dim + 2 * kv_dim;

    int region;
    int local_head;
    int src_off;
    if (head_g < uint(p.q_heads)) {
        region = 0;
        local_head = head_g;
        src_off = tok * qkv_stride + local_head * hd;
    } else if (head_g < uint(p.q_heads + p.kv_heads)) {
        region = 1;
        local_head = head_g - p.q_heads;
        src_off = tok * qkv_stride + q_dim + local_head * hd;
    } else {
        region = 2;
        local_head = head_g - p.q_heads - p.kv_heads;
        src_off = tok * qkv_stride + q_dim + kv_dim + local_head * hd;
    }

    device const float* src = qkv + src_off;
    // Q stays in per-token head-major scratch; K/V go straight into the
    // cache at slot `cache_len + tok`.
    device float* dst;
    if (region == 0) {
        dst = q_out + local_head * p.tokens * hd + tok * hd;
    } else if (region == 1) {
        dst = cache_k + local_head * p.cache_capacity * hd
                      + (p.cache_len + tok) * hd;
    } else {
        dst = cache_v + local_head * p.cache_capacity * hd
                      + (p.cache_len + tok) * hd;
    }

    if (region == 2) {
        for (int i = tiisg; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    const bool apply_norm = (p.qk_mode == 1);
    float scale = 1.0f;
    device const float* norm_w = (region == 0) ? q_norm_w : k_norm_w;
    if (apply_norm) {
        float sum_sq = 0.0f;
        for (int i = tiisg; i < hd; i += 32) {
            float v = src[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        scale = 1.0f / sqrt(sum_sq / float(hd) + p.eps);
    }

    const int pos = p.pos_offset + tok;
    device const float* cos_row = cos_tab + pos * half_d;
    device const float* sin_row = sin_tab + pos * half_d;

    for (int i = tiisg; i < half_d; i += 32) {
        float w0 = apply_norm ? (scale * norm_w[i])          : 1.0f;
        float w1 = apply_norm ? (scale * norm_w[i + half_d]) : 1.0f;
        float x0 = src[i]          * w0;
        float x1 = src[i + half_d] * w1;
        float c = cos_row[i];
        float s = sin_row[i];
        dst[i]          = x0 * c - x1 * s;
        dst[i + half_d] = x1 * c + x0 * s;
    }
}

// ── Paged-KV variant of split_qkv_norm_rope_kvc_f32 ──────────────────
//
// Same fused split + qk-norm + RoPE + cache-write as the contiguous
// variant above. The only change is K/V's `dst` computation: instead
// of writing into `cache_{k,v}[kv_head][cache_len + tok][hd]`, we
// resolve the destination via the block-table:
//
//   global_slot     = cache_len + tok
//   logical_block   = global_slot / block_size
//   slot_in_block   = global_slot % block_size
//   physical_block  = block_table[logical_block]
//   dst             = cache_{k,v}[physical_block][kv_head][slot_in_block][hd]
//
// Cache layout: [num_blocks, kv_heads, block_size, head_dim] (matches
// flash_attn_decode_paged_f32 from the sibling PR). Q stays head-major
// — only K/V live in the paged pool.

struct SplitQkvNormRopePagedKvcParams {
    int tokens;
    int q_heads;
    int kv_heads;
    int head_dim;
    int half_dim;
    int pos_offset;
    float eps;
    int qk_mode;
    int cache_len;              // existing seq length in cache (slot of first new token)
    int block_size;             // KV positions per physical block (16 typical)
    int max_num_blocks_per_seq; // block_table row stride (single-seq case for now)
};

kernel void split_qkv_norm_rope_paged_kvc_f32(
    device const float*    qkv          [[buffer(0)]],
    device const float*    q_norm_w     [[buffer(1)]],
    device const float*    k_norm_w     [[buffer(2)]],
    device const float*    cos_tab      [[buffer(3)]],
    device const float*    sin_tab      [[buffer(4)]],
    device       float*    q_out        [[buffer(5)]],   // [q_heads, tokens, hd]
    device       float*    cache_k      [[buffer(6)]],   // [num_blocks, kv_heads, block_size, hd]
    device       float*    cache_v      [[buffer(7)]],   // [num_blocks, kv_heads, block_size, hd]
    device const uint32_t* block_table  [[buffer(8)]],   // [max_blocks_per_seq] (single seq)
    constant SplitQkvNormRopePagedKvcParams& p [[buffer(9)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const int tok = int(tgpig.x);
    const int head_g = int(tgpig.y);
    if (tok >= p.tokens) return;

    const int hd = p.head_dim;
    const int half_d = p.half_dim;
    const int q_dim = p.q_heads * hd;
    const int kv_dim = p.kv_heads * hd;
    const int qkv_stride = q_dim + 2 * kv_dim;

    int region;
    int local_head;
    int src_off;
    if (head_g < p.q_heads) {
        region = 0;
        local_head = head_g;
        src_off = tok * qkv_stride + local_head * hd;
    } else if (head_g < p.q_heads + p.kv_heads) {
        region = 1;
        local_head = head_g - p.q_heads;
        src_off = tok * qkv_stride + q_dim + local_head * hd;
    } else {
        region = 2;
        local_head = head_g - p.q_heads - p.kv_heads;
        src_off = tok * qkv_stride + q_dim + kv_dim + local_head * hd;
    }

    device const float* src = qkv + src_off;

    // Compute dst pointer.
    // Q (region 0): head-major scratch as before.
    // K/V (region 1/2): paged layout — index via block_table.
    device float* dst;
    if (region == 0) {
        dst = q_out + local_head * p.tokens * hd + tok * hd;
    } else {
        const int global_slot     = p.cache_len + tok;
        const int logical_block   = global_slot / p.block_size;
        const int slot_in_block   = global_slot % p.block_size;
        const uint32_t physical_block = block_table[logical_block];
        const int slot_offset = int(physical_block) * p.kv_heads * p.block_size * hd
                              + local_head * p.block_size * hd
                              + slot_in_block * hd;
        dst = (region == 1 ? cache_k : cache_v) + slot_offset;
    }

    // V: pure copy.
    if (region == 2) {
        for (int i = int(tiisg); i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    // Q / K: optional QK-norm, then RoPE.
    const bool apply_norm = (p.qk_mode == 1);
    float scale = 1.0f;
    device const float* norm_w = (region == 0) ? q_norm_w : k_norm_w;
    if (apply_norm) {
        float sum_sq = 0.0f;
        for (int i = int(tiisg); i < hd; i += 32) {
            float v = src[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        scale = 1.0f / sqrt(sum_sq / float(hd) + p.eps);
    }

    const int pos = p.pos_offset + tok;
    device const float* cos_row = cos_tab + pos * half_d;
    device const float* sin_row = sin_tab + pos * half_d;

    for (int i = int(tiisg); i < half_d; i += 32) {
        float w0 = apply_norm ? (scale * norm_w[i])          : 1.0f;
        float w1 = apply_norm ? (scale * norm_w[i + half_d]) : 1.0f;
        float x0 = src[i]          * w0;
        float x1 = src[i + half_d] * w1;
        float c = cos_row[i];
        float s = sin_row[i];
        dst[i]          = x0 * c - x1 * s;
        dst[i + half_d] = x1 * c + x0 * s;
    }
}
