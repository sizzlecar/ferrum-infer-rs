// Two related fused kernels for the paged-KV write path.
//
// 1. `split_qkv_norm_rope_into_paged_cache_f16` — per-item entry, kept
//    for the legacy Phase 4b batched-decode dispatch. Called once per
//    sequence with that sequence's pos_offset + block_table directly
//    as kernel args. NOT graph-capturable: the scalars change every
//    iter, so a captured launch would always overwrite the same KV
//    slot.
//
// 2. `split_qkv_norm_rope_into_paged_cache_varlen_f16` — varlen entry.
//    Single launch covering ALL sequences in the batch. Reads
//    `pos_offsets[seq]`, `cu_seqlens_q[seq]`, and the per-seq
//    block_table from device buffers, so a captured launch picks up
//    the freshly-written buffer contents from the next iter (the
//    write_u32's that populate them are sync'd outside capture).
//    Used by `unified_forward_internal` once we've validated the
//    capture path.
//
// Both share the same RoPE / QK-norm semantics as `qk_norm_rope.cu`:
//   qk_mode == 0 → copy only (V)
//   qk_mode == 1 → QK-norm + RoPE (Qwen3)
//   qk_mode == 2 → RoPE only        (Llama / Mistral)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum_pgd(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// ───────── Per-item entry (legacy Phase 4b) ─────────

extern "C" __global__ void split_qkv_norm_rope_into_paged_cache_f16(
    const __half* __restrict__ qkv_base,
    const unsigned long long qkv_byte_offset,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const __half* __restrict__ cos_tab,
    const __half* __restrict__ sin_tab,
    __half* __restrict__ q_out_base,
    const unsigned long long q_out_byte_offset,
    __half* __restrict__ cache_k,
    __half* __restrict__ cache_v,
    const int* __restrict__ block_table,
    const int tokens,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int pos_offset,
    const float eps,
    const int qk_mode,
    const int /*cache_len*/,
    const int block_size,
    const int /*max_blocks_per_seq*/
) {
    const int tok      = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int lane     = threadIdx.x;
    const int total_heads = q_heads + 2 * kv_heads;
    if (tok >= tokens || head_idx >= total_heads) return;

    const int hd = head_dim;
    const int half_d = hd / 2;
    const int q_dim  = q_heads  * hd;
    const int kv_dim = kv_heads * hd;
    const int qkv_stride = q_dim + 2 * kv_dim;

    const __half* qkv_ptr = (const __half*)((const char*)qkv_base + qkv_byte_offset);
    const __half* row     = qkv_ptr + tok * qkv_stride;

    bool is_q = (head_idx < q_heads);
    bool is_k = (!is_q) && (head_idx < q_heads + kv_heads);

    int local_head;
    int mode;
    const __half* nw;
    const __half* src;

    if (is_q) {
        local_head = head_idx;
        src        = row + local_head * hd;
        mode       = qk_mode;
        nw         = q_norm_w;
    } else if (is_k) {
        local_head = head_idx - q_heads;
        src        = row + q_dim + local_head * hd;
        mode       = qk_mode;
        nw         = k_norm_w;
    } else {
        local_head = head_idx - q_heads - kv_heads;
        src        = row + q_dim + kv_dim + local_head * hd;
        mode       = 0;
        nw         = nullptr;
    }

    __half* dst;
    if (is_q) {
        // Q output: token-major [tokens, q_heads, hd] for paged_varlen.
        __half* q_out = (__half*)((char*)q_out_base + q_out_byte_offset);
        dst = q_out + (tok * q_heads + local_head) * hd;
    } else {
        const int abs_pos = pos_offset + tok;
        const int logical_block = abs_pos / block_size;
        const int slot          = abs_pos % block_size;
        const int physical_block = block_table[logical_block];
        const int kv_stride    = kv_heads * hd;
        const int block_stride = block_size * kv_stride;
        __half* pool = is_k ? cache_k : cache_v;
        dst = pool + (long long)physical_block * block_stride
                   + slot * kv_stride
                   + local_head * hd;
    }

    if (mode == 0) {
        for (int i = lane; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    float scale = 1.0f;
    if (mode == 1) {
        float sum_sq = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            float v = __half2float(src[i]);
            sum_sq += v * v;
        }
        sum_sq = warp_reduce_sum_pgd(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    const int pos = pos_offset + tok;
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    for (int i = lane; i < half_d; i += 32) {
        float x0 = __half2float(src[i]);
        float x1 = __half2float(src[i + half_d]);
        if (mode == 1) {
            x0 *= scale * __half2float(nw[i]);
            x1 *= scale * __half2float(nw[i + half_d]);
        }
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);
        dst[i]          = __float2half(x0 * c - x1 * s);
        dst[i + half_d] = __float2half(x1 * c + x0 * s);
    }
}

// ───────── Varlen entry (graph-capturable, single launch per layer) ─────────

extern "C" __global__ void split_qkv_norm_rope_into_paged_cache_varlen_f16(
    const __half* __restrict__ qkv_base,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const __half* __restrict__ cos_tab,
    const __half* __restrict__ sin_tab,
    __half* __restrict__ q_out,
    __half* __restrict__ cache_k,
    __half* __restrict__ cache_v,
    const int* __restrict__ cu_seqlens_q,    // [num_seqs + 1]
    const int* __restrict__ pos_offsets,     // [num_seqs]
    const int* __restrict__ block_tables,    // [num_seqs * max_blocks_per_seq]
    const int num_seqs,
    const int m_total,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float eps,
    const int qk_mode,
    const int block_size,
    const int max_blocks_per_seq
) {
    const int global_tok = blockIdx.x;
    const int head_idx   = blockIdx.y;
    const int lane       = threadIdx.x;
    const int total_heads = q_heads + 2 * kv_heads;
    if (global_tok >= m_total || head_idx >= total_heads) return;

    // Locate seq_idx by linear scan (num_seqs is small at decode-time).
    int seq_idx = 0;
    for (int s = 0; s < num_seqs; s++) {
        if (global_tok < cu_seqlens_q[s + 1]) {
            seq_idx = s;
            break;
        }
    }
    const int seq_start = cu_seqlens_q[seq_idx];
    const int local_tok = global_tok - seq_start;
    const int pos_offset = pos_offsets[seq_idx];

    const int hd = head_dim;
    const int half_d = hd / 2;
    const int q_dim  = q_heads  * hd;
    const int kv_dim = kv_heads * hd;
    const int qkv_stride = q_dim + 2 * kv_dim;

    const __half* row = qkv_base + global_tok * qkv_stride;

    bool is_q = (head_idx < q_heads);
    bool is_k = (!is_q) && (head_idx < q_heads + kv_heads);

    int local_head;
    int mode;
    const __half* nw;
    const __half* src;

    if (is_q) {
        local_head = head_idx;
        src        = row + local_head * hd;
        mode       = qk_mode;
        nw         = q_norm_w;
    } else if (is_k) {
        local_head = head_idx - q_heads;
        src        = row + q_dim + local_head * hd;
        mode       = qk_mode;
        nw         = k_norm_w;
    } else {
        local_head = head_idx - q_heads - kv_heads;
        src        = row + q_dim + kv_dim + local_head * hd;
        mode       = 0;
        nw         = nullptr;
    }

    __half* dst;
    if (is_q) {
        // Q output: token-major [m_total, q_heads, hd] — matches paged_varlen.
        dst = q_out + (global_tok * q_heads + local_head) * hd;
    } else {
        const int abs_pos = pos_offset + local_tok;
        const int logical_block = abs_pos / block_size;
        const int slot          = abs_pos % block_size;
        const int physical_block =
            block_tables[seq_idx * max_blocks_per_seq + logical_block];
        const int kv_stride    = kv_heads * hd;
        const int block_stride = block_size * kv_stride;
        __half* pool = is_k ? cache_k : cache_v;
        dst = pool + (long long)physical_block * block_stride
                   + slot * kv_stride
                   + local_head * hd;
    }

    if (mode == 0) {
        for (int i = lane; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    float scale = 1.0f;
    if (mode == 1) {
        float sum_sq = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            float v = __half2float(src[i]);
            sum_sq += v * v;
        }
        sum_sq = warp_reduce_sum_pgd(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    const int pos = pos_offset + local_tok;
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    for (int i = lane; i < half_d; i += 32) {
        float x0 = __half2float(src[i]);
        float x1 = __half2float(src[i + half_d]);
        if (mode == 1) {
            x0 *= scale * __half2float(nw[i]);
            x1 *= scale * __half2float(nw[i + half_d]);
        }
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);
        dst[i]          = __float2half(x0 * c - x1 * s);
        dst[i + half_d] = __float2half(x1 * c + x0 * s);
    }
}
