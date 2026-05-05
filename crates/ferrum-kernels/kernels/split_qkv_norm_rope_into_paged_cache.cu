// Fused: split QKV → per-head QK-norm + RoPE on Q/K → write K/V into paged
// pool slot indexed by `block_table[(pos_offset + token) / block_size]`. Q
// goes to a head-major scratch `q_out`. Mirrors Metal's
// `split_qkv_norm_rope_into_paged_cache` so the unified_forward path on
// CUDA exercises the same kernel surface.
//
// QKV layout (per row): [Q heads | K heads | V heads], each head is `head_dim`
// elements.
//   Q: row[h * hd .. (h+1) * hd] for h in 0..q_heads
//   K: row[q_dim + kh * hd ..] for kh in 0..kv_heads
//   V: row[q_dim + kv_dim + kh * hd ..]
//
// Q output (head-major): q_out[h, tok, d] at q_out_byte_offset.
// K/V output (paged pool): cache_k/v[physical_block, slot, kv_head, d]
//   physical_block = block_table[(pos_offset + tok) / block_size]
//   slot           = (pos_offset + tok) % block_size
//
// `qk_mode` matches `qk_norm_rope.cu`:
//   1 = QK-norm + RoPE (Qwen3)
//   2 = RoPE only       (Llama / Qwen2.5 / Mistral)
// V always uses mode-0 semantics (transpose / copy only).
//
// Launch: grid = (tokens, q_heads + 2 * kv_heads, 1), block = (32, 1, 1).
// One warp per (tok, head). RoPE applied half-dim-pairwise like
// `qk_norm_rope_transpose_f16`.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum_pgd(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

extern "C" __global__ void split_qkv_norm_rope_into_paged_cache_f16(
    const __half* __restrict__ qkv_base,
    const unsigned long long qkv_byte_offset,
    const __half* __restrict__ q_norm_w,        // [head_dim] (used only for mode 1, Q)
    const __half* __restrict__ k_norm_w,        // [head_dim] (used only for mode 1, K)
    const __half* __restrict__ cos_tab,         // [max_seq, head_dim/2]
    const __half* __restrict__ sin_tab,         // [max_seq, head_dim/2]
    __half* __restrict__ q_out_base,
    const unsigned long long q_out_byte_offset,
    __half* __restrict__ cache_k,               // paged pool [num_blocks, block_size, kv_heads, hd]
    __half* __restrict__ cache_v,
    const int* __restrict__ block_table,        // [max_blocks_per_seq]
    const int tokens,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int pos_offset,
    const float eps,
    const int qk_mode,
    const int /*cache_len*/,                    // unused: positions derived from pos_offset+tok
    const int block_size,
    const int /*max_blocks_per_seq*/            // unused at the kernel layer
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
    // is_v = !is_q && !is_k

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
        mode       = 0;        // V: copy only
        nw         = nullptr;
    }

    // Resolve destination pointer.
    __half* dst;
    if (is_q) {
        // Q: token-major [tokens, q_heads, hd] — matches what
        // `paged_varlen_attention.cu` expects (`q[total_q, num_heads, hd]`).
        // The legacy paged decode path uses tokens=1, where token- and
        // head-major collapse to the same flat layout, so this is also
        // a no-op for the Phase 4b batched dispatch.
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
        // Transpose / copy only (V).
        for (int i = lane; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    // RMS-norm scale (mode 1 only).
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

    // RoPE (modes 1 & 2). Pair (i, i+half_d) per the half-rotation layout
    // shared with `qk_norm_rope.cu`.
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
