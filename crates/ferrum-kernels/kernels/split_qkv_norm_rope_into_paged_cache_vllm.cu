// Variant of `split_qkv_norm_rope_into_paged_cache.cu` that writes K and V
// in vLLM's paged_attention_v2 layout instead of ferrum's block-major
// layout. Used with `vllm_attn/launcher.cu`.
//
// Layout (FP16, x = 16/sizeof(half) = 8):
//   K cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
//     element (block, head, dim, slot) is at half index
//       block * (kv_heads * head_dim * block_size) +
//       head  * (head_dim * block_size) +
//       (dim/x) * (block_size * x) +
//       slot * x +
//       (dim % x)
//   V cache: [num_blocks, num_kv_heads, head_dim, block_size]
//     element (block, head, dim, slot) is at
//       block * (kv_heads * head_dim * block_size) +
//       head  * (head_dim * block_size) +
//       dim   * block_size +
//       slot
//
// Q output layout is unchanged (token-major [m_total, q_heads, hd]).
// Compute (QK-norm + RoPE) is the SAME as the block-major variant.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define VLLM_X 8  // 16 / sizeof(half)

__device__ __forceinline__ float warp_reduce_sum_pgd_vllm(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// Returns the half-index inside the K cache for (block, head, dim, slot).
__device__ __forceinline__ long long k_vllm_offset(
    long long physical_block, int local_head, int dim_i, int slot,
    int hd, int kv_heads, int block_size)
{
    constexpr int X = VLLM_X;
    long long per_block = (long long)kv_heads * hd * block_size;
    long long per_head = (long long)hd * block_size;
    int x_chunk = dim_i / X;
    int x_off = dim_i - x_chunk * X;
    return physical_block * per_block + (long long)local_head * per_head +
           (long long)x_chunk * (block_size * X) +
           (long long)slot * X + x_off;
}

// Returns the half-index inside the V cache for (block, head, dim, slot).
__device__ __forceinline__ long long v_vllm_offset(
    long long physical_block, int local_head, int dim_i, int slot,
    int hd, int kv_heads, int block_size)
{
    long long per_block = (long long)kv_heads * hd * block_size;
    long long per_head = (long long)hd * block_size;
    return physical_block * per_block + (long long)local_head * per_head +
           (long long)dim_i * block_size + slot;
}

// Common write helper — `is_k` selects layout. Returns void; writes value at i.
__device__ __forceinline__ void store_kv_vllm(
    __half* k_pool, __half* v_pool, bool is_k,
    long long physical_block, int local_head, int dim_i, int slot,
    int hd, int kv_heads, int block_size, __half value)
{
    if (is_k) {
        long long off =
            k_vllm_offset(physical_block, local_head, dim_i, slot, hd,
                          kv_heads, block_size);
        k_pool[off] = value;
    } else {
        long long off =
            v_vllm_offset(physical_block, local_head, dim_i, slot, hd,
                          kv_heads, block_size);
        v_pool[off] = value;
    }
}

// ───────── Single-seq entry (prefill / m=1 decode) ─────────

extern "C" __global__ void
split_qkv_norm_rope_into_paged_cache_vllm_f16(
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
    const int /*max_blocks_per_seq*/)
{
    const int tok = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int lane = threadIdx.x;
    const int total_heads = q_heads + 2 * kv_heads;
    if (tok >= tokens || head_idx >= total_heads) return;

    const int hd = head_dim;
    const int half_d = hd / 2;
    const int q_dim = q_heads * hd;
    const int kv_dim = kv_heads * hd;
    const int qkv_stride = q_dim + 2 * kv_dim;

    const __half* qkv_ptr =
        (const __half*)((const char*)qkv_base + qkv_byte_offset);
    const __half* row = qkv_ptr + tok * qkv_stride;

    bool is_q = (head_idx < q_heads);
    bool is_k = (!is_q) && (head_idx < q_heads + kv_heads);

    int local_head;
    int mode;
    const __half* nw;
    const __half* src;

    if (is_q) {
        local_head = head_idx;
        src = row + local_head * hd;
        mode = qk_mode;
        nw = q_norm_w;
    } else if (is_k) {
        local_head = head_idx - q_heads;
        src = row + q_dim + local_head * hd;
        mode = qk_mode;
        nw = k_norm_w;
    } else {
        local_head = head_idx - q_heads - kv_heads;
        src = row + q_dim + kv_dim + local_head * hd;
        mode = 0;
        nw = nullptr;
    }

    // V (mode == 0) — straight copy in vLLM V layout.
    if (mode == 0) {
        const int abs_pos = pos_offset + tok;
        const int logical_block = abs_pos / block_size;
        const int slot = abs_pos % block_size;
        const int physical_block = block_table[logical_block];
        for (int i = lane; i < hd; i += 32) {
            store_kv_vllm(cache_k, cache_v, false, physical_block,
                          local_head, i, slot, hd, kv_heads, block_size,
                          src[i]);
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
        sum_sq = warp_reduce_sum_pgd_vllm(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    const int pos = pos_offset + tok;
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    int abs_pos = 0, logical_block = 0, slot = 0, physical_block = 0;
    if (!is_q) {
        abs_pos = pos_offset + tok;
        logical_block = abs_pos / block_size;
        slot = abs_pos % block_size;
        physical_block = block_table[logical_block];
    }

    for (int i = lane; i < half_d; i += 32) {
        float x0 = __half2float(src[i]);
        float x1 = __half2float(src[i + half_d]);
        if (mode == 1) {
            x0 *= scale * __half2float(nw[i]);
            x1 *= scale * __half2float(nw[i + half_d]);
        }
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);
        __half out_lo = __float2half(x0 * c - x1 * s);
        __half out_hi = __float2half(x1 * c + x0 * s);

        if (is_q) {
            __half* q_out = (__half*)((char*)q_out_base + q_out_byte_offset);
            __half* dst = q_out + (tok * q_heads + local_head) * hd;
            dst[i] = out_lo;
            dst[i + half_d] = out_hi;
        } else {
            store_kv_vllm(cache_k, cache_v, true, physical_block, local_head,
                          i, slot, hd, kv_heads, block_size, out_lo);
            store_kv_vllm(cache_k, cache_v, true, physical_block, local_head,
                          i + half_d, slot, hd, kv_heads, block_size, out_hi);
        }
    }
}

// ───────── Varlen entry (graph-capturable, single launch per layer) ─────────

extern "C" __global__ void
split_qkv_norm_rope_into_paged_cache_varlen_vllm_f16(
    const __half* __restrict__ qkv_base,
    const __half* __restrict__ q_norm_w,
    const __half* __restrict__ k_norm_w,
    const __half* __restrict__ cos_tab,
    const __half* __restrict__ sin_tab,
    __half* __restrict__ q_out,
    __half* __restrict__ cache_k,
    __half* __restrict__ cache_v,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ pos_offsets,
    const int* __restrict__ block_tables,
    const int num_seqs,
    const int m_total,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const float eps,
    const int qk_mode,
    const int block_size,
    const int max_blocks_per_seq)
{
    const int global_tok = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int lane = threadIdx.x;
    const int total_heads = q_heads + 2 * kv_heads;
    if (global_tok >= m_total || head_idx >= total_heads) return;

    // Find which seq this token belongs to (linear scan; num_seqs is small).
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
    const int q_dim = q_heads * hd;
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
        src = row + local_head * hd;
        mode = qk_mode;
        nw = q_norm_w;
    } else if (is_k) {
        local_head = head_idx - q_heads;
        src = row + q_dim + local_head * hd;
        mode = qk_mode;
        nw = k_norm_w;
    } else {
        local_head = head_idx - q_heads - kv_heads;
        src = row + q_dim + kv_dim + local_head * hd;
        mode = 0;
        nw = nullptr;
    }

    // V (mode == 0) — straight copy.
    if (mode == 0) {
        const int abs_pos = pos_offset + local_tok;
        const int logical_block = abs_pos / block_size;
        const int slot = abs_pos % block_size;
        const int physical_block =
            block_tables[seq_idx * max_blocks_per_seq + logical_block];
        for (int i = lane; i < hd; i += 32) {
            store_kv_vllm(cache_k, cache_v, false, physical_block,
                          local_head, i, slot, hd, kv_heads, block_size,
                          src[i]);
        }
        return;
    }

    // Q / K with norm + RoPE.
    float scale = 1.0f;
    if (mode == 1) {
        float sum_sq = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            float v = __half2float(src[i]);
            sum_sq += v * v;
        }
        sum_sq = warp_reduce_sum_pgd_vllm(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    const int pos = pos_offset + local_tok;
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    // RoPE pairwise on (i, i+half_d). For each lane, iterate over the low
    // half; compute the rotated pair; write both halves.
    int abs_pos = 0, logical_block = 0, slot = 0, physical_block = 0;
    if (!is_q) {
        abs_pos = pos_offset + local_tok;
        logical_block = abs_pos / block_size;
        slot = abs_pos % block_size;
        physical_block =
            block_tables[seq_idx * max_blocks_per_seq + logical_block];
    }

    for (int i = lane; i < half_d; i += 32) {
        float x0 = __half2float(src[i]);
        float x1 = __half2float(src[i + half_d]);
        if (mode == 1) {
            x0 *= scale * __half2float(nw[i]);
            x1 *= scale * __half2float(nw[i + half_d]);
        }
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);
        __half out_lo = __float2half(x0 * c - x1 * s);
        __half out_hi = __float2half(x1 * c + x0 * s);

        if (is_q) {
            __half* dst = q_out + (global_tok * q_heads + local_head) * hd;
            dst[i] = out_lo;
            dst[i + half_d] = out_hi;
        } else {
            // K: write in vLLM layout.
            store_kv_vllm(cache_k, cache_v, true, physical_block,
                          local_head, i, slot, hd, kv_heads, block_size,
                          out_lo);
            store_kv_vllm(cache_k, cache_v, true, physical_block,
                          local_head, i + half_d, slot, hd, kv_heads,
                          block_size, out_hi);
        }
    }
}
