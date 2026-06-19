// Qwen3.5 full-attention paged-KV writer.
//
// Qwen3.5 materializes separate q_proj/k_proj/v_proj tensors and can store an
// attention gate in the Q projection row. The older paged writer consumes a
// fused [Q|K|V] row, so this kernel is the separated-projection equivalent:
// it applies Q/K norm + partial RoPE, writes K/V into a paged pool through
// block_tables, and emits token-major Q for paged attention.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum_q35(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

extern "C" __global__ void qwen35_split_qkv_norm_rope_into_paged_cache_varlen_f16(
    const __half* __restrict__ query_raw,
    const __half* __restrict__ key_raw,
    const __half* __restrict__ value_raw,
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
    const int total_q_tokens,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int rope_dim,
    const int q_proj_stride,
    const int q_head_stride,
    const int kv_proj_stride,
    const float eps,
    const int qk_mode,
    const int block_size,
    const int max_blocks_per_seq
) {
    const int tok = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int lane = threadIdx.x;
    const int total_heads = q_heads + 2 * kv_heads;
    if (tok >= total_q_tokens || head_idx >= total_heads) return;

    int seq = 0;
    while (seq + 1 < num_seqs && tok >= cu_seqlens_q[seq + 1]) {
        ++seq;
    }
    const int local_tok = tok - cu_seqlens_q[seq];
    const int abs_pos = pos_offsets[seq] + local_tok;

    const bool is_q = head_idx < q_heads;
    const bool is_k = !is_q && head_idx < q_heads + kv_heads;
    const int local_head = is_q ? head_idx : (is_k ? head_idx - q_heads : head_idx - q_heads - kv_heads);

    const __half* src = nullptr;
    const __half* norm_w = nullptr;
    int mode = 0;
    __half* dst = nullptr;

    if (is_q) {
        src = query_raw + tok * q_proj_stride + local_head * q_head_stride;
        norm_w = q_norm_w;
        mode = qk_mode;
        dst = q_out + (tok * q_heads + local_head) * head_dim;
    } else {
        const int logical_block = abs_pos / block_size;
        const int slot = abs_pos % block_size;
        const int physical_block = block_tables[seq * max_blocks_per_seq + logical_block];
        const int kv_stride = kv_heads * head_dim;
        const long long block_stride = (long long)block_size * kv_stride;
        __half* pool = is_k ? cache_k : cache_v;
        dst = pool + (long long)physical_block * block_stride
                   + slot * kv_stride
                   + local_head * head_dim;
        if (is_k) {
            src = key_raw + tok * kv_proj_stride + local_head * head_dim;
            norm_w = k_norm_w;
            mode = qk_mode;
        } else {
            src = value_raw + tok * kv_proj_stride + local_head * head_dim;
            mode = 0;
        }
    }

    if (mode == 0) {
        for (int i = lane; i < head_dim; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    const bool do_norm = (mode == 1 || mode == 3);
    float scale = 1.0f;
    if (do_norm) {
        float sum_sq = 0.0f;
        for (int i = lane; i < head_dim; i += 32) {
            const float x = __half2float(src[i]);
            sum_sq += x * x;
        }
        sum_sq = warp_reduce_sum_q35(sum_sq);
        scale = rsqrtf(sum_sq / (float)head_dim + eps);
    }

    const int rd = rope_dim;
    const int half_rd = rd / 2;
    const __half* cos_row = cos_tab + abs_pos * half_rd;
    const __half* sin_row = sin_tab + abs_pos * half_rd;

    if (mode == 3) {
        for (int pair = lane; pair < half_rd; pair += 32) {
            const int j = 2 * pair;
            float x0 = __half2float(src[j]);
            float x1 = __half2float(src[j + 1]);
            if (do_norm) {
                x0 *= scale * __half2float(norm_w[j]);
                x1 *= scale * __half2float(norm_w[j + 1]);
            }
            const float c = __half2float(cos_row[pair]);
            const float s = __half2float(sin_row[pair]);
            dst[j] = __float2half(x0 * c - x1 * s);
            dst[j + 1] = __float2half(x1 * c + x0 * s);
        }
        for (int i = rd + lane; i < head_dim; i += 32) {
            float x = __half2float(src[i]);
            if (do_norm) {
                x *= scale * __half2float(norm_w[i]);
            }
            dst[i] = __float2half(x);
        }
        return;
    }

    for (int i = lane; i < half_rd; i += 32) {
        float x0 = __half2float(src[i]);
        float x1 = __half2float(src[i + half_rd]);
        if (do_norm) {
            x0 *= scale * __half2float(norm_w[i]);
            x1 *= scale * __half2float(norm_w[i + half_rd]);
        }
        const float c = __half2float(cos_row[i]);
        const float s = __half2float(sin_row[i]);
        dst[i] = __float2half(x0 * c - x1 * s);
        dst[i + half_rd] = __float2half(x1 * c + x0 * s);
    }
    for (int i = rd + lane; i < head_dim; i += 32) {
        float x = __half2float(src[i]);
        if (do_norm) {
            x *= scale * __half2float(norm_w[i]);
        }
        dst[i] = __float2half(x);
    }
}
