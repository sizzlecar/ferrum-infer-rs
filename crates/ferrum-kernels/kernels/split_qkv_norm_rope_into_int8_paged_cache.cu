// Fused INT8 KV write path — Dim 5 PR C2 perf.
//
// Replaces the per-token 4-launch chain
//   `split_qkv` + `qk_norm_rope`×3 + `int8_kv_cache_append`
// with a single launch that fuses split-QKV → QK-norm → RoPE →
// quantize → write into the paged INT8 pool.
//
// One thread block per (token, head). For Q heads the kernel writes
// FP16 head-major scratch (same as the FP16 path); for K/V heads it
// applies QK-norm + RoPE in registers, warp-reduces `max(|x|)` to
// derive the per-(token, kv_head) symmetric scale, quantizes, then
// writes INT8 + the FP16 scale into the paged pool.
//
// Grid:  (tokens, q_heads + 2 * kv_heads, 1)
// Block: (32, 1, 1)  — one warp per (token, head).
//
// Block pool layouts (mirroring `int8_paged_decode_attention.cu`):
//   k_pool / v_pool            : int8  [max_blocks * block_size * num_kv_heads * head_dim]
//   k_scales_pool / v_scales   : __half [max_blocks * block_size * num_kv_heads]

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum_int8pgd(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max_int8pgd(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

extern "C" __global__ void split_qkv_norm_rope_into_int8_paged_cache_f16(
    const __half* __restrict__ qkv,                  // [tokens, q_dim + 2*kv_dim]
    const __half* __restrict__ q_norm_w,             // [head_dim]
    const __half* __restrict__ k_norm_w,             // [head_dim]
    const __half* __restrict__ cos_tab,              // [max_pos, head_dim/2]
    const __half* __restrict__ sin_tab,              // [max_pos, head_dim/2]
    __half* __restrict__ q_out,                      // FP16 [tokens, q_heads, head_dim] head-major
    int8_t* __restrict__ cache_k,                    // INT8 paged pool
    int8_t* __restrict__ cache_v,                    // INT8 paged pool
    __half* __restrict__ cache_k_scales,             // FP16 paged scales
    __half* __restrict__ cache_v_scales,             // FP16 paged scales
    const int* __restrict__ block_table,             // i32 [max_blocks_per_seq]
    const int tokens,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int pos_offset,
    const float eps,
    const int qk_mode,                                // 0=copy, 1=norm+RoPE, 2=RoPE only
    const int cache_len_before,
    const int block_size
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

    const __half* row = qkv + tok * qkv_stride;

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

    // ──────────── Q path: FP16 head-major scratch ────────────
    if (is_q) {
        __half* dst = q_out + (tok * q_heads + local_head) * hd;

        if (mode == 0) {
            for (int i = lane; i < hd; i += 32) {
                dst[i] = src[i];
            }
            return;
        }

        float scale_norm = 1.0f;
        if (mode == 1) {
            float sum_sq = 0.0f;
            for (int i = lane; i < hd; i += 32) {
                float v = __half2float(src[i]);
                sum_sq += v * v;
            }
            sum_sq = warp_reduce_sum_int8pgd(sum_sq);
            scale_norm = rsqrtf(sum_sq / (float)hd + eps);
        }

        const int pos = pos_offset + cache_len_before + tok;
        const __half* cos_row = cos_tab + pos * half_d;
        const __half* sin_row = sin_tab + pos * half_d;

        for (int i = lane; i < half_d; i += 32) {
            float x0 = __half2float(src[i]);
            float x1 = __half2float(src[i + half_d]);
            if (mode == 1) {
                x0 *= scale_norm * __half2float(nw[i]);
                x1 *= scale_norm * __half2float(nw[i + half_d]);
            }
            float c = __half2float(cos_row[i]);
            float s = __half2float(sin_row[i]);
            dst[i]          = __float2half(x0 * c - x1 * s);
            dst[i + half_d] = __float2half(x1 * c + x0 * s);
        }
        return;
    }

    // ──────────── K / V path: norm + RoPE → quantize → INT8 paged ────────────
    // Address translation matches int8_paged_decode_attention.cu:
    //   global_pos = cache_len_before + tok
    //   logical_block = global_pos / block_size
    //   slot          = global_pos % block_size
    //   physical      = block_table[logical_block]
    //   data offset   = physical * block_size * num_kv_heads * hd
    //                 + slot * num_kv_heads * hd
    //                 + local_head * hd
    //   scale offset  = physical * block_size * num_kv_heads
    //                 + slot * num_kv_heads
    //                 + local_head
    const int global_pos     = cache_len_before + tok;
    const int logical_block  = global_pos / block_size;
    const int slot           = global_pos % block_size;
    const int physical_block = block_table[logical_block];

    const int kv_stride         = kv_heads * hd;
    const int block_data_stride = block_size * kv_stride;
    const int block_scale_stride = block_size * kv_heads;

    int8_t* dst_pool       = is_k ? cache_k : cache_v;
    __half* dst_scale_pool = is_k ? cache_k_scales : cache_v_scales;

    int8_t* dst_data = dst_pool
                     + (long long)physical_block * block_data_stride
                     + slot * kv_stride
                     + local_head * hd;
    __half* dst_scale = dst_scale_pool
                      + (long long)physical_block * block_scale_stride
                      + slot * kv_heads
                      + local_head;

    // Norm + RoPE in registers: each lane holds head_dim/32 elements
    // for the (token, head) pair. We need to keep the post-RoPE values
    // in registers across the warp-max reduction so we can quantize
    // them without re-computing.
    //
    // 128 head_dim is the common case (Qwen3-0.6B uses head_dim=128).
    // Allocate a fixed-size register array; head_dim ≤ 128 → 4 elements
    // per lane is enough.
    constexpr int MAX_PER_LANE = 4;  // head_dim / 32 ≤ 4 for hd ≤ 128
    float vals[MAX_PER_LANE * 2];  // hold both halves: vals[i] for i, vals[i+half] for i+half_d
    // Initialize so unused slots don't pollute the max reduce.
    #pragma unroll
    for (int i = 0; i < MAX_PER_LANE * 2; i++) {
        vals[i] = 0.0f;
    }

    if (mode == 0) {
        // V copy: no norm, no RoPE — just read FP16 then quantize.
        // We still need to write per-(token,head) scale.
        int local_idx = 0;
        for (int i = lane; i < hd; i += 32) {
            vals[local_idx++] = __half2float(src[i]);
        }
    } else {
        float scale_norm = 1.0f;
        if (mode == 1) {
            float sum_sq = 0.0f;
            for (int i = lane; i < hd; i += 32) {
                float v = __half2float(src[i]);
                sum_sq += v * v;
            }
            sum_sq = warp_reduce_sum_int8pgd(sum_sq);
            scale_norm = rsqrtf(sum_sq / (float)hd + eps);
        }

        const int pos = pos_offset + cache_len_before + tok;
        const __half* cos_row = cos_tab + pos * half_d;
        const __half* sin_row = sin_tab + pos * half_d;

        // RoPE: process pairs (i, i+half_d). vals[2*idx] holds the
        // post-RoPE value for offset i; vals[2*idx+1] for i+half_d.
        int idx = 0;
        for (int i = lane; i < half_d; i += 32) {
            float x0 = __half2float(src[i]);
            float x1 = __half2float(src[i + half_d]);
            if (mode == 1) {
                x0 *= scale_norm * __half2float(nw[i]);
                x1 *= scale_norm * __half2float(nw[i + half_d]);
            }
            float c = __half2float(cos_row[i]);
            float s = __half2float(sin_row[i]);
            vals[2 * idx]     = x0 * c - x1 * s;       // dst[i]
            vals[2 * idx + 1] = x1 * c + x0 * s;       // dst[i + half_d]
            idx++;
        }
    }

    // Compute max(|x|) over the warp.
    float my_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < MAX_PER_LANE * 2; i++) {
        my_max = fmaxf(my_max, fabsf(vals[i]));
    }
    float warp_max = warp_reduce_max_int8pgd(my_max);
    // Avoid scale=0 (would make dequant always 0); clamp to 1e-8.
    warp_max = fmaxf(warp_max, 1e-8f);
    float scale_quant = warp_max / 127.0f;
    float inv_scale_quant = 1.0f / scale_quant;

    // Lane 0 writes the per-(token, head) scale.
    if (lane == 0) {
        *dst_scale = __float2half(scale_quant);
    }

    // Quantize + write INT8.
    if (mode == 0) {
        int local_idx = 0;
        for (int i = lane; i < hd; i += 32) {
            float v = vals[local_idx++];
            int q = __float2int_rn(fmaxf(-127.0f, fminf(127.0f, v * inv_scale_quant)));
            dst_data[i] = (int8_t)q;
        }
    } else {
        int idx = 0;
        for (int i = lane; i < half_d; i += 32) {
            int q0 = __float2int_rn(fmaxf(-127.0f, fminf(127.0f, vals[2 * idx]     * inv_scale_quant)));
            int q1 = __float2int_rn(fmaxf(-127.0f, fminf(127.0f, vals[2 * idx + 1] * inv_scale_quant)));
            dst_data[i]          = (int8_t)q0;
            dst_data[i + half_d] = (int8_t)q1;
            idx++;
        }
    }
}
