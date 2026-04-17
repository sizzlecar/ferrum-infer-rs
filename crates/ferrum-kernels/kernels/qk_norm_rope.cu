// Fused QK-norm + RoPE + transpose kernel (ported from Metal `norm_rope.metal`).
//
// Input:  [tokens, heads, head_dim]  (token-major, output of split_qkv)
// Output: [heads, tokens, head_dim]  (head-major, ready for flash_attn / kv_cache_append)
//
// mode = 0 : transpose only                 (for V)
// mode = 1 : per-head RMS norm + RoPE + transpose  (Q/K with QK-norm, Qwen3)
// mode = 2 : RoPE + transpose                (Q/K without QK-norm, Llama/Mistral)
//
// Launch: grid = (tokens, heads, 1), block = (warpSize=32, 1, 1).
// Each warp handles one (tok, head) pair; threads in the warp stride by 32 over head_dim.
// RMS norm uses warp-level __shfl_xor_sync reduce — no shared memory needed.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

extern "C" __global__ void qk_norm_rope_transpose_f16(
    const __half* __restrict__ input,     // [tokens, heads, head_dim]
    const __half* __restrict__ norm_w,    // [head_dim] (unused for mode 0/2)
    const __half* __restrict__ cos_tab,   // [max_seq, head_dim/2] (unused for mode 0)
    const __half* __restrict__ sin_tab,   // [max_seq, head_dim/2] (unused for mode 0)
    __half* __restrict__ output,          // [heads, tokens, head_dim]
    const int tokens,
    const int heads,
    const int head_dim,
    const int pos_offset,
    const float eps,
    const int mode
) {
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;          // 0..31
    if (tok >= tokens || head >= heads) return;

    const int hd = head_dim;
    const int half_d = hd / 2;

    const __half* src = input  + (tok * heads + head) * hd;
    __half* dst       = output + (head * tokens + tok) * hd;

    if (mode == 0) {
        // Transpose only (V path).
        for (int i = lane; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    // RMS norm scale (mode == 1 only).
    float scale = 1.0f;
    if (mode == 1) {
        float sum_sq = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            float v = __half2float(src[i]);
            sum_sq += v * v;
        }
        sum_sq = warp_reduce_sum(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    // RoPE table rows for this position.
    const int pos = pos_offset + tok;
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    // Apply norm (if mode 1) + RoPE + write.
    for (int i = lane; i < half_d; i += 32) {
        float x0 = __half2float(src[i]);
        float x1 = __half2float(src[i + half_d]);
        if (mode == 1) {
            x0 *= scale * __half2float(norm_w[i]);
            x1 *= scale * __half2float(norm_w[i + half_d]);
        }
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);
        dst[i]          = __float2half(x0 * c - x1 * s);
        dst[i + half_d] = __float2half(x1 * c + x0 * s);
    }
}
