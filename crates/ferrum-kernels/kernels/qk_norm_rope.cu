// Fused QK-norm + RoPE + transpose kernel (ported from Metal `norm_rope.metal`).
//
// Input:  [tokens, heads, head_dim]  (token-major, output of split_qkv)
// Output: [heads, tokens, head_dim]  (head-major, ready for flash_attn / kv_cache_append)
//
// mode = 0 : transpose only                 (for V)
// mode = 1 : per-head RMS norm + RoPE + transpose  (Q/K with QK-norm, Qwen3)
// mode = 2 : half-split RoPE + transpose     (Q/K without QK-norm)
// mode = 3 : interleaved RoPE + transpose    (GGUF LLaMA / llama.cpp layout)
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

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
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

    if (mode == 3) {
        for (int i = lane; i < half_d; i += 32) {
            const int j = 2 * i;
            float x0 = __half2float(src[j]);
            float x1 = __half2float(src[j + 1]);
            float c = __half2float(cos_row[i]);
            float s = __half2float(sin_row[i]);
            dst[j]     = __float2half(x0 * c - x1 * s);
            dst[j + 1] = __float2half(x1 * c + x0 * s);
        }
        return;
    }

    // Apply norm (if mode 1) + half-split RoPE + write.
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

// Qwen3.5 partial-RoPE variant.
//
// Input:  [tokens, input_stride], with each head at
// input_offset + head * input_head_stride.
// Output: [heads, tokens, head_dim].
//
// mode = 0 : transpose selected slice only (V path)
// mode = 1 : per-head RMSNorm + half-split partial RoPE
// mode = 2 : half-split partial RoPE only
// mode = 3 : per-head RMSNorm + interleaved partial RoPE
extern "C" __global__ void qk_norm_rope_partial_transpose_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ norm_w,
    const __half* __restrict__ cos_tab,
    const __half* __restrict__ sin_tab,
    __half* __restrict__ output,
    const int tokens,
    const int heads,
    const int head_dim,
    const int rope_dim,
    const int input_stride,
    const int input_offset,
    const int input_head_stride,
    const int pos_offset,
    const float eps,
    const int mode
) {
    const int tok = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if (tok >= tokens || head >= heads) return;

    const int hd = head_dim;
    const int rd = rope_dim;
    const int half_rd = rd / 2;

    const __half* src = input + tok * input_stride + input_offset + head * input_head_stride;
    __half* dst = output + (head * tokens + tok) * hd;

    if (mode == 0) {
        for (int i = lane; i < hd; i += 32) {
            dst[i] = src[i];
        }
        return;
    }

    const bool do_norm = (mode == 1 || mode == 3);
    float scale = 1.0f;
    if (do_norm) {
        float sum_sq = 0.0f;
        for (int i = lane; i < hd; i += 32) {
            float v = __half2float(src[i]);
            sum_sq += v * v;
        }
        sum_sq = warp_reduce_sum(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    const int pos = pos_offset + tok;
    const __half* cos_row = cos_tab + pos * half_rd;
    const __half* sin_row = sin_tab + pos * half_rd;

    if (mode == 3) {
        for (int pair = lane; pair < half_rd; pair += 32) {
            const int j = 2 * pair;
            float x0 = __half2float(src[j]);
            float x1 = __half2float(src[j + 1]);
            if (do_norm) {
                x0 *= scale * __half2float(norm_w[j]);
                x1 *= scale * __half2float(norm_w[j + 1]);
            }
            float c = __half2float(cos_row[pair]);
            float s = __half2float(sin_row[pair]);
            dst[j] = __float2half(x0 * c - x1 * s);
            dst[j + 1] = __float2half(x1 * c + x0 * s);
        }
        for (int i = rd + lane; i < hd; i += 32) {
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
        float c = __half2float(cos_row[i]);
        float s = __half2float(sin_row[i]);
        dst[i] = __float2half(x0 * c - x1 * s);
        dst[i + half_rd] = __float2half(x1 * c + x0 * s);
    }
    for (int i = rd + lane; i < hd; i += 32) {
        float x = __half2float(src[i]);
        if (do_norm) {
            x *= scale * __half2float(norm_w[i]);
        }
        dst[i] = __float2half(x);
    }
}

extern "C" __global__ void qwen35_apply_attention_gate_f16(
    __half* __restrict__ context,
    const __half* __restrict__ query_raw,
    const int tokens,
    const int q_total,
    const int q_proj_total,
    const int head_dim
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * q_total;
    if (idx >= total) return;

    const int tok = idx / q_total;
    const int dim = idx - tok * q_total;
    const int head = dim / head_dim;
    const int dim_in_head = dim - head * head_dim;
    const int gate_idx = tok * q_proj_total + head * (2 * head_dim) + head_dim + dim_in_head;
    float value = __half2float(context[idx]);
    float gate = __half2float(query_raw[gate_idx]);
    context[idx] = __float2half(value * sigmoid_f32(gate));
}

extern "C" __global__ void qwen35_apply_token_gate_f16(
    __half* __restrict__ values,
    const __half* __restrict__ gate,
    const int tokens,
    const int hidden_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden_size;
    if (idx >= total) return;

    const int tok = idx / hidden_size;
    float value = __half2float(values[idx]);
    float gate_value = __half2float(gate[tok]);
    values[idx] = __float2half(value * sigmoid_f32(gate_value));
}

extern "C" __global__ void qwen35_apply_token_gate_and_add_inplace_f16(
    __half* __restrict__ dst,
    __half* __restrict__ values,
    const __half* __restrict__ gate,
    const int tokens,
    const int hidden_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden_size;
    if (idx >= total) return;

    const int tok = idx / hidden_size;
    const float gate_value = __half2float(gate[tok]);
    const float gated = __half2float(values[idx]) * sigmoid_f32(gate_value);
    values[idx] = __float2half(gated);
    dst[idx] = __float2half(__half2float(dst[idx]) + gated);
}

extern "C" __global__ void qwen35_apply_token_gate_and_add_inplace_f32(
    float* __restrict__ dst,
    float* __restrict__ values,
    const float* __restrict__ gate,
    const int tokens,
    const int hidden_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden_size;
    if (idx >= total) return;

    const int tok = idx / hidden_size;
    const float gated = values[idx] * sigmoid_f32(gate[tok]);
    values[idx] = gated;
    dst[idx] += gated;
}

extern "C" __global__ void qwen35_interleave_gate_up_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    const int tokens,
    const int intermediate
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * intermediate;
    if (idx >= total) return;

    const int tok = idx / intermediate;
    const int col = idx - tok * intermediate;
    const int src_idx = tok * intermediate + col;
    const int dst_base = tok * 2 * intermediate + col;
    out[dst_base] = gate[src_idx];
    out[dst_base + intermediate] = up[src_idx];
}

extern "C" __global__ void qwen35_interleave_gate_up_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    const int tokens,
    const int intermediate
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * intermediate;
    if (idx >= total) return;

    const int tok = idx / intermediate;
    const int col = idx - tok * intermediate;
    const int src_idx = tok * intermediate + col;
    const int dst_base = tok * 2 * intermediate + col;
    out[dst_base] = gate[src_idx];
    out[dst_base + intermediate] = up[src_idx];
}

// Batched per-item-position variant for batched decode (m sequences, q_len=1
// each, every item at its own absolute position). Replaces M sequential
// `qk_norm_rope_transpose_f16` launches with one launch.
//
// Layout (DECODE only — preserves item-major order for downstream
// per-item flash_attention to slice contiguous chunks):
//   Input:  [m, heads, head_dim] (token-major, output of split_qkv at m=batch)
//   Output: [m, heads, head_dim] (token-major, in-place safe)
//
// `positions[m]` (device-side i32) gives each item's absolute RoPE pos.
// `mode` is the same enum used by qk_norm_rope_transpose_f16:
//   0 = transpose-only path (V) — degenerate to identity here since we
//       keep token-major; included for API symmetry.
//   1 = per-head RMSNorm + RoPE  (Q/K with QK-norm, Qwen3)
//   2 = half-split RoPE-only     (Q/K without QK-norm)
//   3 = interleaved RoPE-only    (GGUF LLaMA / llama.cpp layout)
//
// Launch: grid = (m, heads, 1), block = (warpSize=32, 1, 1).
// Each warp handles one (item, head) pair.
extern "C" __global__ void qk_norm_rope_batched_decode_f16(
    const __half* __restrict__ input,        // [m, heads, head_dim]
    const __half* __restrict__ norm_w,       // [head_dim] (mode 1 only)
    const __half* __restrict__ cos_tab,      // [max_seq, head_dim/2]
    const __half* __restrict__ sin_tab,      // [max_seq, head_dim/2]
    __half* __restrict__ output,             // [m, heads, head_dim]
    const int m,
    const int heads,
    const int head_dim,
    const int* __restrict__ positions,       // [m] absolute rope position per item
    const float eps,
    const int mode
) {
    const int item = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;          // 0..31
    if (item >= m || head >= heads) return;

    const int hd = head_dim;
    const int half_d = hd / 2;

    // Token-major: item i's head h sits at offset (i * heads + h) * hd.
    const __half* src = input  + (item * heads + head) * hd;
    __half* dst       = output + (item * heads + head) * hd;

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
        sum_sq = warp_reduce_sum(sum_sq);
        scale = rsqrtf(sum_sq / (float)hd + eps);
    }

    const int pos = positions[item];
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    if (mode == 3) {
        for (int i = lane; i < half_d; i += 32) {
            const int j = 2 * i;
            float x0 = __half2float(src[j]);
            float x1 = __half2float(src[j + 1]);
            float c = __half2float(cos_row[i]);
            float s = __half2float(sin_row[i]);
            dst[j]     = __float2half(x0 * c - x1 * s);
            dst[j + 1] = __float2half(x1 * c + x0 * s);
        }
        return;
    }

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

// Device-state variant: pos_offset read from a device slot for graph capture.
// Same math as qk_norm_rope_transpose_f16 but `pos_offset` replaced by
// `*pos_ptr`. Decode launches with tokens=1 — we can specialise blockIdx.x
// away but kept for parity with the scalar kernel's launch config.
extern "C" __global__ void qk_norm_rope_transpose_f16_dyn(
    const __half* __restrict__ input,
    const __half* __restrict__ norm_w,
    const __half* __restrict__ cos_tab,
    const __half* __restrict__ sin_tab,
    __half* __restrict__ output,
    const int tokens,
    const int heads,
    const int head_dim,
    const int* __restrict__ pos_ptr,      // device: single int32
    const float eps,
    const int mode
) {
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if (tok >= tokens || head >= heads) return;

    const int hd = head_dim;
    const int half_d = hd / 2;

    const __half* src = input  + (tok * heads + head) * hd;
    __half* dst       = output + (head * tokens + tok) * hd;

    if (mode == 0) {
        for (int i = lane; i < hd; i += 32) dst[i] = src[i];
        return;
    }

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

    const int pos_offset = pos_ptr[0];
    const int pos = pos_offset + tok;
    const __half* cos_row = cos_tab + pos * half_d;
    const __half* sin_row = sin_tab + pos * half_d;

    if (mode == 3) {
        for (int i = lane; i < half_d; i += 32) {
            const int j = 2 * i;
            float x0 = __half2float(src[j]);
            float x1 = __half2float(src[j + 1]);
            float c = __half2float(cos_row[i]);
            float s = __half2float(sin_row[i]);
            dst[j]     = __float2half(x0 * c - x1 * s);
            dst[j + 1] = __float2half(x1 * c + x0 * s);
        }
        return;
    }

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
