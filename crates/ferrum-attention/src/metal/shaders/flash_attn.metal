#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Fused Flash Attention for f32 ──────────────────────────────────────
//
// Single kernel: QK^T * scale + causal_mask → online softmax → attn@V
// No intermediate buffers. All accumulation in registers/threadgroup memory.
//
// Layout:
//   Q: [batch, num_heads, q_len, head_dim]  (contiguous)
//   K: [batch, num_kv_heads, kv_len, head_dim]
//   V: [batch, num_kv_heads, kv_len, head_dim]
//   O: [batch, num_heads, q_len, head_dim]
//
// Grid: (q_len, num_heads, batch) — one threadgroup per query position per head
// Each threadgroup processes one row of the attention matrix.

struct FlashAttnParams {
    int batch;
    int num_heads;
    int num_kv_heads;
    int q_len;
    int kv_len;
    int head_dim;
    float scale;
    int causal;       // 0 or 1
    int pos_offset;
};

// Block size for KV processing — process this many KV positions per iteration
constant int BLOCK_KV = 32;

kernel void flash_attn_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device       float* O       [[buffer(3)]],
    constant FlashAttnParams& p [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],   // (q_pos, head, batch)
    uint  tiisg [[thread_index_in_simdgroup]],       // 0..31
    uint  sgitg [[simdgroup_index_in_threadgroup]])  // simdgroup index
{
    const int qi    = tgpig.x;  // which query position
    const int hi    = tgpig.y;  // which head
    const int bi    = tgpig.z;  // which batch

    const int kv_hi = hi / (p.num_heads / p.num_kv_heads); // GQA: KV head index
    const int d     = p.head_dim;
    const int sk    = p.kv_len;

    // Causal: how many KV positions can we attend to?
    const int attend_len = p.causal ? min(p.pos_offset + qi + 1, sk) : sk;

    // Pointers
    device const float* q_row = Q + ((bi * p.num_heads + hi) * p.q_len + qi) * d;
    device const float* k_base = K + (bi * p.num_kv_heads + kv_hi) * sk * d;
    device const float* v_base = V + (bi * p.num_kv_heads + kv_hi) * sk * d;
    device       float* o_row = O + ((bi * p.num_heads + hi) * p.q_len + qi) * d;

    // Online softmax state (per thread, then reduced)
    float M = -INFINITY;  // running max
    float S = 0.0f;       // running sum of exp

    // Output accumulator in registers (one float per head_dim element)
    // Each thread handles d/32 elements (32 threads in simdgroup)
    // For head_dim=128: each thread handles 4 elements
    const int elems_per_thread = d / 32;

    // Local output accumulator
    float acc[4] = {0, 0, 0, 0}; // supports up to head_dim=128 (4 per thread)

    // Process KV in blocks
    for (int kv_start = 0; kv_start < attend_len; kv_start += BLOCK_KV) {
        int kv_end = min(kv_start + BLOCK_KV, attend_len);

        for (int ki = kv_start; ki < kv_end; ++ki) {
            device const float* k_row = k_base + ki * d;
            device const float* v_row = v_base + ki * d;

            // Compute dot product Q[qi] · K[ki] using simd reduction
            float dot = 0.0f;
            for (int j = tiisg; j < d; j += 32) {
                dot += q_row[j] * k_row[j];
            }
            dot = simd_sum(dot) * p.scale;

            // Online softmax update
            float old_M = M;
            M = max(M, dot);
            float exp_diff = exp(old_M - M);
            float exp_val = exp(dot - M);

            // Rescale existing accumulator and sum
            S = S * exp_diff + exp_val;

            // Update output: O = O * exp_diff + exp_val * V[ki]
            for (int j = 0; j < elems_per_thread; ++j) {
                int idx = tiisg + j * 32;
                if (idx < d) {
                    acc[j] = acc[j] * exp_diff + exp_val * v_row[idx];
                }
            }
        }
    }

    // Final normalization: O = O / S
    float inv_S = (S > 0.0f) ? (1.0f / S) : 0.0f;
    for (int j = 0; j < elems_per_thread; ++j) {
        int idx = tiisg + j * 32;
        if (idx < d) {
            o_row[idx] = acc[j] * inv_S;
        }
    }
}
