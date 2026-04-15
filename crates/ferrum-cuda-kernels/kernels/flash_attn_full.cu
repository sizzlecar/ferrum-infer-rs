// Flash attention for full sequence (prefill + autoregressive with KV cache).
// Used by TTS FusedTransformer. Supports GQA (num_heads != num_kv_heads).
//
// Q: [batch, num_heads, q_len, head_dim]
// K: [batch, num_kv_heads, kv_len, head_dim]
// V: [batch, num_kv_heads, kv_len, head_dim]
// O: [batch, num_heads, q_len, head_dim]
//
// Causal mask: position i can only attend to positions <= pos_offset + i.
// GQA: each KV head serves (num_heads / num_kv_heads) Q heads.
//
// Block: 1 threadgroup per (batch, head, q_tile).
// Each block computes attention for TILE_Q query positions.

#include <cstdint>
#include <cfloat>

#define TILE_Q 16
#define TILE_KV 32

struct FlashAttnParams {
    int batch;
    int num_heads;
    int num_kv_heads;
    int q_len;
    int kv_len;
    int head_dim;
    int causal;      // 1 = causal mask, 0 = no mask
    int pos_offset;  // for decode: attend up to pos_offset + q_pos
    int kv_seq_stride; // stride between KV positions (for GPU cache: max_len, for contiguous: kv_len)
};

extern "C" __global__ void flash_attn_full_f32(
    const float* __restrict__ Q,   // [B, NH, SQ, HD]
    const float* __restrict__ K,   // [B, NKV, SK, HD]
    const float* __restrict__ V,   // [B, NKV, SK, HD]
    float* __restrict__ O,         // [B, NH, SQ, HD]
    FlashAttnParams params
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_tile_start = blockIdx.x * TILE_Q;

    int nh = params.num_heads;
    int nkv = params.num_kv_heads;
    int sq = params.q_len;
    int sk = params.kv_len;
    int hd = params.head_dim;
    int kv_stride = params.kv_seq_stride;

    // GQA: map Q head to KV head
    int kv_h = h / (nh / nkv);

    float scale = rsqrtf((float)hd);

    // Each thread handles one query position in the tile
    int q_local = threadIdx.x % TILE_Q;
    int q_pos = q_tile_start + q_local;
    if (q_pos >= sq) return;

    // Pointers
    const float* q_ptr = Q + ((b * nh + h) * sq + q_pos) * hd;
    float* o_ptr = O + ((b * nh + h) * sq + q_pos) * hd;

    // Online softmax state
    float m_prev = -FLT_MAX;  // running max
    float l_prev = 0.0f;       // running sum of exp
    float acc[128];            // accumulator for output (max head_dim = 128)
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;

    // Iterate over KV tiles
    for (int kv_start = 0; kv_start < sk; kv_start += TILE_KV) {
        // Compute dot products Q @ K^T for this tile
        float scores[TILE_KV];
        for (int kv_local = 0; kv_local < TILE_KV; kv_local++) {
            int kv_pos = kv_start + kv_local;
            if (kv_pos >= sk) {
                scores[kv_local] = -FLT_MAX;
                continue;
            }

            // Causal mask
            if (params.causal && kv_pos > params.pos_offset + q_pos) {
                scores[kv_local] = -FLT_MAX;
                continue;
            }

            const float* k_ptr = K + ((b * nkv + kv_h) * kv_stride + kv_pos) * hd;
            float dot = 0.0f;
            for (int d = 0; d < hd; d++) {
                dot += q_ptr[d] * k_ptr[d];
            }
            scores[kv_local] = dot * scale;
        }

        // Online softmax update
        float m_new = m_prev;
        for (int j = 0; j < TILE_KV; j++) {
            m_new = fmaxf(m_new, scores[j]);
        }

        float correction = expf(m_prev - m_new);
        l_prev *= correction;
        for (int d = 0; d < hd; d++) {
            acc[d] *= correction;
        }

        // Accumulate attention @ V
        for (int kv_local = 0; kv_local < TILE_KV; kv_local++) {
            int kv_pos = kv_start + kv_local;
            if (kv_pos >= sk) continue;

            float w = expf(scores[kv_local] - m_new);
            l_prev += w;

            const float* v_ptr = V + ((b * nkv + kv_h) * kv_stride + kv_pos) * hd;
            for (int d = 0; d < hd; d++) {
                acc[d] += w * v_ptr[d];
            }
        }

        m_prev = m_new;
    }

    // Finalize: divide by sum
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int d = 0; d < hd; d++) {
        o_ptr[d] = acc[d] * inv_l;
    }
}
