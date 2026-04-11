//! ferrum-attention: Fused flash attention for Metal and CPU.
//!
//! Single-kernel attention (QK^T + softmax + attn@V) with no intermediate buffer
//! materialization. Matches PyTorch precision by eliminating intermediate f32 rounding.

pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

/// Attention configuration.
pub struct AttentionParams {
    pub batch: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub pos_offset: usize,
}

/// Run fused attention on CPU (Accelerate sgemm + fused softmax).
pub fn attention_cpu(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    params: &AttentionParams,
) {
    cpu::fused_attention(q, k, v, out, params);
}

/// Run fused attention on Metal GPU (single kernel, no intermediate buffers).
#[cfg(feature = "metal")]
pub fn attention_metal(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    params: &AttentionParams,
) {
    metal::fused_attention(q, k, v, out, params);
}

/// Run fused attention on best available backend.
pub fn attention(
    q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
    params: &AttentionParams,
) {
    #[cfg(feature = "metal")]
    {
        if metal::is_available() {
            metal::fused_attention(q, k, v, out, params);
            return;
        }
    }
    cpu::fused_attention(q, k, v, out, params);
}
