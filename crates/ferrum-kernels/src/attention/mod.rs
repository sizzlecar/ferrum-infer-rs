//! Fused attention for CPU and Metal.
//!
//! Attention backends and shared parameters used by the kernel backends.

pub mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

/// Attention configuration.
#[derive(Clone, Debug, Default)]
pub struct AttentionParams {
    pub batch: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub pos_offset: usize,
    /// Sliding-window size. `0` = full causal (default). Mistral v0.1 and
    /// Gemma use 4096; later Mistral versions set this to 0 to disable.
    pub sliding_window: usize,
}

/// Run fused attention on CPU.
pub fn attention_cpu(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], params: &AttentionParams) {
    cpu::fused_attention(q, k, v, out, params);
}

/// Run fused attention on best available backend.
pub fn attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], params: &AttentionParams) {
    #[cfg(feature = "metal")]
    {
        if metal::is_available() {
            metal::fused_attention(q, k, v, out, params);
            return;
        }
    }
    cpu::fused_attention(q, k, v, out, params);
}
