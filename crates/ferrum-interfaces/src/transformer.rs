//! Transformer model weight abstraction.
//!
//! Different model architectures (Qwen3, Llama, Qwen2) implement
//! `TransformerWeights` to provide a uniform weight access interface.
//! This decouples the execution backend from the model architecture.

use crate::tensor::TensorRef;

/// Configuration for a standard transformer decoder.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
    /// Whether Q/K heads have per-head normalization (Qwen3 has this, Llama doesn't).
    pub has_qk_norm: bool,
}

/// Uniform weight access for transformer decoder models.
///
/// This trait abstracts over different model architectures. All standard
/// transformer decoders share the same layer structure:
///
/// ```text
/// embed → N × (norm → QKV → [Q/K norm] → RoPE → Attention → O → norm → MLP) → norm → lm_head
/// ```
///
/// Variance between architectures (Qwen3 vs Llama vs Qwen2):
/// - head_dim: explicit vs derived from hidden_size / num_heads
/// - Q/K normalization: present in Qwen3, absent in Llama
/// - Bias: present in some Qwen2 layers, absent elsewhere
/// - RoPE parameters: differ between architectures
///
/// Weights are returned as `TensorRef` — the same zero-copy handle used
/// throughout the framework. The backend extracts device-specific pointers
/// (CudaSlice, Metal buffer, etc.) from the TensorRef.
pub trait TransformerWeights: Send + Sync {
    /// Model configuration.
    fn config(&self) -> &TransformerConfig;

    /// Embedding table: [vocab_size, hidden_size]
    fn embed_weight(&self) -> TensorRef;

    /// Input layer norm weight for layer `i`: [hidden_size]
    fn layer_input_norm_weight(&self, layer: usize) -> TensorRef;

    /// Fused QKV projection weight for layer `i`: [q_dim + 2*kv_dim, hidden_size]
    /// If the model stores Q/K/V separately, the implementation fuses them.
    fn layer_qkv_weight(&self, layer: usize) -> TensorRef;

    /// Q-head normalization weight: [head_dim] (None if architecture doesn't have it)
    fn layer_q_norm_weight(&self, layer: usize) -> Option<TensorRef>;

    /// K-head normalization weight: [head_dim] (None if architecture doesn't have it)
    fn layer_k_norm_weight(&self, layer: usize) -> Option<TensorRef>;

    /// Output projection weight for layer `i`: [hidden_size, q_dim]
    fn layer_o_weight(&self, layer: usize) -> TensorRef;

    /// Post-attention layer norm weight for layer `i`: [hidden_size]
    fn layer_post_norm_weight(&self, layer: usize) -> TensorRef;

    /// Fused gate+up projection weight: [2*intermediate_size, hidden_size]
    fn layer_gate_up_weight(&self, layer: usize) -> TensorRef;

    /// Down projection weight: [hidden_size, intermediate_size]
    fn layer_down_weight(&self, layer: usize) -> TensorRef;

    /// Final RMS norm weight: [hidden_size]
    fn final_norm_weight(&self) -> TensorRef;

    /// LM head projection weight: [vocab_size, hidden_size]
    fn lm_head_weight(&self) -> TensorRef;

    /// RoPE cosine table: [max_seq_len, head_dim/2]
    fn rope_cos(&self) -> TensorRef;

    /// RoPE sine table: [max_seq_len, head_dim/2]
    fn rope_sin(&self) -> TensorRef;
}
