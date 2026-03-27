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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{TensorLike, TensorRef};
    use std::any::Any;
    use std::sync::Arc;

    /// Minimal mock tensor for testing (avoids circular dep with ferrum-testkit).
    #[derive(Debug)]
    struct TestTensor {
        shape: Vec<usize>,
    }
    impl TensorLike for TestTensor {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn is_contiguous(&self) -> bool {
            true
        }
        fn dtype(&self) -> ferrum_types::DataType {
            ferrum_types::DataType::FP16
        }
        fn device(&self) -> ferrum_types::Device {
            ferrum_types::Device::CPU
        }
        fn view(&self, _: &[usize], _end: &[usize]) -> ferrum_types::Result<TensorRef> {
            Ok(Arc::new(TestTensor {
                shape: self.shape.clone(),
            }))
        }
        fn reshape(&self, shape: &[usize]) -> ferrum_types::Result<TensorRef> {
            Ok(Arc::new(TestTensor {
                shape: shape.to_vec(),
            }))
        }
        fn to_cpu(&self) -> ferrum_types::Result<TensorRef> {
            Ok(Arc::new(TestTensor {
                shape: self.shape.clone(),
            }))
        }
        fn to_device(&self, _: &ferrum_types::Device) -> ferrum_types::Result<TensorRef> {
            Ok(Arc::new(TestTensor {
                shape: self.shape.clone(),
            }))
        }
        fn to_dtype(&self, _: ferrum_types::DataType) -> ferrum_types::Result<TensorRef> {
            Ok(Arc::new(TestTensor {
                shape: self.shape.clone(),
            }))
        }
    }

    struct MockWeights {
        config: TransformerConfig,
    }

    impl MockWeights {
        fn new(num_layers: usize) -> Self {
            Self {
                config: TransformerConfig {
                    num_layers,
                    hidden_size: 64,
                    num_attention_heads: 4,
                    num_kv_heads: 2,
                    head_dim: 16,
                    intermediate_size: 128,
                    vocab_size: 100,
                    max_seq_len: 512,
                    rms_norm_eps: 1e-6,
                    has_qk_norm: true,
                },
            }
        }

        fn mock_tensor(shape: &[usize]) -> TensorRef {
            Arc::new(TestTensor {
                shape: shape.to_vec(),
            })
        }
    }

    impl TransformerWeights for MockWeights {
        fn config(&self) -> &TransformerConfig {
            &self.config
        }
        fn embed_weight(&self) -> TensorRef {
            Self::mock_tensor(&[self.config.vocab_size, self.config.hidden_size])
        }
        fn layer_input_norm_weight(&self, _layer: usize) -> TensorRef {
            Self::mock_tensor(&[self.config.hidden_size])
        }
        fn layer_qkv_weight(&self, _layer: usize) -> TensorRef {
            let q = self.config.num_attention_heads * self.config.head_dim;
            let kv = self.config.num_kv_heads * self.config.head_dim;
            Self::mock_tensor(&[q + 2 * kv, self.config.hidden_size])
        }
        fn layer_q_norm_weight(&self, _layer: usize) -> Option<TensorRef> {
            if self.config.has_qk_norm {
                Some(Self::mock_tensor(&[self.config.head_dim]))
            } else {
                None
            }
        }
        fn layer_k_norm_weight(&self, _layer: usize) -> Option<TensorRef> {
            self.layer_q_norm_weight(_layer)
        }
        fn layer_o_weight(&self, _layer: usize) -> TensorRef {
            let q = self.config.num_attention_heads * self.config.head_dim;
            Self::mock_tensor(&[self.config.hidden_size, q])
        }
        fn layer_post_norm_weight(&self, _layer: usize) -> TensorRef {
            Self::mock_tensor(&[self.config.hidden_size])
        }
        fn layer_gate_up_weight(&self, _layer: usize) -> TensorRef {
            Self::mock_tensor(&[2 * self.config.intermediate_size, self.config.hidden_size])
        }
        fn layer_down_weight(&self, _layer: usize) -> TensorRef {
            Self::mock_tensor(&[self.config.hidden_size, self.config.intermediate_size])
        }
        fn final_norm_weight(&self) -> TensorRef {
            Self::mock_tensor(&[self.config.hidden_size])
        }
        fn lm_head_weight(&self) -> TensorRef {
            Self::mock_tensor(&[self.config.vocab_size, self.config.hidden_size])
        }
        fn rope_cos(&self) -> TensorRef {
            Self::mock_tensor(&[self.config.max_seq_len, self.config.head_dim / 2])
        }
        fn rope_sin(&self) -> TensorRef {
            Self::mock_tensor(&[self.config.max_seq_len, self.config.head_dim / 2])
        }
    }

    #[test]
    fn transformer_weights_config() {
        let w = MockWeights::new(4);
        assert_eq!(w.config().num_layers, 4);
        assert_eq!(w.config().hidden_size, 64);
        assert!(w.config().has_qk_norm);
    }

    #[test]
    fn transformer_weights_shapes() {
        let w = MockWeights::new(2);
        let cfg = w.config();

        // Embed: [vocab, hidden]
        assert_eq!(w.embed_weight().shape(), &[100, 64]);

        // QKV: [q_dim + 2*kv_dim, hidden]
        let q_dim = cfg.num_attention_heads * cfg.head_dim; // 4*16=64
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 2*16=32
        assert_eq!(w.layer_qkv_weight(0).shape(), &[q_dim + 2 * kv_dim, 64]);

        // Q/K norm: [head_dim]
        assert_eq!(w.layer_q_norm_weight(0).unwrap().shape(), &[16]);
        assert_eq!(w.layer_k_norm_weight(1).unwrap().shape(), &[16]);

        // Gate+Up: [2*inter, hidden]
        assert_eq!(w.layer_gate_up_weight(0).shape(), &[256, 64]);

        // LM head: [vocab, hidden]
        assert_eq!(w.lm_head_weight().shape(), &[100, 64]);

        // RoPE: [max_seq, head_dim/2]
        assert_eq!(w.rope_cos().shape(), &[512, 8]);
    }

    #[test]
    fn transformer_weights_no_qk_norm() {
        let mut w = MockWeights::new(2);
        w.config.has_qk_norm = false;
        assert!(w.layer_q_norm_weight(0).is_none());
        assert!(w.layer_k_norm_weight(0).is_none());
    }

    #[test]
    fn transformer_weights_all_layers() {
        let w = MockWeights::new(36);
        for i in 0..36 {
            // Every layer should return valid tensors
            assert!(!w.layer_input_norm_weight(i).shape().is_empty());
            assert!(!w.layer_qkv_weight(i).shape().is_empty());
            assert!(!w.layer_o_weight(i).shape().is_empty());
        }
    }
}
