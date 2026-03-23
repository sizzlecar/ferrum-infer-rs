//! Kernel backend abstraction layer for LLM-specific fused operations.
//!
//! This module defines a mid-level abstraction between raw `KernelExecutor`
//! (too low-level: grid/block sizes) and `TensorOps` (too high-level: no
//! LLM-specific fused ops). It enables pluggable CUDA/Metal/CPU backends
//! through six focused sub-traits composed into one umbrella `KernelOps`.

use crate::TensorRef;
use ferrum_types::Result;

// ---------------------------------------------------------------------------
// Configuration structs
// ---------------------------------------------------------------------------

/// Rotary position embedding configuration.
#[derive(Debug, Clone)]
pub struct RoPEConfig {
    /// Dimensionality of each attention head.
    pub head_dim: usize,
    /// Maximum sequence length the cache covers.
    pub max_seq_len: usize,
    /// Base frequency (default 10000.0 for standard RoPE).
    pub theta: f32,
}

/// Parameters describing a single attention call.
#[derive(Debug, Clone)]
pub struct AttentionParams {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Softmax scale (typically `1 / sqrt(head_dim)`).
    pub softmax_scale: f32,
    /// Whether to apply a causal mask.
    pub causal: bool,
}

/// Quantization scheme descriptor for quantized linear ops.
#[derive(Debug, Clone)]
pub enum QuantScheme {
    /// 4-bit quantization with group size (e.g. Q4_0 uses group_size=32).
    Q4_0 { group_size: usize },
    /// 8-bit quantization.
    Q8_0 { group_size: usize },
}

/// Sampling parameters for GPU-side token sampling.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Token IDs that have appeared in context (for repetition penalty).
    pub repetition_token_ids: Vec<u32>,
    /// Frequency count for each token in `repetition_token_ids`.
    pub repetition_token_freqs: Vec<u32>,
    /// Per-step RNG seed.
    pub rng_seed: u32,
}

// ---------------------------------------------------------------------------
// Sub-traits
// ---------------------------------------------------------------------------

/// Normalization operations.
pub trait NormOps: Send + Sync {
    /// RMS normalization: `x / rms(x) * weight`.
    fn rms_norm(&self, input: &TensorRef, weight: &TensorRef, eps: f32) -> Result<TensorRef>;

    /// Fused RMS normalization with residual add:
    /// `output = rms_norm(input + residual, weight, eps)`.
    /// Returns `(normed_output, updated_residual)`.
    fn rms_norm_residual(
        &self,
        input: &TensorRef,
        residual: &TensorRef,
        weight: &TensorRef,
        eps: f32,
    ) -> Result<(TensorRef, TensorRef)> {
        // Default: add then norm separately.
        let _ = (input, residual, weight, eps);
        Err(ferrum_types::FerrumError::unsupported(
            "rms_norm_residual not implemented",
        ))
    }
}

/// Positional encoding operations.
pub trait PositionOps: Send + Sync {
    /// Apply rotary position embedding to a Q or K tensor.
    ///
    /// `x` shape: `[batch, seq_len, num_heads, head_dim]`
    /// `position_ids`: position indices for each token in the sequence.
    /// `cos_cache` / `sin_cache`: precomputed `[max_seq_len, head_dim/2]`.
    fn rotary_embedding(
        &self,
        x: &TensorRef,
        cos_cache: &TensorRef,
        sin_cache: &TensorRef,
        position_ids: &[usize],
    ) -> Result<TensorRef>;
}

/// Attention operations.
pub trait AttentionOps: Send + Sync {
    /// Standard multi-head / grouped-query attention.
    ///
    /// * `q` — `[batch, seq_q, num_heads, head_dim]`
    /// * `k` — `[batch, seq_kv, num_kv_heads, head_dim]`
    /// * `v` — `[batch, seq_kv, num_kv_heads, head_dim]`
    ///
    /// Returns attention output `[batch, seq_q, num_heads, head_dim]`.
    fn attention(
        &self,
        q: &TensorRef,
        k: &TensorRef,
        v: &TensorRef,
        params: &AttentionParams,
    ) -> Result<TensorRef>;

    /// Paged attention for KV-cache-based decode.
    ///
    /// Default returns unsupported — backends opt in.
    fn paged_attention(
        &self,
        _q: &TensorRef,
        _k_cache: &TensorRef,
        _v_cache: &TensorRef,
        _block_table: &[u32],
        _params: &AttentionParams,
    ) -> Result<TensorRef> {
        Err(ferrum_types::FerrumError::unsupported(
            "paged_attention not implemented",
        ))
    }
}

/// Activation function operations (including fused variants).
pub trait ActivationOps: Send + Sync {
    /// Fused SiLU-multiply: `silu(gate) * up`.
    ///
    /// This is the SwiGLU building block used in LLaMA/Qwen MLPs.
    fn silu_mul(&self, gate: &TensorRef, up: &TensorRef) -> Result<TensorRef>;

    /// GELU activation.
    fn gelu(&self, input: &TensorRef) -> Result<TensorRef>;
}

/// Linear / matrix-multiply operations.
pub trait LinearOps: Send + Sync {
    /// Dense linear projection (no bias): `input @ weight^T`.
    ///
    /// * `input`  — `[..., in_features]`
    /// * `weight` — `[out_features, in_features]`
    fn linear(&self, input: &TensorRef, weight: &TensorRef) -> Result<TensorRef>;

    /// Quantized linear projection.
    ///
    /// `packed_weight` is backend-specific packed data (e.g. Q4_0 blocks).
    fn quantized_linear(
        &self,
        _input: &TensorRef,
        _packed_weight: &TensorRef,
        _scheme: &QuantScheme,
    ) -> Result<TensorRef> {
        Err(ferrum_types::FerrumError::unsupported(
            "quantized_linear not implemented",
        ))
    }
}

/// Token sampling operations (GPU-side when possible).
pub trait SamplingOps: Send + Sync {
    /// Sample a single token from logits using the full sampling pipeline.
    ///
    /// `logits` shape: `[vocab_size]` or `[1, vocab_size]` (last-token logits).
    fn sample_token(&self, logits: &TensorRef, params: &SamplingParams) -> Result<u32>;

    /// Greedy argmax over the last dimension.
    fn argmax(&self, logits: &TensorRef) -> Result<u32>;
}

// ---------------------------------------------------------------------------
// Umbrella trait
// ---------------------------------------------------------------------------

/// Unified kernel operations interface.
///
/// Backends implement whichever sub-traits they support and return `None` for
/// the rest. Callers use `KernelOpsDispatch` (below) to get automatic fallback
/// to `TensorOps` when a sub-trait is unavailable.
pub trait KernelOps: Send + Sync {
    fn norm_ops(&self) -> Option<&dyn NormOps> {
        None
    }
    fn position_ops(&self) -> Option<&dyn PositionOps> {
        None
    }
    fn attention_ops(&self) -> Option<&dyn AttentionOps> {
        None
    }
    fn activation_ops(&self) -> Option<&dyn ActivationOps> {
        None
    }
    fn linear_ops(&self) -> Option<&dyn LinearOps> {
        None
    }
    fn sampling_ops(&self) -> Option<&dyn SamplingOps> {
        None
    }

    /// Human-readable backend identifier (e.g. `"candle"`, `"metal"`, `"cuda"`).
    fn backend_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Dispatch helper (Step 3)
// ---------------------------------------------------------------------------

/// Dispatch wrapper that tries `KernelOps` first, then falls back to
/// `TensorOps` for operations that have a `TensorOps` equivalent.
///
/// This enables gradual migration: callers use the dispatch without caring
/// which path actually runs.
pub struct KernelOpsDispatch<'a> {
    kernel_ops: Option<&'a dyn KernelOps>,
    tensor_ops: &'a dyn crate::TensorOps,
}

impl<'a> KernelOpsDispatch<'a> {
    pub fn new(
        kernel_ops: Option<&'a dyn KernelOps>,
        tensor_ops: &'a dyn crate::TensorOps,
    ) -> Self {
        Self {
            kernel_ops,
            tensor_ops,
        }
    }

    /// RMS norm: prefer `KernelOps::NormOps`, fall back to `TensorOps::rms_norm`.
    pub fn rms_norm(
        &self,
        input: &TensorRef,
        weight: &TensorRef,
        eps: f32,
    ) -> Result<TensorRef> {
        if let Some(ko) = self.kernel_ops {
            if let Some(norm) = ko.norm_ops() {
                return norm.rms_norm(input, weight, eps);
            }
        }
        self.tensor_ops.rms_norm(input, weight, eps)
    }

    /// GELU: prefer `KernelOps::ActivationOps`, fall back to `TensorOps::gelu`.
    pub fn gelu(&self, input: &TensorRef) -> Result<TensorRef> {
        if let Some(ko) = self.kernel_ops {
            if let Some(act) = ko.activation_ops() {
                return act.gelu(input);
            }
        }
        self.tensor_ops.gelu(input)
    }

    /// SiLU: prefer `KernelOps::ActivationOps::silu_mul` is *fused* so there
    /// is no direct `TensorOps` equivalent. This helper exposes the non-fused
    /// `TensorOps::silu` for callers that only need plain SiLU.
    pub fn silu(&self, input: &TensorRef) -> Result<TensorRef> {
        self.tensor_ops.silu(input)
    }

    /// Fused SiLU-multiply (SwiGLU building block).
    /// Falls back to `silu(gate) * up` via TensorOps when kernel is unavailable.
    pub fn silu_mul(&self, gate: &TensorRef, up: &TensorRef) -> Result<TensorRef> {
        if let Some(ko) = self.kernel_ops {
            if let Some(act) = ko.activation_ops() {
                return act.silu_mul(gate, up);
            }
        }
        // Fallback: silu(gate) * up
        let activated = self.tensor_ops.silu(gate)?;
        self.tensor_ops.mul(&activated, up)
    }

    /// Dense linear (no bias).
    /// Falls back to `TensorOps::matmul`.
    pub fn linear(&self, input: &TensorRef, weight: &TensorRef) -> Result<TensorRef> {
        if let Some(ko) = self.kernel_ops {
            if let Some(lin) = ko.linear_ops() {
                return lin.linear(input, weight);
            }
        }
        self.tensor_ops.matmul(input, weight)
    }

    /// Softmax: always via `TensorOps` (no kernel sub-trait for plain softmax).
    pub fn softmax(&self, input: &TensorRef, dim: i32) -> Result<TensorRef> {
        self.tensor_ops.softmax(input, dim)
    }

    /// Access the underlying `KernelOps` (if any) for ops that have no
    /// `TensorOps` fallback (e.g. rotary_embedding, attention, sampling).
    pub fn kernel_ops(&self) -> Option<&'a dyn KernelOps> {
        self.kernel_ops
    }

    /// Access the underlying `TensorOps`.
    pub fn tensor_ops(&self) -> &'a dyn crate::TensorOps {
        self.tensor_ops
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal KernelOps that returns None for everything.
    struct EmptyKernelOps;
    impl KernelOps for EmptyKernelOps {
        fn backend_name(&self) -> &str {
            "empty"
        }
    }

    #[test]
    fn test_empty_kernel_ops_returns_none() {
        let ops = EmptyKernelOps;
        assert!(ops.norm_ops().is_none());
        assert!(ops.position_ops().is_none());
        assert!(ops.attention_ops().is_none());
        assert!(ops.activation_ops().is_none());
        assert!(ops.linear_ops().is_none());
        assert!(ops.sampling_ops().is_none());
        assert_eq!(ops.backend_name(), "empty");
    }

    #[test]
    fn test_rope_config_default() {
        let cfg = RoPEConfig {
            head_dim: 128,
            max_seq_len: 2048,
            theta: 10000.0,
        };
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_attention_params() {
        let params = AttentionParams {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            softmax_scale: 1.0 / (128.0_f32).sqrt(),
            causal: true,
        };
        assert!(params.causal);
        assert_eq!(params.num_heads / params.num_kv_heads, 4); // GQA ratio
    }

    #[test]
    fn test_quant_scheme_variants() {
        let q4 = QuantScheme::Q4_0 { group_size: 32 };
        let q8 = QuantScheme::Q8_0 { group_size: 128 };
        match q4 {
            QuantScheme::Q4_0 { group_size } => assert_eq!(group_size, 32),
            _ => panic!("expected Q4_0"),
        }
        match q8 {
            QuantScheme::Q8_0 { group_size } => assert_eq!(group_size, 128),
            _ => panic!("expected Q8_0"),
        }
    }
}
