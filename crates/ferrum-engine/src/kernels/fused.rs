//! Fused Kernel Operation Traits
//!
//! This module provides trait definitions for fused kernel operations.
//! Fused operations combine multiple operations into single passes to
//! reduce memory traffic:
//!
//! - RoPE + Attention: Combine rotary position embedding with attention
//! - QKV projection + split: Single matmul then reshape
//! - LayerNorm + Linear: Fuse normalization with projection
//! - SiLU * Gate: Fused activation for MLP
//!
//! Actual implementations are provided by backend-specific modules.

use ferrum_types::Device;
use std::fmt;

/// Fused operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOpType {
    /// RoPE application to Q and K
    RopeQK,
    /// QKV projection and split
    QkvProject,
    /// SiLU activation with gate
    SiluGate,
    /// LayerNorm followed by Linear
    LayerNormLinear,
    /// Add residual + RMSNorm
    AddRmsNorm,
    /// Full RoPE + Attention fusion
    RopeAttention,
}

impl fmt::Display for FusedOpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedOpType::RopeQK => write!(f, "rope_qk"),
            FusedOpType::QkvProject => write!(f, "qkv_project"),
            FusedOpType::SiluGate => write!(f, "silu_gate"),
            FusedOpType::LayerNormLinear => write!(f, "layernorm_linear"),
            FusedOpType::AddRmsNorm => write!(f, "add_rmsnorm"),
            FusedOpType::RopeAttention => write!(f, "rope_attention"),
        }
    }
}

/// Configuration for fused operations
#[derive(Debug, Clone)]
pub struct FusedOpsConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Maximum sequence length (for RoPE cache)
    pub max_seq_len: usize,
    /// RoPE base frequency
    pub rope_base: f32,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// Target device
    pub device: Device,
}

impl Default for FusedOpsConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            hidden_dim: 4096,
            max_seq_len: 4096,
            rope_base: 10000.0,
            rms_norm_eps: 1e-6,
            device: Device::CPU,
        }
    }
}

impl FusedOpsConfig {
    /// Create config for a specific model
    pub fn for_model(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            ..Default::default()
        }
    }

    /// Set RoPE base
    pub fn with_rope_base(mut self, base: f32) -> Self {
        self.rope_base = base;
        self
    }

    /// Set max sequence length
    pub fn with_max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = len;
        self
    }

    /// Set device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
}

/// Fused RoPE + Attention configuration
#[derive(Debug, Clone)]
pub struct FusedRopeAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE base frequency
    pub rope_base: f32,
    /// Softmax scale
    pub softmax_scale: f32,
    /// Use causal masking
    pub causal: bool,
    /// Device
    pub device: Device,
}

impl Default for FusedRopeAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 4096,
            rope_base: 10000.0,
            softmax_scale: 0.088388, // 1/sqrt(128)
            causal: true,
            device: Device::CPU,
        }
    }
}

impl FusedRopeAttentionConfig {
    /// Create from attention config and rope settings
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_base: f32,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            rope_base,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }
}

/// Trait for fused operation providers
///
/// Implementations provide fused operations for specific backends.
pub trait FusedOps: Send + Sync {
    /// Get the supported fused operations
    fn supported_ops(&self) -> Vec<FusedOpType>;

    /// Check if a specific operation is supported
    fn supports(&self, op: FusedOpType) -> bool {
        self.supported_ops().contains(&op)
    }

    /// Get configuration
    fn config(&self) -> &FusedOpsConfig;

    /// Get description
    fn description(&self) -> &str;

    /// Estimated memory savings (percentage)
    fn memory_savings(&self, op: FusedOpType) -> f32 {
        match op {
            FusedOpType::RopeQK => 0.0,     // No memory savings, just compute
            FusedOpType::QkvProject => 0.3, // 30% from avoiding intermediate
            FusedOpType::SiluGate => 0.5,   // 50% from avoiding gate storage
            FusedOpType::LayerNormLinear => 0.3,
            FusedOpType::AddRmsNorm => 0.3,
            FusedOpType::RopeAttention => 0.6, // Significant from fused RoPE
        }
    }

    /// Estimated compute speedup (multiplier)
    fn compute_speedup(&self, op: FusedOpType) -> f32 {
        match op {
            FusedOpType::RopeQK => 1.2,     // 20% faster
            FusedOpType::QkvProject => 1.5, // 50% faster
            FusedOpType::SiluGate => 1.3,   // 30% faster
            FusedOpType::LayerNormLinear => 1.4,
            FusedOpType::AddRmsNorm => 1.3,
            FusedOpType::RopeAttention => 2.0, // 2x faster for full fusion
        }
    }
}

/// CPU fused operations info
pub struct CpuFusedOpsInfo {
    config: FusedOpsConfig,
}

impl CpuFusedOpsInfo {
    pub fn new(config: FusedOpsConfig) -> Self {
        Self { config }
    }
}

impl FusedOps for CpuFusedOpsInfo {
    fn supported_ops(&self) -> Vec<FusedOpType> {
        vec![
            FusedOpType::RopeQK,
            FusedOpType::QkvProject,
            FusedOpType::SiluGate,
            FusedOpType::AddRmsNorm,
        ]
    }

    fn config(&self) -> &FusedOpsConfig {
        &self.config
    }

    fn description(&self) -> &str {
        "CPU fused operations using SIMD where available"
    }
}

/// Metal fused operations info
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub struct MetalFusedOpsInfo {
    config: FusedOpsConfig,
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl MetalFusedOpsInfo {
    pub fn new(config: FusedOpsConfig) -> Self {
        Self { config }
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl FusedOps for MetalFusedOpsInfo {
    fn supported_ops(&self) -> Vec<FusedOpType> {
        vec![
            FusedOpType::RopeQK,
            FusedOpType::QkvProject,
            FusedOpType::SiluGate,
            FusedOpType::LayerNormLinear,
            FusedOpType::AddRmsNorm,
            FusedOpType::RopeAttention, // Full fusion on Metal
        ]
    }

    fn config(&self) -> &FusedOpsConfig {
        &self.config
    }

    fn description(&self) -> &str {
        "Metal GPU fused operations with custom shaders"
    }
}

/// RoPE cache precomputation result
#[derive(Debug, Clone)]
pub struct RopeCache {
    /// Cosine values: [max_seq_len, head_dim/2]
    pub cos: Vec<f32>,
    /// Sine values: [max_seq_len, head_dim/2]
    pub sin: Vec<f32>,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl RopeCache {
    /// Compute RoPE cache
    pub fn compute(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half_dim];
        let mut sin = vec![0.0f32; max_seq_len * half_dim];

        // Compute inverse frequencies
        let inv_freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Compute cos/sin for each position
        for pos in 0..max_seq_len {
            for (i, &inv_freq) in inv_freqs.iter().enumerate() {
                let angle = pos as f32 * inv_freq;
                cos[pos * half_dim + i] = angle.cos();
                sin[pos * half_dim + i] = angle.sin();
            }
        }

        Self {
            cos,
            sin,
            max_seq_len,
            head_dim,
        }
    }

    /// Get cos value at position and dimension
    pub fn cos_at(&self, pos: usize, dim: usize) -> f32 {
        let half_dim = self.head_dim / 2;
        self.cos[pos * half_dim + dim]
    }

    /// Get sin value at position and dimension
    pub fn sin_at(&self, pos: usize, dim: usize) -> f32 {
        let half_dim = self.head_dim / 2;
        self.sin[pos * half_dim + dim]
    }
}

/// Fused RoPE + Attention helper
pub struct FusedRopeAttention {
    config: FusedRopeAttentionConfig,
    rope_cache: RopeCache,
}

impl FusedRopeAttention {
    pub fn new(config: FusedRopeAttentionConfig) -> Self {
        let rope_cache = RopeCache::compute(config.max_seq_len, config.head_dim, config.rope_base);

        Self { config, rope_cache }
    }

    /// Get config
    pub fn config(&self) -> &FusedRopeAttentionConfig {
        &self.config
    }

    /// Get RoPE cache
    pub fn rope_cache(&self) -> &RopeCache {
        &self.rope_cache
    }

    /// Get cos cache as slice
    pub fn cos_cache(&self) -> &[f32] {
        &self.rope_cache.cos
    }

    /// Get sin cache as slice
    pub fn sin_cache(&self) -> &[f32] {
        &self.rope_cache.sin
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_ops_config() {
        let config = FusedOpsConfig::for_model(32, 8, 128, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
    }

    #[test]
    fn test_rope_cache_computation() {
        let cache = RopeCache::compute(1024, 128, 10000.0);
        assert_eq!(cache.cos.len(), 1024 * 64); // max_seq_len * (head_dim/2)
        assert_eq!(cache.sin.len(), 1024 * 64);

        // Verify first position has cos(0) = 1 for first dimension
        assert!((cache.cos_at(0, 0) - 1.0).abs() < 1e-5);
        assert!((cache.sin_at(0, 0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fused_rope_attention() {
        let config = FusedRopeAttentionConfig::default();
        let fused = FusedRopeAttention::new(config);

        assert!(!fused.cos_cache().is_empty());
        assert!(!fused.sin_cache().is_empty());
    }

    #[test]
    fn test_cpu_fused_ops() {
        let config = FusedOpsConfig::default();
        let ops = CpuFusedOpsInfo::new(config);

        assert!(ops.supports(FusedOpType::RopeQK));
        assert!(ops.supports(FusedOpType::SiluGate));
    }

    #[test]
    fn test_fused_op_type_display() {
        assert_eq!(format!("{}", FusedOpType::RopeQK), "rope_qk");
        assert_eq!(format!("{}", FusedOpType::RopeAttention), "rope_attention");
    }
}
