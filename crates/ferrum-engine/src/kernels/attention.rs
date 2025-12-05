//! Attention Kernel Trait Definitions
//!
//! This module provides trait definitions for various attention implementations:
//! - Standard Multi-Head Attention
//! - Flash Attention (memory efficient)
//! - Paged Attention (for KV cache)
//! - Grouped Query Attention (GQA)
//!
//! Actual implementations are provided by backend-specific modules.

use ferrum_types::{Device, Result};
use std::fmt;

/// Type of attention mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard multi-head attention
    Standard,
    /// Flash Attention (memory efficient)
    Flash,
    /// Paged Attention for KV cache
    Paged,
    /// Grouped Query Attention
    GroupedQuery,
    /// Multi-Query Attention
    MultiQuery,
    /// Sliding Window Attention
    SlidingWindow,
}

impl fmt::Display for AttentionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttentionType::Standard => write!(f, "standard"),
            AttentionType::Flash => write!(f, "flash"),
            AttentionType::Paged => write!(f, "paged"),
            AttentionType::GroupedQuery => write!(f, "gqa"),
            AttentionType::MultiQuery => write!(f, "mqa"),
            AttentionType::SlidingWindow => write!(f, "sliding_window"),
        }
    }
}

/// Configuration for attention computation
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Whether to use causal (autoregressive) masking
    pub causal: bool,
    /// Softmax scale (usually 1/sqrt(head_dim))
    pub softmax_scale: f32,
    /// Attention dropout rate (training only)
    pub dropout: f32,
    /// Sliding window size (for sliding window attention)
    pub sliding_window: Option<usize>,
    /// Block size for paged attention
    pub block_size: usize,
    /// Device to run on
    pub device: Device,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len: 4096,
            causal: true,
            softmax_scale: 0.088388, // 1/sqrt(128)
            dropout: 0.0,
            sliding_window: None,
            block_size: 16,
            device: Device::CPU,
        }
    }
}

impl AttentionConfig {
    /// Create config for standard attention
    pub fn standard(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }

    /// Create config for GQA
    pub fn grouped_query(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }

    /// Create config for Flash Attention
    pub fn flash(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        }
    }

    /// Enable sliding window attention
    pub fn with_sliding_window(mut self, window_size: usize) -> Self {
        self.sliding_window = Some(window_size);
        self
    }

    /// Set device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Get groups per query head (for GQA)
    pub fn kv_groups(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Check if this is GQA configuration
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// Check if this is MQA configuration
    pub fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }
}

/// Attention kernel trait
///
/// Implementations of this trait provide specific attention algorithms.
/// The actual tensor operations are handled by the backend.
pub trait AttentionKernel: Send + Sync {
    /// Get the attention type
    fn attention_type(&self) -> AttentionType;

    /// Get configuration
    fn config(&self) -> &AttentionConfig;

    /// Get supported devices
    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    /// Estimated memory usage for given sequence length
    fn memory_estimate(&self, batch_size: usize, seq_len: usize) -> usize;

    /// Get a description of this kernel
    fn description(&self) -> &str;
}

/// Standard attention kernel info
pub struct StandardAttentionInfo {
    config: AttentionConfig,
}

impl StandardAttentionInfo {
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }
}

impl AttentionKernel for StandardAttentionInfo {
    fn attention_type(&self) -> AttentionType {
        AttentionType::Standard
    }

    fn config(&self) -> &AttentionConfig {
        &self.config
    }

    fn memory_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        // Standard attention: O(n^2) memory for attention weights
        let config = self.config();
        let attention_size = batch_size * config.num_heads * seq_len * seq_len * 4; // float32
        let output_size = batch_size * config.num_heads * seq_len * config.head_dim * 4;
        attention_size + output_size
    }

    fn description(&self) -> &str {
        "Standard multi-head attention with O(n^2) memory"
    }
}

/// Flash Attention kernel info
pub struct FlashAttentionInfo {
    config: AttentionConfig,
}

impl FlashAttentionInfo {
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }
}

impl AttentionKernel for FlashAttentionInfo {
    fn attention_type(&self) -> AttentionType {
        AttentionType::Flash
    }

    fn config(&self) -> &AttentionConfig {
        &self.config
    }

    fn memory_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        // Flash attention: O(n) memory instead of O(n^2)
        let config = self.config();
        let chunk_size = 64.min(seq_len);

        // Only need to store one chunk of attention weights at a time
        let attention_chunk = batch_size * config.num_heads * seq_len * chunk_size * 4;
        let output_size = batch_size * config.num_heads * seq_len * config.head_dim * 4;
        let accumulator = batch_size * config.num_heads * seq_len * 2 * 4; // m and l

        attention_chunk + output_size + accumulator
    }

    fn description(&self) -> &str {
        "Flash Attention with O(n) memory using online softmax"
    }
}

/// Paged Attention kernel info
pub struct PagedAttentionInfo {
    config: AttentionConfig,
}

impl PagedAttentionInfo {
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }
}

impl AttentionKernel for PagedAttentionInfo {
    fn attention_type(&self) -> AttentionType {
        AttentionType::Paged
    }

    fn config(&self) -> &AttentionConfig {
        &self.config
    }

    fn memory_estimate(&self, batch_size: usize, seq_len: usize) -> usize {
        // Paged attention uses block-based KV cache
        let config = self.config();
        let num_blocks = (seq_len + config.block_size - 1) / config.block_size;
        let block_memory = num_blocks * config.block_size * config.head_dim * 2 * 4; // K and V
        let output_size = batch_size * config.num_heads * config.head_dim * 4;

        batch_size * config.num_kv_heads * block_memory + output_size
    }

    fn description(&self) -> &str {
        "Paged Attention for efficient KV cache with block-based memory"
    }
}

/// Factory for creating attention kernel info
pub fn create_attention_info(
    attention_type: AttentionType,
    config: AttentionConfig,
) -> Result<Box<dyn AttentionKernel>> {
    match attention_type {
        AttentionType::Standard => Ok(Box::new(StandardAttentionInfo::new(config))),
        AttentionType::Flash => Ok(Box::new(FlashAttentionInfo::new(config))),
        AttentionType::Paged => Ok(Box::new(PagedAttentionInfo::new(config))),
        AttentionType::GroupedQuery => {
            // GQA uses Flash attention with proper head mapping
            Ok(Box::new(FlashAttentionInfo::new(config)))
        }
        AttentionType::MultiQuery => {
            let mut mqa_config = config;
            mqa_config.num_kv_heads = 1;
            Ok(Box::new(FlashAttentionInfo::new(mqa_config)))
        }
        AttentionType::SlidingWindow => Ok(Box::new(StandardAttentionInfo::new(config))),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_creation() {
        let config = AttentionConfig::standard(32, 128);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(!config.is_gqa());
    }

    #[test]
    fn test_gqa_config() {
        let config = AttentionConfig::grouped_query(32, 8, 128);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert!(config.is_gqa());
        assert_eq!(config.kv_groups(), 4);
    }

    #[test]
    fn test_mqa_config() {
        let config = AttentionConfig::grouped_query(32, 1, 128);
        assert!(config.is_mqa());
    }

    #[test]
    fn test_memory_estimate() {
        let config = AttentionConfig::standard(32, 128);

        let standard = StandardAttentionInfo::new(config.clone());
        let flash = FlashAttentionInfo::new(config);

        let seq_len = 1024;
        let batch = 1;

        let standard_mem = standard.memory_estimate(batch, seq_len);
        let flash_mem = flash.memory_estimate(batch, seq_len);

        // Flash attention should use less memory
        assert!(
            flash_mem < standard_mem,
            "Flash: {}, Standard: {}",
            flash_mem,
            standard_mem
        );
    }

    #[test]
    fn test_attention_type_display() {
        assert_eq!(format!("{}", AttentionType::Flash), "flash");
        assert_eq!(format!("{}", AttentionType::Paged), "paged");
    }

    #[test]
    fn test_create_attention_info() {
        let config = AttentionConfig::standard(32, 128);
        let info = create_attention_info(AttentionType::Flash, config);
        assert!(info.is_ok());
        assert_eq!(info.unwrap().attention_type(), AttentionType::Flash);
    }
}
