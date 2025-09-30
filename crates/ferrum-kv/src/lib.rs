//! # Ferrum KV Cache
//!
//! KV-Cache management and PagedAttention block pool implementations.
//!
//! ## Overview
//!
//! This crate provides concrete implementations of the KV-Cache interfaces defined
//! in `ferrum-interfaces`, including:
//!
//! - Block-based memory management (GPU/CPU two-tier)
//! - Block table mapping (logical → physical blocks)
//! - Swap manager for efficient memory usage
//! - Prefix caching for shared prompt optimization  
//! - Eviction policies and memory pressure handling
//! - Optional compression support (int4/fp8)
//!
//! ## Design Principles
//!
//! - **Block-Based Allocation**: Fixed-size blocks for efficient memory management.
//! - **Two-Tier Memory**: GPU primary + CPU swap for large contexts.
//! - **Prefix Sharing**: Automatic detection and reuse of common prefixes.
//! - **Handle-Based API**: Zero-copy operations through opaque handles.

pub mod blocks;
pub mod cache;
pub mod managers;

// Re-exports of interfaces from ferrum-interfaces
pub use ferrum_interfaces::{
    AllocationRequest, BlockTable, CacheStats, CompressionConfig, EvictionPolicy,
    KvCacheHandle as KvCacheHandleInterface, KvCacheManager as KvCacheManagerInterface,
    PrefixCacheConfig,
};

pub use ferrum_types::{DataType, Device, FerrumError, RequestId, Result};

// Re-exports of implementations
pub use blocks::*;
pub use cache::*;
pub use managers::*;

/// Default KV cache manager factory
pub fn default_manager(
    device: Device,
    config: KvCacheConfig,
) -> Result<Box<dyn KvCacheManagerInterface + Send + Sync>> {
    let manager = DefaultKvCacheManager::new(device, config)?;
    Ok(Box::new(manager))
}

/// KV Cache configuration
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Block size in tokens
    pub block_size: usize,
    /// Maximum number of blocks in GPU memory
    pub max_blocks_gpu: usize,
    /// Maximum number of blocks in CPU memory
    pub max_blocks_cpu: usize,
    /// Enable prefix caching
    pub enable_prefix_cache: bool,
    /// Eviction policy to use
    pub eviction_policy: EvictionPolicy,
    /// Optional compression configuration
    pub compression_config: Option<CompressionConfig>,
    /// Enable detailed metrics collection
    pub enable_metrics: bool,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_blocks_gpu: 1024,
            max_blocks_cpu: 4096,
            enable_prefix_cache: true,
            eviction_policy: EvictionPolicy::LRU,
            compression_config: None,
            enable_metrics: true,
        }
    }
}
