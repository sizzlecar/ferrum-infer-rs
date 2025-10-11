//! # Ferrum KV Cache
//!
//! MVP KV-Cache management implementation for Ferrum inference stack.
//!
//! This crate provides block-based KV cache management, implementing the
//! interfaces defined in `ferrum-interfaces::kv_cache`.

pub mod blocks;
pub mod cache;
pub mod managers;

// Re-export interface types
pub use ferrum_interfaces::{
    kv_cache::{
        AllocationRequest, BlockTable, CacheConfig, CacheEvictionPolicy, CacheHandleStats,
        CacheManagerStats, LruEvictionPolicy, PrefixCacheConfig,
    },
    KvCacheHandle as KvCacheHandleInterface,
    KvCacheManager as KvCacheManagerInterface,
};

pub use ferrum_types::{CacheStats, DataType, Device, FerrumError, RequestId, Result};
// Note: ferrum-types::KvCacheConfig exists but has different fields for engine-level config
// This crate uses a simplified internal config

// Re-export implementations
pub use blocks::*;
pub use cache::*;
pub use managers::*;

/// Default KV cache manager factory
pub fn default_manager(
    device: Device,
    block_size: usize,
    max_blocks: usize,
) -> Result<Box<dyn KvCacheManagerInterface + Send + Sync>> {
    let manager = DefaultKvCacheManager::new(device, block_size, max_blocks)?;
    Ok(Box::new(manager))
}

/// Internal KV Cache manager configuration
/// 
/// Note: This is distinct from ferrum_types::KvCacheConfig which is the engine-level
/// configuration. This type is used internally by the KV cache manager implementation.
#[derive(Debug, Clone)]
pub struct KvManagerConfig {
    pub block_size: usize,
    pub max_blocks_gpu: usize,
    pub max_blocks_cpu: usize,
    pub enable_prefix_cache: bool,
    pub enable_metrics: bool,
}