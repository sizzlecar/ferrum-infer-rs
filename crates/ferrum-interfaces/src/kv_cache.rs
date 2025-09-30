//! KV-Cache abstraction with handle semantics and block management
//!
//! This module provides a sentence-handle based abstraction for KV cache management,
//! supporting both contiguous and paged attention patterns with zero-copy operations.

use crate::TensorRef;
use ferrum_types::{BlockId, Device, RequestId, Result};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::{collections::HashMap, sync::Arc};

/// Block table for mapping logical to physical cache blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTable {
    /// Physical block IDs allocated for this sequence
    pub physical_blocks: SmallVec<[BlockId; 8]>,
    /// Mapping from logical to physical block indices
    pub logical_to_physical: SmallVec<[u32; 8]>,
    /// Current sequence length in tokens
    pub sequence_length: usize,
    /// Block size (tokens per block)
    pub block_size: usize,
}

impl BlockTable {
    /// Create new block table
    pub fn new(block_size: usize) -> Self {
        Self {
            physical_blocks: SmallVec::new(),
            logical_to_physical: SmallVec::new(),
            sequence_length: 0,
            block_size,
        }
    }

    /// Get number of blocks allocated
    pub fn num_blocks(&self) -> usize {
        self.physical_blocks.len()
    }

    /// Get required number of blocks for sequence length
    pub fn blocks_needed_for_length(length: usize, block_size: usize) -> usize {
        (length + block_size - 1) / block_size // Ceiling division
    }

    /// Check if can accommodate more tokens without new blocks
    pub fn has_free_space(&self) -> bool {
        let used_blocks = Self::blocks_needed_for_length(self.sequence_length, self.block_size);
        used_blocks < self.num_blocks()
    }

    /// Get number of free tokens in allocated blocks
    pub fn free_tokens(&self) -> usize {
        if self.num_blocks() == 0 {
            0
        } else {
            self.num_blocks() * self.block_size - self.sequence_length
        }
    }

    /// Add blocks to the table
    pub fn add_blocks(&mut self, blocks: &[BlockId]) {
        let start_logical = self.logical_to_physical.len();

        for (i, &block) in blocks.iter().enumerate() {
            self.physical_blocks.push(block);
            self.logical_to_physical.push((start_logical + i) as u32);
        }
    }

    /// Extend sequence length
    pub fn extend_sequence(&mut self, additional_tokens: usize) -> Result<()> {
        let new_length = self.sequence_length + additional_tokens;
        let required_blocks = Self::blocks_needed_for_length(new_length, self.block_size);

        if required_blocks > self.num_blocks() {
            return Err(ferrum_types::FerrumError::backend(format!(
                "Insufficient blocks: need {}, have {}",
                required_blocks,
                self.num_blocks()
            )));
        }

        self.sequence_length = new_length;
        Ok(())
    }
}

/// KV cache handle providing access to cached key-value states
pub trait KvCacheHandle: Send + Sync + std::fmt::Debug {
    /// Get block table for this cache
    fn block_table(&self) -> &BlockTable;

    /// Get mutable block table (for extending)
    fn block_table_mut(&mut self) -> &mut BlockTable;

    /// Downcast support for backend-specific handles
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get device where cache resides
    fn device(&self) -> Device;

    /// Get number of tokens stored in cache
    fn num_tokens(&self) -> usize {
        self.block_table().sequence_length
    }

    /// Get number of layers cached
    fn num_layers(&self) -> usize;

    /// Get number of attention heads
    fn num_heads(&self) -> usize;

    /// Get head dimension
    fn head_dim(&self) -> usize;

    /// Get key cache for specific layer (returns tensor reference)
    fn key_cache(&self, layer: usize) -> Result<Option<TensorRef>>;

    /// Get value cache for specific layer
    fn value_cache(&self, layer: usize) -> Result<Option<TensorRef>>;

    /// Get both key and value caches for layer
    fn kv_cache(&self, layer: usize) -> Result<(Option<TensorRef>, Option<TensorRef>)> {
        Ok((self.key_cache(layer)?, self.value_cache(layer)?))
    }

    /// Clone handle (creates new reference, not deep copy)
    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>>;

    /// Get cache statistics
    fn stats(&self) -> CacheHandleStats;

    /// Check if cache is valid and accessible
    fn is_valid(&self) -> bool;

    /// Get unique identifier for this cache instance
    fn cache_id(&self) -> String;
}

/// Statistics for individual cache handle
#[derive(Debug, Clone)]
pub struct CacheHandleStats {
    /// Total memory usage in bytes
    pub memory_bytes: usize,
    /// Number of blocks allocated
    pub blocks_allocated: usize,
    /// Number of tokens stored
    pub tokens_stored: usize,
    /// Memory utilization ratio
    pub utilization: f32,
    /// Last access timestamp (for LRU)
    pub last_access: std::time::Instant,
}

/// KV cache allocation request
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    /// Request ID this allocation is for
    pub request_id: RequestId,
    /// Initial number of tokens
    pub initial_tokens: usize,
    /// Maximum expected sequence length
    pub max_sequence_length: usize,
    /// Number of layers to cache
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Target device
    pub device: Device,
    /// Data type for cache
    pub dtype: ferrum_types::DataType,
    /// Priority level for allocation
    pub priority: ferrum_types::Priority,
}

impl AllocationRequest {
    /// Calculate estimated memory requirement
    pub fn estimated_memory_bytes(&self) -> usize {
        // Key + Value cache size: layers * heads * max_seq * head_dim * 2 * dtype_size
        let kv_size =
            self.num_layers * self.num_heads * self.max_sequence_length * self.head_dim * 2;
        kv_size * self.dtype.size_bytes()
    }
}

/// KV cache manager for allocation and lifecycle management
#[async_trait::async_trait]
pub trait KvCacheManager: Send + Sync {
    /// Allocate cache for new sequence
    async fn allocate(&self, request: &AllocationRequest) -> Result<Arc<dyn KvCacheHandle>>;

    /// Extend existing cache to accommodate more tokens
    async fn extend(&self, handle: &mut dyn KvCacheHandle, additional_tokens: usize) -> Result<()>;

    /// Deallocate cache (handle becomes invalid)
    async fn deallocate(&self, request_id: RequestId) -> Result<()>;

    /// Check if can allocate requested cache size
    fn can_allocate(&self, request: &AllocationRequest) -> bool;

    /// Get cache statistics
    fn stats(&self) -> CacheManagerStats;

    /// Force garbage collection of unused caches
    async fn gc(&self) -> Result<CacheGcStats>;

    /// Set memory pressure callback
    fn set_pressure_callback(&self, callback: Box<dyn Fn(MemoryPressure) + Send + Sync>);

    /// Get handle for existing request (if exists)
    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn KvCacheHandle>>;

    /// List all active cache handles
    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn KvCacheHandle>)>;
}

/// Cache manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManagerStats {
    /// Total memory allocated in bytes
    pub total_memory_bytes: usize,
    /// Memory currently in use
    pub used_memory_bytes: usize,
    /// Number of active caches
    pub active_caches: usize,
    /// Total blocks allocated
    pub total_blocks: usize,
    /// Free blocks available
    pub free_blocks: usize,
    /// Cache hit rate (for prefix caching)
    pub cache_hit_rate: f32,
    /// Number of evictions performed
    pub eviction_count: u64,
    /// Number of successful allocations
    pub allocation_count: u64,
    /// Number of failed allocations
    pub allocation_failures: u64,
}

/// Garbage collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheGcStats {
    /// Memory freed in bytes
    pub memory_freed: usize,
    /// Number of caches garbage collected
    pub caches_freed: usize,
    /// Time taken for GC
    pub gc_time_ms: u64,
}

/// Memory pressure levels for adaptive management
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// Low memory usage, allocations can proceed freely
    Low,
    /// Moderate usage, start being more conservative
    Medium,
    /// High usage, consider eviction
    High,
    /// Critical usage, must evict or reject allocations
    Critical,
}

/// Advanced KV cache capabilities
pub trait AdvancedKvCacheManager: KvCacheManager {
    /// Enable prefix caching for common prompt prefixes
    async fn enable_prefix_caching(&self, config: PrefixCacheConfig) -> Result<()>;

    /// Share cache blocks between compatible sequences
    async fn share_prefix(
        &self,
        source: RequestId,
        target: RequestId,
        shared_tokens: usize,
    ) -> Result<()>;

    /// Swap cache from GPU to CPU to free GPU memory
    async fn swap_out(&self, request_id: RequestId) -> Result<()>;

    /// Swap cache from CPU back to GPU
    async fn swap_in(&self, request_id: RequestId) -> Result<()>;

    /// Compress cache to reduce memory usage
    async fn compress_cache(&self, request_id: RequestId, compression_ratio: f32) -> Result<()>;

    /// Get cache compression statistics
    fn compression_stats(&self) -> CompressionStats;
}

/// Prefix caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    /// Maximum number of prefixes to cache
    pub max_prefixes: usize,
    /// Minimum prefix length to be eligible for caching
    pub min_prefix_length: usize,
    /// TTL for cached prefixes
    pub prefix_ttl_seconds: u64,
    /// Enable cross-request prefix sharing
    pub enable_cross_request_sharing: bool,
}

/// Cache compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Number of compressed caches
    pub compressed_caches: usize,
    /// Total memory saved by compression
    pub memory_saved_bytes: usize,
    /// Average compression ratio achieved
    pub avg_compression_ratio: f32,
    /// Compression/decompression time overhead
    pub avg_compression_time_ms: f64,
}

/// Block-based cache allocator
pub trait BlockAllocator: Send + Sync {
    /// Allocate specified number of blocks
    fn allocate_blocks(&self, num_blocks: usize) -> Result<Vec<BlockId>>;

    /// Free blocks back to allocator
    fn free_blocks(&self, blocks: &[BlockId]) -> Result<()>;

    /// Get number of free blocks
    fn free_block_count(&self) -> usize;

    /// Get total block count
    fn total_block_count(&self) -> usize;

    /// Get block size in tokens
    fn block_size(&self) -> usize;

    /// Defragment free block list
    fn defragment(&self) -> Result<()>;
}

/// Multi-device cache manager supporting GPU/CPU hierarchies
#[async_trait::async_trait]
pub trait MultiDeviceCacheManager: KvCacheManager {
    /// Get supported devices
    fn supported_devices(&self) -> Vec<Device>;

    /// Set device preference for new allocations
    fn set_device_preference(&self, devices: Vec<Device>);

    /// Move cache between devices
    async fn move_cache(&self, request_id: RequestId, target_device: Device) -> Result<()>;

    /// Get cache location
    fn get_cache_device(&self, request_id: RequestId) -> Option<Device>;

    /// Balance cache distribution across devices
    async fn rebalance_devices(&self) -> Result<()>;

    /// Get per-device statistics
    fn device_stats(&self) -> HashMap<Device, CacheManagerStats>;
}

/// Cache eviction strategies
pub trait CacheEvictionPolicy: Send + Sync {
    /// Select caches to evict to free requested memory
    fn select_eviction_candidates(
        &self,
        required_memory: usize,
        active_caches: &[(RequestId, Arc<dyn KvCacheHandle>)],
    ) -> Vec<RequestId>;

    /// Update cache access information
    fn record_access(&mut self, request_id: RequestId, access_time: std::time::Instant);

    /// Get policy name
    fn name(&self) -> &str;
}

/// Least Recently Used eviction policy
pub struct LruEvictionPolicy {
    access_times: HashMap<RequestId, std::time::Instant>,
}

impl LruEvictionPolicy {
    pub fn new() -> Self {
        Self {
            access_times: HashMap::new(),
        }
    }
}

impl CacheEvictionPolicy for LruEvictionPolicy {
    fn select_eviction_candidates(
        &self,
        required_memory: usize,
        active_caches: &[(RequestId, Arc<dyn KvCacheHandle>)],
    ) -> Vec<RequestId> {
        let mut candidates: Vec<_> = active_caches
            .iter()
            .map(|(req_id, handle)| {
                let access_time = self
                    .access_times
                    .get(req_id)
                    .copied()
                    .unwrap_or_else(std::time::Instant::now);
                (req_id.clone(), handle.stats().memory_bytes, access_time)
            })
            .collect();

        // Sort by access time (oldest first)
        candidates.sort_by(|a, b| a.2.cmp(&b.2));

        let mut freed_memory = 0;
        let mut result = Vec::new();

        for (req_id, memory_bytes, _) in candidates {
            result.push(req_id);
            freed_memory += memory_bytes;
            if freed_memory >= required_memory {
                break;
            }
        }

        result
    }

    fn record_access(&mut self, request_id: RequestId, access_time: std::time::Instant) {
        self.access_times.insert(request_id, access_time);
    }

    fn name(&self) -> &str {
        "lru"
    }
}

impl Default for LruEvictionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Block size in tokens
    pub block_size: usize,
    /// Maximum number of blocks
    pub max_blocks: usize,
    /// Initial number of blocks to allocate
    pub initial_blocks: usize,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Target devices for allocation
    pub target_devices: Vec<Device>,
    /// Enable prefix caching
    pub enable_prefix_caching: bool,
    /// Prefix cache configuration
    pub prefix_cache_config: Option<PrefixCacheConfig>,
    /// Enable multi-device support
    pub enable_multi_device: bool,
    /// Memory pressure thresholds
    pub pressure_thresholds: MemoryPressureThresholds,
}

/// Memory pressure threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureThresholds {
    /// Medium pressure threshold (0.0-1.0)
    pub medium_threshold: f32,
    /// High pressure threshold (0.0-1.0)
    pub high_threshold: f32,
    /// Critical pressure threshold (0.0-1.0)
    pub critical_threshold: f32,
}

impl Default for MemoryPressureThresholds {
    fn default() -> Self {
        Self {
            medium_threshold: 0.6,
            high_threshold: 0.8,
            critical_threshold: 0.95,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_blocks: 1000,
            initial_blocks: 100,
            enable_pooling: true,
            target_devices: vec![Device::CPU],
            enable_prefix_caching: false,
            prefix_cache_config: None,
            enable_multi_device: false,
            pressure_thresholds: MemoryPressureThresholds::default(),
        }
    }
}
