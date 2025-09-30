//! Type definitions for cache management
//!
//! This module defines the core types used throughout the cache system.

use ferrum_types::{BlockId, ModelId, RequestId};
use serde::{Deserialize, Serialize};

/// Cache block containing KV cache data
#[derive(Debug, Clone)]
pub struct CacheBlock {
    /// Block identifier
    pub block_id: BlockId,

    /// Block size in tokens
    pub block_size: usize,

    /// Number of tokens currently stored
    pub num_tokens: usize,

    /// Layer index this block belongs to
    pub layer_idx: usize,

    /// Whether this is a key or value block
    pub is_key: bool,

    /// Reference count for copy-on-write
    pub ref_count: usize,
}

/// Block table mapping logical to physical blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTable {
    /// Request ID this table belongs to
    pub request_id: RequestId,

    /// Physical block IDs in order
    pub blocks: Vec<BlockId>,

    /// Number of tokens per block
    pub block_size: usize,

    /// Total number of tokens
    pub num_tokens: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of cache blocks
    pub total_blocks: usize,

    /// Number of free blocks
    pub free_blocks: usize,

    /// Number of allocated blocks
    pub allocated_blocks: usize,

    /// Cache hit count
    pub hit_count: u64,

    /// Cache miss count
    pub miss_count: u64,

    /// Total allocation requests
    pub allocation_count: u64,

    /// Total deallocation requests
    pub deallocation_count: u64,

    /// Current memory usage in bytes
    pub memory_usage_bytes: usize,

    /// Peak memory usage in bytes
    pub peak_memory_usage_bytes: usize,
}

/// Memory location for blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryLocation {
    /// Block is in GPU memory
    GPU(usize), // GPU device ID

    /// Block is in CPU memory
    CPU,

    /// Block is in disk storage
    Disk,
}

/// Allocation request
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    /// Request ID
    pub request_id: RequestId,

    /// Model ID
    pub model_id: ModelId,

    /// Number of tokens to allocate for
    pub num_tokens: usize,

    /// Block size preference
    pub block_size: usize,

    /// Preferred memory location
    pub preferred_location: MemoryLocation,

    /// Priority for allocation
    pub priority: AllocationPriority,
}

/// Allocation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AllocationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Swap request
#[derive(Debug, Clone)]
pub struct SwapRequest {
    /// Blocks to swap
    pub blocks: Vec<BlockId>,

    /// Source location
    pub from: MemoryLocation,

    /// Destination location
    pub to: MemoryLocation,

    /// Whether this is an urgent swap
    pub urgent: bool,
}

/// Eviction policy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,

    /// Least Frequently Used
    LFU,

    /// First In First Out
    FIFO,

    /// Random eviction
    Random,

    /// Priority-based eviction
    Priority,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Total number of GPU blocks
    pub num_gpu_blocks: usize,

    /// Total number of CPU blocks
    pub num_cpu_blocks: usize,

    /// Block size in tokens
    pub block_size: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Number of KV heads
    pub num_kv_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Data type size in bytes
    pub dtype_size: usize,

    /// Enable swapping
    pub enable_swap: bool,

    /// Enable prefix caching
    pub enable_prefix_cache: bool,

    /// Eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Swap space limit in bytes
    pub swap_space_bytes: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            num_gpu_blocks: 1024,
            num_cpu_blocks: 2048,
            block_size: 16,
            num_layers: 32,
            num_kv_heads: 32,
            head_dim: 128,
            dtype_size: 2, // FP16
            enable_swap: true,
            enable_prefix_cache: true,
            eviction_policy: EvictionPolicy::LRU,
            swap_space_bytes: Some(32 * 1024 * 1024 * 1024), // 32GB
        }
    }
}

/// Block metadata for tracking
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    /// Block ID
    pub block_id: BlockId,

    /// Last access timestamp
    pub last_access: std::time::Instant,

    /// Access count
    pub access_count: u64,

    /// Owning request
    pub owner: Option<RequestId>,

    /// Whether block is shared
    pub is_shared: bool,
}

/// Allocation result
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Allocated block table
    pub block_table: BlockTable,

    /// Whether allocation required eviction
    pub evicted: bool,

    /// Evicted request IDs if any
    pub evicted_requests: Vec<RequestId>,
}
