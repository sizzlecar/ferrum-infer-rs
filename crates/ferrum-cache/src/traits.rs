//! Core cache management traits
//!
//! This module defines the abstract interfaces for cache management in LLM inference,
//! with a focus on PagedAttention and efficient memory utilization.

use crate::types::*;
use async_trait::async_trait;
use ferrum_types::{BlockId, RequestId, Result};

/// Main cache manager trait for KV cache management
#[async_trait]
pub trait CacheManager: Send + Sync {
    /// Allocate cache blocks for a request
    async fn allocate(&self, request: AllocationRequest) -> Result<BlockTable>;

    /// Deallocate cache blocks for a request
    async fn deallocate(&self, request_id: RequestId) -> Result<()>;

    /// Resize allocation for a request (e.g., during generation)
    async fn resize(&self, request_id: RequestId, new_size: usize) -> Result<BlockTable>;

    /// Get block table for a request
    async fn get_block_table(&self, request_id: RequestId) -> Result<BlockTable>;

    /// Get cache statistics
    fn get_stats(&self) -> CacheStats;

    /// Check if cache can accommodate a request
    fn can_allocate(&self, num_blocks: usize) -> bool;

    /// Clear all cache entries
    async fn clear(&self) -> Result<()>;
}

/// Block manager for PagedAttention
#[async_trait]
pub trait BlockManager: Send + Sync {
    /// Allocate physical blocks
    fn allocate_blocks(&self, num_blocks: usize) -> Result<Vec<BlockId>>;

    /// Free physical blocks
    fn free_blocks(&self, blocks: &[BlockId]) -> Result<()>;

    /// Get number of free blocks
    fn num_free_blocks(&self) -> usize;

    /// Get total number of blocks
    fn total_blocks(&self) -> usize;

    /// Fork blocks for copy-on-write
    fn fork_blocks(&self, src_blocks: &[BlockId]) -> Result<Vec<BlockId>>;

    /// Get block utilization
    fn utilization(&self) -> f32;
}

/// Memory pool for managing physical memory
#[async_trait]
pub trait MemoryPool: Send + Sync {
    /// Allocate memory for a block
    async fn allocate(&self, block_id: BlockId) -> Result<()>;

    /// Deallocate memory for a block
    async fn deallocate(&self, block_id: BlockId) -> Result<()>;

    /// Read data from a block
    async fn read(&self, block_id: BlockId) -> Result<CacheBlock>;

    /// Write data to a block
    async fn write(&self, block_id: BlockId, data: &CacheBlock) -> Result<()>;

    /// Get memory location of a block
    fn get_location(&self, block_id: BlockId) -> Option<MemoryLocation>;

    /// Get available memory in bytes
    fn available_memory(&self) -> usize;

    /// Get total memory in bytes
    fn total_memory(&self) -> usize;
}

/// Swap manager for GPU-CPU memory swapping
#[async_trait]
pub trait SwapManager: Send + Sync {
    /// Swap blocks from GPU to CPU
    async fn swap_out(&self, request: SwapRequest) -> Result<()>;

    /// Swap blocks from CPU to GPU
    async fn swap_in(&self, request: SwapRequest) -> Result<()>;

    /// Get blocks that can be swapped out
    fn get_swappable_blocks(&self) -> Vec<BlockId>;

    /// Check if swapping is needed
    fn should_swap(&self) -> bool;

    /// Get swap statistics
    fn get_swap_stats(&self) -> SwapStats;
}

/// Prefix cache manager for sharing common prefixes
#[async_trait]
pub trait PrefixCacheManager: Send + Sync {
    /// Check if a prefix exists in cache
    async fn has_prefix(&self, tokens: &[u32]) -> bool;

    /// Get blocks for a cached prefix
    async fn get_prefix_blocks(&self, tokens: &[u32]) -> Option<Vec<BlockId>>;

    /// Add a prefix to cache
    async fn cache_prefix(&self, tokens: &[u32], blocks: Vec<BlockId>) -> Result<()>;

    /// Remove a prefix from cache
    async fn evict_prefix(&self, tokens: &[u32]) -> Result<()>;

    /// Get prefix cache statistics
    fn get_prefix_stats(&self) -> PrefixCacheStats;
}

/// Cache eviction policy trait
pub trait CacheEvictionPolicy: Send + Sync {
    /// Select blocks to evict
    fn select_victims(&self, num_blocks: usize, candidates: &[BlockId]) -> Vec<BlockId>;

    /// Update access information for a block
    fn access(&self, block_id: BlockId);

    /// Get eviction policy name
    fn name(&self) -> &str;
}

/// Cache allocator trait for different allocation strategies
pub trait CacheAllocator: Send + Sync {
    /// Allocate blocks with a specific strategy
    fn allocate(
        &self,
        request: &AllocationRequest,
        free_blocks: &[BlockId],
    ) -> Result<Vec<BlockId>>;

    /// Suggest blocks to free for a new allocation
    fn suggest_eviction(&self, request: &AllocationRequest, all_blocks: &[BlockId])
        -> Vec<BlockId>;

    /// Get allocator name
    fn name(&self) -> &str;
}

/// Swap statistics
#[derive(Debug, Clone, Default)]
pub struct SwapStats {
    pub total_swaps_out: u64,
    pub total_swaps_in: u64,
    pub current_swapped_blocks: usize,
    pub swap_out_bytes: u64,
    pub swap_in_bytes: u64,
}

/// Prefix cache statistics
#[derive(Debug, Clone, Default)]
pub struct PrefixCacheStats {
    pub num_prefixes: usize,
    pub total_prefix_blocks: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}
