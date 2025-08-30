//! # Ferrum Cache
//! 
//! Cache management traits and abstractions for LLM inference with PagedAttention support.
//! 
//! ## Overview
//! 
//! This module defines the core traits for implementing various caching strategies,
//! particularly focused on PagedAttention for efficient KV cache management.
//! 
//! ## Design Principles
//! 
//! - **Abstract Interfaces**: Only trait definitions, no concrete implementations
//! - **Backend Agnostic**: Can be implemented with different memory management strategies
//! - **Composable**: Traits can be combined to build complex caching systems

pub mod traits;
pub mod types;

// Re-exports
pub use traits::{
    CacheManager, BlockManager, MemoryPool, SwapManager,
    PrefixCacheManager, CacheEvictionPolicy, CacheAllocator
};

pub use types::{
    CacheBlock, BlockTable, CacheStats, MemoryLocation,
    AllocationRequest, SwapRequest, EvictionPolicy
};