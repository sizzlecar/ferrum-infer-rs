//! Default KV cache manager implementation

use crate::{KvCacheConfig, DefaultKvCacheHandle};
use crate::blocks::{BlockPool, PhysicalBlockId, Block};
use ferrum_interfaces::{KvCacheManager, AllocationRequest, CacheStats};
use ferrum_types::{Result, RequestId, Device, DataType, FerrumError};
use parking_lot::{RwLock, Mutex};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn, error};

/// Default KV cache manager implementation
#[derive(Debug)]
pub struct DefaultKvCacheManager {
    /// Configuration
    config: KvCacheConfig,
    /// Primary device (GPU) block pool
    gpu_pool: Option<BlockPool>,
    /// Secondary device (CPU) block pool for swapping
    cpu_pool: Option<BlockPool>,
    /// Active cache handles by request ID
    active_handles: RwLock<HashMap<RequestId, Arc<RwLock<DefaultKvCacheHandle>>>>,
    /// Statistics
    stats: Mutex<CacheStatsInternal>,
}

#[derive(Debug, Default)]
struct CacheStatsInternal {
    total_requests: usize,
    active_requests: usize,
    total_blocks_allocated: usize,
    total_blocks_freed: usize,
    cache_hits: usize,
    cache_misses: usize,
    evictions: usize,
    swap_ins: usize,
    swap_outs: usize,
}

impl DefaultKvCacheManager {
    /// Create new KV cache manager
    pub fn new(device: Device, config: KvCacheConfig) -> Result<Self> {
        debug!("Creating KV cache manager for device: {:?}", device);

        // Create GPU pool if device supports it
        let gpu_pool = match device {
            Device::Cuda(_) | Device::Metal(_) => {
                Some(BlockPool::new(
                    device.clone(),
                    config.block_size,
                    DataType::F16, // Default to FP16 for KV cache
                    config.max_blocks_gpu,
                )?)
            }
            Device::Cpu => None,
        };

        // Create CPU pool for swapping or as primary storage
        let cpu_pool = if config.max_blocks_cpu > 0 {
            Some(BlockPool::new(
                Device::Cpu,
                config.block_size,
                DataType::F16,
                config.max_blocks_cpu,
            )?)
        } else {
            None
        };

        // Ensure we have at least one pool
        if gpu_pool.is_none() && cpu_pool.is_none() {
            return Err(FerrumError::invalid_parameter(
                "Must have at least one block pool (GPU or CPU)"
            ));
        }

        Ok(Self {
            config,
            gpu_pool,
            cpu_pool,
            active_handles: RwLock::new(HashMap::new()),
            stats: Mutex::new(CacheStatsInternal::default()),
        })
    }

    /// Get primary block pool (GPU preferred, fallback to CPU)
    fn primary_pool(&self) -> &BlockPool {
        self.gpu_pool.as_ref().unwrap_or_else(|| {
            self.cpu_pool.as_ref().expect("No block pools available")
        })
    }

    /// Get secondary block pool for swapping
    fn secondary_pool(&self) -> Option<&BlockPool> {
        if self.gpu_pool.is_some() {
            self.cpu_pool.as_ref()
        } else {
            None
        }
    }

    /// Calculate required blocks for allocation
    fn calculate_required_blocks(&self, request: &AllocationRequest) -> usize {
        let tokens_per_block = self.config.block_size;
        (request.num_tokens + tokens_per_block - 1) / tokens_per_block // Ceiling division
    }

    /// Try to allocate blocks from primary pool
    fn try_allocate_blocks(&self, num_blocks: usize) -> Result<Vec<(PhysicalBlockId, Arc<RwLock<Block>>)>> {
        let primary = self.primary_pool();
        let mut allocated_blocks = Vec::new();

        for _ in 0..num_blocks {
            match primary.allocate() {
                Ok(allocation) => {
                    allocated_blocks.push((allocation.physical_id, allocation.block));
                }
                Err(e) => {
                    // Clean up already allocated blocks
                    for (block_id, _) in allocated_blocks {
                        let _ = primary.deallocate(block_id);
                    }
                    return Err(e);
                }
            }
        }

        Ok(allocated_blocks)
    }

    /// Free blocks
    fn free_blocks(&self, blocks: Vec<(PhysicalBlockId, Arc<RwLock<Block>>)>) {
        let primary = self.primary_pool();
        
        for (block_id, _) in blocks {
            if let Err(e) = primary.deallocate(block_id) {
                warn!("Failed to deallocate block {:?}: {}", block_id, e);
            }
        }
    }

    /// Update statistics
    fn update_stats<F>(&self, f: F) 
    where
        F: FnOnce(&mut CacheStatsInternal),
    {
        if self.config.enable_metrics {
            let mut stats = self.stats.lock();
            f(&mut stats);
        }
    }
}

impl KvCacheManager for DefaultKvCacheManager {
    fn allocate(&self, request: &AllocationRequest) -> Result<Box<dyn ferrum_interfaces::KvCacheHandle + Send + Sync>> {
        debug!("Allocating KV cache for request: {:?}", request.request_id);

        let required_blocks = self.calculate_required_blocks(request);
        
        // Check if we can allocate
        if !self.can_allocate(required_blocks) {
            self.update_stats(|s| s.cache_misses += 1);
            return Err(FerrumError::resource_exhausted(
                format!("Cannot allocate {} blocks for request", required_blocks)
            ));
        }

        // Allocate blocks
        let allocated_blocks = self.try_allocate_blocks(required_blocks)?;
        
        // Create handle
        let device = self.primary_pool().device().clone();
        let mut handle = DefaultKvCacheHandle::new(
            request.request_id,
            device,
            request.num_layers.unwrap_or(32),
            request.num_heads.unwrap_or(32), 
            request.head_dim.unwrap_or(128),
        );

        // Add blocks to handle
        for (physical_id, block) in allocated_blocks {
            handle.add_block(physical_id, block);
        }
        
        handle.set_num_tokens(request.num_tokens);

        // Store handle
        let handle_arc = Arc::new(RwLock::new(handle));
        {
            let mut active_handles = self.active_handles.write();
            active_handles.insert(request.request_id, handle_arc.clone());
        }

        // Update statistics
        self.update_stats(|s| {
            s.total_requests += 1;
            s.active_requests = active_handles.len(); // This is approximate due to lock timing
            s.total_blocks_allocated += required_blocks;
            s.cache_hits += 1;
        });

        let boxed_handle: Box<dyn ferrum_interfaces::KvCacheHandle + Send + Sync> = Box::new(
            handle_arc.read().clone()
        );
        Ok(boxed_handle)
    }

    fn resize(&self, request_id: RequestId, new_size: usize) -> Result<Box<dyn ferrum_interfaces::KvCacheHandle + Send + Sync>> {
        debug!("Resizing KV cache for request: {:?} to {} tokens", request_id, new_size);

        let active_handles = self.active_handles.read();
        if let Some(handle_arc) = active_handles.get(&request_id) {
            let mut handle = handle_arc.write();
            let current_tokens = handle.num_tokens();
            
            if new_size > current_tokens {
                // Need to allocate more blocks
                let current_blocks = handle.physical_blocks().len();
                let required_blocks = self.calculate_required_blocks(&AllocationRequest {
                    request_id,
                    num_tokens: new_size,
                    num_layers: Some(handle.dimensions().0),
                    num_heads: Some(handle.dimensions().1),
                    head_dim: Some(handle.dimensions().2),
                });
                
                if required_blocks > current_blocks {
                    let additional_blocks = required_blocks - current_blocks;
                    let new_blocks = self.try_allocate_blocks(additional_blocks)?;
                    
                    // Add new blocks to handle
                    for (physical_id, block) in new_blocks {
                        handle.add_block(physical_id, block);
                    }
                    
                    self.update_stats(|s| s.total_blocks_allocated += additional_blocks);
                }
            } else if new_size < current_tokens {
                // Could potentially free some blocks, but for simplicity we keep them
                // In a more sophisticated implementation, we would free unused blocks
                debug!("Shrinking cache size (blocks kept for simplicity)");
            }
            
            handle.set_num_tokens(new_size);
            
            let boxed_handle: Box<dyn ferrum_interfaces::KvCacheHandle + Send + Sync> = Box::new(handle.clone());
            Ok(boxed_handle)
        } else {
            Err(FerrumError::not_found(format!("Request not found: {:?}", request_id)))
        }
    }

    fn deallocate(&self, request_id: RequestId) -> Result<()> {
        debug!("Deallocating KV cache for request: {:?}", request_id);

        let mut active_handles = self.active_handles.write();
        if let Some(handle_arc) = active_handles.remove(&request_id) {
            let handle = handle_arc.read();
            let blocks_to_free: Vec<_> = handle.physical_blocks()
                .iter()
                .enumerate()
                .map(|(i, block)| {
                    let physical_id = PhysicalBlockId::new(i as u32); // Simplified - should get actual ID
                    (physical_id, block.clone())
                })
                .collect();
            
            let num_blocks = blocks_to_free.len();
            drop(handle); // Release the read lock
            
            self.free_blocks(blocks_to_free);
            
            self.update_stats(|s| {
                s.active_requests = active_handles.len();
                s.total_blocks_freed += num_blocks;
            });
            
            Ok(())
        } else {
            Err(FerrumError::not_found(format!("Request not found: {:?}", request_id)))
        }
    }

    fn can_allocate(&self, num_blocks: usize) -> bool {
        let primary = self.primary_pool();
        let stats = primary.stats();
        let available = stats.max_blocks - stats.allocated_blocks;
        available >= num_blocks
    }

    fn stats(&self) -> CacheStats {
        let primary_stats = self.primary_pool().stats();
        let secondary_stats = self.secondary_pool().map(|p| p.stats());
        
        let internal_stats = if self.config.enable_metrics {
            self.stats.lock().clone()
        } else {
            CacheStatsInternal::default()
        };
        
        CacheStats {
            total_requests: internal_stats.total_requests,
            active_requests: internal_stats.active_requests,
            total_blocks: primary_stats.total_blocks + secondary_stats.map(|s| s.total_blocks).unwrap_or(0),
            free_blocks: primary_stats.free_blocks + secondary_stats.map(|s| s.free_blocks).unwrap_or(0),
            used_blocks: primary_stats.allocated_blocks + secondary_stats.map(|s| s.allocated_blocks).unwrap_or(0),
            cache_hits: internal_stats.cache_hits,
            cache_misses: internal_stats.cache_misses,
            evictions: internal_stats.evictions,
            memory_usage: 0, // TODO: Calculate actual memory usage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let config = KvCacheConfig::default();
        let manager = DefaultKvCacheManager::new(Device::Cpu, config).unwrap();
        
        assert!(manager.cpu_pool.is_some());
        assert!(manager.gpu_pool.is_none());
    }

    #[test]
    fn test_allocation() {
        let config = KvCacheConfig::default();
        let manager = DefaultKvCacheManager::new(Device::Cpu, config).unwrap();
        
        let request = AllocationRequest {
            request_id: RequestId::new(),
            num_tokens: 32,
            num_layers: Some(24),
            num_heads: Some(16),
            head_dim: Some(64),
        };
        
        let handle = manager.allocate(&request).unwrap();
        assert_eq!(handle.num_tokens(), 32);
        assert_eq!(handle.device(), Device::Cpu);
    }

    #[test]
    fn test_deallocation() {
        let config = KvCacheConfig::default();
        let manager = DefaultKvCacheManager::new(Device::Cpu, config).unwrap();
        
        let request_id = RequestId::new();
        let request = AllocationRequest {
            request_id,
            num_tokens: 16,
            num_layers: Some(24),
            num_heads: Some(16),  
            head_dim: Some(64),
        };
        
        let _handle = manager.allocate(&request).unwrap();
        manager.deallocate(request_id).unwrap();
        
        // Should not be able to deallocate again
        assert!(manager.deallocate(request_id).is_err());
    }

    #[test]
    fn test_resize() {
        let config = KvCacheConfig::default();
        let manager = DefaultKvCacheManager::new(Device::Cpu, config).unwrap();
        
        let request_id = RequestId::new();
        let request = AllocationRequest {
            request_id,
            num_tokens: 16,
            num_layers: Some(24),
            num_heads: Some(16),
            head_dim: Some(64),
        };
        
        let _handle = manager.allocate(&request).unwrap();
        let resized_handle = manager.resize(request_id, 32).unwrap();
        
        assert_eq!(resized_handle.num_tokens(), 32);
    }

    #[test]
    fn test_can_allocate() {
        let mut config = KvCacheConfig::default();
        config.max_blocks_cpu = 10;
        
        let manager = DefaultKvCacheManager::new(Device::Cpu, config).unwrap();
        
        assert!(manager.can_allocate(5));
        assert!(manager.can_allocate(10));
        assert!(!manager.can_allocate(11));
    }
}
