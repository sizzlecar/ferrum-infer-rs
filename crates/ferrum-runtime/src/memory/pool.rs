//! Memory pool implementation for efficient allocation

use ferrum_interfaces::memory::{
    DeviceMemoryManager, MemoryHandle, StreamHandle, MemoryTransfer, MemoryInfo, 
    MemoryHandleInfo, MemoryPoolConfig as InterfaceMemoryPoolConfig, DefragmentationStats,
    MemoryPressure, MemoryType
};
use ferrum_types::{Device, Result};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use tracing::{debug, warn};
use async_trait::async_trait;

/// Memory block in the pool
#[derive(Debug, Clone)]
struct MemoryBlock {
    handle: MemoryHandle,
    size: usize,
    is_free: bool,
    allocated_at: std::time::Instant,
}

/// Memory pool for efficient allocation/deallocation
pub struct MemoryPool {
    device: Device,
    blocks: Mutex<VecDeque<MemoryBlock>>,
    free_blocks: Mutex<HashMap<usize, VecDeque<usize>>>, // size -> block indices
    total_allocated: Mutex<usize>,
    peak_allocated: Mutex<usize>,
    allocation_count: Mutex<u64>,
    config: MemoryPoolConfig,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,
    /// Maximum pool size in bytes  
    pub max_size: usize,
    /// Growth factor when expanding pool
    pub growth_factor: f32,
    /// Whether to enable automatic defragmentation
    pub enable_defragmentation: bool,
    /// Minimum block size to pool
    pub min_pooled_size: usize,
    /// Maximum block size to pool
    pub max_pooled_size: usize,
    /// Number of buckets for size-based pooling
    pub size_buckets: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 256 * 1024 * 1024,      // 256MB
            max_size: 8 * 1024 * 1024 * 1024,     // 8GB
            growth_factor: 1.5,
            enable_defragmentation: true,
            min_pooled_size: 256,                  // 256B
            max_pooled_size: 128 * 1024 * 1024,   // 128MB
            size_buckets: 64,
        }
    }
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(device: Device, config: MemoryPoolConfig) -> Self {
        Self {
            device,
            blocks: Mutex::new(VecDeque::new()),
            free_blocks: Mutex::new(HashMap::new()),
            total_allocated: Mutex::new(0),
            peak_allocated: Mutex::new(0),
            allocation_count: Mutex::new(0),
            config,
        }
    }

    /// Allocate memory from pool
    pub fn allocate(&self, size: usize) -> Result<MemoryHandle> {
        let aligned_size = align_size(size, 256); // 256-byte alignment
        
        // Try to find a free block of appropriate size
        if let Some(handle) = self.try_allocate_from_pool(aligned_size) {
            return Ok(handle);
        }
        
        // Allocate new block
        self.allocate_new_block(aligned_size)
    }

    /// Deallocate memory back to pool
    pub fn deallocate(&self, handle: MemoryHandle) -> Result<()> {
        let mut blocks = self.blocks.lock();
        
        // Find the block and mark it as free
        for (index, block) in blocks.iter_mut().enumerate() {
            if block.handle.id() == handle.id() {
                block.is_free = true;
                
                // Add to free blocks index
                let size = block.size;
                drop(blocks);
                
                let mut free_blocks = self.free_blocks.lock();
                free_blocks.entry(size).or_default().push_back(index);
                
                debug!("Deallocated block of size {} bytes", size);
                return Ok(());
            }
        }
        
        warn!("Attempted to deallocate unknown memory handle: {:?}", handle);
        Ok(())
    }

    /// Get memory statistics
    pub fn stats(&self) -> MemoryInfo {
        let blocks = self.blocks.lock();
        let total_allocated = *self.total_allocated.lock();
        
        let used_memory = blocks
            .iter()
            .filter(|b| !b.is_free)
            .map(|b| b.size)
            .sum::<usize>();
        
        let free_memory = blocks
            .iter()
            .filter(|b| b.is_free)
            .map(|b| b.size)
            .sum::<usize>();
        
        let fragmentation_ratio = if total_allocated > 0 {
            let free_blocks_count = blocks.iter().filter(|b| b.is_free).count();
            free_blocks_count as f32 / blocks.len() as f32
        } else {
            0.0
        };
        
        MemoryInfo {
            total_bytes: total_allocated as u64,
            used_bytes: used_memory as u64,
            free_bytes: free_memory as u64,
            reserved_bytes: 0,
            active_allocations: blocks.iter().filter(|b| !b.is_free).count(),
            fragmentation_ratio,
            bandwidth_gbps: None,
        }
    }

    /// Defragment memory pool
    pub fn defragment(&self) -> Result<()> {
        if !self.config.enable_defragmentation {
            return Ok(());
        }
        
        debug!("Starting memory pool defragmentation for device {:?}", self.device);
        
        // Simple defragmentation: compact free blocks
        let mut blocks = self.blocks.lock();
        let mut free_blocks = self.free_blocks.lock();
        
        // Remove freed blocks and rebuild free index
        blocks.retain(|b| !b.is_free);
        free_blocks.clear();
        
        // Rebuild free blocks index
        for (index, block) in blocks.iter().enumerate() {
            if block.is_free {
                free_blocks.entry(block.size).or_default().push_back(index);
            }
        }
        
        debug!("Memory pool defragmentation completed");
        Ok(())
    }

    fn try_allocate_from_pool(&self, size: usize) -> Option<MemoryHandle> {
        let mut free_blocks = self.free_blocks.lock();
        
        // Look for exact size match first
        if let Some(indices) = free_blocks.get_mut(&size) {
            if let Some(index) = indices.pop_front() {
                let mut blocks = self.blocks.lock();
                if let Some(block) = blocks.get_mut(index) {
                    block.is_free = false;
                    return Some(block.handle);
                }
            }
        }
        
        // Look for larger blocks that can be split
        let mut best_fit: Option<(usize, usize)> = None; // (size, index)
        
        for (&block_size, indices) in free_blocks.iter() {
            if block_size >= size && (best_fit.is_none() || block_size < best_fit.unwrap().0) {
                if let Some(&index) = indices.front() {
                    best_fit = Some((block_size, index));
                }
            }
        }
        
        if let Some((block_size, index)) = best_fit {
            // Remove from free list
            free_blocks.get_mut(&block_size)?.pop_front();
            
            let mut blocks = self.blocks.lock();
            if let Some(block) = blocks.get_mut(index) {
                block.is_free = false;
                return Some(block.handle);
            }
        }
        
        None
    }

    fn allocate_new_block(&self, size: usize) -> Result<MemoryHandle> {
        // Check if we would exceed max pool size
        let current_total = *self.total_allocated.lock();
        if current_total + size > self.config.max_size {
            return Err(ferrum_types::FerrumError::backend(
                format!("Memory pool size limit exceeded: {} + {} > {}", 
                        current_total, size, self.config.max_size)
            ));
        }
        
        // Create new memory handle (simplified - real implementation would allocate actual memory)
        let handle_id = {
            let mut count = self.allocation_count.lock();
            *count += 1;
            *count
        };
        
        let handle = MemoryHandle::new(handle_id);
        
        // Add to blocks
        let block = MemoryBlock {
            handle,
            size,
            is_free: false,
            allocated_at: std::time::Instant::now(),
        };
        
        let mut blocks = self.blocks.lock();
        blocks.push_back(block);
        
        // Update statistics
        {
            let mut total = self.total_allocated.lock();
            *total += size;
            
            let mut peak = self.peak_allocated.lock();
            if *total > *peak {
                *peak = *total;
            }
        }
        
        debug!("Allocated new memory block of size {} bytes", size);
        Ok(handle)
    }
}

#[async_trait]
impl DeviceMemoryManager for MemoryPool {
    async fn allocate(&self, size: usize, _device: &Device) -> Result<MemoryHandle> {
        self.allocate(size)
    }

    async fn allocate_aligned(
        &self,
        size: usize,
        alignment: usize,
        _device: &Device,
    ) -> Result<MemoryHandle> {
        let aligned_size = align_size(size, alignment);
        self.allocate(aligned_size)
    }

    async fn deallocate(&self, handle: MemoryHandle) -> Result<()> {
        self.deallocate(handle)
    }

    async fn copy(
        &self,
        _src: MemoryHandle,
        _dst: MemoryHandle,
        _size: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()> {
        // Simplified implementation - real version would do actual copy
        Ok(())
    }

    async fn copy_async(
        &self,
        _transfer: MemoryTransfer,
        _stream: Option<StreamHandle>,
    ) -> Result<()> {
        // Simplified implementation
        Ok(())
    }

    async fn memory_info(&self, _device: &Device) -> Result<MemoryInfo> {
        Ok(self.stats())
    }

    fn handle_info(&self, handle: MemoryHandle) -> Option<MemoryHandleInfo> {
        let blocks = self.blocks.lock();
        blocks.iter().find(|b| b.handle.id() == handle.id()).map(|block| {
            MemoryHandleInfo {
                handle: block.handle,
                size: block.size,
                device: self.device,
                alignment: 256, // Default alignment
                allocated_at: block.allocated_at,
                is_mapped: false,
                memory_type: MemoryType::General,
            }
        })
    }

    async fn configure_pool(&self, _device: &Device, _config: InterfaceMemoryPoolConfig) -> Result<()> {
        // For now, pool config is set at construction
        Ok(())
    }

    async fn defragment(&self, _device: &Device) -> Result<DefragmentationStats> {
        let before_fragmentation = self.stats().fragmentation_ratio;
        self.defragment()?;
        let after_fragmentation = self.stats().fragmentation_ratio;
        
        Ok(DefragmentationStats {
            memory_freed: 0, // Simplified
            blocks_moved: 0, 
            time_taken_ms: 0,
            fragmentation_before: before_fragmentation,
            fragmentation_after: after_fragmentation,
        })
    }

    fn set_pressure_callback(
        &self,
        _callback: Box<dyn Fn(MemoryPressure) + Send + Sync>,
    ) {
        // Simplified - real implementation would store and use callback
    }
}

/// Align size to specified boundary
fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}
