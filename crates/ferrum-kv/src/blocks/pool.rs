//! Block pool implementation for KV-Cache memory management

use ferrum_types::{DataType, Device, FerrumError, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing::{debug, trace, warn};

/// Physical block identifier
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalBlockId(pub u32);

impl PhysicalBlockId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(self) -> u32 {
        self.0
    }
}

/// Logical block identifier  
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalBlockId(pub u32);

impl LogicalBlockId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(self) -> u32 {
        self.0
    }
}

/// Block state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockState {
    Free,
    Allocated,
    InUse,
}

/// Memory block
#[derive(Debug)]
pub struct Block {
    /// Physical block ID
    pub id: PhysicalBlockId,
    /// Device where block is located
    pub device: Device,
    /// Block size in tokens
    pub size: usize,
    /// Data type of stored tensors
    pub data_type: DataType,
    /// Current state
    pub state: BlockState,
    /// Reference count
    pub ref_count: usize,
    /// Last access time (for eviction)
    pub last_access: std::time::Instant,
    /// Memory address/handle (device-specific)
    pub memory_handle: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl Block {
    /// Create new block
    pub fn new(id: PhysicalBlockId, device: Device, size: usize, data_type: DataType) -> Self {
        Self {
            id,
            device,
            size,
            data_type,
            state: BlockState::Free,
            ref_count: 0,
            last_access: std::time::Instant::now(),
            memory_handle: None,
        }
    }

    /// Update last access time
    pub fn touch(&mut self) {
        self.last_access = std::time::Instant::now();
    }

    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
        self.touch();
    }

    /// Decrement reference count
    pub fn remove_ref(&mut self) -> Result<()> {
        if self.ref_count == 0 {
            return Err(FerrumError::invalid_state(
                "Cannot remove reference from block with zero ref count",
            ));
        }
        self.ref_count -= 1;
        Ok(())
    }

    /// Check if block can be evicted
    pub fn can_evict(&self) -> bool {
        self.ref_count == 0 && self.state != BlockState::Free
    }
}

/// Block allocation result
#[derive(Debug)]
pub struct BlockAllocation {
    /// Allocated block
    pub block: Arc<RwLock<Block>>,
    /// Physical block ID
    pub physical_id: PhysicalBlockId,
}

/// Block pool for managing memory blocks
#[derive(Debug)]
pub struct BlockPool {
    /// Device type
    device: Device,
    /// Block size in tokens
    block_size: usize,
    /// Data type for blocks
    data_type: DataType,
    /// Maximum number of blocks
    max_blocks: usize,
    /// Free blocks queue
    free_blocks: Mutex<VecDeque<PhysicalBlockId>>,
    /// All blocks (id -> block)
    blocks: RwLock<HashMap<PhysicalBlockId, Arc<RwLock<Block>>>>,
    /// Next block ID
    next_block_id: AtomicUsize,
    /// Statistics
    allocated_blocks: AtomicUsize,
    total_allocations: AtomicUsize,
    total_deallocations: AtomicUsize,
}

impl BlockPool {
    /// Create new block pool
    pub fn new(
        device: Device,
        block_size: usize,
        data_type: DataType,
        max_blocks: usize,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(FerrumError::invalid_parameter(
                "Block size must be positive",
            ));
        }
        if max_blocks == 0 {
            return Err(FerrumError::invalid_parameter(
                "Max blocks must be positive",
            ));
        }

        debug!(
            "Creating block pool: device={:?}, block_size={}, data_type={:?}, max_blocks={}",
            device, block_size, data_type, max_blocks
        );

        Ok(Self {
            device,
            block_size,
            data_type,
            max_blocks,
            free_blocks: Mutex::new(VecDeque::new()),
            blocks: RwLock::new(HashMap::new()),
            next_block_id: AtomicUsize::new(0),
            allocated_blocks: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
        })
    }

    /// Allocate a block
    pub fn allocate(&self) -> Result<BlockAllocation> {
        // Try to get a free block first
        if let Some(block_id) = self.free_blocks.lock().pop_front() {
            let blocks = self.blocks.read();
            if let Some(block) = blocks.get(&block_id) {
                let mut block_guard = block.write();
                block_guard.state = BlockState::Allocated;
                block_guard.add_ref();

                debug!("Reused free block: {:?}", block_id);
                self.total_allocations.fetch_add(1, Ordering::Relaxed);

                return Ok(BlockAllocation {
                    physical_id: block_id,
                    block: block.clone(),
                });
            }
        }

        // Need to create a new block
        let current_blocks = self.allocated_blocks.load(Ordering::Relaxed);
        if current_blocks >= self.max_blocks {
            return Err(FerrumError::resource_exhausted(format!(
                "Block pool exhausted: {}/{} blocks allocated",
                current_blocks, self.max_blocks
            )));
        }

        let block_id =
            PhysicalBlockId::new(self.next_block_id.fetch_add(1, Ordering::Relaxed) as u32);
        let mut block = Block::new(
            block_id,
            self.device.clone(),
            self.block_size,
            self.data_type,
        );
        block.state = BlockState::Allocated;
        block.add_ref();

        let block = Arc::new(RwLock::new(block));

        {
            let mut blocks = self.blocks.write();
            blocks.insert(block_id, block.clone());
        }

        self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
        self.total_allocations.fetch_add(1, Ordering::Relaxed);

        debug!("Allocated new block: {:?}", block_id);

        Ok(BlockAllocation {
            physical_id: block_id,
            block,
        })
    }

    /// Deallocate a block
    pub fn deallocate(&self, block_id: PhysicalBlockId) -> Result<()> {
        let blocks = self.blocks.read();
        if let Some(block) = blocks.get(&block_id) {
            let mut block_guard = block.write();

            block_guard.remove_ref()?;

            if block_guard.ref_count == 0 {
                block_guard.state = BlockState::Free;
                self.free_blocks.lock().push_back(block_id);
                self.total_deallocations.fetch_add(1, Ordering::Relaxed);
                debug!("Deallocated block: {:?}", block_id);
            }

            Ok(())
        } else {
            Err(FerrumError::not_found(format!(
                "Block not found: {:?}",
                block_id
            )))
        }
    }

    /// Get block by ID
    pub fn get_block(&self, block_id: PhysicalBlockId) -> Option<Arc<RwLock<Block>>> {
        let blocks = self.blocks.read();
        blocks.get(&block_id).cloned()
    }

    /// Get statistics
    pub fn stats(&self) -> BlockPoolStats {
        let blocks = self.blocks.read();
        let free_count = self.free_blocks.lock().len();
        let total_blocks = blocks.len();

        BlockPoolStats {
            total_blocks,
            free_blocks: free_count,
            allocated_blocks: total_blocks - free_count,
            max_blocks: self.max_blocks,
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
        }
    }

    /// Force evict blocks to free memory
    pub fn evict_blocks(&self, count: usize) -> Result<Vec<PhysicalBlockId>> {
        let blocks = self.blocks.read();

        // Find evictable blocks (ref_count == 0, not free)
        let mut evictable: Vec<_> = blocks
            .iter()
            .filter_map(|(&id, block)| {
                let block_guard = block.read();
                if block_guard.can_evict() {
                    Some((id, block_guard.last_access))
                } else {
                    None
                }
            })
            .collect();

        // Sort by last access time (oldest first)
        evictable.sort_by_key(|(_, last_access)| *last_access);

        let mut evicted = Vec::new();
        for (block_id, _) in evictable.iter().take(count) {
            if let Some(block) = blocks.get(block_id) {
                let mut block_guard = block.write();
                if block_guard.can_evict() {
                    block_guard.state = BlockState::Free;
                    self.free_blocks.lock().push_back(*block_id);
                    evicted.push(*block_id);
                    warn!("Force evicted block: {:?}", block_id);
                }
            }
        }

        Ok(evicted)
    }

    /// Get device type
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Block pool statistics
#[derive(Debug, Clone)]
pub struct BlockPoolStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub max_blocks: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
}

impl BlockPoolStats {
    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.max_blocks == 0 {
            0.0
        } else {
            (self.allocated_blocks as f32 / self.max_blocks as f32) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_pool_creation() {
        let pool = BlockPool::new(Device::Cpu, 16, DataType::F16, 100).unwrap();

        assert_eq!(pool.block_size(), 16);
        assert_eq!(pool.device(), &Device::Cpu);
    }

    #[test]
    fn test_block_allocation() {
        let pool = BlockPool::new(Device::Cpu, 16, DataType::F16, 100).unwrap();

        let allocation = pool.allocate().unwrap();
        assert_eq!(allocation.physical_id, PhysicalBlockId::new(0));

        let stats = pool.stats();
        assert_eq!(stats.allocated_blocks, 1);
        assert_eq!(stats.free_blocks, 0);
    }

    #[test]
    fn test_block_deallocation() {
        let pool = BlockPool::new(Device::Cpu, 16, DataType::F16, 100).unwrap();

        let allocation = pool.allocate().unwrap();
        let block_id = allocation.physical_id;

        pool.deallocate(block_id).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.free_blocks, 1);
    }

    #[test]
    fn test_pool_exhaustion() {
        let pool = BlockPool::new(
            Device::Cpu,
            16,
            DataType::F16,
            1, // Only 1 block allowed
        )
        .unwrap();

        // First allocation should succeed
        let _allocation1 = pool.allocate().unwrap();

        // Second allocation should fail
        let result = pool.allocate();
        assert!(result.is_err());
    }

    #[test]
    fn test_block_reuse() {
        let pool = BlockPool::new(Device::Cpu, 16, DataType::F16, 100).unwrap();

        // Allocate and deallocate
        let allocation = pool.allocate().unwrap();
        let block_id = allocation.physical_id;
        pool.deallocate(block_id).unwrap();

        // Next allocation should reuse the same block
        let allocation2 = pool.allocate().unwrap();
        assert_eq!(allocation2.physical_id, block_id);
    }
}
