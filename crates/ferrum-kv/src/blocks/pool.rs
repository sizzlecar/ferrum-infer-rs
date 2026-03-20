//! Block pool implementation for KV-Cache memory management

use super::storage::{BlockStorage, BlockStorageConfig};
use ferrum_types::{DataType, Device, FerrumError, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing::{debug, warn};

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
            return Err(FerrumError::backend(
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
    /// Optional storage config — when set, blocks get KV tensor storage
    storage_config: Option<BlockStorageConfig>,
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
    /// Create new block pool (blocks have no tensor storage).
    pub fn new(
        device: Device,
        block_size: usize,
        data_type: DataType,
        max_blocks: usize,
    ) -> Result<Self> {
        Self::create(device, block_size, data_type, max_blocks, None)
    }

    /// Create a block pool where each block has KV tensor storage.
    pub fn new_with_storage(
        device: Device,
        block_size: usize,
        data_type: DataType,
        max_blocks: usize,
        storage_config: BlockStorageConfig,
    ) -> Result<Self> {
        Self::create(device, block_size, data_type, max_blocks, Some(storage_config))
    }

    fn create(
        device: Device,
        block_size: usize,
        data_type: DataType,
        max_blocks: usize,
        storage_config: Option<BlockStorageConfig>,
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
            "Creating block pool: device={:?}, block_size={}, data_type={:?}, max_blocks={}, has_storage={}",
            device, block_size, data_type, max_blocks, storage_config.is_some()
        );

        Ok(Self {
            device,
            block_size,
            data_type,
            max_blocks,
            storage_config,
            free_blocks: Mutex::new(VecDeque::new()),
            blocks: RwLock::new(HashMap::new()),
            next_block_id: AtomicUsize::new(1), // Start at 1; 0 is reserved as "unmapped"
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

        // Attach tensor storage if configured
        if let Some(cfg) = &self.storage_config {
            let storage: Arc<parking_lot::RwLock<BlockStorage>> =
                Arc::new(parking_lot::RwLock::new(BlockStorage::new(*cfg)));
            block.memory_handle = Some(Arc::new(storage));
        }

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

    /// Whether this pool has tensor storage attached to blocks.
    pub fn has_storage(&self) -> bool {
        self.storage_config.is_some()
    }

    /// Access the storage lock for a given block.
    fn block_storage(
        &self,
        block_id: PhysicalBlockId,
    ) -> Result<Arc<parking_lot::RwLock<BlockStorage>>> {
        let block_arc = self.get_block(block_id).ok_or_else(|| {
            FerrumError::not_found(format!("Block {:?} not found", block_id))
        })?;
        let block = block_arc.read();
        let storage_any = block.memory_handle.as_ref().ok_or_else(|| {
            FerrumError::internal("Block has no tensor storage")
        })?;
        storage_any
            .downcast_ref::<Arc<parking_lot::RwLock<BlockStorage>>>()
            .cloned()
            .ok_or_else(|| FerrumError::internal("Block memory_handle is not BlockStorage"))
    }

    /// Write one token's K/V vectors into a physical block at a given slot and layer.
    pub fn write_kv_slot(
        &self,
        block_id: PhysicalBlockId,
        layer: usize,
        slot: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let storage = self.block_storage(block_id)?;
        let result = storage.write().write_slot(layer, slot, key, value);
        result
    }

    /// Read one token's K/V vectors from a physical block at a given slot and layer.
    ///
    /// Returns `(key, value)` as owned Vec to avoid holding locks.
    pub fn read_kv_slot(
        &self,
        block_id: PhysicalBlockId,
        layer: usize,
        slot: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let storage = self.block_storage(block_id)?;
        let guard = storage.read();
        let (k, v) = guard.read_slot(layer, slot)?;
        Ok((k.to_vec(), v.to_vec()))
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
        let pool = BlockPool::new(Device::CPU, 16, DataType::FP16, 100).unwrap();

        assert_eq!(pool.block_size(), 16);
        assert_eq!(pool.device(), &Device::CPU);
    }

    #[test]
    fn test_block_allocation() {
        let pool = BlockPool::new(Device::CPU, 16, DataType::FP16, 100).unwrap();

        let allocation = pool.allocate().unwrap();
        assert_eq!(allocation.physical_id, PhysicalBlockId::new(1));

        let stats = pool.stats();
        assert_eq!(stats.allocated_blocks, 1);
        assert_eq!(stats.free_blocks, 0);
    }

    #[test]
    fn test_block_deallocation() {
        let pool = BlockPool::new(Device::CPU, 16, DataType::FP16, 100).unwrap();

        let allocation = pool.allocate().unwrap();
        let block_id = allocation.physical_id;

        pool.deallocate(block_id).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.free_blocks, 1);
    }

    #[test]
    fn test_pool_exhaustion() {
        let pool = BlockPool::new(
            Device::CPU,
            16,
            DataType::FP16,
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
    fn test_block_storage_write_read() {
        let cfg = BlockStorageConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 8,
            block_size: 16,
        };
        let pool =
            BlockPool::new_with_storage(Device::CPU, 16, DataType::FP16, 100, cfg).unwrap();
        assert!(pool.has_storage());

        let alloc = pool.allocate().unwrap();
        let bid = alloc.physical_id;

        let tok_size = cfg.num_kv_heads * cfg.head_dim; // 32
        let key: Vec<f32> = (0..tok_size).map(|i| i as f32).collect();
        let val: Vec<f32> = (0..tok_size).map(|i| (i as f32) + 100.0).collect();

        pool.write_kv_slot(bid, 0, 3, &key, &val).unwrap();
        let (k, v) = pool.read_kv_slot(bid, 0, 3).unwrap();
        assert_eq!(k, key);
        assert_eq!(v, val);

        // Different slot should be zeros
        let (k0, _) = pool.read_kv_slot(bid, 0, 0).unwrap();
        assert!(k0.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_block_reuse() {
        let pool = BlockPool::new(Device::CPU, 16, DataType::FP16, 100).unwrap();

        // Allocate and deallocate
        let allocation = pool.allocate().unwrap();
        let block_id = allocation.physical_id;
        pool.deallocate(block_id).unwrap();

        // Next allocation should reuse the same block
        let allocation2 = pool.allocate().unwrap();
        assert_eq!(allocation2.physical_id, block_id);
    }
}
