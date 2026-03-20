//! Mock KV cache manager for testing without GPU memory.

use async_trait::async_trait;
use ferrum_interfaces::{
    kv_cache::{AllocationRequest, CacheGcStats, CacheHandleStats, CacheManagerStats, MemoryPressure},
    BlockTable, KvCacheHandle, KvCacheManager, TensorRef,
};
use ferrum_types::{Device, RequestId, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Mock KV cache handle — tracks block metadata without allocating real memory.
#[derive(Debug)]
pub struct MockKvCacheHandle {
    request_id: RequestId,
    block_table: BlockTable,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
}

impl MockKvCacheHandle {
    pub fn new(request_id: RequestId, num_layers: usize, seq_len: usize) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = seq_len;
        // Add a physical block
        let blocks_needed = BlockTable::blocks_needed_for_length(seq_len, 16);
        let block_ids: Vec<u32> = (0..blocks_needed as u32).collect();
        block_table.add_blocks(&block_ids);

        Self {
            request_id,
            block_table,
            num_layers,
            num_heads: 12,
            head_dim: 64,
            device: Device::CPU,
        }
    }
}

impl KvCacheHandle for MockKvCacheHandle {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn key_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        Ok(None)
    }

    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        Ok(Arc::new(MockKvCacheHandle {
            request_id: self.request_id.clone(),
            block_table: self.block_table.clone(),
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            device: self.device.clone(),
        }))
    }

    fn stats(&self) -> CacheHandleStats {
        CacheHandleStats {
            memory_bytes: self.block_table.num_blocks() * 16 * self.num_layers * self.num_heads * self.head_dim * 2,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.block_table.sequence_length,
            utilization: if self.block_table.num_blocks() > 0 {
                self.block_table.sequence_length as f32
                    / (self.block_table.num_blocks() * 16) as f32
            } else {
                0.0
            },
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        format!("mock_{}", self.request_id)
    }
}

/// Mock KV cache manager — tracks allocations in memory, simulates block limits.
pub struct MockKvCacheManager {
    handles: RwLock<HashMap<RequestId, Arc<dyn KvCacheHandle>>>,
    total_blocks: usize,
    block_size: usize,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
}

impl MockKvCacheManager {
    /// Create with a fixed total block budget.
    pub fn new(total_blocks: usize) -> Self {
        Self {
            handles: RwLock::new(HashMap::new()),
            total_blocks,
            block_size: 16,
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
        }
    }

    pub fn active_count(&self) -> usize {
        self.handles.read().len()
    }
}

#[async_trait]
impl KvCacheManager for MockKvCacheManager {
    async fn allocate(&self, request: &AllocationRequest) -> Result<Arc<dyn KvCacheHandle>> {
        let blocks_needed =
            BlockTable::blocks_needed_for_length(request.initial_tokens, self.block_size);

        // Check block budget
        let used_blocks: usize = self
            .handles
            .read()
            .values()
            .map(|h| h.block_table().num_blocks())
            .sum();

        if used_blocks + blocks_needed > self.total_blocks {
            return Err(ferrum_types::FerrumError::backend(format!(
                "OOM: need {} blocks, have {} free out of {}",
                blocks_needed,
                self.total_blocks - used_blocks,
                self.total_blocks
            )));
        }

        let handle: Arc<dyn KvCacheHandle> = Arc::new(MockKvCacheHandle::new(
            request.request_id.clone(),
            request.num_layers,
            request.initial_tokens,
        ));

        self.handles
            .write()
            .insert(request.request_id.clone(), handle.clone());
        self.allocation_count.fetch_add(1, Ordering::Relaxed);

        Ok(handle)
    }

    async fn extend(
        &self,
        _handle: &mut dyn KvCacheHandle,
        _additional_tokens: usize,
    ) -> Result<()> {
        // Mock: no-op, real impl would allocate more blocks
        Ok(())
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        self.handles.write().remove(&request_id);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn can_allocate(&self, request: &AllocationRequest) -> bool {
        let blocks_needed =
            BlockTable::blocks_needed_for_length(request.initial_tokens, self.block_size);
        let used_blocks: usize = self
            .handles
            .read()
            .values()
            .map(|h| h.block_table().num_blocks())
            .sum();
        used_blocks + blocks_needed <= self.total_blocks
    }

    fn stats(&self) -> CacheManagerStats {
        let handles = self.handles.read();
        let used_blocks: usize = handles.values().map(|h| h.block_table().num_blocks()).sum();
        CacheManagerStats {
            total_memory_bytes: self.total_blocks * self.block_size * 1024,
            used_memory_bytes: used_blocks * self.block_size * 1024,
            active_caches: handles.len(),
            total_blocks: self.total_blocks,
            free_blocks: self.total_blocks - used_blocks,
            cache_hit_rate: 0.0,
            eviction_count: 0,
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            allocation_failures: 0,
        }
    }

    async fn gc(&self) -> Result<CacheGcStats> {
        Ok(CacheGcStats {
            memory_freed: 0,
            caches_freed: 0,
            gc_time_ms: 0,
        })
    }

    fn set_pressure_callback(&self, _callback: Box<dyn Fn(MemoryPressure) + Send + Sync>) {
        // Mock: no-op
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn KvCacheHandle>> {
        self.handles.read().get(&request_id).cloned()
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn KvCacheHandle>)> {
        self.handles
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}
