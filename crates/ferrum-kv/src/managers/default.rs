//! Default KV cache manager - MVP placeholder implementation

use crate::blocks::{BlockPool, DefaultKvCacheHandle};
use async_trait::async_trait;
use ferrum_interfaces::{
    kv_cache::{AllocationRequest, CacheGcStats, CacheManagerStats, MemoryPressure},
    KvCacheHandle, KvCacheManager,
};
use ferrum_types::{DataType, Device, FerrumError, RequestId, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Default KV cache manager - MVP implementation
pub struct DefaultKvCacheManager {
    device: Device,
    block_size: usize,
    max_blocks: usize,
    gpu_pool: Option<BlockPool>,
    cpu_pool: Option<BlockPool>,
    active_handles: RwLock<HashMap<RequestId, Arc<dyn KvCacheHandle>>>,
    stats: Mutex<CacheManagerStats>,
    #[allow(clippy::type_complexity)]
    pressure_callback: Mutex<Option<Box<dyn Fn(MemoryPressure) + Send + Sync>>>,
}

impl DefaultKvCacheManager {
    pub fn new(device: Device, block_size: usize, max_blocks: usize) -> Result<Self> {
        debug!(
            "Creating KV cache manager: device={:?}, block_size={}, max_blocks={}",
            device, block_size, max_blocks
        );

        let gpu_pool = match device {
            Device::CUDA(_) | Device::Metal | Device::ROCm(_) => Some(BlockPool::new(
                device.clone(),
                block_size,
                DataType::FP16,
                max_blocks,
            )?),
            Device::CPU => None,
        };

        let cpu_pool = Some(BlockPool::new(
            Device::CPU,
            block_size,
            DataType::FP16,
            max_blocks / 2,
        )?);

        Ok(Self {
            device,
            block_size,
            max_blocks,
            gpu_pool,
            cpu_pool,
            active_handles: RwLock::new(HashMap::new()),
            stats: Mutex::new(CacheManagerStats {
                total_memory_bytes: 0,
                used_memory_bytes: 0,
                active_caches: 0,
                total_blocks: max_blocks,
                free_blocks: max_blocks,
                cache_hit_rate: 0.0,
                eviction_count: 0,
                allocation_count: 0,
                allocation_failures: 0,
            }),
            pressure_callback: Mutex::new(None),
        })
    }
}

#[async_trait]
impl KvCacheManager for DefaultKvCacheManager {
    async fn allocate(&self, request: &AllocationRequest) -> Result<Arc<dyn KvCacheHandle>> {
        debug!("Allocating KV cache for request: {:?}", request.request_id);

        // MVP: Create a simple handle
        let handle = DefaultKvCacheHandle::new(request.request_id.clone(), self.block_size, 0);

        let handle_arc: Arc<dyn KvCacheHandle> = Arc::new(handle);

        self.active_handles
            .write()
            .insert(request.request_id.clone(), handle_arc.clone());

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.active_caches += 1;
            stats.allocation_count += 1;
        }

        Ok(handle_arc)
    }

    async fn extend(&self, _handle: &mut dyn KvCacheHandle, _additional_tokens: usize) -> Result<()> {
        // MVP: Not yet implemented
        Err(FerrumError::model("MVP: extend not yet implemented"))
    }

    async fn deallocate(&self, request_id: RequestId) -> Result<()> {
        debug!("Deallocating KV cache for request: {:?}", request_id);

        self.active_handles.write().remove(&request_id);

        // Update stats
        {
            let mut stats = self.stats.lock();
            if stats.active_caches > 0 {
                stats.active_caches -= 1;
            }
        }

        Ok(())
    }

    fn can_allocate(&self, _request: &AllocationRequest) -> bool {
        // MVP: always allow allocation
        let active_count = self.active_handles.read().len();
        active_count < self.max_blocks
    }

    fn stats(&self) -> CacheManagerStats {
        self.stats.lock().clone()
    }

    async fn gc(&self) -> Result<CacheGcStats> {
        // MVP: No garbage collection
        Ok(CacheGcStats {
            memory_freed: 0,
            caches_freed: 0,
            gc_time_ms: 0,
        })
    }

    fn set_pressure_callback(&self, callback: Box<dyn Fn(MemoryPressure) + Send + Sync>) {
        *self.pressure_callback.lock() = Some(callback);
    }

    fn get_handle(&self, request_id: RequestId) -> Option<Arc<dyn KvCacheHandle>> {
        self.active_handles.read().get(&request_id).cloned()
    }

    fn list_handles(&self) -> Vec<(RequestId, Arc<dyn KvCacheHandle>)> {
        self.active_handles
            .read()
            .iter()
            .map(|(id, handle)| (id.clone(), handle.clone()))
            .collect()
    }
}

impl std::fmt::Debug for DefaultKvCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultKvCacheManager")
            .field("device", &self.device)
            .field("block_size", &self.block_size)
            .field("max_blocks", &self.max_blocks)
            .field("active_handles_count", &self.active_handles.read().len())
            .finish()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_allocation_request() -> AllocationRequest {
        AllocationRequest {
            request_id: RequestId::new(),
            initial_tokens: 10,
            max_sequence_length: 100,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            device: Device::CPU,
            dtype: DataType::FP16,
            priority: ferrum_types::Priority::Normal,
        }
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = DefaultKvCacheManager::new(Device::CPU, 16, 100);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_allocate_and_deallocate() {
        let manager = DefaultKvCacheManager::new(Device::CPU, 16, 100).unwrap();
        let request = create_test_allocation_request();
        let request_id = request.request_id.clone();

        let handle = manager.allocate(&request).await.unwrap();
        assert!(handle.is_valid());

        let result = manager.deallocate(request_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_can_allocate() {
        let manager = DefaultKvCacheManager::new(Device::CPU, 16, 10).unwrap();
        let request = create_test_allocation_request();

        assert!(manager.can_allocate(&request));
    }

    #[tokio::test]
    async fn test_stats() {
        let manager = DefaultKvCacheManager::new(Device::CPU, 16, 100).unwrap();
        let stats = manager.stats();
        
        assert_eq!(stats.active_caches, 0);
        assert_eq!(stats.total_blocks, 100);
    }

    #[tokio::test]
    async fn test_get_handle() {
        let manager = DefaultKvCacheManager::new(Device::CPU, 16, 100).unwrap();
        let request = create_test_allocation_request();
        let request_id = request.request_id.clone();

        manager.allocate(&request).await.unwrap();

        let handle = manager.get_handle(request_id);
        assert!(handle.is_some());
    }
}