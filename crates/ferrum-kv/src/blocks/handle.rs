//! KV Cache handle - MVP placeholder implementation

use ferrum_interfaces::{kv_cache::CacheHandleStats, BlockTable, KvCacheHandle, TensorRef};
use ferrum_types::{Device, RequestId, Result};
use std::sync::Arc;

/// Default KV cache handle - MVP implementation
#[derive(Debug)]
pub struct DefaultKvCacheHandle {
    request_id: RequestId,
    block_table: BlockTable,
    device: Device,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    cache_id: String,
}

impl DefaultKvCacheHandle {
    pub fn new(request_id: RequestId, block_size: usize, num_tokens: usize) -> Self {
        let mut block_table = BlockTable::new(block_size);
        block_table.sequence_length = num_tokens;

        Self {
            cache_id: format!("cache_{}", request_id.to_string()),
            request_id,
            block_table,
            device: Device::CPU,
            num_layers: 32,    // Default placeholder
            num_heads: 32,     // Default placeholder
            head_dim: 128,     // Default placeholder
        }
    }

    pub fn set_num_tokens(&mut self, num_tokens: usize) {
        self.block_table.sequence_length = num_tokens;
    }
}

impl KvCacheHandle for DefaultKvCacheHandle {
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
        // MVP: return None (not yet implemented)
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> Result<Option<TensorRef>> {
        // MVP: return None (not yet implemented)
        Ok(None)
    }

    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        Err(ferrum_types::FerrumError::model(
            "MVP: Handle cloning not yet implemented",
        ))
    }

    fn stats(&self) -> CacheHandleStats {
        CacheHandleStats {
            memory_bytes: self.block_table.num_blocks() * self.block_table.block_size * 128,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.block_table.sequence_length,
            utilization: if self.block_table.num_blocks() > 0 {
                self.block_table.sequence_length as f32
                    / (self.block_table.num_blocks() * self.block_table.block_size) as f32
            } else {
                0.0
            },
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true // MVP: always valid
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

// ============================================================================
// 内联单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_creation() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id.clone(), 16, 10);

        assert_eq!(handle.request_id, request_id);
        assert_eq!(handle.block_table().block_size, 16);
        assert_eq!(handle.block_table().sequence_length, 10);
        assert!(handle.is_valid());
    }

    #[test]
    fn test_handle_set_num_tokens() {
        let request_id = RequestId::new();
        let mut handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        handle.set_num_tokens(50);
        assert_eq!(handle.block_table().sequence_length, 50);
    }

    #[test]
    fn test_handle_device() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        assert!(matches!(handle.device(), Device::CPU));
    }

    #[test]
    fn test_handle_dimensions() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        assert_eq!(handle.num_layers(), 32);
        assert_eq!(handle.num_heads(), 32);
        assert_eq!(handle.head_dim(), 128);
    }

    #[test]
    fn test_handle_block_table() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        let block_table = handle.block_table();
        assert_eq!(block_table.block_size, 16);
        assert_eq!(block_table.sequence_length, 10);
    }

    #[test]
    fn test_handle_block_table_mut() {
        let request_id = RequestId::new();
        let mut handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        let block_table = handle.block_table_mut();
        block_table.sequence_length = 20;

        assert_eq!(handle.block_table().sequence_length, 20);
    }

    #[test]
    fn test_handle_stats() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        let stats = handle.stats();
        assert_eq!(stats.tokens_stored, 10);
        assert_eq!(stats.blocks_allocated, handle.block_table().num_blocks());
        // memory_bytes 可能为0（如果没有分配blocks）
        assert!(stats.memory_bytes >= 0);
        assert!(stats.utilization >= 0.0 && stats.utilization <= 1.0);
    }

    #[test]
    fn test_handle_cache_id() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id.clone(), 16, 10);

        let cache_id = handle.cache_id();
        assert!(cache_id.contains("cache_"));
        assert!(cache_id.contains(&request_id.to_string()));
    }

    #[test]
    fn test_handle_is_valid() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        // MVP implementation always returns true
        assert!(handle.is_valid());
    }

    #[test]
    fn test_handle_key_cache() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        // MVP: should return None
        let result = handle.key_cache(0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_handle_value_cache() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        // MVP: should return None
        let result = handle.value_cache(0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_handle_clone_not_implemented() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        // MVP: clone_handle not yet implemented
        let result = handle.clone_handle();
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_as_any() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        let any = handle.as_any();
        assert!(any.downcast_ref::<DefaultKvCacheHandle>().is_some());
    }

    #[test]
    fn test_handle_debug_format() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 10);

        let debug_str = format!("{:?}", handle);
        assert!(debug_str.contains("DefaultKvCacheHandle"));
    }

    #[test]
    fn test_handle_with_different_block_sizes() {
        let request_id = RequestId::new();

        let handle_16 = DefaultKvCacheHandle::new(request_id.clone(), 16, 10);
        let handle_32 = DefaultKvCacheHandle::new(request_id.clone(), 32, 10);

        assert_eq!(handle_16.block_table().block_size, 16);
        assert_eq!(handle_32.block_table().block_size, 32);
    }

    #[test]
    fn test_handle_stats_utilization() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 8);

        let stats = handle.stats();
        
        // 如果有blocks，utilization应该在合理范围内
        if stats.blocks_allocated > 0 {
            assert!(stats.utilization >= 0.0);
            assert!(stats.utilization <= 1.0);
        }
    }

    #[test]
    fn test_handle_zero_tokens() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(request_id, 16, 0);

        assert_eq!(handle.block_table().sequence_length, 0);
        let stats = handle.stats();
        assert_eq!(stats.tokens_stored, 0);
    }
}