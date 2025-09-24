//! KV Cache handle implementation

use crate::blocks::pool::{PhysicalBlockId, Block};
use ferrum_interfaces::{KvCacheHandle, BlockTable};
use ferrum_types::{Device, RequestId};
use parking_lot::RwLock;
use std::sync::Arc;
use smallvec::SmallVec;

/// Default implementation of KV cache handle
#[derive(Debug)]
pub struct DefaultKvCacheHandle {
    /// Request ID this handle belongs to
    request_id: RequestId,
    /// Block table mapping
    block_table: BlockTable,
    /// Physical blocks
    physical_blocks: Vec<Arc<RwLock<Block>>>,
    /// Device where blocks are located
    device: Device,
    /// Total number of tokens stored
    num_tokens: usize,
    /// Number of layers
    num_layers: usize,
    /// Number of heads per layer
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
}

impl DefaultKvCacheHandle {
    /// Create new KV cache handle
    pub fn new(
        request_id: RequestId,
        device: Device,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            request_id,
            block_table: BlockTable::default(),
            physical_blocks: Vec::new(),
            device,
            num_tokens: 0,
            num_layers,
            num_heads,
            head_dim,
        }
    }

    /// Add physical block to this handle
    pub fn add_block(&mut self, physical_id: PhysicalBlockId, block: Arc<RwLock<Block>>) {
        let logical_id = self.physical_blocks.len() as u32;
        
        // Update block table
        if logical_id as usize >= self.block_table.logical_to_physical.len() {
            self.block_table.logical_to_physical.resize(logical_id as usize + 1, 0);
        }
        if physical_id.value() as usize >= self.block_table.physical.len() {
            self.block_table.physical.resize(physical_id.value() as usize + 1, 0);
        }
        
        self.block_table.logical_to_physical[logical_id as usize] = physical_id.value();
        self.block_table.physical[physical_id.value() as usize] = 1; // Mark as used
        
        self.physical_blocks.push(block);
    }

    /// Remove last block
    pub fn remove_last_block(&mut self) -> Option<Arc<RwLock<Block>>> {
        if let Some(block) = self.physical_blocks.pop() {
            // Update block table
            if let Some(&physical_id) = self.block_table.logical_to_physical.last() {
                if physical_id as usize < self.block_table.physical.len() {
                    self.block_table.physical[physical_id as usize] = 0;
                }
            }
            self.block_table.logical_to_physical.pop();
            Some(block)
        } else {
            None
        }
    }

    /// Update number of tokens
    pub fn set_num_tokens(&mut self, num_tokens: usize) {
        self.num_tokens = num_tokens;
        self.block_table.seq_len = num_tokens;
    }

    /// Get physical blocks
    pub fn physical_blocks(&self) -> &[Arc<RwLock<Block>>] {
        &self.physical_blocks
    }

    /// Get request ID
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Get model dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.num_layers, self.num_heads, self.head_dim)
    }

    /// Calculate total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Each token needs storage for K and V tensors across all layers
        // K,V shape: [num_heads, head_dim]
        // Total per token: num_layers * num_heads * head_dim * 2 (K + V) * sizeof(data_type)
        let bytes_per_token = self.num_layers * self.num_heads * self.head_dim * 2 * 2; // Assuming FP16
        self.num_tokens * bytes_per_token
    }

    /// Check if handle is valid
    pub fn is_valid(&self) -> bool {
        !self.physical_blocks.is_empty() 
            && self.num_tokens > 0
            && self.block_table.logical_to_physical.len() == self.physical_blocks.len()
    }
}

impl KvCacheHandle for DefaultKvCacheHandle {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn device(&self) -> Device {
        self.device.clone()
    }

    fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
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

    fn key_cache(&self, _layer: usize) -> ferrum_types::Result<Option<ferrum_interfaces::TensorRef>> {
        Ok(None)
    }

    fn value_cache(&self, _layer: usize) -> ferrum_types::Result<Option<ferrum_interfaces::TensorRef>> {
        Ok(None)
    }

    fn clone_handle(&self) -> ferrum_types::Result<Arc<dyn KvCacheHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn stats(&self) -> ferrum_interfaces::CacheHandleStats {
        ferrum_interfaces::CacheHandleStats {
            memory_bytes: self.memory_usage(),
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.num_tokens,
            utilization: 1.0,
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        self.is_valid()
    }

    fn cache_id(&self) -> String {
        self.request_id.to_string()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Clone for DefaultKvCacheHandle {
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id,
            block_table: self.block_table.clone(),
            physical_blocks: self.physical_blocks.clone(),
            device: self.device.clone(),
            num_tokens: self.num_tokens,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::pool::Block;
    use ferrum_types::{DataType, RequestId};

    #[test]
    fn test_handle_creation() {
        let request_id = RequestId::new();
        let handle = DefaultKvCacheHandle::new(
            request_id,
            Device::Cpu,
            32, // layers
            32, // heads  
            128, // head_dim
        );

        assert_eq!(handle.request_id(), request_id);
        assert_eq!(handle.device(), Device::Cpu);
        assert_eq!(handle.num_tokens(), 0);
        assert_eq!(handle.dimensions(), (32, 32, 128));
    }

    #[test]
    fn test_block_management() {
        let request_id = RequestId::new();
        let mut handle = DefaultKvCacheHandle::new(
            request_id,
            Device::Cpu,
            32, 32, 128,
        );

        // Add a block
        let physical_id = PhysicalBlockId::new(0);
        let block = Block::new(physical_id, Device::Cpu, 16, DataType::F16);
        let block_arc = Arc::new(RwLock::new(block));
        
        handle.add_block(physical_id, block_arc.clone());
        
        assert_eq!(handle.physical_blocks().len(), 1);
        assert_eq!(handle.block_table().logical_to_physical.len(), 1);
        assert_eq!(handle.block_table().logical_to_physical[0], 0);

        // Remove the block
        let removed = handle.remove_last_block();
        assert!(removed.is_some());
        assert_eq!(handle.physical_blocks().len(), 0);
    }

    #[test]
    fn test_token_management() {
        let request_id = RequestId::new();
        let mut handle = DefaultKvCacheHandle::new(
            request_id,
            Device::Cpu,
            32, 32, 128,
        );

        handle.set_num_tokens(100);
        assert_eq!(handle.num_tokens(), 100);
        assert_eq!(handle.block_table().seq_len, 100);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let request_id = RequestId::new();
        let mut handle = DefaultKvCacheHandle::new(
            request_id,
            Device::Cpu,
            32, // layers
            32, // heads
            128, // head_dim  
        );

        handle.set_num_tokens(100);
        
        // Expected: 32 layers * 32 heads * 128 head_dim * 2 (K+V) * 2 (FP16) * 100 tokens
        let expected = 32 * 32 * 128 * 2 * 2 * 100;
        assert_eq!(handle.memory_usage(), expected);
    }

    #[test]
    fn test_handle_validity() {
        let request_id = RequestId::new();
        let mut handle = DefaultKvCacheHandle::new(
            request_id,
            Device::Cpu,
            32, 32, 128,
        );

        // Initially invalid (no blocks, no tokens)
        assert!(!handle.is_valid());

        // Add block but no tokens - still invalid
        let physical_id = PhysicalBlockId::new(0);
        let block = Block::new(physical_id, Device::Cpu, 16, DataType::F16);
        let block_arc = Arc::new(RwLock::new(block));
        handle.add_block(physical_id, block_arc);
        assert!(!handle.is_valid());

        // Add tokens - now valid
        handle.set_num_tokens(10);
        assert!(handle.is_valid());
    }
}
