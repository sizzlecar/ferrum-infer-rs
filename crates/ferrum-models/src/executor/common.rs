//! Common executor utilities — extracted from duplicated code across
//! Qwen3, Qwen2, and Llama executors.

use candle_core::{Device as CandleDevice, Tensor};
use ferrum_interfaces::{
    kv_cache::{BlockTable, CacheHandleStats},
    KvCacheHandle, TensorRef,
};
use ferrum_types::{Device, FerrumError, Result};
use std::sync::Arc;
use std::time::Instant;

use crate::tensor_wrapper::CandleTensorWrapper;

// ======================== Tensor Conversion ========================

/// Extract token IDs from a TensorRef.
pub fn tensor_to_tokens(tensor: &TensorRef) -> Result<Vec<u32>> {
    if let Ok(tokens) = tensor.to_vec_u32() {
        if tokens.is_empty() {
            return Err(FerrumError::model("Input token tensor is empty"));
        }
        return Ok(tokens);
    }
    if let Ok(tokens_f32) = tensor.to_vec_f32() {
        let tokens: Vec<u32> = tokens_f32.into_iter().map(|x| x as u32).collect();
        if tokens.is_empty() {
            return Err(FerrumError::model("Input token tensor is empty"));
        }
        return Ok(tokens);
    }
    Err(FerrumError::model(
        "Unable to extract token IDs from input tensor",
    ))
}

/// Convert token IDs to a candle Tensor on the target device.
pub fn tokens_to_tensor(tokens: &[u32], device: &CandleDevice) -> Result<Tensor> {
    let base = Tensor::new(tokens, &CandleDevice::Cpu)
        .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| FerrumError::model(format!("Failed to unsqueeze tensor: {}", e)))?
        .to_dtype(candle_core::DType::I64)
        .map_err(|e| FerrumError::model(format!("Failed to cast tokens to I64: {}", e)))?;

    if matches!(device, CandleDevice::Cpu) {
        Ok(base)
    } else {
        base.to_device(device)
            .map_err(|e| FerrumError::model(format!("Failed to move tensor to device: {}", e)))
    }
}

/// Wrap a candle Tensor as a TensorRef.
pub fn wrap_tensor(tensor: Tensor) -> TensorRef {
    Arc::new(CandleTensorWrapper::new(tensor))
}

// ======================== Generic KV Cache Handle ========================

/// Generic KV cache handle usable by any model architecture.
#[derive(Debug, Clone)]
pub struct GenericKvCacheHandle {
    block_table: BlockTable,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    device: Device,
    cache_id: String,
}

impl GenericKvCacheHandle {
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        device: CandleDevice,
        seq_len: usize,
        cache_id: String,
    ) -> Self {
        let mut block_table = BlockTable::new(16);
        block_table.sequence_length = seq_len;

        Self {
            block_table,
            num_layers,
            num_heads,
            head_dim,
            cache_id,
            device: match device {
                CandleDevice::Cpu => Device::CPU,
                CandleDevice::Cuda(_) => Device::CUDA(0),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                CandleDevice::Metal(_) => Device::Metal,
                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                CandleDevice::Metal(_) => Device::CPU,
            },
        }
    }

    pub fn with_sequence_length(&self, seq_len: usize) -> Self {
        let mut handle = self.clone();
        handle.block_table.sequence_length = seq_len;
        handle
    }

    pub fn request_cache_id(&self) -> &str {
        &self.cache_id
    }
}

impl KvCacheHandle for GenericKvCacheHandle {
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
        Ok(Arc::new(self.clone()))
    }
    fn stats(&self) -> CacheHandleStats {
        CacheHandleStats {
            memory_bytes: 0,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.block_table.sequence_length,
            utilization: 0.0,
            last_access: Instant::now(),
        }
    }
    fn is_valid(&self) -> bool {
        true
    }
    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

// ======================== Default Executor Status ========================

/// Default executor status (all executors return the same thing).
pub fn default_executor_status() -> ferrum_interfaces::model_executor::ExecutorStatus {
    use ferrum_interfaces::model_executor::*;
    ExecutorStatus {
        state: ExecutorState::Ready,
        is_ready: true,
        current_batch_size: 0,
        prefill_operations: 0,
        decode_operations: 0,
        avg_prefill_time_ms: 0.0,
        avg_decode_time_ms: 0.0,
        memory_usage: ExecutorMemoryUsage {
            allocated_bytes: 0,
            used_bytes: 0,
            peak_bytes: 0,
            utilization_percent: 0.0,
        },
        last_operation: Some(Instant::now()),
    }
}
