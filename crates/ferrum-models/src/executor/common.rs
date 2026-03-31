//! Common executor utilities — extracted from duplicated code across
//! Qwen3, Qwen2, and Llama executors.

use candle_core::{Device as CandleDevice, Tensor};
use ferrum_interfaces::{
    kv_cache::{BlockTable, CacheHandleStats},
    KvCacheHandle, TensorRef,
};
use ferrum_types::{Device, FerrumError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Global counter for generating unique clone cache IDs.
static CLONE_COUNTER: AtomicU64 = AtomicU64::new(0);
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
        // Each clone gets a unique cache_id so CUDA runner's kv_states
        // HashMap can distinguish between sequences with same prompt.
        let mut cloned = self.clone();
        let n = CLONE_COUNTER.fetch_add(1, Ordering::Relaxed);
        cloned.cache_id = format!("{}-clone-{n}", self.cache_id);
        Ok(Arc::new(cloned))
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::KvCacheHandle;

    #[test]
    fn tensor_to_tokens_from_u32() {
        let tensor = ferrum_testkit::MockTensor::from_u32(&[1, 2, 3], &[3]);
        let tokens = tensor_to_tokens(&tensor.into_ref()).unwrap();
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn tensor_to_tokens_from_f32() {
        let tensor = ferrum_testkit::MockTensor::from_f32(vec![10.0, 20.0, 30.0], &[3]);
        let tokens = tensor_to_tokens(&tensor.into_ref()).unwrap();
        assert_eq!(tokens, vec![10, 20, 30]);
    }

    #[test]
    fn tensor_to_tokens_empty_fails() {
        let tensor = ferrum_testkit::MockTensor::from_u32(&[], &[0]);
        let result = tensor_to_tokens(&tensor.into_ref());
        assert!(result.is_err());
    }

    #[test]
    fn tokens_to_tensor_cpu() {
        let tensor = tokens_to_tensor(&[42, 100], &CandleDevice::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 2]);
        assert_eq!(tensor.dtype(), candle_core::DType::I64);
    }

    #[test]
    fn wrap_tensor_creates_tensor_ref() {
        let t = Tensor::zeros((2, 3), candle_core::DType::F32, &CandleDevice::Cpu).unwrap();
        let tr = wrap_tensor(t);
        assert_eq!(tr.shape(), &[2, 3]);
    }

    #[test]
    fn generic_kv_cache_handle_basic() {
        let handle = GenericKvCacheHandle::new(
            36,  // num_layers
            32,  // num_heads
            128, // head_dim
            CandleDevice::Cpu,
            10, // seq_len
            "test-cache-1".to_string(),
        );

        assert_eq!(handle.num_layers(), 36);
        assert_eq!(handle.num_heads(), 32);
        assert_eq!(handle.head_dim(), 128);
        assert_eq!(handle.cache_id(), "test-cache-1");
        assert_eq!(handle.block_table().sequence_length, 10);
        assert!(handle.is_valid());
    }

    #[test]
    fn generic_kv_cache_handle_with_sequence_length() {
        let handle =
            GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 5, "cache-1".to_string());
        let updated = handle.with_sequence_length(15);
        assert_eq!(updated.block_table().sequence_length, 15);
        assert_eq!(updated.request_cache_id(), "cache-1");
        // Original unchanged
        assert_eq!(handle.block_table().sequence_length, 5);
    }

    #[test]
    fn generic_kv_cache_handle_clone_handle() {
        let handle =
            GenericKvCacheHandle::new(4, 8, 64, CandleDevice::Cpu, 5, "cache-2".to_string());
        let cloned = handle.clone_handle().unwrap();
        assert_eq!(cloned.cache_id(), "cache-2");
        assert_eq!(cloned.num_layers(), 4);
    }

    #[test]
    fn default_executor_status_is_ready() {
        let status = default_executor_status();
        assert!(status.is_ready);
        assert_eq!(
            status.state,
            ferrum_interfaces::model_executor::ExecutorState::Ready
        );
    }
}
