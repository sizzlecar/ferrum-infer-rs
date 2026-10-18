use ferrum_execution::{
    model_executor::UnifiedBatchItem, DecodeInput, DecodeOutput, PrefillInput, PrefillOutput,
};
use ferrum_interfaces::{
    BlockTable, CacheHandleStats, KvCacheHandle, RecurrentStateHandle, RecurrentStateHandleStats,
    RecurrentStateSpec, RecurrentStateTensorSpec, TensorLike, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, RequestId, Result};
use std::{any::Any, collections::HashMap, sync::Arc, time::Instant};

#[derive(Debug, Clone)]
struct MockRecurrentStateHandle {
    spec: RecurrentStateSpec,
    memory_bytes: usize,
    cache_id: String,
    valid: bool,
    last_access: Instant,
}

impl MockRecurrentStateHandle {
    fn new(spec: RecurrentStateSpec) -> Self {
        let memory_bytes = spec.estimated_memory_bytes();
        let cache_id = format!("mock-recurrent-state-{}", spec.request_id);
        Self {
            spec,
            memory_bytes,
            cache_id,
            valid: true,
            last_access: Instant::now(),
        }
    }
}

impl RecurrentStateHandle for MockRecurrentStateHandle {
    fn request_id(&self) -> RequestId {
        self.spec.request_id.clone()
    }

    fn device(&self) -> Device {
        self.spec.device.clone()
    }

    fn num_layers(&self) -> usize {
        self.spec.num_layers
    }

    fn state_bytes(&self) -> usize {
        self.memory_bytes
    }

    fn clone_handle(&self) -> Result<Arc<dyn RecurrentStateHandle>> {
        Ok(Arc::new(self.clone()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn stats(&self) -> RecurrentStateHandleStats {
        RecurrentStateHandleStats {
            memory_bytes: self.memory_bytes,
            state_tensors: self.spec.tensors.len(),
            batch_slots: self.spec.max_batch_slots,
            last_access: self.last_access,
        }
    }

    fn is_valid(&self) -> bool {
        self.valid
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

#[derive(Debug, Clone)]
struct MockTensor {
    shape: Vec<usize>,
}

impl TensorLike for MockTensor {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> DataType {
        DataType::UINT32
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn is_contiguous(&self) -> bool {
        true
    }

    fn view(&self, _start: &[usize], _end: &[usize]) -> Result<TensorRef> {
        Err(FerrumError::unsupported("mock tensor view"))
    }

    fn reshape(&self, shape: &[usize]) -> Result<TensorRef> {
        Ok(Arc::new(Self {
            shape: shape.to_vec(),
        }))
    }

    fn to_cpu(&self) -> Result<TensorRef> {
        Ok(Arc::new(self.clone()))
    }

    fn to_device(&self, _device: &Device) -> Result<TensorRef> {
        Ok(Arc::new(self.clone()))
    }

    fn to_dtype(&self, _dtype: DataType) -> Result<TensorRef> {
        Ok(Arc::new(self.clone()))
    }
}

#[derive(Debug, Clone)]
struct TestKvCacheHandle {
    block_table: BlockTable,
    cache_id: String,
}

impl TestKvCacheHandle {
    fn new(cache_id: impl Into<String>) -> Self {
        Self {
            block_table: BlockTable::new(16),
            cache_id: cache_id.into(),
        }
    }
}

impl KvCacheHandle for TestKvCacheHandle {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn device(&self) -> Device {
        Device::CPU
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn num_heads(&self) -> usize {
        1
    }

    fn head_dim(&self) -> usize {
        1
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
            blocks_allocated: 0,
            tokens_stored: 0,
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

fn mock_tensor(shape: &[usize]) -> TensorRef {
    Arc::new(MockTensor {
        shape: shape.to_vec(),
    })
}

fn mock_kv(cache_id: &str) -> Arc<dyn KvCacheHandle> {
    Arc::new(TestKvCacheHandle::new(cache_id))
}

#[test]
fn model_executor_inputs_and_outputs_carry_recurrent_state_handles() {
    let recurrent: Arc<dyn RecurrentStateHandle> =
        Arc::new(MockRecurrentStateHandle::new(RecurrentStateSpec {
            request_id: RequestId::new(),
            num_layers: 1,
            tensors: vec![RecurrentStateTensorSpec::new(
                0,
                "delta_state",
                vec![8, 16],
                DataType::BF16,
            )],
            device: Device::CPU,
            max_batch_slots: 1,
        }));
    let kv = mock_kv("kv-cache");

    let prefill = PrefillInput::new(mock_tensor(&[1, 2])).with_recurrent_state(recurrent.clone());
    assert!(prefill.kv_cache.is_none());
    assert_eq!(
        prefill.recurrent_state.as_ref().unwrap().cache_id(),
        recurrent.cache_id()
    );

    let prefill_output = PrefillOutput::new(mock_tensor(&[1, 1, 8]), kv.clone())
        .with_recurrent_state(recurrent.clone());
    assert_eq!(
        prefill_output.recurrent_state.as_ref().unwrap().cache_id(),
        recurrent.cache_id()
    );

    let decode =
        DecodeInput::new(mock_tensor(&[1, 1]), kv.clone()).with_recurrent_state(recurrent.clone());
    assert_eq!(
        decode.recurrent_state.as_ref().unwrap().cache_id(),
        recurrent.cache_id()
    );

    let decode_output =
        DecodeOutput::new(mock_tensor(&[1, 8]), kv.clone()).with_recurrent_state(recurrent.clone());
    assert_eq!(
        decode_output.recurrent_state.as_ref().unwrap().cache_id(),
        recurrent.cache_id()
    );

    let item = UnifiedBatchItem {
        seq_id: "hybrid-seq".to_string(),
        q_tokens: vec![1, 2],
        kv_cache: kv,
        recurrent_state: Some(recurrent),
        pos_offset: 0,
        is_final_chunk: true,
        metadata: HashMap::new(),
        logits_policy: Default::default(),
    };
    assert!(item.recurrent_state.is_some());
    assert!(format!("{item:?}").contains("has_recurrent_state: true"));
}
