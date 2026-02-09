//! Stub model executor for MVP testing and development

use async_trait::async_trait;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorMemoryUsage,
        ExecutorState, ExecutorStatus, MemoryRequirements, PrefillInput, PrefillOutput,
    },
    BlockTable, ComputeBackend, KvCacheHandle, ModelExecutor,
};
use ferrum_types::{DataType, Device, ModelInfo, ModelType, Result};
use std::sync::Arc;
use tracing::debug;

/// Stub model executor - MVP implementation
///
/// Returns dummy outputs to allow pipeline testing without real models.
pub struct StubModelExecutor {
    info: ModelInfo,
    compute_backend: Arc<dyn ComputeBackend>,
}

impl StubModelExecutor {
    pub fn new(
        model_id: impl Into<ferrum_types::ModelId>,
        vocab_size: usize,
        compute_backend: Arc<dyn ComputeBackend>,
    ) -> Self {
        let info = ModelInfo {
            model_id: model_id.into(),
            model_type: ModelType::Custom("stub".into()),
            num_parameters: 1_000_000,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            num_kv_heads: 12,
            vocab_size,
            max_sequence_length: 2048,
            dtype: DataType::FP16,
            device: Device::CPU,
            version: Some("mvp-stub".into()),
            license: Some("Apache-2.0".into()),
            metadata: std::collections::HashMap::new(),
        };

        debug!("Created StubModelExecutor: vocab={}", vocab_size);

        Self {
            info,
            compute_backend,
        }
    }
}

#[async_trait]
impl ModelExecutor for StubModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let batch_size = input.batch_size();
        let seq_len = input.sequence_length();
        let vocab_size = self.info.vocab_size;

        debug!("Stub prefill: batch={}, seq_len={}", batch_size, seq_len);

        // Create dummy logits
        let factory = self.compute_backend.tensor_factory();
        let logits = factory.zeros(
            &[batch_size, seq_len, vocab_size],
            DataType::FP32,
            &self.info.device,
        )?;

        // Create stub KV cache
        let kv_cache = create_stub_kv_cache(
            ferrum_types::RequestId::new(),
            self.info.num_layers,
            seq_len,
        );

        Ok(PrefillOutput::new(logits, kv_cache))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let batch_size = input.batch_size();
        let vocab_size = self.info.vocab_size;

        debug!("Stub decode: batch={}", batch_size);

        let factory = self.compute_backend.tensor_factory();
        let logits = factory.zeros(&[batch_size, vocab_size], DataType::FP32, &self.info.device)?;

        Ok(DecodeOutput::new(logits, input.kv_cache.clone()))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 32,
            max_sequence_length: 2048,
            attention_mechanisms: vec![AttentionType::MultiHead],
            supports_dynamic_batching: false,
            supports_continuous_batching: false,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32, DataType::FP16],
            supported_devices: vec![Device::CPU],
            memory_requirements: MemoryRequirements {
                parameter_memory: 4 * 1024 * 1024, // 4MB
                activation_memory_per_token: 1024,
                kv_cache_memory_per_token: 512,
                overhead_memory: 1024 * 1024,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        ExecutorStatus {
            state: ExecutorState::Ready,
            is_ready: true,
            current_batch_size: 0,
            prefill_operations: 0,
            decode_operations: 0,
            avg_prefill_time_ms: 0.0,
            avg_decode_time_ms: 0.0,
            memory_usage: ExecutorMemoryUsage {
                allocated_bytes: 1024 * 1024,
                used_bytes: 512 * 1024,
                peak_bytes: 1024 * 1024,
                utilization_percent: 50.0,
            },
            last_operation: None,
        }
    }
}

/// Create stub KV cache handle
fn create_stub_kv_cache(
    request_id: ferrum_types::RequestId,
    num_layers: usize,
    seq_len: usize,
) -> Arc<dyn KvCacheHandle> {
    #[derive(Debug)]
    struct StubKvCache {
        request_id: ferrum_types::RequestId,
        block_table: BlockTable,
        num_layers: usize,
    }

    impl KvCacheHandle for StubKvCache {
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
            Device::CPU
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn num_heads(&self) -> usize {
            12
        }

        fn head_dim(&self) -> usize {
            64
        }

        fn key_cache(&self, _layer: usize) -> Result<Option<ferrum_interfaces::TensorRef>> {
            Ok(None)
        }

        fn value_cache(&self, _layer: usize) -> Result<Option<ferrum_interfaces::TensorRef>> {
            Ok(None)
        }

        fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
            Err(ferrum_types::FerrumError::unsupported("Stub cache clone"))
        }

        fn stats(&self) -> ferrum_interfaces::kv_cache::CacheHandleStats {
            ferrum_interfaces::kv_cache::CacheHandleStats {
                memory_bytes: 1024,
                blocks_allocated: 1,
                tokens_stored: self.block_table.sequence_length,
                utilization: 0.5,
                last_access: std::time::Instant::now(),
            }
        }

        fn is_valid(&self) -> bool {
            true
        }

        fn cache_id(&self) -> String {
            format!("stub_{}", self.request_id.to_string())
        }
    }

    let mut block_table = BlockTable::new(16);
    block_table.sequence_length = seq_len;

    Arc::new(StubKvCache {
        request_id,
        block_table,
        num_layers,
    })
}
