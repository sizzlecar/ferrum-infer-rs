//! Tensor-Parallel model executor.
//!
//! Wraps TpDecodeGroup for multi-GPU decode with NCCL all-reduce.
//! Prefill uses candle on GPU 0; decode uses sharded runners on all GPUs.
//!
//! Feature-gated: only available with `tensor-parallel` feature.

#[cfg(feature = "tensor-parallel")]
use async_trait::async_trait;
#[cfg(feature = "tensor-parallel")]
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus,
        MemoryRequirements, PrefillInput, PrefillOutput,
    },
    KvCacheHandle, ModelExecutor, TensorRef,
};
#[cfg(feature = "tensor-parallel")]
use ferrum_types::{DataType, FerrumError, ModelInfo, Result};
#[cfg(feature = "tensor-parallel")]
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
#[cfg(feature = "tensor-parallel")]
use tracing::info;

#[cfg(feature = "tensor-parallel")]
use super::common::{self, GenericKvCacheHandle};
#[cfg(feature = "tensor-parallel")]
use crate::tensor_wrapper::CandleTensorWrapper;
#[cfg(feature = "tensor-parallel")]
use parking_lot::Mutex;

#[cfg(feature = "tensor-parallel")]
use ferrum_cuda_kernels::tp_decode::TpDecodeGroup;

#[cfg(feature = "tensor-parallel")]
struct TpCacheState {
    sequence_length: usize,
}

/// Tensor-parallel executor.
///
/// - Prefill: single GPU (candle FlashAttention-2 on GPU 0)
/// - Decode: all GPUs via TpDecodeGroup with NCCL all-reduce
#[cfg(feature = "tensor-parallel")]
pub struct TpModelExecutor {
    /// Candle model on GPU 0 for prefill
    model: Arc<crate::architectures::llama::LlamaModelWrapper>,
    /// TP decode group (one runner per GPU)
    tp_group: Mutex<TpDecodeGroup>,
    info: ModelInfo,
    states: Mutex<std::collections::HashMap<String, TpCacheState>>,
    next_cache_id: AtomicU64,
    tp_size: usize,
}

#[cfg(feature = "tensor-parallel")]
impl TpModelExecutor {
    pub fn new(
        model: crate::architectures::llama::LlamaModelWrapper,
        tp_group: TpDecodeGroup,
        info: ModelInfo,
        tp_size: usize,
    ) -> Self {
        info!(
            "Created TpModelExecutor: tp_size={}, model={}",
            tp_size, info.model_id
        );
        Self {
            model: Arc::new(model),
            tp_group: Mutex::new(tp_group),
            info,
            states: Mutex::new(std::collections::HashMap::new()),
            next_cache_id: AtomicU64::new(1),
            tp_size,
        }
    }

    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<candle_core::Tensor> {
        candle_core::Tensor::new(token_ids, self.model.device())
            .map_err(|e| FerrumError::model(format!("tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))
    }

    fn wrap_tensor(&self, tensor: candle_core::Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }

    fn tensor_to_tokens(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        common::tensor_to_tokens(tensor)
    }
}

#[cfg(feature = "tensor-parallel")]
#[async_trait]
impl ModelExecutor for TpModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        // Prefill on GPU 0 via candle (all GPUs participate in TP decode only)
        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Empty input"));
        }

        let cache_id = format!(
            "tp-cache-{}",
            self.next_cache_id.fetch_add(1, Ordering::Relaxed)
        );

        let input_tensor = self.tokens_to_tensor(&tokens)?;
        let logits = self.model.forward_prefill(&input_tensor, &cache_id)?;

        let logits = match logits.dims().len() {
            2 => logits
                .unsqueeze(1)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?,
            3 => logits,
            d => return Err(FerrumError::model(format!("Unexpected logits rank: {d}"))),
        };
        let logits_ref = self.wrap_tensor(logits);

        let cfg = self.model.config();
        let handle = Arc::new(GenericKvCacheHandle::new(
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            self.model.device().clone(),
            tokens.len(),
            cache_id.clone(),
        ));

        self.states.lock().insert(
            cache_id,
            TpCacheState {
                sequence_length: tokens.len(),
            },
        );

        Ok(PrefillOutput::new(logits_ref, handle))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let handle = input
            .kv_cache
            .as_any()
            .downcast_ref::<GenericKvCacheHandle>()
            .ok_or_else(|| FerrumError::model("Invalid KV handle"))?;
        let cache_id = handle.request_cache_id().to_string();

        let seq_len = {
            let mut states = self.states.lock();
            if let Some(s) = states.get(&cache_id) {
                s.sequence_length
            } else {
                let len = handle.block_table().sequence_length;
                states.insert(
                    cache_id.clone(),
                    TpCacheState {
                        sequence_length: len,
                    },
                );
                len
            }
        };

        let tokens = self.tensor_to_tokens(&input.input_ids)?;

        // TODO: migrate KV from candle prefill to TP runners
        // For now, use tp_group.decode_step which requires KV already in runners.
        // The full integration needs ensure_runner_kv_cache per rank.

        let logits = {
            let mut group = self.tp_group.lock();
            group
                .decode_step(tokens[0], seq_len, &cache_id)
                .map_err(|e| FerrumError::model(format!("tp_decode: {e}")))?
        };

        let cuda_dev = self
            .model
            .candle_device()
            .as_cuda_device()
            .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
        let vocab = self.info.vocab_size;
        let storage =
            candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(logits, cuda_dev.clone());
        let logits_tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1, 1, vocab),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let logits_ref = self.wrap_tensor(logits_tensor);

        let new_seq_len = seq_len + 1;
        {
            let mut states = self.states.lock();
            if let Some(s) = states.get_mut(&cache_id) {
                s.sequence_length = new_seq_len;
            }
        }
        let new_handle = Arc::new(handle.with_sequence_length(new_seq_len));
        Ok(DecodeOutput::new(logits_ref, new_handle))
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 1,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead, AttentionType::GroupedQuery],
            supports_dynamic_batching: false,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: true,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP16],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2 / self.tp_size) as u64,
                activation_memory_per_token: 4 * self.info.hidden_size,
                kv_cache_memory_per_token: 2 * self.info.num_layers * self.info.hidden_size
                    / self.tp_size,
                overhead_memory: 1024 * 1024 * 1024,
            },
        }
    }

    fn release_cache(&self, cache_id: &str) {
        self.states.lock().remove(cache_id);
        self.model.release_cache(cache_id);
        self.tp_group.lock().release_kv_cache(cache_id);
    }

    fn status(&self) -> ExecutorStatus {
        common::default_executor_status()
    }
}
