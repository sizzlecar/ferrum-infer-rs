//! Qwen3 model executor using Candle

use async_trait::async_trait;
use candle_core::{Device as CandleDevice, Tensor};
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus,
        MemoryRequirements, PrefillInput, PrefillOutput,
    },
    ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, FerrumError, ModelInfo, Result};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tracing::{debug, info};

use crate::architectures::qwen3::Qwen3ModelWrapper;
use crate::executor::common;

#[derive(Debug, Clone)]
struct Qwen3CacheState {
    sequence_length: usize,
}

/// Candle-based Qwen3 model executor with multi-sequence support.
///
/// Each active sequence gets its own KV cache keyed by a unique cache_id.
/// This allows concurrent prefill and decode across many sequences without
/// one sequence's prefill destroying another's KV cache.
///
/// On CUDA devices, lazily creates a `CudaDecodeRunner` that bypasses candle
/// for the decode hot path, using cuBLAS + custom kernels with pre-allocated
/// buffers and optional CUDA Graph acceleration.
pub struct Qwen3ModelExecutor {
    model: Arc<Qwen3ModelWrapper>,
    info: ModelInfo,
    states: Mutex<HashMap<String, Qwen3CacheState>>,
    next_cache_id: AtomicU64,
    /// CUDA decode runner (created lazily on first CUDA decode call).
    #[cfg(feature = "cuda")]
    cuda_runner: Mutex<Option<ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner>>,
    /// Whether CUDA runner initialization has been attempted (avoid retrying on failure).
    #[cfg(feature = "cuda")]
    cuda_runner_init_attempted: std::sync::atomic::AtomicBool,
}

impl Qwen3ModelExecutor {
    pub fn new(model: Qwen3ModelWrapper, info: ModelInfo) -> Self {
        info!("Created Qwen3ModelExecutor for: {}", info.model_id);

        Self {
            model: Arc::new(model),
            info,
            states: Mutex::new(HashMap::new()),
            next_cache_id: AtomicU64::new(1),
            #[cfg(feature = "cuda")]
            cuda_runner: Mutex::new(None),
            #[cfg(feature = "cuda")]
            cuda_runner_init_attempted: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Try to initialize the CUDA decode runner (lazy, first call only).
    /// Returns true if runner is available for use.
    ///
    #[cfg(feature = "cuda")]
    fn ensure_cuda_runner(&self) -> bool {
        // FERRUM_DISABLE_CUDA_RUNNER=1 → always use candle path
        if std::env::var("FERRUM_DISABLE_CUDA_RUNNER").unwrap_or_default() == "1" {
            return false;
        }
        if self.cuda_runner.lock().is_some() {
            return true;
        }
        if self
            .cuda_runner_init_attempted
            .swap(true, Ordering::Relaxed)
        {
            return false;
        }
        if !matches!(self.model.candle_device(), CandleDevice::Cuda(_)) {
            return false;
        }
        match self.model.create_decode_runner() {
            Ok(runner) => {
                info!("CUDA decode runner initialized — decode will bypass candle");
                *self.cuda_runner.lock() = Some(runner);
                true
            }
            Err(e) => {
                tracing::warn!("CUDA decode runner init failed, using candle path: {e}");
                false
            }
        }
    }

    /// Release a sequence's KV cache, freeing GPU memory.
    /// Should be called when a request completes.
    pub fn release_sequence(&self, cache_id: &str) {
        self.states.lock().remove(cache_id);
        self.model.release_cache(cache_id);
        // Also release from CUDA decode runner if active
        #[cfg(feature = "cuda")]
        if let Some(ref mut runner) = *self.cuda_runner.lock() {
            runner.release_kv_cache(cache_id);
        }
        debug!("Released KV cache for sequence: {}", cache_id);
    }

    /// Ensure the CUDA decode runner has KV cache for a sequence.
    /// On first call for a sequence, migrates KV data from candle's PreAllocKvCache.
    #[cfg(feature = "cuda")]
    fn ensure_runner_kv_cache(&self, cache_id: &str, _seq_len: usize) -> Result<()> {
        use candle_core::Storage;

        let mut runner_guard = self.cuda_runner.lock();
        let runner = match runner_guard.as_mut() {
            Some(r) => r,
            None => return Ok(()), // No runner, will fall through to candle
        };

        // Check if runner already has KV cache for this sequence
        if runner.has_kv_cache(cache_id) {
            return Ok(());
        }

        // Export KV data from candle model
        let kv_data_tensors = self.model.export_kv_cache(cache_id).ok_or_else(|| {
            FerrumError::model(format!("No candle KV cache to export for: {cache_id}"))
        })?;

        if kv_data_tensors.is_empty() {
            return Err(FerrumError::model("Empty KV cache export"));
        }
        let prefill_len = kv_data_tensors[0].2;
        let max_len = kv_data_tensors[0].3;

        // Extract CudaSlice from each layer's K/V tensors.
        // clone() on CudaSlice does a D2D copy — we get independent buffers.
        let mut kv_slices = Vec::new();
        for (k_tensor, v_tensor, _len, _max) in &kv_data_tensors {
            let (k_s, _) = k_tensor.storage_and_layout();
            let (v_s, _) = v_tensor.storage_and_layout();
            let k_cuda = match &*k_s {
                Storage::Cuda(cs) => cs
                    .as_cuda_slice::<half::f16>()
                    .map_err(|e| FerrumError::model(format!("KV slice extract: {e}")))?
                    .clone(),
                _ => return Err(FerrumError::model("KV cache not on CUDA")),
            };
            let v_cuda = match &*v_s {
                Storage::Cuda(cs) => cs
                    .as_cuda_slice::<half::f16>()
                    .map_err(|e| FerrumError::model(format!("KV slice extract: {e}")))?
                    .clone(),
                _ => return Err(FerrumError::model("KV cache not on CUDA")),
            };
            drop(k_s);
            drop(v_s);
            kv_slices.push((k_cuda, v_cuda));
        }

        runner
            .init_kv_cache(cache_id, kv_slices, prefill_len, max_len)
            .map_err(|e| FerrumError::model(format!("CUDA runner KV init failed: {e}")))?;

        debug!("Migrated KV cache to CUDA runner for sequence: {cache_id}");
        Ok(())
    }

    fn tensor_to_tokens(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        common::tensor_to_tokens(tensor)
    }

    fn tokens_to_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        common::tokens_to_tensor(tokens, self.model.candle_device())
    }

    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        common::wrap_tensor(tensor)
    }
}

#[async_trait]
impl ModelExecutor for Qwen3ModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!(
            "Qwen3 Prefill: batch={}, seq_len={}",
            input.batch_size(),
            input.sequence_length()
        );

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Prefill input is empty"));
        }

        let cache_id = format!(
            "qwen3-cache-{}",
            self.next_cache_id.fetch_add(1, Ordering::Relaxed)
        );

        let input_tensor = self.tokens_to_tensor(&tokens)?;

        // Each sequence gets its own KV cache slot; no need to clear other sequences.
        let logits = self
            .model
            .forward_prefill(&input_tensor, &cache_id)
            .map_err(|e| FerrumError::model(format!("Qwen3 prefill failed: {}", e)))?;

        let logits = match logits.dims().len() {
            2 => logits
                .unsqueeze(1)
                .map_err(|e| FerrumError::model(format!("Unsqueeze logits failed: {}", e)))?,
            3 => logits,
            dims => {
                return Err(FerrumError::model(format!(
                    "Unexpected Qwen3 prefill logits rank: {} (shape {:?})",
                    dims,
                    logits.dims()
                )))
            }
        };

        let logits_ref = self.wrap_tensor(logits);

        let cfg = self.model.config();
        let kv_handle = Arc::new(common::GenericKvCacheHandle::new(
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.head_dim,
            self.model.device().clone(),
            tokens.len(),
            cache_id.clone(),
        ));

        self.states.lock().insert(
            cache_id,
            Qwen3CacheState {
                sequence_length: tokens.len(),
            },
        );

        Ok(PrefillOutput::new(logits_ref, kv_handle))
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        debug!("Qwen3 Decode: batch={}", input.batch_size());

        let input_handle = input
            .kv_cache
            .as_any()
            .downcast_ref::<common::GenericKvCacheHandle>()
            .ok_or_else(|| FerrumError::model("Invalid KV cache handle type for Qwen3 executor"))?;
        let req_cache_id = input_handle.request_cache_id().to_string();

        let seq_len = {
            let states = self.states.lock();
            let state = states.get(&req_cache_id).ok_or_else(|| {
                FerrumError::model(format!(
                    "Decode called for unknown sequence: {}",
                    req_cache_id
                ))
            })?;
            state.sequence_length
        };

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Decode input is empty"));
        }

        // Try CUDA decode runner path (bypasses candle for the hot path).
        // Falls back to candle if runner fails (e.g., KV cache not yet initialized).
        #[cfg(feature = "cuda")]
        if tokens.len() == 1 && self.ensure_cuda_runner() {
            let token_id = tokens[0];

            // Ensure CUDA runner has KV cache for this sequence
            // (migrated from candle's PreAllocKvCache on first decode call)
            self.ensure_runner_kv_cache(&req_cache_id, seq_len)?;

            let cuda_result = {
                let mut runner = self.cuda_runner.lock();
                if let Some(ref mut runner) = *runner {
                    Some(runner.decode_step_graphed(token_id, seq_len, &req_cache_id))
                } else {
                    None
                }
            };
            if let Some(Ok(logits_slice)) = cuda_result {
                // Wrap CudaSlice into candle Tensor (zero-copy, stays on GPU)
                let cuda_dev = self
                    .model
                    .candle_device()
                    .as_cuda_device()
                    .map_err(|e| FerrumError::model(format!("Not CUDA device: {e}")))?;
                let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    logits_slice,
                    cuda_dev.clone(),
                );
                let logits_tensor = candle_core::Tensor::from_storage(
                    candle_core::Storage::Cuda(storage),
                    (1, 1, self.info.vocab_size),
                    candle_core::op::BackpropOp::none(),
                    false,
                );

                // FERRUM_LOG_TOKENS=1 → log argmax token for every decode step
                if std::env::var("FERRUM_LOG_TOKENS").unwrap_or_default() == "1" || seq_len == 13 {
                    if let Ok(flat) = logits_tensor.flatten_all() {
                        if let Ok(vals) = flat.to_vec1::<half::f16>() {
                            let mut indexed: Vec<(usize, f32)> = vals
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (i, v.to_f32()))
                                .collect();
                            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            let top5: Vec<String> = indexed[..5]
                                .iter()
                                .map(|(i, v)| format!("{}:{:.2}", i, v))
                                .collect();
                            tracing::info!("[CUDA] pos={} top5=[{}]", seq_len, top5.join(", "));
                        }
                    }
                }

                let logits_ref = self.wrap_tensor(logits_tensor);

                let new_seq_len = {
                    let mut states = self.states.lock();
                    if let Some(state) = states.get_mut(&req_cache_id) {
                        state.sequence_length += 1;
                        state.sequence_length
                    } else {
                        seq_len + 1
                    }
                };
                let new_handle = Arc::new(input_handle.with_sequence_length(new_seq_len));

                return Ok(DecodeOutput::new(logits_ref, new_handle));
            } else if let Some(Err(e)) = cuda_result {
                // CUDA runner failed — log and fall through to candle path
                tracing::debug!("CUDA decode runner failed, falling back to candle: {e}");
            }
        }

        // Fallback: standard candle decode path
        let input_tensor = self.tokens_to_tensor(&tokens)?;

        let logits = self
            .model
            .forward_decode(&input_tensor, seq_len, &req_cache_id)
            .map_err(|e| FerrumError::model(format!("Qwen3 decode failed: {}", e)))?;

        if std::env::var("FERRUM_LOG_TOKENS").unwrap_or_default() == "1" || seq_len == 13 {
            if let Ok(flat) = logits.flatten_all() {
                if let Ok(vals) = flat.to_vec1::<half::f16>() {
                    let mut indexed: Vec<(usize, f32)> = vals
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i, v.to_f32()))
                        .collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top5: Vec<String> = indexed[..5]
                        .iter()
                        .map(|(i, v)| format!("{}:{:.2}", i, v))
                        .collect();
                    tracing::info!("[CANDLE] pos={} top5=[{}]", seq_len, top5.join(", "));
                }
            }
        }

        let logits_ref = self.wrap_tensor(logits);

        let new_seq_len = {
            let mut states = self.states.lock();
            if let Some(state) = states.get_mut(&req_cache_id) {
                state.sequence_length += tokens.len();
                state.sequence_length
            } else {
                seq_len + tokens.len()
            }
        };
        let new_handle = Arc::new(input_handle.with_sequence_length(new_seq_len));

        Ok(DecodeOutput::new(logits_ref, new_handle))
    }

    #[cfg(feature = "cuda")]
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        use ferrum_cuda_kernels::cuda_decode::BatchDecodeRequest;

        if inputs.len() <= 1 || !self.ensure_cuda_runner() {
            // Fallback to per-request decode
            let mut outputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                outputs.push(self.decode(input).await?);
            }
            return Ok(outputs);
        }

        // Extract cache_ids, positions, and tokens for all inputs
        let mut requests = Vec::with_capacity(inputs.len());
        let mut cache_ids = Vec::with_capacity(inputs.len());
        let mut seq_lens = Vec::with_capacity(inputs.len());

        for input in inputs {
            let handle = input
                .kv_cache
                .as_any()
                .downcast_ref::<common::GenericKvCacheHandle>()
                .ok_or_else(|| FerrumError::model("Invalid KV cache handle"))?;
            let cache_id = handle.request_cache_id().to_string();
            let seq_len = {
                let states = self.states.lock();
                states
                    .get(&cache_id)
                    .map(|s| s.sequence_length)
                    .unwrap_or(0)
            };
            let tokens = self.tensor_to_tokens(&input.input_ids)?;
            if tokens.len() != 1 {
                return Err(FerrumError::model("batch_decode requires single-token inputs"));
            }

            self.ensure_runner_kv_cache(&cache_id, seq_len)?;
            cache_ids.push(cache_id);
            seq_lens.push(seq_len);
            requests.push((tokens[0], seq_len));
        }

        // Build BatchDecodeRequests
        let batch_requests: Vec<BatchDecodeRequest<'_>> = requests
            .iter()
            .zip(cache_ids.iter())
            .map(|((token_id, position), cache_key)| BatchDecodeRequest {
                token_id: *token_id,
                position: *position,
                cache_key: cache_key.as_str(),
            })
            .collect();

        // Call runner.batch_decode_step
        let logits_slice = {
            let mut runner = self.cuda_runner.lock();
            let runner = runner
                .as_mut()
                .ok_or_else(|| FerrumError::model("CUDA runner not initialized"))?;
            runner.batch_decode_step(&batch_requests)
                .map_err(|e| FerrumError::model(format!("batch_decode_step: {e}")))?
        };

        // Split [B * vocab] logits into per-request outputs
        let batch = inputs.len();
        let vocab = self.info.vocab_size;
        let cuda_dev = self
            .model
            .candle_device()
            .as_cuda_device()
            .map_err(|e| FerrumError::model(format!("Not CUDA: {e}")))?;

        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
            logits_slice,
            cuda_dev.clone(),
        );
        let logits_tensor = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (batch, 1, vocab),
            candle_core::op::BackpropOp::none(),
            false,
        );

        let mut outputs = Vec::with_capacity(batch);
        for (i, input) in inputs.iter().enumerate() {
            let item_logits = logits_tensor
                .narrow(0, i, 1)
                .map_err(|e| FerrumError::model(format!("logits narrow: {e}")))?;
            let logits_ref = self.wrap_tensor(item_logits);

            let handle = input
                .kv_cache
                .as_any()
                .downcast_ref::<common::GenericKvCacheHandle>()
                .unwrap();
            let new_seq_len = {
                let mut states = self.states.lock();
                if let Some(state) = states.get_mut(&cache_ids[i]) {
                    state.sequence_length += 1;
                    state.sequence_length
                } else {
                    seq_lens[i] + 1
                }
            };
            let new_handle = Arc::new(handle.with_sequence_length(new_seq_len));
            outputs.push(DecodeOutput::new(logits_ref, new_handle));
        }

        Ok(outputs)
    }

    #[cfg(not(feature = "cuda"))]
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.decode(input).await?);
        }
        Ok(outputs)
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        ExecutorCapabilities {
            max_batch_size: 256,
            max_sequence_length: self.info.max_sequence_length,
            attention_mechanisms: vec![AttentionType::MultiHead, AttentionType::GroupedQuery],
            supports_dynamic_batching: true,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP16, DataType::FP32, DataType::BF16],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2) as u64,
                activation_memory_per_token: self.info.hidden_size * 4,
                kv_cache_memory_per_token: self.info.hidden_size * 2,
                overhead_memory: 256 * 1024 * 1024,
            },
        }
    }

    fn release_cache(&self, cache_id: &str) {
        self.release_sequence(cache_id);
    }

    fn status(&self) -> ExecutorStatus {
        common::default_executor_status()
    }
}

// Qwen3KvCacheHandle replaced by common::GenericKvCacheHandle
