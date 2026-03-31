//! Llama model executor using our custom Llama implementation.
//!
//! Uses GenericKvCacheHandle (like Qwen3) with per-request cache_id.
//! Supports CUDA decode runner for GPU acceleration.

use async_trait::async_trait;
use candle_core::Tensor;
use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus,
        MemoryRequirements, PrefillInput, PrefillOutput,
    },
    KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_types::{DataType, Device, FerrumError, ModelInfo, Result};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tracing::{debug, info};

use super::common::{self, GenericKvCacheHandle};
use crate::architectures::llama::LlamaModelWrapper;
use crate::tensor_wrapper::CandleTensorWrapper;
use parking_lot::Mutex;

struct LlamaCacheState {
    sequence_length: usize,
}

/// Llama model executor
pub struct CandleModelExecutor {
    model: Arc<LlamaModelWrapper>,
    info: ModelInfo,
    states: Mutex<HashMap<String, LlamaCacheState>>,
    next_cache_id: AtomicU64,
    #[cfg(feature = "cuda")]
    cuda_runner: Mutex<Option<ferrum_cuda_kernels::cuda_decode::CudaDecodeRunner>>,
}

impl CandleModelExecutor {
    pub fn new(model: LlamaModelWrapper, info: ModelInfo) -> Self {
        info!("Created CandleModelExecutor (Llama) for: {}", info.model_id);
        Self {
            model: Arc::new(model),
            info,
            states: Mutex::new(HashMap::new()),
            next_cache_id: AtomicU64::new(1),
            #[cfg(feature = "cuda")]
            cuda_runner: Mutex::new(None),
        }
    }

    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<Tensor> {
        Tensor::new(token_ids, self.model.device())
            .map_err(|e| FerrumError::model(format!("tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))
    }

    fn wrap_tensor(&self, tensor: Tensor) -> TensorRef {
        Arc::new(CandleTensorWrapper::new(tensor))
    }

    fn tensor_to_tokens(&self, tensor: &TensorRef) -> Result<Vec<u32>> {
        common::tensor_to_tokens(tensor)
    }

    #[cfg(feature = "cuda")]
    fn ensure_cuda_runner(&self) -> bool {
        if self.cuda_runner.lock().is_some() {
            return true;
        }
        if std::env::var("FERRUM_DISABLE_CUDA_RUNNER").map_or(false, |v| v == "1") {
            return false;
        }
        match self.model.create_decode_runner() {
            Ok(runner) => {
                info!("CUDA decode runner initialized for Llama");
                *self.cuda_runner.lock() = Some(runner);
                true
            }
            Err(e) => {
                tracing::warn!("CUDA runner init failed for Llama: {e}");
                false
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ensure_runner_kv_cache(&self, cache_id: &str, _seq_len: usize) -> Result<()> {
        use candle_core::Storage;

        let mut runner_guard = self.cuda_runner.lock();
        let runner = match runner_guard.as_mut() {
            Some(r) => r,
            None => return Ok(()),
        };
        if runner.has_kv_cache(cache_id) {
            return Ok(());
        }

        // Export from our custom model's per-request KV cache
        let kv_data = self
            .model
            .export_kv_cache(cache_id)
            .or_else(|| {
                cache_id
                    .rfind("-clone-")
                    .and_then(|pos| self.model.export_kv_cache(&cache_id[..pos]))
            })
            .ok_or_else(|| FerrumError::model(format!("No KV cache to export for: {cache_id}")))?;

        if kv_data.is_empty() {
            return Err(FerrumError::model("Empty KV cache export"));
        }
        let prefill_len = kv_data[0].2;
        let max_len = kv_data[0].3;

        let mut kv_slices = Vec::new();
        for (k_tensor, v_tensor, _len, _max) in &kv_data {
            let (k_s, _) = k_tensor.storage_and_layout();
            let (v_s, _) = v_tensor.storage_and_layout();
            let k_cuda = match &*k_s {
                Storage::Cuda(cs) => cs
                    .as_cuda_slice::<half::f16>()
                    .map_err(|e| FerrumError::model(format!("KV extract: {e}")))?
                    .clone(),
                _ => return Err(FerrumError::model("KV not on CUDA")),
            };
            let v_cuda = match &*v_s {
                Storage::Cuda(cs) => cs
                    .as_cuda_slice::<half::f16>()
                    .map_err(|e| FerrumError::model(format!("KV extract: {e}")))?
                    .clone(),
                _ => return Err(FerrumError::model("KV not on CUDA")),
            };
            drop(k_s);
            drop(v_s);
            kv_slices.push((k_cuda, v_cuda));
        }

        runner
            .init_kv_cache(cache_id, kv_slices, prefill_len, max_len)
            .map_err(|e| FerrumError::model(format!("KV init: {e}")))?;

        if !self.states.lock().contains_key(cache_id) {
            self.states.lock().insert(
                cache_id.to_string(),
                LlamaCacheState {
                    sequence_length: prefill_len,
                },
            );
        }

        debug!("Migrated Llama KV to CUDA runner: {cache_id}");
        Ok(())
    }

    pub fn release_sequence(&self, cache_id: &str) {
        self.states.lock().remove(cache_id);
        self.model.release_cache(cache_id);
        #[cfg(feature = "cuda")]
        if let Some(ref mut runner) = *self.cuda_runner.lock() {
            runner.release_kv_cache(cache_id);
        }
    }
}

#[async_trait]
impl ModelExecutor for CandleModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!("Llama Prefill: seq_len={}", input.sequence_length());

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Empty input"));
        }

        let cache_id = format!(
            "llama-cache-{}",
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
            LlamaCacheState {
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
            .ok_or_else(|| FerrumError::model("Invalid KV handle for Llama"))?;
        let cache_id = handle.request_cache_id().to_string();

        let seq_len = {
            let mut states = self.states.lock();
            if let Some(s) = states.get(&cache_id) {
                s.sequence_length
            } else {
                let len = handle.block_table().sequence_length;
                states.insert(
                    cache_id.clone(),
                    LlamaCacheState {
                        sequence_length: len,
                    },
                );
                len
            }
        };

        let tokens = self.tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Empty decode input"));
        }

        // Try CUDA runner path
        #[cfg(feature = "cuda")]
        {
            if std::env::var("FERRUM_DISABLE_CUDA_RUNNER").map_or(true, |v| v != "1") {
                if self.ensure_cuda_runner() {
                    self.ensure_runner_kv_cache(&cache_id, seq_len)?;

                    let logits = {
                        let mut runner = self.cuda_runner.lock();
                        let runner = runner
                            .as_mut()
                            .ok_or_else(|| FerrumError::model("CUDA runner gone"))?;
                        runner
                            .decode_step(tokens[0], seq_len, &cache_id)
                            .map_err(|e| FerrumError::model(format!("decode: {e}")))?
                    };

                    // Diagnostic: compare runner logits with candle
                    if std::env::var("FERRUM_DIAG").is_ok() {
                        // Compare weight norms
                        if seq_len <= 12 {
                            let model = self.model.model.lock();
                            let qw = model.layers[0].self_attn.q_proj.weight();
                            eprintln!(
                                "[DIAG] candle q_proj shape={:?} dtype={:?}",
                                qw.shape(),
                                qw.dtype()
                            );
                            if let Ok(sum) = qw
                                .to_dtype(candle_core::DType::F32)
                                .and_then(|t| t.abs()?.sum_all()?.to_scalar::<f32>())
                            {
                                eprintln!("[DIAG] candle q_proj abs_sum={sum}");
                            }
                            drop(model);
                            let mut runner = self.cuda_runner.lock();
                            if let Some(ref r) = *runner {
                                eprintln!(
                                    "[DIAG] runner qkv_w len={}",
                                    match &r.weight_layers()[0].qkv_w {
                                        ferrum_cuda_kernels::weight_store::LinearWeight::Fp16(
                                            w,
                                        ) => w.len,
                                        _ => 0,
                                    }
                                );
                            }
                            drop(runner);
                        }
                        // Use same cache_id to compare with identical KV context.
                        // This advances the candle KV by 1 token (side-effect), but
                        // the runner manages its own KV separately.
                        let candle_logits = self.model.forward_decode(
                            &self.tokens_to_tensor(&tokens)?,
                            seq_len,
                            &cache_id,
                        )?;
                        if let Ok(cl) = candle_logits.flatten_all().and_then(|t| t.to_vec1::<f32>())
                        {
                            let mut ci: Vec<(usize, f32)> =
                                cl.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                            ci.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            eprintln!("[DIAG] candle top5: {:?}", &ci[..5]);
                        }
                    }

                    let cuda_dev = self
                        .model
                        .candle_device()
                        .as_cuda_device()
                        .map_err(|e| FerrumError::model(format!("not CUDA: {e}")))?;
                    let vocab = self.info.vocab_size;
                    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                        logits,
                        cuda_dev.clone(),
                    );
                    let logits_tensor = candle_core::Tensor::from_storage(
                        candle_core::Storage::Cuda(storage),
                        (1, 1, vocab),
                        candle_core::op::BackpropOp::none(),
                        false,
                    );

                    // Diagnostic: runner top-5
                    if std::env::var("FERRUM_DIAG").is_ok() {
                        if let Ok(rl) = logits_tensor
                            .flatten_all()
                            .and_then(|t| t.to_vec1::<half::f16>())
                        {
                            let mut ri: Vec<(usize, f32)> = rl
                                .iter()
                                .enumerate()
                                .map(|(i, v)| (i, v.to_f32()))
                                .collect();
                            ri.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            eprintln!("[DIAG] runner top5: {:?}", &ri[..5]);
                        }
                    }

                    let logits_ref = self.wrap_tensor(logits_tensor);

                    let new_seq_len = seq_len + 1;
                    {
                        let mut states = self.states.lock();
                        if let Some(s) = states.get_mut(&cache_id) {
                            s.sequence_length = new_seq_len;
                        }
                    }
                    let new_handle = Arc::new(handle.with_sequence_length(new_seq_len));
                    return Ok(DecodeOutput::new(logits_ref, new_handle));
                }
            }
        }

        // Candle fallback
        let input_tensor = self.tokens_to_tensor(&tokens)?;
        let logits = self
            .model
            .forward_decode(&input_tensor, seq_len, &cache_id)?;
        let logits = match logits.dims().len() {
            2 => logits
                .unsqueeze(1)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?,
            3 => logits,
            _ => logits,
        };
        let logits_ref = self.wrap_tensor(logits);

        let new_seq_len = seq_len + tokens.len();
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
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP16, DataType::FP32],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 2) as u64,
                activation_memory_per_token: 4 * self.info.hidden_size,
                kv_cache_memory_per_token: 2 * self.info.num_layers * self.info.hidden_size,
                overhead_memory: 1024 * 1024 * 1024,
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
