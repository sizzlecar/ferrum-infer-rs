//! `LlmExecutor<M>` — adapts a `DecoderOnlyLLM` to the `ModelExecutor` trait
//! the engine scheduler calls.
//!
//! This is the Model-as-Code equivalent of `GenericModelExecutor`: where
//! `GenericModelExecutor` wraps a `Box<dyn RunnerInterface>` (legacy
//! `ModelRunner<B>`), `LlmExecutor` wraps a `Box<dyn DecoderOnlyLLM>`
//! (new-style per-model code such as `Qwen3Model<B>`).
//!
//! Tokens/logits are currently bridged through candle Tensor for
//! `TensorRef` — Phase C will likely replace that with `SmallTensor` to
//! drop candle from the hot path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::debug;

use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus,
        MemoryRequirements, PrefillInput, PrefillOutput,
    },
    ModelExecutor,
};
use ferrum_types::{DataType, FerrumError, ModelInfo, Result};

use crate::common::DecoderOnlyLLM;

use super::common::{self, GenericKvCacheHandle};

/// Map a `ferrum_types::Device` to the matching `candle_core::Device`.
/// Used when materialising KV cache handles so downstream readers see
/// the real backend the model runs on (Metal / CUDA / CPU) rather than
/// a hard-coded CPU placeholder.
fn ferrum_device_to_candle(d: &ferrum_types::Device) -> candle_core::Device {
    match d {
        ferrum_types::Device::CPU => candle_core::Device::Cpu,
        #[cfg(feature = "cuda")]
        ferrum_types::Device::CUDA(i) => {
            candle_core::Device::new_cuda(*i as usize).unwrap_or(candle_core::Device::Cpu)
        }
        #[cfg(not(feature = "cuda"))]
        ferrum_types::Device::CUDA(_) => candle_core::Device::Cpu,
        #[cfg(all(any(target_os = "macos", target_os = "ios"), feature = "metal"))]
        ferrum_types::Device::Metal => {
            candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
        }
        #[cfg(not(all(any(target_os = "macos", target_os = "ios"), feature = "metal")))]
        ferrum_types::Device::Metal => candle_core::Device::Cpu,
        _ => candle_core::Device::Cpu,
    }
}

pub struct LlmExecutor {
    model: Mutex<Box<dyn DecoderOnlyLLM>>,
    info: ModelInfo,
    next_cache_id: AtomicU64,
}

impl LlmExecutor {
    pub fn new(model: Box<dyn DecoderOnlyLLM>, info: ModelInfo) -> Self {
        Self {
            model: Mutex::new(model),
            info,
            next_cache_id: AtomicU64::new(0),
        }
    }

    fn gen_cache_id(&self) -> String {
        format!(
            "llm-cache-{}",
            self.next_cache_id.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// Roll the KV cache for `cache_id` back to `new_len` positions.
    /// Used by speculative decoding on partial rejection. The caller must
    /// supply a `GenericKvCacheHandle` whose seq_len is also updated.
    pub fn truncate_kv_for_cache_id(&self, cache_id: &str, new_len: usize) {
        let mut model = self.model.lock();
        model.truncate_kv(cache_id, new_len);
    }
}

#[async_trait::async_trait]
impl ModelExecutor for LlmExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let tokens = common::tensor_to_tokens(&input.input_ids)?;

        // Reuse an existing cache_id when the caller supplies a KV handle
        // (chunked prefill) — fresh id only on the very first call for a
        // request. Without this, every chunk would create a new KV cache
        // at position 0 and subsequent chunks wouldn't see prior tokens.
        let supplied_handle_id = input
            .kv_cache
            .as_ref()
            .and_then(|h| {
                h.as_any()
                    .downcast_ref::<GenericKvCacheHandle>()
                    .map(|g| g.request_cache_id().to_string())
            });
        let cache_id = supplied_handle_id
            .clone()
            .unwrap_or_else(|| self.gen_cache_id());

        let logits = {
            let mut model = self.model.lock();
            model.prefill(&cache_id, &tokens)
        };

        // Wrap logits as TensorRef: [1, 1, vocab_size]
        let logits_tensor = candle_core::Tensor::new(&logits[..], &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("logits tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze2: {e}")))?;
        let logits_ref = common::wrap_tensor(logits_tensor);

        let cfg = self.model.lock().config().clone();
        // Sequence-length tracking across chunks: if the caller supplied a
        // GenericKvCacheHandle (chunked prefill continuation), add this
        // chunk's tokens to the prior length. Otherwise this is a fresh
        // prefill so seq_len == this call's token count. Without this the
        // handle would claim only the last chunk's length, misleading
        // decode() into rewriting the KV at an earlier position.
        let seq_len = input
            .kv_cache
            .as_ref()
            .and_then(|h| h.as_any().downcast_ref::<GenericKvCacheHandle>())
            .map(|g| {
                use ferrum_interfaces::KvCacheHandle;
                g.block_table().sequence_length + tokens.len()
            })
            .unwrap_or(tokens.len());

        let kv_handle = Arc::new(GenericKvCacheHandle::new(
            cfg.num_layers,
            cfg.num_kv_heads,
            cfg.head_dim,
            candle_core::Device::Cpu,
            seq_len,
            cache_id,
        ));

        Ok(PrefillOutput::new(logits_ref, kv_handle))
    }

    async fn truncate_kv(
        &self,
        kv_cache: &Arc<dyn ferrum_interfaces::KvCacheHandle>,
        new_len: usize,
    ) -> Result<()> {
        if let Some(g) = kv_cache.as_any().downcast_ref::<GenericKvCacheHandle>() {
            let cache_id = g.request_cache_id();
            self.model.lock().truncate_kv(cache_id, new_len);
        }
        Ok(())
    }

    async fn forward_verify(
        &self,
        inputs: &[ferrum_interfaces::model_executor::DecodeInput],
    ) -> Result<Vec<ferrum_interfaces::model_executor::DecodeOutput>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // All inputs must share the same KV handle (speculative decoding
        // contract). Extract cache_id + starting seq_len once.
        let first_handle = inputs[0].kv_cache.clone();
        let cache_id = first_handle
            .as_any()
            .downcast_ref::<GenericKvCacheHandle>()
            .ok_or_else(|| FerrumError::model(
                "forward_verify requires GenericKvCacheHandle input",
            ))?
            .request_cache_id()
            .to_string();
        let start_seq = {
            use ferrum_interfaces::KvCacheHandle;
            first_handle.block_table().sequence_length
        };

        // Collect the N+1 token ids.
        let mut token_ids: Vec<u32> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let toks = common::tensor_to_tokens(&input.input_ids)?;
            if toks.is_empty() {
                return Err(FerrumError::model("forward_verify input token empty"));
            }
            token_ids.push(toks[0]);
        }

        // One model forward for all N+1 positions → flat seq_len*vocab.
        let flat = {
            let mut model = self.model.lock();
            model.forward_verify(&cache_id, &token_ids)
        };

        let cfg = self.model.lock().config().clone();
        let vocab = cfg.vocab_size;

        // Record the actual backend device so downstream code that reads
        // `KvCacheHandle::device()` sees Metal/CUDA/CPU matching the
        // model's real location. The logits `Tensor` still wraps CPU data
        // because `B::to_vec` already moved it off-device.
        let candle_device = ferrum_device_to_candle(&self.info.device);

        // Split the flat logits into per-position tensors, each wrapped
        // with a handle whose seq_len reflects the positions written so
        // far. Matches what the spec runner expects from sequential
        // decode() calls.
        let mut outputs = Vec::with_capacity(inputs.len());
        for (i, _) in inputs.iter().enumerate() {
            let row = &flat[i * vocab..(i + 1) * vocab];
            let logits_tensor = candle_core::Tensor::new(row, &candle_core::Device::Cpu)
                .map_err(|e| FerrumError::model(format!("logits tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
            let logits_ref = common::wrap_tensor(logits_tensor);
            let handle = Arc::new(GenericKvCacheHandle::new(
                cfg.num_layers,
                cfg.num_kv_heads,
                cfg.head_dim,
                candle_device.clone(),
                start_seq + i + 1,
                cache_id.clone(),
            ));
            outputs.push(
                ferrum_interfaces::model_executor::DecodeOutput::new(logits_ref, handle),
            );
        }
        Ok(outputs)
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let input_handle = input
            .kv_cache
            .as_any()
            .downcast_ref::<GenericKvCacheHandle>()
            .ok_or_else(|| FerrumError::model("Invalid KV cache handle type"))?;

        let cache_id = input_handle.request_cache_id().to_string();
        let seq_len = {
            use ferrum_interfaces::KvCacheHandle;
            input_handle.block_table().sequence_length
        };

        let tokens = common::tensor_to_tokens(&input.input_ids)?;
        if tokens.is_empty() {
            return Err(FerrumError::model("Decode input is empty"));
        }
        let token = tokens[0];

        debug!("LlmExecutor decode: token={token}, pos={seq_len}");

        let logits = {
            let mut model = self.model.lock();
            model.decode(&cache_id, token, seq_len as u32)
        };

        let logits_tensor = candle_core::Tensor::new(&logits[..], &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("logits tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
        let logits_ref = common::wrap_tensor(logits_tensor);

        let kv_handle = Arc::new(input_handle.with_sequence_length(seq_len + 1));
        Ok(DecodeOutput::new(logits_ref, kv_handle))
    }

    /// Override default fallback to acquire the model lock ONCE for the whole
    /// batch, avoiding N round-trips through parking_lot. Does not yet do
    /// true attention batching (each cache has its own kv_len), but removes
    /// mutex churn that was serialising concurrent requests at async level.
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        // Pre-extract all per-input metadata OUTSIDE the lock — this is pure
        // borrow/downcast work that doesn't touch the model.
        struct Prep {
            cache_id: String,
            token: u32,
            seq_len: u32,
            handle: Arc<GenericKvCacheHandle>,
        }
        let mut prepped: Vec<Prep> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let input_handle = input
                .kv_cache
                .as_any()
                .downcast_ref::<GenericKvCacheHandle>()
                .ok_or_else(|| FerrumError::model("Invalid KV cache handle type"))?;
            use ferrum_interfaces::KvCacheHandle;
            let seq_len = input_handle.block_table().sequence_length as u32;
            let tokens = common::tensor_to_tokens(&input.input_ids)?;
            if tokens.is_empty() {
                return Err(FerrumError::model("Decode input is empty"));
            }
            prepped.push(Prep {
                cache_id: input_handle.request_cache_id().to_string(),
                token: tokens[0],
                seq_len,
                handle: Arc::new(input_handle.with_sequence_length((seq_len + 1) as usize)),
            });
        }

        // One lock for the whole batch, dispatch to model's decode_batch —
        // which implementations may fuse into a single forward pass (GEMMs
        // with m=batch, per-item attention) for true concurrency speedup.
        // Trait default falls back to sequential decode per item.
        let all_logits: Vec<Vec<f32>> = {
            let mut model = self.model.lock();
            let tuples: Vec<(String, u32, u32)> = prepped
                .iter()
                .map(|p| (p.cache_id.clone(), p.token, p.seq_len))
                .collect();
            model.decode_batch(&tuples)
        };

        let mut outputs = Vec::with_capacity(prepped.len());
        for (p, logits) in prepped.into_iter().zip(all_logits.into_iter()) {
            debug!(
                "LlmExecutor batch_decode: token={}, pos={}",
                p.token, p.seq_len
            );
            let logits_tensor = candle_core::Tensor::new(&logits[..], &candle_core::Device::Cpu)
                .map_err(|e| FerrumError::model(format!("logits tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?;
            let logits_ref = common::wrap_tensor(logits_tensor);
            outputs.push(DecodeOutput::new(logits_ref, p.handle));
        }
        Ok(outputs)
    }

    fn release_cache(&self, cache_id: &str) {
        self.model.lock().release(cache_id);
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let cfg = self.model.lock().config().clone();
        ExecutorCapabilities {
            max_batch_size: 256,
            max_sequence_length: cfg.max_seq_len,
            attention_mechanisms: vec![AttentionType::GroupedQuery],
            supports_dynamic_batching: true,
            supports_continuous_batching: true,
            supports_speculative_decoding: false,
            supports_tensor_parallelism: false,
            supports_pipeline_parallelism: false,
            supported_dtypes: vec![DataType::FP32],
            supported_devices: vec![self.info.device.clone()],
            memory_requirements: MemoryRequirements {
                parameter_memory: (self.info.num_parameters * 4) as u64,
                activation_memory_per_token: cfg.hidden_size * 4,
                kv_cache_memory_per_token: cfg.hidden_size * 2,
                overhead_memory: 256 * 1024 * 1024,
            },
        }
    }

    fn status(&self) -> ExecutorStatus {
        common::default_executor_status()
    }
}
