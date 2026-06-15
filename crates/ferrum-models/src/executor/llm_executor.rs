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
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, MutexGuard};
use tracing::debug;

use ferrum_interfaces::{
    model_executor::{
        AttentionType, DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus,
        MemoryRequirements, PrefillInput, PrefillOutput, UnifiedBatch,
    },
    ModelExecutor,
};
use ferrum_types::{DataType, FerrumError, ModelInfo, Result};

use crate::common::DecoderOnlyLLM;
use crate::lora::ActiveLoraAdapter;

use super::common::{self, GenericKvCacheHandle};

#[derive(Debug, Clone, PartialEq, Eq)]
struct LlmExecutorRuntimeEnv {
    batch_prefill_prof: bool,
    batch_decode_prof: bool,
}

impl LlmExecutorRuntimeEnv {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
    {
        let mut batch_prefill_prof = false;
        let mut batch_decode_prof = false;

        for (key, _) in vars {
            match key.as_ref() {
                "FERRUM_BATCH_PREFILL_PROF" => batch_prefill_prof = true,
                "FERRUM_BATCH_DECODE_PROF" => batch_decode_prof = true,
                _ => {}
            }
        }

        Self {
            batch_prefill_prof,
            batch_decode_prof,
        }
    }
}

fn llm_executor_runtime_env() -> &'static LlmExecutorRuntimeEnv {
    static CONFIG: OnceLock<LlmExecutorRuntimeEnv> = OnceLock::new();
    CONFIG.get_or_init(LlmExecutorRuntimeEnv::from_env)
}

fn active_lora_from_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<Option<ActiveLoraAdapter>> {
    let name = metadata
        .get("ferrum_lora_adapter")
        .and_then(|value| value.as_str());
    let path = metadata
        .get("ferrum_lora_path")
        .and_then(|value| value.as_str());
    match (name, path) {
        (Some(name), Some(path)) => Ok(Some(ActiveLoraAdapter {
            name: name.to_string(),
            path: std::path::PathBuf::from(path),
        })),
        (None, None) => Ok(None),
        _ => Err(FerrumError::model(
            "incomplete LoRA metadata: expected ferrum_lora_adapter and ferrum_lora_path",
        )),
    }
}

fn metadata_requires_full_logits(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> bool {
    metadata
        .get("ferrum_require_full_logits")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn metadata_kv_capacity_hint(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Option<usize> {
    metadata
        .get("ferrum_kv_capacity_hint")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
}

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
        _ => candle_core::Device::Cpu,
    }
}

pub struct LlmExecutor {
    model: Mutex<Box<dyn DecoderOnlyLLM>>,
    info: ModelInfo,
    next_cache_id: AtomicU64,
    total_model_lock_wait_us: AtomicU64,
    model_lock_wait_samples: AtomicU64,
}

impl LlmExecutor {
    pub fn new(model: Box<dyn DecoderOnlyLLM>, info: ModelInfo) -> Self {
        Self {
            model: Mutex::new(model),
            info,
            next_cache_id: AtomicU64::new(0),
            total_model_lock_wait_us: AtomicU64::new(0),
            model_lock_wait_samples: AtomicU64::new(0),
        }
    }

    fn lock_model(&self) -> MutexGuard<'_, Box<dyn DecoderOnlyLLM>> {
        let start = std::time::Instant::now();
        let guard = self.model.lock();
        self.record_model_lock_wait(start.elapsed());
        guard
    }

    fn record_model_lock_wait(&self, duration: std::time::Duration) {
        self.total_model_lock_wait_us.fetch_add(
            duration.as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.model_lock_wait_samples.fetch_add(1, Ordering::Relaxed);
    }

    fn model_lock_metrics_json(&self) -> serde_json::Value {
        let samples = self.model_lock_wait_samples.load(Ordering::Relaxed);
        let total_us = self.total_model_lock_wait_us.load(Ordering::Relaxed);
        serde_json::json!({
            "schema_version": 1,
            "samples": samples,
            "total_wait_time_us": total_us,
            "avg_wait_time_ms": if samples == 0 {
                0.0
            } else {
                total_us as f64 / samples as f64 / 1000.0
            },
        })
    }

    fn attach_model_lock_metrics(&self, mut snapshot: serde_json::Value) -> serde_json::Value {
        let lock_metrics = self.model_lock_metrics_json();
        if let Some(obj) = snapshot.as_object_mut() {
            obj.insert("executor_model_lock".to_string(), lock_metrics);
            snapshot
        } else {
            serde_json::json!({
                "cache_metrics": snapshot,
                "executor_model_lock": lock_metrics,
            })
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
        let mut model = self.lock_model();
        model.truncate_kv(cache_id, new_len);
    }
}

#[async_trait::async_trait]
impl ModelExecutor for LlmExecutor {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn supports_native_unified_decode(&self) -> bool {
        // CUDA has a native unified mixed prefill+decode forward; CPU and Metal
        // use the legacy split path. The device→capability mapping lives here
        // (the executor is backend-aware) so the engine needs no platform cfg.
        matches!(self.info.device, ferrum_types::Device::CUDA(_))
    }

    fn kv_capacity(&self) -> Option<usize> {
        Some(self.lock_model().kv_capacity())
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        let tokens = common::tensor_to_tokens(&input.input_ids)?;

        // Reuse an existing cache_id when the caller supplies a KV handle
        // (chunked prefill) — fresh id only on the very first call for a
        // request. Without this, every chunk would create a new KV cache
        // at position 0 and subsequent chunks wouldn't see prior tokens.
        let supplied_handle_id = input.kv_cache.as_ref().and_then(|h| {
            h.as_any()
                .downcast_ref::<GenericKvCacheHandle>()
                .map(|g| g.request_cache_id().to_string())
        });
        let cache_id = supplied_handle_id
            .clone()
            .unwrap_or_else(|| self.gen_cache_id());

        // For chunked-prefill continuation, the prior KV length is the seq
        // length already in the supplied handle; for fresh prefill it's 0.
        let prior_seq_len = input
            .kv_cache
            .as_ref()
            .and_then(|h| h.as_any().downcast_ref::<GenericKvCacheHandle>())
            .map(|g| {
                use ferrum_interfaces::KvCacheHandle;
                g.block_table().sequence_length
            })
            .unwrap_or(0);

        // Try the unified_forward path first when the caller can accept the
        // model's fast readback path. Requests that need logits processors or
        // token masks require full logits so the engine sampler can enforce
        // them; unified_forward currently has no per-item metadata channel.
        let force_full_logits = metadata_requires_full_logits(&input.metadata);
        let logits = {
            let mut model = self.lock_model();
            model.set_lora_adapter_for_cache(
                &cache_id,
                active_lora_from_metadata(&input.metadata)?,
            )?;
            if let Some(capacity_hint) = metadata_kv_capacity_hint(&input.metadata) {
                model.prepare_kv_capacity(&cache_id, capacity_hint);
            }
            if force_full_logits {
                model.prefill(&cache_id, &tokens)
            } else {
                let unified_item = vec![(cache_id.clone(), tokens.clone(), prior_seq_len, true)];
                match model.unified_forward(&unified_item) {
                    Ok(mut per_item) => per_item
                        .pop()
                        .flatten()
                        .ok_or_else(|| FerrumError::model("unified_forward returned no logits"))?,
                    Err(FerrumError::Unsupported { .. }) => model.prefill(&cache_id, &tokens),
                    Err(e) => return Err(e),
                }
            }
        };

        // Wrap logits as TensorRef: [1, 1, vocab_size]
        let logits_tensor = candle_core::Tensor::new(&logits[..], &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("logits tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("unsqueeze2: {e}")))?;
        let logits_ref = common::wrap_tensor(logits_tensor);

        let cfg = self.lock_model().config().clone();
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

    /// Batched prefill: combine all prompts into ONE `model.unified_forward`
    /// call so launch / kernel-overhead is amortized across the cohort.
    ///
    /// Falls back to the trait default (serial per-item) when the model
    /// returns `Err(unsupported)` from `unified_forward` — e.g. Qwen3MoeModel
    /// today, until Phase 2 adds its native unified path.
    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let force_full_logits = inputs
            .iter()
            .any(|input| metadata_requires_full_logits(&input.metadata));

        // Per-input: derive cache_id (reuse supplied handle's id or generate
        // fresh) + prior_seq_len. Mirrors the single-prefill path so chunked
        // prefill continuations route correctly when batched.
        let mut cache_ids = Vec::with_capacity(inputs.len());
        let mut prior_seq_lens = Vec::with_capacity(inputs.len());
        let mut tokens_per_input = Vec::with_capacity(inputs.len());
        let mut lora_per_input = Vec::with_capacity(inputs.len());
        for input in inputs {
            let tokens = common::tensor_to_tokens(&input.input_ids)?;
            let supplied_handle_id = input.kv_cache.as_ref().and_then(|h| {
                h.as_any()
                    .downcast_ref::<GenericKvCacheHandle>()
                    .map(|g| g.request_cache_id().to_string())
            });
            let cache_id = supplied_handle_id
                .clone()
                .unwrap_or_else(|| self.gen_cache_id());
            let prior_seq_len = input
                .kv_cache
                .as_ref()
                .and_then(|h| h.as_any().downcast_ref::<GenericKvCacheHandle>())
                .map(|g| {
                    use ferrum_interfaces::KvCacheHandle;
                    g.block_table().sequence_length
                })
                .unwrap_or(0);
            cache_ids.push(cache_id);
            prior_seq_lens.push(prior_seq_len);
            tokens_per_input.push(tokens);
            lora_per_input.push(active_lora_from_metadata(&input.metadata)?);
        }

        // Build unified items and ONE `unified_forward` call. If the model
        // doesn't support it, fall back to the trait-default serial path.
        let unified_items: Vec<(String, Vec<u32>, usize, bool)> = cache_ids
            .iter()
            .zip(tokens_per_input.iter())
            .zip(prior_seq_lens.iter())
            .map(|((cid, toks), &prior)| (cid.clone(), toks.clone(), prior, true))
            .collect();

        let nb_prof = llm_executor_runtime_env().batch_prefill_prof;
        let bp_t0 = if nb_prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut took_fallback = false;
        let per_item_logits: Vec<Vec<f32>> = {
            let mut model = self.lock_model();
            for ((cache_id, adapter), input) in cache_ids
                .iter()
                .zip(lora_per_input.iter())
                .zip(inputs.iter())
            {
                model.set_lora_adapter_for_cache(cache_id, adapter.clone())?;
                if let Some(capacity_hint) = metadata_kv_capacity_hint(&input.metadata) {
                    model.prepare_kv_capacity(cache_id, capacity_hint);
                }
            }
            if force_full_logits {
                took_fallback = true;
                let mut out = Vec::with_capacity(inputs.len());
                for (cid, toks) in cache_ids.iter().zip(tokens_per_input.iter()) {
                    out.push(model.prefill(cid, toks));
                }
                out
            } else {
                match model.unified_forward(&unified_items) {
                    Ok(per_item) => per_item
                        .into_iter()
                        .map(|opt| opt.expect("is_final_chunk=true must yield logits"))
                        .collect(),
                    Err(FerrumError::Unsupported { .. }) => {
                        took_fallback = true;
                        let mut out = Vec::with_capacity(inputs.len());
                        for (cid, toks) in cache_ids.iter().zip(tokens_per_input.iter()) {
                            out.push(model.prefill(cid, toks));
                        }
                        out
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        if let Some(t0) = bp_t0 {
            let total_q: usize = unified_items.iter().map(|it| it.1.len()).sum();
            eprintln!(
                "[batch-prefill] n_items={} total_q={} fallback={} elapsed={}us",
                inputs.len(),
                total_q,
                took_fallback,
                t0.elapsed().as_micros()
            );
        }

        let cfg = self.lock_model().config().clone();
        let mut outputs = Vec::with_capacity(inputs.len());
        for (i, logits) in per_item_logits.into_iter().enumerate() {
            let logits_tensor = candle_core::Tensor::new(&logits[..], &candle_core::Device::Cpu)
                .map_err(|e| FerrumError::model(format!("logits tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("unsqueeze2: {e}")))?;
            let logits_ref = common::wrap_tensor(logits_tensor);
            let seq_len = inputs[i]
                .kv_cache
                .as_ref()
                .and_then(|h| h.as_any().downcast_ref::<GenericKvCacheHandle>())
                .map(|g| {
                    use ferrum_interfaces::KvCacheHandle;
                    g.block_table().sequence_length + tokens_per_input[i].len()
                })
                .unwrap_or(tokens_per_input[i].len());
            let kv_handle = Arc::new(GenericKvCacheHandle::new(
                cfg.num_layers,
                cfg.num_kv_heads,
                cfg.head_dim,
                candle_core::Device::Cpu,
                seq_len,
                cache_ids[i].clone(),
            ));
            outputs.push(PrefillOutput::new(logits_ref, kv_handle));
        }
        Ok(outputs)
    }

    async fn truncate_kv(
        &self,
        kv_cache: &Arc<dyn ferrum_interfaces::KvCacheHandle>,
        new_len: usize,
    ) -> Result<()> {
        if let Some(g) = kv_cache.as_any().downcast_ref::<GenericKvCacheHandle>() {
            let cache_id = g.request_cache_id();
            self.lock_model().truncate_kv(cache_id, new_len);
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
            .ok_or_else(|| {
                FerrumError::model("forward_verify requires GenericKvCacheHandle input")
            })?
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
            let mut model = self.lock_model();
            model.set_lora_adapter_for_cache(
                &cache_id,
                active_lora_from_metadata(&inputs[0].metadata)?,
            )?;
            model.forward_verify(&cache_id, &token_ids)
        };

        let cfg = self.lock_model().config().clone();
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
            outputs.push(ferrum_interfaces::model_executor::DecodeOutput::new(
                logits_ref, handle,
            ));
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

        // Try unified_forward first unless the engine needs full logits for
        // masks/processors. The direct decode path returns vocabulary logits.
        let force_full_logits = metadata_requires_full_logits(&input.metadata);
        let logits = {
            let mut model = self.lock_model();
            model.set_lora_adapter_for_cache(
                &cache_id,
                active_lora_from_metadata(&input.metadata)?,
            )?;
            if force_full_logits {
                model.decode(&cache_id, token, seq_len as u32)
            } else {
                let unified_item = vec![(cache_id.clone(), vec![token], seq_len, true)];
                match model.unified_forward(&unified_item) {
                    Ok(mut per_item) => per_item
                        .pop()
                        .flatten()
                        .ok_or_else(|| FerrumError::model("unified_forward returned no logits"))?,
                    Err(FerrumError::Unsupported { .. }) => {
                        model.decode(&cache_id, token, seq_len as u32)
                    }
                    Err(e) => return Err(e),
                }
            }
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
        let prof = llm_executor_runtime_env().batch_decode_prof;
        let t0 = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Pre-extract all per-input metadata OUTSIDE the lock — this is pure
        // borrow/downcast work that doesn't touch the model.
        struct Prep {
            cache_id: String,
            token: u32,
            seq_len: u32,
            lora: Option<ActiveLoraAdapter>,
            requires_full_logits: bool,
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
                lora: active_lora_from_metadata(&input.metadata)?,
                requires_full_logits: metadata_requires_full_logits(&input.metadata),
                handle: Arc::new(input_handle.with_sequence_length((seq_len + 1) as usize)),
            });
        }
        let t_prep = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // One lock for the whole batch. Try unified_forward first: paged
        // configs route through the varlen kernel (single mixed dispatch
        // for the whole batch); contig configs fall back to model's
        // legacy decode_batch (separate paged_decode_attention call per
        // item, batched matmul for QKV/MLP).
        let (all_logits, t_lock_acq, t_model_call): (Vec<Vec<f32>>, _, _) = {
            let lock_t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let mut model = self.lock_model();
            let lock_acq = lock_t0.map(|t| t.elapsed());
            let model_t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            for p in &prepped {
                model.set_lora_adapter_for_cache(&p.cache_id, p.lora.clone())?;
            }
            let unified_items: Vec<(String, Vec<u32>, usize, bool)> = prepped
                .iter()
                .map(|p| (p.cache_id.clone(), vec![p.token], p.seq_len as usize, true))
                .collect();
            let tuples: Vec<(String, u32, u32)> = prepped
                .iter()
                .map(|p| (p.cache_id.clone(), p.token, p.seq_len))
                .collect();
            let force_full_logits = prepped.iter().any(|p| p.requires_full_logits);
            let logits = if force_full_logits {
                model.decode_batch_with_full_logits(&tuples, true)
            } else {
                match model.unified_forward(&unified_items) {
                    Ok(per_item) => {
                        if per_item.len() != prepped.len() {
                            return Err(FerrumError::model(format!(
                                "unified_forward returned {} entries for {} items",
                                per_item.len(),
                                prepped.len(),
                            )));
                        }
                        let mut out = Vec::with_capacity(prepped.len());
                        for (i, opt) in per_item.into_iter().enumerate() {
                            out.push(opt.ok_or_else(|| {
                                FerrumError::model(format!(
                                    "unified_forward returned None for decode item {i}"
                                ))
                            })?);
                        }
                        out
                    }
                    Err(FerrumError::Unsupported { .. }) => {
                        model.decode_batch_with_full_logits(&tuples, false)
                    }
                    Err(e) => return Err(e),
                }
            };
            let model_call = model_t0.map(|t| t.elapsed());
            (logits, lock_acq, model_call)
        };
        let t_model_done = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let m_count = prepped.len();
        let mut outputs = Vec::with_capacity(m_count);
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
        if let (Some(t0), Some(tp), Some(tm)) = (t0, t_prep, t_model_done) {
            static EX_PROF_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = EX_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n.is_multiple_of(8) {
                let total = t0.elapsed().as_micros();
                let prep = tp.duration_since(t0).as_micros();
                let lock_acq = t_lock_acq.map(|d| d.as_micros()).unwrap_or(0);
                let model_call = t_model_call.map(|d| d.as_micros()).unwrap_or(0);
                let model_block = tm.duration_since(tp).as_micros();
                let wrap = tm.elapsed().as_micros();
                eprintln!(
                    "[exec-batch-decode-prof] call#{} m={} total={}us prep={}us model_block={}us(lock_acq={}us model_call={}us) wrap={}us",
                    n, m_count, total, prep, model_block, lock_acq, model_call, wrap,
                );
            }
        }
        Ok(outputs)
    }

    /// Unified mixed-batch dispatch (chunked-prefill API).
    ///
    /// This impl is a behavior-preserving FALLBACK over the existing
    /// trait methods on `DecoderOnlyLLM`: prefill items go through
    /// `model.prefill(seq_id, &q_tokens)` (one at a time, mirroring the
    /// engine's current sequential prefill loop), decode items
    /// (`q_len == 1 && is_final_chunk`) are grouped into a single
    /// `model.decode_batch(...)` call. Net behavior is identical to the
    /// engine's pre-Phase-13 path; this just changes WHO orchestrates
    /// the prefill/decode split (caller → unified_decode) so the engine
    /// can converge on a single call.
    ///
    /// The real performance unlock comes in Step 5 when models override
    /// this with a true unified-forward (one [M_total, hidden] forward
    /// + varlen attention) — at that point the kernel-level mix replaces
    /// the host-side serial dispatch here.
    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        let mut results: Vec<Option<Vec<f32>>> = vec![None; batch.items.len()];
        if batch.items.is_empty() {
            return Ok(results);
        }

        // ── Real unified path (Step 5b+): if the model implements
        // `DecoderOnlyLLM::unified_forward`, route the entire batch
        // through one model forward (mixed prefill chunks + decode
        // tokens in a single [M_total, hidden] pass). The model returns
        // `Err(unsupported)` if it hasn't been wired yet — fall through
        // to the behaviour-preserving fallback below.
        let unified_items: Vec<(String, Vec<u32>, usize, bool)> = batch
            .items
            .iter()
            .map(|it| {
                (
                    it.seq_id.clone(),
                    it.q_tokens.clone(),
                    it.pos_offset,
                    it.is_final_chunk,
                )
            })
            .collect();
        let force_full_logits = batch
            .items
            .iter()
            .any(|item| metadata_requires_full_logits(&item.metadata));
        if !force_full_logits {
            let model_result = {
                let mut model = self.lock_model();
                for item in &batch.items {
                    model.set_lora_adapter_for_cache(
                        &item.seq_id,
                        active_lora_from_metadata(&item.metadata)?,
                    )?;
                    if item.pos_offset == 0 {
                        if let Some(capacity_hint) = metadata_kv_capacity_hint(&item.metadata) {
                            model.prepare_kv_capacity(&item.seq_id, capacity_hint);
                        }
                    }
                }
                model.unified_forward(&unified_items)
            };
            match model_result {
                Ok(per_item) => {
                    if per_item.len() != batch.items.len() {
                        return Err(FerrumError::model(format!(
                            "unified_forward returned {} entries for {} items",
                            per_item.len(),
                            batch.items.len(),
                        )));
                    }
                    return Ok(per_item);
                }
                Err(FerrumError::Unsupported { .. }) => {
                    // Fall through to the dispatch fallback below.
                }
                Err(e) => return Err(e),
            }
        }

        // Partition: pure decode items vs prefill chunks.
        // A "decode" item has q_len == 1 AND is_final_chunk == true.
        // Anything else (chunked prefill mid-stream OR a single-token
        // prefill that returns logits) goes through the per-item prefill
        // path so the model receives the right pos_offset behaviour.
        let mut prefill_indices: Vec<usize> = Vec::new();
        let mut decode_indices: Vec<usize> = Vec::new();
        for (i, item) in batch.items.iter().enumerate() {
            if item.q_tokens.len() == 1 && item.is_final_chunk {
                decode_indices.push(i);
            } else {
                prefill_indices.push(i);
            }
        }

        // Prefill items — sequential, mirrors current engine behaviour.
        // Held under a single model lock to amortise lock acquire across
        // all prefills in this batch (we may revisit per-call locking
        // when chunked-prefill becomes the perf-critical path).
        if !prefill_indices.is_empty() {
            let mut model = self.lock_model();
            for &i in &prefill_indices {
                let item = &batch.items[i];
                model.set_lora_adapter_for_cache(
                    &item.seq_id,
                    active_lora_from_metadata(&item.metadata)?,
                )?;
                if item.pos_offset == 0 {
                    if let Some(capacity_hint) = metadata_kv_capacity_hint(&item.metadata) {
                        model.prepare_kv_capacity(&item.seq_id, capacity_hint);
                    }
                }
                let logits = model.prefill(&item.seq_id, &item.q_tokens);
                if item.is_final_chunk {
                    results[i] = Some(logits);
                }
            }
        }

        // Decode items — single batched dispatch.
        if !decode_indices.is_empty() {
            let tuples: Vec<(String, u32, u32)> = decode_indices
                .iter()
                .map(|&i| {
                    let it = &batch.items[i];
                    (it.seq_id.clone(), it.q_tokens[0], it.pos_offset as u32)
                })
                .collect();
            let logits_vec = {
                let mut model = self.lock_model();
                for &i in &decode_indices {
                    let item = &batch.items[i];
                    model.set_lora_adapter_for_cache(
                        &item.seq_id,
                        active_lora_from_metadata(&item.metadata)?,
                    )?;
                }
                let policies: Vec<_> = decode_indices
                    .iter()
                    .map(|&i| batch.items[i].logits_policy.clone())
                    .collect();
                if policies.iter().any(
                    ferrum_interfaces::model_executor::LogitsReturnPolicy::requires_full_logits,
                ) {
                    model.decode_batch_with_full_logits(&tuples, true)
                } else {
                    model.decode_batch_with_logits_policy(&tuples, &policies)
                }
            };
            for (j, &i) in decode_indices.iter().enumerate() {
                results[i] = Some(logits_vec[j].clone());
            }
        }

        Ok(results)
    }

    fn release_cache(&self, cache_id: &str) {
        self.lock_model().release(cache_id);
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        let cfg = self.lock_model().config().clone();
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

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        let snapshot = self.lock_model().cache_metrics_snapshot()?;
        Some(self.attach_model_lock_metrics(snapshot))
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        self.lock_model().lora_metrics_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use ferrum_interfaces::model_executor::{DecodeInput, PrefillInput, UnifiedBatchItem};
    use ferrum_interfaces::KvCacheHandle;
    use ferrum_testkit::MockTensor;
    use ferrum_types::{Device, ModelId, ModelType};

    #[derive(Default)]
    struct RecordingCalls {
        unified_forward: usize,
        prefill: usize,
        decode: usize,
        decode_batch_force_full_logits: Vec<bool>,
        prepared_kv: Vec<(String, usize)>,
    }

    struct RecordingLlm {
        calls: Arc<Mutex<RecordingCalls>>,
        config: crate::common::LlmRuntimeConfig,
    }

    impl RecordingLlm {
        fn new(calls: Arc<Mutex<RecordingCalls>>) -> Self {
            Self {
                calls,
                config: crate::common::LlmRuntimeConfig {
                    hidden_size: 4,
                    num_layers: 1,
                    num_kv_heads: 1,
                    head_dim: 4,
                    vocab_size: 4,
                    max_seq_len: 16,
                },
            }
        }
    }

    impl DecoderOnlyLLM for RecordingLlm {
        fn config(&self) -> &crate::common::LlmRuntimeConfig {
            &self.config
        }

        fn prefill(&mut self, _cache_id: &str, _tokens: &[u32]) -> Vec<f32> {
            self.calls.lock().prefill += 1;
            vec![0.0, 1.0, 2.0, 3.0]
        }

        fn decode(&mut self, _cache_id: &str, _token: u32, _pos: u32) -> Vec<f32> {
            self.calls.lock().decode += 1;
            vec![3.0, 2.0, 1.0, 0.0]
        }

        fn decode_batch_with_full_logits(
            &mut self,
            batch: &[(String, u32, u32)],
            force_full_logits: bool,
        ) -> Vec<Vec<f32>> {
            self.calls
                .lock()
                .decode_batch_force_full_logits
                .push(force_full_logits);
            batch.iter().map(|_| vec![3.0, 2.0, 1.0, 0.0]).collect()
        }

        fn unified_forward(
            &mut self,
            items: &[(String, Vec<u32>, usize, bool)],
        ) -> std::result::Result<Vec<Option<Vec<f32>>>, FerrumError> {
            self.calls.lock().unified_forward += 1;
            Ok(items
                .iter()
                .map(|(_, _, _, is_final_chunk)| is_final_chunk.then_some(vec![99.0]))
                .collect())
        }

        fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
            self.calls
                .lock()
                .prepared_kv
                .push((cache_id.to_string(), capacity_hint));
        }

        fn release(&mut self, _cache_id: &str) {}

        fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
            Some(serde_json::json!({
                "position": "recording-test-cache",
            }))
        }
    }

    fn test_model_info() -> ModelInfo {
        ModelInfo {
            model_id: ModelId("recording".to_string()),
            model_type: ModelType::Custom("recording".to_string()),
            num_parameters: 0,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            vocab_size: 4,
            max_sequence_length: 16,
            dtype: DataType::FP32,
            device: Device::CPU,
            version: None,
            license: None,
            metadata: HashMap::new(),
        }
    }

    fn recording_executor(calls: Arc<Mutex<RecordingCalls>>) -> LlmExecutor {
        LlmExecutor::new(Box::new(RecordingLlm::new(calls)), test_model_info())
    }

    fn full_logits_metadata() -> HashMap<String, serde_json::Value> {
        HashMap::from([(
            "ferrum_require_full_logits".to_string(),
            serde_json::json!(true),
        )])
    }

    fn kv_capacity_hint_metadata(capacity_hint: usize) -> HashMap<String, serde_json::Value> {
        HashMap::from([(
            "ferrum_kv_capacity_hint".to_string(),
            serde_json::json!(capacity_hint),
        )])
    }

    fn test_kv_handle(cache_id: &str, seq_len: usize) -> Arc<dyn KvCacheHandle> {
        Arc::new(GenericKvCacheHandle::new(
            1,
            1,
            4,
            candle_core::Device::Cpu,
            seq_len,
            cache_id.to_string(),
        ))
    }

    #[test]
    fn llm_executor_runtime_env_parses_profile_flags_by_presence() {
        let env = LlmExecutorRuntimeEnv::from_env_vars([
            ("FERRUM_BATCH_PREFILL_PROF", ""),
            ("FERRUM_BATCH_DECODE_PROF", "0"),
        ]);

        assert!(env.batch_prefill_prof);
        assert!(env.batch_decode_prof);
    }

    #[test]
    fn llm_executor_runtime_env_defaults_profile_flags_off() {
        let env = LlmExecutorRuntimeEnv::from_env_vars([("UNRELATED", "1")]);

        assert!(!env.batch_prefill_prof);
        assert!(!env.batch_decode_prof);
    }

    #[test]
    fn prefill_skips_unified_forward_when_full_logits_required() {
        let calls = Arc::new(Mutex::new(RecordingCalls::default()));
        let executor = recording_executor(calls.clone());
        let input = PrefillInput::new(MockTensor::from_u32(&[1, 2], &[2]).into_ref())
            .with_metadata(full_logits_metadata());

        let output = tokio_test::block_on(executor.prefill(&input)).unwrap();

        assert_eq!(
            output
                .last_token_logits()
                .unwrap()
                .to_vec_f32()
                .unwrap()
                .len(),
            4
        );
        let calls = calls.lock();
        assert_eq!(calls.unified_forward, 0);
        assert_eq!(calls.prefill, 1);
    }

    #[test]
    fn decode_skips_unified_forward_when_full_logits_required() {
        let calls = Arc::new(Mutex::new(RecordingCalls::default()));
        let executor = recording_executor(calls.clone());
        let input = DecodeInput::new(
            MockTensor::from_u32(&[7], &[1]).into_ref(),
            test_kv_handle("decode-cache", 3),
        )
        .with_metadata(full_logits_metadata());

        let output = tokio_test::block_on(executor.decode(&input)).unwrap();

        assert_eq!(output.logits.to_vec_f32().unwrap().len(), 4);
        let calls = calls.lock();
        assert_eq!(calls.unified_forward, 0);
        assert_eq!(calls.decode, 1);
    }

    #[test]
    fn unified_decode_skips_unified_forward_when_full_logits_required() {
        let calls = Arc::new(Mutex::new(RecordingCalls::default()));
        let executor = recording_executor(calls.clone());
        let mut batch = UnifiedBatch::new();
        batch.items.push(UnifiedBatchItem {
            seq_id: "decode-cache".to_string(),
            q_tokens: vec![7],
            kv_cache: test_kv_handle("decode-cache", 3),
            pos_offset: 3,
            is_final_chunk: true,
            metadata: full_logits_metadata(),
            logits_policy: Default::default(),
        });

        let output = tokio_test::block_on(executor.unified_decode(&batch)).unwrap();

        assert_eq!(output[0].as_ref().unwrap().len(), 4);
        let calls = calls.lock();
        assert_eq!(calls.unified_forward, 0);
        assert_eq!(calls.decode_batch_force_full_logits, vec![true]);
    }

    #[test]
    fn unified_decode_prepares_fresh_prefill_kv_capacity_hint() {
        let calls = Arc::new(Mutex::new(RecordingCalls::default()));
        let executor = recording_executor(calls.clone());
        let mut batch = UnifiedBatch::new();
        batch.items.push(UnifiedBatchItem {
            seq_id: "prefill-cache".to_string(),
            q_tokens: vec![1, 2, 3],
            kv_cache: test_kv_handle("prefill-cache", 0),
            pos_offset: 0,
            is_final_chunk: true,
            metadata: kv_capacity_hint_metadata(7),
            logits_policy: Default::default(),
        });
        batch.items.push(UnifiedBatchItem {
            seq_id: "decode-cache".to_string(),
            q_tokens: vec![7],
            kv_cache: test_kv_handle("decode-cache", 3),
            pos_offset: 3,
            is_final_chunk: true,
            metadata: kv_capacity_hint_metadata(9),
            logits_policy: Default::default(),
        });

        let output = tokio_test::block_on(executor.unified_decode(&batch)).unwrap();

        assert_eq!(output.len(), 2);
        let calls = calls.lock();
        assert_eq!(calls.unified_forward, 1);
        assert_eq!(calls.prepared_kv, vec![("prefill-cache".to_string(), 7)]);
    }

    #[test]
    fn unified_decode_full_logits_prefill_prepares_kv_capacity_hint() {
        let calls = Arc::new(Mutex::new(RecordingCalls::default()));
        let executor = recording_executor(calls.clone());
        let mut metadata = full_logits_metadata();
        metadata.insert("ferrum_kv_capacity_hint".to_string(), serde_json::json!(11));
        let mut batch = UnifiedBatch::new();
        batch.items.push(UnifiedBatchItem {
            seq_id: "prefill-cache".to_string(),
            q_tokens: vec![1, 2, 3],
            kv_cache: test_kv_handle("prefill-cache", 0),
            pos_offset: 0,
            is_final_chunk: true,
            metadata,
            logits_policy: Default::default(),
        });

        let output = tokio_test::block_on(executor.unified_decode(&batch)).unwrap();

        assert_eq!(output[0].as_ref().unwrap().len(), 4);
        let calls = calls.lock();
        assert_eq!(calls.unified_forward, 0);
        assert_eq!(calls.prefill, 1);
        assert_eq!(calls.prepared_kv, vec![("prefill-cache".to_string(), 11)]);
    }

    #[test]
    fn cache_metrics_snapshot_includes_model_lock_wait_metrics() {
        let calls = Arc::new(Mutex::new(RecordingCalls::default()));
        let executor = recording_executor(calls);

        assert_eq!(executor.kv_capacity(), Some(16));
        let metrics = executor.cache_metrics_snapshot().unwrap();

        assert_eq!(metrics["position"], "recording-test-cache");
        assert_eq!(metrics["executor_model_lock"]["schema_version"], 1);
        assert!(
            metrics["executor_model_lock"]["samples"].as_u64().unwrap() >= 2,
            "metrics: {metrics}"
        );
        assert!(
            metrics["executor_model_lock"]["total_wait_time_us"]
                .as_u64()
                .is_some(),
            "metrics: {metrics}"
        );
        assert!(
            metrics["executor_model_lock"]["avg_wait_time_ms"].is_number(),
            "metrics: {metrics}"
        );
    }
}
