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
        MemoryRequirements, PrefillInput, PrefillOutput, UnifiedBatch,
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

        // Try the unified_forward path first: when the model has it wired
        // (paged KV pools allocated), this routes through the chunked-prefill
        // varlen kernel — the same code that handles mixed prefill+decode
        // batches. On Unsupported, fall back to the legacy single-item
        // prefill path so contig-KV configs keep their existing behaviour.
        let logits = {
            let mut model = self.model.lock();
            let unified_item = vec![(cache_id.clone(), tokens.clone(), prior_seq_len, true)];
            match model.unified_forward(&unified_item) {
                Ok(mut per_item) => per_item
                    .pop()
                    .flatten()
                    .ok_or_else(|| FerrumError::model("unified_forward returned no logits"))?,
                Err(FerrumError::Unsupported { .. }) => model.prefill(&cache_id, &tokens),
                Err(e) => return Err(e),
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

        // Try unified_forward first so paged-KV configs route the single
        // decode through the same varlen kernel used by batched mixed
        // batches. Falls back to legacy paged_decode_attention for contig
        // configs that haven't wired unified_forward.
        let logits = {
            let mut model = self.model.lock();
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
        let prof = std::env::var("FERRUM_BATCH_DECODE_PROF").is_ok();
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
            let mut model = self.model.lock();
            let lock_acq = lock_t0.map(|t| t.elapsed());
            let model_t0 = if prof {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let unified_items: Vec<(String, Vec<u32>, usize, bool)> = prepped
                .iter()
                .map(|p| (p.cache_id.clone(), vec![p.token], p.seq_len as usize, true))
                .collect();
            let logits = match model.unified_forward(&unified_items) {
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
                    let tuples: Vec<(String, u32, u32)> = prepped
                        .iter()
                        .map(|p| (p.cache_id.clone(), p.token, p.seq_len))
                        .collect();
                    model.decode_batch(&tuples)
                }
                Err(e) => return Err(e),
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
        let model_result = {
            let mut model = self.model.lock();
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
            let mut model = self.model.lock();
            for &i in &prefill_indices {
                let item = &batch.items[i];
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
                let mut model = self.model.lock();
                model.decode_batch(&tuples)
            };
            for (j, &i) in decode_indices.iter().enumerate() {
                results[i] = Some(logits_vec[j].clone());
            }
        }

        Ok(results)
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
