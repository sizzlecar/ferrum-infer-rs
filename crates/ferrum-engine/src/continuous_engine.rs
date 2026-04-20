//! Continuous Batching Engine
//!
//! Iteration-level continuous batching: each step processes a mixed batch of
//! prefill and decode requests selected by the scheduler.  Multiple callers
//! can submit requests concurrently — an `iteration_lock` serializes the
//! actual engine steps so each batch is processed exactly once.

use async_trait::async_trait;
use ferrum_interfaces::{
    engine::InferenceEngine, kv_cache::AllocationRequest, KvCacheHandle, KvCacheManager,
    ModelExecutor, Sampler, SchedulerInterface as Scheduler, TensorFactory, TensorRef, Tokenizer,
};
use ferrum_kv::cache::prefix::PrefixCache;
use ferrum_sampler::json_mode::JsonModeProcessor;
use ferrum_scheduler::implementations::{ContinuousBatchScheduler, RequestPhase};
use ferrum_types::{
    DataType, Device, EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest,
    InferenceResponse, Priority, RequestId, Result, SamplingParams, StreamChunk, TokenId,
    TokenUsage,
};
use futures::stream::Stream;
use metrics::{counter, gauge, histogram};
use parking_lot::RwLock;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

// ────────────────────────────────────────────────────────────────────────────
// Sequence state
// ────────────────────────────────────────────────────────────────────────────

/// State of a running sequence in the continuous batch.
#[derive(Debug)]
pub struct SequenceState {
    pub request_id: RequestId,
    /// Original request — kept for re-submission after preemption.
    pub original_request: InferenceRequest,
    pub input_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    pub kv_cache: Option<Arc<dyn KvCacheHandle>>,
    pub sampling_params: SamplingParams,
    pub phase: RequestPhase,
    pub rng: StdRng,
    pub prefill_complete: bool,
    pub stream_sender: Option<mpsc::Sender<Result<StreamChunk>>>,
    pub response_sender: Option<tokio::sync::oneshot::Sender<InferenceResponse>>,
    pub start_time: Instant,
    pub tokens_this_iteration: usize,
    /// Number of times this request has been preempted.
    pub preemption_count: usize,
    /// JSON mode logits processor (active when response_format is JsonObject).
    pub json_processor: Option<Arc<JsonModeProcessor>>,
    /// Regex-guided hard-mask processor (active when response_format is Regex).
    pub regex_processor: Option<Arc<ferrum_sampler::guided::RegexGuidedProcessor>>,
    /// Draft-model KV cache (only populated when engine has speculative
    /// decoding enabled). Allocated + prefilled lazily on the first decode.
    pub draft_kv_cache: Option<Arc<dyn KvCacheHandle>>,
    /// Token frequency counts for repetition penalty.
    pub token_frequencies: HashMap<TokenId, usize>,
    /// Model executor's KV cache key for this sequence (for cleanup on completion).
    pub model_cache_id: Option<String>,
}

impl SequenceState {
    pub fn new(request: InferenceRequest, input_tokens: Vec<TokenId>) -> Self {
        Self::new_with_tokenizer(request, input_tokens, None)
    }

    /// Build sequence state, optionally wiring a tokenizer for guided-decoding
    /// processors (`ResponseFormat::Regex` needs vocab access to compile a
    /// token-level mask). Falling back to `None` preserves the old behaviour
    /// for call sites that don't have a tokenizer to hand (smoke tests).
    pub fn new_with_tokenizer(
        request: InferenceRequest,
        input_tokens: Vec<TokenId>,
        tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
    ) -> Self {
        use ferrum_types::ResponseFormat;
        let seed = request.sampling_params.seed.unwrap_or(42);
        let json_processor = match &request.sampling_params.response_format {
            ResponseFormat::JsonObject => Some(Arc::new(JsonModeProcessor::new())),
            _ => None,
        };
        let regex_processor = match (&request.sampling_params.response_format, &tokenizer) {
            (ResponseFormat::JsonSchema(schema), Some(tok)) => {
                // OpenAI-style structured output: compile the JSON Schema
                // to a regex then drive the DFA-guided processor with it.
                match ferrum_sampler::schema_to_regex::schema_to_regex(schema) {
                    Ok(pattern) => {
                        let eos = tok.special_tokens().eos_token;
                        match ferrum_sampler::guided::RegexGuidedProcessor::new(
                            &pattern,
                            tok.clone(),
                            eos,
                        ) {
                            Ok(p) => Some(Arc::new(p)),
                            Err(e) => {
                                tracing::warn!(
                                    "json_schema guided decode disabled (regex build): {e}"
                                );
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "json_schema guided decode disabled (schema translation): {e}"
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        Self {
            request_id: request.id.clone(),
            original_request: request.clone(),
            input_tokens,
            generated_tokens: Vec::new(),
            kv_cache: None,
            sampling_params: request.sampling_params,
            phase: RequestPhase::Waiting,
            rng: StdRng::seed_from_u64(seed),
            prefill_complete: false,
            stream_sender: None,
            response_sender: None,
            start_time: Instant::now(),
            tokens_this_iteration: 0,
            preemption_count: 0,
            json_processor,
            regex_processor,
            draft_kv_cache: None,
            token_frequencies: HashMap::new(),
            model_cache_id: None,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub fn should_stop(&self, vocab_size: usize) -> bool {
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return true;
        }
        if let Some(&last_token) = self.generated_tokens.last() {
            if last_token.get() >= (vocab_size - 10) as u32 {
                return true;
            }
        }
        false
    }

    /// Sample next token with full processor chain (temperature, top-k/p,
    /// repetition penalty, JSON mode, regex-guided mask).
    pub fn sample_with_processors(&mut self, logits: &mut [f32]) -> Result<TokenId> {
        use ferrum_interfaces::sampler::{SamplingConfig, SamplingContext};

        // Regex-guided mask runs FIRST: it's a hard constraint (sets invalid
        // tokens to -inf). Subsequent temperature / top-k / top-p stay
        // correct because -inf tokens can't make it through any softmax.
        if let Some(ref rp) = self.regex_processor {
            rp.advance_with_tokens_public(&self.generated_tokens);
            rp.mask_logits(logits);
        }

        // Apply JSON mode biases before the standard processor chain
        if let Some(ref jp) = self.json_processor {
            let generated: String = self
                .generated_tokens
                .iter()
                .filter_map(|t| {
                    let v = t.get();
                    if v < 128 {
                        Some(v as u8 as char)
                    } else {
                        None
                    }
                })
                .collect();
            jp.apply_biases(logits, &generated);
        }

        // Build SamplingConfig from this request's params (includes temperature, top-k/p, repetition penalty)
        let config = SamplingConfig::from_params(&self.sampling_params);
        let step = self.generated_tokens.len();
        let vocab_size = logits.len();
        let ctx = SamplingContext::new(
            step,
            &self.sampling_params,
            logits,
            &self.generated_tokens,
            &self.token_frequencies,
            vocab_size,
        );
        let token = config.sample(ctx, &mut self.rng)?;

        // Update frequency tracking
        *self.token_frequencies.entry(token).or_insert(0) += 1;

        Ok(token)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Engine inner – shared via Arc so we can spawn tasks
// ────────────────────────────────────────────────────────────────────────────

struct EngineInner {
    config: EngineConfig,
    scheduler: Arc<ContinuousBatchScheduler>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    #[allow(dead_code)]
    // Retained for constructor API; sampling now uses per-request SamplingConfig
    sampler: Arc<dyn Sampler + Send + Sync>,
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    /// Optional draft executor for speculative decoding. When set alongside
    /// `spec_config`, `run_single_decode` routes through `SpeculativeRunner`.
    draft_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
    /// Speculative decoding parameters (N, temperature). `None` = disabled.
    spec_config: Option<crate::speculative::SpeculativeDecodingConfig>,
    tensor_factory: Arc<dyn TensorFactory>,
    sequences: RwLock<HashMap<RequestId, SequenceState>>,
    is_running: AtomicBool,
    shutdown_notify: Arc<Notify>,
    /// Ensures only one iteration step runs at a time.
    iteration_lock: tokio::sync::Mutex<()>,
    /// Wakes callers or a background loop when new work is submitted.
    work_notify: Notify,
    /// Prefix cache: shares KV blocks across requests with common prompts.
    prefix_cache: PrefixCache,
    // stats
    iteration_count: AtomicU64,
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
    total_preemptions: AtomicU64,
    prefix_cache_hits: AtomicU64,
}

impl EngineInner {
    // ── tensor helper ──────────────────────────────────────────────────

    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<TensorRef> {
        let f32_data: Vec<f32> = token_ids.iter().map(|&v| v as f32).collect();
        let len = f32_data.len();
        self.tensor_factory
            .from_slice(&f32_data, &[1, len], DataType::FP32, Device::CPU)
    }

    /// Rebuild a KvCacheHandle with a corrected sequence_length. Used by
    /// speculative-decode rollback — covers the `GenericKvCacheHandle`
    /// that our LLM executors return; for any other handle type we just
    /// return the original clone (stub / mock executors don't care).
    fn make_kv_handle_with_seq(
        &self,
        h: &std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle>,
        new_seq: usize,
    ) -> std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle> {
        if let Some(g) = h
            .as_any()
            .downcast_ref::<ferrum_models::executor::common::GenericKvCacheHandle>()
        {
            std::sync::Arc::new(g.with_sequence_length(new_seq))
        } else {
            h.clone()
        }
    }

    // ── iteration loop ─────────────────────────────────────────────────

    /// Drive iterations until the given request is removed from `sequences`.
    async fn drive_to_completion(&self, request_id: &RequestId) -> Result<()> {
        loop {
            {
                let _guard = self.iteration_lock.lock().await;
                self.run_iteration().await?;
            }
            if !self.sequences.read().contains_key(request_id) {
                return Ok(());
            }
            // Yield to let other tasks progress between iterations.
            tokio::task::yield_now().await;
        }
    }

    /// Run one iteration: ask the scheduler for a batch, then process it.
    async fn run_iteration(&self) -> Result<()> {
        let iteration = self.iteration_count.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.iterations_total").increment(1);

        let hint = ferrum_interfaces::BatchHint {
            max_batch_size: self.config.batching.max_batch_size,
            max_tokens: self.config.batching.max_batch_size * 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: ferrum_interfaces::scheduler::ResourceConstraints::default(),
        };

        let batch = match self.scheduler.next_batch(hint).await {
            Some(b) => b,
            None => {
                tokio::time::sleep(Duration::from_millis(1)).await;
                return Ok(());
            }
        };

        debug!(
            "Iteration {}: batch with {} requests",
            iteration,
            batch.size()
        );

        self.process_batch(&batch).await
    }

    // ── batch processing ───────────────────────────────────────────────

    async fn process_batch(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        let mut prefill_ids = Vec::new();
        let mut decode_ids = Vec::new();

        {
            let mut sequences = self.sequences.write();
            for scheduled_req in &batch.requests {
                let rid = &scheduled_req.request.id;
                let seq = sequences.entry(rid.clone()).or_insert_with(|| {
                    let input_tokens = self
                        .tokenizer
                        .encode(&scheduled_req.request.prompt, true)
                        .unwrap_or_else(|_| vec![TokenId::new(0)]);
                    SequenceState::new_with_tokenizer(
                        scheduled_req.request.clone(),
                        input_tokens,
                        Some(self.tokenizer.clone()),
                    )
                });

                if !seq.prefill_complete {
                    prefill_ids.push(rid.clone());
                } else {
                    decode_ids.push(rid.clone());
                }
            }
        }

        // Prefill new requests
        for rid in &prefill_ids {
            if let Err(e) = self.run_prefill(rid).await {
                warn!("Prefill failed for {}: {}", rid, e);
                self.complete_request(rid, FinishReason::Error).await?;
            }
        }

        // Decode continuing requests (batch when possible)
        if decode_ids.len() > 1 {
            if let Err(e) = self.run_batch_decode(&decode_ids).await {
                warn!("Batch decode failed, falling back to per-request: {}", e);
                for rid in &decode_ids {
                    if let Err(e) = self.run_decode_step(rid).await {
                        warn!("Decode failed for {}: {}", rid, e);
                        self.complete_request(rid, FinishReason::Error).await?;
                    }
                }
            }
        } else {
            for rid in &decode_ids {
                if let Err(e) = self.run_decode_step(rid).await {
                    warn!("Decode failed for {}: {}", rid, e);
                    self.complete_request(rid, FinishReason::Error).await?;
                }
            }
        }

        Ok(())
    }

    // ── preemption ──────────────────────────────────────────────────────

    /// Try to preempt a decoding request to free KV cache blocks.
    ///
    /// Picks the lowest-priority victim (ties broken by fewest generated
    /// tokens — least work lost).  Frees the victim's KV cache, resets
    /// its sequence state, and re-submits it to the scheduler so it will
    /// be re-prefilled in a later iteration.
    ///
    /// Returns `true` if a victim was preempted.
    async fn preempt_victim(&self, exclude_id: &RequestId) -> bool {
        // Select victim: any decoding sequence except the requester
        let victim_id = {
            let sequences = self.sequences.read();
            sequences
                .iter()
                .filter(|(id, s)| *id != exclude_id && s.prefill_complete && s.kv_cache.is_some())
                .min_by(|(_, a), (_, b)| {
                    // Lowest priority first, then fewest generated tokens
                    a.sampling_params
                        .max_tokens // proxy for priority (TODO: use real priority)
                        .cmp(&b.sampling_params.max_tokens)
                        .then_with(|| a.generated_tokens.len().cmp(&b.generated_tokens.len()))
                })
                .map(|(id, _)| id.clone())
        };

        let victim_id = match victim_id {
            Some(id) => id,
            None => return false,
        };

        info!("Preempting request {} to free KV blocks", victim_id);

        // Free model executor's KV cache for this sequence
        {
            let sequences = self.sequences.read();
            if let Some(seq) = sequences.get(&victim_id) {
                if let Some(ref cache_id) = seq.model_cache_id {
                    self.model_executor.release_cache(cache_id);
                }
            }
        }

        // Free KV cache manager blocks
        let _ = self.kv_cache.deallocate(victim_id.clone()).await;

        // Reset sequence state — keep response/stream channels intact
        {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(&victim_id) {
                seq.kv_cache = None;
                seq.model_cache_id = None;
                seq.generated_tokens.clear();
                seq.prefill_complete = false;
                seq.phase = RequestPhase::Waiting;
                seq.tokens_this_iteration = 0;
                seq.preemption_count += 1;
                // Reset RNG to original seed for deterministic re-generation
                let seed = seq.sampling_params.seed.unwrap_or(42);
                seq.rng = StdRng::seed_from_u64(seed);
            }
        }

        // Cancel in scheduler and re-submit so it goes back to waiting queue
        let _ = self.scheduler.cancel(victim_id.clone()).await;
        let request = {
            let sequences = self.sequences.read();
            sequences
                .get(&victim_id)
                .map(|s| s.original_request.clone())
        };
        if let Some(req) = request {
            let _ = self.scheduler.submit(req).await;
        }

        self.total_preemptions.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.preemptions_total").increment(1);
        true
    }

    // ── prefill ────────────────────────────────────────────────────────

    async fn run_prefill(&self, request_id: &RequestId) -> Result<()> {
        let (input_tokens_clone, num_tokens) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            (seq.input_tokens.clone(), seq.input_tokens.len())
        };

        // ── Check prefix cache ──────────────────────────────────────────
        // Exact-match only: on hit, skip executor prefill entirely by cloning
        // the cached KV handle and sampling from the stored last-token logits.
        // Partial matches (stored prefix is a proper prefix of input) fall
        // through to full prefill — supporting them needs incremental prefill
        // on top of a cloned KV, not yet exposed by the executor contract.
        //
        // CUDA default: OFF. Historical candle CUDA decode runner released
        // the KV on completion, invalidating clones ("No candle KV cache to
        // export"). The Phase E Model-as-Code CUDA path may be safe, but
        // end-to-end engine + prefix-cache is not yet validated on GPU.
        // Toggle via env `FERRUM_PREFIX_CACHE=1` to opt in on CUDA; CPU/Metal
        // defaults ON (validated).
        let skip_prefix_cache = if cfg!(feature = "cuda") {
            std::env::var("FERRUM_PREFIX_CACHE").map_or(true, |v| v != "1")
        } else {
            std::env::var("FERRUM_PREFIX_CACHE").map_or(false, |v| v == "0")
        };
        if !skip_prefix_cache {
            let hit = self
                .prefix_cache
                .find_prefix(&input_tokens_clone)
                .filter(|(prefix_id, _, _)| prefix_id.len() == input_tokens_clone.len());
            if let Some((_prefix_id, cached_kv, cached_logits)) = hit {
                debug!(
                    "Prefix cache hit for {}: reusing {} cached tokens",
                    request_id, num_tokens,
                );

                let cloned_kv = cached_kv.clone_handle()?;

                let first_token = {
                    let mut sequences = self.sequences.write();
                    let seq = sequences
                        .get_mut(request_id)
                        .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                    if let Some(ref jp) = seq.json_processor {
                        jp.reset();
                    }
                    let mut logits = cached_logits;
                    let token = seq.sample_with_processors(&mut logits)?;
                    seq.generated_tokens.push(token);
                    seq.model_cache_id = Some(cloned_kv.cache_id());
                    seq.kv_cache = Some(cloned_kv);
                    seq.prefill_complete = true;
                    seq.phase = RequestPhase::Decoding;
                    token
                };

                self.scheduler.mark_prefill_complete(request_id, num_tokens);
                self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                counter!("ferrum.engine.prefix_cache_hits").increment(1);

                debug!(
                    "Prefix cache prefill for {}: first generated: {}",
                    request_id,
                    first_token.get()
                );

                self.send_stream_update(request_id, first_token).await;

                let should_stop = {
                    let sequences = self.sequences.read();
                    sequences.get(request_id).map_or(true, |s| {
                        s.should_stop(self.model_executor.info().vocab_size)
                    })
                };
                if should_stop {
                    self.complete_request(request_id, FinishReason::EOS).await?;
                }

                return Ok(());
            }
        } // skip_prefix_cache

        // ── Cache miss (or prefix cache skipped) — full prefill ─────────
        let model_info = self.model_executor.info();
        let alloc_request = AllocationRequest {
            request_id: request_id.clone(),
            initial_tokens: num_tokens,
            max_sequence_length: model_info.max_sequence_length,
            num_layers: model_info.num_layers,
            num_heads: model_info.num_kv_heads,
            head_dim: model_info.hidden_size / model_info.num_heads.max(1),
            device: self.config.backend.device.clone(),
            dtype: model_info.dtype,
            priority: Priority::Normal,
        };

        // Try allocation, preempting if necessary
        let kv_handle = match self.kv_cache.allocate(&alloc_request).await {
            Ok(h) => h,
            Err(_) => {
                // OOM — try to free blocks by preempting a victim
                if self.preempt_victim(request_id).await {
                    // Retry after preemption
                    self.kv_cache.allocate(&alloc_request).await?
                } else {
                    return Err(FerrumError::resource_exhausted(
                        "No blocks available and no request to preempt",
                    ));
                }
            }
        };

        // Opt-in chunked prefill: `FERRUM_CHUNKED_PREFILL=<chunk_size>` splits
        // the prompt into sequential chunks and runs `prefill` per chunk.
        // Reduces peak activation memory for long prompts; also informs the
        // scheduler so its metrics reflect actual progress. True cross-
        // iteration interleaving with decode is a follow-up refactor.
        let chunk_size = std::env::var("FERRUM_CHUNKED_PREFILL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0 && n < num_tokens);

        let prefill_output = if let Some(csz) = chunk_size {
            let mut current_kv = kv_handle;
            let mut final_output: Option<ferrum_interfaces::model_executor::PrefillOutput> = None;
            let mut processed = 0usize;
            while processed < num_tokens {
                let end = (processed + csz).min(num_tokens);
                let chunk_ids: Vec<u32> = input_tokens_clone[processed..end]
                    .iter()
                    .map(|t| t.get())
                    .collect();
                let chunk_tensor = self.tokens_to_tensor(&chunk_ids)?;
                let input = ferrum_interfaces::model_executor::PrefillInput::new(chunk_tensor)
                    .with_kv_cache(current_kv.clone());
                let out = self.model_executor.prefill(&input).await?;
                current_kv = out.kv_cache.clone();

                self.scheduler
                    .mark_prefill_chunk_processed(request_id, num_tokens, end - processed);

                processed = end;
                if processed >= num_tokens {
                    final_output = Some(out);
                }
            }
            final_output.expect("at least one chunk must run")
        } else {
            let input_tensor = {
                let token_u32s: Vec<u32> = input_tokens_clone.iter().map(|t| t.get()).collect();
                self.tokens_to_tensor(&token_u32s)?
            };
            let prefill_input = ferrum_interfaces::model_executor::PrefillInput::new(input_tensor)
                .with_kv_cache(kv_handle);
            self.model_executor.prefill(&prefill_input).await?
        };

        let last_logits = prefill_output.last_token_logits()?;
        let logits_vec = last_logits.to_vec_f32()?;

        // Store in prefix cache for future reuse
        let _ = self.prefix_cache.store_prefix(
            &input_tokens_clone,
            prefill_output.kv_cache.clone(),
            logits_vec.clone(),
        );

        let first_token = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            if let Some(ref jp) = seq.json_processor {
                jp.reset();
            }
            let mut logits = logits_vec;
            let token = seq.sample_with_processors(&mut logits)?;
            seq.generated_tokens.push(token);
            seq.model_cache_id = Some(prefill_output.kv_cache.cache_id());
            seq.kv_cache = Some(prefill_output.kv_cache.clone());
            seq.prefill_complete = true;
            seq.phase = RequestPhase::Decoding;
            token
        };

        self.scheduler.mark_prefill_complete(request_id, num_tokens);
        self.total_prefill_tokens
            .fetch_add(num_tokens as u64, Ordering::Relaxed);
        counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
        counter!("ferrum.engine.prefills_total").increment(1);

        debug!(
            "Prefill complete for {}: {} prompt tokens, first generated: {}",
            request_id,
            num_tokens,
            first_token.get()
        );

        self.send_stream_update(request_id, first_token).await;

        let should_stop = {
            let sequences = self.sequences.read();
            sequences.get(request_id).map_or(true, |s| {
                s.should_stop(self.model_executor.info().vocab_size)
            })
        };
        if should_stop {
            self.complete_request(request_id, FinishReason::EOS).await?;
        }

        Ok(())
    }

    // ── batch decode ──────────────────────────────────────────────────

    /// Run batch decode for multiple requests in a single forward pass.
    async fn run_batch_decode(&self, request_ids: &[RequestId]) -> Result<()> {
        // Build DecodeInput for each request
        let mut decode_inputs = Vec::with_capacity(request_ids.len());
        let rids: Vec<RequestId> = request_ids.to_vec();
        {
            let sequences = self.sequences.read();
            for rid in &rids {
                let seq = sequences
                    .get(rid)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                let kv_cache = seq
                    .kv_cache
                    .as_ref()
                    .ok_or_else(|| FerrumError::internal("No KV cache"))?
                    .clone();
                let last_token = seq
                    .generated_tokens
                    .last()
                    .copied()
                    .unwrap_or(TokenId::new(0));
                let tensor = self.tokens_to_tensor(&[last_token.get()])?;
                decode_inputs.push(ferrum_interfaces::model_executor::DecodeInput::new(
                    tensor, kv_cache,
                ));
            }
        }

        // Call batch_decode on the executor
        let decode_outputs = self.model_executor.batch_decode(&decode_inputs).await?;

        // Process each result: sample, update state, stream
        for (rid, decode_output) in rids.iter().zip(decode_outputs.iter()) {
            let logits_vec = decode_output.logits.to_vec_f32()?;

            let next_token = {
                let mut sequences = self.sequences.write();
                let seq = sequences
                    .get_mut(rid)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                let mut logits = logits_vec;
                let token = seq.sample_with_processors(&mut logits)?;
                seq.generated_tokens.push(token);
                seq.kv_cache = Some(decode_output.kv_cache.clone());
                seq.tokens_this_iteration += 1;
                token
            };

            let generated_count = {
                let sequences = self.sequences.read();
                sequences
                    .get(rid)
                    .map(|s| s.generated_tokens.len())
                    .unwrap_or(0)
            };
            self.scheduler.update_decode_progress(rid, generated_count);
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);

            self.send_stream_update(rid, next_token).await;

            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(rid).map_or(true, |s| {
                    s.should_stop(self.model_executor.info().vocab_size)
                })
            };
            if should_stop {
                let finish_reason = {
                    let sequences = self.sequences.read();
                    match sequences.get(rid) {
                        Some(seq)
                            if seq.generated_tokens.len() >= seq.sampling_params.max_tokens =>
                        {
                            FinishReason::Length
                        }
                        Some(_) => FinishReason::EOS,
                        None => FinishReason::Error,
                    }
                };
                self.complete_request(rid, finish_reason).await?;
            }
        }

        Ok(())
    }

    // ── decode step ────────────────────────────────────────────────────

    async fn run_decode_step(&self, request_id: &RequestId) -> Result<()> {
        // Speculative decoding path: when both a draft executor and
        // config are set, delegate to the runner and push the accepted
        // tokens onto the sequence in one shot.
        if self.draft_executor.is_some() && self.spec_config.is_some() {
            return self.run_decode_step_speculative(request_id).await;
        }

        let decode_input = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let kv_cache = seq
                .kv_cache
                .as_ref()
                .ok_or_else(|| FerrumError::internal("No KV cache"))?
                .clone();
            let last_token = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0));
            let tensor = self.tokens_to_tensor(&[last_token.get()])?;
            ferrum_interfaces::model_executor::DecodeInput::new(tensor, kv_cache)
        };

        let decode_output = self.model_executor.decode(&decode_input).await?;
        let logits_vec = decode_output.logits.to_vec_f32()?;

        let next_token = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let mut logits = logits_vec;
            let token = seq.sample_with_processors(&mut logits)?;
            seq.generated_tokens.push(token);
            seq.kv_cache = Some(decode_output.kv_cache.clone());
            seq.tokens_this_iteration += 1;
            token
        };

        let generated_count = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.generated_tokens.len())
                .unwrap_or(0)
        };
        self.scheduler
            .update_decode_progress(request_id, generated_count);
        self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.decode_tokens_total").increment(1);

        self.send_stream_update(request_id, next_token).await;

        let should_stop = {
            let sequences = self.sequences.read();
            sequences.get(request_id).map_or(true, |s| {
                s.should_stop(self.model_executor.info().vocab_size)
            })
        };

        if should_stop {
            let finish_reason = {
                let sequences = self.sequences.read();
                match sequences.get(request_id) {
                    Some(seq) if seq.generated_tokens.len() >= seq.sampling_params.max_tokens => {
                        FinishReason::Length
                    }
                    Some(_) => FinishReason::EOS,
                    None => FinishReason::Error,
                }
            };
            self.complete_request(request_id, finish_reason).await?;
        }

        Ok(())
    }

    /// Speculative-decoding variant of `run_decode_step`. Lazily prefills
    /// the draft model's KV cache on first call (same prompt as target).
    /// Then each iteration produces 1..=N+1 tokens via `SpeculativeRunner`.
    async fn run_decode_step_speculative(&self, request_id: &RequestId) -> Result<()> {
        use ferrum_interfaces::model_executor::PrefillInput;

        let (draft_exec, cfg_base) = match (&self.draft_executor, &self.spec_config) {
            (Some(d), Some(c)) => (d.clone(), c.clone()),
            _ => unreachable!("speculative gate checked in run_decode_step"),
        };

        // Use the caller's sampling temperature for accept/reject, not the
        // engine-default from spec_config. Otherwise a greedy request
        // (temperature=0) runs through a T=1 verifier and ULP-level fp32
        // noise between draft/target causes stochastic rejections → KV
        // misalignment → output drift.
        let per_request_temperature = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.sampling_params.temperature)
                .unwrap_or(cfg_base.temperature)
        };
        let cfg = crate::speculative::SpeculativeDecodingConfig {
            num_speculative_tokens: cfg_base.num_speculative_tokens,
            temperature: per_request_temperature,
        };

        // ── 1. Ensure draft KV is prefilled once with the full prompt ────
        let draft_kv_ready = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .and_then(|s| s.draft_kv_cache.clone())
        };
        let draft_kv = if let Some(kv) = draft_kv_ready {
            kv
        } else {
            // First spec iteration — prefill the draft on the prompt only.
            // The already-sampled `last_token` is passed into the runner in
            // step() below (as `last_token`) — the runner's first draft
            // decode consumes it, writing KV at position prompt_len exactly
            // as target's decode path does. DO NOT pre-consume it here.
            let prompt_u32s = {
                let sequences = self.sequences.read();
                let seq = sequences
                    .get(request_id)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                seq.input_tokens.iter().map(|t| t.get()).collect::<Vec<_>>()
            };
            let model_info = draft_exec.info();
            let alloc_request = AllocationRequest {
                request_id: request_id.clone(),
                initial_tokens: prompt_u32s.len(),
                max_sequence_length: model_info.max_sequence_length,
                num_layers: model_info.num_layers,
                num_heads: model_info.num_kv_heads,
                head_dim: model_info.hidden_size / model_info.num_heads.max(1),
                device: self.config.backend.device.clone(),
                dtype: model_info.dtype,
                priority: Priority::Normal,
            };
            let draft_kv_handle = self.kv_cache.allocate(&alloc_request).await?;
            let prompt_tensor = self.tokens_to_tensor(&prompt_u32s)?;
            let pfx = PrefillInput::new(prompt_tensor).with_kv_cache(draft_kv_handle);
            let pfx_out = draft_exec.prefill(&pfx).await?;
            let kv = pfx_out.kv_cache.clone();
            {
                let mut sequences = self.sequences.write();
                if let Some(s) = sequences.get_mut(request_id) {
                    s.draft_kv_cache = Some(kv.clone());
                }
            }
            kv
        };

        // ── 2. Pull current state + run one spec step ────────────────────
        let (target_kv, last_token) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let kv = seq
                .kv_cache
                .as_ref()
                .ok_or_else(|| FerrumError::internal("No target KV"))?
                .clone();
            let last = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0));
            (kv, last)
        };

        let runner = crate::speculative::SpeculativeRunner {
            draft: draft_exec.as_ref(),
            target: self.model_executor.as_ref(),
            tensor_factory: self.tensor_factory.clone(),
            cfg,
        };

        // Snapshot the RNG out of the sequence so the async step can borrow
        // the mutable RNG; reinstall on the way back.
        let rng_seed = {
            let sequences = self.sequences.read();
            let seed = sequences
                .get(request_id)
                .and_then(|s| s.sampling_params.seed)
                .unwrap_or(42);
            // Mix in iteration count for non-deterministic draws across
            // speculative rounds that share the same seed.
            seed.wrapping_add(self.iteration_count.load(Ordering::Relaxed))
        };
        let mut rng = rand::rngs::StdRng::from_seed({
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&rng_seed.to_le_bytes());
            seed
        });

        // Capture entry seq_len so we can compute the correct rollback
        // length on partial rejection below.
        let entry_target_seq = target_kv.block_table().sequence_length;
        let entry_draft_seq = draft_kv.block_table().sequence_length;

        let outcome = runner
            .step(last_token, draft_kv.clone(), target_kv, &mut rng)
            .await?;

        // KV-cache reconciliation post-step.
        //
        //   full accept: target wrote N+1 new positions, draft only N.
        //     Draft lags by one. Feed `draft_catchup_token`
        //     (= draft_tokens[N-1]) into draft so it catches up.
        //
        //   partial reject at k < N: target wrote N+1 positions including
        //     `N-k` that were conditioned on rejected drafts. Truncate
        //     BOTH to entry_seq + k + 1 (keep last_token + k accepted
        //     drafts), then feed the replacement token so both advance
        //     to entry_seq + k + 2 in lockstep.
        let (draft_kv_aligned, target_kv_aligned) = if let Some(catchup) =
            outcome.draft_catchup_token
        {
            let tensor = self.tokens_to_tensor(&[catchup.get()])?;
            let input = ferrum_interfaces::model_executor::DecodeInput::new(
                tensor,
                outcome.draft_kv.clone(),
            );
            let feed_out = draft_exec.decode(&input).await?;
            (feed_out.kv_cache.clone(), outcome.target_kv.clone())
        } else {
            // Partial reject path. k = rejected_at accepted drafts + 1
            // replacement → emitted.len() = k+1. Target wrote positions
            // [entry..entry+N], draft wrote [entry..entry+N-1] during the
            // runner step; only the first k writes are valid. Truncate
            // BOTH caches back to entry+k+1 (keep last_token + k accepted
            // drafts). Do NOT feed replacement here — the next iter's
            // runner will consume replacement as its new last_token and
            // write it at the correct position automatically, mirroring
            // how target self-corrects on the bonus token in full-accept.
            let k = outcome.rejected_at;
            let kept_target = entry_target_seq + k + 1;
            let kept_draft = entry_draft_seq + k + 1;

            draft_exec
                .truncate_kv(&outcome.draft_kv, kept_draft)
                .await?;
            self.model_executor
                .truncate_kv(&outcome.target_kv, kept_target)
                .await?;

            let truncated_draft = self.make_kv_handle_with_seq(&outcome.draft_kv, kept_draft);
            let truncated_target = self.make_kv_handle_with_seq(&outcome.target_kv, kept_target);
            (truncated_draft, truncated_target)
        };

        // ── 3. Install accepted tokens; check stop after each ───────────
        let mut last_emitted = last_token;
        for &tok in &outcome.tokens {
            {
                let mut sequences = self.sequences.write();
                if let Some(seq) = sequences.get_mut(request_id) {
                    seq.generated_tokens.push(tok);
                    seq.tokens_this_iteration += 1;
                }
            }
            last_emitted = tok;
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);

            self.send_stream_update(request_id, tok).await;

            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(request_id).map_or(true, |s| {
                    s.should_stop(self.model_executor.info().vocab_size)
                })
            };
            if should_stop {
                let finish_reason = {
                    let sequences = self.sequences.read();
                    match sequences.get(request_id) {
                        Some(seq)
                            if seq.generated_tokens.len() >= seq.sampling_params.max_tokens =>
                        {
                            FinishReason::Length
                        }
                        Some(_) => FinishReason::EOS,
                        None => FinishReason::Error,
                    }
                };
                self.complete_request(request_id, finish_reason).await?;
                return Ok(());
            }
        }

        // ── 4. Persist updated KV handles ───────────────────────────────
        {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(request_id) {
                seq.kv_cache = Some(target_kv_aligned);
                seq.draft_kv_cache = Some(draft_kv_aligned);
            }
        }
        let _ = last_emitted;
        let generated_count = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.generated_tokens.len())
                .unwrap_or(0)
        };
        self.scheduler
            .update_decode_progress(request_id, generated_count);

        Ok(())
    }

    // ── stream helper ──────────────────────────────────────────────────

    async fn send_stream_update(&self, request_id: &RequestId, token: TokenId) {
        let sender = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .and_then(|s| s.stream_sender.clone())
        };

        if let Some(tx) = sender {
            let token_text = self
                .tokenizer
                .decode(&[token], false)
                .unwrap_or_else(|_| format!("token_{}", token.get()));

            let chunk = StreamChunk {
                request_id: request_id.clone(),
                text: token_text,
                token: Some(token),
                finish_reason: None,
                usage: None,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            let _ = tx.send(Ok(chunk)).await;
        }
    }

    // ── completion ─────────────────────────────────────────────────────

    async fn complete_request(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        let (response, stream_sender, response_sender, has_kv_cache, model_cache_id) = {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.remove(request_id) {
                let text = self
                    .tokenizer
                    .decode(&seq.generated_tokens, true)
                    .unwrap_or_default();

                let response = InferenceResponse {
                    request_id: request_id.clone(),
                    text,
                    tokens: seq.generated_tokens.clone(),
                    finish_reason,
                    usage: TokenUsage::new(seq.input_tokens.len(), seq.generated_tokens.len()),
                    latency_ms: seq.start_time.elapsed().as_millis() as u64,
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                };

                let has_kv = seq.kv_cache.is_some();
                let cache_id = seq.model_cache_id.clone();
                (
                    response,
                    seq.stream_sender,
                    seq.response_sender,
                    has_kv,
                    cache_id,
                )
            } else {
                return Ok(());
            }
        };

        // Release model executor's KV cache for this sequence (frees GPU memory).
        if let Some(ref cache_id) = model_cache_id {
            self.model_executor.release_cache(cache_id);
        }

        if has_kv_cache {
            let _ = self.kv_cache.deallocate(request_id.clone()).await;
        }

        self.scheduler
            .complete(request_id.clone(), &response)
            .await?;

        if let Some(tx) = response_sender {
            let _ = tx.send(response.clone());
        }

        if let Some(tx) = stream_sender {
            let final_chunk = StreamChunk {
                request_id: request_id.clone(),
                text: String::new(),
                token: None,
                finish_reason: Some(finish_reason),
                usage: Some(response.usage.clone()),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
            };
            let _ = tx.send(Ok(final_chunk)).await;
        }

        debug!(
            "Request {} completed: {} tokens, {:?}",
            request_id,
            response.tokens.len(),
            finish_reason
        );

        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public engine wrapper
// ────────────────────────────────────────────────────────────────────────────

/// Continuous batching inference engine.
///
/// Wraps an `Arc<EngineInner>` so it can be cloned and shared freely.
/// Multiple concurrent `infer()` / `infer_stream()` calls are safe —
/// an internal `iteration_lock` serializes engine steps while allowing
/// all pending requests to be processed in each iteration's batch.
pub struct ContinuousBatchEngine {
    inner: Arc<EngineInner>,
}

impl ContinuousBatchEngine {
    pub fn new(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
    ) -> Self {
        Self::new_with_speculation(
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
            tensor_factory,
            None,
            None,
        )
    }

    /// Build an engine with optional speculative decoding. Pass both the
    /// draft executor AND the config together — either both or neither.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_speculation(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
        draft_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
        spec_config: Option<crate::speculative::SpeculativeDecodingConfig>,
    ) -> Self {
        info!(
            "Creating ContinuousBatchEngine (speculative_decoding={})",
            draft_executor.is_some() && spec_config.is_some()
        );

        Self {
            inner: Arc::new(EngineInner {
                config,
                scheduler,
                tokenizer,
                sampler,
                kv_cache,
                model_executor,
                draft_executor,
                spec_config,
                tensor_factory,
                sequences: RwLock::new(HashMap::new()),
                is_running: AtomicBool::new(false),
                shutdown_notify: Arc::new(Notify::new()),
                iteration_lock: tokio::sync::Mutex::new(()),
                work_notify: Notify::new(),
                iteration_count: AtomicU64::new(0),
                prefix_cache: PrefixCache::new(256, 2),
                total_prefill_tokens: AtomicU64::new(0),
                total_decode_tokens: AtomicU64::new(0),
                total_preemptions: AtomicU64::new(0),
                prefix_cache_hits: AtomicU64::new(0),
            }),
        }
    }

    /// Hit count since engine construction (prefix cache). Exposed for
    /// tests + /metrics endpoint; monotonic, Relaxed-ordered.
    pub fn prefix_cache_hits(&self) -> u64 {
        self.inner.prefix_cache_hits.load(Ordering::Relaxed)
    }

    /// Snapshot of prefix cache stats (hits/misses/evictions/active entries).
    pub fn prefix_cache_stats(&self) -> ferrum_kv::cache::prefix::PrefixCacheStats {
        self.inner.prefix_cache.stats()
    }

    /// Start a background iteration loop.  Returns a `JoinHandle` that
    /// runs until `shutdown()` is called.  When a background loop is
    /// active, `infer()` / `infer_stream()` simply submit and wait.
    pub fn start_loop(&self) -> tokio::task::JoinHandle<()> {
        let inner = self.inner.clone();
        inner.is_running.store(true, Ordering::SeqCst);
        tokio::spawn(async move {
            info!("Background iteration loop started");
            loop {
                if !inner.is_running.load(Ordering::SeqCst) {
                    break;
                }
                {
                    let _guard = inner.iteration_lock.lock().await;
                    if let Err(e) = inner.run_iteration().await {
                        warn!("Iteration error: {}", e);
                    }
                }
                tokio::task::yield_now().await;
            }
            info!("Background iteration loop stopped");
        })
    }
}

#[async_trait]
impl InferenceEngine for ContinuousBatchEngine {
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.id.clone();
        let infer_start = Instant::now();
        counter!("ferrum.engine.requests_total").increment(1);
        gauge!("ferrum.engine.active_requests").increment(1.0);

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with oneshot channel
        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut seq_state = SequenceState::new_with_tokenizer(
            request,
            input_tokens,
            Some(self.inner.tokenizer.clone()),
        );
        seq_state.response_sender = Some(resp_tx);
        self.inner
            .sequences
            .write()
            .insert(request_id.clone(), seq_state);

        // Notify any background loop
        self.inner.work_notify.notify_one();

        // Drive iterations until our request completes
        self.inner.drive_to_completion(&request_id).await?;

        let result = resp_rx
            .await
            .map_err(|_| FerrumError::internal("Response channel closed before response was sent"));

        gauge!("ferrum.engine.active_requests").decrement(1.0);
        let elapsed_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("ferrum.engine.request_duration_ms").record(elapsed_ms);

        if let Ok(ref resp) = result {
            counter!("ferrum.engine.requests_completed").increment(1);
            counter!("ferrum.engine.tokens_generated_total").increment(resp.tokens.len() as u64);
            histogram!("ferrum.engine.ttft_ms")
                .record(elapsed_ms / resp.tokens.len().max(1) as f64);
        } else {
            counter!("ferrum.engine.requests_failed").increment(1);
        }

        result
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (tx, rx) = mpsc::channel(100);
        let request_id = request.id.clone();

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with stream sender
        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        let mut seq_state = SequenceState::new_with_tokenizer(
            request,
            input_tokens,
            Some(self.inner.tokenizer.clone()),
        );
        seq_state.stream_sender = Some(tx);
        self.inner
            .sequences
            .write()
            .insert(request_id.clone(), seq_state);

        // Notify any background loop
        self.inner.work_notify.notify_one();

        // Spawn a driver task so we can return the stream immediately
        let inner = self.inner.clone();
        tokio::spawn(async move {
            if let Err(e) = inner.drive_to_completion(&request_id).await {
                warn!("Stream driver error for {}: {}", request_id, e);
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn status(&self) -> EngineStatus {
        let metrics = self.inner.scheduler.metrics();
        EngineStatus {
            is_ready: self.inner.is_running.load(Ordering::SeqCst),
            loaded_models: vec![],
            active_requests: metrics.running_requests,
            queued_requests: metrics.waiting_requests,
            memory_usage: ferrum_types::MemoryUsage {
                total_bytes: 0,
                used_bytes: 0,
                free_bytes: 0,
                gpu_memory_bytes: None,
                cpu_memory_bytes: None,
                cache_memory_bytes: 0,
                utilization_percent: 0.0,
            },
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down continuous batch engine");
        self.inner.is_running.store(false, Ordering::SeqCst);
        self.inner.shutdown_notify.notify_waiters();
        Ok(())
    }

    fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    fn metrics(&self) -> ferrum_types::EngineMetrics {
        let sm = self.inner.scheduler.metrics();
        ferrum_types::EngineMetrics {
            total_requests: sm.completed_requests + sm.failed_requests,
            successful_requests: sm.completed_requests,
            failed_requests: sm.failed_requests,
            avg_request_latency_ms: 0.0,
            p95_request_latency_ms: 0.0,
            p99_request_latency_ms: 0.0,
            throughput_rps: sm.throughput_rps as f32,
            tokens_per_second: 0.0,
            queue_metrics: Default::default(),
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: Default::default(),
        }
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        if self.inner.is_running.load(Ordering::SeqCst) {
            ferrum_types::HealthStatus::healthy()
        } else {
            ferrum_types::HealthStatus {
                status: ferrum_types::HealthStatusType::Unhealthy,
                component_status: ferrum_types::ComponentStatus::healthy(),
                last_check: chrono::Utc::now(),
            }
        }
    }
}

impl std::fmt::Debug for ContinuousBatchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousBatchEngine")
            .field("is_running", &self.inner.is_running.load(Ordering::SeqCst))
            .field(
                "iteration_count",
                &self.inner.iteration_count.load(Ordering::SeqCst),
            )
            .field("active_sequences", &self.inner.sequences.read().len())
            .finish()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_state() {
        let request = InferenceRequest {
            id: RequestId::new(),
            prompt: "test".to_string(),
            model_id: ferrum_types::ModelId::new("test"),
            sampling_params: SamplingParams::default(),
            stream: false,
            priority: Priority::Normal,
            client_id: None,
            session_id: None,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        };

        let tokens = vec![TokenId::new(1), TokenId::new(2)];
        let state = SequenceState::new(request, tokens);

        assert_eq!(state.phase, RequestPhase::Waiting);
        assert_eq!(state.total_tokens(), 2);
        assert!(!state.prefill_complete);
    }
}
