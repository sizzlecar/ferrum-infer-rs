//! Continuous Batching Engine
//!
//! Iteration-level continuous batching: each step processes a mixed batch of
//! prefill and decode requests selected by the scheduler.  Multiple callers
//! can submit requests concurrently — an `iteration_lock` serializes the
//! actual engine steps so each batch is processed exactly once.

use async_trait::async_trait;
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    kv_cache::AllocationRequest,
    KvCacheHandle, KvCacheManager, ModelExecutor, Sampler, SchedulerInterface as Scheduler,
    TensorFactory, TensorRef, Tokenizer,
};
use ferrum_kv::cache::prefix::PrefixCache;
use ferrum_sampler::json_mode::JsonModeProcessor;
use ferrum_scheduler::implementations::{ContinuousBatchScheduler, RequestPhase};
use ferrum_types::{
    DataType, Device, EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest,
    InferenceResponse, Priority, RequestId, Result, SamplingParams, StreamChunk, TokenId,
    TokenUsage, PROMPT_TOKENS_METADATA_KEY,
};
use futures::stream::Stream;
use metrics::{counter, gauge, histogram};
use parking_lot::RwLock;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

/// Resolve per-request stop conditions into (single-token-ids, multi-token-texts).
///
/// Combines:
/// 1. Model EOS reported by the tokenizer (`special_tokens().eos_token`).
/// 2. Common chat-EOS literal names looked up in the tokenizer's vocab —
///    `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `</s>`. Each lookup is
///    model-specific (only IDs that actually exist in this vocab get added),
///    so there's no risk of inserting an unrelated token id from a hard-coded
///    fallback list (e.g. `2` is `</s>` for LLaMA but `!` for Qwen3).
/// 3. User-supplied `stop_sequences` — each is encoded with `add_special=false`;
///    one-token results land in `stop_token_ids`, multi-token results go to
///    `stop_text_seqs` for text-match fallback.
fn resolve_stop_conditions(
    params: &SamplingParams,
    tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
) -> (HashSet<u32>, Vec<String>) {
    let mut ids: HashSet<u32> = HashSet::new();
    let mut text_seqs: Vec<String> = Vec::new();

    if let Some(tok) = tokenizer {
        if let Some(eos) = tok.special_tokens().eos_token {
            ids.insert(eos.get());
        }
        for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
            if let Some(t) = tok.token_id(name) {
                ids.insert(t.get());
            }
        }
        for stop_seq in &params.stop_sequences {
            match tok.encode(stop_seq, false) {
                Ok(toks) if toks.len() == 1 => {
                    ids.insert(toks[0].get());
                }
                _ => text_seqs.push(stop_seq.clone()),
            }
        }
    } else {
        for stop_seq in &params.stop_sequences {
            text_seqs.push(stop_seq.clone());
        }
    }
    (ids, text_seqs)
}

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
    /// Wall-clock `Instant` at which the first SSE chunk was actually
    /// sent to the client stream. Populated lazily by `send_stream_update`
    /// the first time a non-empty delta is emitted (multi-byte UTF-8
    /// buffering can defer that past the first scheduler-completed token).
    /// Used to record `ferrum.engine.ttft_seconds` and as the start point
    /// of the TPOT window.
    pub first_emit_at: Option<Instant>,
    /// Wall-clock `Instant` at the most recent successfully-sent chunk.
    /// Used to compute per-token ITL deltas (`ferrum.engine.itl_seconds`).
    pub last_emit_at: Option<Instant>,
    /// Count of stream chunks successfully sent to the client. Lags
    /// `generated_tokens.len()` by the number of tokens currently buffered
    /// for a multi-byte UTF-8 sequence (so a Chinese char split across
    /// 2 BPE tokens emits once, increments the count by 1).
    pub emitted_chunks: u32,
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
    /// Single-token stop ids: model's EOS + any `stop_sequences` that encode to
    /// exactly one token. Checked against the last generated token each step
    /// — replaces the old "token id near top of vocab = EOS" placeholder. Built
    /// from `tokenizer.eos_token`, a common-EOS fallback list (`</s>`,
    /// `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`), and one-token encodings of
    /// `sampling_params.stop_sequences`.
    pub stop_token_ids: HashSet<u32>,
    /// Multi-token text stop sequences (`stop_sequences` entries that don't
    /// resolve to a single token). Checked via accumulated decoded text.
    pub stop_text_seqs: Vec<String>,
    /// Bytes of decoded `generated_tokens` already flushed via the stream
    /// channel. Used by `send_stream_update` to compute per-call delta from
    /// the full-history decode, so multi-byte UTF-8 sequences (Chinese chars,
    /// emoji) that span several BPE tokens don't get rendered as
    /// `\u{FFFD}` replacement chars when decoded one token at a time.
    pub streamed_text_len: usize,
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
        let (stop_token_ids, stop_text_seqs) =
            resolve_stop_conditions(&request.sampling_params, tokenizer.as_deref());
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
            first_emit_at: None,
            last_emit_at: None,
            emitted_chunks: 0,
            tokens_this_iteration: 0,
            preemption_count: 0,
            json_processor,
            regex_processor,
            draft_kv_cache: None,
            token_frequencies: HashMap::new(),
            model_cache_id: None,
            stop_token_ids,
            stop_text_seqs,
            streamed_text_len: 0,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    /// Should this sequence stop generating?
    ///
    /// Checks: (1) max-tokens budget exhausted, (2) last generated token is in
    /// the resolved `stop_token_ids` set (model EOS + any single-token
    /// `stop_sequences`). The pre-2026-05 placeholder ("last token id near
    /// top of vocab") was wrong for every real model — Qwen3 EOS is 151645
    /// in a 151936-vocab, Llama-3 EOT is 128009 in a 128256-vocab, etc. —
    /// causing chat to never stop until KV overflow.
    ///
    /// Multi-token text stop sequences (`stop_text_seqs`) are NOT checked
    /// here; the chat REPL only uses single-token EOS markers, and adding
    /// a per-step decode just to support a rare case is not free. Callers
    /// that need text stops can layer that check on top.
    pub fn should_stop(&self) -> bool {
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return true;
        }
        if let Some(&last_token) = self.generated_tokens.last() {
            if self.stop_token_ids.contains(&last_token.get()) {
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
    /// Set true the first time `ensure_bg_loop` runs, so per-request
    /// `infer_stream` callers don't each spawn their own competing
    /// driver task (16 streaming requests = 16 drivers thrashing on
    /// `iteration_lock`, ~5ms/iter of tokio scheduling overhead).
    bg_loop_spawned: AtomicBool,
}

impl EngineInner {
    // ── tensor helper ──────────────────────────────────────────────────

    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<TensorRef> {
        let f32_data: Vec<f32> = token_ids.iter().map(|&v| v as f32).collect();
        let len = f32_data.len();
        self.tensor_factory
            .from_slice(&f32_data, &[1, len], DataType::FP32, Device::CPU)
    }

    /// Rebuild a KvCacheHandle with a corrected sequence_length.
    ///
    /// Only meaningful for `GenericKvCacheHandle`, which is what the LLM
    /// executor (`LlmExecutor::prefill` / `decode`) constructs and threads
    /// through speculative decoding. Resource handles minted by
    /// `KvCacheManager` impls (Paged / Default) are returned as a plain
    /// clone — those handles don't track per-iter position (the model's
    /// internal paged_pool does), and the engine no longer reads
    /// `sequence_length` from them for position purposes (see
    /// `process_batch_unified` for the SequenceState-sourced pos_offset).
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

    /// Run one iteration: ask the scheduler for a batch, then process it.
    async fn run_iteration(&self) -> Result<()> {
        let iteration = self.iteration_count.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.iterations_total").increment(1);
        let prof = std::env::var("FERRUM_BATCH_DECODE_PROF").is_ok();
        let t_iter_start = if prof { Some(Instant::now()) } else { None };

        // Phase 3 token-budget hint: scheduler emits a mixed batch
        // summing to at most `max_num_batched_tokens` Q tokens. This
        // replaces the prior `max_batch_size * 2048` heuristic which
        // never bit and left scheduler-side prefill admission capped
        // at `max_prefill_batch=8`. Defaults to 4096 (autosizer can
        // override via `FERRUM_MAX_BATCHED_TOKENS`).
        let hint = ferrum_interfaces::BatchHint {
            max_batch_size: self.config.batching.max_batch_size,
            max_tokens: self.config.batching.max_num_batched_tokens,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: ferrum_interfaces::scheduler::ResourceConstraints::default(),
        };

        // FERRUM_NEXT_BATCH_PROF=1: count Some/None returns to root-cause
        // the apples HTTP-serve 17 ms inter-batch-iter gap. Prints every
        // 1024 run_iteration calls. Set None_size to 0 explicitly so the
        // bg-loop tight-spin theory can be confirmed.
        let nb_prof = std::env::var("FERRUM_NEXT_BATCH_PROF").is_ok();
        let nb_t0 = if nb_prof { Some(Instant::now()) } else { None };
        let nb_result = self.scheduler.next_batch(hint).await;
        if let Some(t0) = nb_t0 {
            use std::sync::atomic::AtomicU64;
            static SOME_N: AtomicU64 = AtomicU64::new(0);
            static NONE_N: AtomicU64 = AtomicU64::new(0);
            static SOME_US: AtomicU64 = AtomicU64::new(0);
            static NONE_US: AtomicU64 = AtomicU64::new(0);
            let us = t0.elapsed().as_micros() as u64;
            let is_some = nb_result.is_some();
            let batch_size = nb_result.as_ref().map_or(0, |b| b.size());
            if is_some {
                SOME_N.fetch_add(1, Ordering::Relaxed);
                SOME_US.fetch_add(us, Ordering::Relaxed);
            } else {
                NONE_N.fetch_add(1, Ordering::Relaxed);
                NONE_US.fetch_add(us, Ordering::Relaxed);
            }
            let total = SOME_N.load(Ordering::Relaxed) + NONE_N.load(Ordering::Relaxed);
            if total.is_multiple_of(1024) {
                let s_n = SOME_N.load(Ordering::Relaxed);
                let n_n = NONE_N.load(Ordering::Relaxed);
                let s_us = SOME_US.load(Ordering::Relaxed);
                let n_us = NONE_US.load(Ordering::Relaxed);
                eprintln!(
                    "[nb-prof] total={} some={} none={} ratio={:.3} | some_avg={}us none_avg={}us last_batch_size={} last_was_some={}",
                    total,
                    s_n,
                    n_n,
                    s_n as f64 / total as f64,
                    if s_n > 0 { s_us / s_n } else { 0 },
                    if n_n > 0 { n_us / n_n } else { 0 },
                    batch_size,
                    is_some,
                );
            }
        }

        let batch = match nb_result {
            Some(b) => b,
            None => {
                tokio::time::sleep(Duration::from_millis(1)).await;
                return Ok(());
            }
        };
        let t_after_sched = if prof { Some(Instant::now()) } else { None };

        debug!(
            "Iteration {}: batch with {} requests",
            iteration,
            batch.size()
        );

        let r = self.process_batch(&batch).await;
        if let (Some(t0), Some(ts)) = (t_iter_start, t_after_sched) {
            let n = self.iteration_count.load(Ordering::Relaxed);
            if n < 64 || n.is_multiple_of(32) {
                let total = t0.elapsed().as_micros();
                let sched = ts.duration_since(t0).as_micros();
                let proc = ts.elapsed().as_micros();
                eprintln!(
                    "[iter-prof] iter#{} total={}us sched={}us process={}us batch_size={}",
                    iteration,
                    total,
                    sched,
                    proc,
                    batch.size()
                );
            }
        }
        r
    }

    // ── batch processing ───────────────────────────────────────────────

    async fn process_batch(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        // Single-shot unified path: prefill + decode items go through ONE
        // `model_executor.unified_decode` call. Phase-2/3 redesign goal —
        // prefill chunks and decode tokens are co-batched at the kernel
        // level, eliminating the cohort gap (decode_queue=0 ↔ 32
        // alternation that consumed half the bench wall on apples).
        //
        // For chunked-prefill (FERRUM_CHUNKED_PREFILL=N) or speculative
        // decoding, fall back to the legacy split path. Phase 3 will
        // extend chunked-prefill into the unified mode.
        let chunked_or_spec =
            std::env::var("FERRUM_CHUNKED_PREFILL").is_ok() || self.spec_config.is_some();
        if chunked_or_spec {
            return self.process_batch_legacy_split(batch).await;
        }
        self.process_batch_unified(batch).await
    }

    /// Legacy split path: separate prefill (batched via run_batch_prefill)
    /// then decode (batched via run_batch_decode). Used for chunked-prefill
    /// and speculative-decoding flows that the unified path doesn't model yet.
    async fn process_batch_legacy_split(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
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
        if !prefill_ids.is_empty() {
            if let Err(e) = self.run_batch_prefill(&prefill_ids).await {
                warn!("Batch prefill failed: {}; falling back to per-request", e);
                for rid in &prefill_ids {
                    if let Err(e) = self.run_prefill(rid).await {
                        warn!("Prefill failed for {}: {}", rid, e);
                        self.complete_request(rid, FinishReason::Error).await?;
                    }
                }
            }
        }
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

    /// Unified path: build ONE `UnifiedBatch` from all requests in the plan
    /// (prefill items get full input_tokens at pos_offset=0; decode items
    /// get [last_token] at pos_offset=current_kv_len), then a single
    /// `model_executor.unified_decode` call drives the entire forward.
    ///
    /// When the model's `unified_forward` returns Unsupported (Qwen3Moe
    /// today, until Phase 2 native), `LlmExecutor.unified_decode`'s fallback
    /// partitions the batch by item shape and serializes prefills under one
    /// model lock — behavior-preserving but no perf gain. Llama already
    /// supports unified_forward, so M2 immediately co-batches prefill+decode.
    async fn process_batch_unified(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        use ferrum_interfaces::model_executor::{UnifiedBatch, UnifiedBatchItem};
        use ferrum_interfaces::KvCacheHandle;

        // ── 0. Materialize SequenceState for every request, classify ──
        let mut prefill_ids: Vec<RequestId> = Vec::new();
        let mut decode_ids: Vec<RequestId> = Vec::new();
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

        // ── 1. Per-prefill setup: prefix-cache check, KV alloc, gather tokens ──
        // Prefix-cache hits short-circuit through the legacy single-prompt
        // path (they don't enter the unified batch — they have no model
        // call to make). The remainder are added to `unified_prefills`.
        let model_info = self.model_executor.info();
        // Prefix cache defaults OFF on every backend. The `clone_handle`
        // path in `crates/ferrum-kv/src/managers/paged.rs` is COW-by-flag
        // but the engine write path doesn't fork blocks on first write,
        // so a second request that hits the cache shares mutated KV from
        // the first request's decode and diverges deterministically
        // (request 1 ≠ request 2 == request 3). Reproduced 2026-05-19;
        // see `~/.claude/projects/*/memory/project_http_server_gaps_2026_05_19.md`.
        // Opt in via `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        let skip_prefix_cache = std::env::var("FERRUM_PREFIX_CACHE").map_or(true, |v| v != "1");
        let mut unified_prefills: Vec<(RequestId, Vec<TokenId>, Arc<dyn KvCacheHandle>)> =
            Vec::new();
        for rid in &prefill_ids {
            let (input_tokens, num_tokens) = {
                let sequences = self.sequences.read();
                let Some(seq) = sequences.get(rid) else {
                    continue;
                };
                (seq.input_tokens.clone(), seq.input_tokens.len())
            };
            if !skip_prefix_cache {
                let hit = self
                    .prefix_cache
                    .find_prefix(&input_tokens)
                    .filter(|(prefix_id, _, _)| prefix_id.len() == input_tokens.len());
                if let Some((_, cached_kv, cached_logits)) = hit {
                    let cloned_kv = cached_kv.clone_handle()?;
                    let first_token = {
                        let mut sequences = self.sequences.write();
                        let Some(seq) = sequences.get_mut(rid) else {
                            continue;
                        };
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
                    self.scheduler.mark_prefill_complete(rid, num_tokens);
                    self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                    counter!("ferrum.engine.prefix_cache_hits").increment(1);
                    self.send_stream_update(rid, first_token).await;
                    let should_stop = {
                        let sequences = self.sequences.read();
                        sequences.get(rid).is_none_or(|s| s.should_stop())
                    };
                    if should_stop {
                        self.complete_request(rid, FinishReason::EOS).await?;
                    }
                    continue;
                }
            }
            // Allocate KV pages (with preempt fallback) for fresh prefill.
            let alloc_request = AllocationRequest {
                request_id: rid.clone(),
                initial_tokens: num_tokens,
                max_sequence_length: model_info.max_sequence_length,
                num_layers: model_info.num_layers,
                num_heads: model_info.num_kv_heads,
                head_dim: model_info.hidden_size / model_info.num_heads.max(1),
                device: self.config.backend.device.clone(),
                dtype: model_info.dtype,
                priority: Priority::Normal,
            };
            let kv_handle = match self.kv_cache.allocate(&alloc_request).await {
                Ok(h) => h,
                Err(_) => {
                    if self.preempt_victim(rid).await {
                        match self.kv_cache.allocate(&alloc_request).await {
                            Ok(h) => h,
                            Err(e) => {
                                warn!("Unified prefill alloc failed for {}: {}", rid, e);
                                self.complete_request(rid, FinishReason::Error).await?;
                                continue;
                            }
                        }
                    } else {
                        warn!("Unified prefill alloc failed for {}: no victim", rid);
                        self.complete_request(rid, FinishReason::Error).await?;
                        continue;
                    }
                }
            };
            unified_prefills.push((rid.clone(), input_tokens, kv_handle));
        }

        // ── 2. Build the UnifiedBatch (prefill chunks + decode tokens) ──
        let mut unified = UnifiedBatch::new();
        let mut prefill_meta: Vec<(RequestId, Vec<TokenId>)> = Vec::new();
        let mut decode_meta: Vec<RequestId> = Vec::new();
        for (rid, tokens, kv_handle) in &unified_prefills {
            let q_tokens: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
            let seq_id = kv_handle.cache_id();
            unified.items.push(UnifiedBatchItem {
                seq_id,
                q_tokens,
                kv_cache: kv_handle.clone(),
                pos_offset: 0,
                is_final_chunk: true,
            });
            prefill_meta.push((rid.clone(), tokens.clone()));
        }
        {
            let sequences = self.sequences.read();
            for rid in &decode_ids {
                let Some(seq) = sequences.get(rid) else {
                    continue;
                };
                let Some(kv) = seq.kv_cache.clone() else {
                    continue;
                };
                let last_token = seq
                    .generated_tokens
                    .last()
                    .copied()
                    .unwrap_or(TokenId::new(0));
                // pos_offset = position of the NEW token in the K/V cache.
                // It must increment by 1 every decode step. Source of truth
                // is the engine's own bookkeeping: input prompt + tokens
                // generated so far (the last one is the one we're about to
                // decode, so its slot is `len - 1` past the prompt). NOT
                // `kv.block_table().sequence_length` — that field is set
                // once at allocate() time and `make_kv_handle_with_seq`
                // doesn't actually update Paged/Default handles, so reading
                // it leaves every decode step at the same position.
                let pos_offset = seq.input_tokens.len() + seq.generated_tokens.len() - 1;
                let seq_id = seq
                    .model_cache_id
                    .clone()
                    .unwrap_or_else(|| rid.to_string());
                unified.items.push(UnifiedBatchItem {
                    seq_id,
                    q_tokens: vec![last_token.get()],
                    kv_cache: kv,
                    pos_offset,
                    is_final_chunk: true,
                });
                decode_meta.push(rid.clone());
            }
        }

        if unified.items.is_empty() {
            return Ok(());
        }

        // ── 3. ONE unified forward call ──
        let unified_prof = std::env::var("FERRUM_UNIFIED_POST_PROF").is_ok();
        let t_unified_model = if unified_prof {
            Some(Instant::now())
        } else {
            None
        };
        let results = match self.model_executor.unified_decode(&unified).await {
            Ok(r) => r,
            Err(e) => {
                warn!("Unified forward failed: {}; falling back to split", e);
                // Release the KV cache slots we just allocated for the
                // unified-path prefills — otherwise the legacy split
                // path's `run_batch_prefill` re-allocates for the same
                // request_id, double-counting `active_caches` (only one
                // of the two pairs ever gets deallocated by
                // `complete_request`). Found via paged_attention_test.
                for (rid, _, _) in &unified_prefills {
                    let _ = self.kv_cache.deallocate(rid.clone()).await;
                }
                return self.process_batch_legacy_split(batch).await;
            }
        };
        let t_unified_model_done = if unified_prof {
            Some(Instant::now())
        } else {
            None
        };
        if results.len() != unified.items.len() {
            return Err(FerrumError::internal(format!(
                "unified_decode returned {} results for {} items",
                results.len(),
                unified.items.len(),
            )));
        }

        // ── 4. Per-item post-process — split by category ──
        // Prefill items come first (in the order added), then decode items.
        let prefill_count = prefill_meta.len();
        let decode_count = decode_meta.len();
        let item_count = unified.items.len();
        let mut t_decode_sample_us: u64 = 0;
        let mut t_decode_sched_us: u64 = 0;
        let mut t_decode_stream_us: u64 = 0;
        let mut t_decode_stop_us: u64 = 0;
        let mut t_decode_complete_us: u64 = 0;
        for (i, (rid, input_tokens)) in prefill_meta.into_iter().enumerate() {
            let logits_vec = match &results[i] {
                Some(l) => l.clone(),
                None => {
                    warn!("Unified prefill result missing for {}", rid);
                    continue;
                }
            };
            let num_tokens = input_tokens.len();
            let kv_handle = unified.items[i].kv_cache.clone();
            // Store in prefix cache (best-effort).
            let _ = self.prefix_cache.store_prefix(
                &input_tokens,
                kv_handle.clone(),
                logits_vec.clone(),
            );
            let first_token = {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&rid) else {
                    continue;
                };
                if let Some(ref jp) = seq.json_processor {
                    jp.reset();
                }
                let mut logits = logits_vec;
                let token = seq.sample_with_processors(&mut logits)?;
                seq.generated_tokens.push(token);
                seq.model_cache_id = Some(kv_handle.cache_id());
                seq.kv_cache = Some(kv_handle);
                seq.prefill_complete = true;
                seq.phase = RequestPhase::Decoding;
                token
            };
            self.scheduler.mark_prefill_complete(&rid, num_tokens);
            self.total_prefill_tokens
                .fetch_add(num_tokens as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
            counter!("ferrum.engine.prefills_total").increment(1);
            self.send_stream_update(&rid, first_token).await;
            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(&rid).is_none_or(|s| s.should_stop())
            };
            if should_stop {
                self.complete_request(&rid, FinishReason::EOS).await?;
            }
        }
        for (j, rid) in decode_meta.into_iter().enumerate() {
            let i = prefill_count + j;
            let t0_sample = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            let logits_vec = match &results[i] {
                Some(l) => l.clone(),
                None => continue,
            };
            let next_token = {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&rid) else {
                    continue;
                };
                let mut logits = logits_vec;
                let token = if logits.len() == 1 {
                    TokenId::new(logits[0] as u32)
                } else {
                    seq.sample_with_processors(&mut logits)?
                };
                seq.generated_tokens.push(token);
                seq.tokens_this_iteration += 1;
                // pos_offset is sourced from SequenceState bookkeeping above
                // (`input_tokens.len() + generated_tokens.len() - 1`); the
                // engine-side KV handle's `sequence_length` field is no
                // longer load-bearing here. Resource handles like
                // PagedKvCacheHandle don't update the field anyway (the
                // model's internal paged_pool is what actually grows), so
                // the previous `make_kv_handle_with_seq` write was a
                // silent no-op for production handles.
                token
            };
            if let Some(t0) = t0_sample {
                t_decode_sample_us += t0.elapsed().as_micros() as u64;
            }
            let t0_sched = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            let generated_count = {
                let sequences = self.sequences.read();
                sequences
                    .get(&rid)
                    .map(|s| s.generated_tokens.len())
                    .unwrap_or(0)
            };
            self.scheduler.update_decode_progress(&rid, generated_count);
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);
            if let Some(t0) = t0_sched {
                t_decode_sched_us += t0.elapsed().as_micros() as u64;
            }
            let t0_stream = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            self.send_stream_update(&rid, next_token).await;
            if let Some(t0) = t0_stream {
                t_decode_stream_us += t0.elapsed().as_micros() as u64;
            }
            let t0_stop = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(&rid).is_none_or(|s| s.should_stop())
            };
            if let Some(t0) = t0_stop {
                t_decode_stop_us += t0.elapsed().as_micros() as u64;
            }
            if should_stop {
                let t0_complete = if unified_prof {
                    Some(Instant::now())
                } else {
                    None
                };
                let finish_reason = {
                    let sequences = self.sequences.read();
                    match sequences.get(&rid) {
                        Some(seq)
                            if seq.generated_tokens.len() >= seq.sampling_params.max_tokens =>
                        {
                            FinishReason::Length
                        }
                        Some(_) => FinishReason::EOS,
                        None => FinishReason::Error,
                    }
                };
                self.complete_request(&rid, finish_reason).await?;
                if let Some(t0) = t0_complete {
                    t_decode_complete_us += t0.elapsed().as_micros() as u64;
                }
            }
        }
        if let (Some(t0), Some(t1)) = (t_unified_model, t_unified_model_done) {
            use std::sync::atomic::AtomicU64;
            static UNIFIED_PROF_N: AtomicU64 = AtomicU64::new(0);
            let n = UNIFIED_PROF_N.fetch_add(1, Ordering::Relaxed);
            if n < 64 || n.is_multiple_of(32) {
                let model_us = t1.duration_since(t0).as_micros() as u64;
                let total_us = t0.elapsed().as_micros() as u64;
                let decode_post_us = t_decode_sample_us
                    + t_decode_sched_us
                    + t_decode_stream_us
                    + t_decode_stop_us
                    + t_decode_complete_us;
                eprintln!(
                    "[unified-prof] iter#{} items={} prefill={} decode={} total={}us model={}us decode_post={}us | sample={} sched={} stream={} stop={} complete={} (us)",
                    n,
                    item_count,
                    prefill_count,
                    decode_count,
                    total_us,
                    model_us,
                    decode_post_us,
                    t_decode_sample_us,
                    t_decode_sched_us,
                    t_decode_stream_us,
                    t_decode_stop_us,
                    t_decode_complete_us,
                );
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
        let prefill_prof = std::env::var("FERRUM_BATCH_DECODE_PROF").is_ok();
        let prefill_t0 = if prefill_prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let res = self.run_prefill_inner(request_id).await;
        if let Some(t0) = prefill_t0 {
            static PREFILL_PROF_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = PREFILL_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let elapsed = t0.elapsed().as_micros();
            eprintln!(
                "[prefill-prof] call#{} req={} elapsed={}us ok={}",
                n,
                request_id,
                elapsed,
                res.is_ok()
            );
        }
        res
    }

    async fn run_prefill_inner(&self, request_id: &RequestId) -> Result<()> {
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
        // CUDA + CPU + Metal: prefix cache defaults OFF. The `clone_handle`
        // path in ferrum-kv flags blocks as COW but the write path doesn't
        // fork before mutating, so cache hits share decode-time mutations
        // back into the cached prefix — first request differs from
        // subsequent ones (reproduced 2026-05-19, see gaps memo). Opt in
        // via env `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        // Prefix cache defaults OFF on every backend. The `clone_handle`
        // path in `crates/ferrum-kv/src/managers/paged.rs` is COW-by-flag
        // but the engine write path doesn't fork blocks on first write,
        // so a second request that hits the cache shares mutated KV from
        // the first request's decode and diverges deterministically
        // (request 1 ≠ request 2 == request 3). Reproduced 2026-05-19;
        // see `~/.claude/projects/*/memory/project_http_server_gaps_2026_05_19.md`.
        // Opt in via `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        let skip_prefix_cache = std::env::var("FERRUM_PREFIX_CACHE").map_or(true, |v| v != "1");
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
                    sequences.get(request_id).is_none_or(|s| s.should_stop())
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

                self.scheduler.mark_prefill_chunk_processed(
                    request_id,
                    num_tokens,
                    end - processed,
                );

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
            sequences.get(request_id).is_none_or(|s| s.should_stop())
        };
        if should_stop {
            self.complete_request(request_id, FinishReason::EOS).await?;
        }

        Ok(())
    }

    // ── batch prefill ─────────────────────────────────────────────────

    /// Run prefill for multiple requests as ONE batched forward pass.
    ///
    /// Replaces the serial `for rid in prefill_ids { run_prefill }` loop
    /// in `process_batch`. Per-request setup (prefix cache check + KV
    /// allocation + tokenization) still happens individually; the GPU
    /// call coalesces into one `model_executor.batch_prefill` invocation.
    ///
    /// Falls back to serial `run_prefill` per request when chunked prefill
    /// is enabled (`FERRUM_CHUNKED_PREFILL=N`) — those paths have
    /// multi-call semantics that the batched path doesn't model yet.
    /// Phase 2 will lift this restriction.
    async fn run_batch_prefill(&self, request_ids: &[RequestId]) -> Result<()> {
        use ferrum_interfaces::model_executor::PrefillInput;

        if request_ids.is_empty() {
            return Ok(());
        }

        // Chunked-prefill opt-in path: fall back to serial.
        if std::env::var("FERRUM_CHUNKED_PREFILL").is_ok() {
            for rid in request_ids {
                if let Err(e) = self.run_prefill(rid).await {
                    warn!("Prefill failed for {}: {}", rid, e);
                    self.complete_request(rid, FinishReason::Error).await?;
                }
            }
            return Ok(());
        }

        // ── Phase 1a: per-request setup (prefix cache → tokens → kv alloc) ──
        // After this loop, `to_prefill` holds only requests that need a real
        // model call. Prefix cache hits + immediate stops are handled inline.
        let mut to_prefill: Vec<(
            RequestId,
            Vec<TokenId>,
            Arc<dyn ferrum_interfaces::KvCacheHandle>,
        )> = Vec::new();

        let model_info = self.model_executor.info();
        // Prefix cache defaults OFF on every backend. The `clone_handle`
        // path in `crates/ferrum-kv/src/managers/paged.rs` is COW-by-flag
        // but the engine write path doesn't fork blocks on first write,
        // so a second request that hits the cache shares mutated KV from
        // the first request's decode and diverges deterministically
        // (request 1 ≠ request 2 == request 3). Reproduced 2026-05-19;
        // see `~/.claude/projects/*/memory/project_http_server_gaps_2026_05_19.md`.
        // Opt in via `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        let skip_prefix_cache = std::env::var("FERRUM_PREFIX_CACHE").map_or(true, |v| v != "1");

        for rid in request_ids {
            let (input_tokens, num_tokens) = {
                let sequences = self.sequences.read();
                let Some(seq) = sequences.get(rid) else {
                    continue; // request gone (cancelled mid-batch)
                };
                (seq.input_tokens.clone(), seq.input_tokens.len())
            };

            // Prefix cache hit short-circuit (mirrors run_prefill_inner).
            if !skip_prefix_cache {
                let hit = self
                    .prefix_cache
                    .find_prefix(&input_tokens)
                    .filter(|(prefix_id, _, _)| prefix_id.len() == input_tokens.len());
                if let Some((_, cached_kv, cached_logits)) = hit {
                    let cloned_kv = cached_kv.clone_handle()?;
                    let first_token = {
                        let mut sequences = self.sequences.write();
                        let Some(seq) = sequences.get_mut(rid) else {
                            continue;
                        };
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
                    self.scheduler.mark_prefill_complete(rid, num_tokens);
                    self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                    counter!("ferrum.engine.prefix_cache_hits").increment(1);
                    self.send_stream_update(rid, first_token).await;
                    let should_stop = {
                        let sequences = self.sequences.read();
                        sequences.get(rid).is_none_or(|s| s.should_stop())
                    };
                    if should_stop {
                        self.complete_request(rid, FinishReason::EOS).await?;
                    }
                    continue;
                }
            }

            // Cache miss — allocate KV pages.
            let alloc_request = AllocationRequest {
                request_id: rid.clone(),
                initial_tokens: num_tokens,
                max_sequence_length: model_info.max_sequence_length,
                num_layers: model_info.num_layers,
                num_heads: model_info.num_kv_heads,
                head_dim: model_info.hidden_size / model_info.num_heads.max(1),
                device: self.config.backend.device.clone(),
                dtype: model_info.dtype,
                priority: Priority::Normal,
            };
            let kv_handle = match self.kv_cache.allocate(&alloc_request).await {
                Ok(h) => h,
                Err(_) => {
                    if self.preempt_victim(rid).await {
                        match self.kv_cache.allocate(&alloc_request).await {
                            Ok(h) => h,
                            Err(e) => {
                                warn!("Prefill alloc failed for {} after preempt: {}", rid, e);
                                self.complete_request(rid, FinishReason::Error).await?;
                                continue;
                            }
                        }
                    } else {
                        warn!("Prefill alloc failed for {}: no preempt victim", rid);
                        self.complete_request(rid, FinishReason::Error).await?;
                        continue;
                    }
                }
            };
            to_prefill.push((rid.clone(), input_tokens, kv_handle));
        }

        if to_prefill.is_empty() {
            return Ok(());
        }

        // ── Phase 1b: ONE batched model_executor.batch_prefill call ──
        let inputs: Vec<PrefillInput> = to_prefill
            .iter()
            .map(|(_, tokens, kv)| {
                let token_u32s: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
                let tensor = self.tokens_to_tensor(&token_u32s)?;
                Ok(PrefillInput::new(tensor).with_kv_cache(kv.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        let outputs = self.model_executor.batch_prefill(&inputs).await?;
        if outputs.len() != to_prefill.len() {
            return Err(FerrumError::internal(format!(
                "batch_prefill returned {} outputs for {} inputs",
                outputs.len(),
                to_prefill.len(),
            )));
        }

        // ── Phase 1c: per-item post-process (sample, update seq, stream, stop) ──
        for ((rid, input_tokens, _), prefill_output) in to_prefill.iter().zip(outputs.iter()) {
            let num_tokens = input_tokens.len();
            let last_logits = prefill_output.last_token_logits()?;
            let logits_vec = last_logits.to_vec_f32()?;
            let _ = self.prefix_cache.store_prefix(
                input_tokens,
                prefill_output.kv_cache.clone(),
                logits_vec.clone(),
            );
            let first_token = {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(rid) else {
                    continue;
                };
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
            self.scheduler.mark_prefill_complete(rid, num_tokens);
            self.total_prefill_tokens
                .fetch_add(num_tokens as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
            counter!("ferrum.engine.prefills_total").increment(1);
            self.send_stream_update(rid, first_token).await;
            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(rid).is_none_or(|s| s.should_stop())
            };
            if should_stop {
                self.complete_request(rid, FinishReason::EOS).await?;
            }
        }
        Ok(())
    }

    // ── batch decode ──────────────────────────────────────────────────

    /// Run batch decode for multiple requests in a single forward pass.
    ///
    /// Dispatches via the unified-batch API: we build a `UnifiedBatch`
    /// of decode-only items (each `q_len = 1`, `is_final_chunk = true`)
    /// and call `model_executor.unified_decode(...)`. The default fallback
    /// in `LlmModelExecutor` recognises the all-decode shape and reroutes
    /// to the existing batched decode path; once `LlmFamilyModel` ships
    /// a real unified forward (Step 5), the same call benefits from the
    /// chunked-prefill kernel work without further engine changes.
    async fn run_batch_decode(&self, request_ids: &[RequestId]) -> Result<()> {
        use ferrum_interfaces::model_executor::{UnifiedBatch, UnifiedBatchItem};

        let rids: Vec<RequestId> = request_ids.to_vec();

        // Build the unified batch from sequence state.
        let mut batch = UnifiedBatch::new();
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
                // pos_offset = position of the NEW token. Compute from
                // engine bookkeeping (see process_batch_unified for why
                // `kv_cache.block_table().sequence_length` is not reliable
                // — Paged/Default handles never increment it).
                let pos_offset = seq.input_tokens.len() + seq.generated_tokens.len() - 1;
                // Use the model-side cache_id (set in `run_prefill_inner`
                // from `prefill_output.kv_cache.cache_id()`), NOT the
                // engine's request_id. The model's `kv_caches` is keyed
                // by the executor-generated id (e.g. "llm-cache-N"); using
                // the request_id (UUID) makes `ensure_kv` allocate a
                // fresh cache + 128 paged blocks for every decode iter,
                // exhausting the pool within ~60 prompts.
                let seq_id = seq
                    .model_cache_id
                    .clone()
                    .unwrap_or_else(|| rid.to_string());
                batch.items.push(UnifiedBatchItem {
                    seq_id,
                    q_tokens: vec![last_token.get()],
                    kv_cache,
                    pos_offset,
                    is_final_chunk: true,
                });
            }
        }

        let prof = std::env::var("FERRUM_RBD_PROF").is_ok();
        let t_decode = if prof { Some(Instant::now()) } else { None };
        let results = self.model_executor.unified_decode(&batch).await?;
        if results.len() != rids.len() {
            return Err(FerrumError::internal(format!(
                "unified_decode returned {} results for {} requests",
                results.len(),
                rids.len(),
            )));
        }
        let t_decode_done = if prof { Some(Instant::now()) } else { None };
        let mut t_sample_us: u64 = 0;
        let mut t_sched_us: u64 = 0;
        let mut t_stream_us: u64 = 0;
        let mut t_stop_us: u64 = 0;
        let mut t_complete_us: u64 = 0;

        // Per-item post-processing: sample, update sequence state, stream
        // the new token, check stop conditions. Decode-only items always
        // produce Some(logits); a None here would indicate a backend bug.
        for (rid, logits_opt) in rids.iter().zip(results.into_iter()) {
            let mut logits = logits_opt.ok_or_else(|| {
                FerrumError::internal(format!(
                    "unified_decode returned None for decode item (rid={rid})"
                ))
            })?;

            let t0_sample = if prof { Some(Instant::now()) } else { None };
            let next_token = {
                let mut sequences = self.sequences.write();
                let seq = sequences
                    .get_mut(rid)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                // Greedy fast path: the model did GPU argmax and emitted
                // one f32 carrying the token id (under `FERRUM_GREEDY_ARGMAX=1`).
                // Skip sample_with_processors entirely — at vocab=152064
                // (Qwen3), the host argmax scan is ~150 µs per item and
                // dominates the engine's per-iter overhead at c=32.
                let token = if logits.len() == 1 {
                    TokenId::new(logits[0] as u32)
                } else {
                    seq.sample_with_processors(&mut logits)?
                };
                seq.generated_tokens.push(token);
                seq.tokens_this_iteration += 1;
                // pos_offset is sourced from SequenceState bookkeeping
                // (see process_batch_unified). The engine-side KV handle's
                // sequence_length is not used for position tracking
                // anymore — production handles (Paged/Default) don't
                // update it across iterations.
                token
            };

            if let Some(t0) = t0_sample {
                t_sample_us += t0.elapsed().as_micros() as u64;
            }

            let t0_sched = if prof { Some(Instant::now()) } else { None };
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
            if let Some(t0) = t0_sched {
                t_sched_us += t0.elapsed().as_micros() as u64;
            }

            let t0_stream = if prof { Some(Instant::now()) } else { None };
            self.send_stream_update(rid, next_token).await;
            if let Some(t0) = t0_stream {
                t_stream_us += t0.elapsed().as_micros() as u64;
            }

            let t0_stop = if prof { Some(Instant::now()) } else { None };
            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(rid).is_none_or(|s| s.should_stop())
            };
            if let Some(t0) = t0_stop {
                t_stop_us += t0.elapsed().as_micros() as u64;
            }
            if should_stop {
                let t0_comp = if prof { Some(Instant::now()) } else { None };
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
                if let Some(t0) = t0_comp {
                    t_complete_us += t0.elapsed().as_micros() as u64;
                }
            }
        }

        if let (Some(t0), Some(t1)) = (t_decode, t_decode_done) {
            use std::sync::atomic::AtomicU64;
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(32) {
                let decode_us = t1.duration_since(t0).as_micros() as u64;
                let total_post_us =
                    t_sample_us + t_sched_us + t_stream_us + t_stop_us + t_complete_us;
                eprintln!(
                    "[rbd-prof] iter#{} m={} decode={}us post={}us | sample={} sched={} stream={} stop={} complete={} (us)",
                    n,
                    rids.len(),
                    decode_us,
                    total_post_us,
                    t_sample_us,
                    t_sched_us,
                    t_stream_us,
                    t_stop_us,
                    t_complete_us,
                );
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
            sequences.get(request_id).is_none_or(|s| s.should_stop())
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
        // length on partial rejection below. Both handles here are
        // `GenericKvCacheHandle` (constructed by `LlmExecutor::prefill` /
        // `decode`, NOT the engine `KvCacheManager`-allocated Paged/Default
        // handles used on the non-spec path), so `sequence_length` is
        // correctly updated each step via `with_sequence_length(new_seq)`.
        // Verified 2026-05-14 with FERRUM_DEBUG_SPEC_POS instrumentation:
        // entry_target_seq grows by N+1 per step, matching the actual
        // model writes (see make_kv_handle_with_seq at L1827-1828 for the
        // update site on the partial-reject path).
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
                sequences.get(request_id).map_or(true, |s| s.should_stop())
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
        // Decode the full generated-token history (skip_special=true matches
        // the final-response decode in `complete_request`) and emit only
        // the delta that hasn't been streamed yet. Per-token decode is
        // wrong for any model whose vocab can split a multi-byte UTF-8
        // sequence across BPE pieces — Qwen3 / Qwen2.5 routinely do this
        // for Chinese chars and emoji, and the single-token decode then
        // returns a `\u{FFFD}` replacement char that renders as a square /
        // `?` glyph in the terminal.
        //
        // Algorithm: hold the write lock once to (a) clone sender, (b)
        // decode current full history, (c) if the decoded text ends in
        // `\u{FFFD}` defer the emit (a later token will complete the
        // multi-byte sequence), (d) otherwise carve off the substring
        // past `streamed_text_len` and bump the watermark. Buffering is
        // bounded — the longest multi-byte sequence is 4 bytes, so at
        // most one or two tokens get deferred before flushing.
        let (sender, delta, ttft_s, itl_s) = {
            let mut sequences = self.sequences.write();
            let Some(seq) = sequences.get_mut(request_id) else {
                return;
            };
            let sender = seq.stream_sender.clone();
            let full = self
                .tokenizer
                .decode(&seq.generated_tokens, true)
                .unwrap_or_else(|_| format!("token_{}", token.get()));
            if full.ends_with('\u{FFFD}') {
                // Partial multi-byte UTF-8 at the tail; wait for the next
                // token. Do NOT advance streamed_text_len so the bytes get
                // re-considered once the sequence completes.
                return;
            }
            let delta = full[seq.streamed_text_len..].to_string();
            seq.streamed_text_len = full.len();

            // Latency-metric tracking (PLAYBOOK § 7 definitions).
            // We capture timestamps in the critical section so the
            // first-emit point matches the moment we commit to streaming
            // the delta — not the moment the chunk actually crosses the
            // socket, which the engine can't observe.
            let mut ttft_s: Option<f64> = None;
            let mut itl_s: Option<f64> = None;
            if !delta.is_empty() {
                let now = Instant::now();
                match seq.first_emit_at {
                    None => {
                        ttft_s = Some(now.duration_since(seq.start_time).as_secs_f64());
                        seq.first_emit_at = Some(now);
                    }
                    Some(_) => {
                        if let Some(prev) = seq.last_emit_at {
                            itl_s = Some(now.duration_since(prev).as_secs_f64());
                        }
                    }
                }
                seq.last_emit_at = Some(now);
                seq.emitted_chunks = seq.emitted_chunks.saturating_add(1);
            }

            (sender, delta, ttft_s, itl_s)
        };

        if let Some(t) = ttft_s {
            histogram!("ferrum.engine.ttft_seconds").record(t);
        }
        if let Some(t) = itl_s {
            histogram!("ferrum.engine.itl_seconds").record(t);
        }

        if let Some(tx) = sender {
            if delta.is_empty() {
                return;
            }
            let chunk = StreamChunk {
                request_id: request_id.clone(),
                text: delta,
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

                // TPOT histogram (PLAYBOOK § 7 definition):
                //   tpot = (e2e − ttft) / (output_tokens − 1)
                // Only meaningful when first_emit_at is set (i.e. at
                // least one stream chunk landed) and ≥ 2 chunks were
                // emitted to give a non-degenerate decode window.
                if let (Some(first), Some(last)) = (seq.first_emit_at, seq.last_emit_at) {
                    if seq.emitted_chunks >= 2 {
                        let decode_s = last.duration_since(first).as_secs_f64();
                        let tpot_s = decode_s / (seq.emitted_chunks - 1) as f64;
                        histogram!("ferrum.engine.tpot_seconds").record(tpot_s);
                    }
                }

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
                bg_loop_spawned: AtomicBool::new(false),
            }),
        }
    }

    /// Spawn the background iteration loop on first request. Without this,
    /// every concurrent infer/infer_stream call spawned its own
    /// drive_to_completion task → 16 streaming requests = 16 tasks all
    /// racing for `iteration_lock` (thundering herd, observed as ~5ms of
    /// per-iter tokio scheduling overhead at c=16). With one bg loop +
    /// per-request tasks just consuming their channel, lock is uncontested.
    fn ensure_bg_loop(&self) {
        if !self.inner.bg_loop_spawned.swap(true, Ordering::SeqCst) {
            let _ = self.start_loop();
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
            let prof = std::env::var("FERRUM_BATCH_DECODE_PROF").is_ok();
            let mut last_iter_end: Option<std::time::Instant> = None;
            static GAP_PROF_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            loop {
                if !inner.is_running.load(Ordering::SeqCst) {
                    break;
                }
                let inter_iter_us = if let Some(prev) = last_iter_end {
                    Some(prev.elapsed().as_micros() as u64)
                } else {
                    None
                };
                {
                    let _guard = inner.iteration_lock.lock().await;
                    if let Err(e) = inner.run_iteration().await {
                        warn!("Iteration error: {}", e);
                    }
                }
                if prof {
                    let n = GAP_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n.is_multiple_of(8) {
                        if let Some(gap_us) = inter_iter_us {
                            eprintln!("[bg-loop-gap] call#{} inter_iter={}us", n, gap_us);
                        }
                    }
                }
                last_iter_end = Some(std::time::Instant::now());
                tokio::task::yield_now().await;
            }
            info!("Background iteration loop stopped");
        })
    }
}

#[async_trait]
impl LlmInferenceEngine for ContinuousBatchEngine {
    async fn infer(&self, mut request: InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.id.clone();
        let infer_start = Instant::now();
        counter!("ferrum.engine.requests_total").increment(1);
        gauge!("ferrum.engine.active_requests").increment(1.0);

        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.to_string(),
            serde_json::Value::from(input_tokens.len() as u64),
        );

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with oneshot channel
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

        // Make sure the single shared bg loop is running, then just wait
        // for our oneshot to fire. Avoids per-request drive_to_completion
        // contention on iteration_lock.
        self.ensure_bg_loop();
        self.inner.work_notify.notify_one();

        let result = resp_rx
            .await
            .map_err(|_| FerrumError::internal("Response channel closed before response was sent"));

        gauge!("ferrum.engine.active_requests").decrement(1.0);
        let elapsed_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("ferrum.engine.request_duration_ms").record(elapsed_ms);

        if let Ok(ref resp) = result {
            counter!("ferrum.engine.requests_completed").increment(1);
            counter!("ferrum.engine.tokens_generated_total").increment(resp.tokens.len() as u64);
            // NOTE: real TTFT lives in `send_stream_update` —
            // emitted as `ferrum.engine.ttft_seconds`. The sync `infer`
            // path returns the whole response at once, so there's no
            // observable first-token moment to record here.
        } else {
            counter!("ferrum.engine.requests_failed").increment(1);
        }

        result
    }

    async fn infer_stream(
        &self,
        mut request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (tx, rx) = mpsc::channel(100);
        let request_id = request.id.clone();

        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.to_string(),
            serde_json::Value::from(input_tokens.len() as u64),
        );

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with stream sender
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

        // Single shared bg loop drives iters; per-request stream just
        // consumes from `rx`. Used to spawn a per-request drive_to_completion
        // task here, but with c=N concurrent streams that produced N
        // tasks all racing for `iteration_lock` — measured ~5ms/iter of
        // tokio thundering-herd overhead at c=16.
        let _ = request_id;
        self.ensure_bg_loop();
        self.inner.work_notify.notify_one();

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

#[async_trait]
impl InferenceEngine for ContinuousBatchEngine {
    async fn status(&self) -> EngineStatus {
        let metrics = self.inner.scheduler.metrics();
        EngineStatus {
            is_ready: self.inner.is_running.load(Ordering::SeqCst),
            loaded_models: vec![self.inner.config.model.model_id.clone()],
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
