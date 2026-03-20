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
use ferrum_scheduler::implementations::{ContinuousBatchScheduler, RequestPhase};
use ferrum_types::{
    DataType, Device, EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest,
    InferenceResponse, Priority, RequestId, Result, SamplingParams, StreamChunk, TokenId,
    TokenUsage,
};
use futures::stream::Stream;
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
}

impl SequenceState {
    pub fn new(request: InferenceRequest, input_tokens: Vec<TokenId>) -> Self {
        let seed = request.sampling_params.seed.unwrap_or(42);
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
}

// ────────────────────────────────────────────────────────────────────────────
// Engine inner – shared via Arc so we can spawn tasks
// ────────────────────────────────────────────────────────────────────────────

struct EngineInner {
    config: EngineConfig,
    scheduler: Arc<ContinuousBatchScheduler>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    sampler: Arc<dyn Sampler + Send + Sync>,
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    tensor_factory: Arc<dyn TensorFactory>,
    sequences: RwLock<HashMap<RequestId, SequenceState>>,
    is_running: AtomicBool,
    shutdown_notify: Arc<Notify>,
    /// Ensures only one iteration step runs at a time.
    iteration_lock: tokio::sync::Mutex<()>,
    /// Wakes callers or a background loop when new work is submitted.
    work_notify: Notify,
    // stats
    iteration_count: AtomicU64,
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
    total_preemptions: AtomicU64,
}

impl EngineInner {
    // ── tensor helper ──────────────────────────────────────────────────

    fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<TensorRef> {
        let f32_data: Vec<f32> = token_ids.iter().map(|&v| v as f32).collect();
        let len = f32_data.len();
        self.tensor_factory
            .from_slice(&f32_data, &[1, len], DataType::FP32, Device::CPU)
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
                    SequenceState::new(scheduled_req.request.clone(), input_tokens)
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

        // Decode continuing requests
        for rid in &decode_ids {
            if let Err(e) = self.run_decode_step(rid).await {
                warn!("Decode failed for {}: {}", rid, e);
                self.complete_request(rid, FinishReason::Error).await?;
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
                .filter(|(id, s)| {
                    *id != exclude_id && s.prefill_complete && s.kv_cache.is_some()
                })
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

        // Free KV cache
        let _ = self.kv_cache.deallocate(victim_id.clone()).await;

        // Reset sequence state — keep response/stream channels intact
        {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(&victim_id) {
                seq.kv_cache = None;
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
        true
    }

    // ── prefill ────────────────────────────────────────────────────────

    async fn run_prefill(&self, request_id: &RequestId) -> Result<()> {
        let (input_tensor, num_tokens) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let token_u32s: Vec<u32> = seq.input_tokens.iter().map(|t| t.get()).collect();
            (self.tokens_to_tensor(&token_u32s)?, seq.input_tokens.len())
        };

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

        let prefill_input = ferrum_interfaces::model_executor::PrefillInput::new(input_tensor)
            .with_kv_cache(kv_handle);
        let prefill_output = self.model_executor.prefill(&prefill_input).await?;

        let last_logits = prefill_output.last_token_logits()?;
        let logits_vec = last_logits.to_vec_f32()?;

        let first_token = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let token = self.sampler.sample(&logits_vec, &mut seq.rng)?;
            seq.generated_tokens.push(token);
            seq.kv_cache = Some(prefill_output.kv_cache.clone());
            seq.prefill_complete = true;
            seq.phase = RequestPhase::Decoding;
            token
        };

        self.scheduler.mark_prefill_complete(request_id, num_tokens);
        self.total_prefill_tokens
            .fetch_add(num_tokens as u64, Ordering::Relaxed);

        debug!(
            "Prefill complete for {}: {} prompt tokens, first generated: {}",
            request_id,
            num_tokens,
            first_token.get()
        );

        self.send_stream_update(request_id, first_token).await;

        let should_stop = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map_or(true, |s| s.should_stop(self.model_executor.info().vocab_size))
        };
        if should_stop {
            self.complete_request(request_id, FinishReason::EOS).await?;
        }

        Ok(())
    }

    // ── decode step ────────────────────────────────────────────────────

    async fn run_decode_step(&self, request_id: &RequestId) -> Result<()> {
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
            let token = self.sampler.sample(&logits_vec, &mut seq.rng)?;
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

        self.send_stream_update(request_id, next_token).await;

        let should_stop = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map_or(true, |s| s.should_stop(self.model_executor.info().vocab_size))
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
        let (response, stream_sender, response_sender, has_kv_cache) = {
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
                (response, seq.stream_sender, seq.response_sender, has_kv)
            } else {
                return Ok(());
            }
        };

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
        info!("Creating ContinuousBatchEngine");

        Self {
            inner: Arc::new(EngineInner {
                config,
                scheduler,
                tokenizer,
                sampler,
                kv_cache,
                model_executor,
                tensor_factory,
                sequences: RwLock::new(HashMap::new()),
                is_running: AtomicBool::new(false),
                shutdown_notify: Arc::new(Notify::new()),
                iteration_lock: tokio::sync::Mutex::new(()),
                work_notify: Notify::new(),
                iteration_count: AtomicU64::new(0),
                total_prefill_tokens: AtomicU64::new(0),
                total_decode_tokens: AtomicU64::new(0),
                total_preemptions: AtomicU64::new(0),
            }),
        }
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

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with oneshot channel
        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut seq_state = SequenceState::new(request, input_tokens);
        seq_state.response_sender = Some(resp_tx);
        self.inner
            .sequences
            .write()
            .insert(request_id.clone(), seq_state);

        // Notify any background loop
        self.inner.work_notify.notify_one();

        // Drive iterations until our request completes
        self.inner.drive_to_completion(&request_id).await?;

        resp_rx.await.map_err(|_| {
            FerrumError::internal("Response channel closed before response was sent")
        })
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
        let mut seq_state = SequenceState::new(request, input_tokens);
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
            .field(
                "is_running",
                &self.inner.is_running.load(Ordering::SeqCst),
            )
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
