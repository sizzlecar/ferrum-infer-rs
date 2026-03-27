//! Main inference engine - MVP implementation

use async_trait::async_trait;
use ferrum_interfaces::sampler::{SamplingConfigBuilder, SamplingContext};
use ferrum_interfaces::{
    engine::InferenceEngine, AllocationRequest, BatchHint, KvCacheHandle, KvCacheManager,
    ModelExecutor, Sampler, SchedulerInterface as Scheduler, Tokenizer,
};
use ferrum_types::{
    EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest, InferenceResponse,
    RequestId, RequestState, Result, SamplingParams, StreamChunk, TokenId, TokenUsage,
};
use futures::stream::Stream;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use rand::RngCore;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tracing::{debug, info, warn};

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use std::sync::OnceLock;

/// Default inference engine - MVP implementation
pub struct DefaultInferenceEngine {
    config: EngineConfig,
    scheduler: Arc<dyn Scheduler + Send + Sync>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    sampler: Arc<dyn Sampler + Send + Sync>,
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    concurrency_limiter: Arc<Semaphore>,
    active_requests: Arc<AtomicUsize>,
    waiting_requests: Arc<AtomicUsize>,
    effective_max_running_requests: usize,
    serialized_executor: bool,
}

struct InflightGuard {
    _permit: OwnedSemaphorePermit,
    active_requests: Arc<AtomicUsize>,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.active_requests.fetch_sub(1, Ordering::SeqCst);
    }
}

fn spawn_cleanup_task<F>(fut: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            handle.spawn(fut);
        }
        Err(e) => {
            warn!(
                "Skipping async cleanup task because no Tokio runtime is active: {}",
                e
            );
        }
    }
}

struct SchedulerFinalizationGuard {
    request_id: RequestId,
    scheduler: Arc<dyn Scheduler + Send + Sync>,
    admitted: bool,
    finalized: bool,
}

impl SchedulerFinalizationGuard {
    fn new(request_id: RequestId, scheduler: Arc<dyn Scheduler + Send + Sync>) -> Self {
        Self {
            request_id,
            scheduler,
            admitted: false,
            finalized: false,
        }
    }

    fn mark_admitted(&mut self) {
        self.admitted = true;
    }

    async fn complete_now(&mut self, response: &InferenceResponse) -> Result<()> {
        self.scheduler
            .complete(self.request_id.clone(), response)
            .await?;
        self.finalized = true;
        Ok(())
    }

    async fn cancel_now(&mut self) -> Result<()> {
        if self.admitted {
            self.scheduler.cancel(self.request_id.clone()).await?;
        }
        self.finalized = true;
        Ok(())
    }
}

impl Drop for SchedulerFinalizationGuard {
    fn drop(&mut self) {
        if !self.admitted || self.finalized {
            return;
        }

        let request_id = self.request_id.clone();
        let scheduler = self.scheduler.clone();
        spawn_cleanup_task(async move {
            if let Err(e) = scheduler.cancel(request_id.clone()).await {
                warn!(
                    "Best-effort cleanup failed to cancel request {} in scheduler: {}",
                    request_id, e
                );
            }
        });
    }
}

struct KvReservationGuard {
    request_id: RequestId,
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    reserved: bool,
    cleaned: bool,
}

impl KvReservationGuard {
    fn new(request_id: RequestId, kv_cache: Arc<dyn KvCacheManager + Send + Sync>) -> Self {
        Self {
            request_id,
            kv_cache,
            reserved: false,
            cleaned: false,
        }
    }

    fn mark_reserved(&mut self) {
        self.reserved = true;
    }

    async fn deallocate_now(&mut self, context: &str) {
        if !self.reserved || self.cleaned {
            return;
        }

        if let Err(e) = self.kv_cache.deallocate(self.request_id.clone()).await {
            warn!(
                "Failed to deallocate KV cache for request {} ({}): {}",
                self.request_id, context, e
            );
        }
        self.cleaned = true;
    }
}

impl Drop for KvReservationGuard {
    fn drop(&mut self) {
        if !self.reserved || self.cleaned {
            return;
        }

        let request_id = self.request_id.clone();
        let kv_cache = self.kv_cache.clone();
        spawn_cleanup_task(async move {
            if let Err(e) = kv_cache.deallocate(request_id.clone()).await {
                warn!(
                    "Best-effort cleanup failed to deallocate KV cache for request {}: {}",
                    request_id, e
                );
            }
        });
    }
}

impl DefaultInferenceEngine {
    pub fn new(
        config: EngineConfig,
        scheduler: Arc<dyn Scheduler + Send + Sync>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    ) -> Self {
        let configured_max_running = config.scheduler.max_running_requests.max(1);
        // Use FERRUM_MAX_RUNNING env var to override max concurrent requests.
        let effective_max_running_requests = std::env::var("FERRUM_MAX_RUNNING")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(configured_max_running)
            .max(1);

        info!(
            "Created DefaultInferenceEngine (configured_max_running_requests={}, effective_max_running_requests={})",
            configured_max_running,
            effective_max_running_requests
        );

        Self {
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
            concurrency_limiter: Arc::new(Semaphore::new(effective_max_running_requests)),
            active_requests: Arc::new(AtomicUsize::new(0)),
            waiting_requests: Arc::new(AtomicUsize::new(0)),
            effective_max_running_requests,
            serialized_executor: effective_max_running_requests <= 1,
        }
    }

    async fn admit_request(&self, request: &InferenceRequest) -> Result<()> {
        let request_id = self.scheduler.submit(request.clone()).await?;
        let max_batch_size = self.effective_max_running_requests.max(1);
        let per_request_tokens = request.sampling_params.max_tokens.max(1);
        let hint = BatchHint {
            max_batch_size,
            max_tokens: per_request_tokens.saturating_mul(max_batch_size),
            target_latency_ms: Some(0),
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let admission_deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            match self.scheduler.request_state(&request_id) {
                Some(RequestState::Running) => return Ok(()),
                Some(RequestState::Cancelled) => {
                    return Err(FerrumError::scheduler(format!(
                        "Scheduler cancelled request {} during admission",
                        request_id
                    )))
                }
                Some(RequestState::Failed) => {
                    return Err(FerrumError::scheduler(format!(
                        "Scheduler failed request {} during admission",
                        request_id
                    )))
                }
                Some(RequestState::Completed) => {
                    return Err(FerrumError::scheduler(format!(
                        "Scheduler completed request {} before execution started",
                        request_id
                    )))
                }
                Some(RequestState::Waiting | RequestState::Preempted) | None => {}
            }

            if let Some(batch) = self.scheduler.next_batch(hint.clone()).await {
                if batch.requests.iter().any(|r| r.request.id == request_id) {
                    return Ok(());
                }
            }

            if let Some(RequestState::Running) = self.scheduler.request_state(&request_id) {
                return Ok(());
            }

            if tokio::time::Instant::now() >= admission_deadline {
                break;
            }
            tokio::task::yield_now().await;
        }

        if let Err(cancel_err) = self.scheduler.cancel(request_id.clone()).await {
            warn!(
                "Failed to rollback scheduler state for request {} after admission timeout: {}",
                request_id, cancel_err
            );
            return Err(FerrumError::scheduler(format!(
                "Timed out admitting request {} and rollback cancel also failed: {}",
                request_id, cancel_err
            )));
        }

        Err(FerrumError::timeout(format!(
            "Timed out admitting request {} after 10s",
            request_id
        )))
    }

    async fn reserve_request_kv_cache_with(
        kv_cache: &Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: &Arc<dyn ModelExecutor + Send + Sync>,
        request: &InferenceRequest,
        prompt_tokens: usize,
    ) -> Result<()> {
        let info = model_executor.info();
        let num_heads = info.num_heads.max(1);
        let head_dim = (info.hidden_size / num_heads).max(1);
        let allocation = AllocationRequest {
            request_id: request.id.clone(),
            initial_tokens: prompt_tokens,
            max_sequence_length: prompt_tokens
                .saturating_add(request.sampling_params.max_tokens)
                .max(1),
            num_layers: info.num_layers.max(1),
            num_heads,
            head_dim,
            device: info.device.clone(),
            dtype: info.dtype,
            priority: request.priority,
        };

        if !kv_cache.can_allocate(&allocation) {
            return Err(FerrumError::resource_exhausted(format!(
                "KV cache cannot allocate for request {}",
                request.id
            )));
        }

        kv_cache
            .allocate(&allocation)
            .await
            .map(|_| ())
            .map_err(|e| {
                FerrumError::resource_exhausted(format!(
                    "KV cache allocation failed for request {}: {}",
                    request.id, e
                ))
            })
    }

    async fn reserve_request_kv_cache(
        &self,
        request: &InferenceRequest,
        prompt_tokens: usize,
    ) -> Result<()> {
        Self::reserve_request_kv_cache_with(
            &self.kv_cache,
            &self.model_executor,
            request,
            prompt_tokens,
        )
        .await
    }

    async fn acquire_inflight_guard(&self) -> Result<InflightGuard> {
        self.waiting_requests.fetch_add(1, Ordering::SeqCst);
        let permit = self
            .concurrency_limiter
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| FerrumError::scheduler("Inference engine concurrency limiter closed"))?;
        self.waiting_requests.fetch_sub(1, Ordering::SeqCst);

        self.active_requests.fetch_add(1, Ordering::SeqCst);
        Ok(InflightGuard {
            _permit: permit,
            active_requests: self.active_requests.clone(),
        })
    }

    /// Execute single inference request
    async fn execute_request(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.id.clone();
        debug!("Executing request: {:?}", request_id);
        let mut kv_guard = KvReservationGuard::new(request.id.clone(), self.kv_cache.clone());

        // 1. Tokenize prompt
        let input_tokens = self.tokenizer.encode(&request.prompt, true)?;
        let prompt_tokens = input_tokens.len();

        debug!("Encoded {} tokens from prompt", prompt_tokens);

        // Reserve KV cache resources for this request lifecycle.
        self.reserve_request_kv_cache(request, prompt_tokens)
            .await?;
        kv_guard.mark_reserved();

        let result = async {
            // 2. Prepare prefill input
            let device = &self.model_executor.info().device;
            let prefill_input = create_prefill_input(&input_tokens, device)?;

            // 3. Execute prefill
            let prefill_output = self.model_executor.prefill(&prefill_input).await?;

            // 4. Generate tokens (decode loop)
            let max_tokens = request.sampling_params.max_tokens;
            let mut generated_tokens = Vec::new();
            let mut all_tokens: Vec<TokenId> = input_tokens.clone();
            let mut kv_cache = prefill_output.kv_cache.clone();
            let mut rng = create_rng(&request.sampling_params);
            let mut stop_reason: Option<FinishReason> = None;

            for step in 0..max_tokens {
                // Get logits from last position
                let logits = if step == 0 {
                    extract_last_token_logits(&prefill_output.logits)?
                } else {
                    // Decode step
                    let decode_input =
                        create_decode_input(&generated_tokens, kv_cache.clone(), device)?;
                    let decode_output = self.model_executor.decode(&decode_input).await?;
                    kv_cache = decode_output.kv_cache.clone();
                    extract_last_token_logits(&decode_output.logits)?
                };

                // Sample next token (use generated_tokens only for repetition penalty
                // so prompt tokens like <|im_end|> are not penalised)
                let next_token = sample_token(
                    &logits,
                    &request.sampling_params,
                    &self.sampler,
                    &mut rng,
                    &generated_tokens,
                )?;

                // Check stop conditions
                if is_stop_token(next_token, self.model_executor.info().vocab_size) {
                    debug!("Hit EOS token at step {}", step);
                    stop_reason = Some(FinishReason::EOS);
                    break;
                }

                generated_tokens.push(next_token);
                all_tokens.push(next_token);

                // Check stop sequences
                if check_stop_sequences(
                    &generated_tokens,
                    &request.sampling_params,
                    &self.tokenizer,
                )? {
                    debug!("Hit stop sequence at step {}", step);
                    stop_reason = Some(FinishReason::Stop);
                    break;
                }
            }

            // 5. Decode output tokens
            let generated_text = self.tokenizer.decode(&generated_tokens, true)?;

            debug!(
                "Generated {} tokens: {}",
                generated_tokens.len(),
                generated_text
            );

            // 6. Build response
            Ok(InferenceResponse {
                request_id,
                text: generated_text,
                tokens: generated_tokens.clone(),
                finish_reason: determine_finish_reason(
                    stop_reason,
                    generated_tokens.len(),
                    max_tokens,
                ),
                usage: TokenUsage::new(prompt_tokens, generated_tokens.len()),
                latency_ms: 0, // TODO: measure actual latency
                created_at: chrono::Utc::now(),
                metadata: std::collections::HashMap::new(),
            })
        }
        .await;

        kv_guard.deallocate_now("sync-infer").await;

        result
    }
}

#[async_trait]
impl InferenceEngine for DefaultInferenceEngine {
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        request.sampling_params.validate()?;
        let _inflight_guard = self.acquire_inflight_guard().await?;
        let mut scheduler_guard =
            SchedulerFinalizationGuard::new(request.id.clone(), self.scheduler.clone());
        self.admit_request(&request).await?;
        scheduler_guard.mark_admitted();
        let result = self.execute_request(&request).await;

        match result {
            Ok(response) => {
                if let Err(complete_err) = scheduler_guard.complete_now(&response).await {
                    warn!(
                        "Failed to mark request {} complete in scheduler: {}",
                        request.id, complete_err
                    );
                    if let Err(cancel_err) = scheduler_guard.cancel_now().await {
                        return Err(FerrumError::scheduler(format!(
                            "Scheduler completion failed for request {}: {}. Rollback cancel also failed: {}",
                            request.id, complete_err, cancel_err
                        )));
                    }

                    return Err(FerrumError::scheduler(format!(
                        "Scheduler completion failed for request {}: {}",
                        request.id, complete_err
                    )));
                }

                Ok(response)
            }
            Err(e) => {
                warn!("Request {} failed: {}", request.id, e);
                if let Err(cancel_err) = scheduler_guard.cancel_now().await {
                    return Err(FerrumError::scheduler(format!(
                        "Request {} failed with error '{}', and scheduler cancel failed: {}",
                        request.id, e, cancel_err
                    )));
                }

                Err(e)
            }
        }
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        request.sampling_params.validate()?;
        let inflight_guard = self.acquire_inflight_guard().await?;
        let mut scheduler_guard =
            SchedulerFinalizationGuard::new(request.id.clone(), self.scheduler.clone());
        self.admit_request(&request).await?;
        scheduler_guard.mark_admitted();
        let (tx, rx) = mpsc::channel(100);

        // Clone components for async task
        let tokenizer = self.tokenizer.clone();
        let sampler = self.sampler.clone();
        let model_executor = self.model_executor.clone();
        let kv_cache = self.kv_cache.clone();
        let request_id = request.id.clone();
        let max_tokens = request.sampling_params.max_tokens;
        let sampling_params = request.sampling_params.clone();

        tokio::spawn(async move {
            let _inflight_guard = inflight_guard;
            let mut scheduler_guard = scheduler_guard;
            let mut kv_guard = KvReservationGuard::new(request_id.clone(), kv_cache.clone());
            let mut last_cache_id = String::new();

            let generation_result: Result<InferenceResponse> = async {
                // 1. Tokenize
                let input_tokens = tokenizer.encode(&request.prompt, true)?;
                let prompt_tokens = input_tokens.len();

                // Reserve KV cache resources for the full stream lifecycle.
                DefaultInferenceEngine::reserve_request_kv_cache_with(
                    &kv_cache,
                    &model_executor,
                    &request,
                    prompt_tokens,
                )
                .await?;
                kv_guard.mark_reserved();

                // 2. Prefill
                let prefill_input =
                    create_prefill_input(&input_tokens, &model_executor.info().device)?;
                let prefill_output = model_executor.prefill(&prefill_input).await?;

                // 3. Decode loop - stream each token
                let mut generated_tokens = Vec::new();
                let mut all_tokens: Vec<TokenId> = input_tokens.iter().copied().collect();
                let mut model_kv_cache = prefill_output.kv_cache.clone();
                last_cache_id = model_kv_cache.cache_id();
                let mut rng = create_rng(&sampling_params);
                let mut stop_reason: Option<FinishReason> = None;

                for step in 0..max_tokens {
                    let logits = if step == 0 {
                        extract_last_token_logits(&prefill_output.logits)?
                    } else {
                        let decode_input = create_decode_input(
                            &generated_tokens,
                            model_kv_cache.clone(),
                            &model_executor.info().device,
                        )?;
                        let decode_output = model_executor.decode(&decode_input).await?;
                        model_kv_cache = decode_output.kv_cache.clone();
                        extract_last_token_logits(&decode_output.logits)?
                    };

                    let next_token = sample_token(
                        &logits,
                        &sampling_params,
                        &sampler,
                        &mut rng,
                        &generated_tokens,
                    )?;

                    if is_stop_token(next_token, model_executor.info().vocab_size) {
                        stop_reason = Some(FinishReason::EOS);
                        break;
                    }

                    generated_tokens.push(next_token);
                    all_tokens.push(next_token);

                    // Decode token to text
                    let token_text = match tokenizer.decode(&[next_token], false) {
                        Ok(text) => text,
                        Err(_) => format!("token_{}", next_token.get()),
                    };

                    // Send streaming chunk
                    let chunk = StreamChunk {
                        request_id: request_id.clone(),
                        text: token_text,
                        token: Some(next_token),
                        finish_reason: None,
                        usage: None,
                        created_at: chrono::Utc::now(),
                        metadata: HashMap::new(),
                    };

                    if tx.send(Ok(chunk)).await.is_err() {
                        return Err(FerrumError::cancelled(format!(
                            "Streaming consumer dropped for request {}",
                            request_id
                        )));
                    }

                    // Check stop sequences
                    if check_stop_sequences(&generated_tokens, &sampling_params, &tokenizer)? {
                        stop_reason = Some(FinishReason::Stop);
                        break;
                    }
                }

                let finish_reason =
                    determine_finish_reason(stop_reason, generated_tokens.len(), max_tokens);
                let generated_len = generated_tokens.len();
                let generated_text = tokenizer.decode(&generated_tokens, true)?;

                Ok(InferenceResponse {
                    request_id: request_id.clone(),
                    text: generated_text,
                    tokens: generated_tokens,
                    finish_reason,
                    usage: TokenUsage::new(prompt_tokens, generated_len),
                    latency_ms: 0, // TODO: measure actual latency
                    created_at: chrono::Utc::now(),
                    metadata: HashMap::new(),
                })
            }
            .await;

            kv_guard.deallocate_now("stream-infer").await;

            match &generation_result {
                Ok(_) => {
                    // Release model executor's KV cache (frees CUDA runner paged blocks)
                    model_executor.release_cache(&last_cache_id);
                }
                Err(_) => {}
            }

            match generation_result {
                Ok(response) => {
                    if let Err(complete_err) = scheduler_guard.complete_now(&response).await {
                        warn!(
                            "Failed to mark streaming request {} complete in scheduler: {}",
                            request_id, complete_err
                        );
                        if let Err(cancel_err) = scheduler_guard.cancel_now().await {
                            warn!(
                                "Failed to rollback streaming request {} after completion failure: {}",
                                request_id, cancel_err
                            );
                        }
                        let _ = tx
                            .send(Err(FerrumError::scheduler(format!(
                                "Scheduler completion failed for request {}: {}",
                                request_id, complete_err
                            ))))
                            .await;
                        return;
                    }

                    let final_chunk = StreamChunk {
                        request_id: request_id.clone(),
                        text: String::new(),
                        token: None,
                        finish_reason: Some(response.finish_reason),
                        usage: Some(response.usage.clone()),
                        created_at: chrono::Utc::now(),
                        metadata: HashMap::new(),
                    };

                    if tx.send(Ok(final_chunk)).await.is_err() {
                        warn!(
                            "Streaming consumer disconnected before final chunk for request {}",
                            request_id
                        );
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e.clone())).await;
                    if let Err(cancel_err) = scheduler_guard.cancel_now().await {
                        warn!(
                            "Failed to cancel request {} after stream error: {}",
                            request_id, cancel_err
                        );
                    }
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn status(&self) -> EngineStatus {
        EngineStatus {
            is_ready: true,
            loaded_models: vec![self.config.model.model_id.clone()],
            active_requests: self.active_requests.load(Ordering::Relaxed),
            queued_requests: self.waiting_requests.load(Ordering::Relaxed),
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
        info!(
            "Shutting down engine (effective_max_running_requests={}, serialized_executor={})",
            self.effective_max_running_requests, self.serialized_executor
        );
        Ok(())
    }

    fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn metrics(&self) -> ferrum_types::EngineMetrics {
        ferrum_types::EngineMetrics {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_request_latency_ms: 0.0,
            p95_request_latency_ms: 0.0,
            p99_request_latency_ms: 0.0,
            throughput_rps: 0.0,
            tokens_per_second: 0.0,
            queue_metrics: Default::default(),
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: Default::default(),
        }
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        ferrum_types::HealthStatus::healthy()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_prefill_input(
    tokens: &[TokenId],
    _device: &ferrum_types::Device,
) -> Result<ferrum_interfaces::model_executor::PrefillInput> {
    use candle_core::{Device as CandleDevice, Tensor};
    use ferrum_models::tensor_wrapper::CandleTensorWrapper;

    // Convert token IDs to u32 vector
    let token_u32s: Vec<u32> = tokens.iter().map(|t| t.get()).collect();

    // Create Candle tensor
    let tensor = Tensor::new(&token_u32s[..], &CandleDevice::Cpu)
        .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))?
        .unsqueeze(0) // Add batch dimension: [seq_len] -> [1, seq_len]
        .map_err(|e| FerrumError::model(format!("Failed to unsqueeze: {}", e)))?;

    // Wrap as TensorRef
    let tensor_ref = Arc::new(CandleTensorWrapper::new(tensor));

    Ok(ferrum_interfaces::model_executor::PrefillInput::new(
        tensor_ref,
    ))
}

fn create_decode_input(
    tokens: &[TokenId],
    kv_cache: Arc<dyn KvCacheHandle>,
    _device: &ferrum_types::Device,
) -> Result<ferrum_interfaces::model_executor::DecodeInput> {
    use candle_core::{Device as CandleDevice, Tensor};
    use ferrum_models::tensor_wrapper::CandleTensorWrapper;

    // Get last token
    let last_token = tokens.last().copied().unwrap_or(TokenId::new(0));

    // Create Candle tensor for single token
    let tensor = Tensor::new(&[last_token.get()], &CandleDevice::Cpu)
        .map_err(|e| FerrumError::model(format!("Failed to create tensor: {}", e)))?
        .unsqueeze(0) // Add batch dimension: [1] -> [1, 1]
        .map_err(|e| FerrumError::model(format!("Failed to unsqueeze: {}", e)))?;

    // Wrap as TensorRef
    let tensor_ref = Arc::new(CandleTensorWrapper::new(tensor));

    Ok(ferrum_interfaces::model_executor::DecodeInput::new(
        tensor_ref, kv_cache,
    ))
}

fn extract_last_token_logits(
    logits: &ferrum_interfaces::TensorRef,
) -> Result<ferrum_interfaces::TensorRef> {
    let shape = logits.shape();
    if let Some(candle_tensor) = logits
        .as_any()
        .downcast_ref::<ferrum_models::tensor_wrapper::CandleTensorWrapper>()
    {
        use candle_core::IndexOp;
        let inner = candle_tensor.inner();
        let extracted = match shape.len() {
            1 => inner.clone(),
            2 => {
                let batch = shape[0];
                let vocab = shape[1];
                if batch == 0 || vocab == 0 {
                    return Err(FerrumError::model("Invalid 2D logits shape"));
                }
                inner
                    .i(batch - 1)
                    .map_err(|e| FerrumError::model(format!("Index 2D logits failed: {}", e)))?
            }
            3 => {
                let batch = shape[0];
                let seq = shape[1];
                let vocab = shape[2];
                if batch == 0 || seq == 0 || vocab == 0 {
                    return Err(FerrumError::model("Invalid 3D logits shape"));
                }
                inner
                    .i((batch - 1, seq - 1))
                    .map_err(|e| FerrumError::model(format!("Index 3D logits failed: {}", e)))?
            }
            4 => {
                let batch = shape[0];
                let seq = shape[1];
                let extra = shape[2];
                let vocab = shape[3];
                if batch == 0 || seq == 0 || extra == 0 || vocab == 0 {
                    return Err(FerrumError::model("Invalid 4D logits shape"));
                }
                inner
                    .i((batch - 1, seq - 1, 0))
                    .map_err(|e| FerrumError::model(format!("Index 4D logits failed: {}", e)))?
            }
            _ => {
                return Err(FerrumError::model(format!(
                    "Unsupported logits rank {}, expected 1D/2D/3D/4D",
                    shape.len()
                )))
            }
        };

        return Ok(Arc::new(
            ferrum_models::tensor_wrapper::CandleTensorWrapper::new(extracted),
        ));
    }

    match shape.len() {
        1..=4 => Ok(logits.clone()),
        _ => Err(FerrumError::model(format!(
            "Unsupported logits rank {}, expected 1D/2D/3D/4D",
            shape.len()
        ))),
    }
}

fn sample_token(
    logits: &ferrum_interfaces::TensorRef,
    params: &SamplingParams,
    sampler: &Arc<dyn Sampler + Send + Sync>,
    rng: &mut StdRng,
    all_tokens: &[TokenId],
) -> Result<TokenId> {
    // Fast path (greedy): avoid transferring full vocab logits to CPU.
    // If the backend provides an on-device argmax, we only read back 1 scalar token id.
    if params.temperature == 0.0 || (sampler.is_deterministic() && sampler.name() == "greedy") {
        if let Ok(idx) = logits.argmax_last_dim_u32() {
            return Ok(TokenId::new(idx));
        }
    }

    // Metal GPU-side sampling path (top-k/top-p/temperature/repetition penalty).
    // This avoids copying full-vocab logits to CPU on every token.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        if let Some(tok) = try_sample_token_metal_gpu(logits, params, rng, all_tokens) {
            return Ok(tok);
        }
    }

    // Fallback: transfer logits to CPU and run the full sampling pipeline so
    // top-k/top-p/repetition/presence/frequency penalties are respected.
    let mut logits_vec = logits.to_vec_f32()?;
    let token_frequencies = build_token_frequencies(all_tokens);
    apply_presence_frequency_penalties(
        &mut logits_vec,
        &token_frequencies,
        params.presence_penalty,
        params.frequency_penalty,
    );

    let mut builder = SamplingConfigBuilder::new()
        .with_temperature(params.temperature)
        .with_repetition_penalty(params.repetition_penalty)
        .with_sampler(Box::new(ArcSampler::new(sampler.clone())));
    if let Some(top_k) = params.top_k {
        builder = builder.with_top_k(top_k);
    }
    if params.top_p < 1.0 {
        builder = builder.with_top_p(params.top_p);
    }
    let config = builder.build();

    let vocab_size = logits_vec.len();
    let step = all_tokens.len();
    let ctx = SamplingContext::new(
        step,
        params,
        &mut logits_vec,
        all_tokens,
        &token_frequencies,
        vocab_size,
    );

    config.sample(ctx, rng)
}

struct ArcSampler {
    inner: Arc<dyn Sampler + Send + Sync>,
}

impl ArcSampler {
    fn new(inner: Arc<dyn Sampler + Send + Sync>) -> Self {
        Self { inner }
    }
}

impl Sampler for ArcSampler {
    fn sample(&self, logits: &[f32], rng: &mut dyn rand::RngCore) -> Result<TokenId> {
        self.inner.sample(logits, rng)
    }

    fn sample_with_context(
        &self,
        ctx: &SamplingContext,
        rng: &mut dyn rand::RngCore,
    ) -> Result<TokenId> {
        self.inner.sample_with_context(ctx, rng)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn is_deterministic(&self) -> bool {
        self.inner.is_deterministic()
    }
}

fn build_token_frequencies(tokens: &[TokenId]) -> HashMap<TokenId, usize> {
    let mut frequencies = HashMap::new();
    for &token in tokens {
        *frequencies.entry(token).or_insert(0) += 1;
    }
    frequencies
}

fn apply_presence_frequency_penalties(
    logits: &mut [f32],
    token_frequencies: &HashMap<TokenId, usize>,
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    if presence_penalty == 0.0 && frequency_penalty == 0.0 {
        return;
    }

    for (&token_id, &freq) in token_frequencies {
        let idx = token_id.get() as usize;
        if idx < logits.len() {
            let penalty = presence_penalty + (frequency_penalty * freq as f32);
            logits[idx] -= penalty;
        }
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn try_sample_token_metal_gpu(
    logits: &ferrum_interfaces::TensorRef,
    params: &SamplingParams,
    rng: &mut StdRng,
    all_tokens: &[TokenId],
) -> Option<TokenId> {
    use ferrum_models::tensor_wrapper::CandleTensorWrapper;
    use ferrum_types::Device;

    if !matches!(logits.device(), Device::Metal) {
        return None;
    }

    // Opt-in switch.
    //
    // Default is OFF because the current GPU sampling implementation uses multiple
    // argmax passes (Top-K via repeated argmax+mask), which can be slower than
    // CPU-side sampling for small models.
    //
    // Enable explicitly when you need to avoid full-vocab GPU→CPU logits transfer:
    //   FERRUM_METAL_GPU_SAMPLING=1
    let enabled = std::env::var("FERRUM_METAL_GPU_SAMPLING")
        .map(|v| v == "1")
        .unwrap_or(false);
    if !enabled {
        return None;
    }

    // Only handle stochastic sampling here (greedy handled above).
    if params.temperature <= 0.0 {
        return None;
    }

    // We approximate nucleus sampling by sampling within Top-K candidates.
    let default_k: usize = std::env::var("FERRUM_METAL_TOPK_DEFAULT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(128);
    let k = params.top_k.unwrap_or(default_k).clamp(1, 256);

    // Prepare repetition penalty sparse list (token_id, freq) over a limited window.
    let rep_pen = params.repetition_penalty;
    let mut rep_ids: Vec<u32> = Vec::new();
    let mut rep_freqs: Vec<u32> = Vec::new();
    if rep_pen != 1.0 && !all_tokens.is_empty() {
        let window: usize = std::env::var("FERRUM_REPETITION_WINDOW")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let start = all_tokens.len().saturating_sub(window);
        let mut map = std::collections::HashMap::<u32, u32>::new();
        for t in &all_tokens[start..] {
            *map.entry(t.get()).or_insert(0) += 1;
        }
        rep_ids = map.keys().copied().collect();
        rep_freqs = rep_ids.iter().map(|id| map[id]).collect();
    }

    static OPS: OnceLock<crate::metal::MetalSamplingOps> = OnceLock::new();
    let ops = OPS.get_or_init(|| {
        let mut ctx =
            crate::metal::MetalContext::new().expect("MetalContext::new failed for sampling");
        ctx.load_shader_library()
            .expect("load_shader_library failed for sampling");
        crate::metal::MetalSamplingOps::new(std::sync::Arc::new(ctx))
            .expect("MetalSamplingOps::new failed")
    });

    let ctw = logits.as_any().downcast_ref::<CandleTensorWrapper>()?;
    let logits_tensor = ctw.inner();

    let seed = rng.next_u32();
    let top_p = params.top_p;

    match ops.sample_token(
        logits_tensor,
        k,
        top_p,
        params.temperature,
        rep_pen,
        &rep_ids,
        &rep_freqs,
        seed,
    ) {
        Ok(tok) => Some(TokenId::new(tok)),
        Err(e) => {
            debug!(
                "Metal GPU sampling failed, falling back to CPU sampler: {}",
                e
            );
            None
        }
    }
}

fn create_rng(params: &SamplingParams) -> StdRng {
    if let Some(seed) = params.seed {
        StdRng::seed_from_u64(seed)
    } else {
        let mut os_rng = rand::rng();
        StdRng::from_rng(&mut os_rng)
    }
}

fn is_stop_token(token: TokenId, _vocab_size: usize) -> bool {
    let token_id = token.get();
    // Common EOS token IDs for various models:
    // - Qwen2.5: <|endoftext|>=151643, <|im_end|>=151645
    // - LLaMA: </s>=2
    // - GPT-2/GPT-J: <|endoftext|>=50256
    matches!(
        token_id,
        2 |          // LLaMA </s>
        50256 |      // GPT-2/GPT-J <|endoftext|>
        151643 |     // Qwen <|endoftext|>
        151645 // Qwen <|im_end|>
    )
}

fn check_stop_sequences(
    tokens: &[TokenId],
    params: &SamplingParams,
    tokenizer: &Arc<dyn Tokenizer + Send + Sync>,
) -> Result<bool> {
    if params.stop_sequences.is_empty() {
        return Ok(false);
    }

    let text = tokenizer.decode(tokens, true)?;
    for stop_seq in &params.stop_sequences {
        if text.contains(stop_seq) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn determine_finish_reason(
    stop_reason: Option<FinishReason>,
    generated_len: usize,
    max_tokens: usize,
) -> FinishReason {
    if let Some(reason) = stop_reason {
        return reason;
    }

    if generated_len >= max_tokens {
        FinishReason::Length
    } else {
        FinishReason::EOS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device as CandleDevice, Tensor};
    use ferrum_models::tensor_wrapper::CandleTensorWrapper;

    #[test]
    fn extract_last_token_logits_uses_last_sequence_position() {
        let values: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let tensor = Tensor::new(values.as_slice(), &CandleDevice::Cpu)
            .expect("create tensor")
            .reshape((1, 3, 4))
            .expect("reshape tensor");
        let logits_ref: ferrum_interfaces::TensorRef = Arc::new(CandleTensorWrapper::new(tensor));

        let last = extract_last_token_logits(&logits_ref).expect("extract last logits");
        let sampled = last.to_vec_f32().expect("to_vec_f32");

        assert_eq!(sampled, vec![8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn determine_finish_reason_prefers_explicit_reason() {
        let reason = determine_finish_reason(Some(FinishReason::Stop), 10, 100);
        assert_eq!(reason, FinishReason::Stop);
    }

    #[test]
    fn determine_finish_reason_falls_back_to_length() {
        let reason = determine_finish_reason(None, 64, 64);
        assert_eq!(reason, FinishReason::Length);
    }
}
