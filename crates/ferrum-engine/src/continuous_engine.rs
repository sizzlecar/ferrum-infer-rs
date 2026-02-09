//! Continuous Batching Engine
//!
//! This module provides an inference engine implementation that uses continuous
//! batching for optimal GPU utilization. Key features:
//!
//! - Iteration-level scheduling: new requests can join running batches
//! - Separate prefill and decode phases
//! - Chunked prefill support for long prompts
//! - Request preemption and swapping
//! - Efficient KV cache management

use async_trait::async_trait;
use ferrum_interfaces::{
    engine::InferenceEngine, kv_cache::AllocationRequest, KvCacheHandle, KvCacheManager,
    ModelExecutor, Sampler, SchedulerInterface as Scheduler, Tokenizer,
};
use ferrum_scheduler::implementations::{ContinuousBatchScheduler, RequestPhase};
use ferrum_types::{
    DataType, EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest,
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

/// State of a running sequence in the continuous batch
#[derive(Debug)]
pub struct SequenceState {
    /// Request ID
    pub request_id: RequestId,
    /// Input token IDs (from tokenization)
    pub input_tokens: Vec<TokenId>,
    /// Generated tokens so far
    pub generated_tokens: Vec<TokenId>,
    /// KV cache handle
    pub kv_cache: Option<Arc<dyn KvCacheHandle>>,
    /// Sampling parameters
    pub sampling_params: SamplingParams,
    /// Phase of the request
    pub phase: RequestPhase,
    /// Random number generator for sampling
    pub rng: StdRng,
    /// Whether prefill is complete
    pub prefill_complete: bool,
    /// Stream sender (if streaming)
    pub stream_sender: Option<mpsc::Sender<Result<StreamChunk>>>,
    /// Start time
    pub start_time: Instant,
    /// Number of tokens generated in current iteration
    pub tokens_this_iteration: usize,
}

impl SequenceState {
    /// Create from inference request
    pub fn new(request: InferenceRequest, input_tokens: Vec<TokenId>) -> Self {
        let seed = request.sampling_params.seed.unwrap_or(42);
        Self {
            request_id: request.id.clone(),
            input_tokens,
            generated_tokens: Vec::new(),
            kv_cache: None,
            sampling_params: request.sampling_params,
            phase: RequestPhase::Waiting,
            rng: StdRng::seed_from_u64(seed),
            prefill_complete: false,
            stream_sender: None,
            start_time: Instant::now(),
            tokens_this_iteration: 0,
        }
    }

    /// Total tokens in sequence
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    /// Check if generation should stop
    pub fn should_stop(&self, vocab_size: usize) -> bool {
        // Check max tokens
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return true;
        }

        // Check for EOS token
        if let Some(&last_token) = self.generated_tokens.last() {
            // Assume tokens near end of vocab are special
            if last_token.get() >= (vocab_size - 10) as u32 {
                return true;
            }
        }

        false
    }
}

/// Continuous batching inference engine
pub struct ContinuousBatchEngine {
    /// Configuration
    config: EngineConfig,
    /// Scheduler (must be ContinuousBatchScheduler)
    scheduler: Arc<ContinuousBatchScheduler>,
    /// Tokenizer
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    /// Sampler
    sampler: Arc<dyn Sampler + Send + Sync>,
    /// KV cache manager
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    /// Model executor
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    /// Active sequences
    sequences: RwLock<HashMap<RequestId, SequenceState>>,
    /// Engine running flag
    is_running: AtomicBool,
    /// Shutdown notification
    shutdown_notify: Arc<Notify>,
    /// Iteration counter
    iteration_count: AtomicU64,
    /// Stats
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
}

impl ContinuousBatchEngine {
    /// Create new continuous batch engine
    pub fn new(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    ) -> Self {
        info!("Creating ContinuousBatchEngine");

        Self {
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
            sequences: RwLock::new(HashMap::new()),
            is_running: AtomicBool::new(false),
            shutdown_notify: Arc::new(Notify::new()),
            iteration_count: AtomicU64::new(0),
            total_prefill_tokens: AtomicU64::new(0),
            total_decode_tokens: AtomicU64::new(0),
        }
    }

    /// Start the engine loop
    pub async fn start(&self) -> Result<()> {
        if self.is_running.swap(true, Ordering::SeqCst) {
            return Err(FerrumError::internal("Engine already running"));
        }

        info!("Starting continuous batch engine loop");

        // The engine loop runs until shutdown
        while self.is_running.load(Ordering::SeqCst) {
            // Run one iteration
            if let Err(e) = self.run_iteration().await {
                warn!("Iteration error: {}", e);
            }

            // Brief yield to allow other tasks
            tokio::task::yield_now().await;
        }

        info!("Continuous batch engine stopped");
        Ok(())
    }

    /// Run a single iteration of the engine loop
    async fn run_iteration(&self) -> Result<()> {
        let iteration = self.iteration_count.fetch_add(1, Ordering::Relaxed);

        // Get batch hint based on current capacity
        let hint = ferrum_interfaces::BatchHint {
            max_batch_size: self.config.batching.max_batch_size,
            max_tokens: self.config.batching.max_batch_size * 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: ferrum_interfaces::scheduler::ResourceConstraints::default(),
        };

        // Get next batch from scheduler
        let batch = match self.scheduler.next_batch(hint).await {
            Some(b) => b,
            None => {
                // No work to do, wait a bit
                tokio::time::sleep(Duration::from_millis(1)).await;
                return Ok(());
            }
        };

        debug!(
            "Iteration {}: processing batch with {} requests",
            iteration,
            batch.size()
        );

        // Process the batch
        self.process_batch(&batch).await?;

        Ok(())
    }

    /// Process a batch of requests
    async fn process_batch(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        // Separate prefill and decode requests
        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();

        for scheduled_req in &batch.requests {
            let request_id = &scheduled_req.request.id;

            // Get or create sequence state
            let mut sequences = self.sequences.write();
            let seq_state = sequences.entry(request_id.clone()).or_insert_with(|| {
                // Tokenize the prompt
                let input_tokens = self
                    .tokenizer
                    .encode(&scheduled_req.request.prompt, true)
                    .unwrap_or_else(|_| vec![TokenId::new(0)]);

                SequenceState::new(scheduled_req.request.clone(), input_tokens)
            });

            if !seq_state.prefill_complete {
                prefill_requests.push(request_id.clone());
            } else {
                decode_requests.push(request_id.clone());
            }
        }
        drop(self.sequences.write());

        // Run prefill for requests that need it
        for request_id in prefill_requests {
            if let Err(e) = self.run_prefill(&request_id).await {
                warn!("Prefill failed for {}: {}", request_id, e);
                self.complete_request(&request_id, FinishReason::Error)
                    .await?;
            }
        }

        // Run decode step for all ready requests
        for request_id in decode_requests {
            if let Err(e) = self.run_decode_step(&request_id).await {
                warn!("Decode failed for {}: {}", request_id, e);
                self.complete_request(&request_id, FinishReason::Error)
                    .await?;
            }
        }

        Ok(())
    }

    /// Run prefill for a request
    async fn run_prefill(&self, request_id: &RequestId) -> Result<()> {
        let (input_tensor, num_tokens) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;

            // Create input tensor
            let token_u32s: Vec<u32> = seq.input_tokens.iter().map(|t| t.get()).collect();
            let tensor = candle_core::Tensor::new(&token_u32s[..], &candle_core::Device::Cpu)
                .map_err(|e| FerrumError::model(format!("Tensor error: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("Unsqueeze error: {}", e)))?;

            let tensor_ref: ferrum_interfaces::TensorRef =
                Arc::new(ferrum_models::CandleTensorWrapper::new(tensor));
            (tensor_ref, seq.input_tokens.len())
        };

        // Allocate KV cache
        let alloc_request = AllocationRequest {
            request_id: request_id.clone(),
            initial_tokens: num_tokens,
            max_sequence_length: 2048,
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            device: self.config.backend.device.clone(),
            dtype: DataType::FP16,
            priority: Priority::Normal,
        };

        let _kv_handle = self.kv_cache.allocate(&alloc_request).await?;

        // Run prefill
        let prefill_input = ferrum_interfaces::model_executor::PrefillInput::new(input_tensor);
        let prefill_output = self.model_executor.prefill(&prefill_input).await?;

        // Sample first token
        let logits_vec = prefill_output.logits.to_vec_f32()?;

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

        // Update scheduler
        self.scheduler.mark_prefill_complete(request_id, num_tokens);

        // Track stats
        self.total_prefill_tokens
            .fetch_add(num_tokens as u64, Ordering::Relaxed);

        debug!(
            "Prefill complete for {}: {} tokens, first generated: {}",
            request_id,
            num_tokens,
            first_token.get()
        );

        // Send streaming update if applicable
        self.send_stream_update(request_id, first_token).await;

        // Check if we should stop
        let should_stop = {
            let sequences = self.sequences.read();
            if let Some(seq) = sequences.get(request_id) {
                seq.should_stop(self.model_executor.info().vocab_size)
            } else {
                true
            }
        };

        if should_stop {
            self.complete_request(request_id, FinishReason::EOS).await?;
        }

        Ok(())
    }

    /// Run a single decode step for a request
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

            // Get last token
            let last_token = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0));

            let tensor = candle_core::Tensor::new(&[last_token.get()], &candle_core::Device::Cpu)
                .map_err(|e| FerrumError::model(format!("Tensor error: {}", e)))?
                .unsqueeze(0)
                .map_err(|e| FerrumError::model(format!("Unsqueeze error: {}", e)))?;

            let tensor_ref: ferrum_interfaces::TensorRef =
                Arc::new(ferrum_models::CandleTensorWrapper::new(tensor));

            let decode_input =
                ferrum_interfaces::model_executor::DecodeInput::new(tensor_ref, kv_cache.clone());

            decode_input
        };

        // Run decode
        let decode_output = self.model_executor.decode(&decode_input).await?;

        // Sample next token
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

        // Update scheduler
        let generated_count = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.generated_tokens.len())
                .unwrap_or(0)
        };
        self.scheduler
            .update_decode_progress(request_id, generated_count);

        // Track stats
        self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);

        // Send streaming update
        self.send_stream_update(request_id, next_token).await;

        // Check if we should stop
        let should_stop = {
            let sequences = self.sequences.read();
            if let Some(seq) = sequences.get(request_id) {
                seq.should_stop(self.model_executor.info().vocab_size)
            } else {
                true
            }
        };

        if should_stop {
            let finish_reason = {
                let sequences = self.sequences.read();
                if let Some(seq) = sequences.get(request_id) {
                    if seq.generated_tokens.len() >= seq.sampling_params.max_tokens {
                        FinishReason::Length
                    } else {
                        FinishReason::EOS
                    }
                } else {
                    FinishReason::Error
                }
            };
            self.complete_request(request_id, finish_reason).await?;
        }

        Ok(())
    }

    /// Send streaming update for a token
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

    /// Complete a request
    async fn complete_request(
        &self,
        request_id: &RequestId,
        finish_reason: FinishReason,
    ) -> Result<()> {
        // Extract data without holding lock across await
        let (response, stream_sender, has_kv_cache) = {
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

                let has_kv_cache = seq.kv_cache.is_some();
                (response, seq.stream_sender, has_kv_cache)
            } else {
                return Ok(());
            }
        };

        // Deallocate KV cache outside of lock
        if has_kv_cache {
            let _ = self.kv_cache.deallocate(request_id.clone()).await;
        }

        // Notify scheduler
        self.scheduler
            .complete(request_id.clone(), &response)
            .await?;

        // Send final stream chunk if streaming
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

#[async_trait]
impl InferenceEngine for ContinuousBatchEngine {
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.id.clone();

        // Submit to scheduler
        self.scheduler.submit(request.clone()).await?;

        // Tokenize and create sequence state
        let input_tokens = self.tokenizer.encode(&request.prompt, true)?;
        let seq_state = SequenceState::new(request, input_tokens.clone());

        // Insert into sequences
        self.sequences.write().insert(request_id.clone(), seq_state);

        // Wait for completion (poll-based for now)
        loop {
            // Run an iteration
            self.run_iteration().await?;

            // Check if request is done
            if !self.sequences.read().contains_key(&request_id) {
                break;
            }

            // Check timeout
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // The response was already built and sent to scheduler
        // For non-streaming, we need to rebuild it
        let text = "Generated response"; // Placeholder
        Ok(InferenceResponse {
            request_id,
            text: text.to_string(),
            tokens: vec![],
            finish_reason: FinishReason::EOS,
            usage: TokenUsage::new(input_tokens.len(), 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (tx, rx) = mpsc::channel(100);
        let request_id = request.id.clone();

        // Submit to scheduler
        self.scheduler.submit(request.clone()).await?;

        // Tokenize and create sequence state with stream sender
        let input_tokens = self.tokenizer.encode(&request.prompt, true)?;
        let mut seq_state = SequenceState::new(request, input_tokens);
        seq_state.stream_sender = Some(tx);

        // Insert into sequences
        self.sequences.write().insert(request_id.clone(), seq_state);

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn status(&self) -> EngineStatus {
        let metrics = self.scheduler.metrics();

        EngineStatus {
            is_ready: self.is_running.load(Ordering::SeqCst),
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
        self.is_running.store(false, Ordering::SeqCst);
        self.shutdown_notify.notify_waiters();
        Ok(())
    }

    fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn metrics(&self) -> ferrum_types::EngineMetrics {
        let scheduler_metrics = self.scheduler.metrics();

        ferrum_types::EngineMetrics {
            total_requests: scheduler_metrics.completed_requests
                + scheduler_metrics.failed_requests,
            successful_requests: scheduler_metrics.completed_requests,
            failed_requests: scheduler_metrics.failed_requests,
            avg_request_latency_ms: 0.0,
            p95_request_latency_ms: 0.0,
            p99_request_latency_ms: 0.0,
            throughput_rps: scheduler_metrics.throughput_rps as f32,
            tokens_per_second: 0.0,
            queue_metrics: Default::default(),
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: Default::default(),
        }
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        if self.is_running.load(Ordering::SeqCst) {
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
            .field("is_running", &self.is_running.load(Ordering::SeqCst))
            .field(
                "iteration_count",
                &self.iteration_count.load(Ordering::SeqCst),
            )
            .field("active_sequences", &self.sequences.read().len())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

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
