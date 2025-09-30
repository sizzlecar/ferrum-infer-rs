//! Main inference engine - MVP implementation

use async_trait::async_trait;
use ferrum_interfaces::{
    engine::InferenceEngine, KvCacheHandle, KvCacheManager, ModelExecutor, Sampler,
    SchedulerInterface as Scheduler, Tokenizer,
};
use ferrum_types::{
    EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest, InferenceResponse,
    Result, SamplingParams, StreamChunk, TokenId, TokenUsage,
};
use futures::stream::{Stream, StreamExt};
use rand::{rngs::StdRng, SeedableRng};
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Default inference engine - MVP implementation
pub struct DefaultInferenceEngine {
    config: EngineConfig,
    scheduler: Arc<dyn Scheduler + Send + Sync>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    sampler: Arc<dyn Sampler + Send + Sync>,
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
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
        info!("Created DefaultInferenceEngine");

        Self {
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
        }
    }

    /// Execute single inference request
    async fn execute_request(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.id.clone();
        debug!("Executing request: {:?}", request_id);

        // 1. Tokenize prompt
        let input_tokens = self.tokenizer.encode(&request.prompt, true)?;
        let prompt_tokens = input_tokens.len();

        debug!("Encoded {} tokens from prompt", prompt_tokens);

        // 2. Prepare prefill input
        let factory = self.model_executor.info().device;
        let prefill_input = create_prefill_input(&input_tokens, &factory)?;

        // 3. Execute prefill
        let prefill_output = self.model_executor.prefill(&prefill_input).await?;

        // 4. Generate tokens (decode loop)
        let max_tokens = request.sampling_params.max_tokens;
        let mut generated_tokens = Vec::new();
        let mut kv_cache = prefill_output.kv_cache.clone();
        let mut rng = create_rng(&request.sampling_params);

        for step in 0..max_tokens {
            // Get logits from last position
            let logits = if step == 0 {
                extract_last_token_logits(&prefill_output.logits)?
            } else {
                // Decode step
                let decode_input = create_decode_input(&generated_tokens, kv_cache.clone(), &factory)?;
                let decode_output = self.model_executor.decode(&decode_input).await?;
                kv_cache = decode_output.kv_cache.clone();
                decode_output.logits.clone()
            };

            // Sample next token
            let next_token = sample_token(&logits, &request.sampling_params, &self.sampler, &mut rng)?;

            // Check stop conditions
            if is_stop_token(next_token, self.model_executor.info().vocab_size) {
                debug!("Hit EOS token at step {}", step);
                break;
            }

            generated_tokens.push(next_token);

            // Check stop sequences
            if check_stop_sequences(&generated_tokens, &request.sampling_params, &self.tokenizer)? {
                debug!("Hit stop sequence at step {}", step);
                break;
            }
        }

        // 5. Decode output tokens
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, true)?;

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
                generated_tokens.len(),
                max_tokens,
                &request.sampling_params,
            ),
            usage: TokenUsage::new(prompt_tokens, generated_tokens.len()),
            latency_ms: 0, // TODO: measure actual latency
            created_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        })
    }
}

#[async_trait]
impl InferenceEngine for DefaultInferenceEngine {
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        self.execute_request(&request).await
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (tx, rx) = mpsc::channel(100);
        let engine = self.clone_engine();

        tokio::spawn(async move {
            match engine.execute_request(&request).await {
                Ok(response) => {
                    let chunk = StreamChunk {
                        request_id: response.request_id.clone(),
                        text: response.text.clone(),
                        token: None,
                        finish_reason: Some(response.finish_reason),
                        usage: Some(response.usage.clone()),
                        created_at: response.created_at,
                        metadata: response.metadata.clone(),
                    };
                    let _ = tx.send(Ok(chunk)).await;
                }
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn status(&self) -> EngineStatus {
        EngineStatus {
            is_ready: true,
            loaded_models: vec![],
            active_requests: 0,
            queued_requests: 0,
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
        info!("Shutting down engine");
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

impl DefaultInferenceEngine {
    fn clone_engine(&self) -> Arc<Self> {
        Arc::new(Self {
            config: self.config.clone(),
            scheduler: self.scheduler.clone(),
            tokenizer: self.tokenizer.clone(),
            sampler: self.sampler.clone(),
            kv_cache: self.kv_cache.clone(),
            model_executor: self.model_executor.clone(),
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_prefill_input(
    tokens: &[TokenId],
    _device: &ferrum_types::Device,
) -> Result<ferrum_interfaces::model_executor::PrefillInput> {
    // MVP: create dummy tensor - in real implementation would use tensor factory
    Err(FerrumError::model(
        "MVP: Prefill input creation needs tensor factory integration",
    ))
}

fn create_decode_input(
    _tokens: &[TokenId],
    kv_cache: Arc<dyn KvCacheHandle>,
    _device: &ferrum_types::Device,
) -> Result<ferrum_interfaces::model_executor::DecodeInput> {
    Err(FerrumError::model(
        "MVP: Decode input creation needs tensor factory integration",
    ))
}

fn extract_last_token_logits(logits: &ferrum_interfaces::TensorRef) -> Result<ferrum_interfaces::TensorRef> {
    // MVP: return as-is for now
    Ok(logits.clone())
}

fn sample_token(
    logits: &ferrum_interfaces::TensorRef,
    _params: &SamplingParams,
    sampler: &Arc<dyn Sampler>,
    rng: &mut StdRng,
) -> Result<TokenId> {
    // MVP: simplified sampling - need to extract logits to Vec<f32>
    // For now, return token 0
    Ok(TokenId::new(0))
}

fn create_rng(params: &SamplingParams) -> StdRng {
    if let Some(seed) = params.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_entropy()
    }
}

fn is_stop_token(token: TokenId, vocab_size: usize) -> bool {
    // Common EOS tokens: typically vocab_size - 1 or 2
    token.get() >= (vocab_size - 10) as u32
}

fn check_stop_sequences(
    tokens: &[TokenId],
    params: &SamplingParams,
    tokenizer: &Arc<dyn Tokenizer>,
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

fn determine_finish_reason(generated_len: usize, max_tokens: usize, _params: &SamplingParams) -> FinishReason {
    if generated_len >= max_tokens {
        FinishReason::Length
    } else {
        FinishReason::EOS
    }
}