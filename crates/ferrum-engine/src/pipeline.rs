//! Inference pipeline implementation

use std::sync::Arc;
use async_trait::async_trait;
use tracing::{debug, instrument, trace, warn};
use ferrum_types::{
    Result,
    InferenceRequest,
    InferenceResponse,
    StreamChunk,
    FinishReason,
    SamplingParams,
    TokenId,
    FerrumError,
};
use ferrum_interfaces::{
    BatchHint,
    BatchPlan,
    IncrementalTokenizer,
    KvCacheHandle,
    KvCacheManager,
    LogitsProcessor,
    ModelExecutor,
    PrefillInput,
    PrefillOutput,
    DecodeInput,
    DecodeOutput,
    Sampler,
    SamplingContext,
    Scheduler as SchedulerInterface,
    TensorFactory,
    TensorLike,
    Tokenizer,
};
use ferrum_runtime::TensorFactoryHandle;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

/// Streaming token event
#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub token_id: TokenId,
    pub text_delta: String,
}

/// Pipeline components required for inference
pub struct PipelineComponents {
    pub scheduler: Arc<dyn Scheduler + Send + Sync>,
    pub tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    pub incremental_tokenizer: Arc<dyn IncrementalTokenizer + Send + Sync>,
    pub tensor_factory: TensorFactoryHandle,
    pub sampler: Arc<dyn Sampler + Send + Sync>,
    pub logits_processors: Vec<Arc<dyn LogitsProcessor + Send + Sync>>,
    pub kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    pub model_executor: Arc<dyn ModelExecutor + Send + Sync>,
}

/// Inference pipeline orchestrating scheduler, kv cache, tokenizer, model executor and sampler
pub struct InferencePipeline {
    components: PipelineComponents,
    batch_hint: BatchHint,
}

impl InferencePipeline {
    pub fn new(components: PipelineComponents, batch_hint: BatchHint) -> Self {
        Self { components, batch_hint }
    }

    /// Submit request to scheduler
    pub async fn submit_request(&self, request: InferenceRequest) -> Result<()> {
        let scheduler_request_id = self.components.scheduler.submit(request.clone()).await?;
        trace!(?scheduler_request_id, "request submitted to scheduler");
        Ok(())
    }

    /// Process ready batches and run inference until completion for target request id
    #[instrument(skip(self, target_request_id, tx))]
    pub async fn process_batches(
        &self,
        target_request_id: &ferrum_types::RequestId,
        tx: UnboundedSender<Result<StreamChunk>>,
    ) -> Result<()> {
        // TODO: Phase 1 implementation pending: tokenize prompts, manage KV allocation, run model executor and sampling loop.
        loop {
            // Pull next batch from scheduler
            let batch_plan = self.components.scheduler.next_batch(self.batch_hint.clone()).await;
            let Some(batch) = batch_plan else {
                trace!("no batch available yet");
                tokio::task::yield_now().await;
                continue;
            };

            trace!(batch_id = %batch.batch_id, size = batch.requests.len(), "processing batch");
            // Tokenize prompts for new requests and allocate KV
            let mut batch_prefill_inputs = Vec::new();
            let mut batch_targets = Vec::new();

            for scheduled in &batch.requests {
                if scheduled.request.id == *target_request_id {
                    batch_targets.push(scheduled.request.clone());
                }
                // Tokenize prompt
                let encode_result = self.components.tokenizer.encode(
                    &scheduled.request.prompt,
                    true,
                )?;

                let input_ids_tensor = self.create_tensor_from_tokens(&encode_result)?;

                // Build PrefillInput
                let prefill_input = PrefillInput {
                    input_ids: input_ids_tensor,
                    attention_mask: None,
                    position_ids: None,
                };

                batch_prefill_inputs.push((scheduled.request.clone(), prefill_input));
            }

            if batch_prefill_inputs.is_empty() {
                continue;
            }

            // Prefill stage - one request at a time for MVP
            for (request, prefill_input) in batch_prefill_inputs.into_iter() {
                let PrefillOutput { logits, kv } = self.components.model_executor.prefill(&prefill_input).await?;

                // Sample next token
                let token_id = self.sample_from_logits(&logits, request.sampling_params.clone())?;

                // Stream delta text if request matches target
                if &request.id == target_request_id {
                    let delta = self.components.incremental_tokenizer.decode_incremental(
                        &[],
                        token_id,
                    )?;

                    let event = TokenEvent {
                        token_id,
                        text_delta: delta.clone(),
                    };

                    self.send_stream_chunk(&tx, &request, event, None, None)?;
                }

                // Decode loop for target request
                if &request.id == target_request_id {
                    self.decode_loop(request.clone(), token_id, kv, tx.clone()).await?;
                    return Ok(());
                }
            }
        }
    }

    /// Decode loop for a given request
    async fn decode_loop(
        &self,
        request: InferenceRequest,
        mut last_token: TokenId,
        mut kv: Arc<dyn KvCacheHandle>,
        tx: UnboundedSender<Result<StreamChunk>>,
    ) -> Result<()> {
        let mut tokens = vec![last_token];
        let mut text_acc = String::new();

        for step in 0..request.sampling_params.max_tokens {
            // Prepare input tensor for decode step (batch size 1)
            let decode_tensor = self.create_tensor_from_tokens(&tokens)?;

            let decode_input = DecodeInput {
                input_ids: decode_tensor,
                kv: kv.clone(),
                position_ids: None,
            };

            let DecodeOutput { logits, kv: new_kv } = self.components.model_executor.decode(&decode_input).await?;
            kv = new_kv;

            let token_id = self.sample_from_logits(&logits, request.sampling_params.clone())?;
            tokens.push(token_id);

            let delta = self.components.incremental_tokenizer.decode_incremental(
                tokens.iter().take(tokens.len() - 1).collect::<Vec<_>>().as_slice(),
                token_id,
            )?;
            text_acc.push_str(&delta);

            let event = TokenEvent {
                token_id,
                text_delta: delta.clone(),
            };

            self.send_stream_chunk(&tx, &request, event, None, None)?;

            if self.is_stop_token(token_id, &request.sampling_params) {
                let response = self.build_final_response(request.clone(), text_acc.clone(), tokens.clone(), FinishReason::Stop);
                self.send_completion(&tx, response)?;
                return Ok(());
            }

            if step + 1 >= request.sampling_params.max_tokens {
                let response = self.build_final_response(request.clone(), text_acc.clone(), tokens.clone(), FinishReason::Length);
                self.send_completion(&tx, response)?;
                return Ok(());
            }
        }

        Ok(())
    }

    fn create_tensor_from_tokens(&self, tokens: &[TokenId]) -> Result<ferrum_interfaces::TensorRef> {
        let shape = [1, tokens.len()];
        let data: Vec<f32> = tokens.iter().map(|t| *t as f32).collect();
        self
            .components
            .tensor_factory
            .as_ref()
            .from_slice(&data, &shape, ferrum_types::Device::Cuda(0))
            .map_err(|e| FerrumError::backend(format!("Failed to build input tensor: {}", e)))
    }

    fn sample_from_logits(
        &self,
        logits: &ferrum_interfaces::TensorRef,
        sampling_params: SamplingParams,
    ) -> Result<TokenId> {
        let logits_slice = logits
            .data_f32()
            .ok_or_else(|| FerrumError::backend("Logits tensor must expose f32 data"))?;

        let mut logits_buf = logits_slice.to_vec();
        let mut rng = if let Some(seed) = sampling_params.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let mut ctx = SamplingContext::new(
            0,
            &sampling_params,
            &mut logits_buf,
            &[],
            &std::collections::HashMap::new(),
            logits_buf.len(),
        );

        for processor in &self.components.logits_processors {
            processor.process(&mut ctx)?;
        }

        self.components.sampler.sample_with_context(&ctx, &mut rng)
    }

    fn is_stop_token(&self, _token_id: TokenId, sampling: &SamplingParams) -> bool {
        if let Some(stop_tokens) = &sampling.stop_tokens {
            stop_tokens.contains(&_token_id)
        } else {
            false
        }
    }

    fn send_stream_chunk(
        &self,
        tx: &UnboundedSender<Result<StreamChunk>>,
        request: &InferenceRequest,
        event: TokenEvent,
        finish_reason: Option<FinishReason>,
        usage: Option<ferrum_types::TokenUsage>,
    ) -> Result<()> {
        let chunk = StreamChunk {
            request_id: request.id,
            text: event.text_delta,
            token: Some(event.token_id),
            finish_reason,
            usage,
            created_at: chrono::Utc::now(),
            metadata: request.metadata.clone(),
        };
        tx.send(Ok(chunk)).map_err(|_| FerrumError::channel_closed("failed to send stream chunk"))
    }

    fn send_completion(
        &self,
        tx: &UnboundedSender<Result<StreamChunk>>,
        response: InferenceResponse,
    ) -> Result<()> {
        tx.send(Ok(StreamChunk::Complete { response })).map_err(|_| FerrumError::channel_closed("failed to send completion chunk"))
    }

    fn build_final_response(
        &self,
        request: InferenceRequest,
        text: String,
        tokens: Vec<TokenId>,
        reason: FinishReason,
    ) -> InferenceResponse {
        InferenceResponse {
            request_id: request.id,
            text,
            tokens,
            finish_reason: reason,
            usage: ferrum_types::TokenUsage::default(),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: request.metadata,
        }
    }
}

impl Default for InferencePipeline {
    fn default() -> Self {
        panic!("InferencePipeline requires components; use InferencePipeline::new")
    }
}
