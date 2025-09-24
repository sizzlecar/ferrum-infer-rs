//! Inference pipeline implementation

use std::collections::HashMap;
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
        loop {
            let Some(batch_plan) = self.components.scheduler.next_batch(self.batch_hint.clone()).await else {
                trace!("no batch available yet");
                tokio::task::yield_now().await;
                continue;
            };

            if batch_plan.is_empty() {
                trace!(batch_id = %batch_plan.batch_id, "received empty batch");
                continue;
            }

            trace!(batch_id = %batch_plan.batch_id, size = batch_plan.size(), "processing batch");

            let prefill_work = self.prepare_prefill_work(&batch_plan)?;

            for (request, prefill_input) in prefill_work {
                let PrefillOutput { logits, kv_cache, .. } = self
                    .components
                    .model_executor
                    .prefill(&prefill_input)
                    .await?;

                let last_logits = self.extract_last_logits(&logits)?;
                let token_id = self.sample_from_logits(&last_logits, request.sampling_params.clone())?;

                if &request.id == target_request_id {
                    let delta = self
                        .components
                        .incremental_tokenizer
                        .decode_incremental(&[], token_id)?;

                    let event = TokenEvent {
                        token_id,
                        text_delta: delta.clone(),
                    };
                    self.send_stream_chunk(&tx, &request, event, None, None)?;

                    self.decode_loop(request.clone(), token_id, kv_cache, tx.clone()).await?;
                    return Ok(());
                }

                // TODO: 对于非目标请求，将初始 token 推入队列或存入状态，等待批次协同。
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
        let mut token_frequencies: HashMap<TokenId, usize> = HashMap::new();
        *token_frequencies.entry(last_token).or_insert(0) += 1;

        let mut stop_reason = None;

        for step in 0..request.sampling_params.max_tokens {
            // Prepare input tensor for decode step (batch size 1)
            let decode_tensor = self.create_tensor_from_tokens(&tokens[tokens.len() - 1..])?;

            let decode_input = DecodeInput {
                input_ids: decode_tensor,
                kv: kv.clone(),
                position_ids: None,
            };

            let DecodeOutput { logits, kv: new_kv } = self.components.model_executor.decode(&decode_input).await?;
            kv = new_kv;

            let mut logits_buf = logits
                .data_f32()
                .ok_or_else(|| FerrumError::backend("Logits tensor must expose f32 data"))?
                .to_vec();

            let mut ctx = SamplingContext::new(
                step + 1,
                &request.sampling_params,
                &mut logits_buf,
                &tokens,
                &token_frequencies,
                logits_buf.len(),
            );

            for processor in &self.components.logits_processors {
                processor.process(&mut ctx)?;
            }

            let mut rng = if let Some(seed) = request.sampling_params.seed {
                StdRng::seed_from_u64(seed + step as u64 + 1)
            } else {
                StdRng::from_entropy()
            };

            let token_id = self.components.sampler.sample_with_context(&ctx, &mut rng)?;
            tokens.push(token_id);
            *token_frequencies.entry(token_id).or_insert(0) += 1;

            let delta = self
                .components
                .incremental_tokenizer
                .decode_incremental(tokens.iter().take(tokens.len() - 1).collect::<Vec<_>>().as_slice(), token_id)?;
            text_acc.push_str(&delta);

            let event = TokenEvent {
                token_id,
                text_delta: delta.clone(),
            };

            self.send_stream_chunk(&tx, &request, event, None, None)?;

            if self.is_stop_token(token_id, &request.sampling_params) {
                stop_reason = Some(FinishReason::Stop);
                break;
            }

            if self.contains_stop_sequence(&text_acc, &request.sampling_params.stop_sequences) {
                stop_reason = Some(FinishReason::Stop);
                break;
            }

            if step + 1 >= request.sampling_params.max_tokens {
                stop_reason = Some(FinishReason::Length);
                break;
            }
        }

        let reason = stop_reason.unwrap_or(FinishReason::Stop);
        let response = self.build_final_response(request.clone(), text_acc.clone(), tokens.clone(), reason);
        self.send_completion(&tx, response)?;
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
            &HashMap::new(),
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

    fn prepare_prefill_work(&self, batch_plan: &BatchPlan) -> Result<Vec<(InferenceRequest, PrefillInput)>> {
        batch_plan
            .requests
            .iter()
            .map(|scheduled| {
                let token_ids = self.components.tokenizer.encode(&scheduled.request.prompt, true)?;
                let input_tensor = self.create_tensor_from_tokens(&token_ids)?;
                Ok((
                    scheduled.request.clone(),
                    PrefillInput {
                        input_ids: input_tensor,
                        attention_mask: None,
                        position_ids: None,
                    },
                ))
            })
            .collect()
    }

    fn extract_last_logits(&self, logits: &ferrum_interfaces::TensorRef) -> Result<ferrum_interfaces::TensorRef> {
        let shape = logits.shape();
        if shape.len() <= 2 {
            return Ok(logits.clone());
        }

        let seq_len = shape[1];
        if seq_len == 0 {
            return Err(FerrumError::backend("Empty logits sequence"));
        }

        logits.view(&[0, seq_len - 1, 0], &[shape[0], seq_len, shape[2]])
    }

    fn contains_stop_sequence(&self, text: &str, stop_sequences: &[String]) -> bool {
        stop_sequences.iter().any(|seq| !seq.is_empty() && text.ends_with(seq))
    }
}

impl Default for InferencePipeline {
    fn default() -> Self {
        panic!("InferencePipeline requires components; use InferencePipeline::new")
    }
}
