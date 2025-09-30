//! Main inference engine implementation

use crate::pipeline::{InferencePipeline, PipelineComponents};
use async_trait::async_trait;
use ferrum_interfaces::{EngineStatus, InferenceEngine, StreamChunk};
use ferrum_types::{EngineConfig, FerrumError, InferenceRequest, InferenceResponse, Result};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

/// Default inference engine implementation
#[derive(Debug)]
pub struct DefaultInferenceEngine {
    /// Configuration
    config: EngineConfig,
    /// Scheduler component
    scheduler: Arc<dyn ferrum_scheduler::Scheduler + Send + Sync>,
    /// Tokenizer component
    tokenizer: Arc<dyn ferrum_tokenizer::Tokenizer + Send + Sync>,
    /// Sampler component
    sampler: Arc<dyn ferrum_sampler::Sampler + Send + Sync>,
    /// KV cache manager
    kv_cache: Arc<dyn ferrum_kv::KvCacheManager + Send + Sync>,
    /// Model executor
    model_executor: Arc<dyn ferrum_interfaces::ModelExecutor + Send + Sync>,
    /// Pipeline (built lazily)
    pipeline: Arc<InferencePipeline>,
}

impl DefaultInferenceEngine {
    /// Create new inference engine
    pub fn new(
        config: EngineConfig,
        scheduler: Arc<dyn ferrum_scheduler::Scheduler + Send + Sync>,
        tokenizer: Arc<dyn ferrum_tokenizer::Tokenizer + Send + Sync>,
        sampler: Arc<dyn ferrum_sampler::Sampler + Send + Sync>,
        kv_cache: Arc<dyn ferrum_kv::KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ferrum_interfaces::ModelExecutor + Send + Sync>,
    ) -> Self {
        info!(
            "Created inference engine with model: {:?}",
            config.model.model_id
        );

        let components = PipelineComponents {
            scheduler: scheduler.clone(),
            tokenizer: tokenizer.clone(),
            incremental_tokenizer: tokenizer.clone(),
            tensor_factory: ferrum_runtime::TensorFactoryHandle::default(),
            sampler: sampler.clone(),
            logits_processors: Vec::new(),
            kv_cache: kv_cache.clone(),
            model_executor: model_executor.clone(),
        };
        let batch_hint = ferrum_interfaces::BatchHint::simple(config.batching.max_batch_size);
        let pipeline = Arc::new(InferencePipeline::new(components, batch_hint));

        Self {
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
            pipeline,
        }
    }
}

#[async_trait]
impl InferenceEngine for DefaultInferenceEngine {
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let mut stream = self.infer_stream(request).await?;
        let mut final_response = None;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(StreamChunk::Complete { response }) => {
                    final_response = Some(response);
                    break;
                }
                Ok(StreamChunk::Delta { .. }) => continue,
                Ok(StreamChunk::Error { error }) => return Err(error),
                Err(e) => return Err(e),
            }
        }
        final_response.ok_or_else(|| FerrumError::engine_error("No response generated"))
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send + Unpin>> {
        info!(?request.id, "Starting streaming inference");

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let pipeline = self.pipeline.clone();
        let target_request_id = request.id;

        pipeline.submit_request(request.clone()).await?;

        tokio::spawn(async move {
            if let Err(e) = pipeline
                .process_batches(&target_request_id, tx.clone())
                .await
            {
                let _ = tx.send(Ok(StreamChunk::Error { error: e }));
            }
        });

        Ok(Box::new(ReceiverStream::new(rx)))
    }

    async fn status(&self) -> EngineStatus {
        // TODO: gather real metrics from scheduler/KV/pipeline
        EngineStatus {
            is_ready: true,
            loaded_models: vec![self.model_executor.info().model_id.clone()],
            active_requests: 0,
            queued_requests: 0,
            memory_usage: ferrum_types::MemoryUsage::default(),
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: "mvp".to_string(),
            component_status: ferrum_interfaces::ComponentStatus {
                scheduler: ferrum_interfaces::ComponentHealth::healthy("scheduler"),
                model_executor: ferrum_interfaces::ComponentHealth::healthy("model"),
                tokenizer: ferrum_interfaces::ComponentHealth::healthy("tokenizer"),
                kv_cache: ferrum_interfaces::ComponentHealth::healthy("kv"),
                memory_manager: ferrum_interfaces::ComponentHealth::healthy("memory"),
                backend: ferrum_interfaces::ComponentHealth::healthy("backend"),
            },
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down inference engine");
        Ok(())
    }

    fn config(&self) -> &ferrum_types::EngineConfig {
        &self.config
    }

    fn metrics(&self) -> ferrum_interfaces::EngineMetrics {
        ferrum_interfaces::EngineMetrics::default()
    }

    async fn health_check(&self) -> ferrum_interfaces::HealthStatus {
        ferrum_interfaces::HealthStatus::healthy()
    }
}

impl DefaultInferenceEngine {
    // Process streaming request is now handled via pipeline.process_batches
}
