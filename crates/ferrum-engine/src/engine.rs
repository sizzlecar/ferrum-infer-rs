//! Main inference engine implementation

use async_trait::async_trait;
use ferrum_core::{
    Backend, BatchManager, CacheManager, EngineStatus, Error, FinishReason, InferenceEngine, InferenceRequest,
    InferenceResponse, MemoryManager, Model, ModelConfig, ModelId, ModelLoader, Result, Scheduler, StreamChunk,
    ScheduledBatch, ScheduledRequest, RequestState, BatchId, BatchOutput, BatchInfo, BlockId, KVBlock, MemoryHandle, MemoryPressure,
    SchedulerStats, CacheStats, MemoryUsage, TokenUsage,
};
use ferrum_models::{DefaultModelRegistry, DefaultModelSourceResolver, ModelSourceConfig, ModelSourceResolver};
use crate::CandleBackend;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use crate::metal::MetalBackend;
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};

/// Engine configuration
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EngineConfig {
    /// Maximum batch size for inference
    pub max_batch_size: usize,

    /// Maximum sequence length
    pub max_sequence_length: usize,

    /// Number of GPU blocks for KV cache
    pub num_gpu_blocks: usize,

    /// Block size for paged attention
    pub block_size: usize,

    /// Enable continuous batching
    pub enable_continuous_batching: bool,

    /// Enable prefix caching
    pub enable_prefix_caching: bool,

    /// GPU memory fraction to use
    pub gpu_memory_fraction: f32,

    /// Scheduling interval in milliseconds
    pub scheduling_interval_ms: u64,

    /// Model ID to load
    pub model_id: String,

    /// Device to use (cuda:0, cpu, etc.)
    pub device: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            max_sequence_length: 4096,
            num_gpu_blocks: 512,
            block_size: 16,
            enable_continuous_batching: true,
            enable_prefix_caching: true,
            gpu_memory_fraction: 0.9,
            scheduling_interval_ms: 10,
            model_id: "llama-2-7b".to_string(),
            device: "cuda:0".to_string(),
        }
    }
}

/// Main inference engine
pub struct Engine {
    config: EngineConfig,
    scheduler: Arc<dyn Scheduler>,
    batch_manager: Arc<dyn BatchManager>,
    cache_manager: Arc<dyn CacheManager>,
    memory_manager: Arc<dyn MemoryManager>,
    model_loader: Arc<dyn ModelLoader>,
    engine_state: Arc<RwLock<EngineState>>,
    shutdown_signal: Arc<RwLock<bool>>,
}

/// Internal engine state
struct EngineState {
    is_ready: bool,
    loaded_models: Vec<ModelId>,
    active_requests: usize,
    total_requests: u64,
    start_time: std::time::Instant,
}

impl Engine {
    /// Create a new inference engine
    pub async fn new(
        config: EngineConfig,
        scheduler: Arc<dyn Scheduler>,
        batch_manager: Arc<dyn BatchManager>,
        cache_manager: Arc<dyn CacheManager>,
        memory_manager: Arc<dyn MemoryManager>,
        model_loader: Arc<dyn ModelLoader>,
    ) -> Result<Self> {
        debug!("Initializing Ferrum Engine with config: {:?}", config);

        // Initialize engine state
        let engine_state = Arc::new(RwLock::new(EngineState {
            is_ready: false,
            loaded_models: vec![],
            active_requests: 0,
            total_requests: 0,
            start_time: std::time::Instant::now(),
        }));

        let engine = Self {
            config,
            scheduler,
            batch_manager,
            cache_manager,
            memory_manager,
            model_loader,
            engine_state,
            shutdown_signal: Arc::new(RwLock::new(false)),
        };

        // Initialize subsystems
        engine.initialize().await?;

        Ok(engine)
    }

    /// Initialize engine subsystems
    pub async fn initialize(&self) -> Result<()> {
        debug!("Initializing engine subsystems...");

        // Use new model maintenance system to resolve model path
        debug!("Resolving model using new maintenance system: {}", self.config.model_id);
        
        let registry = DefaultModelRegistry::with_defaults();
        let resolved_id = registry.resolve_model_id(&self.config.model_id);
        
        let source_config = ModelSourceConfig::default();
        let resolver = DefaultModelSourceResolver::new(source_config);
        
        let resolved_source = resolver.resolve(&resolved_id, None).await
            .map_err(|e| Error::internal(format!("Failed to resolve model '{}': {}", resolved_id, e)))?;
            
        debug!("Model resolved to path: {:?}", resolved_source.local_path);
        
        // Load the model with resolved path
        let model_config = ferrum_core::ModelConfig {
            model_id: ModelId(resolved_id.clone()),
            model_path: resolved_source.local_path.to_string_lossy().to_string(),
            model_type: ferrum_core::ModelType::Llama, // TODO: Get from config
            dtype: ferrum_core::DataType::FP32, // Use FP32 for MVP
            device: self.parse_device()?,
            max_batch_size: self.config.max_batch_size,
            max_sequence_length: self.config.max_sequence_length,
            tensor_parallel_size: None,
            pipeline_parallel_size: None,
            quantization: None,
        };

        self.model_loader.load_model(&model_config).await?;

        // Update state
        {
            let mut state = self.engine_state.write();
            state.loaded_models.push(model_config.model_id);
            state.is_ready = true;
        }

        // Start background scheduler if continuous batching is enabled
        if self.config.enable_continuous_batching {
            self.start_scheduler_loop().await;
        }

        debug!("Engine initialized successfully");
        Ok(())
    }

    /// Parse device string to Device enum
    fn parse_device(&self) -> Result<ferrum_core::Device> {
        match self.config.device.as_str() {
            "cpu" => Ok(ferrum_core::Device::CPU),
            s if s.starts_with("cuda:") => {
                let id = s
                    .trim_start_matches("cuda:")
                    .parse::<usize>()
                    .map_err(|_| Error::internal("Invalid CUDA device ID"))?;
                Ok(ferrum_core::Device::CUDA(id))
            }
            s if s.starts_with("rocm:") => {
                let id = s
                    .trim_start_matches("rocm:")
                    .parse::<usize>()
                    .map_err(|_| Error::internal("Invalid ROCm device ID"))?;
                Ok(ferrum_core::Device::ROCm(id))
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            "metal" => Ok(ferrum_core::Device::Metal),
            _ => Err(Error::internal(format!(
                "Unknown device: {}",
                self.config.device
            ))),
        }
    }

    /// Start the background scheduler loop
    async fn start_scheduler_loop(&self) {
        let scheduler = Arc::clone(&self.scheduler);
        let batch_manager = Arc::clone(&self.batch_manager);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let interval_ms = self.config.scheduling_interval_ms;

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));

            loop {
                if *shutdown_signal.read() {
                    info!("Scheduler loop shutting down");
                    break;
                }

                interval.tick().await;

                // Get next batch to execute
                if let Some(batch) = scheduler.get_next_batch().await {
                    debug!("Executing batch {:?}", batch.batch_id);

                    // Execute the batch
                    match batch_manager.execute_batch(batch.batch_id).await {
                        Ok(_output) => {
                            debug!("Batch execution successful");
                            // Process outputs will be handled by response callbacks
                        }
                        Err(e) => {
                            error!("Batch execution failed: {}", e);
                        }
                    }
                }
            }
        });
    }

    /// Generate completion directly for MVP
    async fn generate_completion_direct(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        // Get the model
        let model = self.model_loader.get_model(&request.model_id.0).await
            .ok_or_else(|| Error::internal("Model not loaded"))?;

        self.generate_completion_with_model(request, &model).await
    }

    /// Generate completion with a specific model
    async fn generate_completion_with_model(&self, request: &InferenceRequest, model: &Arc<dyn Model>) -> Result<InferenceResponse> {
        let prompt = &request.prompt;
        debug!("Generating completion for prompt: {}", prompt);

        // Encode the prompt
        let input_ids = model.encode(prompt)?;
        debug!("Encoded prompt to {} tokens", input_ids.len());

        let mut generated_tokens = Vec::new();
        let mut current_ids = input_ids.clone();
        let mut kv_cache = None;

        // Generate tokens one by one
        for _step in 0..request.sampling_params.max_tokens {
            let output = model.generate_next_token(
                &current_ids,
                kv_cache.as_ref(),
                &request.sampling_params,
            ).await?;

            let token_id = output.token_id;
            generated_tokens.push(token_id);
            current_ids.push(token_id);
            kv_cache = output.kv_cache;

            // Check for stop sequences
            if !request.sampling_params.stop_sequences.is_empty() {
                let decoded_text = model.decode(&generated_tokens)?;
                if request.sampling_params.stop_sequences.iter().any(|stop| decoded_text.contains(stop)) {
                    break;
                }
            }

            // Check for EOS token (typically 2 for LLaMA models)
            if token_id == 2 {
                break;
            }
        }

        let generated_text = model.decode(&generated_tokens)?;
        
        Ok(InferenceResponse {
            request_id: request.id.clone(),
            text: generated_text,
            tokens: generated_tokens.clone(),
            finish_reason: FinishReason::Length, // Simplified for MVP
            usage: TokenUsage {
                prompt_tokens: input_ids.len(),
                completion_tokens: generated_tokens.len(),
                total_tokens: input_ids.len() + generated_tokens.len(),
            },
            latency_ms: 0, // TODO: measure actual latency
            created_at: chrono::Utc::now(),
        })
    }

}

#[async_trait]
impl InferenceEngine for Engine {
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Check if engine is ready
        {
            let state = self.engine_state.read();
            if !state.is_ready {
                return Err(Error::internal("Engine not ready"));
            }
        }

        // Update stats
        {
            let mut state = self.engine_state.write();
            state.active_requests += 1;
            state.total_requests += 1;
        }

        // For MVP, directly generate completion without complex scheduling
        let response = self.generate_completion_direct(&request).await;

        // Update stats
        {
            let mut state = self.engine_state.write();
            state.active_requests -= 1;
        }

        response
    }

    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send + Unpin>> {
        // Check if engine is ready
        {
            let state = self.engine_state.read();
            if !state.is_ready {
                return Err(Error::internal("Engine not ready"));
            }
        }

        // Create stream channel
        let (tx, rx) = mpsc::channel(100);

        // Clone necessary data for the async task
        let model_loader = Arc::clone(&self.model_loader);
        let request_id = request.id.clone();
        let model_id = request.model_id.clone();
        let prompt = request.prompt.clone();
        let sampling_params = request.sampling_params.clone();

        // Spawn async task to generate tokens
        tokio::spawn(async move {
            let result = async {
                // Get the model
                let model = model_loader.get_model(&model_id.0).await
                    .ok_or_else(|| Error::internal("Model not loaded"))?;

                // Encode the prompt
                let input_ids = model.encode(&prompt)?;
                let mut generated_tokens = Vec::new();
                let mut current_ids = input_ids.clone();
                let mut kv_cache = None;

                // Generate tokens one by one, streaming each
                for _step in 0..sampling_params.max_tokens {
                    let output = model.generate_next_token(
                        &current_ids,
                        kv_cache.as_ref(),
                        &sampling_params,
                    ).await?;

                    let token_id = output.token_id;
                    generated_tokens.push(token_id);
                    current_ids.push(token_id);
                    kv_cache = output.kv_cache;

                    // Decode incrementally to preserve proper spacing
                    let token_text = if generated_tokens.len() == 1 {
                        // First token, decode directly
                        model.decode(&[token_id]).unwrap_or_default()
                    } else {
                        // For subsequent tokens, decode full sequence and extract incremental text
                        match (model.decode(&generated_tokens), model.decode(&generated_tokens[..generated_tokens.len()-1])) {
                            (Ok(full_text), Ok(prev_text)) => {
                                if let Some(incremental) = full_text.strip_prefix(&prev_text) {
                                    incremental.to_string()
                                } else {
                                    // Fallback: try single token decode
                                    model.decode(&[token_id]).unwrap_or_default()
                                }
                            }
                            _ => {
                                // Fallback: try single token decode
                                model.decode(&[token_id]).unwrap_or_default()
                            }
                        }
                    };

                    // Send stream chunk (只发送非空文本)
                    if !token_text.is_empty() {
                        let chunk = StreamChunk {
                            request_id: request_id.clone(),
                            text: token_text,
                            token: Some(token_id),
                            finish_reason: None,
                        };

                        if tx.send(Ok(chunk)).await.is_err() {
                            break; // Stream closed
                        }
                        
                        // 添加小延迟来演示流式效果
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }

                    // Check for stop sequences
                    if !sampling_params.stop_sequences.is_empty() {
                        let decoded_text = model.decode(&generated_tokens)?;
                        if sampling_params.stop_sequences.iter().any(|stop| decoded_text.contains(stop)) {
                            break;
                        }
                    }

                    // Check for EOS token
                    if token_id == 2 {
                        break;
                    }
                }

                // Send final chunk
                let final_chunk = StreamChunk {
                    request_id: request_id.clone(),
                    text: "".to_string(),
                    token: None,
                    finish_reason: Some(FinishReason::Length),
                };

                tx.send(Ok(final_chunk)).await.ok();
                Ok::<(), Error>(())
            }.await;

            if let Err(e) = result {
                tx.send(Err(e)).await.ok();
            }
        });

        // Convert to futures::Stream
        let stream = ReceiverStream::new(rx);
        Ok(Box::new(stream))
    }

    async fn get_status(&self) -> EngineStatus {
        let state = self.engine_state.read();
        let memory_usage = self.memory_manager.get_memory_usage();

        EngineStatus {
            is_ready: state.is_ready,
            loaded_models: state.loaded_models.clone(),
            active_requests: state.active_requests,
            memory_usage,
            uptime_seconds: state.start_time.elapsed().as_secs(),
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down engine...");

        // Signal shutdown
        {
            let mut shutdown = self.shutdown_signal.write();
            *shutdown = true;
        }

        // Wait for active requests to complete (with timeout)
        let timeout = tokio::time::Duration::from_secs(30);
        let start = tokio::time::Instant::now();

        loop {
            let active = {
                let state = self.engine_state.read();
                state.active_requests
            };

            if active == 0 {
                break;
            }

            if start.elapsed() > timeout {
                warn!("Shutdown timeout reached with {} active requests", active);
                break;
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        info!("Engine shutdown complete");
        Ok(())
    }
}

/// Generic ModelLoader implementation for MVP supporting multiple backends
pub struct GenericModelLoader {
    backend: Arc<dyn Backend>,
    loaded_models: Arc<RwLock<std::collections::HashMap<String, Arc<dyn Model>>>>,
}

impl GenericModelLoader {
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            loaded_models: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
}

#[async_trait]
impl ModelLoader for GenericModelLoader {
    async fn load_model(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        info!("Loading model: {}", config.model_id.0);
        
        // Use the model_path from config (should be resolved by new model maintenance system)
        let model_path = if config.model_path.is_empty() {
            warn!("Empty model path, using model ID as fallback: {}", config.model_id.0);
            config.model_id.0.as_str()
        } else {
            debug!("Using resolved model path: {}", config.model_path);
            config.model_path.as_str()
        };
        
        let model = Backend::load_weights(&*self.backend, model_path, config.dtype, &config.device).await?;
        let model_arc = Arc::from(model);
        
        // Cache the loaded model
        {
            let mut models = self.loaded_models.write();
            models.insert(config.model_id.0.clone(), Arc::clone(&model_arc));
        }
        
        Ok(model_arc)
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        info!("Unloading model: {}", model_id);
        let mut models = self.loaded_models.write();
        models.remove(model_id);
        Ok(())
    }

    async fn get_model(&self, model_id: &str) -> Option<Arc<dyn Model>> {
        let models = self.loaded_models.read();
        models.get(model_id).cloned()
    }

    async fn list_models(&self) -> Vec<ferrum_core::ModelInfo> {
        let models = self.loaded_models.read();
        models.values().map(|model| model.info().clone()).collect()
    }
}

/// Create a simplified Engine for MVP
pub async fn create_mvp_engine(config: EngineConfig) -> Result<Engine> {
            debug!("Creating MVP Engine with simplified components");

    // Parse device
    let device = match config.device.as_str() {
        "cpu" => ferrum_core::Device::CPU,
        s if s.starts_with("cuda:") => {
            let id = s.trim_start_matches("cuda:")
                .parse::<usize>()
                .map_err(|_| Error::internal("Invalid CUDA device ID"))?;
            ferrum_core::Device::CUDA(id)
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        "metal" => ferrum_core::Device::Metal,
        _ => ferrum_core::Device::CPU,
    };

    // Create and initialize backend based on device
    let backend: Arc<dyn Backend> = match device {
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        ferrum_core::Device::Metal => {
            debug!("Using Metal backend for Apple GPU acceleration");
            let mut backend = crate::metal::MetalBackend::new(device)?;
            Backend::initialize(&mut backend).await?;
            Arc::new(backend)
        }
        _ => {
            debug!("Using Candle backend");
            let mut backend = CandleBackend::new(device)?;
            Backend::initialize(&mut backend).await?;
            Arc::new(backend)
        }
    };

    // Create simplified components  
    let model_loader = Arc::new(GenericModelLoader::new(backend.clone()));
    let scheduler = Arc::new(SimpleScheduler::new());
    let batch_manager = Arc::new(SimpleBatchManager::new());
    let cache_manager = Arc::new(SimpleCacheManager::new());
    let memory_manager = Arc::new(SimpleMemoryManager::new());

    Engine::new(
        config,
        scheduler,
        batch_manager,
        cache_manager,
        memory_manager,
        model_loader,
    ).await
}

/// Simple scheduler implementation for MVP
pub struct SimpleScheduler {
    requests: Arc<RwLock<std::collections::VecDeque<InferenceRequest>>>,
}

impl SimpleScheduler {
    pub fn new() -> Self {
        Self {
            requests: Arc::new(RwLock::new(std::collections::VecDeque::new())),
        }
    }
}

#[async_trait]
impl Scheduler for SimpleScheduler {
    async fn schedule_request(&self, request: InferenceRequest) -> Result<ferrum_core::RequestId> {
        let mut queue = self.requests.write();
        let request_id = request.id.clone();
        queue.push_back(request);
        Ok(request_id)
    }

    async fn get_next_batch(&self) -> Option<ScheduledBatch> {
        let mut queue = self.requests.write();
        if let Some(request) = queue.pop_front() {
            Some(ScheduledBatch {
                batch_id: BatchId(uuid::Uuid::new_v4()),
                requests: vec![ScheduledRequest {
                    request,
                    state: RequestState::Running,
                    allocated_blocks: vec![],
                }],
                created_at: chrono::Utc::now(),
            })
        } else {
            None
        }
    }

    async fn preempt_request(&self, _request_id: ferrum_core::RequestId) -> Result<()> {
        // For MVP, don't support preemption
        Ok(())
    }

    async fn complete_request(&self, _request_id: ferrum_core::RequestId, _response: InferenceResponse) -> Result<()> {
        // For MVP, just acknowledge completion
        Ok(())
    }



    async fn get_stats(&self) -> SchedulerStats {
        SchedulerStats {
            waiting_requests: self.requests.read().len(),
            running_requests: 0,
            preempted_requests: 0,
            completed_requests: 0,
            failed_requests: 0,
            avg_wait_time_ms: 0.0,
            avg_execution_time_ms: 0.0,
        }
    }
}

/// Simple batch manager implementation for MVP
pub struct SimpleBatchManager;

impl SimpleBatchManager {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl BatchManager for SimpleBatchManager {
    async fn create_batch(&self, _requests: Vec<InferenceRequest>) -> Result<BatchId> {
        Ok(BatchId(uuid::Uuid::new_v4()))
    }

    async fn add_to_batch(&self, _batch_id: BatchId, _request: InferenceRequest) -> Result<()> {
        Ok(())
    }

    async fn remove_from_batch(&self, _batch_id: BatchId, _request_id: ferrum_core::RequestId) -> Result<()> {
        Ok(())
    }

    async fn execute_batch(&self, _batch_id: BatchId) -> Result<BatchOutput> {
        // For MVP, batch execution is handled directly in Engine
        Ok(BatchOutput {
            batch_id: BatchId(uuid::Uuid::new_v4()),
            outputs: std::collections::HashMap::new(),
        })
    }

    async fn get_batch_info(&self, _batch_id: BatchId) -> Option<BatchInfo> {
        None
    }
}

/// Simple cache manager implementation for MVP
pub struct SimpleCacheManager;

impl SimpleCacheManager {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CacheManager for SimpleCacheManager {
    async fn allocate_blocks(&self, _num_blocks: usize) -> Result<Vec<BlockId>> {
        Ok(vec![])
    }

    async fn free_blocks(&self, _block_ids: &[BlockId]) -> Result<()> {
        Ok(())
    }

    async fn get_block(&self, _block_id: BlockId) -> Option<KVBlock> {
        None
    }

    async fn update_block(&self, _block_id: BlockId, _block: KVBlock) -> Result<()> {
        Ok(())
    }

    fn get_stats(&self) -> CacheStats {
        CacheStats {
            total_blocks: 0,
            used_blocks: 0,
            free_blocks: 0,
            cache_hit_rate: 0.0,
            eviction_count: 0,
        }
    }

    async fn defragment(&self) -> Result<()> {
        Ok(())
    }
}

/// Simple memory manager implementation for MVP
pub struct SimpleMemoryManager;

impl SimpleMemoryManager {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl MemoryManager for SimpleMemoryManager {
    async fn allocate(&self, _size: usize) -> Result<MemoryHandle> {
        Ok(MemoryHandle(0))
    }

    async fn deallocate(&self, _handle: MemoryHandle) -> Result<()> {
        Ok(())
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            used_bytes: 0,
            free_bytes: 0,
            total_bytes: 0,
            gpu_memory_bytes: Some(0),
            cpu_memory_bytes: Some(0),
        }
    }

    async fn swap_out(&self, _handle: MemoryHandle) -> Result<()> {
        Ok(())
    }

    async fn swap_in(&self, _handle: MemoryHandle) -> Result<()> {
        Ok(())
    }

    fn set_pressure_callback(&self, _callback: Box<dyn Fn(MemoryPressure) + Send + Sync>) {
        // For MVP, ignore pressure callbacks
    }
}
