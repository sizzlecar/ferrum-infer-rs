//! Core trait definitions for the Ferrum inference framework

use async_trait::async_trait;
use std::sync::Arc;
use crate::{Result, types::*};

/// Core trait for inference engine
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Execute an inference request
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;
    
    /// Execute streaming inference
    async fn infer_stream(
        &self, 
        request: InferenceRequest
    ) -> Result<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send + Unpin>>;
    
    /// Get engine status
    async fn get_status(&self) -> EngineStatus;
    
    /// Shutdown the engine gracefully
    async fn shutdown(&self) -> Result<()>;
}

/// Model loading and management trait
#[async_trait]
pub trait ModelLoader: Send + Sync {
    /// Load a model with given configuration
    async fn load_model(&self, config: &ModelConfig) -> Result<Arc<dyn Model>>;
    
    /// Unload a model by ID
    async fn unload_model(&self, model_id: &str) -> Result<()>;
    
    /// Get a loaded model by ID
    async fn get_model(&self, model_id: &str) -> Option<Arc<dyn Model>>;
    
    /// List all loaded models
    async fn list_models(&self) -> Vec<ModelInfo>;
}

/// Model trait for LLM operations
#[async_trait]
pub trait Model: Send + Sync {
    /// Get model information
    fn info(&self) -> &ModelInfo;
    
    /// Forward pass through the model
    async fn forward(&self, input: &Tensor) -> Result<Tensor>;
    
    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    
    /// Decode token IDs to text
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
    
    /// Generate next token given input and KV cache
    async fn generate_next_token(
        &self,
        input_ids: &[TokenId],
        past_kv: Option<&KVCache>,
        sampling_params: &SamplingParams,
    ) -> Result<GenerateOutput>;
}

/// Request scheduler trait for managing inference queue
#[async_trait]
pub trait Scheduler: Send + Sync {
    /// Schedule a new inference request
    async fn schedule_request(&self, request: InferenceRequest) -> Result<RequestId>;
    
    /// Get next batch of requests to execute
    async fn get_next_batch(&self) -> Option<ScheduledBatch>;
    
    /// Preempt a running request
    async fn preempt_request(&self, request_id: RequestId) -> Result<()>;
    
    /// Mark request as completed
    async fn complete_request(&self, request_id: RequestId, response: InferenceResponse) -> Result<()>;
    
    /// Get scheduler statistics
    async fn get_stats(&self) -> SchedulerStats;
}

/// Cache manager trait for KV cache blocks
#[async_trait]
pub trait CacheManager: Send + Sync {
    /// Allocate cache blocks
    async fn allocate_blocks(&self, num_blocks: usize) -> Result<Vec<BlockId>>;
    
    /// Free cache blocks
    async fn free_blocks(&self, block_ids: &[BlockId]) -> Result<()>;
    
    /// Get a cache block by ID
    async fn get_block(&self, block_id: BlockId) -> Option<KVBlock>;
    
    /// Update a cache block
    async fn update_block(&self, block_id: BlockId, block: KVBlock) -> Result<()>;
    
    /// Get cache statistics
    fn get_stats(&self) -> CacheStats;
    
    /// Defragment cache memory
    async fn defragment(&self) -> Result<()>;
}

/// Batch manager trait for continuous batching
#[async_trait]
pub trait BatchManager: Send + Sync {
    /// Create a new batch from requests
    async fn create_batch(&self, requests: Vec<InferenceRequest>) -> Result<BatchId>;
    
    /// Add request to existing batch
    async fn add_to_batch(&self, batch_id: BatchId, request: InferenceRequest) -> Result<()>;
    
    /// Remove request from batch
    async fn remove_from_batch(&self, batch_id: BatchId, request_id: RequestId) -> Result<()>;
    
    /// Execute batch inference
    async fn execute_batch(&self, batch_id: BatchId) -> Result<BatchOutput>;
    
    /// Get batch information
    async fn get_batch_info(&self, batch_id: BatchId) -> Option<BatchInfo>;
}

/// Memory manager trait for GPU/CPU memory
#[async_trait]
pub trait MemoryManager: Send + Sync {
    /// Allocate memory
    async fn allocate(&self, size: usize) -> Result<MemoryHandle>;
    
    /// Deallocate memory
    async fn deallocate(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Get current memory usage
    fn get_memory_usage(&self) -> MemoryUsage;
    
    /// Swap memory from GPU to CPU
    async fn swap_out(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Swap memory from CPU to GPU
    async fn swap_in(&self, handle: MemoryHandle) -> Result<()>;
    
    /// Set callback for memory pressure events
    fn set_pressure_callback(&self, callback: Box<dyn Fn(MemoryPressure) + Send + Sync>);
}

/// Executor trait for running inference tasks
#[async_trait]
pub trait Executor: Send + Sync {
    /// Execute an inference task
    async fn execute(&self, task: ExecutionTask) -> Result<ExecutionResult>;
    
    /// Get executor capabilities
    fn capabilities(&self) -> ExecutorCapabilities;
    
    /// Get executor status
    fn status(&self) -> ExecutorStatus;
}

/// Quantization trait for model compression
pub trait Quantization: Send + Sync {
    /// Quantize a tensor
    fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor>;
    
    /// Dequantize a tensor
    fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Tensor>;
    
    /// Get quantization configuration
    fn config(&self) -> &QuantizationConfig;
}
