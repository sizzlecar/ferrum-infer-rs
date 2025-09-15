//! Core type definitions for the Ferrum inference framework

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ==================== Basic ID Types ====================

/// Request identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(pub Uuid);

impl RequestId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Batch identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BatchId(pub Uuid);

/// Model identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// Cache block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

/// Token identifier type
pub type TokenId = u32;

// ==================== Request and Response Types ====================

/// Inference request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: RequestId,
    pub prompt: String,
    pub model_id: ModelId,
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub priority: Priority,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Inference response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: RequestId,
    pub text: String,
    pub tokens: Vec<TokenId>,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
    pub latency_ms: u64,
    pub created_at: DateTime<Utc>,
}

/// Streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub request_id: RequestId,
    pub text: String,
    pub token: Option<TokenId>,
    pub finish_reason: Option<FinishReason>,
}

/// Sampling parameters for generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: vec![],
            seed: None,
        }
    }
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Reason for completion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Length,
    Stop,
    EOS,
    Cancelled,
    Error,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ==================== Model Types ====================

/// Model configuration (Runtime/Execution layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub model_id: ModelId,
    pub model_path: String,
    pub model_type: ModelType,
    pub dtype: DataType,
    pub device: Device,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub tensor_parallel_size: Option<usize>,
    pub pipeline_parallel_size: Option<usize>,
    pub quantization: Option<QuantizationConfig>,
    
    // Runtime attention configuration
    pub use_flash_attention: bool,
    pub use_paged_attention: bool,
}

/// Model type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    Llama,
    Mistral,
    Qwen,
    Custom(String),
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: ModelId,
    pub model_type: ModelType,
    pub num_parameters: u64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub dtype: DataType,
    pub device: Device,
}

/// Data type for tensors
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataType {
    FP32,
    FP16,
    BF16,
    INT8,
    FP8,
}

/// Device type for computation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    CPU,
    CUDA(usize),
    ROCm(usize),
    /// Apple GPU using Metal Performance Shaders
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal,
}

// ==================== Scheduling Types ====================

/// Scheduled batch of requests
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub batch_id: BatchId,
    pub requests: Vec<ScheduledRequest>,
    pub created_at: DateTime<Utc>,
}

/// Scheduled request with state
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub request: InferenceRequest,
    pub state: RequestState,
    pub allocated_blocks: Vec<BlockId>,
}

/// Request state in scheduler
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestState {
    Waiting,
    Running,
    Preempted,
    Completed,
    Failed,
}

/// Scheduler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub waiting_requests: usize,
    pub running_requests: usize,
    pub preempted_requests: usize,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub avg_wait_time_ms: f64,
    pub avg_execution_time_ms: f64,
}

// ==================== Cache Types ====================

/// Key-Value cache for attention
#[derive(Debug, Clone)]
pub struct KVCache {
    pub key_cache: Vec<Tensor>,
    pub value_cache: Vec<Tensor>,
    pub sequence_length: usize,
}

/// KV cache block
#[derive(Debug, Clone)]
pub struct KVBlock {
    pub block_id: BlockId,
    pub token_ids: Vec<TokenId>,
    pub key_cache: Tensor,
    pub value_cache: Tensor,
    pub ref_count: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_blocks: usize,
    pub used_blocks: usize,
    pub free_blocks: usize,
    pub cache_hit_rate: f32,
    pub eviction_count: u64,
}

// ==================== Batch Processing Types ====================

/// Batch information
#[derive(Debug, Clone)]
pub struct BatchInfo {
    pub batch_id: BatchId,
    pub requests: Vec<RequestId>,
    pub max_sequence_length: usize,
    pub created_at: DateTime<Utc>,
}

/// Batch output results
#[derive(Debug, Clone)]
pub struct BatchOutput {
    pub batch_id: BatchId,
    pub outputs: HashMap<RequestId, GenerateOutput>,
}

/// Generation output for a single token
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    pub token_id: TokenId,
    pub logits: Tensor,
    pub kv_cache: Option<KVCache>,
}

// ==================== Memory Management Types ====================

/// Memory handle for allocation tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryHandle(pub u64);

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total_bytes: usize,
    pub used_bytes: usize,
    pub free_bytes: usize,
    pub gpu_memory_bytes: Option<usize>,
    pub cpu_memory_bytes: Option<usize>,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

// ==================== Executor Types ====================

/// Execution task for inference
#[derive(Debug, Clone)]
pub struct ExecutionTask {
    pub task_id: Uuid,
    pub task_type: TaskType,
    pub input: Tensor,
    pub model_id: ModelId,
    pub batch_size: usize,
}

/// Task type for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Initial prompt processing
    Prefill,
    /// Token-by-token generation
    Decode,
    /// Forward pass through model
    ModelForward,
    /// Token sampling
    Sampling,
    /// Input preprocessing
    Preprocessing,
    /// Output postprocessing
    Postprocessing,
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub task_id: Uuid,
    pub output: Tensor,
    pub execution_time_ms: u64,
}

/// Executor capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorCapabilities {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub supports_flash_attention: bool,
    pub supports_paged_attention: bool,
    pub supports_continuous_batching: bool,
    pub supports_tensor_parallelism: bool,
    pub supports_pipeline_parallelism: bool,
    pub supported_dtypes: Vec<DataType>,
}

/// Executor status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorStatus {
    Idle,
    Running,
    Busy,
    Error,
}

/// Detailed executor state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorState {
    pub status: ExecutorStatus,
    pub is_ready: bool,
    pub current_batch_size: usize,
    pub queued_tasks: usize,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
}

// ==================== Engine Status ====================

/// Engine status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub is_ready: bool,
    pub loaded_models: Vec<ModelId>,
    pub active_requests: usize,
    pub memory_usage: MemoryUsage,
    pub uptime_seconds: u64,
}

// ==================== Quantization Types ====================

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationConfig {
    GPTQ { bits: u8, group_size: usize },
    AWQ { bits: u8, zero_point: bool },
    FP8 { e4m3: bool },
    INT8 { symmetric: bool },
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<u8>,
    pub scale: f32,
    pub zero_point: Option<f32>,
    pub config: QuantizationConfig,
}

// ==================== Tensor Types (Simplified) ====================

/// Simplified tensor type (should use dedicated tensor library in production)
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DataType,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            dtype: DataType::FP32,
        }
    }
}
