//! Model execution interface with clear prefill/decode separation
//!
//! This module provides the ModelExecutor trait that replaces the "fat" Model
//! interface, focusing purely on tensor operations without tokenization or sampling.

use crate::{KvCacheHandle, TensorRef};
use ferrum_types::{ModelInfo, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

/// Input for prefill phase (processing the initial prompt)
#[derive(Debug, Clone)]
pub struct PrefillInput {
    /// Input token IDs [batch_size, sequence_length]
    pub input_ids: TensorRef,
    /// Attention mask [batch_size, sequence_length] (optional)
    pub attention_mask: Option<TensorRef>,
    /// Position IDs [batch_size, sequence_length] (optional, for RoPE)
    pub position_ids: Option<TensorRef>,
}

impl PrefillInput {
    /// Create new prefill input
    pub fn new(input_ids: TensorRef) -> Self {
        Self {
            input_ids,
            attention_mask: None,
            position_ids: None,
        }
    }
    
    /// Add attention mask
    pub fn with_attention_mask(mut self, mask: TensorRef) -> Self {
        self.attention_mask = Some(mask);
        self
    }
    
    /// Add position IDs
    pub fn with_position_ids(mut self, positions: TensorRef) -> Self {
        self.position_ids = Some(positions);
        self
    }
    
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.shape()[0]
    }
    
    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        if self.input_ids.shape().len() >= 2 {
            self.input_ids.shape()[1]
        } else {
            1
        }
    }
}

/// Output from prefill phase
#[derive(Debug, Clone)]
pub struct PrefillOutput {
    /// Logits for all positions [batch_size, sequence_length, vocab_size]
    pub logits: TensorRef,
    /// KV cache handle populated with prompt states
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Hidden states at each layer (optional, for analysis)
    pub hidden_states: Option<Vec<TensorRef>>,
    /// Attention weights (optional, for analysis)
    pub attention_weights: Option<Vec<TensorRef>>,
}

impl PrefillOutput {
    /// Create new prefill output
    pub fn new(logits: TensorRef, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        Self {
            logits,
            kv_cache,
            hidden_states: None,
            attention_weights: None,
        }
    }
    
    /// Get logits for last position (for next token generation)
    pub fn last_token_logits(&self) -> Result<TensorRef> {
        let shape = self.logits.shape();
        if shape.len() != 3 {
            return Err(ferrum_types::FerrumError::backend(
                "Expected 3D logits tensor [batch, seq, vocab]"
            ));
        }
        
        let seq_len = shape[1];
        if seq_len == 0 {
            return Err(ferrum_types::FerrumError::backend("Empty sequence"));
        }
        
        // Extract last position: [batch, seq-1:seq, vocab] -> [batch, vocab]
        self.logits.view(&[0, seq_len - 1, 0], &[shape[0], seq_len, shape[2]])
    }
}

/// Input for decode phase (generating one token at a time)
#[derive(Debug, Clone)]
pub struct DecodeInput {
    /// Input token ID for current step [batch_size, 1]
    pub input_ids: TensorRef,
    /// Existing KV cache from previous steps
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Position IDs for current step [batch_size, 1] (optional)
    pub position_ids: Option<TensorRef>,
}

impl DecodeInput {
    /// Create new decode input
    pub fn new(input_ids: TensorRef, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        Self {
            input_ids,
            kv_cache,
            position_ids: None,
        }
    }
    
    /// Add position IDs
    pub fn with_position_ids(mut self, positions: TensorRef) -> Self {
        self.position_ids = Some(positions);
        self
    }
    
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.shape()[0]
    }
}

/// Output from decode phase
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    /// Logits for next token [batch_size, vocab_size]
    pub logits: TensorRef,
    /// Updated KV cache with new token state
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Hidden state for current token (optional)
    pub hidden_state: Option<TensorRef>,
    /// Attention weights for current token (optional)
    pub attention_weights: Option<Vec<TensorRef>>,
}

impl DecodeOutput {
    /// Create new decode output
    pub fn new(logits: TensorRef, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        Self {
            logits,
            kv_cache,
            hidden_state: None,
            attention_weights: None,
        }
    }
}

/// Core model executor trait focusing on tensor operations
#[async_trait]
pub trait ModelExecutor: Send + Sync {
    /// Get model information and metadata
    fn info(&self) -> &ModelInfo;
    
    /// Execute prefill phase (process initial prompt)
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput>;
    
    /// Execute decode phase (generate next token)
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput>;
    
    /// Optional: full forward pass (for non-autoregressive use cases)
    async fn forward(&self, input: &TensorRef) -> Result<TensorRef> {
        // Default implementation not supported
        Err(ferrum_types::FerrumError::unsupported(
            "Full forward pass not supported by this executor"
        ))
    }
    
    /// Get executor capabilities
    fn capabilities(&self) -> ExecutorCapabilities;
    
    /// Get current executor status
    fn status(&self) -> ExecutorStatus;
    
    /// Warm up executor (load model, allocate memory, etc.)
    async fn warmup(&mut self) -> Result<()> {
        // Default no-op implementation
        Ok(())
    }
    
    /// Shutdown executor gracefully
    async fn shutdown(&mut self) -> Result<()> {
        // Default no-op implementation
        Ok(())
    }
}

/// Executor capabilities and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorCapabilities {
    /// Maximum supported batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Supported attention mechanisms
    pub attention_mechanisms: Vec<AttentionType>,
    /// Whether executor supports dynamic batching
    pub supports_dynamic_batching: bool,
    /// Whether executor supports continuous batching
    pub supports_continuous_batching: bool,
    /// Whether executor supports speculative decoding
    pub supports_speculative_decoding: bool,
    /// Whether executor supports tensor parallelism
    pub supports_tensor_parallelism: bool,
    /// Whether executor supports pipeline parallelism
    pub supports_pipeline_parallelism: bool,
    /// Supported data types
    pub supported_dtypes: Vec<ferrum_types::DataType>,
    /// Supported devices
    pub supported_devices: Vec<ferrum_types::Device>,
    /// Memory requirements estimation
    pub memory_requirements: MemoryRequirements,
}

/// Attention mechanism types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard multi-head attention
    MultiHead,
    /// Multi-query attention (MQA)
    MultiQuery,
    /// Grouped-query attention (GQA)
    GroupedQuery,
    /// Flash attention
    Flash,
    /// Paged attention
    Paged,
    /// Sliding window attention
    SlidingWindow,
}

/// Memory requirements for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Model parameter memory in bytes
    pub parameter_memory: u64,
    /// Minimum activation memory per token
    pub activation_memory_per_token: usize,
    /// KV cache memory per token per layer
    pub kv_cache_memory_per_token: usize,
    /// Additional overhead memory
    pub overhead_memory: u64,
}

impl MemoryRequirements {
    /// Calculate total memory for given configuration
    pub fn calculate_total_memory(
        &self,
        batch_size: usize,
        sequence_length: usize,
        num_layers: usize,
    ) -> u64 {
        let activation_mem = (self.activation_memory_per_token * batch_size * sequence_length) as u64;
        let kv_cache_mem = (self.kv_cache_memory_per_token * batch_size * sequence_length * num_layers) as u64;
        
        self.parameter_memory + activation_mem + kv_cache_mem + self.overhead_memory
    }
}

/// Executor status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorStatus {
    /// Current executor state
    pub state: ExecutorState,
    /// Whether executor is ready to accept requests
    pub is_ready: bool,
    /// Current batch size being processed
    pub current_batch_size: usize,
    /// Number of prefill operations completed
    pub prefill_operations: u64,
    /// Number of decode operations completed
    pub decode_operations: u64,
    /// Average prefill time in milliseconds
    pub avg_prefill_time_ms: f64,
    /// Average decode time in milliseconds
    pub avg_decode_time_ms: f64,
    /// Memory usage statistics
    pub memory_usage: ExecutorMemoryUsage,
    /// Last operation timestamp
    #[serde(skip)]
    pub last_operation: Option<std::time::Instant>,
}

/// Executor state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorState {
    /// Executor is initializing
    Initializing,
    /// Executor is ready to accept requests
    Ready,
    /// Executor is processing requests
    Busy,
    /// Executor encountered an error
    Error,
    /// Executor is shutting down
    Shutdown,
}

/// Executor memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorMemoryUsage {
    /// Total allocated memory in bytes
    pub allocated_bytes: usize,
    /// Currently used memory in bytes
    pub used_bytes: usize,
    /// Peak memory usage
    pub peak_bytes: usize,
    /// Memory utilization percentage
    pub utilization_percent: f32,
}

/// Batch model executor for processing multiple requests efficiently
#[async_trait]
pub trait BatchModelExecutor: ModelExecutor {
    /// Execute batch prefill for multiple sequences
    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>>;
    
    /// Execute batch decode for multiple sequences
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>>;
    
    /// Get optimal batch size for current conditions
    fn optimal_batch_size(&self) -> usize;
    
    /// Check if batch size is supported
    fn supports_batch_size(&self, batch_size: usize) -> bool;
}

/// Speculative execution support
#[async_trait]
pub trait SpeculativeExecutor: ModelExecutor {
    /// Execute speculative decoding with draft model
    async fn speculative_decode(
        &self,
        input: &DecodeInput,
        draft_tokens: &[ferrum_types::TokenId],
        acceptance_threshold: f32,
    ) -> Result<SpeculativeDecodeOutput>;
}

/// Output from speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeDecodeOutput {
    /// Accepted tokens (subset of draft tokens)
    pub accepted_tokens: Vec<ferrum_types::TokenId>,
    /// Logits for the next token after last accepted
    pub next_logits: TensorRef,
    /// Updated KV cache
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Number of draft tokens accepted
    pub acceptance_count: usize,
}

/// Model executor factory
#[async_trait]
pub trait ModelExecutorFactory: Send + Sync {
    /// Create executor from model configuration
    async fn create_executor(
        &self,
        config: &ExecutorConfig,
    ) -> Result<Box<dyn ModelExecutor>>;
    
    /// Create batch executor
    async fn create_batch_executor(
        &self,
        config: &ExecutorConfig,
    ) -> Result<Box<dyn BatchModelExecutor>>;
    
    /// Get supported executor types
    fn supported_types(&self) -> Vec<ExecutorType>;
    
    /// Validate configuration
    fn validate_config(&self, config: &ExecutorConfig) -> Result<()>;
}

/// Executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Model information
    pub model_info: ModelInfo,
    /// Target device
    pub device: ferrum_types::Device,
    /// Data type for computation
    pub dtype: ferrum_types::DataType,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Attention configuration
    pub attention_config: AttentionConfig,
    /// Memory configuration
    pub memory_config: ExecutorMemoryConfig,
    /// Optimization settings
    pub optimization_config: OptimizationConfig,
    /// Additional executor-specific options
    pub executor_options: HashMap<String, serde_json::Value>,
}

/// Attention mechanism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Type of attention to use
    pub attention_type: AttentionType,
    /// Enable flash attention if available
    pub enable_flash_attention: bool,
    /// Enable paged attention
    pub enable_paged_attention: bool,
    /// Block size for paged attention
    pub block_size: Option<usize>,
    /// Sliding window size (if using sliding window attention)
    pub sliding_window_size: Option<usize>,
}

/// Memory configuration for executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorMemoryConfig {
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Memory pool size in bytes (None for auto)
    pub memory_pool_size: Option<usize>,
    /// Enable KV cache sharing
    pub enable_kv_cache_sharing: bool,
    /// Maximum memory usage percentage
    pub max_memory_usage: f32,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable CUDA graphs (if supported)
    pub enable_cuda_graphs: bool,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Enable mixed precision
    pub enable_mixed_precision: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Custom optimization flags
    pub custom_flags: HashMap<String, bool>,
}

/// Supported executor types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutorType {
    /// Standard sequential executor
    Sequential,
    /// Batch executor for parallel processing
    Batch,
    /// Continuous batching executor
    ContinuousBatch,
    /// Speculative decoding executor
    Speculative,
    /// Pipeline parallel executor
    PipelineParallel,
    /// Tensor parallel executor
    TensorParallel,
}

/// Executor performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorMetrics {
    /// Total operations executed
    pub total_operations: u64,
    /// Prefill operations
    pub prefill_operations: u64,
    /// Decode operations
    pub decode_operations: u64,
    /// Average prefill latency (ms)
    pub avg_prefill_latency: f64,
    /// Average decode latency (ms)
    pub avg_decode_latency: f64,
    /// P95 prefill latency (ms)
    pub p95_prefill_latency: f64,
    /// P95 decode latency (ms)
    pub p95_decode_latency: f64,
    /// Throughput (tokens per second)
    pub throughput_tps: f64,
    /// Memory efficiency (used/allocated)
    pub memory_efficiency: f32,
    /// Batch utilization
    pub batch_utilization: f32,
}

/// Executor registry for managing multiple executors
pub trait ExecutorRegistry: Send + Sync {
    /// Register executor with name
    fn register(
        &mut self,
        name: &str,
        executor: Box<dyn ModelExecutor>,
    ) -> Result<()>;
    
    /// Get executor by name
    fn get(&self, name: &str) -> Option<&dyn ModelExecutor>;
    
    /// Remove executor by name
    fn remove(&mut self, name: &str) -> Option<Box<dyn ModelExecutor>>;
    
    /// List registered executor names
    fn list_names(&self) -> Vec<String>;
    
    /// Get executor metrics
    fn get_metrics(&self, name: &str) -> Option<ExecutorMetrics>;
}
