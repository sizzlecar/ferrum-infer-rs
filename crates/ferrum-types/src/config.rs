//! Configuration types for Ferrum components

use crate::{DataType, Device, ModelId, ModelInfo, SamplingParams};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model: EngineModelConfig,
    pub scheduler: SchedulerConfig,
    pub sampling: SamplingConfig,
    pub backend: BackendConfig,
    pub kv_cache: KvCacheConfig,
    pub memory: MemoryConfig,
    pub batching: BatchConfig,
    pub monitoring: MonitoringConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: EngineModelConfig::default(),
            scheduler: SchedulerConfig::default(),
            sampling: SamplingConfig::default(),
            backend: BackendConfig::default(),
            kv_cache: KvCacheConfig::default(),
            memory: MemoryConfig::default(),
            batching: BatchConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineModelConfig {
    pub model_id: ModelId,
    pub model_info: Option<ModelInfo>,
    pub tokenizer: TokenizerConfig,
}

impl Default for EngineModelConfig {
    fn default() -> Self {
        Self {
            model_id: ModelId::new("default"),
            model_info: None,
            tokenizer: TokenizerConfig::default(),
        }
    }
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Maximum waiting queue size
    pub max_waiting_requests: usize,
    /// Maximum running requests
    pub max_running_requests: usize,
    /// Enable request preemption
    pub enable_preemption: bool,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Fair share weights per client
    pub fair_share_weights: HashMap<String, f32>,
    /// SLA enforcement enabled
    pub enable_sla_enforcement: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Priority,
            max_waiting_requests: 1000,
            max_running_requests: 256,
            enable_preemption: true,
            enable_load_balancing: false,
            fair_share_weights: HashMap::new(),
            enable_sla_enforcement: false,
        }
    }
}

/// Scheduling policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served
    FCFS,
    /// Priority-based scheduling
    Priority,
    /// Fair-share scheduling
    FairShare,
    /// Shortest-Job-First
    SJF,
    /// Round-Robin
    RoundRobin,
}

/// KV Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Cache implementation type
    pub cache_type: KvCacheType,
    /// Block size for paged attention
    pub block_size: usize,
    /// Maximum number of blocks
    pub max_blocks: usize,
    /// Enable cache compression
    pub enable_compression: bool,
    /// Compression ratio target
    pub compression_ratio: f32,
    /// Enable multi-level caching (GPU + CPU)
    pub enable_multi_level: bool,
    /// Swap threshold (when to move to CPU)
    pub swap_threshold: f32,
    /// Enable prefix caching
    pub enable_prefix_caching: bool,
    /// Prefix cache size
    pub prefix_cache_size: usize,
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            cache_type: KvCacheType::Contiguous,
            block_size: 16,
            max_blocks: 1024,
            enable_compression: false,
            compression_ratio: 0.5,
            enable_multi_level: true,
            swap_threshold: 0.8,
            enable_prefix_caching: true,
            prefix_cache_size: 100,
        }
    }
}

/// KV Cache implementation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum KvCacheType {
    /// Simple contiguous memory allocation
    Contiguous,
    /// Paged attention with block-based allocation
    Paged,
    /// Tree-based cache for prefix sharing
    Tree,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory pool size in bytes
    pub pool_size: Option<usize>,
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Memory alignment in bytes
    pub alignment: usize,
    /// Enable memory defragmentation
    pub enable_defragmentation: bool,
    /// Defragmentation threshold
    pub defragmentation_threshold: f32,
    /// Enable memory statistics tracking
    pub enable_memory_stats: bool,
    /// Memory pressure warning threshold
    pub pressure_warning_threshold: f32,
    /// Memory pressure critical threshold
    pub pressure_critical_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            pool_size: None,
            enable_pooling: true,
            alignment: 256,
            enable_defragmentation: false,
            defragmentation_threshold: 0.7,
            enable_memory_stats: true,
            pressure_warning_threshold: 0.8,
            pressure_critical_threshold: 0.95,
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type
    pub backend_type: BackendType,
    /// Target device
    pub device: Device,
    /// Data type for computation
    pub dtype: DataType,
    /// Enable optimizations
    pub enable_optimizations: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable CUDA graphs
    pub enable_cuda_graphs: bool,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Custom backend-specific options
    pub backend_options: HashMap<String, serde_json::Value>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Candle,
            device: Device::Cpu,
            dtype: DataType::F16,
            enable_optimizations: true,
            optimization_level: 2,
            enable_cuda_graphs: false,
            enable_kernel_fusion: true,
            backend_options: HashMap::new(),
        }
    }
}

/// Supported backend types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BackendType {
    /// Candle framework
    Candle,
    /// ONNX Runtime
    OnnxRuntime,
    /// TensorRT
    TensorRT,
    /// Custom backend
    Custom,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
    /// Path to tokenizer files
    pub tokenizer_path: Option<String>,
    /// Enable fast tokenization
    pub enable_fast: bool,
    /// Add special tokens
    pub add_special_tokens: bool,
    /// Truncation strategy
    pub truncation: Option<TruncationConfig>,
    /// Padding strategy
    pub padding: Option<PaddingConfig>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_type: TokenizerType::BPE,
            tokenizer_path: None,
            enable_fast: true,
            add_special_tokens: true,
            truncation: None,
            padding: None,
        }
    }
}

/// Truncation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationConfig {
    /// Maximum length
    pub max_length: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
}

/// Truncation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Remove from the beginning
    TruncateStart,
    /// Remove from the end
    TruncateEnd,
    /// Remove from both sides
    TruncateBoth,
}

/// Padding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingConfig {
    /// Padding strategy
    pub strategy: PaddingStrategy,
    /// Padding token ID
    pub token_id: u32,
    /// Target length
    pub target_length: Option<usize>,
}

/// Padding strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to maximum length in batch
    MaxLength,
    /// Pad to specific length
    FixedLength,
}

/// Sampling configuration presets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingPresets {
    /// Available presets
    pub presets: HashMap<String, SamplingParams>,
}

impl Default for SamplingPresets {
    fn default() -> Self {
        let mut presets = HashMap::new();
        presets.insert("greedy".to_string(), SamplingParams::greedy());
        presets.insert(
            "creative".to_string(),
            SamplingParams {
                temperature: 1.2,
                top_p: 0.9,
                top_k: Some(50),
                repetition_penalty: 1.1,
                ..Default::default()
            },
        );
        presets.insert(
            "precise".to_string(),
            SamplingParams {
                temperature: 0.3,
                top_p: 0.95,
                top_k: Some(20),
                repetition_penalty: 1.05,
                ..Default::default()
            },
        );
        Self { presets }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable API authentication
    pub enable_auth: bool,
    /// API keys for authentication
    pub api_keys: Vec<String>,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Rate limit per client (requests per minute)
    pub rate_limit_rpm: u32,
    /// Enable content filtering
    pub enable_content_filter: bool,
    /// Maximum prompt length
    pub max_prompt_length: usize,
    /// Enable prompt validation
    pub enable_prompt_validation: bool,
    /// Allowed file extensions for uploads
    pub allowed_extensions: Vec<String>,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_auth: false,
            api_keys: vec![],
            enable_rate_limiting: true,
            rate_limit_rpm: 60,
            enable_content_filter: false,
            max_prompt_length: 32768,
            enable_prompt_validation: true,
            allowed_extensions: vec!["txt".to_string(), "json".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub default_params: SamplingParams,
    pub presets: SamplingPresets,
    pub enable_custom_processors: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            default_params: SamplingParams::default(),
            presets: SamplingPresets::default(),
            enable_custom_processors: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub export_interval: Duration,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            export_interval: Duration::from_secs(5),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_wait_ms: u64,
    pub enable_dynamic: bool,
    pub enable_continuous: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 16,
            max_wait_ms: 8,
            enable_dynamic: true,
            enable_continuous: false,
        }
    }
}
