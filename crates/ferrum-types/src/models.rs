//! Model-related types and configurations

use crate::{ids::ModelId, devices::*, FerrumError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// LLaMA family models
    Llama,
    /// Mistral family models
    Mistral,
    /// Qwen family models
    Qwen,
    /// Phi family models
    Phi,
    /// Gemma family models
    Gemma,
    /// Code-specific models
    Code(String),
    /// Custom model implementation
    Custom(String),
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Llama => write!(f, "llama"),
            ModelType::Mistral => write!(f, "mistral"),
            ModelType::Qwen => write!(f, "qwen"),
            ModelType::Phi => write!(f, "phi"),
            ModelType::Gemma => write!(f, "gemma"),
            ModelType::Code(name) => write!(f, "code-{}", name),
            ModelType::Custom(name) => write!(f, "custom-{}", name),
        }
    }
}

/// Model information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub model_id: ModelId,
    /// Model type/architecture
    pub model_type: ModelType,
    /// Number of parameters
    pub num_parameters: u64,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Data type used by the model
    pub dtype: DataType,
    /// Device where model is loaded
    pub device: Device,
    /// Model version or revision
    pub version: Option<String>,
    /// Model license
    pub license: Option<String>,
    /// Additional model metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ModelInfo {
    /// Calculate approximate model size in bytes
    pub fn estimated_size_bytes(&self) -> u64 {
        // Rough estimation: parameters * dtype size + some overhead
        let param_size = self.num_parameters * self.dtype.size_bytes() as u64;
        // Add ~20% overhead for embeddings, activations, etc.
        (param_size as f64 * 1.2) as u64
    }

    /// Check if model supports a specific sequence length
    pub fn supports_sequence_length(&self, length: usize) -> bool {
        length <= self.max_sequence_length
    }

    /// Get memory requirements for inference
    pub fn memory_requirements(&self, batch_size: usize, sequence_length: usize) -> ModelMemoryRequirements {
        let param_memory = self.estimated_size_bytes();
        
        // Estimate KV cache size: layers * heads * seq_len * head_dim * 2 (key + value) * dtype * batch_size
        let head_dim = self.hidden_size / self.num_heads;
        let kv_cache_per_token = self.num_layers * self.num_kv_heads * head_dim * 2 * self.dtype.size_bytes();
        let kv_cache_memory = (kv_cache_per_token * sequence_length * batch_size) as u64;
        
        // Estimate activation memory (rough approximation)
        let activation_memory = (self.hidden_size * sequence_length * batch_size * self.dtype.size_bytes()) as u64 * 4;

        ModelMemoryRequirements {
            parameter_memory: param_memory,
            kv_cache_memory,
            activation_memory,
            total_estimated: param_memory + kv_cache_memory + activation_memory,
        }
    }
}

/// Memory requirements for model inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMemoryRequirements {
    /// Memory required for model parameters
    pub parameter_memory: u64,
    /// Memory required for KV cache
    pub kv_cache_memory: u64,
    /// Memory required for activations
    pub activation_memory: u64,
    /// Total estimated memory requirement
    pub total_estimated: u64,
}

/// Model configuration for runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model identifier
    pub model_id: ModelId,
    /// Path to model files
    pub model_path: String,
    /// Model type/architecture
    pub model_type: ModelType,
    /// Data type to use for inference
    pub dtype: DataType,
    /// Target device
    pub device: Device,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Tensor parallelism size
    pub tensor_parallel_size: Option<usize>,
    /// Pipeline parallelism size  
    pub pipeline_parallel_size: Option<usize>,
    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,
    /// Use flash attention if available
    pub use_flash_attention: bool,
    /// Use paged attention for KV cache
    pub use_paged_attention: bool,
    /// Enable CUDA graphs for optimization
    pub enable_cuda_graphs: bool,
    /// Additional configuration parameters
    pub extra_config: HashMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Create a new model configuration
    pub fn new(model_id: impl Into<ModelId>, model_path: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            model_path: model_path.into(),
            model_type: ModelType::Custom("unknown".to_string()),
            dtype: DataType::FP16,
            device: Device::CPU,
            max_batch_size: 1,
            max_sequence_length: 2048,
            tensor_parallel_size: None,
            pipeline_parallel_size: None,
            quantization: None,
            use_flash_attention: false,
            use_paged_attention: false,
            enable_cuda_graphs: false,
            extra_config: HashMap::new(),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.model_path.is_empty() {
            return Err(FerrumError::config("Model path cannot be empty"));
        }
        
        if self.max_batch_size == 0 {
            return Err(FerrumError::config("Max batch size must be positive"));
        }
        
        if self.max_sequence_length == 0 {
            return Err(FerrumError::config("Max sequence length must be positive"));
        }

        if let Some(tp_size) = self.tensor_parallel_size {
            if tp_size == 0 {
                return Err(FerrumError::config("Tensor parallel size must be positive"));
            }
        }

        if let Some(pp_size) = self.pipeline_parallel_size {
            if pp_size == 0 {
                return Err(FerrumError::config("Pipeline parallel size must be positive"));
            }
        }

        Ok(())
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// GPTQ quantization
    GPTQ { 
        bits: u8, 
        group_size: usize,
        desc_act: bool,
    },
    /// AWQ quantization
    AWQ { 
        bits: u8, 
        zero_point: bool,
        version: String,
    },
    /// FP8 quantization
    FP8 { 
        e4m3: bool,
        kv_cache: bool,
    },
    /// INT8 quantization
    INT8 { 
        symmetric: bool,
        per_channel: bool,
    },
    /// INT4 quantization
    INT4 {
        symmetric: bool,
        group_size: usize,
    },
    /// SmoothQuant
    SmoothQuant {
        alpha: f32,
        calibration_size: usize,
    },
}

impl QuantizationConfig {
    /// Get the number of bits used by this quantization method
    pub fn bits(&self) -> u8 {
        match self {
            QuantizationConfig::GPTQ { bits, .. } => *bits,
            QuantizationConfig::AWQ { bits, .. } => *bits,
            QuantizationConfig::FP8 { .. } => 8,
            QuantizationConfig::INT8 { .. } => 8,
            QuantizationConfig::INT4 { .. } => 4,
            QuantizationConfig::SmoothQuant { .. } => 8,
        }
    }

    /// Check if this quantization preserves accuracy well
    pub fn is_high_accuracy(&self) -> bool {
        match self {
            QuantizationConfig::FP8 { .. } => true,
            QuantizationConfig::INT8 { .. } => true,
            QuantizationConfig::SmoothQuant { .. } => true,
            _ => false,
        }
    }
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Number of tokens in the prompt
    pub prompt_tokens: usize,
    /// Number of tokens generated
    pub completion_tokens: usize,
    /// Total tokens processed
    pub total_tokens: usize,
}

impl TokenUsage {
    /// Create new token usage
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }

    /// Add completion tokens
    pub fn add_completion_tokens(&mut self, tokens: usize) {
        self.completion_tokens += tokens;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }
}

/// Model loading source specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    /// Local file path
    Local(String),
    /// Hugging Face Hub model
    HuggingFace {
        repo_id: String,
        revision: Option<String>,
        cache_dir: Option<String>,
    },
    /// URL download
    Url {
        url: String,
        headers: HashMap<String, String>,
    },
    /// S3-compatible storage
    S3 {
        bucket: String,
        key: String,
        region: Option<String>,
        endpoint: Option<String>,
    },
}
