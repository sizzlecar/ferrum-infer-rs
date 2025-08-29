//! Model configuration management

use ferrum_core::{ModelConfig, ModelType, DataType, Device, ModelId, Result, Error};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;
use tracing::info;

/// Model configuration manager
pub struct ModelConfigManager {
    configs: HashMap<String, ModelConfig>,
}

/// Model configuration file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigFile {
    pub model_id: String,
    pub model_type: String,
    pub model_path: String,
    pub dtype: String,
    pub device: String,
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub tensor_parallel_size: Option<usize>,
    pub pipeline_parallel_size: Option<usize>,
    pub quantization: Option<QuantizationConfigFile>,
}

/// Quantization configuration file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfigFile {
    pub method: String,
    pub bits: Option<u8>,
    pub group_size: Option<usize>,
    pub zero_point: Option<bool>,
    pub symmetric: Option<bool>,
}

impl ModelConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }
    
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        info!("Loading model config from {:?}", path.as_ref());
        
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::configuration(format!("Failed to read config file: {}", e)))?;
        
        let config_file: ModelConfigFile = toml::from_str(&content)
            .map_err(|e| Error::configuration(format!("Failed to parse config file: {}", e)))?;
        
        let config = self.convert_config_file(config_file)?;
        self.configs.insert(config.model_id.0.clone(), config);
        
        Ok(())
    }
    
    /// Convert config file to ModelConfig
    fn convert_config_file(&self, file: ModelConfigFile) -> Result<ModelConfig> {
        let model_type = match file.model_type.as_str() {
            "llama" | "Llama" => ModelType::Llama,
            "mistral" | "Mistral" => ModelType::Mistral,
            "qwen" | "Qwen" => ModelType::Qwen,
            custom => ModelType::Custom(custom.to_string()),
        };
        
        let dtype = match file.dtype.as_str() {
            "fp32" | "f32" => DataType::FP32,
            "fp16" | "f16" => DataType::FP16,
            "bf16" => DataType::BF16,
            "int8" | "i8" => DataType::INT8,
            "fp8" => DataType::FP8,
            _ => return Err(Error::configuration(format!("Unknown dtype: {}", file.dtype))),
        };
        
        let device = self.parse_device(&file.device)?;
        
        let quantization = file.quantization
            .map(|q| self.convert_quantization_config(q))
            .transpose()?;
        
        Ok(ModelConfig {
            model_id: ModelId(file.model_id),
            model_path: file.model_path,
            model_type,
            dtype,
            device,
            max_batch_size: file.max_batch_size,
            max_sequence_length: file.max_sequence_length,
            tensor_parallel_size: file.tensor_parallel_size,
            pipeline_parallel_size: file.pipeline_parallel_size,
            quantization,
        })
    }
    
    /// Parse device string
    fn parse_device(&self, device_str: &str) -> Result<Device> {
        match device_str {
            "cpu" | "CPU" => Ok(Device::CPU),
            s if s.starts_with("cuda:") => {
                let id = s.trim_start_matches("cuda:")
                    .parse::<usize>()
                    .map_err(|_| Error::configuration(format!("Invalid CUDA device: {}", s)))?;
                Ok(Device::CUDA(id))
            }
            s if s.starts_with("rocm:") => {
                let id = s.trim_start_matches("rocm:")
                    .parse::<usize>()
                    .map_err(|_| Error::configuration(format!("Invalid ROCm device: {}", s)))?;
                Ok(Device::ROCm(id))
            }
            _ => Err(Error::configuration(format!("Unknown device: {}", device_str))),
        }
    }
    
    /// Convert quantization config
    fn convert_quantization_config(&self, config: QuantizationConfigFile) -> Result<ferrum_core::QuantizationConfig> {
        match config.method.as_str() {
            "gptq" | "GPTQ" => Ok(ferrum_core::QuantizationConfig::GPTQ {
                bits: config.bits.unwrap_or(4),
                group_size: config.group_size.unwrap_or(128),
            }),
            "awq" | "AWQ" => Ok(ferrum_core::QuantizationConfig::AWQ {
                bits: config.bits.unwrap_or(4),
                zero_point: config.zero_point.unwrap_or(true),
            }),
            "fp8" | "FP8" => Ok(ferrum_core::QuantizationConfig::FP8 {
                e4m3: true,
            }),
            "int8" | "INT8" => Ok(ferrum_core::QuantizationConfig::INT8 {
                symmetric: config.symmetric.unwrap_or(true),
            }),
            _ => Err(Error::configuration(format!("Unknown quantization method: {}", config.method))),
        }
    }
    
    /// Get configuration for a model
    pub fn get(&self, model_id: &str) -> Option<&ModelConfig> {
        self.configs.get(model_id)
    }
    
    /// Add configuration
    pub fn add(&mut self, config: ModelConfig) {
        self.configs.insert(config.model_id.0.clone(), config);
    }
    
    /// List all configurations
    pub fn list(&self) -> Vec<&ModelConfig> {
        self.configs.values().collect()
    }
    
    /// Create default configuration for common models
    pub fn create_default(model_id: &str) -> Result<ModelConfig> {
        let (model_type, default_size) = match model_id {
            s if s.contains("llama") => (ModelType::Llama, 4096),
            s if s.contains("mistral") => (ModelType::Mistral, 8192),
            s if s.contains("qwen") => (ModelType::Qwen, 8192),
            _ => (ModelType::Custom(model_id.to_string()), 4096),
        };
        
        Ok(ModelConfig {
            model_id: ModelId(model_id.to_string()),
            model_path: format!("models/{}", model_id),
            model_type,
            dtype: DataType::FP16,
            device: Device::cuda_if_available().unwrap_or(Device::CPU),
            max_batch_size: 256,
            max_sequence_length: default_size,
            tensor_parallel_size: None,
            pipeline_parallel_size: None,
            quantization: None,
        })
    }
}

impl Device {
    /// Get CUDA device if available, otherwise CPU
    pub fn cuda_if_available() -> Option<Device> {
        // Check if CUDA is available (simplified)
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            Some(Device::CUDA(0))
        } else {
            None
        }
    }
}

impl Default for ModelConfigManager {
    fn default() -> Self {
        Self::new()
    }
}
