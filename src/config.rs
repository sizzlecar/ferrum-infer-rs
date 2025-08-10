//! Configuration management for the LLM inference engine
//!
//! This module handles all configuration settings, including server settings,
//! model configuration, cache settings, and performance tuning parameters.

use crate::error::{EngineError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Model configuration
    pub model: ModelConfig,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Performance tuning configuration
    pub performance: PerformanceConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Metrics configuration
    pub metrics: MetricsConfig,
}

/// Server configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server bind address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Enable CORS
    pub enable_cors: bool,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
}

/// Model configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name or identifier
    pub name: String,
    /// Path to model files (local path or HuggingFace model ID)
    pub model_path: String,
    /// Path to tokenizer files
    pub tokenizer_path: Option<String>,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Device to use for inference (cpu, cuda, mps)
    pub device: String,
    /// Data type for model weights (f16, f32)
    pub dtype: String,
    /// Whether to use flash attention
    pub use_flash_attention: bool,
    /// Model revision/version
    pub revision: Option<String>,
}

/// Cache configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable KV caching
    pub enabled: bool,
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Cache eviction policy (lru, lfu, fifo)
    pub eviction_policy: String,
    /// Maximum number of cached sequences
    pub max_sequences: usize,
    /// Cache key TTL in seconds
    pub ttl_seconds: Option<u64>,
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads for inference
    pub num_threads: Option<usize>,
    /// Batch size for inference
    pub batch_size: usize,
    /// Enable tensor parallelism
    pub enable_tensor_parallel: bool,
    /// Memory pool size in MB
    pub memory_pool_size_mb: usize,
    /// Enable memory mapping for model files
    pub use_mmap: bool,
    /// Prefill chunk size
    pub prefill_chunk_size: usize,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Log format (json, pretty)
    pub format: String,
    /// Enable request logging
    pub log_requests: bool,
    /// Enable performance logging
    pub log_performance: bool,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics endpoint path
    pub endpoint: String,
    /// Metrics port (if different from main server)
    pub port: Option<u16>,
    /// Enable detailed inference metrics
    pub detailed_inference_metrics: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            model: ModelConfig::default(),
            cache: CacheConfig::default(),
            performance: PerformanceConfig::default(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_concurrent_requests: 100,
            request_timeout_secs: 300,
            enable_cors: true,
            api_key: None,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "microsoft/DialoGPT-medium".to_string(),
            model_path: "microsoft/DialoGPT-medium".to_string(),
            tokenizer_path: None,
            max_sequence_length: 2048,
            device: "cpu".to_string(),
            dtype: "f32".to_string(),
            use_flash_attention: false,
            revision: None,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size_mb: 1024,
            eviction_policy: "lru".to_string(),
            max_sequences: 1000,
            ttl_seconds: Some(3600),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use system default
            batch_size: 1,
            enable_tensor_parallel: false,
            memory_pool_size_mb: 512,
            use_mmap: true,
            prefill_chunk_size: 512,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            log_requests: true,
            log_performance: true,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "/metrics".to_string(),
            port: None,
            detailed_inference_metrics: false,
        }
    }
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        // Server configuration
        if let Ok(host) = std::env::var("FERRUM_INFER_HOST") {
            config.server.host = host;
        }
        if let Ok(port) = std::env::var("FERRUM_INFER_PORT") {
            config.server.port = port
                .parse()
                .map_err(|_| EngineError::config("Invalid port number"))?;
        }
        if let Ok(max_requests) = std::env::var("FERRUM_INFER_MAX_CONCURRENT_REQUESTS") {
            config.server.max_concurrent_requests = max_requests
                .parse()
                .map_err(|_| EngineError::config("Invalid max concurrent requests"))?;
        }
        if let Ok(api_key) = std::env::var("FERRUM_INFER_API_KEY") {
            config.server.api_key = Some(api_key);
        }

        // Model configuration
        if let Ok(model_path) = std::env::var("FERRUM_INFER_MODEL_PATH") {
            config.model.model_path = model_path.clone();
            config.model.name = model_path;
        }
        if let Ok(device) = std::env::var("FERRUM_INFER_DEVICE") {
            config.model.device = device;
        }
        if let Ok(max_seq_len) = std::env::var("FERRUM_INFER_MAX_SEQUENCE_LENGTH") {
            config.model.max_sequence_length = max_seq_len
                .parse()
                .map_err(|_| EngineError::config("Invalid max sequence length"))?;
        }

        // Cache configuration
        if let Ok(cache_enabled) = std::env::var("FERRUM_INFER_CACHE_ENABLED") {
            config.cache.enabled = cache_enabled
                .parse()
                .map_err(|_| EngineError::config("Invalid cache enabled flag"))?;
        }
        if let Ok(cache_size) = std::env::var("FERRUM_INFER_CACHE_SIZE_MB") {
            config.cache.max_size_mb = cache_size
                .parse()
                .map_err(|_| EngineError::config("Invalid cache size"))?;
        }

        // Logging configuration
        if let Ok(log_level) = std::env::var("FERRUM_INFER_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| EngineError::config(format!("Failed to read config file: {}", e)))?;

        let config: Self = toml::from_str(&content)
            .map_err(|e| EngineError::config(format!("Failed to parse config file: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate server configuration
        if self.server.port == 0 {
            return Err(EngineError::config("Server port cannot be 0"));
        }
        if self.server.max_concurrent_requests == 0 {
            return Err(EngineError::config(
                "Max concurrent requests must be greater than 0",
            ));
        }

        // Validate model configuration
        if self.model.model_path.is_empty() {
            return Err(EngineError::config("Model path cannot be empty"));
        }
        if self.model.max_sequence_length == 0 {
            return Err(EngineError::config(
                "Max sequence length must be greater than 0",
            ));
        }
        if !["cpu", "cuda", "mps"].contains(&self.model.device.as_str()) {
            return Err(EngineError::config("Device must be one of: cpu, cuda, mps"));
        }
        if !["f16", "f32"].contains(&self.model.dtype.as_str()) {
            return Err(EngineError::config("Data type must be one of: f16, f32"));
        }

        // Validate cache configuration
        if !["lru", "lfu", "fifo"].contains(&self.cache.eviction_policy.as_str()) {
            return Err(EngineError::config(
                "Cache eviction policy must be one of: lru, lfu, fifo",
            ));
        }

        // Validate performance configuration
        if self.performance.batch_size == 0 {
            return Err(EngineError::config("Batch size must be greater than 0"));
        }

        // Validate logging configuration
        if !["trace", "debug", "info", "warn", "error"].contains(&self.logging.level.as_str()) {
            return Err(EngineError::config(
                "Log level must be one of: trace, debug, info, warn, error",
            ));
        }

        Ok(())
    }

    /// Get the full server address
    pub fn server_address(&self) -> String {
        format!("{}:{}", self.server.host, self.server.port)
    }

    /// Check if GPU is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        matches!(self.model.device.as_str(), "cuda" | "mps")
    }

    /// Get the model directory path
    pub fn model_dir(&self) -> PathBuf {
        PathBuf::from(&self.model.model_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.model.device, "cpu");
        assert!(config.cache.enabled);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        config.server.port = 0;
        assert!(config.validate().is_err());

        config.server.port = 8080;
        config.model.device = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_server_address() {
        let config = Config::default();
        assert_eq!(config.server_address(), "0.0.0.0:8080");
    }

    #[test]
    fn test_gpu_detection() {
        let mut config = Config::default();
        assert!(!config.is_gpu_enabled());

        config.model.device = "cuda".to_string();
        assert!(config.is_gpu_enabled());
    }
}
