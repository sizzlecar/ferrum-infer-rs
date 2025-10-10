//! CLI configuration management
//!
//! Handles loading and parsing of configuration files for the CLI tool.

use ferrum_types::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Server configuration
    pub server: ServerCliConfig,

    /// Model configuration
    pub models: ModelCliConfig,

    /// Benchmark configuration
    pub benchmark: BenchmarkConfig,

    /// Client configuration
    pub client: ClientConfig,

    /// Development configuration
    pub dev: DevConfig,
}

/// Server CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCliConfig {
    /// Default host
    pub host: String,

    /// Default port
    pub port: u16,

    /// Configuration file path
    pub config_path: String,

    /// Log level
    pub log_level: String,

    /// Enable hot reload
    pub hot_reload: bool,
}

/// Model CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCliConfig {
    /// Default model directory
    pub model_dir: String,

    /// Model cache directory
    pub cache_dir: String,

    /// Default model
    pub default_model: Option<String>,

    /// Model aliases
    pub aliases: HashMap<String, String>,

    /// Download settings
    pub download: DownloadConfig,
}

/// Download configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadConfig {
    /// HuggingFace cache directory
    pub hf_cache_dir: String,

    /// Download timeout in seconds
    pub timeout_seconds: u64,

    /// Max concurrent downloads
    pub max_concurrent: usize,

    /// Retry attempts
    pub retry_attempts: u32,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Default number of requests
    pub num_requests: usize,

    /// Default concurrency level
    pub concurrency: usize,

    /// Default prompt length
    pub prompt_length: usize,

    /// Default max tokens
    pub max_tokens: usize,

    /// Warmup requests
    pub warmup_requests: usize,

    /// Output directory for reports
    pub output_dir: String,
}

/// Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Default API base URL
    pub base_url: String,

    /// Default API key
    pub api_key: Option<String>,

    /// Request timeout
    pub timeout_seconds: u64,

    /// Retry configuration
    pub retry: RetryConfig,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,

    /// Initial delay in milliseconds
    pub initial_delay_ms: u64,

    /// Maximum delay in milliseconds
    pub max_delay_ms: u64,

    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

/// Development configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevConfig {
    /// Enable debug mode
    pub debug: bool,

    /// Profile memory usage
    pub profile_memory: bool,

    /// Enable GPU profiling
    pub profile_gpu: bool,

    /// Mock backends for testing
    pub mock_backends: bool,

    /// Test data directory
    pub test_data_dir: String,
}

impl CliConfig {
    /// Load configuration from file
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            // Create default config file
            let default_config = Self::default();
            let content = toml::to_string_pretty(&default_config).map_err(|e| {
                ferrum_types::FerrumError::configuration(format!(
                    "Failed to serialize default config: {}",
                    e
                ))
            })?;

            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await.map_err(|e| {
                    ferrum_types::FerrumError::io_str(format!("Failed to create config directory: {}", e))
                })?;
            }

            fs::write(path, content).await.map_err(|e| {
                ferrum_types::FerrumError::io_str(format!("Failed to write default config: {}", e))
            })?;

            return Ok(default_config);
        }

        let content = fs::read_to_string(path).await.map_err(|e| {
            ferrum_types::FerrumError::io_str(format!("Failed to read config file: {}", e))
        })?;

        toml::from_str(&content).map_err(|e| {
            ferrum_types::FerrumError::configuration(format!("Failed to parse config: {}", e))
        })
    }

    /// Save configuration to file
    pub async fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            ferrum_types::FerrumError::configuration(format!("Failed to serialize config: {}", e))
        })?;

        fs::write(path, content)
            .await
            .map_err(|e| ferrum_types::FerrumError::io_str(format!("Failed to write config file: {}", e)))
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate server config
        if self.server.port == 0 {
            return Err(ferrum_types::FerrumError::configuration(
                "Server port cannot be 0".to_string(),
            ));
        }

        // Validate model config
        if !Path::new(&self.models.model_dir).exists() {
            return Err(ferrum_types::FerrumError::configuration(format!(
                "Model directory does not exist: {}",
                self.models.model_dir
            )));
        }

        // Validate benchmark config
        if self.benchmark.num_requests == 0 {
            return Err(ferrum_types::FerrumError::configuration(
                "Number of requests cannot be 0".to_string(),
            ));
        }

        if self.benchmark.concurrency == 0 {
            return Err(ferrum_types::FerrumError::configuration(
                "Concurrency cannot be 0".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            server: ServerCliConfig::default(),
            models: ModelCliConfig::default(),
            benchmark: BenchmarkConfig::default(),
            client: ClientConfig::default(),
            dev: DevConfig::default(),
        }
    }
}

impl Default for ServerCliConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8000,
            config_path: "server.toml".to_string(),
            log_level: "info".to_string(),
            hot_reload: false,
        }
    }
}

impl Default for ModelCliConfig {
    fn default() -> Self {
        Self {
            model_dir: "./models".to_string(),
            cache_dir: "./cache".to_string(),
            default_model: None,
            aliases: HashMap::new(),
            download: DownloadConfig::default(),
        }
    }
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            hf_cache_dir: dirs::cache_dir()
                .map(|p| p.join("huggingface").to_string_lossy().to_string())
                .unwrap_or_else(|| "./hf_cache".to_string()),
            timeout_seconds: 300,
            max_concurrent: 4,
            retry_attempts: 3,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_requests: 100,
            concurrency: 10,
            prompt_length: 512,
            max_tokens: 256,
            warmup_requests: 10,
            output_dir: "./benchmark_results".to_string(),
        }
    }
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8000".to_string(),
            api_key: None,
            timeout_seconds: 30,
            retry: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for DevConfig {
    fn default() -> Self {
        Self {
            debug: false,
            profile_memory: false,
            profile_gpu: false,
            mock_backends: false,
            test_data_dir: "./test_data".to_string(),
        }
    }
}
