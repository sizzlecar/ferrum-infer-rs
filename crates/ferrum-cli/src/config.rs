//! CLI configuration management
//!
//! Handles loading and parsing of configuration files for the CLI tool.

use ferrum_types::{Result, RuntimeConfigEntry, RuntimeConfigSource};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

    /// Runtime overrides loaded from the CLI config file.
    #[serde(default)]
    pub runtime: RuntimeCliConfig,
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

/// Runtime knobs that can be sourced from the CLI config file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimeCliConfig {
    /// Named startup/runtime preset. Presets provide product-owned default
    /// bundles and can still be overridden by explicit runtime keys below,
    /// environment variables, or CLI flags.
    #[serde(default)]
    pub preset: Option<String>,

    /// KV cache dtype override, equivalent to `--kv-dtype` or
    /// `FERRUM_KV_DTYPE`.
    #[serde(default)]
    pub kv_dtype: Option<String>,

    /// KV block budget, equivalent to `FERRUM_KV_MAX_BLOCKS`.
    #[serde(default)]
    pub kv_max_blocks: Option<usize>,

    /// Per-sequence KV token capacity, equivalent to `FERRUM_KV_CAPACITY`.
    #[serde(default)]
    pub kv_capacity: Option<usize>,

    /// Maximum paged-KV sequence count, equivalent to
    /// `FERRUM_PAGED_MAX_SEQS`.
    #[serde(default)]
    pub paged_max_seqs: Option<usize>,

    /// Scheduler/model max batched-token budget, equivalent to
    /// `FERRUM_MAX_BATCHED_TOKENS`.
    #[serde(default)]
    pub max_batched_tokens: Option<usize>,

    /// Prefer prefilling until this many requests are active, equivalent to
    /// `FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE`.
    #[serde(default)]
    pub scheduler_prefill_first_until_active: Option<usize>,

    /// Prefix cache opt-in, equivalent to `FERRUM_PREFIX_CACHE`.
    #[serde(default)]
    pub prefix_cache: Option<bool>,

    /// Layer-split decode pipeline mode, equivalent to
    /// `FERRUM_LAYER_SPLIT_PIPELINE_MODE`.
    #[serde(default)]
    pub layer_split_pipeline_mode: Option<String>,

    /// MoE CUDA graph policy override, equivalent to `FERRUM_MOE_GRAPH`.
    #[serde(default)]
    pub moe_graph: Option<bool>,

    /// Legacy Llama/Gemma batched decode CUDA graph policy override,
    /// equivalent to `FERRUM_BATCHED_GRAPH`.
    #[serde(default)]
    pub batched_graph: Option<bool>,

    /// Emit engine batch iteration profile logs, equivalent to
    /// `FERRUM_BATCH_DECODE_PROF`.
    #[serde(default)]
    pub batch_decode_prof: Option<bool>,

    /// Emit executor batch prefill profile logs, equivalent to
    /// `FERRUM_BATCH_PREFILL_PROF`.
    #[serde(default)]
    pub batch_prefill_prof: Option<bool>,

    /// Emit engine next-batch scheduler profile logs, equivalent to
    /// `FERRUM_NEXT_BATCH_PROF`.
    #[serde(default)]
    pub next_batch_prof: Option<bool>,

    /// Emit route/batched-decode profile logs, equivalent to
    /// `FERRUM_RBD_PROF`.
    #[serde(default)]
    pub rbd_prof: Option<bool>,

    /// Emit unified decode postprocess profile logs, equivalent to
    /// `FERRUM_UNIFIED_POST_PROF`.
    #[serde(default)]
    pub unified_post_prof: Option<bool>,

    /// Emit model decode operator profile logs, equivalent to
    /// `FERRUM_DECODE_OP_PROFILE`.
    #[serde(default)]
    pub decode_op_profile: Option<bool>,

    /// Emit model prefill operator profile logs, equivalent to
    /// `FERRUM_PREFILL_OP_PROFILE`.
    #[serde(default)]
    pub prefill_op_profile: Option<bool>,

    /// vLLM paged attention policy, equivalent to
    /// `FERRUM_USE_VLLM_PAGED_ATTN`.
    #[serde(default)]
    pub use_vllm_paged_attn: Option<bool>,

    /// Short-context vLLM paged-attention v1 policy, equivalent to
    /// `FERRUM_VLLM_PAGED_ATTN_V1_SHORT`.
    #[serde(default)]
    pub vllm_paged_attn_v1_short: Option<bool>,

    /// vLLM-Marlin MoE dispatch policy, equivalent to `FERRUM_VLLM_MOE`.
    #[serde(default)]
    pub vllm_moe: Option<bool>,

    /// vLLM-MoE pair-id route layout policy, equivalent to
    /// `FERRUM_VLLM_MOE_PAIR_IDS`.
    #[serde(default)]
    pub vllm_moe_pair_ids: Option<bool>,

    /// GPU greedy argmax readback policy, equivalent to
    /// `FERRUM_GREEDY_ARGMAX`.
    #[serde(default)]
    pub greedy_argmax: Option<bool>,

    /// FA-compatible varlen K/V layout policy, equivalent to
    /// `FERRUM_FA_LAYOUT_VARLEN`.
    #[serde(default)]
    pub fa_layout_varlen: Option<bool>,

    /// Source-linked FA2 policy, equivalent to `FERRUM_FA2_SOURCE`.
    #[serde(default)]
    pub fa2_source: Option<bool>,

    /// Runtime-loaded FA2 direct FFI policy, equivalent to
    /// `FERRUM_FA2_DIRECT_FFI`.
    #[serde(default)]
    pub fa2_direct_ffi: Option<bool>,

    /// Runtime-loaded FA2 direct FFI shim path, equivalent to
    /// `FERRUM_FA2_DIRECT_FFI_SHIM`.
    #[serde(default)]
    pub fa2_direct_ffi_shim: Option<String>,

    /// Requested max model length, equivalent to `FERRUM_MAX_MODEL_LEN`.
    #[serde(default)]
    pub max_model_len: Option<usize>,

    /// Minimum MoE batch size for the batched expert path, equivalent to
    /// `FERRUM_MOE_BATCH_THRESHOLD`.
    #[serde(default)]
    pub moe_batch_threshold: Option<usize>,
}

impl RuntimeCliConfig {
    pub fn runtime_config_entries(&self) -> Vec<RuntimeConfigEntry> {
        let mut entries = Vec::new();
        push_string_entry(&mut entries, "FERRUM_KV_DTYPE", self.kv_dtype.as_deref());
        push_usize_entry(&mut entries, "FERRUM_KV_MAX_BLOCKS", self.kv_max_blocks);
        push_usize_entry(&mut entries, "FERRUM_KV_CAPACITY", self.kv_capacity);
        push_usize_entry(&mut entries, "FERRUM_PAGED_MAX_SEQS", self.paged_max_seqs);
        push_usize_entry(
            &mut entries,
            "FERRUM_MAX_BATCHED_TOKENS",
            self.max_batched_tokens,
        );
        push_usize_entry(
            &mut entries,
            "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE",
            self.scheduler_prefill_first_until_active,
        );
        push_bool_entry(&mut entries, "FERRUM_PREFIX_CACHE", self.prefix_cache);
        push_string_entry(
            &mut entries,
            "FERRUM_LAYER_SPLIT_PIPELINE_MODE",
            self.layer_split_pipeline_mode.as_deref(),
        );
        push_bool_entry(&mut entries, "FERRUM_MOE_GRAPH", self.moe_graph);
        push_bool_entry(&mut entries, "FERRUM_BATCHED_GRAPH", self.batched_graph);
        push_true_entry(
            &mut entries,
            "FERRUM_BATCH_DECODE_PROF",
            self.batch_decode_prof,
        );
        push_true_entry(
            &mut entries,
            "FERRUM_BATCH_PREFILL_PROF",
            self.batch_prefill_prof,
        );
        push_true_entry(&mut entries, "FERRUM_NEXT_BATCH_PROF", self.next_batch_prof);
        push_true_entry(&mut entries, "FERRUM_RBD_PROF", self.rbd_prof);
        push_true_entry(
            &mut entries,
            "FERRUM_UNIFIED_POST_PROF",
            self.unified_post_prof,
        );
        push_true_entry(
            &mut entries,
            "FERRUM_DECODE_OP_PROFILE",
            self.decode_op_profile,
        );
        push_true_entry(
            &mut entries,
            "FERRUM_PREFILL_OP_PROFILE",
            self.prefill_op_profile,
        );
        push_bool_entry(
            &mut entries,
            "FERRUM_USE_VLLM_PAGED_ATTN",
            self.use_vllm_paged_attn,
        );
        push_bool_entry(
            &mut entries,
            "FERRUM_VLLM_PAGED_ATTN_V1_SHORT",
            self.vllm_paged_attn_v1_short,
        );
        push_bool_entry(&mut entries, "FERRUM_VLLM_MOE", self.vllm_moe);
        push_bool_entry(
            &mut entries,
            "FERRUM_VLLM_MOE_PAIR_IDS",
            self.vllm_moe_pair_ids,
        );
        push_bool_entry(&mut entries, "FERRUM_GREEDY_ARGMAX", self.greedy_argmax);
        push_bool_entry(
            &mut entries,
            "FERRUM_FA_LAYOUT_VARLEN",
            self.fa_layout_varlen,
        );
        push_bool_entry(&mut entries, "FERRUM_FA2_SOURCE", self.fa2_source);
        push_bool_entry(&mut entries, "FERRUM_FA2_DIRECT_FFI", self.fa2_direct_ffi);
        push_string_entry(
            &mut entries,
            "FERRUM_FA2_DIRECT_FFI_SHIM",
            self.fa2_direct_ffi_shim.as_deref(),
        );
        push_usize_entry(&mut entries, "FERRUM_MAX_MODEL_LEN", self.max_model_len);
        push_usize_entry(
            &mut entries,
            "FERRUM_MOE_BATCH_THRESHOLD",
            self.moe_batch_threshold,
        );
        entries
    }
}

fn push_string_entry(entries: &mut Vec<RuntimeConfigEntry>, key: &str, value: Option<&str>) {
    if let Some(value) = value.filter(|value| !value.trim().is_empty()) {
        entries.push(RuntimeConfigEntry::new(
            key,
            value.to_string(),
            RuntimeConfigSource::ConfigFile,
        ));
    }
}

fn push_usize_entry(entries: &mut Vec<RuntimeConfigEntry>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        entries.push(RuntimeConfigEntry::new(
            key,
            value.to_string(),
            RuntimeConfigSource::ConfigFile,
        ));
    }
}

fn push_bool_entry(entries: &mut Vec<RuntimeConfigEntry>, key: &str, value: Option<bool>) {
    if let Some(value) = value {
        entries.push(RuntimeConfigEntry::new(
            key,
            if value { "1" } else { "0" },
            RuntimeConfigSource::ConfigFile,
        ));
    }
}

fn push_true_entry(entries: &mut Vec<RuntimeConfigEntry>, key: &str, value: Option<bool>) {
    if value == Some(true) {
        entries.push(RuntimeConfigEntry::new(
            key,
            "1".to_string(),
            RuntimeConfigSource::ConfigFile,
        ));
    }
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
                    ferrum_types::FerrumError::io_str(format!(
                        "Failed to create config directory: {}",
                        e
                    ))
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

        fs::write(path, content).await.map_err(|e| {
            ferrum_types::FerrumError::io_str(format!("Failed to write config file: {}", e))
        })
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
            hf_cache_dir: std::env::var("HF_HOME")
                .ok()
                .or_else(|| {
                    dirs::home_dir()
                        .map(|h| h.join(".cache/huggingface").to_string_lossy().to_string())
                })
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::RuntimeConfigEffect;

    #[test]
    fn runtime_cli_config_emits_config_file_source_entries() {
        let runtime = RuntimeCliConfig {
            preset: Some("m3_qwen3_30b_a3b_int4".to_string()),
            kv_dtype: Some("int8".to_string()),
            kv_max_blocks: Some(4096),
            kv_capacity: Some(2048),
            paged_max_seqs: Some(64),
            max_batched_tokens: Some(2048),
            scheduler_prefill_first_until_active: Some(16),
            prefix_cache: Some(false),
            layer_split_pipeline_mode: Some("batch".to_string()),
            moe_graph: Some(true),
            batched_graph: Some(true),
            batch_decode_prof: Some(true),
            batch_prefill_prof: Some(true),
            next_batch_prof: Some(true),
            rbd_prof: Some(true),
            unified_post_prof: Some(true),
            decode_op_profile: Some(true),
            prefill_op_profile: Some(true),
            use_vllm_paged_attn: Some(true),
            vllm_paged_attn_v1_short: Some(false),
            vllm_moe: Some(true),
            vllm_moe_pair_ids: Some(true),
            greedy_argmax: Some(true),
            fa_layout_varlen: Some(true),
            fa2_source: Some(true),
            fa2_direct_ffi: Some(false),
            fa2_direct_ffi_shim: Some("/tmp/libferrum_fa2_shim.so".to_string()),
            max_model_len: Some(4096),
            moe_batch_threshold: Some(4),
            ..Default::default()
        };
        let entries = runtime.runtime_config_entries();
        assert_eq!(entries.len(), 28);
        let entry = |key: &str| {
            entries
                .iter()
                .find(|entry| entry.key == key)
                .unwrap_or_else(|| panic!("missing {key}"))
        };
        assert_eq!(entry("FERRUM_KV_DTYPE").effective_value, "int8");
        assert_eq!(
            entry("FERRUM_KV_DTYPE").source,
            RuntimeConfigSource::ConfigFile
        );
        assert!(entry("FERRUM_KV_DTYPE")
            .affects
            .contains(&RuntimeConfigEffect::Correctness));
        assert_eq!(entry("FERRUM_KV_MAX_BLOCKS").effective_value, "4096");
        assert_eq!(entry("FERRUM_KV_CAPACITY").effective_value, "2048");
        assert_eq!(entry("FERRUM_PAGED_MAX_SEQS").effective_value, "64");
        assert_eq!(entry("FERRUM_MAX_BATCHED_TOKENS").effective_value, "2048");
        assert_eq!(
            entry("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE").effective_value,
            "16"
        );
        assert_eq!(
            entry("FERRUM_LAYER_SPLIT_PIPELINE_MODE").effective_value,
            "batch"
        );
        assert_eq!(entry("FERRUM_PREFIX_CACHE").effective_value, "0");
        assert_eq!(entry("FERRUM_MOE_GRAPH").effective_value, "1");
        assert_eq!(entry("FERRUM_BATCHED_GRAPH").effective_value, "1");
        assert_eq!(entry("FERRUM_BATCH_DECODE_PROF").effective_value, "1");
        assert!(entry("FERRUM_BATCH_DECODE_PROF")
            .affects
            .contains(&RuntimeConfigEffect::Diagnostics));
        assert_eq!(entry("FERRUM_BATCH_PREFILL_PROF").effective_value, "1");
        assert_eq!(entry("FERRUM_NEXT_BATCH_PROF").effective_value, "1");
        assert_eq!(entry("FERRUM_RBD_PROF").effective_value, "1");
        assert_eq!(entry("FERRUM_UNIFIED_POST_PROF").effective_value, "1");
        assert_eq!(entry("FERRUM_DECODE_OP_PROFILE").effective_value, "1");
        assert_eq!(entry("FERRUM_PREFILL_OP_PROFILE").effective_value, "1");
        assert_eq!(entry("FERRUM_USE_VLLM_PAGED_ATTN").effective_value, "1");
        assert_eq!(
            entry("FERRUM_VLLM_PAGED_ATTN_V1_SHORT").effective_value,
            "0"
        );
        assert_eq!(entry("FERRUM_VLLM_MOE").effective_value, "1");
        assert_eq!(entry("FERRUM_VLLM_MOE_PAIR_IDS").effective_value, "1");
        assert_eq!(entry("FERRUM_GREEDY_ARGMAX").effective_value, "1");
        assert_eq!(entry("FERRUM_FA_LAYOUT_VARLEN").effective_value, "1");
        assert_eq!(entry("FERRUM_FA2_SOURCE").effective_value, "1");
        assert_eq!(entry("FERRUM_FA2_DIRECT_FFI").effective_value, "0");
        assert_eq!(
            entry("FERRUM_FA2_DIRECT_FFI_SHIM").effective_value,
            "/tmp/libferrum_fa2_shim.so"
        );
        assert_eq!(entry("FERRUM_MAX_MODEL_LEN").effective_value, "4096");
        assert_eq!(entry("FERRUM_MOE_BATCH_THRESHOLD").effective_value, "4");
    }

    #[test]
    fn runtime_cli_config_diagnostic_presence_flags_are_opt_in() {
        let entries = RuntimeCliConfig {
            batch_decode_prof: Some(false),
            batch_prefill_prof: Some(false),
            next_batch_prof: Some(false),
            rbd_prof: Some(false),
            unified_post_prof: Some(false),
            decode_op_profile: Some(false),
            prefill_op_profile: Some(false),
            ..Default::default()
        }
        .runtime_config_entries();

        assert!(
            entries.is_empty(),
            "false diagnostic presence flags must not materialize as FERRUM_*_PROF=0"
        );
    }

    #[test]
    fn runtime_cli_config_defaults_when_missing_from_toml() {
        let config: CliConfig = toml::from_str(
            r#"
            [server]
            host = "127.0.0.1"
            port = 8000
            config_path = "server.toml"
            log_level = "info"
            hot_reload = false

            [models]
            model_dir = "./models"
            cache_dir = "./cache"

            [models.aliases]

            [models.download]
            hf_cache_dir = "./hf_cache"
            timeout_seconds = 300
            max_concurrent = 4
            retry_attempts = 3

            [benchmark]
            num_requests = 100
            concurrency = 10
            prompt_length = 512
            max_tokens = 256
            warmup_requests = 10
            output_dir = "./benchmark_results"

            [client]
            base_url = "http://127.0.0.1:8000"
            timeout_seconds = 30

            [client.retry]
            max_attempts = 3
            initial_delay_ms = 100
            max_delay_ms = 5000
            backoff_multiplier = 2.0

            [dev]
            debug = false
            profile_memory = false
            profile_gpu = false
            mock_backends = false
            test_data_dir = "./test_data"
            "#,
        )
        .unwrap();
        assert!(config.runtime.preset.is_none());
        assert!(config.runtime.kv_dtype.is_none());
        assert!(config.runtime.runtime_config_entries().is_empty());
    }
}
