//! Configuration types for Ferrum components

use crate::{
    parse_bool_env_value, parse_path_env_value, parse_usize_env_value, DataType, Device, ModelId,
    ModelInfo, RuntimeConfigSnapshot, SamplingParams, SamplingPresets,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, time::Duration};

/// Engine runtime knobs the CLI/autosizer resolves and injects via the
/// runtime-config snapshot. The continuous engine reads these from the typed
/// config instead of `std::env::vars()`, so the env bridge stays at the
/// composition root and tests can vary the knobs per `EngineConfig`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeKnobs {
    pub kv_capacity: Option<usize>,
    pub max_model_len: Option<usize>,
    pub chunked_prefill_size: Option<usize>,
    pub batch_decode_prof: bool,
    pub next_batch_prof: bool,
    pub rbd_prof: bool,
    pub scheduler_trace_jsonl: Option<PathBuf>,
    pub unified_post_prof: bool,
    pub prefix_cache_enabled: bool,

    // Engine-build composition knobs. Previously read directly from the
    // environment by `builder.rs` (FERRUM_MODEL_PATH / FERRUM_SPEC_DRAFT /
    // FERRUM_SPEC_N) and `registry.rs` (FERRUM_DTYPE / FERRUM_METAL_DTYPE /
    // FERRUM_TP). The CLI composition root now resolves them into this typed
    // field so the engine builder and component registry read the snapshot,
    // not `std::env`.
    pub model_path: Option<String>,
    pub spec_draft: Option<String>,
    pub spec_n: Option<usize>,
    pub dtype: Option<String>,
    pub metal_dtype: Option<String>,
    pub tp: Option<usize>,
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    pub model: EngineModelConfig,
    pub scheduler: SchedulerConfig,
    pub sampling: SamplingConfig,
    pub backend: BackendConfig,
    pub kv_cache: KvCacheConfig,
    pub memory: MemoryConfig,
    pub batching: BatchConfig,
    pub monitoring: MonitoringConfig,
    #[serde(default)]
    pub runtime: RuntimeKnobs,
}

impl EngineConfig {
    pub fn apply_runtime_config_snapshot(
        &mut self,
        snapshot: &RuntimeConfigSnapshot,
    ) -> std::result::Result<(), String> {
        self.scheduler.apply_runtime_config_snapshot(snapshot)?;
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_KV_MAX_BLOCKS") {
            self.kv_cache.max_blocks =
                parse_required_positive_usize("FERRUM_KV_MAX_BLOCKS", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_MAX_BATCHED_TOKENS") {
            self.batching.max_num_batched_tokens =
                parse_required_positive_usize("FERRUM_MAX_BATCHED_TOKENS", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_PAGED_MAX_SEQS") {
            self.scheduler.max_running_requests =
                parse_required_positive_usize("FERRUM_PAGED_MAX_SEQS", value)?;
        }
        // Engine runtime knobs (previously read by the engine from env). The
        // CLI/autosizer resolves these into the snapshot; the engine reads the
        // typed `runtime` field instead of `std::env::vars()`.
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_KV_CAPACITY") {
            self.runtime.kv_capacity =
                Some(parse_required_positive_usize("FERRUM_KV_CAPACITY", value)?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_MAX_MODEL_LEN") {
            self.runtime.max_model_len = Some(parse_required_positive_usize(
                "FERRUM_MAX_MODEL_LEN",
                value,
            )?);
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_CHUNKED_PREFILL") {
            self.runtime.chunked_prefill_size =
                parse_usize_env_value(value).ok().filter(|&v| v > 0);
        }
        self.runtime.batch_decode_prof |=
            runtime_config_value(snapshot, "FERRUM_BATCH_DECODE_PROF").is_some();
        self.runtime.next_batch_prof |=
            runtime_config_value(snapshot, "FERRUM_NEXT_BATCH_PROF").is_some();
        self.runtime.rbd_prof |= runtime_config_value(snapshot, "FERRUM_RBD_PROF").is_some();
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHEDULER_TRACE_JSONL") {
            self.runtime.scheduler_trace_jsonl = Some(parse_path_env_value(value)?);
        }
        self.runtime.unified_post_prof |=
            runtime_config_value(snapshot, "FERRUM_UNIFIED_POST_PROF").is_some();
        self.runtime.prefix_cache_enabled |=
            runtime_config_value(snapshot, "FERRUM_WHOLE_PROMPT_PREFIX_CACHE")
                .map(|v| v == "1")
                .unwrap_or(false);

        // Engine-build composition knobs (previously read by builder.rs /
        // registry.rs from env). Only overwrite when the key is present so a
        // later snapshot apply without the key keeps an earlier value.
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_MODEL_PATH") {
            self.runtime.model_path = Some(value.to_string());
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SPEC_DRAFT") {
            self.runtime.spec_draft = if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SPEC_N") {
            self.runtime.spec_n = value.parse::<usize>().ok();
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_DTYPE") {
            self.runtime.dtype = Some(value.to_string());
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_METAL_DTYPE") {
            self.runtime.metal_dtype = Some(value.to_string());
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_TP") {
            self.runtime.tp = value.parse::<usize>().ok();
        }

        // Publish the resolved snapshot process-wide. Model code (which is not
        // threaded an EngineConfig) reads `active_runtime_snapshot()` for the
        // remaining FERRUM_* toggles instead of `std::env`, keeping the env
        // bridge at this single composition-root call.
        crate::install_runtime_snapshot(snapshot.clone());
        Ok(())
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
    /// Use prompt-token metadata for initial continuous-batch admission estimates.
    #[serde(default = "default_prompt_token_estimate")]
    pub prompt_token_estimate: bool,
    /// Prefer new prefills over early decodes until this many requests are active.
    #[serde(default)]
    pub prefill_first_until_active: Option<usize>,
    /// Cap per-request prefill chunks at the scheduler token-budget layer.
    #[serde(default)]
    pub prefill_step_chunk: Option<usize>,
    /// Cap prefill admission chunks only while decode requests are already active.
    #[serde(default)]
    pub active_decode_prefill_chunk: Option<usize>,
    /// Emit diagnostic scheduler None/SOME decisions.
    #[serde(default)]
    pub scheduler_none_prof: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Priority,
            max_waiting_requests: 1000,
            max_running_requests: 32,
            enable_preemption: true,
            enable_load_balancing: false,
            fair_share_weights: HashMap::new(),
            enable_sla_enforcement: false,
            prompt_token_estimate: default_prompt_token_estimate(),
            prefill_first_until_active: None,
            prefill_step_chunk: None,
            active_decode_prefill_chunk: None,
            scheduler_none_prof: false,
        }
    }
}

fn default_prompt_token_estimate() -> bool {
    true
}

impl SchedulerConfig {
    pub fn apply_runtime_config_snapshot(
        &mut self,
        snapshot: &RuntimeConfigSnapshot,
    ) -> std::result::Result<(), String> {
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE") {
            self.prompt_token_estimate = parse_bool_env_value(value)
                .map_err(|reason| format!("FERRUM_SCHED_PROMPT_TOKEN_ESTIMATE: {reason}"))?;
        }
        if let Some(value) =
            runtime_config_value(snapshot, "FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE")
        {
            self.prefill_first_until_active =
                parse_optional_positive_usize("FERRUM_SCHED_PREFILL_FIRST_UNTIL_ACTIVE", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHED_PREFILL_STEP_CHUNK") {
            self.prefill_step_chunk =
                parse_optional_positive_usize("FERRUM_SCHED_PREFILL_STEP_CHUNK", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK") {
            self.active_decode_prefill_chunk =
                parse_optional_positive_usize("FERRUM_ACTIVE_DECODE_PREFILL_CHUNK", value)?;
        }
        if let Some(value) = runtime_config_value(snapshot, "FERRUM_SCHED_NONE_PROF") {
            self.scheduler_none_prof = parse_presence_bool(value)
                .map_err(|reason| format!("FERRUM_SCHED_NONE_PROF: {reason}"))?;
        }
        Ok(())
    }
}

fn runtime_config_value<'a>(snapshot: &'a RuntimeConfigSnapshot, key: &str) -> Option<&'a str> {
    snapshot
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.effective_value.as_str())
}

fn parse_optional_positive_usize(
    key: &str,
    value: &str,
) -> std::result::Result<Option<usize>, String> {
    let parsed = parse_usize_env_value(value).map_err(|reason| format!("{key}: {reason}"))?;
    Ok((parsed > 0).then_some(parsed))
}

fn parse_required_positive_usize(key: &str, value: &str) -> std::result::Result<usize, String> {
    let parsed = parse_usize_env_value(value).map_err(|reason| format!("{key}: {reason}"))?;
    if parsed == 0 {
        Err(format!("{key}: must be greater than zero"))
    } else {
        Ok(parsed)
    }
}

fn parse_presence_bool(value: &str) -> std::result::Result<bool, String> {
    if value.trim().is_empty() {
        Ok(true)
    } else {
        parse_bool_env_value(value)
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
    /// Iteration-level continuous batching with preemption
    ContinuousBatch,
}

/// KV Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Cache implementation type
    pub cache_type: KvCacheType,
    /// Element dtype (Dim 5 polymorphism point). FP16 is the
    /// validated production path; INT8 / FP8 require a backend impl
    /// of `BackendKvDtype<KvInt8>` / `BackendKvDtype<KvFp8>` and a
    /// model wired through `KvCacheQuant<B, K>`.
    #[serde(default)]
    pub dtype: KvCacheDtype,
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
        // 2048 blocks covers c=32 ShareGPT prompts (~32×500/16 = 1000
        // blocks). The previous 1024 floor crashed at c≥16 on real
        // workloads with "Block pool exhausted". Runtime overrides are
        // applied through EngineConfig::apply_runtime_config_snapshot.
        Self {
            cache_type: KvCacheType::Contiguous,
            dtype: KvCacheDtype::default(),
            block_size: 16,
            max_blocks: 2048,
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

/// KV Cache element dtype (Dim 5 polymorphism point).
///
/// Mirrors `ferrum_interfaces::kv_dtype::KvDtypeKind` markers but
/// lives here because `KvCacheConfig` is part of the user-facing
/// `EngineConfig` and needs `Serialize` / `Deserialize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KvCacheDtype {
    /// FP16 K/V — the validated production path on every backend.
    #[default]
    Fp16,
    /// BF16 K/V — same memory cost as FP16, slightly different precision.
    /// Marker only; no backend impl ships yet.
    Bf16,
    /// INT8 K/V with per-token per-kv-head FP16 scale (vLLM-style).
    /// Halves KV memory at small (<1%) accuracy hit. CUDA kernels
    /// land via `BackendKvDtype<KvInt8>` (PR #131); model wire-up
    /// (`KvCacheQuant<B, KvInt8>` through the model decode loop) is
    /// the only remaining step.
    Int8,
    /// FP8 (E4M3) K/V. Marker only; CUDA kernels pending.
    Fp8,
}

impl KvCacheDtype {
    /// Parse from a CLI / env-var string. Accepts fp16/f16/bf16/int8/fp8/f8e4m3.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fp16" | "f16" | "float16" => Some(Self::Fp16),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "int8" | "i8" => Some(Self::Int8),
            "fp8" | "f8" | "f8e4m3" | "e4m3" => Some(Self::Fp8),
            _ => None,
        }
    }

    /// Short label for display + telemetry.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
        }
    }
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
            device: Device::CPU,
            dtype: DataType::FP16,
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

/// Tokenizer algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerType {
    /// Byte Pair Encoding
    BPE,
    /// WordPiece tokenizer (BERT-style)
    WordPiece,
    /// SentencePiece tokenizer
    SentencePiece,
    /// Tiktoken tokenizer family
    Tiktoken,
    /// Any custom tokenizer implementation
    Custom,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SamplingConfig {
    pub default_params: SamplingParams,
    pub presets: SamplingPresets,
    pub enable_custom_processors: bool,
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
    /// vLLM-style per-iteration token budget. The scheduler emits a
    /// mixed prefill+decode batch summing to at most this many Q
    /// tokens (decode = 1 each, prefill chunk = its chunk size).
    /// Default 2048. Runtime snapshots can override this with
    /// `FERRUM_MAX_BATCHED_TOKENS`, usually from the GPU autosizer or a
    /// named workload preset rather than a user hand-written env bundle.
    #[serde(default = "BatchConfig::default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
}

impl BatchConfig {
    fn default_max_num_batched_tokens() -> usize {
        2048
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 8,
            enable_dynamic: true,
            enable_continuous: false,
            max_num_batched_tokens: Self::default_max_num_batched_tokens(),
        }
    }
}
