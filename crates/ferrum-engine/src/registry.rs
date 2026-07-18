//! Component registry for dynamic component creation
//!
//! This module provides a registry pattern implementation that allows
//! dynamic registration and lookup of component factories. The registry
//! replaces hardcoded factory implementations with a flexible, extensible
//! system that supports:
//!
//! - Dynamic component registration and discovery
//! - Multiple implementations for each component type
//! - Configuration-driven component selection
//! - Easy testing with mock components

use async_trait::async_trait;
use ferrum_interfaces::{
    KvCacheManager, ModelExecutor, Sampler, SchedulerInterface as Scheduler, Tokenizer,
};
use ferrum_types::{DataType, Device, EngineConfig, FerrumError, Result, RuntimeKnobs};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tracing::{debug, info};

// ============================================================================
// Core Types
// ============================================================================

/// Factory trait for creating components
#[async_trait]
pub trait ComponentFactory<T>: Send + Sync {
    /// Create a component instance
    async fn create(&self, config: &ComponentConfig) -> Result<T>;

    /// Get component metadata
    fn metadata(&self) -> ComponentMetadata;
}

/// Component configuration passed to factories
#[derive(Debug, Clone)]
pub struct ComponentConfig {
    /// Full engine configuration
    pub engine_config: EngineConfig,
    /// Target device for the component
    pub device: Device,
    /// Additional component-specific options
    pub component_options: HashMap<String, serde_json::Value>,
}

impl ComponentConfig {
    /// Create from engine config
    pub fn from_engine_config(config: &EngineConfig) -> Self {
        Self {
            engine_config: config.clone(),
            device: config.backend.device.clone(),
            component_options: config.backend.backend_options.clone(),
        }
    }

    /// Get an option value
    pub fn get_option<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.component_options
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get string option
    pub fn get_string_option(&self, key: &str) -> Option<String> {
        self.component_options
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

/// Component metadata
#[derive(Debug, Clone)]
pub struct ComponentMetadata {
    /// Component name
    pub name: String,
    /// Version string
    pub version: String,
    /// Human-readable description
    pub description: String,
    /// Supported devices
    pub supported_devices: Vec<Device>,
    /// Capability flags
    pub capabilities: Vec<String>,
}

impl Default for ComponentMetadata {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            version: "0.0.0".to_string(),
            description: String::new(),
            supported_devices: vec![Device::CPU],
            capabilities: vec![],
        }
    }
}

fn cpu_cuda_and_optional_metal_devices() -> Vec<Device> {
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        vec![Device::CPU, Device::CUDA(0), Device::Metal]
    }
    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    {
        vec![Device::CPU, Device::CUDA(0)]
    }
}

fn parse_executor_dtype(s: &str) -> Option<candle_core::DType> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fp16" | "f16" | "float16" => Some(candle_core::DType::F16),
        "fp32" | "f32" | "float32" => Some(candle_core::DType::F32),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RegistryRuntimeEnv {
    model_path: Option<String>,
    metal_dtype: Option<String>,
    dtype: Option<String>,
    tp: usize,
}

impl RegistryRuntimeEnv {
    /// Build from the typed runtime snapshot resolved at the composition root.
    /// Replaces the former `std::env`-reading `from_env`/`OnceLock` pair; the
    /// CLI lands FERRUM_MODEL_PATH / FERRUM_DTYPE / FERRUM_METAL_DTYPE /
    /// FERRUM_TP into `EngineConfig.runtime`, and the registry reads that.
    fn from_runtime_knobs(knobs: &RuntimeKnobs) -> Self {
        Self {
            model_path: knobs.model_path.clone(),
            metal_dtype: knobs.metal_dtype.clone(),
            dtype: knobs.dtype.clone(),
            tp: knobs.tp.unwrap_or(0),
        }
    }

    #[cfg(test)]
    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let mut model_path = None;
        let mut metal_dtype = None;
        let mut dtype = None;
        let mut tp = None;

        for (key, value) in vars {
            let value = value.into();
            match key.as_ref() {
                "FERRUM_MODEL_PATH" => model_path = Some(value),
                "FERRUM_METAL_DTYPE" => metal_dtype = Some(value),
                "FERRUM_DTYPE" => dtype = Some(value),
                "FERRUM_TP" => tp = value.parse::<usize>().ok(),
                _ => {}
            }
        }

        Self {
            model_path,
            metal_dtype,
            dtype,
            tp: tp.unwrap_or(0),
        }
    }

    #[cfg(test)]
    fn model_path(&self) -> Option<String> {
        self.model_path.clone()
    }

    fn dtype_for_device(&self, device: &Device) -> candle_core::DType {
        match device {
            Device::CPU => candle_core::DType::F32,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => self
                .metal_dtype
                .as_deref()
                .and_then(parse_executor_dtype)
                .or_else(|| self.dtype.as_deref().and_then(parse_executor_dtype))
                .unwrap_or(candle_core::DType::F32),
            Device::CUDA(_) | Device::ROCm(_) => self
                .dtype
                .as_deref()
                .and_then(parse_executor_dtype)
                .unwrap_or(candle_core::DType::F16),
        }
    }

    #[allow(dead_code)]
    fn metal_dtype_hint(&self) -> String {
        self.metal_dtype
            .clone()
            .unwrap_or_else(|| "f32".to_string())
    }
}

// ============================================================================
// Component Registry
// ============================================================================

/// Global component registry for managing component factories
pub struct ComponentRegistry {
    tokenizer_factories:
        RwLock<HashMap<String, Arc<dyn ComponentFactory<Arc<dyn Tokenizer + Send + Sync>>>>>,
    sampler_factories:
        RwLock<HashMap<String, Arc<dyn ComponentFactory<Arc<dyn Sampler + Send + Sync>>>>>,
    scheduler_factories:
        RwLock<HashMap<String, Arc<dyn ComponentFactory<Arc<dyn Scheduler + Send + Sync>>>>>,
    kv_cache_factories:
        RwLock<HashMap<String, Arc<dyn ComponentFactory<Arc<dyn KvCacheManager + Send + Sync>>>>>,
    executor_factories:
        RwLock<HashMap<String, Arc<dyn ComponentFactory<Arc<dyn ModelExecutor + Send + Sync>>>>>,
}

impl ComponentRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tokenizer_factories: RwLock::new(HashMap::new()),
            sampler_factories: RwLock::new(HashMap::new()),
            scheduler_factories: RwLock::new(HashMap::new()),
            kv_cache_factories: RwLock::new(HashMap::new()),
            executor_factories: RwLock::new(HashMap::new()),
        }
    }

    /// Create a registry with all default factories pre-registered
    pub fn with_defaults() -> Self {
        let registry = Self::new();
        registry.register_defaults();
        registry
    }

    /// Register all default component factories
    pub fn register_defaults(&self) {
        info!("Registering default component factories");

        // Tokenizer factories
        self.register_tokenizer_factory("huggingface", Arc::new(HuggingFaceTokenizerFactory));
        self.register_tokenizer_factory("stub", Arc::new(StubTokenizerFactory));

        // Sampler factories
        self.register_sampler_factory("multinomial", Arc::new(MultinomialSamplerFactory));
        self.register_sampler_factory("greedy", Arc::new(GreedySamplerFactory));

        // Scheduler factories
        self.register_scheduler_factory("fifo", Arc::new(FifoSchedulerFactory));
        self.register_scheduler_factory("priority", Arc::new(PrioritySchedulerFactory));
        self.register_scheduler_factory("continuous", Arc::new(ContinuousBatchSchedulerFactory));

        // KV cache factories
        self.register_kv_cache_factory("default", Arc::new(DefaultKvCacheFactory));
        self.register_kv_cache_factory("paged", Arc::new(PagedKvCacheFactory));

        // Executor factories
        self.register_executor_factory("stub", Arc::new(StubExecutorFactory));
        self.register_executor_factory("llm", Arc::new(LlmExecutorFactory));

        debug!(
            "Registered factories - tokenizers: {:?}, samplers: {:?}, schedulers: {:?}, kv_caches: {:?}, executors: {:?}",
            self.list_tokenizers(),
            self.list_samplers(),
            self.list_schedulers(),
            self.list_kv_caches(),
            self.list_executors()
        );
    }

    // ========================================================================
    // Registration methods
    // ========================================================================

    /// Register a tokenizer factory
    pub fn register_tokenizer_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn ComponentFactory<Arc<dyn Tokenizer + Send + Sync>>>,
    ) {
        let name = name.into();
        debug!("Registering tokenizer factory: {}", name);
        self.tokenizer_factories.write().insert(name, factory);
    }

    /// Register a sampler factory
    pub fn register_sampler_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn ComponentFactory<Arc<dyn Sampler + Send + Sync>>>,
    ) {
        let name = name.into();
        debug!("Registering sampler factory: {}", name);
        self.sampler_factories.write().insert(name, factory);
    }

    /// Register a scheduler factory
    pub fn register_scheduler_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn ComponentFactory<Arc<dyn Scheduler + Send + Sync>>>,
    ) {
        let name = name.into();
        debug!("Registering scheduler factory: {}", name);
        self.scheduler_factories.write().insert(name, factory);
    }

    /// Register a KV cache factory
    pub fn register_kv_cache_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn ComponentFactory<Arc<dyn KvCacheManager + Send + Sync>>>,
    ) {
        let name = name.into();
        debug!("Registering KV cache factory: {}", name);
        self.kv_cache_factories.write().insert(name, factory);
    }

    /// Register a model executor factory
    pub fn register_executor_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn ComponentFactory<Arc<dyn ModelExecutor + Send + Sync>>>,
    ) {
        let name = name.into();
        debug!("Registering executor factory: {}", name);
        self.executor_factories.write().insert(name, factory);
    }

    // ========================================================================
    // Lookup methods
    // ========================================================================

    /// Get a tokenizer factory by name
    pub fn get_tokenizer_factory(
        &self,
        name: &str,
    ) -> Option<Arc<dyn ComponentFactory<Arc<dyn Tokenizer + Send + Sync>>>> {
        self.tokenizer_factories.read().get(name).cloned()
    }

    /// Get a sampler factory by name
    pub fn get_sampler_factory(
        &self,
        name: &str,
    ) -> Option<Arc<dyn ComponentFactory<Arc<dyn Sampler + Send + Sync>>>> {
        self.sampler_factories.read().get(name).cloned()
    }

    /// Get a scheduler factory by name
    pub fn get_scheduler_factory(
        &self,
        name: &str,
    ) -> Option<Arc<dyn ComponentFactory<Arc<dyn Scheduler + Send + Sync>>>> {
        self.scheduler_factories.read().get(name).cloned()
    }

    /// Get a KV cache factory by name
    pub fn get_kv_cache_factory(
        &self,
        name: &str,
    ) -> Option<Arc<dyn ComponentFactory<Arc<dyn KvCacheManager + Send + Sync>>>> {
        self.kv_cache_factories.read().get(name).cloned()
    }

    /// Get a model executor factory by name
    pub fn get_executor_factory(
        &self,
        name: &str,
    ) -> Option<Arc<dyn ComponentFactory<Arc<dyn ModelExecutor + Send + Sync>>>> {
        self.executor_factories.read().get(name).cloned()
    }

    // ========================================================================
    // List methods
    // ========================================================================

    /// List all registered tokenizer names
    pub fn list_tokenizers(&self) -> Vec<String> {
        self.tokenizer_factories.read().keys().cloned().collect()
    }

    /// List all registered sampler names
    pub fn list_samplers(&self) -> Vec<String> {
        self.sampler_factories.read().keys().cloned().collect()
    }

    /// List all registered scheduler names
    pub fn list_schedulers(&self) -> Vec<String> {
        self.scheduler_factories.read().keys().cloned().collect()
    }

    /// List all registered KV cache names
    pub fn list_kv_caches(&self) -> Vec<String> {
        self.kv_cache_factories.read().keys().cloned().collect()
    }

    /// List all registered executor names
    pub fn list_executors(&self) -> Vec<String> {
        self.executor_factories.read().keys().cloned().collect()
    }

    // ========================================================================
    // Convenience creation methods
    // ========================================================================

    /// Create a tokenizer by name
    pub async fn create_tokenizer(
        &self,
        name: &str,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
        let factory = self.get_tokenizer_factory(name).ok_or_else(|| {
            FerrumError::tokenizer(format!(
                "Tokenizer '{}' not found. Available: {:?}",
                name,
                self.list_tokenizers()
            ))
        })?;
        factory.create(config).await
    }

    /// Create a sampler by name
    pub async fn create_sampler(
        &self,
        name: &str,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn Sampler + Send + Sync>> {
        let factory = self.get_sampler_factory(name).ok_or_else(|| {
            FerrumError::internal(format!(
                "Sampler '{}' not found. Available: {:?}",
                name,
                self.list_samplers()
            ))
        })?;
        factory.create(config).await
    }

    /// Create a scheduler by name
    pub async fn create_scheduler(
        &self,
        name: &str,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn Scheduler + Send + Sync>> {
        let factory = self.get_scheduler_factory(name).ok_or_else(|| {
            FerrumError::scheduler(format!(
                "Scheduler '{}' not found. Available: {:?}",
                name,
                self.list_schedulers()
            ))
        })?;
        factory.create(config).await
    }

    /// Create a KV cache manager by name
    pub async fn create_kv_cache(
        &self,
        name: &str,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn KvCacheManager + Send + Sync>> {
        let factory = self.get_kv_cache_factory(name).ok_or_else(|| {
            FerrumError::internal(format!(
                "KV cache '{}' not found. Available: {:?}",
                name,
                self.list_kv_caches()
            ))
        })?;
        factory.create(config).await
    }

    /// Create a model executor by name
    pub async fn create_executor(
        &self,
        name: &str,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn ModelExecutor + Send + Sync>> {
        let factory = self.get_executor_factory(name).ok_or_else(|| {
            FerrumError::model(format!(
                "Executor '{}' not found. Available: {:?}",
                name,
                self.list_executors()
            ))
        })?;
        factory.create(config).await
    }
}

impl Default for ComponentRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl std::fmt::Debug for ComponentRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentRegistry")
            .field("tokenizers", &self.list_tokenizers())
            .field("samplers", &self.list_samplers())
            .field("schedulers", &self.list_schedulers())
            .field("kv_caches", &self.list_kv_caches())
            .field("executors", &self.list_executors())
            .finish()
    }
}

// ============================================================================
// Default Factory Implementations
// ============================================================================

// ----------------------------------------------------------------------------
// Backend Factories
// ----------------------------------------------------------------------------

// CandleBackendFactory deleted in Phase: legacy `ComputeBackend` trait
// gone; the registry's component graph never produced a runtime backend
// (`_backend` in builder.rs was unused). The stub-executor factory now
// constructs `CandleTensorFactory` directly when it needs to mint dummy
// logits; everything else dispatches via `Backend<B>` in ferrum-kernels.

// ----------------------------------------------------------------------------
// Tokenizer Factories
// ----------------------------------------------------------------------------

/// HuggingFace tokenizer factory
pub struct HuggingFaceTokenizerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Tokenizer + Send + Sync>> for HuggingFaceTokenizerFactory {
    async fn create(&self, config: &ComponentConfig) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
        // Try to find tokenizer path from config or environment
        let tokenizer_path = config
            .get_string_option("tokenizer_path")
            .or_else(|| config.get_string_option("model_path"))
            .or_else(|| config.engine_config.runtime.model_path.clone());

        if let Some(model_path) = tokenizer_path {
            let path = std::path::Path::new(&model_path);
            // GGUF path: model_path is a file. Auto-discover a sibling
            // tokenizer.json (or the convention used by ~/ferrum-bench).
            let tokenizer_file = if ferrum_models::gguf_engine_loader::is_gguf_path(&model_path) {
                ferrum_models::gguf_engine_loader::auto_discover_tokenizer_path(path).ok_or_else(
                    || {
                        FerrumError::tokenizer(format!(
                            "Could not find tokenizer.json for {} — \
                             place it next to the .gguf file or in a \
                             sibling tokenizers/ directory",
                            path.display()
                        ))
                    },
                )?
            } else {
                // HF safetensors layout: model_path is a directory.
                path.join("tokenizer.json")
            };

            if tokenizer_file.exists() {
                info!("Loading HuggingFace tokenizer from: {:?}", tokenizer_file);
                match ferrum_tokenizer::implementations::HuggingFaceTokenizer::from_file(
                    &tokenizer_file.to_string_lossy(),
                )
                .await
                {
                    Ok(tokenizer) => {
                        info!("HuggingFace tokenizer loaded successfully");
                        return Ok(Arc::new(tokenizer));
                    }
                    Err(e) => {
                        tracing::warn!("Failed to load tokenizer: {}, falling back to stub", e);
                    }
                }
            }
        }

        // Fallback to stub
        Err(FerrumError::tokenizer(
            "HuggingFace tokenizer path not found or invalid",
        ))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "huggingface".to_string(),
            version: "0.1.0".to_string(),
            description: "HuggingFace tokenizers library integration".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec![
                "bpe".to_string(),
                "wordpiece".to_string(),
                "sentencepiece".to_string(),
                "chat_template".to_string(),
            ],
        }
    }
}

/// Stub tokenizer factory for testing
pub struct StubTokenizerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Tokenizer + Send + Sync>> for StubTokenizerFactory {
    async fn create(&self, config: &ComponentConfig) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
        let vocab_size = config
            .engine_config
            .model
            .model_info
            .as_ref()
            .map(|info| info.vocab_size)
            .unwrap_or(32000);

        info!("Creating stub tokenizer with vocab_size: {}", vocab_size);
        Ok(Arc::new(StubTokenizer::new(vocab_size)))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "stub".to_string(),
            version: "0.1.0".to_string(),
            description: "Stub tokenizer for testing".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec!["testing".to_string()],
        }
    }
}

// ----------------------------------------------------------------------------
// Sampler Factories
// ----------------------------------------------------------------------------

/// Multinomial sampler factory
pub struct MultinomialSamplerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Sampler + Send + Sync>> for MultinomialSamplerFactory {
    async fn create(&self, _config: &ComponentConfig) -> Result<Arc<dyn Sampler + Send + Sync>> {
        info!("Creating multinomial sampler");
        Ok(Arc::new(ferrum_interfaces::sampler::MultinomialSampler))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "multinomial".to_string(),
            version: "0.1.0".to_string(),
            description: "Multinomial sampling with temperature and top-k/top-p".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec![
                "temperature".to_string(),
                "top_k".to_string(),
                "top_p".to_string(),
            ],
        }
    }
}

/// Greedy sampler factory
pub struct GreedySamplerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Sampler + Send + Sync>> for GreedySamplerFactory {
    async fn create(&self, _config: &ComponentConfig) -> Result<Arc<dyn Sampler + Send + Sync>> {
        info!("Creating greedy sampler");
        Ok(Arc::new(GreedySampler))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "greedy".to_string(),
            version: "0.1.0".to_string(),
            description: "Greedy decoding (always pick highest probability)".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec!["deterministic".to_string()],
        }
    }
}

/// Greedy sampler implementation
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(
        &self,
        logits: &[f32],
        _rng: &mut dyn rand::RngCore,
    ) -> Result<ferrum_types::TokenId> {
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| FerrumError::internal("Empty logits"))?;

        Ok(ferrum_types::TokenId::new(max_idx as u32))
    }

    fn name(&self) -> &str {
        "greedy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// ----------------------------------------------------------------------------
// Scheduler Factories
// ----------------------------------------------------------------------------

/// FIFO scheduler factory
pub struct FifoSchedulerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Scheduler + Send + Sync>> for FifoSchedulerFactory {
    async fn create(&self, config: &ComponentConfig) -> Result<Arc<dyn Scheduler + Send + Sync>> {
        info!("Creating FIFO scheduler");
        let scheduler_config = config.engine_config.scheduler.clone();
        let scheduler = ferrum_scheduler::implementations::FifoScheduler::new(scheduler_config);
        Ok(Arc::new(scheduler))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "fifo".to_string(),
            version: "0.1.0".to_string(),
            description: "First-In-First-Out scheduler".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec!["simple".to_string(), "fair".to_string()],
        }
    }
}

/// Priority scheduler factory
pub struct PrioritySchedulerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Scheduler + Send + Sync>> for PrioritySchedulerFactory {
    async fn create(&self, config: &ComponentConfig) -> Result<Arc<dyn Scheduler + Send + Sync>> {
        info!("Creating priority scheduler");
        let scheduler_config = config.engine_config.scheduler.clone();
        let scheduler = ferrum_scheduler::implementations::PriorityScheduler::new(scheduler_config);
        Ok(Arc::new(scheduler))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "priority".to_string(),
            version: "0.1.0".to_string(),
            description: "Priority-based scheduler".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec!["priority".to_string(), "preemption".to_string()],
        }
    }
}

/// Continuous batching scheduler factory
pub struct ContinuousBatchSchedulerFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn Scheduler + Send + Sync>> for ContinuousBatchSchedulerFactory {
    async fn create(&self, config: &ComponentConfig) -> Result<Arc<dyn Scheduler + Send + Sync>> {
        info!("Creating continuous batch scheduler");
        let scheduler_config = config.engine_config.scheduler.clone();
        let scheduler =
            ferrum_scheduler::implementations::ContinuousBatchScheduler::new(scheduler_config);
        Ok(Arc::new(scheduler))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "continuous".to_string(),
            version: "0.1.0".to_string(),
            description: "Continuous batching scheduler with iteration-level scheduling"
                .to_string(),
            supported_devices: cpu_cuda_and_optional_metal_devices(),
            capabilities: vec![
                "continuous_batching".to_string(),
                "preemption".to_string(),
                "chunked_prefill".to_string(),
                "iteration_level".to_string(),
            ],
        }
    }
}

// ----------------------------------------------------------------------------
// KV Cache Factories
// ----------------------------------------------------------------------------

/// Default KV cache factory
pub struct DefaultKvCacheFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn KvCacheManager + Send + Sync>> for DefaultKvCacheFactory {
    async fn create(
        &self,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn KvCacheManager + Send + Sync>> {
        let block_size = config.engine_config.kv_cache.block_size;
        let max_blocks = config.engine_config.kv_cache.max_blocks;

        info!(
            "Creating default KV cache manager: device={:?}, block_size={}, max_blocks={}",
            config.device, block_size, max_blocks
        );

        let manager = ferrum_kv::managers::DefaultKvCacheManager::new(
            config.device.clone(),
            block_size,
            max_blocks,
        )?;
        Ok(Arc::new(manager))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "default".to_string(),
            version: "0.1.0".to_string(),
            description: "Default contiguous KV cache manager".to_string(),
            supported_devices: cpu_cuda_and_optional_metal_devices(),
            capabilities: vec!["contiguous".to_string()],
        }
    }
}

/// Paged KV cache factory (for PagedAttention)
pub struct PagedKvCacheFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn KvCacheManager + Send + Sync>> for PagedKvCacheFactory {
    async fn create(
        &self,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn KvCacheManager + Send + Sync>> {
        let block_size = config.engine_config.kv_cache.block_size;
        let max_blocks = config.engine_config.kv_cache.max_blocks;

        info!(
            "Creating paged KV cache manager: device={:?}, block_size={}, max_blocks={}",
            config.device, block_size, max_blocks
        );

        // Use the PagedKvCacheManager for PagedAttention support
        let paged_config = ferrum_kv::managers::PagedKvCacheConfig {
            block_size,
            max_gpu_blocks: max_blocks,
            max_cpu_blocks: max_blocks / 2,
            enable_cow: true,
            enable_swapping: true,
            ..Default::default()
        };

        let manager =
            ferrum_kv::managers::PagedKvCacheManager::new(config.device.clone(), paged_config)?;
        Ok(Arc::new(manager))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "paged".to_string(),
            version: "0.1.0".to_string(),
            description: "Paged KV cache manager for PagedAttention".to_string(),
            supported_devices: cpu_cuda_and_optional_metal_devices(),
            capabilities: vec![
                "paged".to_string(),
                "copy_on_write".to_string(),
                "swap".to_string(),
            ],
        }
    }
}

// ----------------------------------------------------------------------------
// Executor Factories
// ----------------------------------------------------------------------------

/// Stub executor factory
pub struct StubExecutorFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn ModelExecutor + Send + Sync>> for StubExecutorFactory {
    async fn create(
        &self,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn ModelExecutor + Send + Sync>> {
        let vocab_size = config
            .engine_config
            .model
            .model_info
            .as_ref()
            .map(|info| info.vocab_size)
            .unwrap_or(32000);

        info!(
            "Creating stub executor for model: {}",
            config.engine_config.model.model_id
        );

        // The stub executor only needs a `TensorFactory` to mint dummy
        // logits — no real backend dispatch involved. Wire the candle
        // tensor factory directly without the legacy `ComputeBackend`
        // wrapper.
        let tensor_factory: Arc<dyn ferrum_interfaces::TensorFactory> = Arc::new(
            crate::tensor_factory::candle::CandleTensorFactory::new(config.device.clone()),
        );

        let executor = ferrum_models::StubModelExecutor::new(
            config.engine_config.model.model_id.clone(),
            vocab_size,
            tensor_factory,
        );

        Ok(Arc::new(executor))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "stub".to_string(),
            version: "0.1.0".to_string(),
            description: "Stub executor for testing".to_string(),
            supported_devices: vec![Device::CPU],
            capabilities: vec!["testing".to_string()],
        }
    }
}

/// Candle executor factory
/// Builds a [`ModelExecutor`] from a resolved model path. Despite the
/// historical "candle" name (PR #127 deleted the legacy `CandleBackend`
/// trait), this factory now produces real `Backend<B>`-based executors
/// for LLMs (LlamaFamilyModel / Qwen3MoeModel) and candle-based
/// executors for embedding / multimodal / ASR (Bert, CLIP, Whisper).
///
/// The factory dispatches over four axes:
///   - Dim 1 (architecture): LlamaFamilyModel vs Qwen3MoeModel — runtime
///     value, decided inside [`build_llm`].
///   - Dim 3 (weight format): safetensors vs GGUF — runtime branch on
///     `WeightFormat::detect()` at the top of [`Self::create`].
///   - Dim 4 (device): CpuBackend / MetalBackend / CudaBackend — type
///     parameter `B`, picked by the `(device, kv_dtype)` cascade below.
///   - Dim 5 (kv dtype): KvFp16 / KvInt8 / KvFp8 — type parameter `K`,
///     same cascade. Only `KvFp16` is wired today; `KvInt8` lands in
///     PR C (see `docs/dim5-model-wireup-plan.md`).
pub struct LlmExecutorFactory;

fn resolve_llama_layer_split_plan(
    config: &ComponentConfig,
    num_layers: usize,
) -> Result<Option<crate::layer_split::ParsedLayerSplitPlan>> {
    if config
        .get_string_option("selected_distributed_strategy")
        .as_deref()
        != Some("layer_split")
    {
        return Ok(None);
    }

    let selected = config
        .get_option::<Vec<usize>>("selected_gpu_devices")
        .unwrap_or_default();
    let plan_raw = config.get_string_option("selected_layer_split_plan");
    let parsed_plan =
        if let Some(stages) = config.component_options.get("selected_layer_split_stages") {
            crate::layer_split::parse_layer_split_stage_documents(stages)?
        } else {
            let plan_raw = plan_raw.as_deref().ok_or_else(|| {
                FerrumError::config(
                    "selected_distributed_strategy=layer_split requires selected_layer_split_plan",
                )
            })?;
            crate::layer_split::parse_layer_split_plan(plan_raw)?
        };
    crate::layer_split::validate_layer_split_plan_for_devices(&parsed_plan, &selected)?;
    if parsed_plan.total_layers() != num_layers {
        return Err(FerrumError::config(format!(
            "selected_layer_split_plan covers {} layers but model has {num_layers}",
            parsed_plan.total_layers()
        )));
    }
    Ok(Some(parsed_plan))
}

fn resolve_llama_layer_split_pipeline_mode(
    config: &ComponentConfig,
    stage_count: usize,
) -> Result<ferrum_models::models::LlamaPipelineMode> {
    let Some(mode) = config.get_string_option("layer_split_pipeline_mode") else {
        return Ok(ferrum_models::models::LlamaPipelineMode::default_for_stage_count(stage_count));
    };
    let mode = ferrum_models::models::LlamaPipelineMode::from_config_value(&mode)?;
    if mode == ferrum_models::models::LlamaPipelineMode::Overlapped && stage_count != 2 {
        return Err(FerrumError::config(
            "layer_split_pipeline_mode=overlapped requires exactly two pipeline stages",
        ));
    }
    Ok(mode)
}

#[cfg(test)]
fn resolve_llama_layer_stage_config(
    config: &ComponentConfig,
    num_layers: usize,
) -> Result<Option<ferrum_models::models::llama_family::LlamaFamilyLayerStageConfig>> {
    let Some(parsed_plan) = resolve_llama_layer_split_plan(config, num_layers)? else {
        return Ok(None);
    };
    let device_id = match &config.device {
        Device::CUDA(device_id) => *device_id,
        other => {
            return Err(FerrumError::unsupported(format!(
                "selected_distributed_strategy=layer_split requires a CUDA stage device, got {other:?}",
            )));
        }
    };
    parsed_plan
        .llama_stage_config_for_device(device_id)
        .map(Some)
}

/// Generic LLM construction helper. Picks the model type (LlamaFamilyModel
/// or Qwen3MoeModel) by `arch`, opens a `NativeSafetensorsLoader<B>`, and
/// returns a `Box<dyn DecoderOnlyLLM>` ready to be wrapped in `LlmExecutor`.
///
/// Generic over both Dim 4 (`B`: hardware backend) and Dim 5 (`K`: KV
/// element type). Adding INT8 KV in PR C means adding the
/// `(Device::CUDA, KvCacheDtype::Int8) => build_llm::<CudaBackend, KvInt8>(...)`
/// arm to the cascade — this helper already accepts `K` so the callers
/// don't need to be touched again.
fn build_llm<B, K>(
    arch: ferrum_models::Architecture,
    qcfg: ferrum_models::models::LlamaFamilyConfig,
    moe_cfg: Option<ferrum_models::moe_config::Qwen3MoeConfig>,
    model_path: &str,
    llama_layer_split_plan: Option<crate::layer_split::ParsedLayerSplitPlan>,
    llama_layer_split_pipeline_mode: Option<ferrum_models::models::LlamaPipelineMode>,
) -> Result<Box<dyn ferrum_models::common::DecoderOnlyLLM>>
where
    B: ferrum_kernels::backend::MoeLlmBackend,
    K: ferrum_kernels::backend::KvLayer<B>,
    ferrum_models::models::LlamaFamilyModel<B, K>: ferrum_models::common::DecoderOnlyLLM,
    ferrum_models::models::LlamaFamilyPipelineModel<B, K>: ferrum_models::common::DecoderOnlyLLM,
{
    if matches!(arch, ferrum_models::Architecture::Qwen3Moe) {
        if llama_layer_split_plan.is_some() {
            return Err(FerrumError::unsupported(
                "CUDA layer_split stage loading is wired only for Llama-family dense models; \
                 Qwen3MoeModel requires a separate MoE stage loader.",
            ));
        }
        let weight_loader = ferrum_quantization::NativeSafetensorsLoader::<B>::open(model_path)?;
        let mc = moe_cfg.ok_or_else(|| {
            FerrumError::internal(
                "Qwen3Moe arch reached build_llm without Qwen3MoeConfig (caller bug)",
            )
        })?;
        Ok(Box::new(
            ferrum_models::models::Qwen3MoeModel::<B, K>::new_safetensors(mc, &weight_loader)?,
        ))
    } else {
        if llama_layer_split_plan.is_some() && !B::supports_device_ordinal_scope() {
            return Err(FerrumError::unsupported(
                "selected_distributed_strategy=layer_split requires a backend with \
                 device-scoped execution; refusing to silently use the default device",
            ));
        }
        let weight_loader = ferrum_quantization::NativeSafetensorsLoader::<B>::open(model_path)?;
        if let Some(plan) = llama_layer_split_plan {
            let stage_configs = plan.to_llama_stage_configs();
            let stage_device_ordinals = plan
                .stages
                .iter()
                .map(|stage| Some(stage.device))
                .collect::<Vec<_>>();
            let mut stages = Vec::with_capacity(stage_configs.len());
            for (idx, (stage_config, device_ordinal)) in stage_configs
                .into_iter()
                .zip(stage_device_ordinals.iter().copied())
                .enumerate()
            {
                tracing::info!(
                    "Loading Llama layer_split stage {idx} on backend device {:?}",
                    device_ordinal
                );
                let stage = B::with_device_ordinal(device_ordinal, || {
                    ferrum_models::models::LlamaFamilyModel::<B, K>::new_layer_stage(
                        qcfg.clone(),
                        &weight_loader,
                        stage_config,
                    )
                })?;
                stages.push(stage);
            }
            Ok(Box::new(ferrum_models::models::LlamaFamilyPipelineModel::<
                B,
                K,
            >::new_with_placement(
                stages,
                ferrum_models::models::LlamaPipelinePlacement::from_backend_device_ordinals(
                    stage_device_ordinals,
                )
                .with_pipeline_mode(
                    llama_layer_split_pipeline_mode.expect("layer split pipeline mode resolved"),
                ),
            )?))
        } else {
            Ok(Box::new(
                ferrum_models::models::LlamaFamilyModel::<B, K>::new(qcfg, &weight_loader)?,
            ))
        }
    }
}

fn create_registered_vnext_executor(
    config: &ComponentConfig,
    model_path: &std::path::Path,
    registration: ferrum_models::vnext::RegisteredProductionModel,
) -> Result<Arc<dyn ModelExecutor + Send + Sync>> {
    use ferrum_models::vnext::ProductionExecutionKind;

    match (registration.execution_kind(), &config.device) {
        (ProductionExecutionKind::CausalLanguage, Device::CUDA(ordinal)) => {
            #[cfg(feature = "cuda")]
            {
                let prepared = registration.prepare(model_path)?;
                let model_info = prepared.model_info(
                    config.engine_config.model.model_id.clone(),
                    config.device.clone(),
                );
                let family = prepared.family();
                let family_fingerprint = family
                    .fingerprint()
                    .map_err(|error| FerrumError::model(error.to_string()))?;
                let program_fingerprint = family
                    .program()
                    .fingerprint()
                    .map_err(|error| FerrumError::model(error.to_string()))?;
                info!(
                    external_metadata_id = %registration.external_metadata_id(),
                    family_id = %family.family_id(),
                    family_fingerprint,
                    program_fingerprint,
                    backend = "cuda",
                    "Building registered model from a typed vNext execution plan"
                );
                let device_id = ferrum_interfaces::vnext::DeviceId::new(format!(
                    "device.cuda.{ordinal}"
                ))
                .map_err(|error| FerrumError::device(error.to_string()))?;
                let composition =
                    ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition::create(
                        *ordinal, device_id,
                    )
                    .map_err(|error| {
                        FerrumError::device(format!("create vNext CUDA runtime: {error}"))
                    })?;
                let (runtime, operation_registry, catalog) = composition.into_parts();
                let executor = crate::product_composition::create_vnext_executor(
                    model_path,
                    &config.engine_config,
                    prepared,
                    model_info,
                    runtime,
                    operation_registry,
                    catalog,
                )?;
                info!(
                    resolved_plan_fingerprint = executor
                        .resolved_model_plan()
                        .map(|plan| plan.fingerprint())
                        .unwrap_or("missing"),
                    "Resolved product model plan is authoritative for vNext execution"
                );
                Ok(Arc::new(executor))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (ordinal, model_path);
                Err(FerrumError::device(
                    "registered vNext CUDA composition requires the 'cuda' feature",
                ))
            }
        }
        (kind, device) => Err(FerrumError::unsupported(format!(
            "registered vNext model metadata {} requires a {kind:?} backend composition, but {device} is not registered",
            registration.external_metadata_id()
        ))),
    }
}

/// Back-compat alias; retained so external test fixtures referencing
/// the old name continue to compile while we sweep call sites.
#[deprecated(note = "use `LlmExecutorFactory` (renamed PR A — Dim 1/3 cleanup)")]
pub type CandleExecutorFactory = LlmExecutorFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn ModelExecutor + Send + Sync>> for LlmExecutorFactory {
    async fn create(
        &self,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn ModelExecutor + Send + Sync>> {
        use candle_core::{DType, Device as CandleDevice};
        use ferrum_models::weight_format::WeightFormat;

        // Try to load model from path
        let model_path = config
            .get_string_option("model_path")
            .or_else(|| config.engine_config.runtime.model_path.clone());

        let model_path = match model_path {
            Some(path) => path,
            None => {
                info!("No model path found, falling back to stub executor");
                return StubExecutorFactory.create(config).await;
            }
        };

        // Dim 3 dispatch — peer-level enum (no GGUF special-case at the
        // top of an architecture cascade). Adding a new format (AWQ /
        // EXL2 / HQQ) means a new `WeightFormat` variant + a matching
        // `WeightLoader<B>` impl in `ferrum-quantization`.
        let weight_fmt = WeightFormat::detect(std::path::Path::new(&model_path))?;
        info!(
            "Loading model from {} (format: {})",
            model_path,
            weight_fmt.label()
        );

        if let WeightFormat::Gguf { ref path } = weight_fmt {
            // GGUF goes through its own loader — single-file format with
            // architecture + tokenizer baked in, no separate config.json
            // step. Output is a `Box<dyn DecoderOnlyLLM>` that plugs into
            // the same LlmExecutor as the safetensors path, so the rest of
            // the engine is unchanged.
            let (llm, model_info) = ferrum_models::gguf_engine_loader::load_gguf_decoder_with_info(
                path,
                &config.device,
                config.engine_config.model.model_id.clone(),
            )?;
            return Ok(Arc::new(ferrum_models::LlmExecutor::new(llm, model_info)));
        }

        // Registered vNext packages resolve from immutable external metadata
        // before the legacy ConfigManager/Architecture cascade. Backend
        // compatibility is checked before the package opens model weights;
        // once a package is registered, any preparation error fails closed and
        // cannot fall back to a legacy executor.
        let legacy_reference_enabled = config
            .get_option::<bool>("qwen35_reference")
            .unwrap_or(false);
        let production_registration = ferrum_models::vnext::resolve_registered_model_from_dir(
            std::path::Path::new(&model_path),
        )?;
        if legacy_reference_enabled && !production_registration.allows_legacy_reference() {
            return Err(FerrumError::unsupported(format!(
                "--qwen35-reference is not permitted for model metadata {}",
                production_registration.external_metadata_id()
            )));
        }
        if legacy_reference_enabled {
            info!(
                external_metadata_id = %production_registration.external_metadata_id(),
                "Entering the explicitly allowed Qwen3.5 CPU reference path"
            );
        } else {
            match production_registration {
                ferrum_models::vnext::ProductionModelRegistration::Registered(registration) => {
                    return create_registered_vnext_executor(
                        config,
                        std::path::Path::new(&model_path),
                        registration,
                    );
                }
                ferrum_models::vnext::ProductionModelRegistration::LegacyRegistered {
                    external_metadata_id,
                    ..
                } => {
                    info!(
                        %external_metadata_id,
                        "Entering the explicitly registered legacy model path"
                    );
                }
            }
        }

        // Explicitly registered legacy safetensors and the diagnostic CPU
        // reference path continue through the old registry until each family
        // migration deletes its legacy row and architecture branch together.
        let mut config_manager = ferrum_models::ConfigManager::new();
        let model_def = config_manager
            .load_from_path(std::path::Path::new(&model_path))
            .await?;

        info!(
            "  Architecture: {:?}, Layers: {}, Vocab: {}",
            model_def.architecture, model_def.num_hidden_layers, model_def.vocab_size
        );

        // Determine device
        let candle_device = match &config.device {
            Device::CPU => CandleDevice::Cpu,
            Device::CUDA(id) => CandleDevice::new_cuda(*id)
                .map_err(|e| FerrumError::device(format!("CUDA error: {}", e)))?,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            Device::Metal => CandleDevice::new_metal(0)
                .map_err(|e| FerrumError::device(format!("Metal error: {}", e)))?,
            Device::ROCm(_) => {
                return Err(FerrumError::device("ROCm not yet supported"));
            }
        };

        // Select dtype.
        //
        // IMPORTANT (correctness): Metal defaults to FP32 for stability.
        // This was introduced to address cases where CPU inference is correct but Metal results
        // can deviate. If you want higher performance and accept potential numerical risk, you
        // can explicitly opt-in via env:
        // - FERRUM_METAL_DTYPE=fp16|fp32 (takes precedence on Metal)
        // - FERRUM_DTYPE=fp16|fp32 (global override for non-CPU)
        let dtype: DType = RegistryRuntimeEnv::from_runtime_knobs(&config.engine_config.runtime)
            .dtype_for_device(&config.device);

        // Create model based on architecture
        info!("Building model...");
        match model_def.architecture {
            // All Llama-family decoders (Llama / Llama-2 / Llama-3 / Qwen2 /
            // Qwen2.5 / Qwen3) share `LlamaFamilyModel<B>` + `LlmExecutor`.
            // Only the config constructor differs.
            arch @ (ferrum_models::Architecture::Llama
            | ferrum_models::Architecture::Qwen2
            | ferrum_models::Architecture::Qwen3
            | ferrum_models::Architecture::Qwen3Moe
            | ferrum_models::Architecture::Gemma3
            | ferrum_models::Architecture::Mistral) => {
                let _loader = ferrum_models::SafeTensorsLoader::new(&model_path);
                let model_dir_path: std::path::PathBuf = model_path.clone().into();

                // Tensor parallelism (FERRUM_TP>1) was wired against the
                // pre-Architecture-v2 CandleBackend and hasn't been ported
                // to the Backend<B> trait stack. Reject explicitly so users
                // don't get a silent single-GPU fallback.
                let tp_size = config.engine_config.runtime.tp.unwrap_or(0);
                if tp_size > 1 {
                    return Err(FerrumError::unsupported(
                        "FERRUM_TP>1 not supported on the Backend<B> path. \
                         Run with FERRUM_TP=1 (default) for single-GPU inference.",
                    ));
                }
                // GPTQ is loaded via NativeSafetensorsLoader's load_linear:
                // it auto-detects `<name>.qweight` tensors and constructs
                // a GptqLinear via Backend::load_gptq. The QuantizeConfig
                // probe is kept for back-compat with old configs that
                // explicitly listed a quantization method.
                let _ = ferrum_models::loader::QuantizeConfig::from_model_dir(&model_dir_path);

                // Resolve the architecture-specific config in one place.
                // Qwen3-MoE needs both `LlamaFamilyConfig` (dense attention
                // shares the Llama path) AND `Qwen3MoeConfig` (router +
                // experts); other arches only need `LlamaFamilyConfig`.
                let (qcfg, moe_cfg): (
                    ferrum_models::models::LlamaFamilyConfig,
                    Option<ferrum_models::moe_config::Qwen3MoeConfig>,
                ) = match arch {
                    ferrum_models::Architecture::Qwen3Moe => {
                        info!("Loading Qwen3-MoE via Qwen3MoeModel (safetensors GPTQ)");
                        let mc = ferrum_models::moe_config::Qwen3MoeConfig::from_def(&model_def)?;
                        // dense attention reuses the Qwen3 dense config —
                        // build_llm only consumes `qcfg` on the LlamaFamily
                        // arm, so passing the MoE's `base` is fine.
                        (mc.base.clone(), Some(mc))
                    }
                    ferrum_models::Architecture::Qwen3 => {
                        info!("Loading Qwen3 via LlamaFamilyModel (QK-norm on)");
                        (
                            ferrum_models::models::LlamaFamilyConfig::qwen3_from_def(&model_def),
                            None,
                        )
                    }
                    ferrum_models::Architecture::Qwen2 => {
                        info!("Loading Qwen2 via LlamaFamilyModel");
                        (
                            ferrum_models::models::LlamaFamilyConfig::qwen2_from_def(&model_def),
                            None,
                        )
                    }
                    ferrum_models::Architecture::Mistral => {
                        info!("Loading Mistral via LlamaFamilyModel (sliding_window from config)");
                        (
                            ferrum_models::models::LlamaFamilyConfig::mistral_from_def(&model_def),
                            None,
                        )
                    }
                    ferrum_models::Architecture::Gemma3 => {
                        info!(
                            "Loading Gemma3 via LlamaFamilyModel (5:1 SWA, dual rope, GeGLU, \
                             sandwich norms)"
                        );
                        (
                            ferrum_models::models::LlamaFamilyConfig::gemma3_from_def(&model_def),
                            None,
                        )
                    }
                    _ => {
                        info!("Loading Llama via LlamaFamilyModel");
                        (
                            ferrum_models::models::LlamaFamilyConfig::llama_from_def(&model_def),
                            None,
                        )
                    }
                };

                let model_info =
                    model_def.to_model_info(config.engine_config.model.model_id.to_string());
                let llama_layer_split_plan =
                    resolve_llama_layer_split_plan(config, qcfg.num_layers)?;
                let llama_layer_split_pipeline_mode = llama_layer_split_plan
                    .as_ref()
                    .map(|plan| resolve_llama_layer_split_pipeline_mode(config, plan.stages.len()))
                    .transpose()?;

                // (Dim 4, Dim 5) cascade: pick `B` from device, `K` from
                // kv-dtype, and dispatch to the generic `build_llm` helper.
                // PR C will extend this with `(CUDA, Int8) => build_llm::<CudaBackend, KvInt8>(...)`
                // — model wire-up already accepts `K`, so adding INT8 only
                // touches this match.
                use ferrum_interfaces::kv_dtype::KvFp16;
                #[cfg(feature = "cuda")]
                use ferrum_interfaces::kv_dtype::KvInt8;
                use ferrum_types::KvCacheDtype;
                let kv_dtype = config.engine_config.kv_cache.dtype;
                let llm: Box<dyn ferrum_models::common::DecoderOnlyLLM> =
                    match (&config.device, kv_dtype) {
                        (Device::CPU, KvCacheDtype::Fp16) => {
                            info!("  Backend: CPU, KV: fp16");
                            build_llm::<ferrum_kernels::backend::cpu::CpuBackend, KvFp16>(
                                arch,
                                qcfg,
                                moe_cfg,
                                &model_path,
                                llama_layer_split_plan,
                                llama_layer_split_pipeline_mode,
                            )?
                        }
                        #[cfg(any(target_os = "macos", target_os = "ios"))]
                        (Device::Metal, KvCacheDtype::Fp16) => {
                            #[cfg(feature = "metal")]
                            {
                                // FERRUM_METAL_DTYPE=f16 toggles fp16 weight storage
                                // inside MetalBackend. Halves big-tensor RAM;
                                // recommended for 4B+ models on 16 GB Macs.
                                let dtype_hint = RegistryRuntimeEnv::from_runtime_knobs(
                                    &config.engine_config.runtime,
                                )
                                .metal_dtype_hint();
                                info!("  Backend: Metal (weights {}), KV: fp16", dtype_hint);
                                build_llm::<ferrum_kernels::backend::metal::MetalBackend, KvFp16>(
                                    arch,
                                    qcfg,
                                    moe_cfg,
                                    &model_path,
                                    llama_layer_split_plan,
                                    llama_layer_split_pipeline_mode,
                                )?
                            }
                            #[cfg(not(feature = "metal"))]
                            {
                                return Err(FerrumError::device(
                                    "Metal requested but 'metal' feature not enabled",
                                ));
                            }
                        }
                        (Device::CUDA(_), KvCacheDtype::Fp16) => {
                            #[cfg(feature = "cuda")]
                            {
                                info!("  Backend: CUDA, KV: fp16");
                                build_llm::<ferrum_kernels::backend::cuda::CudaBackend, KvFp16>(
                                    arch,
                                    qcfg,
                                    moe_cfg,
                                    &model_path,
                                    llama_layer_split_plan,
                                    llama_layer_split_pipeline_mode,
                                )?
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                return Err(FerrumError::device(
                                    "CUDA requested but 'cuda' feature not enabled",
                                ));
                            }
                        }
                        (Device::CUDA(_), KvCacheDtype::Int8) => {
                            #[cfg(feature = "cuda")]
                            {
                                // Dim 5 PR C: only CudaBackend implements
                                // BackendInt8KvOps with real launchers; LlamaFamilyModel<B, KvInt8>
                                // uses LayerKvCache::Int8 + paged INT8 KV. Qwen3-MoE
                                // INT8 KV is a follow-up — reject here so users
                                // see a clear error instead of a panic at
                                // alloc_paged_int8_layer time.
                                if matches!(arch, ferrum_models::Architecture::Qwen3Moe) {
                                    return Err(FerrumError::unsupported(
                                        "INT8 KV cache is not yet wired through Qwen3MoeModel \
                                     (LlamaFamilyModel-only in PR C). Use --kv-dtype fp16 \
                                     for MoE models or wait for the follow-up PR.",
                                    ));
                                }
                                info!("  Backend: CUDA, KV: int8 (paged, vLLM-style)");
                                build_llm::<ferrum_kernels::backend::cuda::CudaBackend, KvInt8>(
                                    arch,
                                    qcfg,
                                    moe_cfg,
                                    &model_path,
                                    llama_layer_split_plan,
                                    llama_layer_split_pipeline_mode,
                                )?
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                return Err(FerrumError::device(
                                    "CUDA requested but 'cuda' feature not enabled",
                                ));
                            }
                        }
                        (dev, dt) => {
                            return Err(FerrumError::unsupported(format!(
                                "(device={dev:?}, kv_dtype={dt:?}) not implemented — \
                             see docs/dim5-model-wireup-plan.md"
                            )));
                        }
                    };

                Ok(Arc::new(ferrum_models::LlmExecutor::new(llm, model_info)))
            }
            ferrum_models::Architecture::Bert => {
                info!("Using BERT executor for embeddings");
                let executor = ferrum_models::BertModelExecutor::from_path(
                    &model_path,
                    &model_def,
                    candle_device.clone(),
                )
                .await?;

                Ok(Arc::new(executor))
            }
            ferrum_models::Architecture::Clip => {
                info!("Using CLIP executor for multimodal embeddings");
                let executor = ferrum_models::ClipModelExecutor::from_path(
                    &model_path,
                    candle_device.clone(),
                    dtype,
                )?;
                Ok(Arc::new(executor))
            }
            ferrum_models::Architecture::Whisper => {
                info!("Using Whisper executor for ASR");
                let executor = ferrum_models::WhisperModelExecutor::from_path(
                    &model_path,
                    candle_device.clone(),
                    dtype,
                )?;
                Ok(Arc::new(executor))
            }
            ferrum_models::Architecture::Qwen35 | ferrum_models::Architecture::Qwen35Moe => {
                if !legacy_reference_enabled {
                    return Err(FerrumError::unsupported(format!(
                        "model metadata for {:?} is registered for legacy reference execution only; enable --qwen35-reference with CPU/FP32 or migrate the family to vNext",
                        model_def.architecture
                    )));
                }
                if config.device != Device::CPU || dtype != DType::F32 {
                    return Err(FerrumError::unsupported(
                        "Qwen3.5/Qwen3.6 reference execution requires --backend cpu and FP32; \
                         CUDA/Metal release execution is not wired yet.",
                    ));
                }
                info!("Using Qwen3.5/Qwen3.6 explicit CPU/FP32 reference executor");
                let model_dir = std::path::Path::new(&model_path);
                let model_id = config.engine_config.model.model_id.to_string();
                let executor = match model_def.architecture {
                    ferrum_models::Architecture::Qwen35 => {
                        ferrum_models::Qwen35W3Executor::from_definition_with_dense_reference_cpu_safetensors(
                            model_id,
                            &model_def,
                            model_dir,
                            DataType::FP32,
                            Device::CPU,
                        )?
                    }
                    ferrum_models::Architecture::Qwen35Moe => {
                        ferrum_models::Qwen35W3Executor::from_definition_with_sparse_moe_reference_cpu_safetensors(
                            model_id,
                            &model_def,
                            model_dir,
                            DataType::FP32,
                            Device::CPU,
                        )?
                    }
                    _ => unreachable!("matched Qwen35 architectures above"),
                };
                Ok(Arc::new(executor))
            }
            _ => Err(FerrumError::model(format!(
                "Architecture {:?} not supported",
                model_def.architecture
            ))),
        }
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "llm".to_string(),
            version: "0.2.0".to_string(),
            description: "LLM executor (LlamaFamily / Qwen3MoE via Backend<B>; \
                 BERT / CLIP / Whisper via candle)"
                .to_string(),
            supported_devices: cpu_cuda_and_optional_metal_devices(),
            capabilities: vec![
                "llama".to_string(),
                "qwen2".to_string(),
                "qwen3".to_string(),
                "qwen3_moe".to_string(),
                "mistral".to_string(),
                "bert".to_string(),
                "clip".to_string(),
                "whisper".to_string(),
                "safetensors".to_string(),
                "gguf".to_string(),
                "fp16".to_string(),
                "fp32".to_string(),
            ],
        }
    }
}

// ============================================================================
// Stub Tokenizer Implementation
// ============================================================================

/// Stub tokenizer for testing
pub struct StubTokenizer {
    vocab_size: usize,
    info: ferrum_interfaces::TokenizerInfo,
}

impl StubTokenizer {
    /// Create a new stub tokenizer
    pub fn new(vocab_size: usize) -> Self {
        let info = ferrum_interfaces::TokenizerInfo {
            tokenizer_type: ferrum_interfaces::tokenizer::TokenizerType::BPE,
            vocab_size,
            special_tokens: ferrum_types::SpecialTokens::default(),
            supports_incremental: false,
            supports_chat_template: false,
            max_token_length: None,
            model_name: Some("stub".into()),
        };

        Self { vocab_size, info }
    }
}

impl std::fmt::Debug for StubTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StubTokenizer")
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

impl Tokenizer for StubTokenizer {
    fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<ferrum_types::TokenId>> {
        let tokens: Vec<ferrum_types::TokenId> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| ferrum_types::TokenId::new((i % self.vocab_size) as u32))
            .collect();

        Ok(if tokens.is_empty() {
            vec![ferrum_types::TokenId::new(0)]
        } else {
            tokens
        })
    }

    fn decode(&self, tokens: &[ferrum_types::TokenId], _skip_special: bool) -> Result<String> {
        Ok(tokens
            .iter()
            .map(|t| format!("token_{}", t.get()))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn decode_incremental(
        &self,
        _prev: &[ferrum_types::TokenId],
        next: ferrum_types::TokenId,
    ) -> Result<String> {
        Ok(format!("token_{} ", next.get()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        &self.info.special_tokens
    }

    fn token_id(&self, _text: &str) -> Option<ferrum_types::TokenId> {
        Some(ferrum_types::TokenId::new(0))
    }

    fn token_text(&self, _token_id: ferrum_types::TokenId) -> Option<&str> {
        None
    }

    fn info(&self) -> ferrum_interfaces::TokenizerInfo {
        self.info.clone()
    }
}

// ============================================================================
// Global Registry
// ============================================================================

/// Global registry instance
static GLOBAL_REGISTRY: OnceLock<Arc<ComponentRegistry>> = OnceLock::new();

/// Get the global registry, initializing with defaults if needed
pub fn global_registry() -> Arc<ComponentRegistry> {
    GLOBAL_REGISTRY
        .get_or_init(|| {
            info!("Initializing global component registry");
            Arc::new(ComponentRegistry::with_defaults())
        })
        .clone()
}

/// Initialize global registry with a custom registry
pub fn set_global_registry(registry: Arc<ComponentRegistry>) -> Result<()> {
    GLOBAL_REGISTRY
        .set(registry)
        .map_err(|_| FerrumError::internal("Global registry already initialized"))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::model_executor::PrefillInput;
    use ferrum_testkit::MockTensor;
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use std::path::{Path, PathBuf};

    fn unique_test_dir(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "ferrum-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_qwen35_reference_config(dir: &Path) {
        let config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vocab_size": 3,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2,
                "intermediate_size": 2,
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1,
                "linear_conv_kernel_dim": 1,
                "head_dim": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "tie_word_embeddings": false
            }
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
    }

    fn write_qwen35_reference_safetensors(dir: &Path) {
        let tensors: Vec<(String, Vec<f32>)> = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            ("model.norm.weight".to_string(), vec![0.0, 0.0]),
            (
                "model.lm_head.weight".to_string(),
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            (
                "model.layers.0.input_layernorm.weight".to_string(),
                vec![0.0, 0.0],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_string(),
                vec![0.0, 0.0],
            ),
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight".to_string(),
                vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight".to_string(),
                vec![1.0, -1.0],
            ),
            (
                "model.layers.0.linear_attn.in_proj_b.weight".to_string(),
                vec![0.5, 0.25],
            ),
            (
                "model.layers.0.linear_attn.in_proj_a.weight".to_string(),
                vec![-0.25, 0.75],
            ),
            (
                "model.layers.0.linear_attn.conv1d.weight".to_string(),
                vec![1.0, 1.0, 1.0],
            ),
            ("model.layers.0.linear_attn.A_log".to_string(), vec![0.0]),
            ("model.layers.0.linear_attn.dt_bias".to_string(), vec![0.0]),
            (
                "model.layers.0.linear_attn.norm.weight".to_string(),
                vec![1.0],
            ),
            (
                "model.layers.0.linear_attn.out_proj.weight".to_string(),
                vec![1.0, -0.5],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight".to_string(),
                vec![0.2, 0.1, -0.1, 0.3],
            ),
            (
                "model.layers.0.mlp.up_proj.weight".to_string(),
                vec![0.4, -0.2, 0.3, 0.5],
            ),
            (
                "model.layers.0.mlp.down_proj.weight".to_string(),
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.1.input_layernorm.weight".to_string(),
                vec![0.0, 0.0],
            ),
            (
                "model.layers.1.post_attention_layernorm.weight".to_string(),
                vec![0.0, 0.0],
            ),
            (
                "model.layers.1.self_attn.q_proj.weight".to_string(),
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.1.self_attn.k_proj.weight".to_string(),
                vec![0.5, 0.0, 0.0, 0.5],
            ),
            (
                "model.layers.1.self_attn.v_proj.weight".to_string(),
                vec![1.0, 1.0, -0.5, 0.5],
            ),
            (
                "model.layers.1.self_attn.o_proj.weight".to_string(),
                vec![1.0, 0.0, 0.0, 1.0],
            ),
            (
                "model.layers.1.self_attn.q_norm.weight".to_string(),
                vec![1.0, 1.0],
            ),
            (
                "model.layers.1.self_attn.k_norm.weight".to_string(),
                vec![1.0, 1.0],
            ),
            (
                "model.layers.1.mlp.gate_proj.weight".to_string(),
                vec![-0.2, 0.2, 0.1, 0.3],
            ),
            (
                "model.layers.1.mlp.up_proj.weight".to_string(),
                vec![0.25, 0.5, -0.3, 0.4],
            ),
            (
                "model.layers.1.mlp.down_proj.weight".to_string(),
                vec![0.5, 0.25, -0.2, 0.75],
            ),
        ];
        let views = tensors
            .into_iter()
            .map(|(name, values)| {
                let bytes = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let bytes: &'static [u8] = Box::leak(bytes);
                (
                    name,
                    TensorView::new(Dtype::F32, vec![values.len()], bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            views,
            &None::<std::collections::HashMap<String, String>>,
            &dir.join("model.safetensors"),
        )
        .unwrap();
    }

    fn write_qwen35_reference_model_dir() -> PathBuf {
        let dir = unique_test_dir("qwen35-reference");
        write_qwen35_reference_config(&dir);
        write_qwen35_reference_safetensors(&dir);
        dir
    }

    fn qwen35_reference_component_config(model_dir: &Path, enabled: bool) -> ComponentConfig {
        let mut engine_config = EngineConfig::default();
        engine_config.backend.device = Device::CPU;
        engine_config.backend.backend_options.insert(
            "model_path".to_string(),
            serde_json::Value::String(model_dir.to_string_lossy().to_string()),
        );
        if enabled {
            engine_config
                .backend
                .backend_options
                .insert("qwen35_reference".to_string(), serde_json::json!(true));
        }
        ComponentConfig::from_engine_config(&engine_config)
    }

    #[test]
    fn qwen35_registry_routes_registered_package_before_legacy_loading() {
        let dir = write_qwen35_reference_model_dir();
        std::fs::remove_file(dir.join("model.safetensors")).unwrap();
        let config = qwen35_reference_component_config(&dir, false);

        let err = match tokio_test::block_on(LlmExecutorFactory.create(&config)) {
            Ok(_) => panic!("registered Qwen3.5 package unexpectedly accepted CPU execution"),
            Err(err) => err.to_string(),
        };

        assert!(
            err.contains(ferrum_models::vnext::qwen35::EXTERNAL_METADATA_ID),
            "{err}"
        );
        assert!(err.contains("but cpu is not registered"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn qwen35_reference_option_is_rejected_for_other_model_metadata() {
        let dir = write_qwen35_reference_model_dir();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        let config = qwen35_reference_component_config(&dir, true);

        let err = match tokio_test::block_on(LlmExecutorFactory.create(&config)) {
            Ok(_) => panic!("Qwen3.5 reference option unexpectedly accepted Llama metadata"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("--qwen35-reference is not permitted"), "{err}");
        assert!(err.contains("hf.architecture.LlamaForCausalLM"), "{err}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn qwen35_registry_loads_explicit_cpu_reference_executor() {
        let dir = write_qwen35_reference_model_dir();
        let config = qwen35_reference_component_config(&dir, true);

        let executor = tokio_test::block_on(LlmExecutorFactory.create(&config))
            .expect("explicit Qwen3.5 reference path should load toy safetensors");
        let input = PrefillInput::new(MockTensor::from_u32(&[0, 1], &[2]).into_ref());
        let output = tokio_test::block_on(executor.prefill(&input))
            .expect("explicit Qwen3.5 reference path should prefill");

        assert_eq!(output.logits.shape(), &[1, 1, 3]);
        assert_eq!(output.kv_cache.block_table().sequence_length, 2);
        let status = executor.status();
        assert!(status.is_ready, "{status:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_registry_creation() {
        let registry = ComponentRegistry::new();
        assert!(registry.list_tokenizers().is_empty());
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = ComponentRegistry::with_defaults();
        assert!(registry.list_tokenizers().contains(&"stub".to_string()));
        assert!(registry
            .list_samplers()
            .contains(&"multinomial".to_string()));
        assert!(registry.list_schedulers().contains(&"fifo".to_string()));
    }

    #[test]
    fn test_component_config() {
        let mut options = HashMap::new();
        options.insert("test_key".to_string(), serde_json::json!("test_value"));

        let config = ComponentConfig {
            engine_config: EngineConfig::default(),
            device: Device::CPU,
            component_options: options,
        };

        assert_eq!(
            config.get_string_option("test_key"),
            Some("test_value".to_string())
        );
        assert!(config.get_string_option("missing").is_none());
    }

    fn layer_split_component_config_for_device(device: usize) -> ComponentConfig {
        let mut config = EngineConfig::default();
        config.backend.device = Device::CUDA(device);
        config.backend.backend_options.insert(
            "selected_distributed_strategy".to_string(),
            serde_json::Value::String("layer_split".to_string()),
        );
        config.backend.backend_options.insert(
            "selected_gpu_devices".to_string(),
            serde_json::json!([0, 1]),
        );
        config.backend.backend_options.insert(
            "selected_layer_split_plan".to_string(),
            serde_json::Value::String(
                "stage0:cuda:0:layers=0-39;stage1:cuda:1:layers=40-79".to_string(),
            ),
        );
        config.backend.backend_options.insert(
            "selected_layer_split_stages".to_string(),
            serde_json::json!([
                {"stage": 0, "device": 0, "layer_start": 0, "layer_end": 39},
                {"stage": 1, "device": 1, "layer_start": 40, "layer_end": 79}
            ]),
        );
        ComponentConfig::from_engine_config(&config)
    }

    #[test]
    fn resolves_llama_stage_config_for_selected_cuda_device() {
        let config = layer_split_component_config_for_device(1);

        let stage = resolve_llama_layer_stage_config(&config, 80)
            .unwrap()
            .unwrap();

        assert_eq!(stage.source_layers, 40..80);
        assert!(!stage.load_embedding);
        assert!(stage.load_lm_head);
    }

    #[test]
    fn resolves_layer_split_pipeline_mode_default_and_explicit_batch() {
        let mut config = layer_split_component_config_for_device(0);

        assert_eq!(
            resolve_llama_layer_split_pipeline_mode(&config, 2).unwrap(),
            ferrum_models::models::LlamaPipelineMode::Overlapped
        );

        config.component_options.insert(
            "layer_split_pipeline_mode".to_string(),
            serde_json::Value::String("batch".to_string()),
        );
        assert_eq!(
            resolve_llama_layer_split_pipeline_mode(&config, 2).unwrap(),
            ferrum_models::models::LlamaPipelineMode::Batch
        );
    }

    #[test]
    fn rejects_llama_stage_config_when_plan_layer_count_mismatches_model() {
        let config = layer_split_component_config_for_device(0);

        let err = resolve_llama_layer_stage_config(&config, 81)
            .unwrap_err()
            .to_string();

        assert!(err.contains("covers 80 layers but model has 81"));
    }

    #[test]
    fn build_llm_rejects_layer_split_before_weight_open_without_device_scope() {
        use ferrum_interfaces::kv_dtype::KvFp16;
        use ferrum_models::models::LlamaFamilyConfig;

        let qcfg = LlamaFamilyConfig {
            hidden_size: 1,
            intermediate_size: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            num_layers: 2,
            vocab_size: 1,
            max_seq_len: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: None,
            rope_interleaved: false,
            has_qk_norm: false,
            sliding_window: 0,
            ..Default::default()
        };
        let plan = crate::layer_split::parse_layer_split_plan(
            "stage0:cuda:0:layers=0-0;stage1:cuda:1:layers=1-1",
        )
        .unwrap();

        let err = match build_llm::<ferrum_kernels::backend::cpu::CpuBackend, KvFp16>(
            ferrum_models::Architecture::Llama,
            qcfg,
            None,
            "/missing/model/path",
            Some(plan),
            Some(ferrum_models::models::LlamaPipelineMode::Overlapped),
        ) {
            Ok(_) => panic!("layer_split build unexpectedly succeeded"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("device-scoped execution"));
        assert!(err.contains("refusing to silently use the default device"));
    }

    #[test]
    fn test_registry_runtime_env_parses_model_dtype_and_tp() {
        let env = RegistryRuntimeEnv::from_env_vars([
            ("FERRUM_MODEL_PATH", "/models/qwen"),
            ("FERRUM_METAL_DTYPE", "fp16"),
            ("FERRUM_DTYPE", "fp32"),
            ("FERRUM_TP", "4"),
        ]);

        assert_eq!(env.model_path.as_deref(), Some("/models/qwen"));
        assert_eq!(env.metal_dtype.as_deref(), Some("fp16"));
        assert_eq!(env.dtype.as_deref(), Some("fp32"));
        assert_eq!(env.tp, 4);
    }

    #[test]
    fn test_registry_runtime_env_defaults_invalid_tp_and_cuda_dtype() {
        let env = RegistryRuntimeEnv::from_env_vars([
            ("FERRUM_DTYPE", "not-a-dtype"),
            ("FERRUM_TP", "not-a-number"),
        ]);

        assert_eq!(env.model_path(), None);
        assert_eq!(env.tp, 0);
        assert_eq!(
            env.dtype_for_device(&Device::CUDA(0)),
            candle_core::DType::F16
        );
        assert_eq!(env.dtype_for_device(&Device::CPU), candle_core::DType::F32);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    #[test]
    fn test_registry_runtime_env_metal_dtype_precedence() {
        let env = RegistryRuntimeEnv::from_env_vars([
            ("FERRUM_METAL_DTYPE", "fp16"),
            ("FERRUM_DTYPE", "fp32"),
        ]);

        assert_eq!(
            env.dtype_for_device(&Device::Metal),
            candle_core::DType::F16
        );
        assert_eq!(env.metal_dtype_hint(), "fp16");
    }

    #[test]
    fn test_stub_tokenizer() {
        let tokenizer = StubTokenizer::new(100);
        assert_eq!(tokenizer.vocab_size(), 100);

        let tokens = tokenizer.encode("hello world", false).unwrap();
        assert!(!tokens.is_empty());

        let text = tokenizer.decode(&tokens, false).unwrap();
        assert!(text.contains("token_"));
    }

    #[test]
    fn test_greedy_sampler() {
        use rand::SeedableRng;

        let sampler = GreedySampler;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = sampler.sample(&logits, &mut rng).unwrap();

        assert_eq!(token.get(), 3); // Index of 0.9
    }
}
