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
    ComputeBackend, KvCacheManager, ModelExecutor, Sampler, SchedulerInterface as Scheduler,
    Tokenizer,
};
use ferrum_types::{Device, EngineConfig, FerrumError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
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

// ============================================================================
// Component Registry
// ============================================================================

/// Global component registry for managing component factories
pub struct ComponentRegistry {
    backend_factories: RwLock<HashMap<String, Arc<dyn ComponentFactory<Arc<dyn ComputeBackend>>>>>,
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
            backend_factories: RwLock::new(HashMap::new()),
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

        // Backend factories
        self.register_backend_factory("candle", Arc::new(CandleBackendFactory));

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
        self.register_executor_factory("candle", Arc::new(CandleExecutorFactory));

        debug!(
            "Registered factories - backends: {:?}, tokenizers: {:?}, samplers: {:?}, schedulers: {:?}, kv_caches: {:?}, executors: {:?}",
            self.list_backends(),
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

    /// Register a backend factory
    pub fn register_backend_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn ComponentFactory<Arc<dyn ComputeBackend>>>,
    ) {
        let name = name.into();
        debug!("Registering backend factory: {}", name);
        self.backend_factories.write().insert(name, factory);
    }

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

    /// Get a backend factory by name
    pub fn get_backend_factory(
        &self,
        name: &str,
    ) -> Option<Arc<dyn ComponentFactory<Arc<dyn ComputeBackend>>>> {
        self.backend_factories.read().get(name).cloned()
    }

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

    /// List all registered backend names
    pub fn list_backends(&self) -> Vec<String> {
        self.backend_factories.read().keys().cloned().collect()
    }

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

    /// Create a backend by name
    pub async fn create_backend(
        &self,
        name: &str,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn ComputeBackend>> {
        let factory = self.get_backend_factory(name).ok_or_else(|| {
            FerrumError::backend(format!(
                "Backend '{}' not found. Available: {:?}",
                name,
                self.list_backends()
            ))
        })?;
        factory.create(config).await
    }

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
            .field("backends", &self.list_backends())
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

/// Candle backend factory
pub struct CandleBackendFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn ComputeBackend>> for CandleBackendFactory {
    async fn create(&self, config: &ComponentConfig) -> Result<Arc<dyn ComputeBackend>> {
        use ferrum_runtime::backends::CandleBackend;
        info!("Creating Candle backend for device: {:?}", config.device);
        let backend = CandleBackend::new(config.device.clone()).await?;
        Ok(Arc::new(backend))
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "candle".to_string(),
            version: "0.1.0".to_string(),
            description: "Candle compute backend for CPU/GPU inference".to_string(),
            supported_devices: vec![Device::CPU, Device::CUDA(0), Device::Metal],
            capabilities: vec!["fp16".to_string(), "fp32".to_string(), "bf16".to_string()],
        }
    }
}

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
            .or_else(|| std::env::var("FERRUM_MODEL_PATH").ok());

        if let Some(model_path) = tokenizer_path {
            let tokenizer_file = std::path::Path::new(&model_path).join("tokenizer.json");

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
            supported_devices: vec![Device::CPU, Device::CUDA(0), Device::Metal],
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
            supported_devices: vec![Device::CPU, Device::CUDA(0), Device::Metal],
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
            supported_devices: vec![Device::CPU, Device::CUDA(0), Device::Metal],
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

        // Create a stub compute backend for the executor
        let backend = ferrum_runtime::backends::CandleBackend::new(config.device.clone()).await?;

        let executor = ferrum_models::StubModelExecutor::new(
            config.engine_config.model.model_id.clone(),
            vocab_size,
            Arc::new(backend),
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
pub struct CandleExecutorFactory;

#[async_trait]
impl ComponentFactory<Arc<dyn ModelExecutor + Send + Sync>> for CandleExecutorFactory {
    async fn create(
        &self,
        config: &ComponentConfig,
    ) -> Result<Arc<dyn ModelExecutor + Send + Sync>> {
        use candle_core::{DType, Device as CandleDevice};

        // Try to load model from path
        let model_path = config
            .get_string_option("model_path")
            .or_else(|| std::env::var("FERRUM_MODEL_PATH").ok());

        let model_path = match model_path {
            Some(path) => path,
            None => {
                info!("No model path found, falling back to stub executor");
                return StubExecutorFactory.create(config).await;
            }
        };

        info!("Loading Candle model from: {}", model_path);

        // Load model definition
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
        fn parse_dtype(s: &str) -> Option<DType> {
            match s.trim().to_ascii_lowercase().as_str() {
                "fp16" | "f16" | "float16" => Some(DType::F16),
                "fp32" | "f32" | "float32" => Some(DType::F32),
                _ => None,
            }
        }

        let dtype = match &config.device {
            Device::CPU => DType::F32,
            Device::Metal => std::env::var("FERRUM_METAL_DTYPE")
                .ok()
                .and_then(|v| parse_dtype(&v))
                .or_else(|| {
                    std::env::var("FERRUM_DTYPE")
                        .ok()
                        .and_then(|v| parse_dtype(&v))
                })
                .unwrap_or(DType::F32),
            _ => std::env::var("FERRUM_DTYPE")
                .ok()
                .and_then(|v| parse_dtype(&v))
                .unwrap_or(DType::F16),
        };

        // Create model based on architecture
        info!("Building model...");
        match model_def.architecture {
            ferrum_models::Architecture::Llama => {
                // Use Metal LLaMA executor for Metal device
                #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
                if matches!(&config.device, Device::Metal) {
                    info!("Using Metal-accelerated LLaMA executor with custom RMS Norm kernel");
                    let executor = crate::metal::MetalLlamaExecutor::from_path(
                        &model_path,
                        &model_def,
                        candle_device.clone(),
                        dtype,
                    )
                    .await?;
                    return Ok(Arc::new(executor));
                }

                // Load weights (non-Metal path)
                info!("Loading model weights...");
                let loader = ferrum_models::SafeTensorsLoader::new(&model_path);
                let vb = loader.load_varbuilder(&candle_device, dtype)?;

                // Standard Candle executor for CPU/CUDA
                let llama_model = ferrum_models::LlamaModelWrapper::from_varbuilder(
                    vb,
                    &model_def,
                    candle_device.clone(),
                    dtype,
                )?;

                let model_info =
                    model_def.to_model_info(config.engine_config.model.model_id.to_string());
                let executor = ferrum_models::CandleModelExecutor::new(llama_model, model_info);

                Ok(Arc::new(executor))
            }
            ferrum_models::Architecture::Qwen2 => {
                // Use Metal Qwen2 executor for Metal device
                #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
                if matches!(&config.device, Device::Metal) {
                    info!("Using Metal-accelerated Qwen2 executor");
                    let executor = crate::metal::MetalQwen2Executor::from_path(
                        &model_path,
                        &model_def,
                        candle_device.clone(),
                        dtype,
                    )
                    .await?;
                    return Ok(Arc::new(executor));
                }

                // Load weights (non-Metal path)
                info!("Loading model weights...");
                let loader = ferrum_models::SafeTensorsLoader::new(&model_path);
                let vb = loader.load_varbuilder(&candle_device, dtype)?;

                // Standard Candle executor for CPU
                let qwen2_model = ferrum_models::Qwen2ModelWrapper::from_varbuilder(
                    vb,
                    &model_def,
                    candle_device.clone(),
                    dtype,
                )?;

                let model_info =
                    model_def.to_model_info(config.engine_config.model.model_id.to_string());
                let executor = ferrum_models::Qwen2ModelExecutor::new(qwen2_model, model_info);

                Ok(Arc::new(executor))
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
            _ => Err(FerrumError::model(format!(
                "Architecture {:?} not supported",
                model_def.architecture
            ))),
        }
    }

    fn metadata(&self) -> ComponentMetadata {
        ComponentMetadata {
            name: "candle".to_string(),
            version: "0.1.0".to_string(),
            description: "Candle-based model executor".to_string(),
            supported_devices: vec![Device::CPU, Device::CUDA(0), Device::Metal],
            capabilities: vec![
                "llama".to_string(),
                "qwen2".to_string(),
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

use std::sync::OnceLock;

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

    #[test]
    fn test_registry_creation() {
        let registry = ComponentRegistry::new();
        assert!(registry.list_backends().is_empty());
        assert!(registry.list_tokenizers().is_empty());
    }

    #[test]
    fn test_registry_with_defaults() {
        let registry = ComponentRegistry::with_defaults();
        assert!(registry.list_backends().contains(&"candle".to_string()));
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

    #[tokio::test]
    async fn test_factory_metadata() {
        let factory = CandleBackendFactory;
        let metadata = factory.metadata();

        assert_eq!(metadata.name, "candle");
        assert!(!metadata.supported_devices.is_empty());
    }
}
