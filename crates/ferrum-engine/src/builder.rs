//! Engine builder with registry-based component creation
//!
//! This module provides a fluent builder API for creating inference engines
//! using the component registry pattern. The builder supports:
//!
//! - Configuration-driven component selection
//! - Custom component overrides
//! - Automatic fallback to defaults
//! - Validation before engine creation

use crate::registry::{ComponentConfig, ComponentRegistry};
use crate::DefaultInferenceEngine;
use ferrum_interfaces::{
    ComputeBackend, InferenceEngine, KvCacheManager, ModelExecutor, Sampler,
    SchedulerInterface as Scheduler, Tokenizer,
};
use ferrum_types::{EngineConfig, Result};
use std::sync::Arc;
use tracing::{debug, info};

/// Engine builder for creating inference engines with registry-based components
pub struct EngineBuilder {
    /// Component registry to use
    registry: Arc<ComponentRegistry>,
    /// Engine configuration
    config: EngineConfig,
    /// Override: custom backend name
    backend_name: Option<String>,
    /// Override: custom tokenizer name
    tokenizer_name: Option<String>,
    /// Override: custom sampler name
    sampler_name: Option<String>,
    /// Override: custom scheduler name
    scheduler_name: Option<String>,
    /// Override: custom KV cache name
    kv_cache_name: Option<String>,
    /// Override: custom executor name
    executor_name: Option<String>,
    /// Pre-created backend (skip factory)
    custom_backend: Option<Arc<dyn ComputeBackend>>,
    /// Pre-created tokenizer (skip factory)
    custom_tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
    /// Pre-created sampler (skip factory)
    custom_sampler: Option<Arc<dyn Sampler + Send + Sync>>,
    /// Pre-created scheduler (skip factory)
    custom_scheduler: Option<Arc<dyn Scheduler + Send + Sync>>,
    /// Pre-created KV cache (skip factory)
    custom_kv_cache: Option<Arc<dyn KvCacheManager + Send + Sync>>,
    /// Pre-created executor (skip factory)
    custom_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
}

impl EngineBuilder {
    /// Create a new engine builder with default registry
    pub fn new(config: EngineConfig) -> Self {
        Self::with_registry(config, crate::registry::global_registry())
    }

    /// Create a new engine builder with a custom registry
    pub fn with_registry(config: EngineConfig, registry: Arc<ComponentRegistry>) -> Self {
        Self {
            registry,
            config,
            backend_name: None,
            tokenizer_name: None,
            sampler_name: None,
            scheduler_name: None,
            kv_cache_name: None,
            executor_name: None,
            custom_backend: None,
            custom_tokenizer: None,
            custom_sampler: None,
            custom_scheduler: None,
            custom_kv_cache: None,
            custom_executor: None,
        }
    }

    /// Set the backend to use by name
    pub fn with_backend(mut self, name: impl Into<String>) -> Self {
        self.backend_name = Some(name.into());
        self
    }

    /// Set a pre-created backend
    pub fn with_custom_backend(mut self, backend: Arc<dyn ComputeBackend>) -> Self {
        self.custom_backend = Some(backend);
        self
    }

    /// Set the tokenizer to use by name
    pub fn with_tokenizer(mut self, name: impl Into<String>) -> Self {
        self.tokenizer_name = Some(name.into());
        self
    }

    /// Set a pre-created tokenizer
    pub fn with_custom_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer + Send + Sync>) -> Self {
        self.custom_tokenizer = Some(tokenizer);
        self
    }

    /// Set the sampler to use by name
    pub fn with_sampler(mut self, name: impl Into<String>) -> Self {
        self.sampler_name = Some(name.into());
        self
    }

    /// Set a pre-created sampler
    pub fn with_custom_sampler(mut self, sampler: Arc<dyn Sampler + Send + Sync>) -> Self {
        self.custom_sampler = Some(sampler);
        self
    }

    /// Set the scheduler to use by name
    pub fn with_scheduler(mut self, name: impl Into<String>) -> Self {
        self.scheduler_name = Some(name.into());
        self
    }

    /// Set a pre-created scheduler
    pub fn with_custom_scheduler(mut self, scheduler: Arc<dyn Scheduler + Send + Sync>) -> Self {
        self.custom_scheduler = Some(scheduler);
        self
    }

    /// Set the KV cache to use by name
    pub fn with_kv_cache(mut self, name: impl Into<String>) -> Self {
        self.kv_cache_name = Some(name.into());
        self
    }

    /// Set a pre-created KV cache manager
    pub fn with_custom_kv_cache(mut self, kv_cache: Arc<dyn KvCacheManager + Send + Sync>) -> Self {
        self.custom_kv_cache = Some(kv_cache);
        self
    }

    /// Set the executor to use by name
    pub fn with_executor(mut self, name: impl Into<String>) -> Self {
        self.executor_name = Some(name.into());
        self
    }

    /// Set a pre-created model executor
    pub fn with_custom_executor(mut self, executor: Arc<dyn ModelExecutor + Send + Sync>) -> Self {
        self.custom_executor = Some(executor);
        self
    }

    /// Determine which backend to use based on config and overrides
    fn resolve_backend_name(&self) -> String {
        if let Some(ref name) = self.backend_name {
            return name.clone();
        }

        // Derive from config
        match self.config.backend.backend_type {
            ferrum_types::BackendType::Candle => "candle".to_string(),
            ferrum_types::BackendType::OnnxRuntime => "onnx".to_string(),
            ferrum_types::BackendType::TensorRT => "tensorrt".to_string(),
            ferrum_types::BackendType::Custom => self
                .config
                .backend
                .backend_options
                .get("backend_name")
                .and_then(|v| v.as_str())
                .unwrap_or("candle")
                .to_string(),
        }
    }

    /// Determine which tokenizer to use based on config and overrides
    fn resolve_tokenizer_name(&self) -> String {
        if let Some(ref name) = self.tokenizer_name {
            return name.clone();
        }

        // If model path is set, try huggingface first
        if std::env::var("FERRUM_MODEL_PATH").is_ok() {
            return "huggingface".to_string();
        }

        "stub".to_string()
    }

    /// Determine which sampler to use based on config and overrides
    fn resolve_sampler_name(&self) -> String {
        if let Some(ref name) = self.sampler_name {
            return name.clone();
        }

        // Could derive from sampling params in the future
        "multinomial".to_string()
    }

    /// Determine which scheduler to use based on config and overrides
    fn resolve_scheduler_name(&self) -> String {
        if let Some(ref name) = self.scheduler_name {
            return name.clone();
        }

        match self.config.scheduler.policy {
            ferrum_types::SchedulingPolicy::FCFS => "fifo".to_string(),
            ferrum_types::SchedulingPolicy::Priority => "priority".to_string(),
            ferrum_types::SchedulingPolicy::FairShare => "fifo".to_string(), // Fallback
            ferrum_types::SchedulingPolicy::SJF => "fifo".to_string(),       // Fallback
            ferrum_types::SchedulingPolicy::RoundRobin => "fifo".to_string(), // Fallback
        }
    }

    /// Determine which KV cache to use based on config and overrides
    fn resolve_kv_cache_name(&self) -> String {
        if let Some(ref name) = self.kv_cache_name {
            return name.clone();
        }

        match self.config.kv_cache.cache_type {
            ferrum_types::KvCacheType::Contiguous => "default".to_string(),
            ferrum_types::KvCacheType::Paged => "paged".to_string(),
            ferrum_types::KvCacheType::Tree => "default".to_string(), // Fallback
        }
    }

    /// Determine which executor to use based on config and overrides
    fn resolve_executor_name(&self) -> String {
        if let Some(ref name) = self.executor_name {
            return name.clone();
        }

        // If model path is set, try candle executor
        if std::env::var("FERRUM_MODEL_PATH").is_ok() {
            return "candle".to_string();
        }

        "stub".to_string()
    }

    /// Build the inference engine
    pub async fn build(self) -> Result<Box<dyn InferenceEngine + Send + Sync>> {
        info!(
            "Building inference engine for model: {}",
            self.config.model.model_id
        );

        // Pre-compute all component names before consuming self
        let backend_name = self.resolve_backend_name();
        let tokenizer_name = self.resolve_tokenizer_name();
        let sampler_name = self.resolve_sampler_name();
        let scheduler_name = self.resolve_scheduler_name();
        let kv_cache_name = self.resolve_kv_cache_name();
        let executor_name = self.resolve_executor_name();

        let component_config = ComponentConfig::from_engine_config(&self.config);
        let registry = self.registry.clone();
        let config = self.config;

        // Extract custom components
        let custom_backend = self.custom_backend;
        let custom_tokenizer = self.custom_tokenizer;
        let custom_sampler = self.custom_sampler;
        let custom_scheduler = self.custom_scheduler;
        let custom_kv_cache = self.custom_kv_cache;
        let custom_executor = self.custom_executor;

        // 1. Create or use provided backend (not used directly but may be needed for executor)
        let _backend = if let Some(backend) = custom_backend {
            debug!("Using custom backend");
            Some(backend)
        } else {
            debug!("Creating backend: {}", backend_name);
            Some(
                registry
                    .create_backend(&backend_name, &component_config)
                    .await?,
            )
        };

        // 2. Create or use provided tokenizer
        let tokenizer = if let Some(tokenizer) = custom_tokenizer {
            debug!("Using custom tokenizer");
            tokenizer
        } else {
            debug!("Creating tokenizer: {}", tokenizer_name);

            // Try primary tokenizer, fallback to stub
            match registry
                .create_tokenizer(&tokenizer_name, &component_config)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        "Failed to create tokenizer '{}': {}, falling back to stub",
                        tokenizer_name,
                        e
                    );
                    registry.create_tokenizer("stub", &component_config).await?
                }
            }
        };

        // 3. Create or use provided sampler
        let sampler = if let Some(sampler) = custom_sampler {
            debug!("Using custom sampler");
            sampler
        } else {
            debug!("Creating sampler: {}", sampler_name);
            registry
                .create_sampler(&sampler_name, &component_config)
                .await?
        };

        // 4. Create or use provided scheduler
        let scheduler = if let Some(scheduler) = custom_scheduler {
            debug!("Using custom scheduler");
            scheduler
        } else {
            debug!("Creating scheduler: {}", scheduler_name);
            registry
                .create_scheduler(&scheduler_name, &component_config)
                .await?
        };

        // 5. Create or use provided KV cache
        let kv_cache = if let Some(kv_cache) = custom_kv_cache {
            debug!("Using custom KV cache");
            kv_cache
        } else {
            debug!("Creating KV cache: {}", kv_cache_name);
            registry
                .create_kv_cache(&kv_cache_name, &component_config)
                .await?
        };

        // 6. Create or use provided executor
        let executor = if let Some(executor) = custom_executor {
            debug!("Using custom executor");
            executor
        } else {
            debug!("Creating executor: {}", executor_name);

            // Try primary executor, fallback to stub
            match registry
                .create_executor(&executor_name, &component_config)
                .await
            {
                Ok(e) => e,
                Err(err) => {
                    tracing::warn!(
                        "Failed to create executor '{}': {}, falling back to stub",
                        executor_name,
                        err
                    );
                    registry.create_executor("stub", &component_config).await?
                }
            }
        };

        // 7. Create the engine
        info!("All components created, building engine");
        let engine =
            DefaultInferenceEngine::new(config, scheduler, tokenizer, sampler, kv_cache, executor);

        Ok(Box::new(engine))
    }
}

/// Create an engine with the default configuration and registry
pub async fn create_engine(config: EngineConfig) -> Result<Box<dyn InferenceEngine + Send + Sync>> {
    EngineBuilder::new(config).build().await
}

/// Create an engine with a custom registry
pub async fn create_engine_with_registry(
    config: EngineConfig,
    registry: Arc<ComponentRegistry>,
) -> Result<Box<dyn InferenceEngine + Send + Sync>> {
    EngineBuilder::with_registry(config, registry).build().await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config);

        assert!(builder.backend_name.is_none());
        assert!(builder.tokenizer_name.is_none());
    }

    #[test]
    fn test_builder_with_overrides() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config)
            .with_backend("custom_backend")
            .with_tokenizer("custom_tokenizer")
            .with_sampler("greedy")
            .with_scheduler("priority")
            .with_kv_cache("paged")
            .with_executor("custom_executor");

        assert_eq!(builder.backend_name, Some("custom_backend".to_string()));
        assert_eq!(builder.tokenizer_name, Some("custom_tokenizer".to_string()));
        assert_eq!(builder.sampler_name, Some("greedy".to_string()));
        assert_eq!(builder.scheduler_name, Some("priority".to_string()));
        assert_eq!(builder.kv_cache_name, Some("paged".to_string()));
        assert_eq!(builder.executor_name, Some("custom_executor".to_string()));
    }

    #[test]
    fn test_resolve_defaults() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config);

        assert_eq!(builder.resolve_backend_name(), "candle");
        assert_eq!(builder.resolve_sampler_name(), "multinomial");
        // Default SchedulerConfig uses Priority policy
        assert_eq!(builder.resolve_scheduler_name(), "priority");
        assert_eq!(builder.resolve_kv_cache_name(), "default");
    }

    #[tokio::test]
    async fn test_build_with_defaults() {
        let config = EngineConfig::default();
        let result = EngineBuilder::new(config).build().await;

        // Should succeed with stub components
        assert!(result.is_ok());
    }
}
