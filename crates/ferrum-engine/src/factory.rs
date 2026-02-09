//! Engine factory implementations
//!
//! This module provides factory implementations for creating inference engines.
//! The `DefaultEngineFactory` now uses the registry pattern internally while
//! maintaining backward compatibility with the original API.

use crate::builder::EngineBuilder;
use crate::registry::ComponentRegistry;
use crate::InferenceEngineInterface;
use ferrum_types::{EngineConfig, Result};
use std::sync::Arc;
use tracing::info;

/// Default engine factory using registry-based component creation
///
/// This factory uses the component registry to create all engine components,
/// allowing for dynamic component selection and easy testing.
#[derive(Debug, Clone)]
pub struct DefaultEngineFactory {
    /// Component registry
    registry: Arc<ComponentRegistry>,
}

impl Default for DefaultEngineFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultEngineFactory {
    /// Create a new factory with the default global registry
    pub fn new() -> Self {
        Self {
            registry: crate::registry::global_registry(),
        }
    }

    /// Create a new factory with a custom registry
    pub fn with_registry(registry: Arc<ComponentRegistry>) -> Self {
        Self { registry }
    }

    /// Create inference engine with all components from registry
    pub async fn create_engine(
        &self,
        config: EngineConfig,
    ) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
        info!(
            "Creating inference engine with config: {:?}",
            config.model.model_id
        );

        EngineBuilder::with_registry(config, self.registry.clone())
            .build()
            .await
    }

    /// Create engine with specific component overrides
    pub async fn create_engine_with_components(
        &self,
        config: EngineConfig,
        backend_name: Option<&str>,
        tokenizer_name: Option<&str>,
        sampler_name: Option<&str>,
        scheduler_name: Option<&str>,
        kv_cache_name: Option<&str>,
        executor_name: Option<&str>,
    ) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
        let mut builder = EngineBuilder::with_registry(config, self.registry.clone());

        if let Some(name) = backend_name {
            builder = builder.with_backend(name);
        }
        if let Some(name) = tokenizer_name {
            builder = builder.with_tokenizer(name);
        }
        if let Some(name) = sampler_name {
            builder = builder.with_sampler(name);
        }
        if let Some(name) = scheduler_name {
            builder = builder.with_scheduler(name);
        }
        if let Some(name) = kv_cache_name {
            builder = builder.with_kv_cache(name);
        }
        if let Some(name) = executor_name {
            builder = builder.with_executor(name);
        }

        builder.build().await
    }

    /// Get the registry used by this factory
    pub fn registry(&self) -> Arc<ComponentRegistry> {
        self.registry.clone()
    }

    /// List available backends
    pub fn list_backends(&self) -> Vec<String> {
        self.registry.list_backends()
    }

    /// List available tokenizers
    pub fn list_tokenizers(&self) -> Vec<String> {
        self.registry.list_tokenizers()
    }

    /// List available samplers
    pub fn list_samplers(&self) -> Vec<String> {
        self.registry.list_samplers()
    }

    /// List available schedulers
    pub fn list_schedulers(&self) -> Vec<String> {
        self.registry.list_schedulers()
    }

    /// List available KV caches
    pub fn list_kv_caches(&self) -> Vec<String> {
        self.registry.list_kv_caches()
    }

    /// List available executors
    pub fn list_executors(&self) -> Vec<String> {
        self.registry.list_executors()
    }
}

/// Registry-based engine factory (explicit name for clarity)
///
/// This is an alias for `DefaultEngineFactory` that makes the
/// registry-based nature explicit.
pub type RegistryBasedEngineFactory = DefaultEngineFactory;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creation() {
        let factory = DefaultEngineFactory::new();
        assert!(!factory.list_backends().is_empty());
    }

    #[test]
    fn test_factory_with_custom_registry() {
        let registry = Arc::new(ComponentRegistry::with_defaults());
        let factory = DefaultEngineFactory::with_registry(registry);
        assert!(factory.list_backends().contains(&"candle".to_string()));
    }

    #[test]
    fn test_list_components() {
        let factory = DefaultEngineFactory::new();

        let backends = factory.list_backends();
        let tokenizers = factory.list_tokenizers();
        let samplers = factory.list_samplers();
        let schedulers = factory.list_schedulers();
        let kv_caches = factory.list_kv_caches();
        let executors = factory.list_executors();

        assert!(backends.contains(&"candle".to_string()));
        assert!(tokenizers.contains(&"stub".to_string()));
        assert!(samplers.contains(&"multinomial".to_string()));
        assert!(schedulers.contains(&"fifo".to_string()));
        assert!(kv_caches.contains(&"default".to_string()));
        assert!(executors.contains(&"stub".to_string()));
    }

    #[tokio::test]
    async fn test_create_engine() {
        let factory = DefaultEngineFactory::new();
        let config = EngineConfig::default();

        let result = factory.create_engine(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_engine_with_components() {
        let factory = DefaultEngineFactory::new();
        let config = EngineConfig::default();

        let result = factory
            .create_engine_with_components(
                config,
                Some("candle"),
                Some("stub"),
                Some("greedy"),
                Some("priority"),
                Some("default"),
                Some("stub"),
            )
            .await;

        assert!(result.is_ok());
    }
}
