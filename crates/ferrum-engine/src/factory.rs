//! Engine factory for creating configured engines

use crate::{EngineConfig, DefaultInferenceEngine};
use ferrum_interfaces::InferenceEngine;
use ferrum_types::{Result, Device, FerrumError, EngineConfig};
use std::sync::Arc;
use tracing::{debug, info};

/// Default engine factory
#[derive(Debug, Clone)]
pub struct DefaultEngineFactory;

impl DefaultEngineFactory {
    /// Create new factory
    pub fn new() -> Self {
        Self
    }

    /// Create inference engine from configuration
    pub async fn create_engine(
        &self,
        config: EngineConfig,
    ) -> Result<Box<dyn InferenceEngine + Send + Sync>> {
        info!("Creating inference engine", model_id = ?config.model.model_id);

        // Create scheduler
        let scheduler = self.create_scheduler(&config).await?;
        
        // Create tokenizer
        let tokenizer = self.create_tokenizer(&config).await?;
        
        // Create sampler
        let sampler = self.create_sampler(&config).await?;
        
        // Create KV cache manager
        let kv_cache = self.create_kv_cache_manager(&config).await?;
        
        // Create model executor
        let model_executor = self.create_model_executor(&config).await?;
        
        // Create engine
        let engine = DefaultInferenceEngine::new(
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
        );

        info!("Successfully created inference engine");
        Ok(Box::new(engine))
    }

    /// Create scheduler component
    async fn create_scheduler(
        &self,
        config: &EngineConfig,
    ) -> Result<Arc<dyn ferrum_scheduler::Scheduler + Send + Sync>> {
        debug!("Initializing scheduler", policy = ?config.scheduler.policy);
        // Create default FIFO scheduler
        let scheduler = ferrum_scheduler::implementations::FifoScheduler::new();
        Ok(Arc::new(scheduler))
    }

    /// Create tokenizer component
    async fn create_tokenizer(
        &self,
        config: &EngineConfig,
    ) -> Result<Arc<dyn ferrum_tokenizer::Tokenizer + Send + Sync>> {
        debug!("Creating tokenizer", tokenizer = ?config.model.tokenizer.tokenizer_type);
        // Create HuggingFace tokenizer
        let factory = ferrum_tokenizer::implementations::HuggingFaceTokenizerFactory;
        // For now, create a placeholder - in real implementation would load from model
        Err(FerrumError::not_implemented("Tokenizer creation not yet implemented"))
    }

    /// Create sampler component
    async fn create_sampler(
        &self,
        config: &EngineConfig,
    ) -> Result<Arc<dyn ferrum_sampler::Sampler + Send + Sync>> {
        debug!("Creating sampler", enable_custom = config.sampling.enable_custom_processors);
        // Create default greedy sampler
        let sampler = ferrum_sampler::implementations::GreedySampler::new();
        Ok(Arc::new(sampler))
    }

    /// Create KV cache manager
    async fn create_kv_cache_manager(
        &self,
        config: &EngineConfig,
    ) -> Result<Arc<dyn ferrum_kv::KvCacheManager + Send + Sync>> {
        debug!(
            "Creating KV cache manager",
            cache_type = ?config.kv_cache.cache_type,
            max_blocks = config.kv_cache.max_blocks
        );
        // Create KV cache manager
        let manager = ferrum_kv::managers::DefaultKvCacheManager::new(
            config.backend.device,
            config.kv_cache.clone(),
        )?;
        Ok(Arc::new(manager))
    }

    /// Create model executor
    async fn create_model_executor(
        &self,
        config: &EngineConfig,
    ) -> Result<Arc<dyn ferrum_interfaces::ModelExecutor + Send + Sync>> {
        // Create compute backend
        let backend = self.create_compute_backend(&config.backend.device).await?;
        
        // Create weight loader (placeholder)
        let weight_loader = Arc::new(ferrum_models::loader::SafeTensorsLoader::new());
        
        // Create model builder
        let builder_factory = ferrum_models::builder::DefaultModelBuilderFactory::new();
        let builder = builder_factory.get_builder(config.model.model_info.as_ref()
            .map(|info| info.architecture)
            .unwrap_or(ferrum_types::Architecture::Llama))?;
        
        // Build model executor
        let model_config = config
            .model
            .model_info
            .clone()
            .ok_or_else(|| FerrumError::configuration("Missing model_info in EngineConfig"))?;
        let executor = builder.build(&model_config, backend, weight_loader).await?;
        
        Ok(executor)
    }

    /// Create compute backend
    async fn create_compute_backend(
        &self,
        device: &Device,
    ) -> Result<Arc<dyn ferrum_interfaces::ComputeBackend + Send + Sync>> {
        let registry = ferrum_runtime::global_backend_registry();
        let backend = registry.create_default_backend(device).await?;
        Ok(Arc::new(backend))
    }
}

impl Default for DefaultEngineFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> EngineConfig {
        let mut config = EngineConfig::default();
        config.model.model_id = ferrum_types::ModelId::new("test-model");
        config
    }

    #[test]
    fn test_factory_creation() {
        let factory = DefaultEngineFactory::new();
        // Just verify it can be created
        drop(factory);
    }

    #[tokio::test]
    async fn test_scheduler_creation() {
        let factory = DefaultEngineFactory::new();
        let config = create_test_config();
        
        let scheduler = factory.create_scheduler(&config).await.unwrap();
        // Verify scheduler was created
        drop(scheduler);
    }
}
