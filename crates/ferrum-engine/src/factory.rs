//! Engine factory - MVP implementation

use crate::{DefaultInferenceEngine, InferenceEngineInterface, Sampler, Scheduler, Tokenizer};
use ferrum_types::{Device, EngineConfig, FerrumError, Result, TokenId};
use std::sync::Arc;
use tracing::info;

/// Default engine factory
#[derive(Debug, Default, Clone)]
pub struct DefaultEngineFactory;

impl DefaultEngineFactory {
    pub fn new() -> Self {
        Self
    }

    /// Create inference engine with all components
    pub async fn create_engine(
        &self,
        config: EngineConfig,
    ) -> Result<Box<dyn InferenceEngineInterface + Send + Sync>> {
        info!("Creating inference engine with config: {:?}", config.model.model_id);

        // 1. Create compute backend
        let device = config.backend.device.clone();
        let compute_backend = self.create_compute_backend(device.clone()).await?;

        // 2. Create tokenizer
        let tokenizer = self.create_tokenizer(&config).await?;

        // 3. Create sampler
        let sampler = self.create_sampler(&config.model.model_id.to_string()).await?;

        // 4. Create scheduler
        let scheduler = self.create_scheduler(&config).await?;

        // 5. Create KV cache manager
        let kv_cache = self.create_kv_cache(&config, device.clone()).await?;

        // 6. Create model executor
        let model_executor = self.create_model_executor(&config, compute_backend.clone()).await?;

        // 7. Build engine
        let engine = DefaultInferenceEngine::new(
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
        );

        Ok(Box::new(engine))
    }

    async fn create_compute_backend(
        &self,
        device: Device,
    ) -> Result<Arc<dyn ferrum_interfaces::ComputeBackend>> {
        info!("Creating Candle backend for device: {:?}", device);
        let backend = ferrum_runtime::backends::CandleBackend::new(device).await?;
        Ok(Arc::new(backend))
    }

    async fn create_tokenizer(
        &self,
        _config: &EngineConfig,
    ) -> Result<Arc<dyn ferrum_interfaces::Tokenizer + Send + Sync>> {
        info!("Creating stub tokenizer");
        // MVP: Create a simple stub tokenizer
        let tokenizer = StubTokenizer::new(32000); // Default vocab size
        Ok(Arc::new(tokenizer))
    }

    async fn create_sampler(
        &self,
        _model_id: &str,
    ) -> Result<Arc<dyn ferrum_interfaces::Sampler + Send + Sync>> {
        info!("Creating multinomial sampler");
        Ok(Arc::new(ferrum_interfaces::sampler::MultinomialSampler))
    }

    async fn create_scheduler(
        &self,
        config: &EngineConfig,
    ) -> Result<Arc<dyn Scheduler + Send + Sync>> {
        info!("Creating FIFO scheduler");
        let scheduler_config = config.scheduler.clone();
        let scheduler = ferrum_scheduler::implementations::FifoScheduler::new(scheduler_config);
        Ok(Arc::new(scheduler))
    }

    async fn create_kv_cache(
        &self,
        _config: &EngineConfig,
        device: Device,
    ) -> Result<Arc<dyn ferrum_interfaces::KvCacheManager + Send + Sync>> {
        info!("Creating KV cache manager for device: {:?}", device);
        let manager = ferrum_kv::managers::DefaultKvCacheManager::new(device, 16, 1024)?;
        Ok(Arc::new(manager))
    }

    async fn create_model_executor(
        &self,
        config: &EngineConfig,
        compute_backend: Arc<dyn ferrum_interfaces::ComputeBackend>,
    ) -> Result<Arc<dyn ferrum_interfaces::ModelExecutor + Send + Sync>> {
        info!("Creating model executor");

        let vocab_size = config
            .model
            .model_info
            .as_ref()
            .map(|info| info.vocab_size)
            .unwrap_or(32000);

        let executor = ferrum_models::StubModelExecutor::new(
            config.model.model_id.clone(),
            vocab_size,
            compute_backend,
        );

        Ok(Arc::new(executor))
    }
}

// ============================================================================
// Stub Tokenizer
// ============================================================================

struct StubTokenizer {
    vocab_size: usize,
    info: ferrum_interfaces::TokenizerInfo,
}

impl StubTokenizer {
    fn new(vocab_size: usize) -> Self {
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

impl ferrum_interfaces::Tokenizer for StubTokenizer {
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

    fn decode_incremental(&self, _prev: &[ferrum_types::TokenId], next: ferrum_types::TokenId) -> Result<String> {
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
