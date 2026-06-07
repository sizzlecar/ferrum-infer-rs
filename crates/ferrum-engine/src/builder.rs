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
use ferrum_interfaces::engine::LlmInferenceEngine;
use ferrum_interfaces::{
    KvCacheManager, ModelExecutor, Sampler, SchedulerInterface as Scheduler, TensorFactory,
    Tokenizer,
};
use ferrum_types::{EngineConfig, FerrumError, Result, SchedulingPolicy};
use std::sync::{Arc, OnceLock};
use tracing::{debug, info};

#[derive(Debug, Clone, PartialEq, Eq)]
struct EngineBuilderRuntimeEnv {
    model_path: Option<String>,
    spec_draft: Option<String>,
    spec_n: usize,
}

impl EngineBuilderRuntimeEnv {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let mut model_path = None;
        let mut spec_draft = None;
        let mut spec_n = None;

        for (key, value) in vars {
            let value = value.into();
            match key.as_ref() {
                "FERRUM_MODEL_PATH" => model_path = Some(value),
                "FERRUM_SPEC_DRAFT" if !value.is_empty() => spec_draft = Some(value),
                "FERRUM_SPEC_DRAFT" => spec_draft = None,
                "FERRUM_SPEC_N" => spec_n = value.parse::<usize>().ok(),
                _ => {}
            }
        }

        Self {
            model_path,
            spec_draft,
            spec_n: spec_n.unwrap_or(4),
        }
    }

    fn has_model_path(&self) -> bool {
        self.model_path.is_some()
    }
}

fn engine_builder_runtime_env() -> &'static EngineBuilderRuntimeEnv {
    static CONFIG: OnceLock<EngineBuilderRuntimeEnv> = OnceLock::new();
    CONFIG.get_or_init(EngineBuilderRuntimeEnv::from_env)
}

/// Engine builder for creating inference engines with registry-based components
pub struct EngineBuilder {
    /// Component registry to use
    registry: Arc<ComponentRegistry>,
    /// Engine configuration
    config: EngineConfig,
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
            tokenizer_name: None,
            sampler_name: None,
            scheduler_name: None,
            kv_cache_name: None,
            executor_name: None,
            custom_tokenizer: None,
            custom_sampler: None,
            custom_scheduler: None,
            custom_kv_cache: None,
            custom_executor: None,
        }
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

    /// Determine which tokenizer to use based on config and overrides
    fn resolve_tokenizer_name(&self) -> String {
        if let Some(ref name) = self.tokenizer_name {
            return name.clone();
        }

        // If model path is set, try huggingface first
        if self.has_typed_model_path() || engine_builder_runtime_env().has_model_path() {
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
            SchedulingPolicy::FCFS => "fifo".to_string(),
            SchedulingPolicy::Priority => "priority".to_string(),
            SchedulingPolicy::FairShare => "fifo".to_string(), // Fallback
            SchedulingPolicy::SJF => "fifo".to_string(),       // Fallback
            SchedulingPolicy::RoundRobin => "fifo".to_string(), // Fallback
            SchedulingPolicy::ContinuousBatch => "continuous".to_string(),
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
        if self.has_typed_model_path() || engine_builder_runtime_env().has_model_path() {
            return "llm".to_string();
        }

        "stub".to_string()
    }

    fn has_typed_model_path(&self) -> bool {
        self.config
            .backend
            .backend_options
            .get("model_path")
            .and_then(|value| value.as_str())
            .is_some()
    }

    /// Build the inference engine
    pub async fn build(self) -> Result<Box<dyn LlmInferenceEngine + Send + Sync>> {
        info!(
            "Building inference engine for model: {}",
            self.config.model.model_id
        );

        // Pre-compute all component names before consuming self
        let tokenizer_name = self.resolve_tokenizer_name();
        let sampler_name = self.resolve_sampler_name();
        let scheduler_name = self.resolve_scheduler_name();
        let kv_cache_name = self.resolve_kv_cache_name();
        let executor_name = self.resolve_executor_name();

        let component_config = ComponentConfig::from_engine_config(&self.config);
        reject_unsupported_layer_split(&component_config)?;
        let typed_model_path = component_config.get_string_option("model_path");
        let has_model_path =
            typed_model_path.is_some() || engine_builder_runtime_env().has_model_path();
        let runtime_env = engine_builder_runtime_env();
        let registry = self.registry.clone();
        let config = self.config;

        // Extract custom components. Phase 3e+ deleted the legacy
        // `ComputeBackend` trait, so there's no "backend" component to
        // build here — real GPU dispatch goes through `Backend<B>` in
        // `ferrum-kernels`. The stub executor now wires
        // `CandleTensorFactory` directly from the registry.
        let custom_tokenizer = self.custom_tokenizer;
        let custom_sampler = self.custom_sampler;
        let custom_scheduler = self.custom_scheduler;
        let custom_kv_cache = self.custom_kv_cache;
        let custom_executor = self.custom_executor;

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
                    if has_model_path {
                        return Err(FerrumError::config(format!(
                            "Failed to create tokenizer '{}' in model mode: {}",
                            tokenizer_name, e
                        )));
                    }

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
                    if has_model_path {
                        return Err(FerrumError::config(format!(
                            "Failed to create executor '{}' in model mode: {}",
                            executor_name, err
                        )));
                    }

                    tracing::warn!(
                        "Failed to create executor '{}': {}, falling back to stub",
                        executor_name,
                        err
                    );
                    registry.create_executor("stub", &component_config).await?
                }
            }
        };

        // 7. Create the engine — always ContinuousBatchEngine.
        info!("All components created, building ContinuousBatchEngine");

        // The registry-resolved scheduler from steps 4 was used by the
        // legacy DefaultInferenceEngine path that Phase 5b deleted; the
        // ContinuousBatchEngine needs a concrete ContinuousBatchScheduler.
        let _ = scheduler;
        let cb_scheduler = Arc::new(
            ferrum_scheduler::implementations::ContinuousBatchScheduler::new(
                config.scheduler.clone(),
            ),
        );

        // Create TensorFactory for the configured device
        let tensor_factory: Arc<dyn TensorFactory> = Arc::new(
            crate::tensor_factory::candle::CandleTensorFactory::new(config.backend.device.clone()),
        );

        // Opt-in speculative decoding: provide an absolute HF snapshot path
        // for a second smaller model. The draft must use the same tokenizer
        // + vocab as the target (same family e.g. Qwen3). Backend options are
        // the typed startup path; the legacy speculative env names remain
        // compatibility aliases.
        let spec_draft = component_config
            .get_string_option("spec_draft")
            .or_else(|| runtime_env.spec_draft.clone());
        let spec_n = component_config
            .get_option::<usize>("spec_n")
            .unwrap_or(runtime_env.spec_n);
        let (draft_executor, spec_config) = match spec_draft.as_ref() {
            Some(draft_path) => {
                info!("Speculative decoding: loading draft model from {draft_path}");
                let mut draft_cfg = component_config.clone();
                draft_cfg.component_options.insert(
                    "model_path".to_string(),
                    serde_json::Value::String(draft_path.to_string()),
                );
                match registry.create_executor(&executor_name, &draft_cfg).await {
                    Ok(draft) => (
                        Some(draft),
                        Some(crate::speculative::SpeculativeDecodingConfig {
                            num_speculative_tokens: spec_n,
                            temperature: 1.0,
                        }),
                    ),
                    Err(e) => {
                        tracing::warn!("Speculative decoding disabled — draft load failed: {e}");
                        (None, None)
                    }
                }
            }
            _ => (None, None),
        };

        let engine = crate::ContinuousBatchEngine::new_with_speculation(
            config,
            cb_scheduler,
            tokenizer,
            sampler,
            kv_cache,
            executor,
            tensor_factory,
            draft_executor,
            spec_config,
        );
        Ok(Box::new(engine))
    }
}

fn reject_unsupported_layer_split(component_config: &ComponentConfig) -> Result<()> {
    if component_config
        .get_string_option("selected_distributed_strategy")
        .as_deref()
        != Some("layer_split")
    {
        return Ok(());
    }
    let requested = component_config
        .get_option::<Vec<usize>>("requested_gpu_devices")
        .unwrap_or_default();
    let selected = component_config
        .get_option::<Vec<usize>>("selected_gpu_devices")
        .unwrap_or_default();
    let plan = component_config
        .get_string_option("selected_layer_split_plan")
        .unwrap_or_else(|| "missing".to_string());
    Err(FerrumError::unsupported(format!(
        "CUDA layer_split execution is not implemented yet; requested_gpu_devices={requested:?} selected_gpu_devices={selected:?} selected_layer_split_plan={plan}. Refusing to fall back to a single GPU."
    )))
}

/// Create an engine with the default configuration and registry
pub async fn create_engine(
    config: EngineConfig,
) -> Result<Box<dyn LlmInferenceEngine + Send + Sync>> {
    EngineBuilder::new(config).build().await
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

        assert!(builder.tokenizer_name.is_none());
    }

    #[test]
    fn test_builder_with_overrides() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config)
            .with_tokenizer("custom_tokenizer")
            .with_sampler("greedy")
            .with_scheduler("priority")
            .with_kv_cache("paged")
            .with_executor("custom_executor");

        assert_eq!(builder.tokenizer_name, Some("custom_tokenizer".to_string()));
        assert_eq!(builder.sampler_name, Some("greedy".to_string()));
        assert_eq!(builder.scheduler_name, Some("priority".to_string()));
        assert_eq!(builder.kv_cache_name, Some("paged".to_string()));
        assert_eq!(builder.executor_name, Some("custom_executor".to_string()));
    }

    #[test]
    fn test_builder_typed_model_path_selects_model_components() {
        let mut config = EngineConfig::default();
        config.backend.backend_options.insert(
            "model_path".to_string(),
            serde_json::Value::String("/models/target".to_string()),
        );
        let builder = EngineBuilder::new(config);

        assert!(builder.has_typed_model_path());
        assert_eq!(builder.resolve_tokenizer_name(), "huggingface");
        assert_eq!(builder.resolve_executor_name(), "llm");
    }

    #[test]
    fn test_builder_typed_spec_options_parse_from_component_config() {
        let mut config = EngineConfig::default();
        config.backend.backend_options.insert(
            "model_path".to_string(),
            serde_json::Value::String("/models/target".to_string()),
        );
        config.backend.backend_options.insert(
            "spec_draft".to_string(),
            serde_json::Value::String("/models/draft".to_string()),
        );
        config.backend.backend_options.insert(
            "spec_n".to_string(),
            serde_json::Value::Number(serde_json::Number::from(6)),
        );
        let component_config = ComponentConfig::from_engine_config(&config);

        assert_eq!(
            component_config.get_string_option("spec_draft").as_deref(),
            Some("/models/draft")
        );
        assert_eq!(component_config.get_option::<usize>("spec_n"), Some(6));
    }

    #[tokio::test]
    async fn test_builder_rejects_layer_split_until_executor_supports_it() {
        let mut config = EngineConfig::default();
        config.backend.backend_options.insert(
            "model_path".to_string(),
            serde_json::Value::String("/models/target".to_string()),
        );
        config.backend.backend_options.insert(
            "selected_distributed_strategy".to_string(),
            serde_json::Value::String("layer_split".to_string()),
        );
        config.backend.backend_options.insert(
            "requested_gpu_devices".to_string(),
            serde_json::json!([0, 1]),
        );
        config.backend.backend_options.insert(
            "selected_gpu_devices".to_string(),
            serde_json::json!([0, 1]),
        );
        config.backend.backend_options.insert(
            "selected_layer_split_plan".to_string(),
            serde_json::Value::String(
                "stage0:cuda:0:layers=auto;stage1:cuda:1:layers=auto".to_string(),
            ),
        );

        let err = match EngineBuilder::new(config).build().await {
            Ok(_) => panic!("layer_split build unexpectedly succeeded"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("layer_split execution is not implemented yet"));
        assert!(err.to_string().contains("selected_gpu_devices=[0, 1]"));
    }

    #[test]
    fn test_engine_builder_runtime_env_parses_model_and_spec() {
        let env = EngineBuilderRuntimeEnv::from_env_vars([
            ("FERRUM_MODEL_PATH", "/models/target"),
            ("FERRUM_SPEC_DRAFT", "/models/draft"),
            ("FERRUM_SPEC_N", "8"),
        ]);

        assert_eq!(env.model_path.as_deref(), Some("/models/target"));
        assert_eq!(env.spec_draft.as_deref(), Some("/models/draft"));
        assert_eq!(env.spec_n, 8);
        assert!(env.has_model_path());
    }

    #[test]
    fn test_engine_builder_runtime_env_defaults_and_ignores_empty_draft() {
        let env = EngineBuilderRuntimeEnv::from_env_vars([
            ("FERRUM_MODEL_PATH", ""),
            ("FERRUM_SPEC_DRAFT", ""),
            ("FERRUM_SPEC_N", "not-a-number"),
        ]);

        assert_eq!(env.model_path.as_deref(), Some(""));
        assert_eq!(env.spec_draft, None);
        assert_eq!(env.spec_n, 4);
        assert!(env.has_model_path());
    }

    #[test]
    fn test_resolve_defaults() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config);

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
