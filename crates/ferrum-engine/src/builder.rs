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
    KvCacheManager, ModelExecutor, RecurrentStateManager, Sampler, SchedulerInterface as Scheduler,
    TensorFactory, Tokenizer,
};
use ferrum_models::vnext::{PreparedProductionModel, ProductionModelSourceBundle};
use ferrum_types::{EngineConfig, FerrumError, Result};
use std::sync::Arc;
use tracing::{debug, info};

// Engine-build composition knobs (FERRUM_MODEL_PATH / FERRUM_SPEC_DRAFT /
// FERRUM_SPEC_N) are no longer read from the environment here. The CLI
// composition root captures them via `RuntimeConfigSnapshot::capture_current()`
// and lands them in `EngineConfig.runtime` through
// `apply_runtime_config_snapshot`; the builder reads `self.config.runtime`.

/// Engine builder for creating inference engines with registry-based components
pub struct EngineBuilder {
    /// Component registry to use
    registry: Arc<ComponentRegistry>,
    /// Engine configuration
    config: EngineConfig,
    /// Product-resolved semantic, tokenizer, and weight sources.
    model_sources: Option<Arc<ProductionModelSourceBundle>>,
    /// Immutable typed model package prepared once by the product composition
    /// root and reused by startup policy and executor construction.
    prepared_model: Option<Arc<PreparedProductionModel>>,
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
    /// Pre-created recurrent-state manager (skip factory)
    custom_recurrent_state_manager: Option<Arc<dyn RecurrentStateManager + Send + Sync>>,
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
            model_sources: None,
            prepared_model: None,
            tokenizer_name: None,
            sampler_name: None,
            scheduler_name: None,
            kv_cache_name: None,
            executor_name: None,
            custom_tokenizer: None,
            custom_sampler: None,
            custom_scheduler: None,
            custom_kv_cache: None,
            custom_recurrent_state_manager: None,
            custom_executor: None,
        }
    }

    pub fn with_model_sources(mut self, sources: Arc<ProductionModelSourceBundle>) -> Self {
        self.model_sources = Some(sources);
        self.prepared_model = None;
        self
    }

    pub fn with_prepared_model(mut self, prepared: Arc<PreparedProductionModel>) -> Self {
        self.model_sources = Some(Arc::clone(prepared.sources()));
        self.prepared_model = Some(prepared);
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

    /// Set a pre-created recurrent-state manager.
    pub fn with_custom_recurrent_state_manager(
        mut self,
        manager: Arc<dyn RecurrentStateManager + Send + Sync>,
    ) -> Self {
        self.custom_recurrent_state_manager = Some(manager);
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
        if self.has_typed_model_path() || self.config.runtime.model_path.is_some() {
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
        if self.has_typed_model_path() || self.config.runtime.model_path.is_some() {
            return "llm".to_string();
        }

        "stub".to_string()
    }

    fn has_typed_model_path(&self) -> bool {
        self.model_sources.is_some()
            || self
                .config
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

        if self.scheduler_name.is_some() || self.custom_scheduler.is_some() {
            return Err(FerrumError::config(
                "EngineBuilder scheduler component overrides are no longer accepted; configure the typed EngineConfig.scheduler used by ContinuousBatchScheduler",
            ));
        }

        // Pre-compute all component names before consuming self
        let tokenizer_name = self.resolve_tokenizer_name();
        let sampler_name = self.resolve_sampler_name();
        let kv_cache_name = self.resolve_kv_cache_name();
        let executor_name = self.resolve_executor_name();
        let explicit_kv_cache_override = self.kv_cache_name.is_some();

        let component_config = ComponentConfig::from_engine_config_and_product_model(
            &self.config,
            self.model_sources.clone(),
            self.prepared_model.clone(),
        );
        validate_layer_split_plan(&component_config)?;
        let typed_model_path = component_config.get_string_option("model_path");
        let has_model_path = typed_model_path.is_some() || self.config.runtime.model_path.is_some();
        let registry = self.registry.clone();
        let config = self.config;

        // Extract custom components. Phase 3e+ deleted the legacy
        // `ComputeBackend` trait, so there's no "backend" component to
        // build here — real GPU dispatch goes through `Backend<B>` in
        // `ferrum-kernels`. The stub executor now wires
        // `CandleTensorFactory` directly from the registry.
        let custom_tokenizer = self.custom_tokenizer;
        let custom_sampler = self.custom_sampler;
        let custom_kv_cache = self.custom_kv_cache;
        let custom_recurrent_state_manager = self.custom_recurrent_state_manager;
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

        // 4. Resolve the executor before resource managers. Its typed
        // authority decides whether engine-side KV/recurrent managers may
        // exist at all.
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
        let execution_resource_authority = executor.execution_resource_authority();

        let (kv_cache, recurrent_state_manager) = match execution_resource_authority {
            ferrum_interfaces::model_executor::ExecutionResourceAuthority::PlanRuntime => {
                if explicit_kv_cache_override || custom_kv_cache.is_some() {
                    return Err(FerrumError::config(
                        "plan runtime cannot be combined with a legacy engine KV-cache override",
                    ));
                }
                if custom_recurrent_state_manager.is_some() {
                    return Err(FerrumError::config(
                        "plan runtime cannot be combined with a legacy engine recurrent-state manager",
                    ));
                }
                if executor.resolved_model_plan().is_none() {
                    return Err(FerrumError::config(
                        "plan-runtime executor did not expose its authoritative ResolvedModelPlan",
                    ));
                }
                (None, None)
            }
            ferrum_interfaces::model_executor::ExecutionResourceAuthority::LegacyEngine => {
                let kv_cache = if let Some(kv_cache) = custom_kv_cache {
                    debug!("Using custom KV cache");
                    kv_cache
                } else {
                    debug!("Creating KV cache: {}", kv_cache_name);
                    registry
                        .create_kv_cache(&kv_cache_name, &component_config)
                        .await?
                };
                let recurrent_state_manager = custom_recurrent_state_manager
                    .or_else(|| default_recurrent_state_manager(&config, &component_config));
                (Some(kv_cache), recurrent_state_manager)
            }
        };

        // 5. Create the engine — always ContinuousBatchEngine.
        info!("All components created, building ContinuousBatchEngine");

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
            .or_else(|| config.runtime.spec_draft.clone());
        if execution_resource_authority
            == ferrum_interfaces::model_executor::ExecutionResourceAuthority::PlanRuntime
            && spec_draft.is_some()
        {
            return Err(FerrumError::unsupported(
                "speculative decoding is not yet part of the plan-runtime contract",
            ));
        }
        let spec_n = component_config
            .get_option::<usize>("spec_n")
            .unwrap_or(config.runtime.spec_n.unwrap_or(4));
        let (draft_executor, spec_config) = match spec_draft.as_ref() {
            Some(draft_path) => {
                info!("Speculative decoding: loading draft model from {draft_path}");
                let mut draft_cfg = component_config.clone();
                draft_cfg.component_options.insert(
                    "model_path".to_string(),
                    serde_json::Value::String(draft_path.to_string()),
                );
                let draft = registry
                    .create_executor(&executor_name, &draft_cfg)
                    .await
                    .map_err(|error| {
                        FerrumError::config(format!(
                            "requested speculative draft executor failed to load: {error}"
                        ))
                    })?;
                if draft.execution_resource_authority() != execution_resource_authority {
                    return Err(FerrumError::config(
                        "target and speculative draft executors declare different resource authority",
                    ));
                }
                (
                    Some(draft),
                    Some(crate::speculative::SpeculativeDecodingConfig {
                        num_speculative_tokens: spec_n,
                        temperature: 1.0,
                    }),
                )
            }
            _ => (None, None),
        };

        // This is the single product readiness boundary shared by `run` and
        // `serve`. Executor-owned compilation and warmup must finish here so
        // it cannot leak into the first request or diverge by entrypoint.
        executor.prepare_startup().await?;
        if let Some(draft) = draft_executor.as_ref() {
            draft.prepare_startup().await?;
        }

        let engine = match execution_resource_authority {
            ferrum_interfaces::model_executor::ExecutionResourceAuthority::PlanRuntime => {
                crate::ContinuousBatchEngine::new_plan_runtime(
                    config,
                    cb_scheduler,
                    tokenizer,
                    sampler,
                    executor,
                    tensor_factory,
                )?
            }
            ferrum_interfaces::model_executor::ExecutionResourceAuthority::LegacyEngine => {
                let kv_cache = kv_cache.ok_or_else(|| {
                    FerrumError::internal("legacy-engine composition lost its KV-cache manager")
                })?;
                crate::ContinuousBatchEngine::new_with_speculation_and_recurrent_state_manager(
                    config,
                    cb_scheduler,
                    tokenizer,
                    sampler,
                    kv_cache,
                    executor,
                    tensor_factory,
                    draft_executor,
                    spec_config,
                    recurrent_state_manager,
                )?
            }
        };
        Ok(Box::new(engine))
    }
}

fn default_recurrent_state_manager(
    config: &EngineConfig,
    _component_config: &ComponentConfig,
) -> Option<Arc<dyn RecurrentStateManager + Send + Sync>> {
    let total_batch_slots = config
        .runtime
        .recurrent_state_max_slots
        .unwrap_or(usize::MAX);
    #[cfg(any(test, feature = "legacy-qwen35-reference-test"))]
    if _component_config
        .get_option::<bool>("qwen35_reference")
        .unwrap_or(false)
    {
        return Some(
            Arc::new(ferrum_models::models::Qwen35RecurrentStateManager::<
                ferrum_kernels::backend::cpu::CpuBackend,
            >::new(
                ferrum_models::models::Qwen35RecurrentStateManagerConfig {
                    total_memory_bytes: usize::MAX,
                    total_batch_slots,
                },
            )) as Arc<dyn RecurrentStateManager + Send + Sync>,
        );
    }
    Some(
        Arc::new(crate::recurrent_state::InMemoryRecurrentStateManager::new(
            crate::recurrent_state::InMemoryRecurrentStateConfig {
                total_memory_bytes: usize::MAX,
                total_batch_slots,
            },
        )) as Arc<dyn RecurrentStateManager + Send + Sync>,
    )
}

fn validate_layer_split_plan(component_config: &ComponentConfig) -> Result<()> {
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
    let plan_raw = component_config.get_string_option("selected_layer_split_plan");
    let parsed_plan = if let Some(stages) = component_config
        .component_options
        .get("selected_layer_split_stages")
    {
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
    let execution_plan = parsed_plan.to_execution_plan();
    let stage_ranges = execution_plan
        .layer_distribution
        .stage_layers
        .iter()
        .map(|range| format!("{}-{}", range.start, range.end.saturating_sub(1)))
        .collect::<Vec<_>>()
        .join(",");
    let plan_label = plan_raw.unwrap_or_else(|| format!("{:?}", parsed_plan.stages));
    tracing::info!(
        "validated CUDA layer_split plan: requested_gpu_devices={requested:?} selected_gpu_devices={selected:?} selected_layer_split_plan={plan_label} total_layers={} pipeline_stages={} stage_ranges={stage_ranges} communication_backend={}",
        parsed_plan.total_layers(),
        execution_plan.parallel_config.pipeline_parallel_size,
        execution_plan.parallel_config.communication_backend,
    );
    Ok(())
}

/// Create an engine with the default configuration and registry
pub async fn create_engine(
    config: EngineConfig,
) -> Result<Box<dyn LlmInferenceEngine + Send + Sync>> {
    EngineBuilder::new(config).build().await
}

/// Create a product engine from one immutable role-specific source bundle.
pub async fn create_product_engine(
    config: EngineConfig,
    sources: Arc<ProductionModelSourceBundle>,
) -> Result<Box<dyn LlmInferenceEngine + Send + Sync>> {
    EngineBuilder::new(config)
        .with_model_sources(sources)
        .build()
        .await
}

/// Create a product engine from the exact typed model package already used by
/// startup capability and resource-policy resolution.
pub async fn create_prepared_product_engine(
    config: EngineConfig,
    prepared: Arc<PreparedProductionModel>,
) -> Result<Box<dyn LlmInferenceEngine + Send + Sync>> {
    EngineBuilder::new(config)
        .with_prepared_model(prepared)
        .build()
        .await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::{
        model_executor::{
            DecodeInput, DecodeOutput, ExecutionResourceAuthority, ExecutorCapabilities,
            ExecutorStatus, PlanRuntimeResourceSnapshot, PrefillInput, PrefillOutput,
        },
        RecurrentStateHandle, RecurrentStateManager, RecurrentStateManagerStats,
        RecurrentStateSpec, RecurrentStateTensorSpec,
    };
    use ferrum_types::{DataType, Device, RequestId};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct NoopRecurrentStateManager;

    struct PlanRuntimeBuilderExecutor {
        inner: ferrum_testkit::MockModelExecutor,
    }

    struct StartupProbeExecutor {
        inner: ferrum_testkit::MockModelExecutor,
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl ModelExecutor for StartupProbeExecutor {
        fn info(&self) -> &ferrum_types::ModelInfo {
            self.inner.info()
        }

        async fn prepare_startup(&self) -> Result<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                return Err(FerrumError::backend("startup preparation rejected"));
            }
            Ok(())
        }

        async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
            self.inner.prefill(input).await
        }

        async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
            self.inner.decode(input).await
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            self.inner.capabilities()
        }

        fn status(&self) -> ExecutorStatus {
            self.inner.status()
        }
    }

    #[async_trait::async_trait]
    impl ModelExecutor for PlanRuntimeBuilderExecutor {
        fn info(&self) -> &ferrum_types::ModelInfo {
            self.inner.info()
        }

        fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
            ExecutionResourceAuthority::PlanRuntime
        }

        fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
            PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).map(Some)
        }

        async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
            self.inner.prefill(input).await
        }

        async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
            self.inner.decode(input).await
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            self.inner.capabilities()
        }

        fn status(&self) -> ExecutorStatus {
            self.inner.status()
        }
    }

    struct CountingKvFactory {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl crate::registry::ComponentFactory<Arc<dyn KvCacheManager + Send + Sync>>
        for CountingKvFactory
    {
        async fn create(
            &self,
            _config: &ComponentConfig,
        ) -> Result<Arc<dyn KvCacheManager + Send + Sync>> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(ferrum_testkit::MockKvCacheManager::new(8)))
        }

        fn metadata(&self) -> crate::registry::ComponentMetadata {
            crate::registry::ComponentMetadata::default()
        }
    }

    #[async_trait::async_trait]
    impl RecurrentStateManager for NoopRecurrentStateManager {
        async fn allocate(
            &self,
            _spec: &RecurrentStateSpec,
        ) -> Result<Arc<dyn RecurrentStateHandle>> {
            Err(FerrumError::unsupported(
                "noop recurrent-state manager does not allocate",
            ))
        }

        async fn deallocate(&self, _request_id: RequestId) -> Result<()> {
            Ok(())
        }

        fn can_allocate(&self, _spec: &RecurrentStateSpec) -> bool {
            false
        }

        fn get_handle(&self, _request_id: RequestId) -> Option<Arc<dyn RecurrentStateHandle>> {
            None
        }

        fn list_handles(&self) -> Vec<(RequestId, Arc<dyn RecurrentStateHandle>)> {
            Vec::new()
        }

        fn stats(&self) -> RecurrentStateManagerStats {
            RecurrentStateManagerStats {
                total_memory_bytes: 0,
                used_memory_bytes: 0,
                active_states: 0,
                active_state_tensors: 0,
                total_batch_slots: 0,
                used_batch_slots: 0,
                allocation_count: 0,
                allocation_failures: 0,
                eviction_count: 0,
            }
        }

        async fn reset(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn test_builder_creation() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config);

        assert!(builder.tokenizer_name.is_none());
        assert!(builder.custom_recurrent_state_manager.is_none());
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
    fn test_builder_with_custom_recurrent_state_manager() {
        let config = EngineConfig::default();
        let manager = Arc::new(NoopRecurrentStateManager);
        let builder = EngineBuilder::new(config).with_custom_recurrent_state_manager(manager);

        assert!(builder.custom_recurrent_state_manager.is_some());
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
    fn test_builder_retains_one_typed_source_bundle_for_components() {
        let root = std::env::temp_dir().join(format!(
            "ferrum-builder-source-bundle-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("config.json"),
            br#"{"architectures":["Fixture"]}"#,
        )
        .unwrap();
        std::fs::write(root.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(root.join("model.safetensors"), b"fixture").unwrap();
        let original = ferrum_interfaces::vnext::OriginalModelSource {
            kind: ferrum_interfaces::vnext::ModelSourceKind::LocalDirectory,
            location: root.display().to_string(),
            requested_revision: None,
        };
        let sources = Arc::new(
            ProductionModelSourceBundle::open(
                &root,
                &root,
                ferrum_models::vnext::ProductionWeightArtifact::safetensors_directory(&root),
                ferrum_interfaces::vnext::OriginalModelSources {
                    semantic: original.clone(),
                    tokenizer: original.clone(),
                    weights: original,
                },
            )
            .unwrap(),
        );

        let builder =
            EngineBuilder::new(EngineConfig::default()).with_model_sources(Arc::clone(&sources));
        assert!(builder.has_typed_model_path());
        assert!(Arc::ptr_eq(
            builder.model_sources.as_ref().unwrap(),
            &sources
        ));
        assert_eq!(builder.resolve_tokenizer_name(), "huggingface");
        assert_eq!(builder.resolve_executor_name(), "llm");
        std::fs::remove_dir_all(root).unwrap();
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

    #[test]
    fn test_builder_qwen35_reference_uses_typed_recurrent_state_manager() {
        let mut config = EngineConfig::default();
        config.backend.device = Device::CPU;
        config.runtime.recurrent_state_max_slots = Some(4);
        config.backend.backend_options.insert(
            "qwen35_reference".to_string(),
            serde_json::Value::Bool(true),
        );
        let component_config = ComponentConfig::from_engine_config(&config);
        let manager = default_recurrent_state_manager(&config, &component_config)
            .expect("qwen35 reference CPU path should install a recurrent-state manager");
        let spec = RecurrentStateSpec {
            request_id: RequestId::new(),
            num_layers: 2,
            tensors: vec![RecurrentStateTensorSpec::new(
                0,
                "delta_state",
                vec![1, 1, 1],
                DataType::FP32,
            )],
            device: Device::CPU,
            max_batch_slots: 1,
        };

        let handle = tokio_test::block_on(manager.allocate(&spec)).unwrap();

        assert!(
            handle
                .as_any()
                .is::<ferrum_models::models::Qwen35RecurrentStateHandle<
                    ferrum_kernels::backend::cpu::CpuBackend,
                >>(),
            "qwen35 reference should allocate typed Qwen35 recurrent-state handles"
        );
    }

    #[test]
    fn test_builder_cuda_recurrent_state_manager_uses_recurrent_state_slot_cap() {
        let mut config = EngineConfig::default();
        config.backend.device = Device::CUDA(0);
        config.runtime.recurrent_state_max_slots = Some(2);
        let component_config = ComponentConfig::from_engine_config(&config);
        let manager = default_recurrent_state_manager(&config, &component_config)
            .expect("cuda product path should install admission recurrent-state manager");
        let spec = |request_id| RecurrentStateSpec {
            request_id,
            num_layers: 1,
            tensors: vec![RecurrentStateTensorSpec::new(
                0,
                "delta_state",
                vec![1, 1, 1],
                DataType::FP32,
            )],
            device: Device::CUDA(0),
            max_batch_slots: 1,
        };

        tokio_test::block_on(manager.allocate(&spec(RequestId::new()))).unwrap();
        tokio_test::block_on(manager.allocate(&spec(RequestId::new()))).unwrap();
        let err = tokio_test::block_on(manager.allocate(&spec(RequestId::new())))
            .expect_err("third recurrent allocation should exceed the two-slot cap");

        assert!(matches!(err, FerrumError::ResourceExhausted { .. }));
        let stats = manager.stats();
        assert_eq!(stats.total_batch_slots, 2);
        assert_eq!(stats.used_batch_slots, 2);
        assert_eq!(stats.allocation_failures, 1);
    }

    #[test]
    fn test_builder_validates_layer_split_plan_without_executor_reject() {
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
        let component_config = ComponentConfig::from_engine_config(&config);

        validate_layer_split_plan(&component_config).unwrap();
    }

    #[tokio::test]
    async fn test_builder_rejects_invalid_layer_split_plan_before_executor_build() {
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
        assert!(err.to_string().contains("expected START-END"));
    }

    #[test]
    fn test_resolve_defaults() {
        let config = EngineConfig::default();
        let builder = EngineBuilder::new(config);

        assert_eq!(builder.resolve_sampler_name(), "multinomial");
        assert_eq!(builder.resolve_kv_cache_name(), "default");
    }

    #[tokio::test]
    async fn test_build_with_defaults() {
        let config = EngineConfig::default();
        let result = EngineBuilder::new(config).build().await;

        // Should succeed with stub components
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn startup_preparation_runs_once_and_blocks_engine_construction_on_failure() {
        let success_calls = Arc::new(AtomicUsize::new(0));
        let success: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(StartupProbeExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
            calls: Arc::clone(&success_calls),
            fail: false,
        });
        EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(success)
            .build()
            .await
            .expect("successful startup preparation builds the engine");
        assert_eq!(success_calls.load(Ordering::Relaxed), 1);

        let failure_calls = Arc::new(AtomicUsize::new(0));
        let failure: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(StartupProbeExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
            calls: Arc::clone(&failure_calls),
            fail: true,
        });
        let error = EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(failure)
            .build()
            .await
            .err()
            .expect("failed startup preparation must stop engine construction");
        assert!(error.to_string().contains("startup preparation rejected"));
        assert_eq!(failure_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn plan_runtime_without_resolved_plan_rejects_before_legacy_kv_factory() {
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = Arc::new(ComponentRegistry::with_defaults());
        registry.register_kv_cache_factory(
            "default",
            Arc::new(CountingKvFactory {
                calls: Arc::clone(&calls),
            }),
        );
        let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(PlanRuntimeBuilderExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
        });

        let result = EngineBuilder::with_registry(EngineConfig::default(), registry)
            .with_custom_executor(executor)
            .build()
            .await;

        let error = result
            .err()
            .expect("plan runtime without a resolved plan must fail closed");
        assert!(error
            .to_string()
            .contains("authoritative ResolvedModelPlan"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn plan_runtime_build_rejects_legacy_resource_override() {
        let executor: Arc<dyn ModelExecutor + Send + Sync> = Arc::new(PlanRuntimeBuilderExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
        });
        let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
            Arc::new(ferrum_testkit::MockKvCacheManager::new(8));

        let error = EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(executor)
            .with_custom_kv_cache(kv_cache)
            .build()
            .await
            .err()
            .expect("plan runtime must reject a legacy KV manager override");

        assert!(error
            .to_string()
            .contains("cannot be combined with a legacy engine KV-cache override"));
    }
}
