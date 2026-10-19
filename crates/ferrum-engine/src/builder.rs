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
use crate::{EngineExecutor, IntoEngineExecutor};
use ferrum_execution::execution_attempt::ExecutionAttemptObserver;
use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorScope};
#[cfg(test)]
use ferrum_execution::ModelExecutor;
use ferrum_interfaces::engine::{InferenceEngine, LlmInferenceEngine};
use ferrum_interfaces::{Sampler, Tokenizer};
use ferrum_models::vnext::{DefinedProductionModel, ProductionModelSourceBundle};
use ferrum_scheduler::cost_prediction::CostSnapshotSource;
use ferrum_types::{EngineConfig, FerrumError, Result};
use std::sync::Arc;
use tracing::{debug, info};

mod source_binding;

// The product composition root captures model-loading options in the typed
// configuration. The builder does not read process environment variables.

/// Engine builder for creating inference engines with registry-based components
pub struct EngineBuilder {
    journal_scopes: Option<crate::EngineJournalScopes>,
    execution_attempt_observer: Option<Arc<dyn ExecutionAttemptObserver>>,
    cost_snapshot_source: Option<Arc<dyn CostSnapshotSource>>,
    /// Component registry to use
    registry: Arc<ComponentRegistry>,
    /// Engine configuration
    config: EngineConfig,
    /// Product-resolved semantic, tokenizer, and weight sources.
    model_sources: Option<Arc<ProductionModelSourceBundle>>,
    /// Immutable typed model package prepared once by the product composition
    /// root and reused by startup policy and executor construction.
    defined_model: Option<Arc<DefinedProductionModel>>,
    /// Override: custom tokenizer name
    tokenizer_name: Option<String>,
    /// Override: custom sampler name
    sampler_name: Option<String>,
    /// Override: custom executor name
    executor_name: Option<String>,
    /// Pre-created tokenizer (skip factory)
    custom_tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
    /// Pre-created sampler (skip factory)
    custom_sampler: Option<Arc<dyn Sampler + Send + Sync>>,
    /// Pre-created executor (skip factory)
    custom_executor: Option<EngineExecutor>,
}

impl EngineBuilder {
    /// Create a new engine builder with default registry
    pub fn new(config: EngineConfig) -> Self {
        Self::with_registry(config, crate::registry::global_registry())
    }

    /// Create a new engine builder with a custom registry
    pub fn with_registry(config: EngineConfig, registry: Arc<ComponentRegistry>) -> Self {
        Self {
            journal_scopes: None,
            execution_attempt_observer: None,
            cost_snapshot_source: None,
            registry,
            config,
            model_sources: None,
            defined_model: None,
            tokenizer_name: None,
            sampler_name: None,
            executor_name: None,
            custom_tokenizer: None,
            custom_sampler: None,
            custom_executor: None,
        }
    }

    pub fn with_model_sources(mut self, sources: Arc<ProductionModelSourceBundle>) -> Self {
        self.model_sources = Some(sources);
        self.defined_model = None;
        self
    }

    /// Use assembly-owned journal destinations. Engine shutdown retires these
    /// producer scopes; the assembly retains final writer-close authority.
    pub fn with_journal_scopes(mut self, scopes: crate::EngineJournalScopes) -> Self {
        self.journal_scopes = Some(scopes);
        self
    }

    /// Observe bounded product execution facts independently of detailed traces.
    /// The observer is fixed before startup; omitting it keeps collection off.
    pub fn with_execution_attempt_observer(
        mut self,
        observer: Arc<dyn ExecutionAttemptObserver>,
    ) -> Self {
        self.execution_attempt_observer = Some(observer);
        self
    }

    /// Install one feedback assembly before startup. The engine only observes
    /// facts and reads published predictions; its owner lives in the assembly.
    pub fn with_execution_feedback(
        mut self,
        observer: Arc<dyn ExecutionAttemptObserver>,
        source: Arc<dyn CostSnapshotSource>,
    ) -> Self {
        self.execution_attempt_observer = Some(observer);
        self.cost_snapshot_source = Some(source);
        self
    }

    pub fn with_defined_model(mut self, prepared: Arc<DefinedProductionModel>) -> Self {
        self.model_sources = Some(Arc::clone(prepared.sources()));
        self.defined_model = Some(prepared);
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

    /// Set the executor to use by name
    pub fn with_executor(mut self, name: impl Into<String>) -> Self {
        self.executor_name = Some(name.into());
        self
    }

    /// Set a pre-created model executor
    pub fn with_custom_executor(mut self, executor: ExecutorScope) -> Self {
        self.custom_executor = Some(
            executor
                .into_engine_executor()
                .expect("existing scope is infallible"),
        );
        self
    }

    /// Transfer final-close authority with a custom executor.
    pub fn with_owned_executor(mut self, executor: ExecutorOwner) -> Result<Self> {
        self.custom_executor = Some(executor.into_engine_executor()?);
        Ok(self)
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

    /// Determine which executor to use based on config and overrides
    fn resolve_executor_name(&self) -> String {
        if let Some(ref name) = self.executor_name {
            return name.clone();
        }

        // A model source selects the registered language-model executor.
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
        crate::continuous_engine::validate_execution_feedback(
            &self.config,
            self.execution_attempt_observer.as_ref(),
            self.cost_snapshot_source.as_ref(),
        )?;
        info!(
            "Building inference engine for model: {}",
            self.config.model.model_id
        );

        for key in ["spec_draft", "spec_n"] {
            if self.config.backend.backend_options.contains_key(key) {
                return Err(FerrumError::config(format!(
                    "backend option '{key}' belongs to retired experimental speculative decoding"
                )));
            }
        }

        // Pre-compute all component names before consuming self
        let tokenizer_name = self.resolve_tokenizer_name();
        let sampler_name = self.resolve_sampler_name();
        let executor_name = self.resolve_executor_name();

        let component_config = ComponentConfig::from_engine_config_and_product_model(
            &self.config,
            self.model_sources.clone(),
            self.defined_model.clone(),
        );
        validate_layer_split_plan(&component_config)?;
        let typed_model_path = component_config.get_string_option("model_path");
        let has_model_path = typed_model_path.is_some() || self.config.runtime.model_path.is_some();
        let registry = self.registry.clone();
        let mut config = self.config;

        // Extract custom components. Phase 3e+ deleted the legacy
        // `ComputeBackend` trait, so there's no "backend" component to
        // build here — real GPU dispatch goes through `Backend<B>` in
        // `ferrum-kernels` and stays owned by the original executor.
        let custom_tokenizer = self.custom_tokenizer;
        let custom_sampler = self.custom_sampler;
        let custom_executor = self.custom_executor;
        let journal_scopes = self.journal_scopes;
        let execution_attempt_observer = self.execution_attempt_observer;
        let cost_snapshot_source = self.cost_snapshot_source;

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
                Ok(e) => e.into_engine_executor()?,
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
                    registry
                        .create_executor("stub", &component_config)
                        .await?
                        .into_engine_executor()?
                }
            }
        };
        source_binding::validate(
            &config,
            component_config.model_sources.as_ref(),
            component_config.defined_model.as_ref(),
            executor.as_ref(),
        )?;
        if let Some(plan) = executor.startup_memory_plan() {
            plan.apply_to_engine_config(&mut config)
                .map_err(FerrumError::config)?;
        }
        let execution_planning_model = executor.execution_planning_model();

        if execution_planning_model
            == ferrum_execution::model_executor::ExecutionPlanningModel::CompiledPlan
            && executor.resolved_model_plan().is_none()
        {
            return Err(FerrumError::config(
                "plan-runtime executor did not expose its authoritative ResolvedModelPlan",
            ));
        }

        // 5. Create the engine — always ContinuousBatchEngine.
        info!("All components created, building ContinuousBatchEngine");

        let cb_scheduler = Arc::new(
            ferrum_scheduler::implementations::ContinuousBatchScheduler::new(
                config.scheduler.clone(),
            ),
        );

        // Construct the unpublished engine shell first so its typed profile
        // sink is attached before executor startup emits warmup/capture events.
        let engine = crate::ContinuousBatchEngine::new_observed(
            config,
            cb_scheduler,
            tokenizer,
            sampler,
            executor,
            journal_scopes,
            execution_attempt_observer,
            cost_snapshot_source,
        )?;

        // This is the single product readiness boundary shared by `run` and
        // `serve`. The engine is not exposed until executor-owned compilation
        // and warmup complete, but those events now share the product trace.
        let startup_result = engine.prepare_startup().await;
        if let Err(startup_error) = startup_result {
            if let Err(shutdown_error) = engine.shutdown().await {
                tracing::warn!(
                    "Failed to close engine resources after startup rejection: {shutdown_error}"
                );
            }
            return Err(startup_error);
        }
        Ok(Box::new(engine))
    }
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
pub async fn create_defined_product_engine(
    config: EngineConfig,
    prepared: Arc<DefinedProductionModel>,
) -> Result<Box<dyn LlmInferenceEngine + Send + Sync>> {
    EngineBuilder::new(config)
        .with_defined_model(prepared)
        .build()
        .await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_execution::executor_lifecycle::{ExecutorShutdownObserver, ExecutorShutdownPermit};
    use ferrum_execution::model_executor::{
        ExecutionPlanningModel, ExecutorCapabilities, ExecutorStatus, PlanRuntimeResourceSnapshot,
    };
    use ferrum_execution::ProductExecutionEventSink;
    use ferrum_runtime::ExecutionEventSink;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    };

    struct PlanRuntimeBuilderExecutor {
        inner: ferrum_testkit::MockModelExecutor,
    }

    struct StartupProbeExecutor {
        inner: ferrum_testkit::MockModelExecutor,
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    struct ProfileStartupProbeExecutor {
        inner: ferrum_testkit::MockModelExecutor,
        event_sink: Mutex<Option<Arc<dyn ProductExecutionEventSink>>>,
        saw_sink_during_startup: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl ModelExecutor for StartupProbeExecutor {
        fn begin_shutdown(
            self: Arc<Self>,
            permit: &ExecutorShutdownPermit,
        ) -> ExecutorShutdownObserver {
            if let Err(error) = permit.check(self.as_ref()) {
                return ExecutorShutdownObserver::retained(Arc::new(self), error);
            }
            match Arc::try_unwrap(self) {
                Ok(original) => ExecutorOwner::new(Box::new(original.inner)).begin_shutdown(),
                Err(original) => ExecutorShutdownObserver::retained(
                    Arc::new(original),
                    FerrumError::internal("startup probe still has execution borrowers"),
                ),
            }
        }

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

        fn capabilities(&self) -> ExecutorCapabilities {
            self.inner.capabilities()
        }

        fn status(&self) -> ExecutorStatus {
            self.inner.status()
        }
    }

    #[async_trait::async_trait]
    impl ModelExecutor for ProfileStartupProbeExecutor {
        fn begin_shutdown(
            self: Arc<Self>,
            permit: &ExecutorShutdownPermit,
        ) -> ExecutorShutdownObserver {
            if let Err(error) = permit.check(self.as_ref()) {
                return ExecutorShutdownObserver::retained(Arc::new(self), error);
            }
            match Arc::try_unwrap(self) {
                Ok(original) => ExecutorOwner::new(Box::new(original.inner)).begin_shutdown(),
                Err(original) => ExecutorShutdownObserver::retained(
                    Arc::new(original),
                    FerrumError::internal("profile probe still has execution borrowers"),
                ),
            }
        }

        fn info(&self) -> &ferrum_types::ModelInfo {
            self.inner.info()
        }

        async fn prepare_startup(&self) -> Result<()> {
            use ferrum_runtime::{ExecutionEventEmitter, TrustedExecutionEventContext};

            let sink = self
                .event_sink
                .lock()
                .expect("profile startup probe sink lock")
                .clone();
            self.saw_sink_during_startup
                .store(sink.is_some(), Ordering::Release);
            let Some(sink) = sink else {
                return Ok(());
            };
            let (run_id, request_id, event) = startup_profile_test_event();
            let runtime_sink: Arc<dyn ExecutionEventSink> = sink;
            ExecutionEventEmitter::from_shared(runtime_sink, run_id.clone(), request_id.clone())
                .emit(
                    event,
                    &TrustedExecutionEventContext::pre_plan(&run_id, &request_id),
                )
                .map_err(|error| {
                    FerrumError::internal(format!("emit startup profile probe: {error}"))
                })
        }

        fn attach_execution_event_sink(&self, sink: Arc<dyn ProductExecutionEventSink>) {
            *self
                .event_sink
                .lock()
                .expect("profile startup probe sink lock") = Some(sink);
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            self.inner.capabilities()
        }

        fn status(&self) -> ExecutorStatus {
            self.inner.status()
        }
    }

    fn startup_profile_test_event() -> (
        ferrum_runtime::RunId,
        ferrum_runtime::RequestIdentity,
        ferrum_runtime::ExecutionEvent,
    ) {
        use ferrum_runtime::{
            ExecutionEvent, ExecutionEventDetail, ExecutionEventKind, ExecutionIdentityEnvelope,
            ExecutionIdentityParts, ExecutionPhase, MonotonicTimestamp, RequestIdentity, RunId,
            SpanId, EXECUTION_IDENTITY_VERSION,
        };

        let run_id = RunId::new("run.vnext.builder-startup-profile").unwrap();
        let request_id = RequestIdentity::new("request.vnext.builder-startup-profile").unwrap();
        let event = ExecutionEvent::new(
            MonotonicTimestamp {
                nanos_since_run_start: 1,
            },
            ExecutionPhase::Resolution,
            ExecutionEventKind::RequestAccepted,
            ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
                version: EXECUTION_IDENTITY_VERSION,
                run_id: run_id.clone(),
                request_id: request_id.clone(),
                sequence: 1,
                plan_id: None,
                plan_hash: None,
                frame_id: None,
                node_invocation_id: None,
                node_id: None,
                operation_id: None,
                provider_id: None,
                device_id: None,
                resource_pool_id: None,
                resource_pool_identity_fingerprint: None,
                provisioning_run_id: None,
                provisioning_request_id: None,
                transaction_id: None,
                active_sequence_slot: None,
                admission_generation: None,
                activation_epoch: None,
                runtime_implementation_fingerprint: None,
                active_sequence_fingerprint: None,
                completed_sequence_fingerprint: None,
                aborted_sequence_fingerprint: None,
                resource_id: None,
                resource_generation: None,
                resource_batch_fingerprint: None,
                span_id: SpanId::new("vnext/request/builder-startup-profile").unwrap(),
                parent_span_id: None,
                async_links: Vec::new(),
            })
            .unwrap(),
            ExecutionEventDetail::None,
        )
        .unwrap();
        (run_id, request_id, event)
    }

    #[async_trait::async_trait]
    impl ModelExecutor for PlanRuntimeBuilderExecutor {
        fn begin_shutdown(
            self: Arc<Self>,
            permit: &ExecutorShutdownPermit,
        ) -> ExecutorShutdownObserver {
            if let Err(error) = permit.check(self.as_ref()) {
                return ExecutorShutdownObserver::retained(Arc::new(self), error);
            }
            match Arc::try_unwrap(self) {
                Ok(original) => ExecutorOwner::new(Box::new(original.inner)).begin_shutdown(),
                Err(original) => ExecutorShutdownObserver::retained(
                    Arc::new(original),
                    FerrumError::internal("plan probe still has execution borrowers"),
                ),
            }
        }

        fn info(&self) -> &ferrum_types::ModelInfo {
            self.inner.info()
        }

        fn execution_planning_model(&self) -> ExecutionPlanningModel {
            ExecutionPlanningModel::CompiledPlan
        }

        fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
            PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).map(Some)
        }

        fn capabilities(&self) -> ExecutorCapabilities {
            self.inner.capabilities()
        }

        fn status(&self) -> ExecutorStatus {
            self.inner.status()
        }
    }

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
            .with_executor("custom_executor");

        assert_eq!(builder.tokenizer_name, Some("custom_tokenizer".to_string()));
        assert_eq!(builder.sampler_name, Some("greedy".to_string()));
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
        let original = ferrum_runtime::OriginalModelSource {
            kind: ferrum_runtime::ModelSourceKind::LocalDirectory,
            location: root.display().to_string(),
            requested_revision: None,
        };
        let sources = Arc::new(
            ProductionModelSourceBundle::open(
                &root,
                &root,
                ferrum_models::vnext::ProductionWeightArtifact::safetensors_directory(&root),
                ferrum_runtime::OriginalModelSources {
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

    #[tokio::test]
    async fn retired_speculative_options_fail_before_component_loading() {
        for key in ["spec_draft", "spec_n"] {
            for value in [
                serde_json::Value::Null,
                serde_json::json!(""),
                serde_json::json!(6),
            ] {
                let mut config = EngineConfig::default();
                config.backend.backend_options.insert(key.to_owned(), value);
                let error = EngineBuilder::new(config)
                    .with_tokenizer("not-registered")
                    .build()
                    .await
                    .err()
                    .expect("retired options must not silently select ordinary decoding");
                assert!(matches!(&error, FerrumError::Config { .. }), "{error}");
                let message = error.to_string();
                assert!(
                    message.contains(key) && message.contains("retired"),
                    "{message}"
                );
            }
        }
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
    }

    #[tokio::test]
    async fn test_build_with_defaults() {
        let config = EngineConfig::default();
        let result = EngineBuilder::new(config).build().await;

        // Should succeed with stub components
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn startup_preparation_runs_once_and_blocks_engine_publication_on_failure() {
        let success_calls = Arc::new(AtomicUsize::new(0));
        let success = ExecutorOwner::new(Box::new(StartupProbeExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
            calls: Arc::clone(&success_calls),
            fail: false,
        }));
        let engine = EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(success.scope().unwrap())
            .build()
            .await
            .expect("successful startup preparation builds the engine");
        assert_eq!(success_calls.load(Ordering::Relaxed), 1);
        engine.shutdown().await.unwrap();
        success.begin_shutdown().wait().await.result().unwrap();

        let failure_calls = Arc::new(AtomicUsize::new(0));
        let failure = ExecutorOwner::new(Box::new(StartupProbeExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
            calls: Arc::clone(&failure_calls),
            fail: true,
        }));
        let error = EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(failure.scope().unwrap())
            .build()
            .await
            .err()
            .expect("failed startup preparation must stop engine construction");
        assert!(error.to_string().contains("startup preparation rejected"));
        assert_eq!(failure_calls.load(Ordering::Relaxed), 1);
        failure.begin_shutdown().wait().await.result().unwrap();
    }

    #[tokio::test]
    async fn product_profile_captures_startup_events_before_engine_readiness() {
        let trace_path = std::env::temp_dir().join(format!(
            "ferrum-builder-startup-profile-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&trace_path);
        let saw_sink_during_startup = Arc::new(AtomicBool::new(false));
        let executor = ExecutorOwner::new(Box::new(ProfileStartupProbeExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
            event_sink: Mutex::new(None),
            saw_sink_during_startup: Arc::clone(&saw_sink_during_startup),
        }));
        let mut config = EngineConfig::default();
        config.runtime.profile_jsonl = Some(trace_path.clone());
        config.runtime.profile_entrypoint = Some(ferrum_types::ProfileEntrypoint::Run);

        // Product assembly owns the executor and may install its startup sink.
        // A borrowed scope cannot replace another assembly's event sink.
        let engine = EngineBuilder::new(config)
            .with_owned_executor(executor)
            .expect("original executor owner provides the product scope")
            .build()
            .await
            .expect("profile-enabled startup builds the engine");
        assert!(saw_sink_during_startup.load(Ordering::Acquire));
        engine.shutdown().await.unwrap();

        let startup_events = std::fs::read_to_string(&trace_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|event| event["phase"] == "vnext.request_accepted")
            .count();
        assert_eq!(startup_events, 1);
        let _ = std::fs::remove_file(trace_path);
    }

    #[tokio::test]
    async fn product_without_profile_does_not_attach_execution_event_sink() {
        let saw_sink_during_startup = Arc::new(AtomicBool::new(false));
        let executor = ExecutorOwner::new(Box::new(ProfileStartupProbeExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
            event_sink: Mutex::new(None),
            saw_sink_during_startup: Arc::clone(&saw_sink_during_startup),
        }));

        let engine = EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(executor.scope().unwrap())
            .build()
            .await
            .expect("profile-disabled engine builds");
        assert!(!saw_sink_during_startup.load(Ordering::Acquire));
        engine.shutdown().await.unwrap();
        executor.begin_shutdown().wait().await.result().unwrap();
    }

    #[tokio::test]
    async fn plan_runtime_without_resolved_plan_rejects_before_startup() {
        let executor = ExecutorOwner::new(Box::new(PlanRuntimeBuilderExecutor {
            inner: ferrum_testkit::MockModelExecutor::instant(128),
        }));

        let result = EngineBuilder::new(EngineConfig::default())
            .with_custom_executor(executor.scope().unwrap())
            .build()
            .await;

        let error = result
            .err()
            .expect("plan runtime without a resolved plan must fail closed");
        assert!(error
            .to_string()
            .contains("authoritative ResolvedModelPlan"));
        executor.begin_shutdown().wait().await.result().unwrap();
    }
}
