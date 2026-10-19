use super::*;
use crate::builder::EngineBuilder;
use ferrum_execution::executor_lifecycle::{
    ExecutorOwner, ExecutorScope, ExecutorShutdownObserver, ExecutorShutdownPermit,
};
use ferrum_execution::model_executor::{
    ExecutorCapabilities, ExecutorStatus, PlanRuntimeResourceSnapshot,
};
use ferrum_runtime::{
    DeviceId, ModelSourceKind, OriginalModelSource, OriginalModelSources, ResolvedModelSources,
    WeightMaterializerSelection,
};
use ferrum_types::{Device, ExecutorAdmissionLimits, ModelInfo};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct FixtureDir(PathBuf);

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

enum SourceReport {
    Missing,
    Bound(ResolvedModelSources),
    Delegate,
}

struct BindingProbe {
    inner: ExecutorScope,
    owner: ExecutorOwner,
    report: SourceReport,
    observation: Arc<BindingObservation>,
}

struct BindingObservation {
    startup_calls: AtomicUsize,
    memory_plan_reads: AtomicUsize,
}

impl BindingProbe {
    fn new(owner: ExecutorOwner, report: SourceReport) -> (ExecutorOwner, Arc<BindingObservation>) {
        let observation = Arc::new(BindingObservation {
            startup_calls: AtomicUsize::new(0),
            memory_plan_reads: AtomicUsize::new(0),
        });
        let probe = Self {
            inner: owner.scope().unwrap(),
            owner,
            report,
            observation: Arc::clone(&observation),
        };
        (ExecutorOwner::new(Box::new(probe)), observation)
    }

    fn mock(report: SourceReport) -> (ExecutorOwner, Arc<BindingObservation>) {
        Self::new(
            ExecutorOwner::new(Box::new(ferrum_testkit::MockModelExecutor::instant(128))),
            report,
        )
    }
}

#[async_trait::async_trait]
impl ModelExecutor for BindingProbe {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(Self {
                inner,
                owner,
                report: _,
                observation: _,
            }) => {
                drop(inner);
                owner.begin_shutdown()
            }
            Err(original) => ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("source binding probe still has execution borrowers"),
            ),
        }
    }

    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn bound_resolved_sources(&self) -> Option<&ResolvedModelSources> {
        match &self.report {
            SourceReport::Missing => None,
            SourceReport::Bound(sources) => Some(sources),
            SourceReport::Delegate => self.inner.bound_resolved_sources(),
        }
    }

    fn resolved_model_plan(&self) -> Option<&ResolvedModelPlan> {
        self.inner.resolved_model_plan()
    }

    fn execution_planning_model(&self) -> ExecutionPlanningModel {
        self.inner.execution_planning_model()
    }

    fn admission_limits(&self) -> Result<Option<ExecutorAdmissionLimits>> {
        self.inner.admission_limits()
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        self.inner.plan_runtime_resource_snapshot()
    }

    fn kv_capacity(&self) -> Option<usize> {
        self.inner.kv_capacity()
    }

    fn startup_memory_plan(&self) -> Option<&ferrum_types::StartupMemoryPlan> {
        self.observation
            .memory_plan_reads
            .fetch_add(1, Ordering::Relaxed);
        self.inner.startup_memory_plan()
    }

    async fn prepare_startup(&self) -> Result<()> {
        self.observation
            .startup_calls
            .fetch_add(1, Ordering::Relaxed);
        self.inner.prepare_startup().await
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

fn unregistered_sources() -> (FixtureDir, Arc<ProductionModelSourceBundle>) {
    let dir = FixtureDir(std::env::temp_dir().join(format!(
        "ferrum-builder-source-binding-{}",
        uuid::Uuid::new_v4()
    )));
    std::fs::create_dir_all(&dir.0).unwrap();
    std::fs::write(
        dir.0.join("config.json"),
        br#"{"architectures":["CustomFixtureForCausalLM"]}"#,
    )
    .unwrap();
    std::fs::write(dir.0.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
    std::fs::write(dir.0.join("model.safetensors"), b"source-identity-only").unwrap();
    let original = OriginalModelSource {
        kind: ModelSourceKind::LocalDirectory,
        location: dir.0.display().to_string(),
        requested_revision: None,
    };
    let sources = ProductionModelSourceBundle::open(
        &dir.0,
        &dir.0,
        ferrum_models::vnext::ProductionWeightArtifact::safetensors_directory(&dir.0),
        OriginalModelSources {
            semantic: original.clone(),
            tokenizer: original.clone(),
            weights: original,
        },
    )
    .unwrap();
    (dir, Arc::new(sources))
}

fn custom_builder(
    sources: Arc<ProductionModelSourceBundle>,
    owner: &ExecutorOwner,
) -> EngineBuilder {
    EngineBuilder::new(EngineConfig::default())
        .with_model_sources(sources)
        .with_tokenizer("stub")
        .with_custom_executor(owner.scope().unwrap())
}

async fn assert_rejected_before_startup(
    builder: EngineBuilder,
    probe: &BindingObservation,
    owner: &ExecutorOwner,
    reason: &str,
) {
    let error = builder
        .build()
        .await
        .err()
        .expect("mismatched product binding must reject construction");
    assert!(
        error.to_string().contains(reason),
        "unexpected error: {error}"
    );
    assert_eq!(probe.startup_calls.load(Ordering::Relaxed), 0);
    assert_eq!(probe.memory_plan_reads.load(Ordering::Relaxed), 0);
    owner.begin_shutdown().wait().await.result().unwrap();
}

#[tokio::test]
async fn source_binding_preserves_source_independent_custom_executor() {
    let (owner, probe) = BindingProbe::mock(SourceReport::Missing);
    let engine = EngineBuilder::new(EngineConfig::default())
        .with_custom_executor(owner.scope().unwrap())
        .build()
        .await
        .unwrap();
    assert_eq!(probe.startup_calls.load(Ordering::Relaxed), 1);
    engine.shutdown().await.unwrap();
    owner.begin_shutdown().wait().await.result().unwrap();
}

#[tokio::test]
async fn source_binding_accepts_matching_custom_and_delegated_sources() {
    let (_dir, sources) = unregistered_sources();
    for delegate in [false, true] {
        let (inner_owner, inner) =
            BindingProbe::mock(SourceReport::Bound(sources.resolved_sources().clone()));
        let (owner, probe) = if delegate {
            BindingProbe::new(inner_owner, SourceReport::Delegate)
        } else {
            (inner_owner, Arc::clone(&inner))
        };
        let engine = custom_builder(sources.clone(), &owner)
            .build()
            .await
            .unwrap();
        assert_eq!(probe.startup_calls.load(Ordering::Relaxed), 1);
        assert_eq!(inner.startup_calls.load(Ordering::Relaxed), 1);
        engine.shutdown().await.unwrap();
        owner.begin_shutdown().wait().await.result().unwrap();
    }
}

#[tokio::test]
async fn source_binding_rejects_missing_and_each_mismatched_role_before_startup() {
    let (_dir, sources) = unregistered_sources();
    let (owner, missing) = BindingProbe::mock(SourceReport::Missing);
    assert_rejected_before_startup(
        custom_builder(sources.clone(), &owner),
        &missing,
        &owner,
        "did not report its bound product model sources",
    )
    .await;
    for role in ModelArtifactSourceRole::ALL {
        let mut reported = sources.resolved_sources().clone();
        let changed = match role {
            ModelArtifactSourceRole::Semantic => &mut reported.semantic,
            ModelArtifactSourceRole::Tokenizer => &mut reported.tokenizer,
            ModelArtifactSourceRole::Weights => &mut reported.weights,
        };
        changed.files[0].sha256 = "a".repeat(64);
        let (owner, probe) = BindingProbe::mock(SourceReport::Bound(reported));
        assert_rejected_before_startup(
            custom_builder(sources.clone(), &owner),
            &probe,
            &owner,
            "executor bound model sources differ",
        )
        .await;
    }
}

#[tokio::test]
async fn source_binding_legacy_custom_cannot_bypass_required_numerical_profile() {
    let (_dir, sources) = unregistered_sources();
    let (owner, probe) =
        BindingProbe::mock(SourceReport::Bound(sources.resolved_sources().clone()));
    let mut builder = custom_builder(sources, &owner);
    builder.config.numerical_execution = "fixture.explicit-profile".parse().unwrap();
    assert_rejected_before_startup(
        builder,
        &probe,
        &owner,
        "requires a registered vNext model package",
    )
    .await;
}

fn defined_fixture(context: usize) -> (FixtureDir, Arc<DefinedProductionModel>) {
    let dir = FixtureDir(crate::registry::tests::write_qwen35_fixture_model_dir());
    let mut semantic: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.0.join("config.json")).unwrap()).unwrap();
    semantic["max_position_embeddings"] = context.into();
    semantic["text_config"]["max_position_embeddings"] = context.into();
    std::fs::write(
        dir.0.join("config.json"),
        serde_json::to_vec(&semantic).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.0.join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
    std::fs::write(dir.0.join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
    let defined = ferrum_models::vnext::qwen35::define_from_model_dir(&dir.0).unwrap();
    (dir, Arc::new(defined))
}

fn typed_executor(defined: &DefinedProductionModel) -> (EngineConfig, ExecutorOwner) {
    use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;

    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    config.backend.enable_reusable_execution = false;
    config.scheduler.max_running_requests = 4;
    config.batching.max_num_batched_tokens = 8;
    let composition = CpuVNextComposition::create(
        DeviceId::new(format!(
            "device.cpu.source-binding-{}",
            uuid::Uuid::new_v4()
        ))
        .unwrap(),
        16 * 1024 * 1024,
    )
    .unwrap();
    let (runtime, operations, materializers, materializer, catalog) =
        composition.into_parts().unwrap();
    let executor = crate::product_composition::create_program_executor(
        &config,
        defined,
        runtime,
        operations,
        materializers,
        catalog,
        |_| Ok(WeightMaterializerSelection::exact(materializer.clone())),
        None,
    )
    .unwrap();
    (config, ExecutorOwner::new(Box::new(executor)))
}

#[tokio::test]
async fn source_binding_typed_delegation_accepts_frozen_sources_and_prepared_profile() {
    let (dir, defined) = defined_fixture(16);
    let (config, executor) = typed_executor(&defined);
    let (owner, probe) = BindingProbe::new(executor, SourceReport::Delegate);
    // Validation consumes the retained source bytes, never a later semantic
    // reread that could choose another architecture or definition.
    std::fs::write(
        dir.0.join("config.json"),
        br#"{"architectures":["ChangedAfterConstruction"]}"#,
    )
    .unwrap();
    let engine = EngineBuilder::new(config)
        .with_defined_model(defined)
        .with_custom_executor(owner.scope().unwrap())
        .build()
        .await
        .unwrap();
    assert_eq!(probe.startup_calls.load(Ordering::Relaxed), 1);
    engine.shutdown().await.unwrap();
    owner.begin_shutdown().wait().await.result().unwrap();
}

#[tokio::test]
async fn source_binding_typed_plan_cannot_hide_other_sources_or_definition() {
    let (_dir, defined) = defined_fixture(16);
    let (_other_dir, other_defined) = defined_fixture(32);
    let (config, executor) = typed_executor(&defined);
    {
        let scope = executor.scope().unwrap();
        let plan = scope.resolved_model_plan().unwrap();
        validate_defined_family(&defined, plan).unwrap();
        let error = validate_defined_family(&other_defined, plan).unwrap_err();
        assert!(error.to_string().contains("prepared family differs"));
    }

    // Matching wrapper metadata does not replace the actual typed plan's
    // bound source identity.
    let (owner, probe) = BindingProbe::new(
        executor,
        SourceReport::Bound(other_defined.sources().resolved_sources().clone()),
    );
    assert_rejected_before_startup(
        EngineBuilder::new(config)
            .with_defined_model(other_defined)
            .with_custom_executor(owner.scope().unwrap()),
        &probe,
        &owner,
        "resolved plan sources differ",
    )
    .await;
}
