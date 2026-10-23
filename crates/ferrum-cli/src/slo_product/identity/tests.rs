use super::*;
use ferrum_execution::model_executor::{ExecutorCapabilities, ExecutorStatus};
use ferrum_models::executor::StubModelExecutor;
use ferrum_runtime::{FileFingerprint, ResolvedModelSource, ResolvedModelSources};

struct LoadedExecutor {
    stub: StubModelExecutor,
    info: ferrum_types::ModelInfo,
    sources: ResolvedModelSources,
    implementation: Option<&'static str>,
}

impl LoadedExecutor {
    fn new() -> Self {
        let stub = StubModelExecutor::new("actual-loaded-model", 128);
        let source = ResolvedModelSource {
            canonical_location: "fixture:immutable-model".into(),
            resolved_revision: "revision-a".into(),
            files: vec![FileFingerprint {
                relative_path: "config.json".into(),
                size_bytes: 12,
                sha256: "a".repeat(64),
            }],
        };
        Self {
            info: stub.info().clone(),
            stub,
            sources: ResolvedModelSources {
                semantic: source.clone(),
                tokenizer: source.clone(),
                weights: source,
            },
            implementation: Some("fixture:concrete-executor"),
        }
    }
}

#[async_trait::async_trait]
impl ModelExecutor for LoadedExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        &self.info
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.stub.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.stub.status()
    }
    fn implementation_identity(&self) -> Option<&'static str> {
        self.implementation
    }
    fn bound_resolved_sources(&self) -> Option<&ResolvedModelSources> {
        Some(&self.sources)
    }
}

fn binding(executor: &LoadedExecutor, physical: &str, executable: &str) -> BoundIdentity {
    bind_with_evidence(
        &EngineConfig::default(),
        executor,
        Ok(json!({"physical_device": physical})),
        Ok(executable.into()),
    )
    .unwrap()
}

#[test]
fn actual_model_source_device_and_implementation_evidence_control_reuse() {
    let mut executor = LoadedExecutor::new();
    let initial = binding(&executor, "physical-device-a", "binary-a");
    assert!(matches!(initial.scope, BindingScope::Reusable));
    assert_eq!(
        initial.identity,
        binding(&executor, "physical-device-a", "binary-a").identity
    );
    assert_ne!(
        initial.identity,
        binding(&executor, "physical-device-b", "binary-a").identity
    );
    assert_ne!(
        initial.identity,
        binding(&executor, "physical-device-a", "binary-b").identity
    );
    executor.info.dtype = ferrum_types::DataType::FP32;
    assert_ne!(
        initial.identity,
        binding(&executor, "physical-device-a", "binary-a").identity
    );
    executor.info.dtype = ferrum_types::DataType::FP16;
    executor.sources.weights.files[0].sha256 = "b".repeat(64);
    assert_ne!(
        initial.identity,
        binding(&executor, "physical-device-a", "binary-a").identity
    );
    let encoded = serde_json::to_value(&initial.identity).unwrap();
    let fields: Value = serde_json::from_str(encoded.as_str().unwrap()).unwrap();
    assert_eq!(fields["actual_model"]["model_id"], "actual-loaded-model");
    assert_eq!(fields["physical"]["physical_device"], "physical-device-a");
}

#[test]
fn incomplete_hardware_or_executor_evidence_never_becomes_a_reusable_key() {
    let mut executor = LoadedExecutor::new();
    let incomplete = bind_with_evidence(
        &EngineConfig::default(),
        &executor,
        Err("driver identity unavailable".into()),
        Ok("binary-a".into()),
    )
    .unwrap();
    assert!(matches!(
        incomplete.scope,
        BindingScope::InstanceScoped { .. }
    ));
    executor.implementation = None;
    let first = binding(&executor, "physical-device-a", "binary-a");
    let second = binding(&executor, "physical-device-a", "binary-a");
    assert!(matches!(first.scope, BindingScope::InstanceScoped { .. }));
    assert_ne!(first.identity, second.identity);
}

#[test]
fn canonical_config_map_order_does_not_invalidate_same_execution() {
    let executor = LoadedExecutor::new();
    let mut first = EngineConfig::default();
    first
        .backend
        .backend_options
        .insert("first".into(), 1.into());
    first
        .backend
        .backend_options
        .insert("second".into(), 2.into());
    let mut second = EngineConfig::default();
    second
        .backend
        .backend_options
        .insert("second".into(), 2.into());
    second
        .backend
        .backend_options
        .insert("first".into(), 1.into());
    let get = |config: &EngineConfig| {
        bind_with_evidence(
            config,
            &executor,
            Ok(json!({"device": "same"})),
            Ok("binary".into()),
        )
        .unwrap()
        .identity
    };
    assert_eq!(get(&first), get(&second));
    first.runtime.profile_jsonl = Some("first-output.jsonl".into());
    second.runtime.profile_jsonl = Some("second-output.jsonl".into());
    assert_eq!(get(&first), get(&second));
    second.runtime.profile_jsonl = None;
    assert_ne!(get(&first), get(&second));
    second.runtime.profile_jsonl = first.runtime.profile_jsonl.clone();
    second.batching.max_num_batched_tokens += 1;
    assert_ne!(get(&first), get(&second));
}
