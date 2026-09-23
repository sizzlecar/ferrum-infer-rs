use super::*;
use crate::vnext::qwen35::define_from_model_dir;

type Runtime = MetalDeviceRuntime;
pub(super) struct Fixture {
    pub executor: VNextModelExecutor<Runtime>,
    _directory: tempfile::TempDir,
}
impl Fixture {
    pub async fn new(maximum_batch_tokens: usize, prefix: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        weights::write_config(directory.path());
        weights::write_weights(directory.path());
        std::fs::write(directory.path().join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
        std::fs::write(directory.path().join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
        let defined = define_from_model_dir(directory.path()).unwrap();
        let mut engine = EngineConfig::default();
        engine.backend.device = Device::Metal;
        engine.backend.enable_reusable_execution = false;
        engine.runtime.prefix_state_cache_enabled = prefix;
        // Keep the existing tiny product fixture capacity; the observed and
        // ordinary arms own separate executors and request lifecycles.
        engine.scheduler.max_running_requests = 8;
        engine.scheduler.prefill_step_chunk = Some(maximum_batch_tokens);
        engine.batching.max_num_batched_tokens = maximum_batch_tokens;
        engine.memory.usable_capacity_bytes = Some(64 << 20);
        let profiles = defined
            .definition()
            .numerical_profiles()
            .candidates(
                &engine.numerical_execution,
                ferrum_types::KvStorageFormat::F16,
            )
            .unwrap();
        let prepared = defined.prepare(&profiles[0].id).unwrap();
        let (runtime, operations, materializers, materializer, catalog) =
            MetalVNextComposition::create(
                DeviceId::new(format!("device.observed-product.{}", uuid::Uuid::new_v4())).unwrap(),
            )
            .unwrap()
            .into_parts();
        let info = prepared.model_info(ModelId::new("observed-product"), Device::Metal);
        let config =
            VNextExecutorConfig::from_engine_config(&engine, &info, runtime.as_ref()).unwrap();
        let composition = VNextRuntimeComposition::new(runtime, operations, materializers, catalog);
        let compiled = composition
            .compile_model(
                &prepared,
                info,
                &engine,
                config,
                WeightMaterializerSelection::exact(materializer),
            )
            .unwrap();
        let executor = compiled
            .initialize(|prepared, runtime, catalog, compilation| {
                let numerical = NumericalProfileResolution::from_static_plan(
                    engine.numerical_execution.clone(),
                    ferrum_types::KvStorageFormat::F16,
                    defined.definition(),
                    prepared.family(),
                    catalog,
                    runtime,
                    compilation.executable().execution_plan(),
                    Vec::new(),
                )
                .unwrap();
                resolver::resolve(prepared, runtime, catalog, compilation, numerical)
            })
            .unwrap();
        executor.prepare_startup().await.unwrap();
        Self {
            executor,
            _directory: directory,
        }
    }
    pub fn submissions(&self) -> u64 {
        self.executor
            .metrics
            .submitted_waves
            .load(Ordering::Relaxed)
    }
    pub fn admit(&self, input: &PlanRuntimePrefillInput) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(Instant::now() < deadline, "tiny admission made no progress");
            let admission = ExecutorPrefillAdmission::for_product_request(
                &input.request_id,
                &input.input_tokens,
                input.maximum_sequence_tokens,
                input.input_tokens.len(),
                0,
            )
            .unwrap();
            match self.executor.try_admit_prefill(admission).unwrap() {
                ExecutorPrefillAdmissionDecision::Admitted(_) => break,
                ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                    assert!(matches!(
                        self.executor
                            .maintain_prefill_backing(&input.request_id)
                            .unwrap(),
                        ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                            | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                    ));
                }
                other => panic!("fixture admission: {other:?}"),
            }
        }
    }
}

pub(super) fn prompt(tokens: &[u32], chunk: usize) -> PlanRuntimePrefillInput {
    PlanRuntimePrefillInput::new(
        RequestId::new(),
        tokens.iter().copied().map(TokenId::new).collect::<Vec<_>>(),
        16,
        PrefillChunk::new(0, chunk, tokens.len()).unwrap(),
    )
    .unwrap()
}
pub(super) fn assert_logits_same(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5),
        "actual {actual:?}, expected {expected:?}"
    );
}
pub(super) fn assert_prefill_same(
    actual: &PlanRuntimePrefillCompletion,
    expected: &PlanRuntimePrefillCompletion,
) {
    assert_eq!(actual.completed_chunk(), expected.completed_chunk());
    assert_eq!(
        actual.output().committed_tokens(),
        expected.output().committed_tokens()
    );
    match (actual.output().product(), expected.output().product()) {
        (PlanRuntimePrefillProduct::Intermediate, PlanRuntimePrefillProduct::Intermediate) => {}
        (PlanRuntimePrefillProduct::FinalLogits(a), PlanRuntimePrefillProduct::FinalLogits(b)) => {
            assert_logits_same(a, b)
        }
        _ => panic!("prefill product differs"),
    }
}
