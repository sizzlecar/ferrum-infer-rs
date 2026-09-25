//! Real product CUDA lifecycle, using the same two-layer recurrent/causal
//! model and resolver as the Metal product tests. Resource preparation is not
//! graph preparation, and neither is evidence that future costs are Known.
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_weight_materializer_selection, CudaVNextComposition},
    vnext_runtime::CudaDeviceRuntime,
};
use ferrum_types::{KvStorageFormat, ModelId};

#[path = "../cost_observation/product_tests/resolver.rs"]
mod resolver;
#[path = "../cost_observation/product_tests/weights.rs"]
mod weights;

const CONTEXT: usize = 512;
const WIDTHS: [usize; 4] = [1, 2, 4, 8];

// CUDA reports capture/replay counts through the opt-in host dispatch sink.
// This enables those real counters without device events or frame recording.
struct HostCounters;

impl ExecutionEventSink for HostCounters {
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::None
    }

    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        false
    }

    fn host_dispatch_timing_enabled(&self) -> bool {
        true
    }

    fn record(&self, _: EventEmissionPermit) -> std::result::Result<(), ExecutionEventSinkError> {
        panic!("host counters must not record execution frames")
    }
}

struct Fixture {
    executor: VNextModelExecutor<CudaDeviceRuntime>,
    _directory: tempfile::TempDir,
}

impl Fixture {
    fn observe_host_counters(&self) {
        self.executor
            .attach_execution_event_sink(Arc::new(HostCounters));
        assert!(self.executor.host_dispatch_timing_enabled());
        assert_eq!(self.executor.device_timing_mode(), DeviceTimingMode::Off);
    }

    async fn new(
        workspace: WorkspacePreparationMode,
        programs: ReusableExecutionPreparationMode,
    ) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let geometry = weights::CausalGeometry {
            context: CONTEXT,
            ..weights::CausalGeometry::TINY
        };
        weights::write_config(directory.path(), geometry);
        weights::write_weights(directory.path(), geometry);
        std::fs::write(directory.path().join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
        std::fs::write(directory.path().join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
        let defined = crate::vnext::qwen35::define_from_model_dir(directory.path()).unwrap();
        let mut engine = EngineConfig::default();
        engine.backend.device = Device::CUDA(0);
        engine.backend.enable_reusable_execution = true;
        engine.backend.workspace_preparation = workspace;
        engine.backend.reusable_execution_capture = ReusableExecutionCaptureConfig {
            preparation: programs,
            exact_decode_widths: Some(WIDTHS.to_vec()),
            ..Default::default()
        };
        engine.runtime.prefix_state_cache_enabled = false;
        engine.runtime.attention_execution_policy = AttentionExecutionPolicy::Portable;
        engine.scheduler.max_running_requests = *WIDTHS.last().unwrap();
        engine.scheduler.prefill_step_chunk = Some(8);
        engine.batching.max_num_batched_tokens = 8;
        // Fixed for all arms. This accommodates static staging plus the real
        // lane/program bindings; failures must not reduce shapes or capacity.
        engine.memory.usable_capacity_bytes = Some(256 << 20);
        let profiles = defined
            .definition()
            .numerical_profiles()
            .candidates(&engine.numerical_execution, KvStorageFormat::F16)
            .unwrap();
        let prepared = defined.prepare(&profiles[0].id).unwrap();
        let selection = cuda_weight_materializer_selection(prepared.family()).unwrap();
        let (runtime, operations, materializers, catalog) = CudaVNextComposition::create(
            0,
            DeviceId::new(format!("device.workspace-startup.{}", uuid::Uuid::new_v4())).unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap()
        .into_parts();
        let info = prepared.model_info(ModelId::new("workspace-startup-product"), Device::CUDA(0));
        let config =
            VNextExecutorConfig::from_engine_config(&engine, &info, runtime.as_ref()).unwrap();
        assert_eq!(config.maximum_model_tokens, CONTEXT);
        let composition = VNextRuntimeComposition::new(runtime, operations, materializers, catalog);
        let compiled = composition
            .compile_model(&prepared, info, &engine, config, selection)
            .unwrap();
        let executor = compiled
            .initialize(|prepared, runtime, catalog, compilation| {
                let numerical = NumericalProfileResolution::from_static_plan(
                    engine.numerical_execution.clone(),
                    KvStorageFormat::F16,
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
        assert_eq!(executor.device_timing_mode(), DeviceTimingMode::Off);
        Self {
            executor,
            _directory: directory,
        }
    }

    fn report(&self) -> VNextReusableExecutionStartupReport {
        let state = self.executor.startup_preparation.lock();
        let VNextStartupPreparationState::Ready { report } = &*state else {
            panic!("real product startup did not reach Ready")
        };
        report.clone()
    }

    fn assert_capacity_and_no_owners(&self) {
        assert_eq!(self.executor.sequences.lock().total_len(), 0);
        let status = self.executor.plan_resources.dynamic_pool_status().unwrap();
        assert!(status.process_claimed_bytes() <= status.effective_device_usable_ceiling_bytes());
        for pool in status.pools() {
            assert!(pool.free_bytes() <= pool.resident_bytes());
        }
        // This performs the real quiescent-lane check, not a cached flag.
        self.executor
            .lane
            .reusable_executable_preparation()
            .unwrap();
    }

    fn assert_resource_report(&self, report: &VNextReusableExecutionStartupReport) {
        let resources = report.workspace_preparation.as_ref().unwrap();
        assert_eq!(resources.mode, WorkspacePreparationMode::Startup);
        assert_eq!(resources.encoded_model_waves, 0);
        assert_eq!(resources.submitted_model_waves, 0);
        assert_eq!(resources.maximum_simultaneous_startup_owners, 8);
        assert!(
            resources.after.process_claimed_bytes
                <= resources.after.effective_device_usable_ceiling_bytes
        );
        let memory = self
            .executor
            .resolved_plan
            .execution_plan()
            .payload()
            .memory();
        let expected = memory
            .reusable_execution()
            .unwrap()
            .buckets()
            .iter()
            .map(|bucket| bucket.bucket().bucket_id().clone())
            .collect::<BTreeSet<_>>();
        let actual = resources
            .prepared_buckets
            .iter()
            .map(|bucket| bucket.bucket.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(resources.prepared_buckets.len(), actual.len());
        assert_eq!(
            actual, expected,
            "every declared bucket must prepare; no shrinking"
        );
    }

    async fn assert_idempotent(&self) {
        let report = serde_json::to_value(self.report()).unwrap();
        let before = self.executor.plan_resources.dynamic_pool_status().unwrap();
        let preparation = self
            .executor
            .lane
            .reusable_executable_preparation()
            .unwrap();
        self.executor.prepare_startup().await.unwrap();
        let after = self.executor.plan_resources.dynamic_pool_status().unwrap();
        assert_eq!(before.epochs(), after.epochs());
        assert_eq!(
            before.process_claimed_bytes(),
            after.process_claimed_bytes()
        );
        assert_eq!(
            preparation,
            self.executor
                .lane
                .reusable_executable_preparation()
                .unwrap()
        );
        assert_eq!(serde_json::to_value(self.report()).unwrap(), report);
    }

    fn admit(&self, input: &PlanRuntimePrefillInput) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(
                Instant::now() < deadline,
                "real backing maintenance made no progress"
            );
            let admission = ExecutorPrefillAdmission::for_product_request(
                &input.request_id,
                &input.input_tokens,
                input.maximum_sequence_tokens,
                input.input_tokens.len(),
                0,
            )
            .unwrap();
            match self.executor.try_admit_prefill(admission).unwrap() {
                ExecutorPrefillAdmissionDecision::Admitted(_) => return,
                ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => assert!(matches!(
                    self.executor
                        .maintain_prefill_backing(&input.request_id)
                        .unwrap(),
                    ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                )),
                other => panic!("real product admission failed: {other:?}"),
            }
        }
    }

    async fn prefill(&self, input: &PlanRuntimePrefillInput) -> PlanRuntimePrefillCompletion {
        self.admit(input);
        match self
            .executor
            .plan_runtime_prefill_with_capacity(input)
            .await
            .unwrap()
        {
            PlanRuntimePrefillOutcome::Completed(output) => output,
            PlanRuntimePrefillOutcome::Deferred(reason) => {
                panic!("real product prefill deferred: {reason:?}")
            }
        }
    }

    async fn decode(&self, inputs: &[PlanRuntimeDecodeInput]) -> Vec<PlanRuntimeDecodeOutput> {
        match self
            .executor
            .plan_runtime_batch_decode_with_capacity(inputs)
            .await
            .unwrap()
        {
            PlanRuntimeBatchDecodeOutcome::Completed(outputs) => outputs,
            PlanRuntimeBatchDecodeOutcome::Deferred(reason) => {
                panic!("real product decode deferred: {reason:?}")
            }
        }
    }

    fn release(&self, inputs: Vec<PlanRuntimeDecodeInput>) {
        for input in &inputs {
            self.executor.release_cache(&input.kv_cache.cache_id());
        }
        drop(inputs);
        self.assert_capacity_and_no_owners();
    }
}

fn prompt() -> PlanRuntimePrefillInput {
    let tokens = [0, 1, 2, 1, 0, 1, 2, 1].map(TokenId::new).to_vec();
    PlanRuntimePrefillInput::new(
        RequestId::new(),
        tokens,
        CONTEXT,
        PrefillChunk::new(0, 8, 8).unwrap(),
    )
    .unwrap()
}

fn assert_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    for (a, b) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5,
            "workspace preparation changed actual logits: {actual:?} != {expected:?}"
        );
    }
}

#[tokio::test]
#[ignore = "requires a CUDA device and the compiled native operator set"]
async fn workspace_startup_cuda_on_demand_prepares_resources_without_programs() {
    let fixture = Fixture::new(
        WorkspacePreparationMode::Startup,
        ReusableExecutionPreparationMode::OnDemand,
    )
    .await;
    let report = fixture.report();
    fixture.assert_resource_report(&report);
    assert!(report.enabled && report.supported);
    assert_eq!(report.eager_warmup_waves, 0);
    assert_eq!(report.capture_waves, 0);
    assert_eq!(report.replay_inventory_check_waves, 0);
    assert_eq!(report.synthetic_sequences, 0);
    assert_eq!(report.prepared_programs, 0);
    assert_eq!(
        fixture
            .executor
            .metrics
            .submitted_waves
            .load(Ordering::Acquire),
        0
    );
    assert_eq!(
        report.device_preparation.state(),
        DeviceReusableExecutionPreparationState::Ready
    );
    assert_eq!(report.device_preparation.resident_executables(), 0);
    assert_eq!(report.device_preparation.captured_executables(), 0);
    assert_eq!(report.device_preparation.uploaded_executables(), 0);
    assert!(fixture
        .executor
        .lane
        .reusable_execution_catalog()
        .unwrap()
        .programs()
        .is_empty());
    fixture.assert_capacity_and_no_owners();
    fixture.assert_idempotent().await;
}

#[tokio::test]
#[ignore = "requires a CUDA device and the compiled native operator set"]
async fn workspace_startup_cuda_matches_demand_driven_through_capture_replay_and_cleanup() {
    let actual = Fixture::new(
        WorkspacePreparationMode::Startup,
        ReusableExecutionPreparationMode::OnDemand,
    )
    .await;
    let expected = Fixture::new(
        WorkspacePreparationMode::DemandDriven,
        ReusableExecutionPreparationMode::OnDemand,
    )
    .await;
    assert!(expected.report().workspace_preparation.is_none());
    actual.observe_host_counters();
    expected.observe_host_counters();
    let mut actual_inputs = Vec::new();
    let mut expected_inputs = Vec::new();
    for _ in 0..4 {
        let a = prompt();
        let b = prompt();
        let a_output = actual.prefill(&a).await;
        let b_output = expected.prefill(&b).await;
        assert_eq!(a_output.completed_chunk(), b_output.completed_chunk());
        assert_eq!(
            a_output.output().committed_tokens(),
            b_output.output().committed_tokens()
        );
        let (
            PlanRuntimePrefillProduct::FinalLogits(a_logits),
            PlanRuntimePrefillProduct::FinalLogits(b_logits),
        ) = (a_output.output().product(), b_output.output().product())
        else {
            panic!("full product prefill did not return final logits")
        };
        assert_logits(a_logits, b_logits);
        actual_inputs.push(PlanRuntimeDecodeInput::new(
            a.request_id,
            TokenId::new(1),
            Arc::clone(a_output.output().kv_cache()),
        ));
        expected_inputs.push(PlanRuntimeDecodeInput::new(
            b.request_id,
            TokenId::new(1),
            Arc::clone(b_output.output().kv_cache()),
        ));
    }
    // Predeclared, bounded real decode history. Do not stop when a graph first
    // appears: validate outputs before, during, and after on-demand capture.
    for turn in 0..12 {
        let a = actual.decode(&actual_inputs).await;
        let b = expected.decode(&expected_inputs).await;
        assert_eq!(a.len(), actual_inputs.len());
        assert_eq!(b.len(), expected_inputs.len());
        for (((a, b), next_a), next_b) in a
            .into_iter()
            .zip(b)
            .zip(&mut actual_inputs)
            .zip(&mut expected_inputs)
        {
            assert_eq!(a.kv_cache.num_tokens(), b.kv_cache.num_tokens());
            let (
                ExecutorSamplingOutput::FullLogits(a_logits),
                ExecutorSamplingOutput::FullLogits(b_logits),
            ) = (&a.sampling_output, &b.sampling_output)
            else {
                panic!("ordinary decode changed product output mode")
            };
            assert_logits(a_logits, b_logits);
            next_a.kv_cache = a.kv_cache;
            next_b.kv_cache = b.kv_cache;
            next_a.input_token = TokenId::new((turn % 2) as u32);
            next_b.input_token = next_a.input_token;
        }
    }
    for fixture in [&actual, &expected] {
        let metrics = &fixture.executor.metrics.wave_timing.reusable_execution;
        assert!(
            metrics.captured_segments.load(Ordering::Acquire) > 0,
            "real on-demand capture did not happen"
        );
        assert!(
            metrics.replayed_segments.load(Ordering::Acquire) > 0,
            "real graph replay did not happen"
        );
        assert!(!fixture
            .executor
            .lane
            .reusable_execution_catalog()
            .unwrap()
            .programs()
            .is_empty());
    }
    actual.release(actual_inputs);
    expected.release(expected_inputs);
}

#[tokio::test]
#[ignore = "requires a CUDA device and the compiled native operator set"]
async fn workspace_startup_cuda_keeps_independent_startup_graph_lifecycle_usable() {
    let fixture = Fixture::new(
        WorkspacePreparationMode::Startup,
        ReusableExecutionPreparationMode::Startup,
    )
    .await;
    let report = fixture.report();
    fixture.assert_resource_report(&report);
    assert!(report.eager_warmup_waves > 0 && report.capture_waves > 0);
    assert!(report.replay_inventory_check_waves > 0 && report.prepared_programs > 0);
    assert_eq!(
        report.device_preparation.state(),
        DeviceReusableExecutionPreparationState::Ready
    );
    assert!(report.device_preparation.captured_executables() > 0);
    assert_eq!(
        report.device_preparation.captured_executables(),
        report.device_preparation.resident_executables()
    );
    assert_eq!(
        report.requested_decode_widths,
        report.prepared_decode_widths
    );
    assert_eq!(
        report
            .prepared_decode_widths
            .iter()
            .copied()
            .collect::<BTreeSet<_>>(),
        WIDTHS.into_iter().collect()
    );
    assert_eq!(
        report.requested_prefill_chunks,
        report.prepared_prefill_chunks
    );
    assert!(!fixture
        .executor
        .lane
        .reusable_execution_catalog()
        .unwrap()
        .programs()
        .is_empty());
    fixture.assert_capacity_and_no_owners();
    fixture.assert_idempotent().await;
    fixture.observe_host_counters();
    let mut inputs = Vec::new();
    for _ in 0..8 {
        let input = prompt();
        let output = fixture.prefill(&input).await;
        inputs.push(PlanRuntimeDecodeInput::new(
            input.request_id,
            TokenId::new(1),
            Arc::clone(output.output().kv_cache()),
        ));
    }
    // Each explicit prepared width is exercised with real active owners.
    for width in WIDTHS.into_iter().rev() {
        let outputs = fixture.decode(&inputs[..width]).await;
        assert_eq!(outputs.len(), width);
        for (input, output) in inputs[..width].iter_mut().zip(outputs) {
            let ExecutorSamplingOutput::FullLogits(logits) = output.sampling_output else {
                panic!("ordinary startup-graph decode changed product mode")
            };
            assert!(!logits.is_empty() && logits.iter().all(|value| value.is_finite()));
            input.kv_cache = output.kv_cache;
        }
    }
    // Startup resets product metrics. This count therefore proves replay by
    // the real requests above, not only by synthetic startup validation.
    assert!(
        fixture
            .executor
            .metrics
            .wave_timing
            .reusable_execution
            .replayed_segments
            .load(Ordering::Acquire)
            > 0
    );
    fixture.release(inputs);
}
