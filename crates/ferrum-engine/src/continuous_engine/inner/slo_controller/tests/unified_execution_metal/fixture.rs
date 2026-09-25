//! Shared product constructors and one manual driver; no replacement executor.
use super::*;
use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
use ferrum_interfaces::{
    model_executor::ModelExecutor,
    vnext::{DeviceId, WeightMaterializerSelection},
};
use ferrum_kernels::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_tokenizer::HuggingFaceTokenizer;

pub(super) struct ModelDirectory(std::path::PathBuf);
impl Drop for ModelDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) async fn fixture() -> (CalibrationSession, ModelDirectory) {
    fixture_with_structured_capture(false).await
}
pub(super) async fn fixture_with_structured_capture(
    enabled: bool,
) -> (CalibrationSession, ModelDirectory) {
    let directory = ModelDirectory(
        std::env::temp_dir().join(format!("ferrum-unified-metal-{}", uuid::Uuid::new_v4())),
    );
    std::fs::create_dir(&directory.0).unwrap();
    weights::write_config(&directory.0, weights::CausalGeometry::TINY);
    weights::write_weights(&directory.0, weights::CausalGeometry::TINY);
    std::fs::write(directory.0.join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
    std::fs::write(directory.0.join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
    let mut raw = tokenizers::Tokenizer::from_file(directory.0.join("tokenizer.json")).unwrap();
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    let tokenizer = Arc::new(HuggingFaceTokenizer::new(raw).await.unwrap());
    let defined = ferrum_models::vnext::qwen35::define_from_model_dir(&directory.0).unwrap();
    let mut config = ferrum_types::EngineConfig::default();
    config.backend.device = ferrum_types::Device::Metal;
    config.backend.enable_reusable_execution = false;
    config.backend.workspace_preparation = ferrum_types::WorkspacePreparationMode::Startup;
    config.runtime.prefix_state_cache_enabled = false;
    config.scheduler.max_running_requests = 8;
    config.scheduler.prefill_step_chunk = Some(8);
    config.batching.max_num_batched_tokens = 8;
    config.memory.usable_capacity_bytes = Some(64 << 20);
    config.scheduler.slo.mode = ferrum_types::SloMode::Observe;
    if enabled {
        config.scheduler.slo.cost_observation.structured_capture =
            ferrum_types::SloStructuredCostCapture::HostSettledV1;
    }
    config.scheduler.slo.default_service_class = Some("native-transition-test".into());
    config
        .scheduler
        .slo
        .services
        .push(ferrum_types::ServiceSloConfig {
            id: "native-transition-test".into(),
            server_token_commit: ferrum_types::SloLatencyBudgets {
                ttft_ms: NonZeroU64::new(30_000).unwrap(),
                tpot_ms: NonZeroU64::new(30_000).unwrap(),
                itl_ms: NonZeroU64::new(30_000).unwrap(),
            },
            client_visible: None,
            attainment: Default::default(),
        });
    // Functional boundary tests do not impose a hardware performance claim.
    config.scheduler.slo.planner.max_planning_us = NonZeroU64::new(30_000_000).unwrap();
    let (runtime, operations, materializers, materializer, catalog) =
        MetalVNextComposition::create(
            DeviceId::new(format!(
                "device.unified-transition.{}",
                uuid::Uuid::new_v4()
            ))
            .unwrap(),
        )
        .unwrap()
        .into_parts();
    let executor = crate::product_composition::create_vnext_executor(
        &config,
        &defined,
        runtime,
        operations,
        materializers,
        catalog,
        |_| Ok(WeightMaterializerSelection::exact(materializer.clone())),
    )
    .unwrap();
    executor.prepare_startup().await.unwrap();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let mut engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        Arc::new(executor),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    // This shared typed reference fixture supplies fixed N=4 work scores only.
    // It is not a measured Metal reference artifact or a latency qualification.
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .prefill_reference_runtime = Some(
        crate::continuous_engine::inner::prefill_reference_runtime::test_calibration_runtime(),
    );
    let session = CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::new(8).unwrap()).unwrap(),
    )
    .unwrap();
    (session, directory)
}

pub(super) async fn ready(inner: &EngineInner, id: &RequestId) {
    let mut changed = inner.sequences.read()[id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .subscribe();
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if inner.sequences.read()[id]
                .credited_output
                .as_ref()
                .unwrap()
                .port
                .readiness()
                == OutputReadinessState::Ready
            {
                return;
            }
            changed.changed().await.unwrap();
        }
    })
    .await
    .expect("actual output actor did not become ready");
}

pub(super) async fn add(
    session: &mut CalibrationSession,
) -> (RequestId, tokio::task::JoinHandle<()>) {
    let mut request = ferrum_types::InferenceRequest::new(
        "hello hello hello hello",
        session.configuration().model.model_id.clone(),
    );
    request.stream = true;
    // Five permits the extra real mixed workspace preparation wave before
    // the parity wave, keeping both nonterminal. This is a tiny test request,
    // not a change to product output limits or benchmark inputs.
    request.sampling_params.max_tokens = 5;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    let id = request.id.clone();
    let mut output = session
        .add_request(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    // Drain leases without assuming a generated token produces visible text.
    let drain = tokio::spawn(async move {
        while let Some(frame) = output.frames.next().await {
            drop(frame);
        }
        let _ = output.completion.await;
    });
    let inner = session.test_engine_inner();
    ready(&inner, &id).await;
    assert_eq!(inner.sequences.read()[&id].input_tokens.len(), 4);
    (id, drain)
}

pub(super) fn frontier(session: &CalibrationSession, id: &RequestId) -> CalibrationFrontier {
    session
        .frontiers()
        .unwrap()
        .into_iter()
        .find(|f| f.request_id() == id)
        .unwrap()
}

pub(super) async fn admit(session: &mut CalibrationSession, id: &RequestId) {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match session.step(CalibrationAction::AdmitOne).await.unwrap() {
                CalibrationTurn::AdmittedOrMaintained => {}
                CalibrationTurn::Blocked(CalibrationBlockReason::AdmissionUnavailable) => {}
                other => panic!("unexpected actual admission turn: {other:?}"),
            }
            let inner = session.test_engine_inner();
            let mut available = Vec::new();
            let epochs = inner
                .model_executor
                .write_execution_capacity_snapshot(&mut available)
                .unwrap()
                .unwrap();
            let queue = inner
                .scheduler
                .planning_state(
                    NonZeroUsize::new(8).unwrap(),
                    AdmissionWakeSnapshot::new(
                        AdmissionWakeEpochs::new(
                            epochs.coordinator_id,
                            epochs.release_epoch,
                            epochs.capacity_epoch,
                            0,
                        ),
                        &available,
                    ),
                )
                .unwrap();
            if queue
                .requests()
                .iter()
                .any(|r| r.key.request_id == *id && r.queue == PlanningQueueKind::Prefill)
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("actual admission did not reconcile");
}

pub(super) async fn wave(
    session: &mut CalibrationSession,
    mut rows: Vec<CalibrationWork>,
) -> CalibrationWaveReport {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match session
                .step(CalibrationAction::Wave(rows.clone()))
                .await
                .unwrap()
            {
                CalibrationTurn::Wave(report)
                    if report.submission == CalibrationSubmissionState::NotSubmitted =>
                {
                    assert!(report.error.is_none(), "{report:?}");
                    assert!(matches!(
                        &report.observation,
                        CalibrationObservation::Rejected { reason } if reason == "NoPhysicalWave"
                    ));
                    // A cold workspace can require a real retained capacity
                    // continuation. Neither the unsubmitted attempt nor that
                    // maintenance may perform the selected model work.
                    for row in &rows {
                        let before = row.frontier();
                        let after = frontier(session, before.request_id());
                        assert_eq!(after.owner_incarnation(), before.owner_incarnation());
                        assert_eq!(after.work_generation(), before.work_generation());
                        assert_eq!(after.generated_tokens(), before.generated_tokens());
                        assert_eq!(after.prefill_progress(), before.prefill_progress());
                        assert_eq!(after.kv_tokens(), before.kv_tokens());
                    }
                    assert!(
                        matches!(
                            session.step(CalibrationAction::Maintenance).await.unwrap(),
                            CalibrationTurn::MaintenanceReconciled
                        ),
                        "unsubmitted preparation has no real maintenance continuation: {report:?}"
                    );
                    for row in &mut rows {
                        let before = row.frontier();
                        let after = frontier(session, before.request_id());
                        assert_eq!(after.owner_incarnation(), before.owner_incarnation());
                        assert_eq!(after.generated_tokens(), before.generated_tokens());
                        assert_eq!(after.prefill_progress(), before.prefill_progress());
                        assert_eq!(after.kv_tokens(), before.kv_tokens());
                        row.frontier = after;
                    }
                    tokio::task::yield_now().await;
                }
                CalibrationTurn::Wave(report) => {
                    assert_eq!(
                        report.submission,
                        CalibrationSubmissionState::HostReconciled,
                        "actual native submission: {report:?}"
                    );
                    assert!(
                        report.error.is_none(),
                        "actual native wave: {:?}",
                        report.error
                    );
                    return report;
                }
                CalibrationTurn::Blocked(CalibrationBlockReason::ResourceUnavailable(
                    ResourcePlanningUnknown::ReadUnavailable(_)
                    | ResourcePlanningUnknown::BusyOrUnavailable,
                )) => tokio::time::sleep(Duration::from_millis(1)).await,
                other => panic!("exact native wave was not executed: {other:?}"),
            }
        }
    })
    .await
    .expect("actual native wave did not reconcile")
}
