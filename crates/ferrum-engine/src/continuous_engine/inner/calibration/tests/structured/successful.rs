//! Real manual session, controlled CPU producer, output owner and FIFO worker.
//! This is protocol correctness, not native-kernel or performance evidence.
use super::*;
use crate::continuous_engine::inner::cost_observation::EngineCostRuntime;
use crate::continuous_engine::inner::slo_controller::tests::fixture::fixture_with_custom_config;
use ferrum_interfaces::execution_cost::*;
use ferrum_interfaces::output_flow::OutputCompletion;
use ferrum_scheduler::implementations::continuous::cost_model::structured::StructuredInputV1;
use sha2::{Digest, Sha256};
use std::sync::atomic::AtomicU64;

const WARMUP_REQUESTS: usize = 1;
const PHASE_REQUESTS: [usize; 3] = [4, 4, 2];
const OUTPUTS_PER_REQUEST: usize = 5;
// The existing controlled fixture binds each distinct request permanently to
// its own real CoreEvidence owner. Allocate the complete declared population;
// do not erase mappings or claim this test proves owner-index reuse.
const OWNER_SLOTS: usize =
    WARMUP_REQUESTS + PHASE_REQUESTS[0] + PHASE_REQUESTS[1] + PHASE_REQUESTS[2];

struct AdvancingCostClock(AtomicU64);
impl CostObservationClock for AdvancingCostClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.fetch_add(1, Ordering::Relaxed))
    }
}
async fn observed_session() -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut engine, _, executor) = fixture_with_custom_config(OWNER_SLOTS, |config| {
        config.scheduler.slo.cost_observation.structured_capture =
            ferrum_types::SloStructuredCostCapture::HostSettledV1;
    })
    .await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::build(
            identity,
            Arc::new(AdvancingCostClock(AtomicU64::new(100))),
            &inner.config.scheduler.slo.cost_observation,
            true,
        )
        .unwrap(),
    ));
    inner.bg_loop_spawned.store(false, Ordering::Release);
    let session = CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::MIN).unwrap(),
    )
    .unwrap();
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    executor
        .emit_structured_cost_observations
        .store(true, Ordering::Release);
    executor
        .completion_work_known
        .store(true, Ordering::Release);
    (session, executor)
}

/// Complete the declared output policy. No owner is cancelled to create cuts.
async fn completed_request(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
) -> Vec<CalibrationWaveReport> {
    let mut request =
        ferrum_types::InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.max_tokens = OUTPUTS_PER_REQUEST;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let mut output = session
        .add_request(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(session, &id, false).await;
    admit(session).await;
    let mut reports = Vec::new();
    for generated in 1..=OUTPUTS_PER_REQUEST {
        let current = frontier(session, &id);
        let work = if generated == 1 {
            current.prefill_work(NonZeroU32::MIN).unwrap()
        } else {
            current
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        };
        let report = wave(session, executor, vec![work]).await;
        assert!(report.error.is_none(), "{:?}", report.error);
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        let stages = report.host_stages.as_ref().unwrap();
        let qualified = stages
            .structured_evidence
            .as_ref()
            .expect("actual structured producer")
            .as_ref()
            .unwrap_or_else(|error| panic!("structured settlement: {error:?}; {stages:?}"));
        qualified.validate_host_stages(stages).unwrap();
        qualified
            .recipe()
            .algorithm_work()
            .unwrap()
            .validate_structure(qualified.recipe())
            .unwrap();
        let frame = bounded(output.frames.next()).await.unwrap();
        assert_eq!(frame.metadata().generated_tokens, generated);
        assert!(!frame.metadata().terminal);
        drop(frame);
        reports.push(report);
        if generated < OUTPUTS_PER_REQUEST {
            ready(session, &id, false).await;
        }
    }
    let terminal = bounded(output.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    assert_eq!(terminal.metadata().generated_tokens, OUTPUTS_PER_REQUEST);
    assert!(terminal.metadata().token.is_none());
    drop(terminal);
    let completion = bounded(output.completion).await.unwrap();
    assert!(matches!(
        completion.payload(),
        OutputCompletion::Succeeded {
            reason: ferrum_types::FinishReason::Length,
            ..
        }
    ));
    assert!(session.frontiers().unwrap().is_empty());
    reports
}

/// Independent discovery reads the real qualified shape. It does not forge a
/// numeric observation or bypass the collector's later source/session binding.
fn discovery_input(report: &CalibrationWaveReport) -> StructuredInputV1 {
    let stages = report.host_stages.as_ref().unwrap();
    let shape = stages.actual_shape.as_ref().unwrap();
    let selected = stages.statistical_evidence.as_ref().unwrap();
    let recipe = stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .recipe();
    let exact = CanonicalWaveCostShape {
        kind: ActualWaveKind::Decode,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: shape.provider_signature,
        output_policy_signature: shape.output_policy_signature,
        numeric_features: shape.numeric_features.clone(),
        host_content_features: shape.host_content_features,
        row_multiset_features: shape.row_multiset_features.clone(),
        rows: stages
            .rows
            .iter()
            .map(|row| match row.actual_work {
                HostStageWork::Decode { kv_tokens } => ActualRowWork::Decode { kv_tokens },
                _ => panic!("discovery scope must be actual decode"),
            })
            .collect(),
        recurrent_state_bytes: shape.recurrent_state_bytes,
    };
    StructuredInputV1::from_future(&exact, selected, recipe).unwrap()
}

#[tokio::test]
async fn structured_collector_real_driver_completes_three_frozen_populations_and_raw_recipe() {
    let (mut session, executor) = observed_session().await;
    let path = SourcePath::new();
    // Declared before executing discovery or warmup; fixed scope candidate and
    // output policy are repeated, never selected from qualification outcomes.
    let protocol: [u8;32]=Sha256::digest(b"controlled-cpu/full-logits/plain-text/rows1/max5;warmup=one-complete-request;fit=4;residual=4;qualification=2").into();
    let warmup = completed_request(&mut session, &executor).await;
    let independent = discovery_input(&warmup[1]);
    let mut config = options(&path.0);
    config.protocol_sha256 = protocol;
    config.scope.domain_signature = *independent.domain_signature();
    let members = PHASE_REQUESTS
        .map(|requests| NonZeroUsize::new(requests * (OUTPUTS_PER_REQUEST - 1)).unwrap());
    config.fit_members = members[0];
    config.residual_members = members[1];
    config.qualification_members = members[2];
    config.maximum_file_bytes = NonZeroU64::new(4 << 20).unwrap();
    // Declared empirical margin for this protocol-correctness fixture. The
    // virtual cost clock never changes request clocks or any captured receipt.
    config.settings.static_margin_ns = 100_000;
    bounded(session.begin_structured_cost_calibration(config))
        .await
        .unwrap();
    let mut receipts = Vec::new();
    for (phase, requests) in [
        (StructuredCapturePhase::Fit, PHASE_REQUESTS[0]),
        (StructuredCapturePhase::Residual, PHASE_REQUESTS[1]),
        (StructuredCapturePhase::Qualification, PHASE_REQUESTS[2]),
    ] {
        for _ in 0..requests {
            completed_request(&mut session, &executor).await;
        }
        let progress = session.structured_cost_progress().unwrap();
        assert_eq!(progress.phase, phase);
        assert_eq!(progress.failed_members, [0, 0, 0]);
        receipts.push(
            bounded(session.freeze_structured_cost_phase())
                .await
                .unwrap(),
        );
    }
    let artifact = bounded(session.finish_structured_cost_calibration())
        .await
        .unwrap();
    assert_eq!(artifact.phase, StructuredCapturePhase::Qualified);
    assert!(artifact.model.is_some());
    assert_eq!(artifact.scope_members, 40);
    assert_eq!(artifact.scope_failures, 0);
    assert!(artifact.failure.is_none());
    let bytes = std::fs::read(&path.0).unwrap();
    assert_eq!(
        artifact.source_sha256,
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    for (index, receipt) in receipts.iter().enumerate() {
        assert_eq!(receipt.member_cutoff, [16, 32, 40][index]);
        assert_eq!(
            receipt.source_prefix_sha256,
            <[u8; 32]>::from(Sha256::digest(
                &bytes[..receipt.source_prefix_bytes as usize]
            ))
        );
        if index > 0 {
            assert!(receipt.frozen_at_ns >= receipts[index - 1].frozen_at_ns);
        }
    }
    let records: Vec<serde_json::Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["source_record_ordinal"], (index + 1) as u64);
    }
    let producer = &records[0]["record"]["producer"];
    assert!(producer["executable_bytes"].as_u64().unwrap() > 0);
    assert!(!producer["executable_sha256"].as_str().unwrap().is_empty());
    let expected_algorithm = serde_json::to_value(
        SelectedAlgorithmClassV1::new("fixture.controlled.full_logits_fill", 1, [1; 32], [2; 32])
            .unwrap(),
    )
    .unwrap();
    let mut terminals = 0;
    let mut nonterminals = 0;
    for record in records
        .iter()
        .map(|record| &record["record"])
        .filter(|record| record["kind"] == "completed" && !record["member"].is_null())
    {
        let recipe = &record["host_stages"]["structured_evidence"]["Ok"]["recipe"];
        assert_eq!(recipe, &record["selected_structured_capture"]["Ok"]);
        assert_eq!(recipe["device"]["physical_commands"], 1);
        assert_eq!(
            recipe["device"]["algorithm_work"]["Ok"]["entries"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            recipe["device"]["algorithm_work"]["Ok"]["entries"][0]["algorithm"],
            expected_algorithm
        );
        assert_eq!(
            recipe["device"]["algorithm_work"]["Ok"]["entries"][0]["work"]["logical_units"],
            64
        );
        match recipe["physical_host_rows"][0]["terminal_expectation"]
            .as_str()
            .unwrap()
        {
            "length_boundary" => terminals += 1,
            "token_may_terminate" => nonterminals += 1,
            other => panic!("unexpected scope {other}"),
        }
        let phase_start = match record["phase"].as_str().unwrap() {
            "fit" => records[0]["record"]["opened_at_ns"].as_u64().unwrap(),
            "residual" => receipts[0].frozen_at_ns,
            "qualification" => receipts[1].frozen_at_ns,
            other => panic!("unexpected phase {other}"),
        };
        assert!(record["numeric"]["observed_at_ns"].as_u64().unwrap() >= phase_start);
    }
    assert_eq!((terminals, nonterminals), (10, 30));
    assert_eq!(
        executor.completion_calls.load(Ordering::Acquire),
        OWNER_SLOTS
    );
    session.shutdown().await.unwrap();
}
