//! Real ingress, scheduler and output owners. The controlled backend tests
//! lifecycle publication, not native numerical correctness or measured cost.
use super::super::tests::{
    bounded,
    fixture::{cleanup, fixture_with_width, ControlledExecutor},
    prefill, ready,
};
use super::*;
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    execution_cost::CompletionOnlyReason,
    output_flow::OutputProjectionContract,
    InferenceRequestContext,
};
use ferrum_types::{InferenceRequest, SloMode};
use futures::{FutureExt, StreamExt};
use std::{sync::atomic::Ordering, time::Duration};

async fn pending(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    id: &RequestId,
) -> PendingTimeWitness {
    // Explicit finite receipt fixture: no synthetic observation is trained or
    // exported, and this helper is never used as proof of planner feasibility.
    let captured = prefill::captured(engine, executor).await;
    let sequences = engine.inner.sequences.read();
    let sequence = &sequences[id];
    let state = sequence.time_admission.as_ref().unwrap();
    PendingTimeWitness {
        request_id: id.clone(),
        owner: sequence.stream_projection_identity.clone(),
        work_generation: sequence.cost_frontier.unwrap().work_generation.get(),
        ingress: state.ingress,
        original_input_tokens: state.original_input_tokens,
        maximum_output_tokens: state.maximum_output_tokens,
        evidence: StartedTimeWitness {
            snapshot_generation: captured.snapshot.generation,
            model_version: captured.snapshot.cost_model_version,
            validated_through: captured
                .origin
                .instant_at_ns(captured.snapshot.scope.horizon_end_ns)
                .unwrap(),
            obligations_beyond_horizon: captured.snapshot.requests.len(),
        },
    }
}

fn partial(id: &RequestId) -> [(RequestId, PlanningWorkAction); 1] {
    [(
        id.clone(),
        PlanningWorkAction::Prefill {
            offset: 0,
            count: NonZeroUsize::new(2).unwrap(),
        },
    )]
}

#[tokio::test]
async fn ingress_records_original_budget_and_never_calls_acceptance_a_time_witness() {
    let (engine, _, _) = fixture_with_width(1).await;
    let (id, session) = prefill::request(&engine, 4, 8).await;
    let original = {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        let state = sequence.time_admission.as_ref().unwrap();
        assert_eq!(state.ingress, sequence.slo.as_ref().unwrap().ingress());
        assert_eq!(
            (state.original_input_tokens, state.maximum_output_tokens),
            (4, 8)
        );
        assert!(state.started.is_none());
        assert!(state.last_assessment.is_none());
        state.ingress
    };
    // Reinitialization must not reset the original arrival or accept twice.
    let mut sequences = engine.inner.sequences.write();
    let sequence = sequences.get_mut(&id).unwrap();
    engine.inner.initialize_sequence_time_admission(sequence);
    assert_eq!(sequence.time_admission.as_ref().unwrap().ingress, original);
    drop(sequences);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn off_ingress_adds_no_time_admission_state() {
    let (mut engine, _, _) = fixture_with_width(1).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = SloMode::Off;
    let (id, session) = prefill::request(&engine, 1, 2).await;
    assert!(engine.inner.sequences.read()[&id].time_admission.is_none());
    cleanup(engine, session).await;
}

#[tokio::test]
async fn ordinary_run_and_stream_ingress_use_the_same_untimed_registration() {
    let (engine, _, _) = fixture_with_width(2).await;
    let request = InferenceRequest::new("test", engine.inner.config.model.model_id.clone());
    let stream_id = request.id.clone();
    let stream = engine
        .infer_stream_with_context(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
        )
        .await
        .unwrap();
    let request = InferenceRequest::new("test", engine.inner.config.model.model_id.clone());
    let response_id = request.id.clone();
    let mut response = Box::pin(engine.infer_with_context(
        request,
        InferenceRequestContext::from_ingress(slo_clock_now()),
    ));
    assert!(response.as_mut().now_or_never().is_none());
    {
        let sequences = engine.inner.sequences.read();
        for id in [&stream_id, &response_id] {
            let state = sequences[id].time_admission.as_ref().unwrap();
            assert_eq!(state.original_input_tokens, 1);
            assert!(state.started.is_none());
        }
    }
    drop(response);
    drop(stream);
    // The fixture deliberately has no background loop. Drive the real
    // abandoned-request cleanup before dropping legacy request-slot owners.
    {
        let _iteration = engine.inner.iteration_lock.lock().await;
        engine.inner.cancel_abandoned_requests().await.unwrap();
    }
    assert!(engine.inner.sequences.read().is_empty());
    assert!(engine.inner.scheduler.trace_phase(&stream_id).is_none());
    assert!(engine.inner.scheduler.trace_phase(&response_id).is_none());
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn new_targets_rotate_after_assessment_without_changing_any_peer_or_time_origin() {
    let (engine, _, executor) = fixture_with_width(2).await;
    let (first_id, first) = prefill::request(&engine, 4, 8).await;
    let (second_id, second) = prefill::request(&engine, 2, 8).await;
    prefill::admit(&engine, 2).await;
    let captured = prefill::captured(&engine, &executor).await;
    let before = captured.snapshot.clone();
    let first_target = engine
        .inner
        .time_admission_target(&captured)
        .unwrap()
        .unwrap();
    assert_eq!(first_target.request_id, first_id);
    let proposal = engine.inner.record_slo_time_assessment(
        &captured,
        &first_target,
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::CostUnavailable),
            continuation: TimeAdmissionContinuation::BestEffort,
            wait: TimeAdmissionWait {
                review_at_ns: None,
                strict_expiry_at_ns: None,
                snapshot_generation: captured.snapshot.generation,
                cost_model_version: captured.snapshot.cost_model_version,
            },
        },
    );
    assert!(proposal.pending.is_none());
    assert_eq!(
        engine
            .inner
            .time_admission_target(&captured)
            .unwrap()
            .unwrap()
            .request_id,
        second_id
    );
    assert_eq!(
        captured.snapshot, before,
        "no request is removed or re-anchored by assessment"
    );
    drop(captured);
    drop(second);
    cleanup(engine, first).await;
}

#[tokio::test]
async fn actual_admission_adapter_retains_unknown_backend_route_without_minting_a_promise() {
    let (engine, _, executor) = fixture_with_width(1).await;
    let (id, session) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    let captured = prefill::captured(&engine, &executor).await;
    // This controlled executor deliberately has no complete future-route
    // producer. An imported model alone must not bypass that missing evidence.
    let proposal = engine.inner.propose_slo_time_admission(&captured).unwrap();
    assert!(matches!(
        proposal.decision,
        Some(PlanningDecision::Unknown { .. })
    ));
    assert!(proposal.pending.is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let sequences = engine.inner.sequences.read();
    let state = sequences[&id].time_admission.as_ref().unwrap();
    assert!(state.started.is_none());
    assert!(matches!(
        state.last_assessment.unwrap().kind,
        TimeAdmissionAssessmentKind::Unknown(_)
    ));
    drop(sequences);
    drop(captured);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn stale_finite_receipt_cannot_mark_reused_request_id() {
    let (engine, _, executor) = fixture_with_width(1).await;
    let (id, old) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    let old_receipt = pending(&engine, &executor, &id).await;
    drop(old);
    bounded(async {
        while engine.inner.sequences.read().contains_key(&id) {
            engine.inner.cancel_abandoned_requests().await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
    let mut request = InferenceRequest::new(
        "test test test test",
        engine.inner.config.model.model_id.clone(),
    );
    request.id = id.clone();
    request.stream = true;
    request.sampling_params.max_tokens = 8;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let replacement = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    engine.inner.record_started_time_witness(&old_receipt);
    assert!(engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .started
        .is_none());
    cleanup(engine, replacement).await;
}

#[tokio::test]
async fn only_actual_guarded_submission_starts_one_finite_witness() {
    let (engine, scheduler, executor) = fixture_with_width(1).await;
    let (id, session) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    let receipt = pending(&engine, &executor, &id).await;
    let ingress = receipt.ingress;
    let through = receipt.evidence.validated_through;
    let repeated = pending(&engine, &executor, &id).await;
    let prepared = prefill::install_with_admission(
        &engine,
        &executor,
        &scheduler,
        &partial(&id),
        Some(receipt),
    )
    .await;
    assert!(engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .started
        .is_none());
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    engine.inner.record_started_time_witness(&repeated);
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 2);
        let state = sequence.time_admission.as_ref().unwrap();
        assert_eq!(state.ingress, ingress);
        assert_eq!(state.started.unwrap().validated_through, through);
    }
    let captured = prefill::captured(&engine, &executor).await;
    assert!(
        engine
            .inner
            .time_admission_target(&captured)
            .unwrap()
            .is_none(),
        "the same partial owner cannot receive a new admission promise"
    );
    drop(captured);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn real_no_submission_withdrawal_keeps_request_untimed_and_retryable() {
    let (engine, scheduler, executor) = fixture_with_width(1).await;
    let (id, session) = prefill::request(&engine, 4, 8).await;
    prefill::admit(&engine, 1).await;
    let prepared = prefill::install_with_admission(
        &engine,
        &executor,
        &scheduler,
        &partial(&id),
        Some(pending(&engine, &executor, &id).await),
    )
    .await;
    executor.replan_before_encode.store(true, Ordering::Release);
    let _ = bounded(engine.inner.execute_slo_controller_wave(prepared))
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 0);
        assert!(sequence.time_admission.as_ref().unwrap().started.is_none());
    }
    let captured = prefill::captured(&engine, &executor).await;
    assert_eq!(
        engine
            .inner
            .time_admission_target(&captured)
            .unwrap()
            .unwrap()
            .request_id,
        id
    );
    drop(captured);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn missing_reference_keeps_real_complete_requests_progress_and_no_time_promise() {
    let (mut engine, _, executor) = fixture_with_width(1).await;
    {
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        inner.config.scheduler.slo.mode = SloMode::Enforce;
        inner.prefill_reference_runtime = None;
    }
    let (id, mut session) = prefill::request(&engine, 1, 2).await;
    prefill::admit(&engine, 1).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        engine.inner.prepare_slo_controller(&hint)
    })
    .await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    drop(bounded(session.frames.next()).await.unwrap());
    ready(&engine, &id).await;
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    assert!(engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .started
        .is_none());
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
        engine.inner.prepare_completion_controller(
            &hint,
            &budget,
            CompletionOnlyReason::CostUnavailable,
        )
    })
    .await;
    bounded(engine.inner.execute_slo_controller_wave(prepared))
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    cleanup(engine, session).await;
}
