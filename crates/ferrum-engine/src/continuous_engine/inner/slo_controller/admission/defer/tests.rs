use super::*;
use crate::continuous_engine::inner::slo_controller::tests::{
    bounded,
    fixture::{cleanup, fixture_with_width, ControlledExecutor},
    prefill,
};
use ferrum_interfaces::{output_flow::CreditedOutputSession, scheduler::Scheduler};
use ferrum_scheduler::implementations::continuous::{ContinuousBatchScheduler, RequestPhase};
use ferrum_scheduler::vnext::AdmissionDeferral;
use ferrum_types::SloMode;
use futures::FutureExt;
use std::{sync::atomic::Ordering, time::Duration};

async fn engine(
    width: usize,
    mode: SloMode,
    cap: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (mut engine, scheduler, executor) = fixture_with_width(width).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = mode;
    inner.config.scheduler.slo.admission.max_active_requests = NonZeroUsize::new(cap).unwrap();
    inner.config.scheduler.slo.admission.max_wait_ms = NonZeroU64::new(100).unwrap();
    (engine, scheduler, executor)
}

async fn turn(engine: &ContinuousBatchEngine, width: usize) -> bool {
    let _iteration = engine.inner.iteration_lock.lock().await;
    engine
        .inner
        .prepare_slo_admission_turn(width)
        .await
        .unwrap()
}

async fn cancel(engine: &ContinuousBatchEngine, id: &RequestId, session: CreditedOutputSession) {
    drop(session);
    bounded(async {
        while engine.inner.sequences.read().contains_key(id) {
            engine.inner.cancel_abandoned_requests().await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
}

#[tokio::test(start_paused = true)]
async fn enforce_time_limit_retains_waiting_and_one_overdue_owner_without_resetting_ingress() {
    let (engine, scheduler, executor) = engine(4, SloMode::Enforce, 1).await;
    let (active_id, active) = prefill::request(&engine, 4, 8).await;
    let (late_id, late) = prefill::request(&engine, 4, 8).await;
    let (third_id, third) = prefill::request(&engine, 4, 8).await;
    let ingress = engine.inner.sequences.read()[&late_id]
        .slo
        .as_ref()
        .unwrap()
        .ingress();
    assert!(turn(&engine, 4).await);
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(
        scheduler.trace_phase(&active_id),
        Some(RequestPhase::Prefilling)
    );
    assert!(!turn(&engine, 4).await);
    assert_eq!(scheduler.waiting_count(), 2);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    let state = engine.inner.sequences.read()[&late_id]
        .time_admission
        .as_ref()
        .unwrap()
        .last_assessment
        .unwrap();
    assert_eq!(
        state.kind,
        TimeAdmissionAssessmentKind::Deferred(TimeAdmissionDeferReason::ActiveLimit)
    );
    let before = scheduler.trace_snapshot();
    let mut wake = Box::pin(engine.inner.wait_for_slo_time_admission());
    assert!(wake.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(50)).await;
    assert!(wake.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(51)).await;
    assert!(wake.as_mut().now_or_never().is_some());
    drop(wake);
    assert_eq!(
        scheduler.trace_snapshot(),
        before,
        "timer alone cannot alter physical or queue authority"
    );
    assert!(turn(&engine, 4).await);
    assert_eq!(scheduler.active_count(), 2);
    assert_eq!(
        scheduler.trace_phase(&late_id),
        Some(RequestPhase::Prefilling)
    );
    for _ in 0..3 {
        assert!(
            !turn(&engine, 4).await,
            "an expired timer must not mint another exception per loop"
        );
    }
    assert_eq!(
        scheduler.trace_phase(&third_id),
        Some(RequestPhase::Waiting)
    );
    assert!(engine
        .inner
        .wait_for_slo_time_admission()
        .now_or_never()
        .is_none());
    assert!(scheduler.defer_prefill_to_waiting(&late_id));
    assert!(
        turn(&engine, 4).await,
        "an admitted overflow continuation may rebuild its own backing"
    );
    assert_eq!(scheduler.active_count(), 2);
    assert_eq!(
        scheduler.trace_phase(&third_id),
        Some(RequestPhase::Waiting)
    );
    assert!(!turn(&engine, 4).await);
    assert_eq!(
        engine.inner.sequences.read()[&late_id]
            .slo
            .as_ref()
            .unwrap()
            .ingress(),
        ingress
    );
    assert_eq!(
        engine.inner.sequences.read()[&late_id]
            .sampling_params
            .max_tokens,
        8
    );
    assert!(engine.inner.sequences.read()[&late_id]
        .time_admission
        .as_ref()
        .unwrap()
        .started
        .is_none());
    cancel(&engine, &active_id, active).await;
    // The previous excess owner now fits the normal cap. A different overdue
    // owner may compete for the single extra slot, after a real departure.
    assert!(turn(&engine, 4).await);
    assert_eq!(scheduler.active_count(), 2);
    drop(third);
    cleanup(engine, late).await;
}

#[tokio::test]
async fn active_completion_serves_while_peer_waits_then_frees_time_slot() {
    let (mut engine, scheduler, executor) = engine(2, SloMode::Enforce, 1).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .prefill_reference_runtime = None;
    let (active_id, active) = prefill::request(&engine, 1, 1).await;
    let (waiting_id, waiting) = prefill::request(&engine, 1, 1).await;
    assert!(turn(&engine, 2).await);
    assert!(!turn(&engine, 2).await);
    // Missing reference/model coverage does not add any time hold to the
    // already active request. Use the actual guarded completion path.
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
        engine.inner.prepare_completion_controller(
            &ferrum_interfaces::BatchHint::simple(2),
            &budget,
            ferrum_interfaces::execution_cost::CompletionOnlyReason::CostUnavailable,
        )
    })
    .await;
    bounded(engine.inner.execute_slo_controller_wave(prepared))
        .await
        .unwrap();
    bounded(async {
        while engine.inner.sequences.read().contains_key(&active_id) {
            engine.inner.drain_slo_execution().await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(
        scheduler.trace_phase(&waiting_id),
        Some(RequestPhase::Waiting)
    );
    assert!(
        turn(&engine, 2).await,
        "actual completion, without timer expiry, must free the time slot"
    );
    assert_eq!(
        scheduler.trace_phase(&waiting_id),
        Some(RequestPhase::Prefilling)
    );
    drop(active);
    cleanup(engine, waiting).await;
}

#[tokio::test]
async fn physical_recompute_keeps_original_activation_instead_of_reentering_fresh_limit() {
    let (engine, scheduler, _) = engine(2, SloMode::Enforce, 1).await;
    let (active_id, active) = prefill::request(&engine, 4, 8).await;
    let (waiting_id, waiting) = prefill::request(&engine, 4, 8).await;
    assert!(turn(&engine, 2).await);
    assert!(scheduler.defer_prefill_to_waiting(&active_id));
    assert!(turn(&engine, 2).await);
    assert_eq!(
        scheduler.trace_phase(&active_id),
        Some(RequestPhase::Prefilling)
    );
    assert_eq!(
        scheduler.trace_phase(&waiting_id),
        Some(RequestPhase::Waiting)
    );
    assert_eq!(scheduler.active_count(), 1);
    drop(waiting);
    cleanup(engine, active).await;
}

#[tokio::test(start_paused = true)]
async fn reused_id_cannot_inherit_cancelled_overflow_or_original_wait_expiry() {
    use crate::continuous_engine::inner::slo_controller::tests::ready;
    use ferrum_interfaces::{
        engine::LlmInferenceEngine, output_flow::OutputProjectionContract, InferenceRequestContext,
    };
    let (engine, scheduler, _) = engine(2, SloMode::Enforce, 1).await;
    let (_, active) = prefill::request(&engine, 4, 8).await;
    let (id, old) = prefill::request(&engine, 4, 8).await;
    assert!(turn(&engine, 2).await);
    assert!(!turn(&engine, 2).await);
    tokio::time::advance(Duration::from_millis(101)).await;
    assert!(turn(&engine, 2).await);
    let (old_owner, old_ingress) = {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        (
            Arc::clone(&sequence.stream_projection_identity),
            sequence.slo.as_ref().unwrap().ingress(),
        )
    };
    cancel(&engine, &id, old).await;
    let mut request = InferenceRequest::new(
        "test test test test",
        engine.inner.config.model.model_id.clone(),
    );
    request.id = id.clone();
    request.stream = true;
    request.sampling_params.max_tokens = 8;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let new = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(&engine, &id).await;
    assert!(!turn(&engine, 2).await);
    assert_eq!(scheduler.trace_phase(&id), Some(RequestPhase::Waiting));
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert!(!Arc::ptr_eq(
            &old_owner,
            &sequence.stream_projection_identity
        ));
        let state = sequence.time_admission.as_ref().unwrap();
        assert!(!state.activated);
        assert!(state.ingress > old_ingress);
    }
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .time_activation
        .overflow
        .is_none());
    drop(new);
    cleanup(engine, active).await;
}

#[tokio::test(start_paused = true)]
async fn off_and_observe_preserve_original_admission_and_do_not_arm_time_wake() {
    for mode in [SloMode::Off, SloMode::Observe] {
        let (engine, scheduler, _) = engine(2, mode, 1).await;
        let (_, first) = prefill::request(&engine, 1, 2).await;
        let (_, second) = prefill::request(&engine, 1, 2).await;
        assert!(turn(&engine, 2).await);
        assert_eq!(scheduler.active_count(), 2, "time gate is Enforce-only");
        tokio::time::advance(Duration::from_secs(1)).await;
        assert!(engine
            .inner
            .wait_for_slo_time_admission()
            .now_or_never()
            .is_none());
        drop(second);
        cleanup(engine, first).await;
    }
}

#[tokio::test]
async fn unknown_cost_below_time_cap_does_not_hold_new_requests_until_timeout() {
    let (mut engine, scheduler, _) = engine(2, SloMode::Enforce, 2).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .prefill_reference_runtime = None;
    let (first_id, first) = prefill::request(&engine, 4, 8).await;
    let (second_id, second) = prefill::request(&engine, 4, 8).await;
    assert!(turn(&engine, 2).await);
    assert_eq!(scheduler.active_count(), 2);
    for id in [&first_id, &second_id] {
        let sequences = engine.inner.sequences.read();
        let state = sequences[id].time_admission.as_ref().unwrap();
        assert!(state.activated);
        assert!(state.last_assessment.is_none());
        assert!(
            state.started.is_none(),
            "physical admission cannot fabricate a time witness"
        );
    }
    assert!(engine
        .inner
        .wait_for_slo_time_admission()
        .now_or_never()
        .is_none());
    drop(second);
    cleanup(engine, first).await;
}

#[tokio::test(start_paused = true)]
async fn defer_retains_real_review_time_and_unknown_does_not_create_a_hold() {
    let (engine, _, executor) = engine(1, SloMode::Enforce, 1).await;
    let (id, session) = prefill::request(&engine, 4, 8).await;
    assert!(turn(&engine, 1).await);
    let captured = prefill::captured(&engine, &executor).await;
    let target = captured.snapshot.requests[0].key.clone();
    let at = slo_clock_now() + Duration::from_millis(10);
    let wait = TimeAdmissionWait {
        review_at_ns: Some(captured.origin.at_ns(at).unwrap()),
        strict_expiry_at_ns: None,
        snapshot_generation: captured.snapshot.generation,
        cost_model_version: captured.snapshot.cost_model_version,
    };
    let proposal = engine.inner.record_slo_time_assessment(
        &captured,
        &target,
        TimeAdmissionDecision::Defer {
            reason: TimeAdmissionDeferReason::ExistingObligationAtRisk,
            wait,
        },
    );
    assert!(proposal.decision.is_none());
    assert_eq!(
        proposal.deferred,
        Some(TimeAdmissionDeferReason::ExistingObligationAtRisk)
    );
    assert!(engine
        .inner
        .time_admission_target(&captured)
        .unwrap()
        .is_none());
    let mut wake = Box::pin(engine.inner.wait_for_slo_time_admission());
    assert!(wake.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(11)).await;
    assert!(wake.as_mut().now_or_never().is_some());
    drop(wake);
    assert_eq!(
        engine.inner.time_admission_target(&captured).unwrap(),
        Some(target.clone())
    );
    let unknown = engine.inner.record_slo_time_assessment(
        &captured,
        &target,
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::CostUnavailable),
            continuation: TimeAdmissionContinuation::BestEffort,
            wait,
        },
    );
    assert!(unknown.deferred.is_none());
    assert!(matches!(
        unknown.decision,
        Some(PlanningDecision::Unknown {
            reason: PlanningUnknownReason::CostUnavailable,
            ..
        })
    ));
    assert!(engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .deferred
        .is_none());
    let later = slo_clock_now() + Duration::from_millis(10);
    engine.inner.record_slo_time_assessment(
        &captured,
        &target,
        TimeAdmissionDecision::Defer {
            reason: TimeAdmissionDeferReason::ExistingObligationAtRisk,
            wait: TimeAdmissionWait {
                review_at_ns: Some(captured.origin.at_ns(later).unwrap()),
                ..wait
            },
        },
    );
    // Thread/host delay can pass review_at before Idle registers its select.
    tokio::time::advance(Duration::from_millis(11)).await;
    assert!(engine
        .inner
        .wait_for_slo_time_admission()
        .now_or_never()
        .is_some());
    assert!(engine
        .inner
        .wait_for_slo_time_admission()
        .now_or_never()
        .is_none());
    drop(captured);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn time_hold_does_not_hide_real_capacity_wake_or_reprobe_unchanged_source() {
    use ferrum_interfaces::model_executor::ExecutorPrefillAdmissionReceipt;
    use ferrum_interfaces::vnext::{
        CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
        DeferredAction,
    };
    let scheduler = ContinuousBatchScheduler::new(Default::default());
    let first = scheduler
        .submit(InferenceRequest::new(
            "a",
            ferrum_types::ModelId::new("fixture"),
        ))
        .await
        .unwrap();
    let held = scheduler
        .submit(InferenceRequest::new(
            "b",
            ferrum_types::ModelId::new("fixture"),
        ))
        .await
        .unwrap();
    let epochs = AdmissionWakeEpochs::new(NonZeroU64::new(47).unwrap(), 0, 0, 0);
    let source = CapacityAvailabilitySource::ActiveSequenceSlots;
    let sources = [CapacityAvailabilityEpoch::new(source, 1).unwrap()];
    let condition = CapacityWaitCondition::from_observation(47, sources.to_vec()).unwrap();
    let mut probes = Vec::new();
    let first_receipt = scheduler
        .prepare_dynamic_admission_observed_with_eligibility(
            1,
            AdmissionWakeSnapshot::new(epochs, &sources),
            &mut |request| {
                probes.push(request.id.clone());
                AdmissionProbeOutcome::Deferred(AdmissionDeferral::new(
                    DeferredAction::WaitForRelease,
                    epochs,
                    condition.clone(),
                ))
            },
            &mut |_| {},
            &|request| request.id != held,
        )
        .unwrap();
    assert_eq!(first_receipt.probed(), 1);
    assert_eq!(probes, vec![first.clone()]);
    assert!(scheduler
        .passive_capacity_wait_condition()
        .unwrap()
        .is_none());
    assert_eq!(
        scheduler
            .passive_capacity_wait_condition_with_eligibility(&|request| request.id != held)
            .unwrap(),
        Some(condition)
    );
    let unchanged = scheduler
        .prepare_dynamic_admission_observed_with_eligibility(
            1,
            AdmissionWakeSnapshot::new(epochs, &sources),
            &mut |_| panic!("unchanged capacity cannot trigger a probe"),
            &mut |_| {},
            &|request| request.id != held,
        )
        .unwrap();
    assert_eq!(unchanged.probed(), 0);
    let changed = [CapacityAvailabilityEpoch::new(source, 2).unwrap()];
    let admitted = scheduler
        .prepare_dynamic_admission_observed_with_eligibility(
            1,
            AdmissionWakeSnapshot::new(epochs, &changed),
            &mut |request| {
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                    request_id: request.id.clone(),
                })
            },
            &mut |_| {},
            &|request| request.id != held,
        )
        .unwrap();
    assert_eq!(admitted.admitted(), 1);
    assert_eq!(
        scheduler.trace_phase(&first),
        Some(RequestPhase::Prefilling)
    );
    assert_eq!(scheduler.trace_phase(&held), Some(RequestPhase::Waiting));
}
