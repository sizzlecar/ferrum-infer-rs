//! Real qualified import + original recorder/FIFO/settlement, using synthetic clocks.
//! These validate issued error accounting, not hardware prediction accuracy.
use super::*;
use crate::continuous_engine::inner::cost_observation::prospective_capture::{
    IssuedPredictionAuditSnapshot, ProspectiveCapture, ProspectiveCaptureOutcomeV1 as Outcome,
};
use std::time::{Duration, Instant};

fn fixture() -> Fixture {
    let mut f = Fixture::new();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    assert!(f.config.prospective_structured_capture.is_disabled());
    f
}
fn issued(f: &Fixture, runtime: &EngineCostRuntime, wave: &Wave) -> (Arc<ProspectiveCapture>, u64) {
    let bound = runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .unwrap()
        .planning_ns;
    let row = &wave.actual.rows[0];
    let pending = ProspectiveCapture::from_fixture(
        runtime,
        wave.prepared.exact.clone(),
        wave.prepared.selected.clone(),
        &f.queries[0],
        vec![CostObservationParticipant {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            input_index: row.input_index,
            output_policy_signature: Some([6; 32]),
            host_features: Some(wave.host),
        }],
        Instant::now() + Duration::from_secs(30),
    )
    .with_issued_bound_for_test(runtime, bound);
    (pending, bound)
}
fn observe(
    f: &Fixture,
    runtime: &EngineCostRuntime,
    pending: &Arc<ProspectiveCapture>,
    wave: &Wave,
    extra_wall: u64,
    terminal: Option<ferrum_types::FinishReason>,
    cancel: bool,
    expected: CostCallDisposition,
) -> Arc<HostStageEvidenceV1> {
    record_with_hooks_expecting(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        wave.actual.clone(),
        wave.host,
        None,
        extra_wall,
        terminal,
        |_| None,
        |call| {
            call.attach_prospective_capture(pending.clone(), Instant::now());
            assert!(call.prospective_capture.is_some());
            if cancel {
                call.host_cancelled(&call.participants[0].request_id.clone());
            }
        },
        expected,
    )
}
fn audit(runtime: &EngineCostRuntime) -> IssuedPredictionAuditSnapshot {
    let snapshot = runtime.audit_snapshot();
    assert!(
        snapshot.prospective_capture.is_none(),
        "explicit passive capture stays disabled"
    );
    snapshot
        .issued_structured_prediction
        .expect("normal structured issued audit")
}

#[tokio::test]
async fn issued_structured_prediction_counts_original_retained_settlement_once() {
    let f = fixture();
    let runtime = f.build();
    let original = runtime.snapshot().unwrap();
    let w = wave("fixture.feedback.a");
    let (pending, planning) = issued(&f, &runtime, &w);
    let stages = observe(
        &f,
        &runtime,
        &pending,
        &w,
        30,
        None,
        false,
        CostCallDisposition::Queued,
    );
    let receipt = stages.prospective_capture.as_ref().unwrap();
    assert_eq!(receipt.outcome(), Outcome::Matched);
    assert!(receipt.validates_host_stages(&stages));
    assert_eq!(
        audit(&runtime).paired_count,
        0,
        "a diagnostic receipt before FIFO resolution is not retained evidence"
    );
    runtime.consume_samples();
    let actual = stages.full_wall_ns.unwrap();
    let a = audit(&runtime);
    assert_eq!(a.paired_count, 1);
    assert_eq!((a.total_planning_ns, a.total_actual_ns), (planning, actual));
    assert_eq!(a.total_absolute_error_ns, actual.abs_diff(planning));
    assert_eq!(a.total_underestimate_ns, actual.saturating_sub(planning));
    assert_eq!(a.total_overestimate_ns, planning.saturating_sub(actual));
    assert_eq!(a.underestimates, u64::from(actual > planning));
    assert_eq!(a.overestimates, u64::from(actual < planning));
    assert_eq!(a.invalid_measurement, 0);
    pending.record_issued_settlement(1);
    runtime.consume_samples();
    assert_eq!(
        audit(&runtime).paired_count,
        1,
        "duplicate views/consumer polls cannot recount a call"
    );
    assert!(Arc::ptr_eq(&original, &runtime.snapshot().unwrap()));
    f.unchanged();
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn issued_structured_prediction_keeps_original_bound_across_real_catalog_replacement() {
    let f = fixture();
    let runtime = f.build();
    let original = runtime.snapshot().unwrap();
    let original_epoch = original.model_version();
    let w = wave("fixture.feedback.a");
    let (pending, planning) = issued(&f, &runtime, &w);
    let stages = observe(
        &f,
        &runtime,
        &pending,
        &w,
        0,
        None,
        false,
        CostCallDisposition::Queued,
    );
    let now = f.clock.now_ns().unwrap();
    let new_epoch = runtime
        .training
        .publish_live_catalog(
            runtime.training.live_catalog_children(now).unwrap(),
            runtime.profile_receipt().unwrap(),
            now,
        )
        .unwrap();
    assert!(new_epoch > original_epoch);
    assert!(!original.current());
    runtime.consume_samples();
    let a = audit(&runtime);
    assert_eq!(
        a.paired_count, 1,
        "a valid issued call does not need a later current-model lookup"
    );
    assert_eq!(a.total_planning_ns, planning);
    assert_eq!(a.total_actual_ns, stages.full_wall_ns.unwrap());
    let serialized = serde_json::to_value(stages.prospective_capture.as_ref().unwrap()).unwrap();
    assert_eq!(serialized["model_epoch"], original_epoch);
    assert_eq!(serialized["issued_planning_ns"], planning);
    assert_eq!(runtime.snapshot().unwrap().model_version(), new_epoch);
    f.unchanged();
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn issued_structured_prediction_excludes_queue_loss_mismatch_failure_and_unsupported_terminal(
) {
    let mut f = fixture();
    f.config.max_queued_samples = NonZeroUsize::new(1).unwrap();
    f.config.max_samples_per_update = NonZeroUsize::new(1).unwrap();
    let runtime = f.build();
    for index in 0..2 {
        let w = wave("fixture.feedback.a");
        let (pending, _) = issued(&f, &runtime, &w);
        let stages = observe(
            &f,
            &runtime,
            &pending,
            &w,
            0,
            None,
            false,
            if index == 0 {
                CostCallDisposition::Queued
            } else {
                CostCallDisposition::Dropped(CostSampleDrop::Capacity)
            },
        );
        assert_eq!(
            stages.prospective_capture.as_ref().unwrap().outcome(),
            Outcome::Matched
        );
        drop(pending);
    }
    assert_eq!(audit(&runtime).paired_count, 0);
    runtime.consume_samples();
    assert_eq!(audit(&runtime).paired_count, 1);
    assert_eq!(audit(&runtime).matched_but_not_retained, 1);
    for (algorithm, terminal, cancel, outcome) in [
        ("fixture.feedback.b", None, false, Outcome::ActualMismatch),
        (
            "fixture.feedback.a",
            Some(ferrum_types::FinishReason::EOS),
            false,
            Outcome::UnsupportedTerminal,
        ),
        (
            "fixture.feedback.a",
            None,
            true,
            Outcome::SettlementRejected,
        ),
    ] {
        let expected = wave("fixture.feedback.a");
        let (pending, _) = issued(&f, &runtime, &expected);
        let mut actual = wave(algorithm);
        actual.actual.rows = expected.actual.rows.clone();
        let stages = observe(
            &f,
            &runtime,
            &pending,
            &actual,
            0,
            terminal,
            cancel,
            CostCallDisposition::Queued,
        );
        assert_eq!(
            stages.prospective_capture.as_ref().unwrap().outcome(),
            outcome
        );
        runtime.consume_samples();
        assert_eq!(audit(&runtime).paired_count, 1);
    }
    let w = wave("fixture.feedback.a");
    let (unsubmitted, _) = issued(&f, &runtime, &w);
    drop(unsubmitted);
    let a = serde_json::to_value(audit(&runtime)).unwrap();
    assert_eq!(a["population"]["declared"], 6);
    assert_eq!(a["population"]["queued"], 4);
    assert_eq!(a["population"]["queue_dropped"], 1);
    for reason in [
        "actual_mismatch",
        "unsupported_terminal",
        "settlement_rejected",
        "abandoned",
    ] {
        assert_eq!(
            a["population"]["outcomes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|v| v[0].as_str() == Some(reason))
                .unwrap()[1],
            1
        );
    }
    f.unchanged();
    runtime.shutdown().await.unwrap();
}
