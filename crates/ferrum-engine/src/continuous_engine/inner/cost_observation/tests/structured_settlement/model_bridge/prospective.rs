use super::*;
use crate::continuous_engine::inner::cost_observation::prospective_capture::{
    ProspectiveCapture, ProspectiveCaptureOutcomeV1 as Outcome,
};
use std::time::{Duration, Instant};

fn fixture() -> Fixture {
    let mut f = Fixture::new();
    f.config.prospective_structured_capture =
        ferrum_types::SloProspectiveStructuredCapture::ReplayedFirstWaveV1;
    f
}
fn capture(
    f: &Fixture,
    runtime: &EngineCostRuntime,
    wave: &Wave,
    deadline: Instant,
) -> Arc<ProspectiveCapture> {
    let row = &wave.actual.rows[0];
    ProspectiveCapture::from_fixture(
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
        deadline,
    )
}
fn attach(call: &mut EngineCostCall, pending: &Arc<ProspectiveCapture>, expected: bool) {
    call.attach_prospective_capture(pending.clone(), Instant::now());
    assert_eq!(
        call.prospective_capture.is_some(),
        expected,
        "test must reach the intended attachment boundary"
    );
}
fn observe(
    f: &Fixture,
    runtime: &EngineCostRuntime,
    pending: &Arc<ProspectiveCapture>,
    wave: &Wave,
    terminal: Option<ferrum_types::FinishReason>,
    cancel: bool,
    expect_attached: bool,
) -> Arc<HostStageEvidenceV1> {
    record_with_hook(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        wave.actual.clone(),
        wave.host,
        None,
        0,
        terminal,
        |call| {
            attach(call, pending, expect_attached);
            if cancel {
                call.host_cancelled(&call.participants[0].request_id.clone());
            }
        },
    )
}
fn count(runtime: &EngineCostRuntime, outcome: &str) -> u64 {
    let value = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    value["prospective_capture"]["outcomes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row[0].as_str() == Some(outcome))
        .unwrap()[1]
        .as_u64()
        .unwrap()
}

#[tokio::test]
async fn prospective_real_settlement_binds_original_call_without_source_membership() {
    let f = fixture();
    let runtime = f.build();
    let w = wave("fixture.feedback.a");
    let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
    let stages = observe(&f, &runtime, &pending, &w, None, false, true);
    let receipt = stages.prospective_capture.as_ref().unwrap();
    assert_eq!(receipt.outcome(), Outcome::Matched);
    assert!(receipt.validates_host_stages(&stages));
    let mut changed = stages.as_ref().clone();
    changed.rows[0].work_generation += 1;
    assert!(!receipt.validates_host_stages(&changed));
    let mut changed = stages.as_ref().clone();
    changed.call_id += 1;
    assert!(!receipt.validates_host_stages(&changed));
    assert_eq!(
        count(&runtime, "matched"),
        1,
        "repeated stage views count one call"
    );
    // No calibration capture/session was attached. The live observer still
    // cannot create the original protocol's membership/phase/cohort receipt.
    f.unchanged();
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_different_request_is_rejected_before_context() {
    let mut f = fixture();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
    let declared = wave("fixture.feedback.a");
    let other = wave("fixture.feedback.a");
    assert_ne!(
        declared.actual.rows[0].request_id,
        other.actual.rows[0].request_id
    );
    let pending = capture(
        &f,
        &runtime,
        &declared,
        Instant::now() + Duration::from_secs(30),
    );
    let stages = observe(&f, &runtime, &pending, &other, None, false, false);
    assert!(stages.prospective_capture.is_none());
    assert_eq!(count(&runtime, "participant_mismatch"), 1);
    assert_eq!(count(&runtime, "matched"), 0);
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_expiry_and_epoch_change_reject_capture_without_resurrecting_model() {
    let f = fixture();
    let runtime = f.build();
    let w = wave("fixture.feedback.a");
    let expired = capture(&f, &runtime, &w, Instant::now() - Duration::from_nanos(1));
    let stages = observe(&f, &runtime, &expired, &w, None, false, false);
    assert!(stages.prospective_capture.is_none());
    assert_eq!(count(&runtime, "expired"), 1);
    runtime.consume_samples();
    let old = runtime.snapshot().unwrap();
    let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
    f.observe(&runtime, 30);
    f.observe(&runtime, 30);
    assert!(!old.current());
    let stages = observe(&f, &runtime, &pending, &w, None, false, false);
    assert!(stages.prospective_capture.is_none());
    assert_eq!(count(&runtime, "epoch_changed"), 1);
    assert!(old
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .is_err());
    f.unchanged();
    drop(expired);
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_actual_route_eos_cancellation_and_unsubmitted_drop_fail_closed() {
    let mut f = fixture();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
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
        let pending = capture(
            &f,
            &runtime,
            &expected,
            Instant::now() + Duration::from_secs(30),
        );
        let mut actual = wave(algorithm);
        // The same declared request takes a different actual algorithm/terminal.
        // Preserve its identity so the negative case reaches reconciliation.
        actual.actual.rows = expected.actual.rows.clone();
        let stages = observe(&f, &runtime, &pending, &actual, terminal, cancel, true);
        let receipt = stages.prospective_capture.as_ref().unwrap();
        assert_eq!(receipt.outcome(), outcome);
        assert!(!receipt.validates_host_stages(&stages));
    }
    let w = wave("fixture.feedback.a");
    let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
    drop(pending);
    assert_eq!(count(&runtime, "abandoned"), 1);
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_fifo_loss_does_not_become_accepted_evidence() {
    let mut f = fixture();
    f.config.max_queued_samples = NonZeroUsize::new(1).unwrap();
    f.config.max_samples_per_update = NonZeroUsize::new(1).unwrap();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
    for _ in 0..2 {
        let w = wave("fixture.feedback.a");
        let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
        let stages = observe(&f, &runtime, &pending, &w, None, false, true);
        assert_eq!(
            stages.prospective_capture.as_ref().unwrap().outcome(),
            Outcome::Matched
        );
    }
    let audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    assert_eq!(audit["prospective_capture"]["queued"], 1);
    assert_eq!(audit["prospective_capture"]["queue_dropped"], 1);
    assert_eq!(
        count(&runtime, "matched"),
        2,
        "match is distinct from FIFO acceptance"
    );
    f.unchanged();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_two_calls_cannot_share_a_declaration_even_when_both_settle() {
    let mut f = fixture();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
    let w = wave("fixture.feedback.a");
    let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
    let mut second = None;
    let first_stages = record_with_hook(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        w.actual.clone(),
        w.host,
        None,
        0,
        None,
        |first| {
            attach(first, &pending, true);
            let mut other = EngineCostCall::begin(
                &runtime.ids,
                f.clock.clone(),
                runtime.sink.clone(),
                EngineCostCallSpec {
                    identity: identity(),
                    participants: first.participants.clone(),
                    prepare_started_at_ns: first.prepare_started_at_ns,
                    boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
                    recorder_limits: runtime.recorder_limits,
                },
            )
            .unwrap()
            .with_structured_capture(true);
            attach(&mut other, &pending, false);
            second = Some(other);
        },
    );
    assert_eq!(
        first_stages.prospective_capture.as_ref().unwrap().outcome(),
        Outcome::AttachmentConflict
    );
    let second_stages = record_with_hook(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        w.actual,
        w.host,
        None,
        0,
        None,
        |call| {
            std::mem::swap(call, second.as_mut().unwrap());
        },
    );
    assert_eq!(
        second_stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    assert_ne!(first_stages.call_id, second_stages.call_id);
    assert!(second_stages.prospective_capture.is_none());
    assert_eq!(count(&runtime, "matched"), 0);
    assert_eq!(count(&runtime, "attachment_conflict"), 1);
    drop(second);
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_compares_independent_sidecar_beyond_v1_equality() {
    let mut f = fixture();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = f.build();
    let mut w = wave("fixture.feedback.a");
    let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
    let original = w.actual.statistical_evidence.take().unwrap();
    let mut wire =
        serde_json::to_value(original.independent_attention_v2().unwrap().to_wire_v2()).unwrap();
    let n = wire["family_signature"][0].as_u64().unwrap();
    wire["family_signature"][0] = serde_json::json!(n ^ 1);
    let sidecar = IndependentAttentionWaveEvidenceV2::from_wire_v2(
        serde_json::from_value(wire).unwrap(),
        &w.prepared.exact,
    )
    .unwrap();
    let changed = original
        .clone()
        .with_independent_attention_v2(sidecar, &w.prepared.exact)
        .unwrap();
    assert_eq!(
        original, changed,
        "V1 equality deliberately ignores independent sidecars"
    );
    w.actual.statistical_evidence = Some(changed);
    let stages = observe(&f, &runtime, &pending, &w, None, false, true);
    assert_eq!(
        stages.prospective_capture.as_ref().unwrap().outcome(),
        Outcome::ActualMismatch
    );
    drop(pending);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn prospective_duplicate_at_finish_boundary_uses_one_terminal_result() {
    let mut f = fixture();
    f.config.structured_feedback = SloStructuredFeedbackPolicy::Disabled;
    let runtime = Arc::new(f.build());
    let runtime_for_hook = runtime.clone();
    let w = wave("fixture.feedback.a");
    let pending = capture(&f, &runtime, &w, Instant::now() + Duration::from_secs(30));
    let second = Arc::new(parking_lot::Mutex::new(None));
    let second_for_hook = second.clone();
    let weak = Arc::downgrade(&pending);
    let clock = f.clock.clone();
    let queue = runtime.sink.clone();
    let row = w.actual.rows[0].clone();
    let host = w.host;
    pending.before_receipt_finish_for_test(move |proposed| {
        assert_eq!(
            proposed,
            Outcome::Matched,
            "interleave after successful reconciliation, before its terminal CAS"
        );
        let mut other = EngineCostCall::begin(
            &runtime_for_hook.ids,
            clock.clone(),
            queue,
            EngineCostCallSpec {
                identity: identity(),
                participants: vec![CostObservationParticipant {
                    request_id: row.request_id,
                    owner_incarnation: row.owner_incarnation,
                    work_generation: row.work_generation,
                    input_index: row.input_index,
                    output_policy_signature: Some([6; 32]),
                    host_features: Some(host),
                }],
                prepare_started_at_ns: clock.now_ns(),
                boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
                recorder_limits: runtime_for_hook.recorder_limits,
            },
        )
        .unwrap()
        .with_structured_capture(true);
        attach(&mut other, &weak.upgrade().unwrap(), false);
        *second_for_hook.lock() = Some(other);
    });
    let first = observe(&f, &runtime, &pending, &w, None, false, true);
    assert_eq!(
        first.prospective_capture.as_ref().unwrap().outcome(),
        Outcome::AttachmentConflict
    );
    assert_eq!(count(&runtime, "matched"), 0);
    assert_eq!(count(&runtime, "attachment_conflict"), 1);
    let other = record_with_hook(
        &runtime.ids,
        &runtime.sink,
        &f.clock,
        w.actual,
        w.host,
        None,
        0,
        None,
        |call| {
            std::mem::swap(call, second.lock().as_mut().unwrap());
        },
    );
    assert_eq!(
        other.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    assert_ne!(first.call_id, other.call_id);
    assert!(other.prospective_capture.is_none());
    drop(second);
    drop(pending);
    runtime.shutdown().await.unwrap();
}
