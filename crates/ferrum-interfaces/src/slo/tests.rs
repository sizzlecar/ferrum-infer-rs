use super::*;
use std::num::NonZeroU64;

#[test]
fn slo_violation_wake_allows_deadline_equality_and_does_not_spin_after_recording() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    let wake = timing.next_violation_wake().unwrap().unwrap();
    assert_eq!(wake.boundary, SloTimingBoundary::FirstToken);
    assert_eq!(
        wake.at,
        t + Duration::from_millis(100) + Duration::from_nanos(1)
    );
    timing.observe_wait(t + Duration::from_millis(100)).unwrap();
    assert_eq!(timing.next_violation_wake().unwrap(), Some(wake));
    timing.observe_wait(wake.at).unwrap();
    assert!(timing.violations().ttft);
    assert_eq!(timing.next_violation_wake().unwrap(), None);
}

#[test]
fn slo_violation_wake_retains_the_other_decode_budget_after_one_is_missed() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    timing.record_commit(t + Duration::from_millis(10)).unwrap();
    let prefix = timing.next_violation_wake().unwrap().unwrap();
    assert_eq!(prefix.boundary, SloTimingBoundary::TokenPrefix);
    timing.observe_wait(prefix.at).unwrap();
    let interval = timing.next_violation_wake().unwrap().unwrap();
    assert_eq!(interval.boundary, SloTimingBoundary::InterToken);
    assert_eq!(
        interval.at,
        t + Duration::from_millis(90) + Duration::from_nanos(1)
    );
    timing.observe_wait(interval.at).unwrap();
    assert_eq!(timing.next_violation_wake().unwrap(), None);
    assert!(timing.violations().tpot && timing.violations().itl);
}

#[test]
fn slo_violation_wake_rearms_from_commit_without_clearing_historical_failure() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    timing.observe_wait(t + Duration::from_millis(101)).unwrap();
    timing
        .record_commit(t + Duration::from_millis(102))
        .unwrap();
    assert!(timing.violations().ttft);
    let wake = timing.next_violation_wake().unwrap().unwrap();
    assert_eq!(wake.boundary, SloTimingBoundary::TokenPrefix);
    assert_eq!(
        wake.at,
        t + Duration::from_millis(132) + Duration::from_nanos(1)
    );
    timing.observe_wait(t).unwrap_err();
    assert!(timing.next_violation_wake().is_err());
}

#[test]
fn slo_violation_wake_arithmetic_failure_invalidates_owner_evidence() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    timing.record_commit(t).unwrap();
    timing.committed_tokens = u64::MAX;
    assert!(timing.is_trusted());
    assert!(timing.arm_violation_wake().is_err());
    assert!(!timing.is_trusted());
    assert!(timing.next_deadline().is_err());
}

fn state(at: Instant, f: u64, p: u64, i: u64) -> RequestSloState {
    RequestSloState::new(
        at,
        "interactive".into(),
        SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(f).unwrap(),
            tpot_ms: NonZeroU64::new(p).unwrap(),
            itl_ms: NonZeroU64::new(i).unwrap(),
        },
    )
    .unwrap()
}

#[test]
fn slo_deadline_uses_ingress_and_current_committed_count() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    assert_eq!(
        timing.next_deadline().unwrap(),
        t + Duration::from_millis(100)
    );
    timing.record_commit(t + Duration::from_millis(90)).unwrap();
    assert_eq!(
        timing.next_deadline().unwrap(),
        t + Duration::from_millis(120)
    );
    timing
        .record_commit(t + Duration::from_millis(115))
        .unwrap();
    assert_eq!(
        timing.next_deadline().unwrap(),
        t + Duration::from_millis(150)
    );
    assert_eq!(timing.committed_tokens(), 2);
    assert!(!timing.violations().any());
}

#[test]
fn slo_itl_binds_even_with_earlier_fast_tokens() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 40);
    for ms in [10, 11, 12] {
        timing.record_commit(t + Duration::from_millis(ms)).unwrap();
    }
    assert_eq!(
        timing.next_deadline().unwrap(),
        t + Duration::from_millis(52)
    );
    timing.record_commit(t + Duration::from_millis(53)).unwrap();
    assert!(timing.violations().itl);
    assert!(!timing.violations().tpot);
}

#[test]
fn slo_waiting_and_later_fast_tokens_do_not_erase_missed_deadline() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    timing.observe_wait(t + Duration::from_millis(101)).unwrap();
    timing
        .record_commit(t + Duration::from_millis(102))
        .unwrap();
    timing
        .record_commit(t + Duration::from_millis(140))
        .unwrap();
    timing
        .record_commit(t + Duration::from_millis(141))
        .unwrap();
    assert_eq!(
        timing.violations(),
        SloTimingViolations {
            ttft: true,
            tpot: true,
            itl: false
        }
    );
}

#[test]
fn slo_exact_deadline_is_allowed_and_single_token_has_no_decode_failure() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    timing
        .record_commit(t + Duration::from_millis(100))
        .unwrap();
    assert!(!timing.violations().any());
    timing
        .record_commit(t + Duration::from_millis(130))
        .unwrap();
    assert!(!timing.violations().any());
}

#[test]
fn slo_nonmonotonic_commit_is_rejected_without_advancing() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    assert!(timing.record_commit(t - Duration::from_millis(1)).is_err());
    assert_eq!(timing.committed_tokens(), 0);
    timing.record_commit(t + Duration::from_millis(50)).unwrap();
    assert!(timing.record_commit(t + Duration::from_millis(49)).is_err());
    assert_eq!(timing.committed_tokens(), 1);
    assert_eq!(timing.first_commit(), timing.last_commit());
    assert!(!timing.is_trusted());
    assert!(timing.next_deadline().is_err());
}

#[test]
fn slo_overflow_is_an_error_instead_of_an_unlimited_deadline() {
    let t = Instant::now();
    let mut timing = state(t, 100, 30, 80);
    timing.record_commit(t).unwrap();
    timing.committed_tokens = u64::MAX;
    assert!(timing.next_deadline().is_err());
    assert!(timing.record_commit(t).is_err());
    assert_eq!(timing.committed_tokens(), u64::MAX);
}

#[test]
fn slo_off_context_keeps_original_ingress_without_enabling_policy() {
    let t = Instant::now() - Duration::from_millis(500);
    let context = InferenceRequestContext::from_ingress(t);
    assert_eq!(context.ingress(), t);
    assert!(context
        .resolve_slo(&SloConfig::default())
        .unwrap()
        .is_none());
}
