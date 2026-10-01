//! Monitor boundary only: production obtains OutsideRoute exclusively from
//! the live consumer's validated route exclusion, never from actual Unknown.
use super::*;

fn monitor(p: &SloSelectedFeedbackSettingsV1) -> Monitor {
    Monitor::open_bound(
        p,
        &Storage::MemoryOnly,
        binding(),
        2,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32], [8; 32]].into()),
    )
    .unwrap()
}

fn outside_route() -> FeedbackObservation {
    FeedbackObservation::OutsideRoute {
        observed_at_ns: 1,
        consumed_at_ns: 2,
    }
}

#[test]
fn automatic_outside_route_preserves_drift_window_and_unrelated_owner_margins() {
    let mut monitor = monitor(&policy());
    monitor.observe_classified(1, FeedbackObservation::Compared(comparison(120)));
    let before = serde_json::to_vec(&monitor.state).unwrap();
    let original = monitor.current_view();
    monitor.observe_classified(2, outside_route());
    assert_eq!(serde_json::to_vec(&monitor.state).unwrap(), before);
    assert!(Arc::ptr_eq(&original, &monitor.current_view()));
    assert!(monitor.publish(Some(0)).is_none());
    assert!(original.current());

    // Exclusion cannot clear an earlier underestimate or fill the residual
    // window: the next actual comparison must still trigger its correction.
    monitor.observe_classified(3, FeedbackObservation::Compared(comparison(120)));
    let corrected = monitor.publish(Some(0)).unwrap();
    corrected.activate();
    assert_eq!(corrected.margin(&[7; 32]), 25);
    assert_eq!(corrected.margin(&[8; 32]), 0);
    let before = serde_json::to_vec(&monitor.state).unwrap();
    monitor.observe_classified(4, outside_route());
    assert_eq!(serde_json::to_vec(&monitor.state).unwrap(), before);
    assert!(Arc::ptr_eq(&corrected, &monitor.current_view()));
    assert!(monitor.publish(Some(0)).is_none());
    assert!(corrected.current());

    let audit = monitor.audit();
    assert_eq!(audit.outside_route_observations, 2);
    assert_eq!(audit.outside_catalog_observations, 0);
    assert_eq!(audit.outside_support_observations, 0);
    assert_eq!(audit.uncomparable_observations, 0);
    assert_eq!(audit.failed_or_partial, 0);
    assert_eq!(audit.compared, 2);
    assert_eq!(audit.corrections, 1);
    assert_eq!(audit.scopes[0].compared, 2);
    assert_eq!(audit.scopes[1].compared, 0);
    assert_eq!(audit.revoked, None);
    let wire = serde_json::to_value(audit).unwrap();
    assert_eq!(wire["outside_route_observations"], 2);
    monitor.finish();
}

#[test]
fn automatic_outside_route_enforces_clock_lag_ordinal_and_counter_bounds() {
    for (ordinal, observed_at_ns, consumed_at_ns, expected) in [
        (0, 1, 2, Some(Revocation::IdentityOrClock)),
        (1, 2, 1, Some(Revocation::IdentityOrClock)),
        (1, 1, 52, Some(Revocation::ObservationLag)),
        (1, 1, 51, None),
    ] {
        let mut monitor = monitor(&policy());
        let original = monitor.current_view();
        monitor.observe_classified(
            ordinal,
            FeedbackObservation::OutsideRoute {
                observed_at_ns,
                consumed_at_ns,
            },
        );
        let audit = monitor.audit();
        assert_eq!(audit.revoked, expected);
        assert_eq!(
            audit.outside_route_observations,
            u64::from(expected.is_none())
        );
        assert_eq!(audit.compared, 0);
        assert_eq!(audit.uncomparable_observations, 0);
        assert_eq!(original.current(), expected.is_none());
        if expected.is_none() {
            monitor.observe_classified(ordinal, outside_route());
            assert_eq!(monitor.audit().revoked, Some(Revocation::IdentityOrClock));
            assert_eq!(monitor.audit().outside_route_observations, 1);
            assert!(!original.current());
        }
        monitor.finish();
    }

    let mut monitor = monitor(&policy());
    monitor.outside_route_observations = u64::MAX;
    let original = monitor.current_view();
    monitor.observe_classified(1, outside_route());
    assert_eq!(monitor.audit().revoked, Some(Revocation::Arithmetic));
    assert_eq!(monitor.audit().outside_route_observations, u64::MAX);
    assert!(!original.current());
    monitor.finish();
}

#[test]
fn automatic_outside_route_does_not_exempt_unknown_failure_queue_loss_or_worker_stop() {
    let mut p = policy();
    p.maximum_uncomparable_observations = 0;
    p.maximum_failed_or_partial = 0;
    p.maximum_queue_drops = 0;
    for expected in [
        Revocation::UncomparableObservation,
        Revocation::FailedOrPartial,
        Revocation::QueueLoss,
        Revocation::WorkerStopped,
    ] {
        let mut monitor = monitor(&p);
        let original = monitor.current_view();
        monitor.observe_classified(1, outside_route());
        match expected {
            Revocation::UncomparableObservation => {
                monitor.observe_classified(2, FeedbackObservation::Uncomparable);
                assert_eq!(monitor.audit().uncomparable_observations, 1);
            }
            Revocation::FailedOrPartial => {
                monitor.observe_classified(2, FeedbackObservation::FailedOrPartial);
                assert_eq!(monitor.audit().failed_or_partial, 1);
            }
            Revocation::QueueLoss => {
                assert!(monitor.publish(Some(1)).is_some());
                assert_eq!(monitor.audit().queue_drops, 1);
            }
            Revocation::WorkerStopped => monitor.worker_stopped_unclean(),
            _ => unreachable!(),
        }
        let audit = monitor.audit();
        assert_eq!(audit.revoked, Some(expected));
        assert_eq!(audit.outside_route_observations, 1);
        assert_eq!(audit.outside_catalog_observations, 0);
        assert_eq!(audit.outside_support_observations, 0);
        assert_eq!(audit.compared, 0);
        assert!(!original.current());
        monitor.finish();
    }
}
