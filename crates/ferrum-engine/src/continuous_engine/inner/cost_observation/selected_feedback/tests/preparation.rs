//! Monitor boundary. Production can classify preparation only from an
//! original private call and its complete validated settlement witness.
use super::*;

fn monitor() -> Monitor {
    let mut settings = policy();
    settings.maximum_uncomparable_observations = 0;
    settings.maximum_failed_or_partial = 0;
    Monitor::open_bound(
        &settings,
        &Storage::MemoryOnly,
        binding(),
        1,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32]].into()),
    )
    .unwrap()
}

#[test]
fn outside_preparation_preserves_model_but_never_exempts_ordinary_unknown_or_failure() {
    for (ordinary, expected) in [
        (
            FeedbackObservation::Uncomparable,
            Revocation::UncomparableObservation,
        ),
        (
            FeedbackObservation::FailedOrPartial,
            Revocation::FailedOrPartial,
        ),
    ] {
        let mut monitor = monitor();
        let original = monitor.current_view();
        let state = serde_json::to_vec(&monitor.state).unwrap();
        monitor.observe_classified(
            1,
            FeedbackObservation::OutsidePreparation {
                observed_at_ns: 1,
                consumed_at_ns: 2,
            },
        );
        assert!(original.current());
        assert!(Arc::ptr_eq(&original, &monitor.current_view()));
        assert_eq!(serde_json::to_vec(&monitor.state).unwrap(), state);
        assert_eq!(monitor.audit().outside_preparation_observations, 1);
        assert_eq!(monitor.audit().compared, 0);
        assert!(monitor.publish(Some(0)).is_none());
        monitor.observe_classified(2, ordinary);
        let audit = monitor.audit();
        assert_eq!(audit.revoked, Some(expected));
        assert!(!original.current());
        assert_eq!(audit.outside_preparation_observations, 1);
        assert_eq!(
            audit.uncomparable_observations,
            u64::from(expected == Revocation::UncomparableObservation)
        );
        assert_eq!(
            audit.failed_or_partial,
            u64::from(expected == Revocation::FailedOrPartial)
        );
        monitor.finish();
    }
}

#[test]
fn outside_preparation_still_requires_monotonic_identity_clock_and_bounded_lag() {
    for (ordinal, observed_at_ns, consumed_at_ns, expected) in [
        (0, 1, 2, Some(Revocation::IdentityOrClock)),
        (1, 2, 1, Some(Revocation::IdentityOrClock)),
        (1, 1, 52, Some(Revocation::ObservationLag)),
        (1, 1, 51, None),
    ] {
        let mut monitor = monitor();
        let original = monitor.current_view();
        monitor.observe_classified(
            ordinal,
            FeedbackObservation::OutsidePreparation {
                observed_at_ns,
                consumed_at_ns,
            },
        );
        let audit = monitor.audit();
        assert_eq!(audit.revoked, expected);
        assert_eq!(
            audit.outside_preparation_observations,
            u64::from(expected.is_none())
        );
        assert_eq!(audit.compared, 0);
        assert_eq!(original.current(), expected.is_none());
        if expected.is_none() {
            monitor.observe_classified(
                ordinal,
                FeedbackObservation::OutsidePreparation {
                    observed_at_ns,
                    consumed_at_ns,
                },
            );
            assert_eq!(monitor.audit().revoked, Some(Revocation::IdentityOrClock));
            assert!(!original.current());
        }
        monitor.finish();
    }
}
