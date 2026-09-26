//! The shared durable monitor must retain all declared V2 owner identities.
//! Product integration supplies actual private qualified host-settled receipts.
use super::*;

fn monitor(directory: &Directory, p: &SloSelectedFeedbackSettingsV1) -> Monitor {
    Monitor::open_bound(
        p,
        &Storage::CreateNew {
            path: directory.path(),
        },
        binding(),
        2,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32], [8; 32]].into()),
    )
    .unwrap()
}

#[test]
fn structured_feedback_declared_inventory_includes_unobserved_owners_and_updates_one_epoch() {
    let directory = Directory::new();
    let mut monitor = monitor(&directory, &policy());
    let initial = monitor.current_view();
    let audit = monitor.audit();
    assert_eq!(audit.scopes.len(), 2);
    assert!(audit
        .scopes
        .iter()
        .all(|scope| scope.compared == 0 && scope.margin_ns == 0));
    monitor.observe_classified(1, FeedbackObservation::Compared(comparison(120)));
    assert!(initial.current());
    assert!(monitor.publish(Some(0)).is_none());
    monitor.observe_classified(2, FeedbackObservation::Compared(comparison(120)));
    assert!(
        !initial.current(),
        "decision closes every old owner before durable publication"
    );
    let next = monitor.publish(Some(0)).unwrap();
    assert_eq!(next.epoch, initial.epoch + 1);
    assert!(
        !next.current(),
        "persisted view needs the runtime Arc swap before activation"
    );
    next.activate();
    assert!(next.current());
    assert!(!initial.current());
    assert_eq!(next.margin(&[7; 32]), 25);
    assert_eq!(next.margin(&[8; 32]), 0);
    let audit = monitor.audit();
    assert_eq!(audit.compared, 2);
    assert_eq!(audit.scopes[0].compared, 2);
    assert_eq!(audit.scopes[0].underestimates, 2);
    assert_eq!(audit.scopes[0].maximum_base_excess_ns, 20);
    assert_eq!(audit.scopes[1].compared, 0);
    for ordinal in 3..=5 {
        monitor.observe_classified(ordinal, FeedbackObservation::Compared(comparison(120)));
    }
    assert!(monitor.publish(Some(0)).is_none());
    assert_eq!(monitor.audit().scopes[0].underestimates, 2);
    assert_eq!(monitor.audit().scopes[0].compared, 5);
    assert_eq!(monitor.audit().corrections, 1);
    monitor.finish();
    assert!(!next.current());
}

#[test]
fn structured_feedback_unknown_owner_and_repeated_ordinals_close_the_whole_catalog() {
    for repeat_ordinal in [false, true] {
        let directory = Directory::new();
        let mut monitor = monitor(&directory, &policy());
        let old = monitor.current_view();
        monitor.observe_classified(1, FeedbackObservation::Compared(comparison(100)));
        let mut next = comparison(100);
        if !repeat_ordinal {
            next.family = [9; 32];
        }
        monitor.observe_classified(
            if repeat_ordinal { 1 } else { 2 },
            FeedbackObservation::Compared(next),
        );
        assert!(!old.current());
        let revoked = monitor.publish(Some(0)).unwrap();
        revoked.activate();
        assert!(!revoked.current());
        let audit = monitor.audit();
        assert_eq!(audit.revoked, Some(Revocation::IdentityOrClock));
        assert_eq!(audit.compared, 1);
        assert_eq!(audit.scopes.len(), 2);
        monitor.finish();
    }
}

#[test]
fn structured_feedback_queue_loss_and_persistence_failure_cannot_reauthorize_a_view() {
    for fail_persistence in [false, true] {
        let directory = Directory::new();
        let mut monitor = monitor(&directory, &policy());
        let old = monitor.current_view();
        let drops = if fail_persistence {
            monitor.observe_classified(1, FeedbackObservation::Compared(comparison(120)));
            monitor.observe_classified(2, FeedbackObservation::Compared(comparison(120)));
            fs::create_dir(directory.0.join("receipt.json.pending")).unwrap();
            0
        } else {
            2
        };
        let revoked = monitor.publish(Some(drops)).unwrap();
        assert!(!old.current());
        revoked.activate();
        assert!(!revoked.current());
        assert_eq!(
            monitor.audit().revoked,
            Some(if fail_persistence {
                Revocation::Persistence
            } else {
                Revocation::QueueLoss
            })
        );
        monitor.finish();
        if fail_persistence {
            assert!(monitor.check_finished().is_err());
            drop(monitor);
            assert!(Monitor::open_bound(
                &policy(),
                &Storage::Resume {
                    path: directory.path()
                },
                binding(),
                2,
                FeedbackKind::StructuredV2,
                Some(vec![[7; 32], [8; 32]].into())
            )
            .is_err());
        }
    }
}

#[test]
fn structured_feedback_resume_preserves_owner_margins_and_rejects_rebound_source() {
    let directory = Directory::new();
    let mut initial = monitor(&directory, &policy());
    initial.observe_classified(1, FeedbackObservation::Compared(comparison(120)));
    initial.observe_classified(2, FeedbackObservation::Compared(comparison(120)));
    let published = initial.publish(Some(0)).unwrap();
    initial.finish();
    drop(initial);
    let mut resumed = Monitor::open_bound(
        &policy(),
        &Storage::Resume {
            path: directory.path(),
        },
        binding(),
        2,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32], [8; 32]].into()),
    )
    .unwrap();
    assert_eq!(resumed.current_view().epoch, published.epoch);
    assert_eq!(resumed.current_view().margin(&[7; 32]), 25);
    assert_eq!(resumed.audit().session, 2);
    assert_eq!(resumed.audit().scopes[0].compared, 2);
    assert_eq!(resumed.audit().scopes[1].compared, 0);
    resumed.finish();
    drop(resumed);
    let mut changed = binding();
    changed.source_sha256 = [99; 32];
    assert!(Monitor::open_bound(
        &policy(),
        &Storage::Resume {
            path: directory.path()
        },
        changed,
        2,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32], [8; 32]].into())
    )
    .is_err());
}

#[test]
fn selected_feedback_old_receipt_state_without_owner_counters_keeps_canonical_bytes() {
    let mut state = State::new(binding());
    state.compare(&policy(), 1, comparison(120));
    let mut wire = serde_json::to_value(&state).unwrap();
    let family = wire["families"][0].as_object_mut().unwrap();
    family.remove("compared");
    family.remove("underestimates");
    family.remove("maximum_base_excess_ns");
    let restored: State = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), wire);
    assert!(restored.validate(&binding(), &policy(), 1));
}
