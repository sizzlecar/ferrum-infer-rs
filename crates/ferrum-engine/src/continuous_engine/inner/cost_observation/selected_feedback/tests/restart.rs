//! State/store boundary tests. Catalog equivalence is separately established
//! by the non-deserializable profile proof and real source replay integration.
use super::super::{RestartFeedbackAllowance, RestartFeedbackBudget};
use super::*;

fn allowance(budget: RestartFeedbackBudget) -> RestartFeedbackAllowance {
    RestartFeedbackAllowance {
        transient_bytes: budget.maximum_transient_bytes,
        disk_bytes: budget.maximum_disk_bytes,
    }
}
fn scopes() -> Arc<[[u8; 32]]> {
    vec![[7; 32], [8; 32]].into()
}
fn original() -> Monitor {
    Monitor::open_bound(
        &policy(),
        &Storage::MemoryOnly,
        binding(),
        2,
        FeedbackKind::StructuredV2,
        Some(scopes()),
    )
    .unwrap()
}
fn file_binding() -> Binding {
    Binding {
        profile_sha256: [9; 32],
        ..binding()
    }
}
fn populated() -> Monitor {
    let mut monitor = original();
    monitor.observe_classified(1, FeedbackObservation::Compared(comparison(120)));
    monitor.observe_classified(2, FeedbackObservation::Compared(comparison(120)));
    monitor.publish(Some(0)).unwrap().activate();
    let outside = [
        FeedbackObservation::OutsideCatalog {
            observed_at_ns: 1,
            consumed_at_ns: 2,
        },
        FeedbackObservation::OutsideSupport {
            observed_at_ns: 1,
            consumed_at_ns: 2,
        },
        FeedbackObservation::OutsideRoute {
            observed_at_ns: 1,
            consumed_at_ns: 2,
        },
        FeedbackObservation::ProvenNoSubmission {
            observed_at_ns: 1,
            consumed_at_ns: 2,
        },
        FeedbackObservation::OutsidePreparation {
            observed_at_ns: 1,
            consumed_at_ns: 2,
        },
    ];
    for (i, value) in outside.into_iter().enumerate() {
        monitor.observe_classified(i as u64 + 3, value);
    }
    // Keep one original residual window item as well as an existing margin.
    monitor.observe_classified(8, FeedbackObservation::Compared(comparison(130)));
    monitor.publish(Some(1));
    monitor
}

#[test]
fn restart_feedback_handoff_preserves_margin_windows_counters_and_original_fifo() {
    let directory = Directory::new();
    let mut monitor = populated();
    let before = serde_json::to_value(&monitor.state).unwrap();
    let old_view = monitor.current_view();
    let receipt = monitor
        .finish_bound_for_restart(
            &binding(),
            &file_binding(),
            &scopes(),
            &directory.path(),
            11,
            allowance(monitor.restart_budget(&directory.path()).unwrap()),
            || Ok(()),
        )
        .unwrap();
    assert!(!old_view.current());
    let bytes = fs::read(&receipt.path).unwrap();
    assert_eq!(receipt.bytes, bytes.len() as u64);
    assert_eq!(receipt.sha256, <[u8; 32]>::from(Sha256::digest(&bytes)));
    let mut resumed = Monitor::resume_for_restart(
        &policy(),
        &directory.path(),
        file_binding(),
        scopes(),
        allowance(Monitor::restart_resume_budget(&policy(), &directory.path()).unwrap()),
    )
    .unwrap();
    assert!(
        !resumed.current_view().current(),
        "coordinator must install the new epoch before activation"
    );
    assert_eq!(resumed.current_view().margin(&[7; 32]), 25);
    let after = serde_json::to_value(&resumed.state).unwrap();
    for field in [
        "families",
        "compared",
        "corrections",
        "uncomparable_observations",
        "failed_or_partial",
        "queue_drops",
        "revoked",
    ] {
        assert_eq!(
            before[field], after[field],
            "original state changed: {field}"
        );
    }
    assert_eq!(resumed.state.epoch, monitor.state.epoch + 1);
    assert_eq!(resumed.state.session, monitor.state.session + 1);
    let history = resumed.audit().previous_process.unwrap();
    assert_eq!(
        (
            history.previous_processed_fifo,
            history.previous_feedback_fifo,
            history.previous_queue_drops
        ),
        (11, 8, 1)
    );
    let audit = resumed.audit();
    assert_eq!(
        (
            audit.outside_catalog_observations,
            audit.outside_support_observations,
            audit.outside_route_observations,
            audit.no_submission_observations,
            audit.outside_preparation_observations
        ),
        (1, 1, 1, 1, 1)
    );
    assert_eq!((resumed.last_ordinal, resumed.last_drops), (0, 0));
    resumed.observe_classified(1, FeedbackObservation::Compared(comparison(130)));
    assert_eq!(
        resumed.audit().revoked,
        None,
        "old FIFO cannot reject a new process's first original observation"
    );
    assert_eq!(resumed.audit().compared, monitor.audit().compared + 1);
    resumed.observe_classified(1, FeedbackObservation::NotSubmitted);
    assert_eq!(
        resumed.audit().revoked,
        Some(Revocation::IdentityOrClock),
        "new process retains its own no-repeat gate"
    );
    resumed.finish();
}

#[test]
fn restart_feedback_handoff_preserves_sticky_revocation_instead_of_base_known() {
    let directory = Directory::new();
    let mut monitor = populated();
    monitor.observe_classified(9, FeedbackObservation::FailedOrPartial);
    monitor.observe_classified(10, FeedbackObservation::FailedOrPartial);
    monitor.publish(Some(1));
    assert_eq!(monitor.audit().revoked, Some(Revocation::FailedOrPartial));
    let allowed = allowance(monitor.restart_budget(&directory.path()).unwrap());
    monitor
        .finish_bound_for_restart(
            &binding(),
            &file_binding(),
            &scopes(),
            &directory.path(),
            12,
            allowed,
            || Ok(()),
        )
        .unwrap();
    let mut resumed = Monitor::resume_for_restart(
        &policy(),
        &directory.path(),
        file_binding(),
        scopes(),
        allowance(Monitor::restart_resume_budget(&policy(), &directory.path()).unwrap()),
    )
    .unwrap();
    resumed.current_view().activate();
    assert!(!resumed.current_view().current());
    assert_eq!(resumed.audit().revoked, Some(Revocation::FailedOrPartial));
    assert_eq!(resumed.audit().failed_or_partial, 2);
    assert_eq!(resumed.current_view().margin(&[7; 32]), 25);
    resumed.finish();
}

#[test]
fn restart_feedback_handoff_rejects_changed_source_bad_cut_and_budget_before_io() {
    for kind in 0..4 {
        let directory = Directory::new();
        let mut monitor = populated();
        let mut allowed = allowance(monitor.restart_budget(&directory.path()).unwrap());
        let mut target = file_binding();
        let cut = if kind == 0 { 7 } else { 11 };
        if kind == 1 {
            target.source_sha256[0] ^= 1;
        }
        if kind == 2 {
            allowed.transient_bytes -= 1;
        }
        if kind == 3 {
            allowed.disk_bytes -= 1;
        }
        let old = monitor.current_view();
        assert!(monitor
            .finish_bound_for_restart(
                &binding(),
                &target,
                &scopes(),
                &directory.path(),
                cut,
                allowed,
                || Ok(())
            )
            .is_err());
        assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
        assert!(Arc::ptr_eq(&old, &monitor.current_view()));
        assert_eq!(monitor.state.binding, binding());
    }
}

#[test]
fn restart_feedback_handoff_incomplete_validation_keeps_dirty_receipt_unloadable() {
    let directory = Directory::new();
    let mut monitor = populated();
    let allowed = allowance(monitor.restart_budget(&directory.path()).unwrap());
    assert!(monitor
        .finish_bound_for_restart(
            &binding(),
            &file_binding(),
            &scopes(),
            &directory.path(),
            11,
            allowed,
            || Err(FerrumError::config(
                "original source expired during cold write"
            ))
        )
        .is_err());
    assert!(Monitor::resume_for_restart(
        &policy(),
        &directory.path(),
        file_binding(),
        scopes(),
        allowance(Monitor::restart_resume_budget(&policy(), &directory.path()).unwrap())
    )
    .is_err());
    assert_eq!(monitor.state.binding, binding());
    assert!(!monitor.current_view().current());
}

#[test]
fn restart_feedback_handoff_absent_history_does_not_upgrade_legacy_feedback_receipt() {
    let directory = Directory::new();
    let (mut store, state) = store::Store::open(
        &Storage::CreateNew {
            path: directory.path(),
        },
        &policy(),
        file_binding(),
        2,
    )
    .unwrap();
    assert!(!serde_json::to_value(&state)
        .unwrap()
        .as_object()
        .unwrap()
        .contains_key("restart_progress"));
    store.finish(&state).unwrap();
    assert!(Monitor::resume_for_restart(
        &policy(),
        &directory.path(),
        file_binding(),
        scopes(),
        allowance(Monitor::restart_resume_budget(&policy(), &directory.path()).unwrap())
    )
    .is_err());
}
