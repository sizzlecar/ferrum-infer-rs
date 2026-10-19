use super::*;
use crate::test_support::resource_support::{
    configured_driver, isolated_static_initialization_plan, transaction, TestDriver, Trace,
};
use std::sync::{Arc, Mutex};

// These two tests exercise the one existing fallback domain. Their actual
// resource/capacity domains are separately allocated by the fixture.
use super::STATIC_PROVISION_CLEANUP_TEST as FALLBACK_TEST;

fn failed_with_release_fault(
    panic: bool,
) -> (
    StaticInitializationFailure<TestDriver>,
    ExecutionPlan,
    Arc<Mutex<Trace>>,
    ResourceId,
) {
    let (family, plan) = isolated_static_initialization_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let committed = transaction(&plan, driver, "release-fault")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let reservations = committed.reservations().reservations();
    for weight in ["resource.weight.left", "resource.weight.right"] {
        assert!(reservations
            .iter()
            .any(|entry| entry.resource_id().as_str() == weight));
    }
    let second = reservations[1].resource_id().clone();
    {
        let mut trace = trace.lock().unwrap();
        if panic {
            trace.release_panics.insert(second.clone());
        } else {
            trace.failures.insert(format!("release:{second}"), 1);
        }
    }
    struct BadSource;
    impl WeightComponentSource for BadSource {
        fn component<'a>(
            &'a self,
            _component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            Err(VNextError::InvalidExecutionPlan {
                reason: "original static source failed before upload".into(),
            })
        }
    }
    let failure = committed
        .initialize_static(
            &family,
            &plan,
            &BadSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .err()
        .expect("real static initialization must report the source error");
    assert!(!failure.is_indeterminate());
    (failure, plan, trace, second)
}

#[test]
fn static_retirement_drop_completes_original_partial_release_without_repeating_prefix() {
    let _serial = FALLBACK_TEST.lock().unwrap();
    let (mut failure, plan, trace, second) = failed_with_release_fault(false);
    let allocations = plan.payload().memory().static_allocations().len();
    let cause = failure.failure().clone();
    assert!(failure.retire_in_place().is_err());
    assert_eq!(failure.failure(), &cause);
    assert!(!failure.is_retired());
    let retained_bytes = failure
        .transaction
        .as_ref()
        .unwrap()
        .maintenance_controller()
        .status()
        .unwrap()
        .budget_claimed_bytes();
    assert!(retained_bytes > 0);
    assert_eq!(trace.lock().unwrap().buffer_drops, 1);
    let before = static_initialization_cleanup_status();
    drop(failure);
    assert_eq!(
        trace.lock().unwrap().buffer_drops,
        1,
        "dropping the failure cannot release its unknown suffix"
    );
    assert_eq!(
        static_initialization_cleanup_status().submitted_total(),
        before.submitted_total() + 1
    );
    // A peer on this exact device sees the original capacity claim after Drop.
    let (peer_driver, _) = configured_driver(&plan, &[], &[]);
    let peer = transaction(&plan, peer_driver, "peer")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let peer_bytes = peer
        .maintenance_controller()
        .status()
        .unwrap()
        .budget_claimed_bytes();
    assert_eq!(
        peer.maintenance_controller()
            .status()
            .unwrap()
            .process_claimed_bytes(),
        peer_bytes + retained_bytes
    );
    let receipt =
        maintain_static_initialization_cleanups(MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS)
            .unwrap();
    assert!(receipt.completed() >= 1);
    assert_eq!(trace.lock().unwrap().buffer_drops, allocations as u64);
    assert_eq!(
        peer.maintenance_controller()
            .status()
            .unwrap()
            .process_claimed_bytes(),
        peer_bytes
    );
    let calls = &trace.lock().unwrap().calls;
    let releases: Vec<_> = calls
        .iter()
        .filter(|call| call.starts_with("release:"))
        .collect();
    assert_eq!(releases.len(), allocations + 1);
    assert_eq!(
        releases
            .iter()
            .filter(|call| call.as_str() == format!("release:{second}"))
            .count(),
        2
    );
    drop(peer.release().unwrap());
}

#[test]
fn static_retirement_drop_quarantines_release_panic_with_original_capacity_and_buffers() {
    let _serial = FALLBACK_TEST.lock().unwrap();
    let (mut failure, plan, trace, _second) = failed_with_release_fault(true);
    let cause = failure.failure().clone();
    assert_eq!(
        failure.retire_in_place().unwrap_err().code(),
        "static_retirement_release_panic"
    );
    assert_eq!(failure.failure(), &cause);
    let retained_bytes = failure
        .transaction
        .as_ref()
        .unwrap()
        .maintenance_controller()
        .status()
        .unwrap()
        .budget_claimed_bytes();
    assert!(retained_bytes > 0);
    assert_eq!(trace.lock().unwrap().buffer_drops, 1);
    let before_calls = trace.lock().unwrap().calls.clone();
    assert!(failure.retire_in_place().is_err());
    assert_eq!(
        trace.lock().unwrap().calls,
        before_calls,
        "unknown release side effects cannot be replayed"
    );
    drop(failure);
    assert_eq!(trace.lock().unwrap().buffer_drops, 1);
    let (peer_driver, _) = configured_driver(&plan, &[], &[]);
    let peer = transaction(&plan, peer_driver, "poison-peer")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let peer_bytes = peer
        .maintenance_controller()
        .status()
        .unwrap()
        .budget_claimed_bytes();
    let receipt =
        maintain_static_initialization_cleanups(MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS)
            .unwrap();
    assert!(receipt.status_after().quarantined() >= 1);
    assert_eq!(trace.lock().unwrap().calls, before_calls);
    assert_eq!(trace.lock().unwrap().buffer_drops, 1);
    assert_eq!(
        peer.maintenance_controller()
            .status()
            .unwrap()
            .process_claimed_bytes(),
        peer_bytes + retained_bytes
    );
    drop(peer.release().unwrap());
}

#[test]
fn static_retirement_fresh_quiescent_drop_releases_inline_without_global_task() {
    let _serial = FALLBACK_TEST.lock().unwrap();
    let (failure, plan, trace, _) = failed_with_release_fault(false);
    let allocations = plan.payload().memory().static_allocations().len();
    trace.lock().unwrap().failures.clear();
    let before = static_initialization_cleanup_status();
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    drop(failure);
    assert_eq!(trace.lock().unwrap().buffer_drops, allocations as u64);
    assert_eq!(trace.lock().unwrap().runtime_synchronize_calls, 0);
    assert_eq!(
        static_initialization_cleanup_status().submitted_total(),
        before.submitted_total()
    );
}

fn initialized_for_activation_test() -> (
    InitializedResourceTransaction<TestDriver>,
    Arc<Mutex<Trace>>,
    usize,
) {
    let (family, plan) = isolated_static_initialization_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let committed = transaction(&plan, driver, "activation-retirement")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let allocations = committed.reservations().reservations().len();
    struct Source;
    impl WeightComponentSource for Source {
        fn component<'a>(
            &'a self,
            component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            WeightComponentPayload::new(
                component,
                component.external_names[0].clone(),
                "model.safetensors",
                component.dimensions.clone(),
                component.physical_element_type(),
                vec![0; usize::try_from(component.physical_bytes()?).unwrap()],
            )
        }
    }
    let initialized = committed
        .initialize_static(
            &family,
            &plan,
            &Source,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert!(trace.lock().unwrap().submit_calls > 0);
    (initialized, trace, allocations)
}

#[test]
fn static_retirement_initialized_activation_error_releases_original_without_reupload() {
    let (initialized, trace, allocations) = initialized_for_activation_test();
    let submitted = trace.lock().unwrap().submit_calls;
    let uploaded = trace.lock().unwrap().upload_calls;
    let mut original = initialized.abort_activation(VNextError::InvalidExecutionPlan {
        reason: "original activation attribution rejected".into(),
    });
    assert!(!original.is_indeterminate());
    let cause = original.failure().clone();
    original.retire_in_place().unwrap();
    assert!(original.is_retired());
    assert_eq!(original.failure(), &cause);
    assert_eq!(trace.lock().unwrap().buffer_drops, allocations as u64);
    assert_eq!(trace.lock().unwrap().submit_calls, submitted);
    assert_eq!(trace.lock().unwrap().upload_calls, uploaded);
    assert_eq!(trace.lock().unwrap().runtime_synchronize_calls, 0);
    assert!(trace.lock().unwrap().abandon.is_empty());
}

#[test]
fn static_retirement_initialized_handoff_error_preserves_real_committed_owner() {
    let (mut initialized, trace, allocations) = initialized_for_activation_test();
    // This is a real lease-policy change: handoff requires every lease Active.
    // It does not alter the physical allocation or its completed upload proof.
    initialized.transaction.cancel_all().unwrap();
    let submitted = trace.lock().unwrap().submit_calls;
    let mut original = initialized
        .into_plan_runtime()
        .err()
        .expect("cancelled lease cannot be activated");
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    assert!(!original.is_indeterminate());
    original.retire_in_place().unwrap();
    assert!(original.is_retired());
    assert_eq!(trace.lock().unwrap().buffer_drops, allocations as u64);
    assert_eq!(trace.lock().unwrap().submit_calls, submitted);
    assert!(trace.lock().unwrap().abandon.is_empty());
}
