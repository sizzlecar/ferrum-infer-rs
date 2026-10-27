use super::*;
use std::task::Wake;

fn account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            100,
            HostMemoryClassLimits {
                input_preprocess_bytes: 100,
                transfer_staging_bytes: 100,
                output_bytes: 40,
                weight_backing_bytes: 100,
            },
        )
        .unwrap(),
    )
}

#[test]
fn hard_class_limit_is_not_hidden_by_temporary_total_pressure() {
    let host = account();
    let input = host.reserve(HostMemoryClass::InputPreprocess, 90).unwrap();
    let impossible = [(HostMemoryClass::Output, 41)];
    for error in [
        host.check_scope_capacity(&impossible).unwrap_err(),
        host.check_scope_admission(&impossible).unwrap_err(),
        host.reserve_scope(&impossible).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            HostMemoryError::Capacity {
                scope: HostMemoryPressureScope::Class(HostMemoryClass::Output),
                requested: 41,
                limit: 40,
                ..
            }
        ));
        assert!(!error.is_temporary_capacity());
    }
    let possible = [(HostMemoryClass::Output, 20)];
    host.check_scope_capacity(&possible).unwrap();
    let error = host.check_scope_admission(&possible).unwrap_err();
    assert!(error.is_temporary_capacity());
    assert!(matches!(
        error,
        HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Total,
            requested: 20,
            committed: 90,
            limit: 100,
        }
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 90);
    drop(input);
    host.check_scope_admission(&possible).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(
        host.reserve_scope(&possible)
            .unwrap()
            .account()
            .usage()
            .unwrap()
            .committed_bytes(),
        20
    );
}

struct ReadUsageOnWake {
    account: HostMemoryAccount,
    committed: Mutex<Vec<usize>>,
}

impl Wake for ReadUsageOnWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // This must run after the account mutex has been released.
        let committed = self.account.usage().unwrap().committed_bytes();
        self.committed.lock().unwrap().push(committed);
    }
}

#[test]
fn scope_close_and_final_backing_release_wake_after_refund_and_allow_unsubscribe() {
    let host = account();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, 40)])
        .unwrap();
    let output = scope
        .account()
        .allocate_zeroed(HostMemoryClass::Output, 30)
        .unwrap()
        .freeze();
    let probe = Arc::new(ReadUsageOnWake {
        account: host.clone(),
        committed: Mutex::new(Vec::new()),
    });
    let subscription = host.subscribe_capacity(Waker::from(probe.clone()));
    assert!(host
        .check_scope_admission(&[(HostMemoryClass::Output, 20)])
        .unwrap_err()
        .is_temporary_capacity());
    scope.close().unwrap();
    assert_eq!(*probe.committed.lock().unwrap(), [30]);
    assert!(host
        .check_scope_admission(&[(HostMemoryClass::Output, 20)])
        .is_err());
    let retained = output.clone();
    drop(output);
    assert_eq!(*probe.committed.lock().unwrap(), [30]);
    drop(retained);
    assert_eq!(*probe.committed.lock().unwrap(), [30, 0]);
    host.check_scope_admission(&[(HostMemoryClass::Output, 20)])
        .unwrap();
    drop(subscription);
    drop(host.reserve(HostMemoryClass::Output, 20).unwrap());
    assert_eq!(*probe.committed.lock().unwrap(), [30, 0]);
}

#[test]
fn reservation_cancellation_wakes_before_any_backing_was_allocated() {
    let host = account();
    let held = host.reserve(HostMemoryClass::Output, 40).unwrap();
    let probe = Arc::new(ReadUsageOnWake {
        account: host.clone(),
        committed: Mutex::new(Vec::new()),
    });
    let _subscription = host.subscribe_capacity(Waker::from(probe.clone()));
    assert!(host
        .reserve_scope(&[(HostMemoryClass::Output, 1)])
        .unwrap_err()
        .is_temporary_capacity());
    drop(held);
    assert_eq!(*probe.committed.lock().unwrap(), [0]);
    host.reserve_scope(&[(HostMemoryClass::Output, 40)])
        .unwrap();
}

fn mixed_account() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            200,
            HostMemoryClassLimits {
                input_preprocess_bytes: 100,
                transfer_staging_bytes: 40,
                output_bytes: 100,
                weight_backing_bytes: 100,
            },
        )
        .unwrap(),
    )
}

#[test]
fn output_release_check_keeps_selected_encoding_credit_in_hard_requirement() {
    let host = mixed_account();
    let response = host
        .reserve_scope(&[(HostMemoryClass::Output, 60)])
        .unwrap();
    let raw = [(HostMemoryClass::Output, 50)];
    host.check_scope_capacity(&raw).unwrap();
    assert!(host
        .check_scope_admission(&raw)
        .unwrap_err()
        .is_temporary_capacity());
    let error = host.check_scope_after_output_release(&raw, 60).unwrap_err();
    assert_eq!(
        error,
        HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Class(HostMemoryClass::Output),
            requested: 110,
            committed: 0,
            limit: 100,
        }
    );
    assert!(!error.is_temporary_capacity());
    assert_eq!(host.usage().unwrap().committed_bytes(), 60);
    assert_eq!(
        host.check_scope_after_output_release(&raw, 61),
        Err(HostMemoryError::InvalidReservation)
    );
    drop(response);
}

#[test]
fn output_release_check_keeps_queue_input_and_warm_weight_owners() {
    for class in [
        HostMemoryClass::InputPreprocess,
        HostMemoryClass::WeightBacking,
    ] {
        let host = mixed_account();
        let held = host.reserve(class, 90).unwrap();
        let external_output = host.reserve(HostMemoryClass::Output, 100).unwrap();
        let demand = [(class, 20)];
        host.check_scope_capacity(&demand).unwrap();
        let error = host
            .check_scope_after_output_release(&demand, 0)
            .unwrap_err();
        assert_eq!(
            error,
            HostMemoryError::Capacity {
                scope: HostMemoryPressureScope::Class(class),
                requested: 110,
                committed: 0,
                limit: 100,
            }
        );
        assert!(!error.is_temporary_capacity());
        assert_eq!(host.usage().unwrap().committed_bytes(), 190);
        drop((held, external_output));
    }
}

#[test]
fn output_release_check_waits_only_for_external_output_that_can_make_room() {
    let host = mixed_account();
    let input = host.reserve(HostMemoryClass::InputPreprocess, 70).unwrap();
    let weights = host.reserve(HostMemoryClass::WeightBacking, 40).unwrap();
    let response = host
        .reserve_scope(&[(HostMemoryClass::Output, 20)])
        .unwrap();
    let external_output = host.reserve(HostMemoryClass::Output, 60).unwrap();
    let demand = [
        (HostMemoryClass::Output, 40),
        (HostMemoryClass::TransferStaging, 10),
    ];
    host.check_scope_after_output_release(&demand, 20).unwrap();
    assert!(host
        .check_scope_admission(&demand)
        .unwrap_err()
        .is_temporary_capacity());
    assert_eq!(host.usage().unwrap().committed_bytes(), 190);
    drop(external_output);
    host.check_scope_admission(&demand).unwrap();
    let execution = host.reserve_scope(&demand).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 180);
    drop((input, weights, response, execution));
}

#[test]
fn output_release_check_applies_total_ceiling_across_retained_classes() {
    let host = mixed_account();
    let inputs = host.reserve(HostMemoryClass::InputPreprocess, 90).unwrap();
    let weights = host.reserve(HostMemoryClass::WeightBacking, 90).unwrap();
    let response = host
        .reserve_scope(&[(HostMemoryClass::Output, 10)])
        .unwrap();
    let demand = [(HostMemoryClass::Output, 20)];
    let error = host
        .check_scope_after_output_release(&demand, 10)
        .unwrap_err();
    assert_eq!(
        error,
        HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Total,
            requested: 210,
            committed: 0,
            limit: 200,
        }
    );
    assert!(!error.is_temporary_capacity());
    drop((inputs, weights, response));
}
