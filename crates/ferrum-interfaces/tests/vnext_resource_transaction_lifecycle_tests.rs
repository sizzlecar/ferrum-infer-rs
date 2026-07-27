#![allow(unused_imports)]

use ferrum_interfaces::vnext::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, Weak};
use std::time::Duration;

#[path = "vnext_resource_contract/support.rs"]
mod support;
use support::*;

fn admission_and_success(plan: &ExecutionPlan, passed: &mut usize) {
    let request_a: RequestIdentity = id("request.provision.admission-a");
    let request_b: RequestIdentity = id("request.provision.admission-b");
    let permit_a = admit_resources(plan, request_a.clone()).unwrap();
    let generation_a = permit_a.binding().admission_generation();
    check(
        passed,
        permit_a.binding().plan_id() == plan.payload().plan_id(),
    );
    check(passed, permit_a.binding().plan_hash() == plan.plan_hash());
    check(
        passed,
        permit_a.binding().pool_identity().plan_id() == plan.payload().plan_id()
            && permit_a.binding().pool_identity().plan_hash() == plan.plan_hash()
            && permit_a.binding().pool_identity().device_id() == plan.payload().device_id(),
    );
    check(
        passed,
        permit_a.binding().pool_identity().admission_generation() == generation_a
            && permit_a.binding().pool_id().get() == generation_a,
    );
    check(
        passed,
        permit_a.binding().admitted_bytes() == plan.payload().memory().static_bytes(),
    );
    check(
        passed,
        permit_a.binding().usable_capacity_bytes()
            == plan.payload().memory().usable_capacity_bytes(),
    );
    check(
        passed,
        permit_a.binding().maximum_active_sequences()
            == plan.payload().memory().maximum_active_sequences(),
    );
    check(
        passed,
        permit_a.reservations().reservations().len()
            == plan.payload().memory().static_allocations().len(),
    );
    let permit_b = admit_resources(plan, request_b).unwrap();
    check(
        passed,
        permit_b.binding().admission_generation() > generation_a,
    );

    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut wrong_device = driver.clone();
    wrong_device.device_id = id("device.reference.other");
    let wrong_device_identity = ResourceTransactionIdentity::for_admission(
        permit_a.binding(),
        id("run.wrong-device"),
        id("transaction.wrong-device"),
    );
    check(
        passed,
        ResourceTransaction::begin(wrong_device, wrong_device_identity, permit_a).is_err(),
    );

    let request_capacity: RequestIdentity = id("request.provision.wrong-capacity");
    let permit_capacity = admit_resources(plan, request_capacity.clone()).unwrap();
    let mut wrong_capacity = driver.clone();
    wrong_capacity.device_capacity_bytes -= 1;
    let wrong_capacity_identity = ResourceTransactionIdentity::for_admission(
        permit_capacity.binding(),
        id("run.wrong-capacity"),
        id("transaction.wrong-capacity"),
    );
    check(
        passed,
        ResourceTransaction::begin(wrong_capacity, wrong_capacity_identity, permit_capacity)
            .is_err(),
    );

    let request_identity: RequestIdentity = id("request.provision.identity-mismatch");
    let permit_identity = admit_resources(plan, request_identity).unwrap();
    let other_permit = admit_resources(plan, id("request.provision.other-pool")).unwrap();
    let mismatched_identity = ResourceTransactionIdentity::for_admission(
        other_permit.binding(),
        id("run.identity-mismatch"),
        id("transaction.identity-mismatch"),
    );
    check(
        passed,
        ResourceTransaction::begin(driver.clone(), mismatched_identity, permit_identity).is_err(),
    );
    drop(other_permit);

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "success").reserve().unwrap();
    let resource_count = plan.payload().memory().static_allocations().len();
    check(
        passed,
        reserved
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Reserved),
    );
    check(passed, reserved.receipts().len() == 1);
    check(
        passed,
        reserved.receipts()[0].records().len() == resource_count,
    );
    check(
        passed,
        reserved.latest_transition_validation_context().is_some(),
    );
    let committed = reserved.commit().unwrap();
    check(
        passed,
        committed.identity().pool_id() == committed.admission().pool_id(),
    );
    check(
        passed,
        committed.lease().entries().count() == resource_count,
    );
    check(
        passed,
        committed
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Committed),
    );
    let first = committed.lease().entries().next().unwrap();
    let first_id = first.resource_id().clone();
    let first_generation = first.generation();
    check(
        passed,
        plan.payload()
            .memory()
            .static_allocations()
            .iter()
            .any(|allocation| allocation.resource_id() == &first_id),
    );
    check(
        passed,
        first_generation == committed.admission().admission_generation(),
    );
    check(
        passed,
        first.size_bytes() > 0 && first.alignment_bytes().is_power_of_two(),
    );
    let released = committed.release().unwrap();
    check(
        passed,
        released
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Released),
    );
    check(passed, trace.lock().unwrap().abandon.is_empty());
}

fn reverse_incremental_recovery(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    assert_eq!(resources.len(), 2);

    let reserve_failure = failure_key("reserve", &resources[1]);
    let undo_failure = failure_key("undo-reserve", &resources[0]);
    let failures = [(reserve_failure.as_str(), 1), (undo_failure.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let error = expect_err(transaction(plan, driver, "reserve-recovery").reserve());
    check(passed, error.failure().completed().len() == 1);
    check(
        passed,
        error.failure().ledger_after()[0].transaction_state() == ResourceTransactionState::Reserved,
    );
    check(
        passed,
        error.failure().ledger_after()[1].transaction_state() == ResourceTransactionState::New,
    );
    let error = expect_err(error.recover());
    check(passed, error.failure().compensation().is_empty());
    check(
        passed,
        error.failure().recovery_failures().len() == 1
            && error.failure().recovery_failures()[0]
                .resource()
                .unwrap()
                .resource_id()
                == &resources[0],
    );
    let recovered = error.recover().unwrap();
    check(
        passed,
        recovered
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::New),
    );
    check(
        passed,
        calls(&trace, "undo-reserve")
            == [
                failure_key("undo-reserve", &resources[0]),
                failure_key("undo-reserve", &resources[0]),
            ],
    );
    let _terminal = recovered.reserve().unwrap().rollback().unwrap();

    let commit_failure = failure_key("commit", &resources[1]);
    let undo_commit_failure = failure_key("undo-commit", &resources[0]);
    let failures = [
        (commit_failure.as_str(), 1),
        (undo_commit_failure.as_str(), 1),
    ];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let reserved = transaction(plan, driver, "commit-recovery")
        .reserve()
        .unwrap();
    let error = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Recoverable(error) => error,
        ResourceCommitTransitionError::Poisoned(_) => panic!("driver error is recoverable"),
    };
    check(passed, error.failure().completed().len() == 1);
    check(
        passed,
        error.failure().ledger_after()[0].buffer_present()
            && !error.failure().ledger_after()[1].buffer_present(),
    );
    let error = expect_err(error.recover());
    check(passed, error.failure().compensation().is_empty());
    let reserved = error.recover().unwrap();
    check(
        passed,
        reserved
            .ledger_snapshot()
            .entries()
            .iter()
            .all(|entry| !entry.buffer_present()),
    );
    check(
        passed,
        calls(&trace, "undo-commit")
            == [
                failure_key("undo-commit", &resources[0]),
                failure_key("undo-commit", &resources[0]),
            ],
    );
    let _terminal = reserved.commit().unwrap().release().unwrap();
}

fn poison_reconcile_and_quarantine(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    let malformed = resources[1].clone();

    let (driver, trace) =
        configured_driver(plan, &[], &[(malformed.clone(), InvalidCommit::Descriptor)]);
    let reserved = transaction(plan, driver, "poison-reconcile")
        .reserve()
        .unwrap();
    let poisoned = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Poisoned(poisoned) => poisoned,
        ResourceCommitTransitionError::Recoverable(_) => panic!("bad descriptor must poison"),
    };
    check(
        passed,
        poisoned.failure().recovery_strategy() == ResourceRecoveryStrategy::ReconcileOrQuarantine,
    );
    check(
        passed,
        poisoned.failure().failure_point().unwrap().resource_id() == &malformed,
    );
    check(
        passed,
        poisoned.failure().failure_point().unwrap().actual_before()
            == ResourceTransactionState::Reserved,
    );
    check(passed, poisoned.failure().completed().len() == 1);
    check(
        passed,
        poisoned.failure().ledger_after()[0].transaction_state()
            == ResourceTransactionState::Committed,
    );
    check(
        passed,
        poisoned.failure().ledger_after()[1].transaction_state()
            == ResourceTransactionState::Reserved,
    );
    let recovery = poisoned.reconcile().unwrap();
    check(
        passed,
        recovery.failure().recovery_strategy() == ResourceRecoveryStrategy::ReverseCompensation,
    );
    let reserved = recovery.recover().unwrap();
    check(passed, calls(&trace, "reconcile").len() == 1);
    check(passed, calls(&trace, "undo-commit").len() == 1);
    let _terminal = reserved.rollback().unwrap();

    let (driver, trace) =
        configured_driver(plan, &[], &[(malformed.clone(), InvalidCommit::Generation)]);
    let reserved = transaction(plan, driver, "poison-quarantine")
        .reserve()
        .unwrap();
    let poisoned = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Poisoned(poisoned) => poisoned,
        ResourceCommitTransitionError::Recoverable(_) => panic!("bad generation must poison"),
    };
    let quarantined = poisoned.quarantine().unwrap();
    check(
        passed,
        quarantined
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Quarantined),
    );
    let quarantine_receipt = quarantined.receipts().last().unwrap();
    check(
        passed,
        quarantine_receipt.records()[0].before() == ResourceTransactionState::Committed,
    );
    check(
        passed,
        quarantine_receipt.records()[1].before() == ResourceTransactionState::Reserved,
    );
    check(
        passed,
        quarantine_receipt
            .records()
            .iter()
            .all(|record| record.after() == ResourceTransactionState::Quarantined),
    );
    check(
        passed,
        quarantined
            .ledger_snapshot()
            .entries()
            .iter()
            .all(|entry| !entry.buffer_present()),
    );
    check(passed, trace.lock().unwrap().quarantine_sizes == [2]);
    check(
        passed,
        trace.lock().unwrap().quarantine_actual_mismatch == [true],
    );

    let reconcile_failure = failure_key("reconcile", &malformed);
    let failures = [(reconcile_failure.as_str(), 1)];
    let (driver, trace) =
        configured_driver(plan, &failures, &[(malformed, InvalidCommit::Descriptor)]);
    let reserved = transaction(plan, driver, "poison-reconcile-failure")
        .reserve()
        .unwrap();
    let poisoned = match expect_err(reserved.commit()) {
        ResourceCommitTransitionError::Poisoned(poisoned) => poisoned,
        ResourceCommitTransitionError::Recoverable(_) => unreachable!(),
    };
    let poisoned = expect_err(poisoned.reconcile());
    check(passed, poisoned.failure().recovery_failures().len() == 1);
    check(passed, poisoned.quarantine().is_ok());
    check(passed, calls(&trace, "quarantine").len() == 1);
}

fn forward_only_recovery(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    assert_eq!(resources.len(), 2);
    let rollback_first = failure_key("rollback", &resources[0]);
    let rollback_second = failure_key("rollback", &resources[1]);
    let failures = [(rollback_first.as_str(), 1), (rollback_second.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let reserved = transaction(plan, driver, "rollback-forward")
        .reserve()
        .unwrap();
    let error = expect_err(reserved.rollback());
    check(passed, error.failure().completed().is_empty());
    check(
        passed,
        error.failure().ledger_after()[0].transaction_state() == ResourceTransactionState::Reserved,
    );
    check(
        passed,
        error.failure().ledger_after()[1].transaction_state() == ResourceTransactionState::Reserved,
    );
    let error = expect_err(error.complete());
    check(passed, error.failure().completed().len() == 1);
    let rolled_back = error.complete().unwrap();
    check(
        passed,
        rolled_back
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::RolledBack),
    );
    check(
        passed,
        calls(&trace, "rollback")
            == [
                failure_key("rollback", &resources[0]),
                failure_key("rollback", &resources[0]),
                failure_key("rollback", &resources[1]),
                failure_key("rollback", &resources[1]),
            ],
    );

    let release_first = failure_key("release", &resources[0]);
    let release_second = failure_key("release", &resources[1]);
    let failures = [(release_first.as_str(), 1), (release_second.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let committed = transaction(plan, driver, "release-forward")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let error = expect_err(committed.release());
    let failure_anchor = error.failure().clone();
    check(passed, failure_anchor.failure_id().get() != 0);
    check(passed, error.failure().completed().is_empty());
    check(
        passed,
        error.failure().ledger_after()[0].transaction_state()
            == ResourceTransactionState::Committed,
    );
    check(
        passed,
        error.failure().ledger_after()[1].transaction_state()
            == ResourceTransactionState::Committed,
    );
    let error = expect_err(error.complete());
    check(
        passed,
        error
            .failure()
            .validate_recovery_continuation(&failure_anchor)
            .is_ok()
            && error.failure().failure_id() == failure_anchor.failure_id(),
    );
    check(passed, error.failure().completed().len() == 1);
    let released = error.complete().unwrap();
    check(
        passed,
        released
            .recovery_history()
            .last()
            .unwrap()
            .validate_recovery_continuation(&failure_anchor)
            .is_ok(),
    );
    check(
        passed,
        released
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Released),
    );
    check(
        passed,
        calls(&trace, "release")
            == [
                failure_key("release", &resources[0]),
                failure_key("release", &resources[0]),
                failure_key("release", &resources[1]),
                failure_key("release", &resources[1]),
            ],
    );
}

#[test]
fn transaction_lifecycle_contracts_are_exhaustive() {
    const EXPECTED_CASES: usize = 70;
    let plan = execution_plan();
    let mut passed = 0;
    admission_and_success(&plan, &mut passed);
    reverse_incremental_recovery(&plan, &mut passed);
    poison_reconcile_and_quarantine(&plan, &mut passed);
    forward_only_recovery(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT TRANSACTION LIFECYCLE PASS: {passed}/{EXPECTED_CASES}");
}
