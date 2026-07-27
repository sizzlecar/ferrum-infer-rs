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

fn full_pool_retention_and_release(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "full-pool-retention")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let deferred = committed.defer_all().unwrap();
    check(passed, deferred.entries().len() == resources.len());
    check(passed, deferred.before() == ResourceLeaseState::Active);
    check(passed, deferred.after() == ResourceLeaseState::Deferred);
    check(
        passed,
        deferred.decision() == ResourceRetentionDecision::Retain,
    );
    check(
        passed,
        committed.lease().state() == ResourceLeaseState::Deferred,
    );
    check(
        passed,
        committed
            .lease()
            .entries()
            .all(|entry| entry.state() == ResourceLeaseState::Deferred),
    );
    let resumed = committed.resume_all().unwrap();
    check(passed, resumed.before() == ResourceLeaseState::Deferred);
    check(passed, resumed.after() == ResourceLeaseState::Active);
    check(
        passed,
        committed.lease().state() == ResourceLeaseState::Active,
    );
    let cancelled = committed.cancel_all().unwrap();
    check(
        passed,
        cancelled.decision() == ResourceRetentionDecision::ReturnRequested,
    );
    check(passed, cancelled.after() == ResourceLeaseState::Cancelled);
    check(passed, cancelled.entries().len() == resources.len());
    check(
        passed,
        committed.lease().state() == ResourceLeaseState::Cancelled,
    );

    let expected_retention = plan
        .payload()
        .memory()
        .static_allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.resource_id(),
                ResourceRetentionPolicy::from(allocation.lifetime()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    check(
        passed,
        committed.lease().entries().all(|entry| {
            expected_retention.get(entry.resource_id()) == Some(&entry.retention_policy())
        }),
    );

    let released = committed.release().unwrap();
    check(
        passed,
        released
            .actual_states()
            .iter()
            .all(|state| *state == ResourceTransactionState::Released),
    );
    check(
        passed,
        released.receipts().last().unwrap().records().len() == resources.len(),
    );
}

fn failure_identity_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "failure-id-sequence")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let first = committed.complete_pending_release().unwrap_err();
    let second = committed.complete_pending_release().unwrap_err();
    check(
        passed,
        first.failure_id().get() != 0 && second.failure_id().get() > first.failure_id().get(),
    );
    check(
        passed,
        first.action() == ResourceTransactionAction::Release
            && first.failure_point().is_none()
            && first.completed().is_empty()
            && first.recovery_strategy() == ResourceRecoveryStrategy::ForwardCompletion,
    );
    check(
        passed,
        second.validate_recovery_continuation(&first).is_err(),
    );
    let _released = committed.release().unwrap();
}

fn context_bound_wire_validation(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "wire-transition")
        .reserve()
        .unwrap();
    let receipt = reserved.receipts()[0].clone();
    let context = reserved
        .latest_transition_validation_context()
        .unwrap()
        .clone();
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    check(passed, wire.clone().try_validate().is_err());
    check(passed, wire.try_validate_against(&context).is_ok());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["records"][0]["generation"] = json!(0);
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["records"].as_array_mut().unwrap().pop();
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    let extra = value["records"][0].clone();
    value["records"].as_array_mut().unwrap().push(extra);
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["admission"]["plan_hash"] = json!(sha('f'));
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let mut value = serde_json::to_value(&receipt).unwrap();
    value["identity"]["transaction_id"] = json!("transaction.mutated");
    let wire = UnvalidatedResourceTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&context).is_err());

    let committed = reserved.commit().unwrap();
    let mut committed = committed;
    let lease_receipt = committed.defer_all().unwrap();
    let lease_context = committed.latest_lease_validation_context().unwrap().clone();
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&lease_receipt).unwrap(),
    )
    .unwrap();
    check(passed, wire.clone().try_validate().is_err());
    check(passed, wire.try_validate_against(&lease_context).is_ok());

    let mut value = serde_json::to_value(&lease_receipt).unwrap();
    value["entries"][0]["generation"] = json!(0);
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&lease_context).is_err());

    let mut value = serde_json::to_value(&lease_receipt).unwrap();
    value["entries"].as_array_mut().unwrap().pop();
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&lease_context).is_err());

    let mut value = serde_json::to_value(&lease_receipt).unwrap();
    let extra = value["entries"][0].clone();
    value["entries"].as_array_mut().unwrap().push(extra);
    let wire = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(
        &serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    check(passed, wire.try_validate_against(&lease_context).is_err());

    let at_limit = vec![b' '; MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES];
    let error = UnvalidatedResourceTransitionReceipt::decode_untrusted(&at_limit).unwrap_err();
    check(passed, !error.to_string().contains("exceeds limit"));
    let over_limit = vec![b' '; MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES + 1];
    let error = UnvalidatedResourceTransitionReceipt::decode_untrusted(&over_limit).unwrap_err();
    check(passed, error.to_string().contains("exceeds limit"));
    let at_limit = vec![b' '; MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES];
    let error = UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(&at_limit).unwrap_err();
    check(passed, !error.to_string().contains("exceeds limit"));
    let over_limit = vec![b' '; MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES + 1];
    let error =
        UnvalidatedResourceLeaseTransitionReceipt::decode_untrusted(&over_limit).unwrap_err();
    check(passed, error.to_string().contains("exceeds limit"));

    let _terminal = committed.release().unwrap();
}

fn drop_abandon_exactly_once(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    drop(transaction(plan, driver, "drop-new"));
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(
        passed,
        trace.lock().unwrap().abandon[0].state() == ResourceTransactionState::New,
    );
    check(
        passed,
        trace.lock().unwrap().abandon[0]
            .ledger()
            .iter()
            .all(|entry| entry.transaction_state() == ResourceTransactionState::New),
    );

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let committed = transaction(plan, driver, "drop-committed")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    drop(committed);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(
        passed,
        trace.lock().unwrap().abandon[0]
            .ledger()
            .iter()
            .all(ResourceLedgerEntrySnapshot::buffer_present),
    );

    let resources = plan_resources(plan);
    let reserve_failure = failure_key("reserve", &resources[1]);
    let failures = [(reserve_failure.as_str(), 1)];
    let (driver, trace) = configured_driver(plan, &failures, &[]);
    let error = expect_err(transaction(plan, driver, "drop-recovery-owner").reserve());
    drop(error);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(
        passed,
        trace.lock().unwrap().abandon[0].pending_action()
            == Some(ResourceTransactionAction::Reserve),
    );

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let terminal = transaction(plan, driver, "drop-terminal")
        .reserve()
        .unwrap()
        .rollback()
        .unwrap();
    drop(terminal);
    check(passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "drop-deferred-pool")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    committed.defer_all().unwrap();
    drop(committed);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    let signal = trace.lock().unwrap().abandon[0].clone();
    check(
        passed,
        signal
            .ledger()
            .iter()
            .all(|entry| entry.transaction_state() == ResourceTransactionState::Committed),
    );
    check(
        passed,
        trace.lock().unwrap().abandon_buffer_counts == [resources.len()],
    );
}

fn abandon_callback_panic_during_unwind_is_contained(passed: &mut usize) {
    let executable = std::env::current_exe().expect("test executable is discoverable");
    let status = Command::new(executable)
        .arg("resource_transaction_abandon_panic_child")
        .arg("--exact")
        .arg("--test-threads=1")
        .arg("--nocapture")
        .env("FERRUM_VNEXT_ABANDON_PANIC_CHILD", "1")
        .env("RUST_BACKTRACE", "0")
        .status()
        .expect("abandon panic child starts");
    check(passed, status.success());
}

#[test]
fn resource_transaction_abandon_panic_child() {
    if std::env::var_os("FERRUM_VNEXT_ABANDON_PANIC_CHILD").is_none() {
        return;
    }

    let plan = execution_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let runtime = Arc::downgrade(driver.runtime());
    trace.lock().unwrap().panic_on_abandon = true;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _committed = transaction(&plan, driver, "abandon-panic-during-unwind")
            .reserve()
            .unwrap()
            .commit()
            .unwrap();
        panic!("primary transaction owner panic");
    }));
    assert!(unwind.is_err());
    assert_eq!(trace.lock().unwrap().abandon.len(), 1);
    assert_eq!(trace.lock().unwrap().buffer_drops, 0);
    assert!(runtime.upgrade().is_some());
}

fn allocation_withholding_is_core_owned(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "allocation-forget-after-success")
        .reserve()
        .unwrap();
    let first = reserved.reservations().reservations()[0]
        .resource_id()
        .clone();
    trace
        .lock()
        .unwrap()
        .post_allocation
        .insert(first.to_string(), PostAllocationBehavior::ForgetThenError);
    let poisoned = match reserved.commit() {
        Err(ResourceCommitTransitionError::Poisoned(owner)) => owner,
        Err(ResourceCommitTransitionError::Recoverable(_)) => {
            panic!("withheld allocation was treated as unowned")
        }
        Ok(_) => panic!("withheld allocation unexpectedly committed"),
    };
    check(
        passed,
        poisoned.failure().failure().code() == "commit-after-allocation_failed",
    );
    check(
        passed,
        poisoned.failure().recovery_strategy() == ResourceRecoveryStrategy::ReconcileOrQuarantine,
    );
    check(
        passed,
        poisoned
            .failure()
            .ledger_after()
            .iter()
            .any(|entry| entry.entry().resource_id() == &first && entry.buffer_present()),
    );
    check(passed, trace.lock().unwrap().runtime_allocate_calls == 1);
    let quarantined = poisoned.quarantine().unwrap();
    check(
        passed,
        quarantined.state() == ResourceTransactionState::Quarantined,
    );
    check(passed, trace.lock().unwrap().quarantine_sizes == [1]);
    drop(quarantined);
    check(passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "allocation-drop-after-success")
        .reserve()
        .unwrap();
    let first = reserved.reservations().reservations()[0]
        .resource_id()
        .clone();
    {
        let mut trace = trace.lock().unwrap();
        trace
            .post_allocation
            .insert(first.to_string(), PostAllocationBehavior::DropThenError);
        trace.failures.insert(failure_key("reconcile", &first), 1);
    }
    let poisoned = match reserved.commit() {
        Err(ResourceCommitTransitionError::Poisoned(owner)) => owner,
        Err(ResourceCommitTransitionError::Recoverable(_)) => {
            panic!("dropped allocation receipt lost core ownership")
        }
        Ok(_) => panic!("dropped allocation receipt unexpectedly committed"),
    };
    check(
        passed,
        poisoned.failure().failure().code() == "commit-after-allocation_failed",
    );
    let retry_owner = match poisoned.reconcile() {
        Ok(_) => panic!("injected reconcile failure unexpectedly succeeded"),
        Err(owner) => owner,
    };
    check(passed, retry_owner.failure().recovery_failures().len() == 1);
    check(
        passed,
        retry_owner
            .failure()
            .ledger_after()
            .iter()
            .any(|entry| entry.entry().resource_id() == &first && entry.buffer_present()),
    );
    let recovery = retry_owner.reconcile().unwrap();
    check(
        passed,
        recovery.failure().recovery_strategy() == ResourceRecoveryStrategy::ReverseCompensation,
    );
    check(passed, calls(&trace, "reconcile:").len() == 2);
    let reserved = recovery.recover().unwrap();
    check(
        passed,
        reserved.state() == ResourceTransactionState::Reserved,
    );
    let rolled_back = reserved.rollback().unwrap();
    check(
        passed,
        rolled_back.state() == ResourceTransactionState::RolledBack,
    );
    drop(rolled_back);
    check(passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(plan, &[], &[]);
    let reserved = transaction(plan, driver, "allocation-panic-after-success")
        .reserve()
        .unwrap();
    let first = reserved.reservations().reservations()[0]
        .resource_id()
        .clone();
    trace
        .lock()
        .unwrap()
        .post_allocation
        .insert(first.to_string(), PostAllocationBehavior::Panic);
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = reserved.commit();
    }));
    check(passed, panic_result.is_err());
    check(passed, trace.lock().unwrap().runtime_allocate_calls == 1);
    check(passed, trace.lock().unwrap().abandon.len() == 1);
    check(passed, trace.lock().unwrap().abandon_buffer_counts == [1]);
    check(
        passed,
        trace.lock().unwrap().abandon_claimed_bytes == [plan.payload().memory().static_bytes()],
    );
    check(
        passed,
        trace.lock().unwrap().abandon[0]
            .ledger()
            .iter()
            .any(|entry| entry.entry().resource_id() == &first && entry.buffer_present()),
    );
    check(
        passed,
        admit_resources(plan, id("request.allocation-panic.after-abandon")).is_ok(),
    );
}

#[test]
fn transaction_evidence_contracts_are_exhaustive() {
    const EXPECTED_CASES: usize = 69;
    let plan = execution_plan();
    let mut passed = 0;
    full_pool_retention_and_release(&plan, &mut passed);
    failure_identity_contract(&plan, &mut passed);
    context_bound_wire_validation(&plan, &mut passed);
    drop_abandon_exactly_once(&plan, &mut passed);
    abandon_callback_panic_during_unwind_is_contained(&mut passed);
    allocation_withholding_is_core_owned(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT TRANSACTION EVIDENCE PASS: {passed}/{EXPECTED_CASES}");
}
