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

fn logical_sequence_drop_recovery_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "logical-drop-recovery");
    let run_id = "run.logical.drop-recovery";
    let request_id = "request.logical.drop-recovery";
    let request = admit_logical_request(&root, run_id, request_id);
    let resources = admit_logical_child_sequence(&root, &request);
    let sibling_resources = admit_logical_child_sequence(&root, &request);
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    drop(permit);

    check(passed, resources.is_poisoned()); // 1
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 0); // 2
    check(
        passed,
        matches!(
            resources.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                slot,
                activation_epoch: epoch,
            }) if slot == resources.sequence_authority().sparse_id()
                && epoch == activation_epoch
        ),
    ); // 3

    // Dropping the exact bound stream transfers it into the private recovery
    // registry. Recovery then drains and retires only this sequence authority.
    drop(stream);
    let recovered = resources.recover_abandoned_sequence().unwrap();
    check(
        passed,
        recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 4
    check(
        passed,
        recovered.sequence_authority() == resources.sequence_authority(),
    ); // 5
    check(passed, recovered.activation_epoch() == activation_epoch); // 6
    check(passed, recovered.run_id() == &id::<RunId>(run_id)); // 7
    check(
        passed,
        recovered.request_id() == &id::<RequestIdentity>(request_id),
    ); // 8
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 1); // 9
    check(
        passed,
        matches!(
            resources.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ) && matches!(
            resources.create_execution_stream(),
            Err(ExecutionStreamCreationError::Contract(_))
        ),
    ); // 10

    let mut sibling_stream = sibling_resources.create_execution_stream().unwrap();
    let sibling_permit = sibling_resources.activate(&mut sibling_stream).unwrap();
    let sibling_completion = sibling_permit.synchronize().unwrap().complete().unwrap();
    drop(sibling_stream);
    drop(sibling_resources);
    drop(resources);
    drop(request);
    let close = close_plan_runtime(root);
    let trace = trace.lock().unwrap();
    check(
        passed,
        sibling_completion.sequence_authority() != recovered.sequence_authority()
            && trace.runtime_synchronize_calls == 2
            && close.released_static_resources() == plan_resources(plan).len()
            && trace.quarantine_sizes.is_empty()
            && trace.abandon.is_empty()
            && trace.stream_drops == 2,
    ); // 11
}

fn forgotten_live_permit_fails_closed(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "forgotten-logical-permit");
    let sequence = admit_logical_sequence(
        &root,
        "run.forgotten-logical-permit",
        "request.forgotten-logical-permit",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    check(passed, activation_epoch != 0); // 1/16

    // This is intentionally the hostile path: Drop cannot mark the sequence
    // poisoned, so the registry must retain exact epoch metadata and refuse
    // recovery until the externally-owned bound stream is returned.
    std::mem::forget(permit);
    check(passed, !sequence.is_poisoned()); // 2/16
    check(passed, sequence.activate(&mut stream).is_err()); // 3/16
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                slot,
                activation_epoch: epoch,
            }) if slot == sequence_authority.sparse_id() && epoch == activation_epoch
        ),
    ); // 4/16

    // The logical sequence owns a request binding, which owns the root Arc.
    // Close therefore enters Closing but cannot tear down resources underneath
    // the forgotten permit's recovery authority.
    let (root, strong_count) = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            ..
        }) => (resources, strong_count),
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("forgotten logical permit unexpectedly allowed root close")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };
    check(passed, strong_count >= 2); // 5/16
    check(passed, root.is_closing()); // 6/16
    check(passed, root.trusted_runtime_binding().is_err()); // 7/16

    drop(stream);
    check(passed, sequence.is_poisoned()); // 8/16
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(passed, recovered.sequence_authority() == sequence_authority); // 9/16
    check(passed, recovered.activation_epoch() == activation_epoch); // 10/16
    check(passed, recovered.run_id() == sequence.run_id()); // 11/16
    check(passed, recovered.request_id() == sequence.request_id()); // 12/16
    check(
        passed,
        recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned
            && recovered.runtime_implementation_fingerprint()
                == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 13/16
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ),
    ); // 14/16
    check(
        passed,
        matches!(
            sequence.create_execution_stream(),
            Err(ExecutionStreamCreationError::Contract(_))
        ),
    ); // 15/16

    drop(sequence);
    let close = close_plan_runtime(root);
    let final_trace = trace.lock().unwrap();
    check(
        passed,
        close.released_static_resources() == plan_resources(plan).len()
            && final_trace.runtime_synchronize_calls == 1
            && final_trace.stream_drops == 1
            && final_trace.abandon.is_empty(),
    ); // 16/16
}

fn abandoned_sequence_recovery_contract(plan: &ExecutionPlan, passed: &mut usize) {
    // Backend synchronization returning Ok is insufficient unless the exact
    // stream also reports Ready. The failure owner is dropped intentionally so
    // the registry must recover the abandoned stream.
    let (driver, trace) = configured_driver(plan, &[], &[]);
    trace.lock().unwrap().synchronize_returns_not_ready = true;
    let root = plan_runtime(plan, driver, "recovery-not-ready");
    let sequence = admit_logical_sequence(
        &root,
        "run.recovery-not-ready",
        "request.recovery-not-ready",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    let failure = match permit.synchronize() {
        Ok(_) => panic!("non-ready stream produced synchronization evidence"),
        Err(failure) => failure,
    };
    check(
        passed,
        matches!(failure.error(), SequenceSynchronizationError::Contract(_)),
    ); // 1/14
    drop(failure);
    check(passed, sequence.is_poisoned()); // 2/14
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                slot,
                activation_epoch: epoch,
            }) if slot == sequence_authority.sparse_id() && epoch == activation_epoch
        ),
    ); // 3/14
    drop(stream);
    trace.lock().unwrap().synchronize_returns_not_ready = false;
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(passed, recovered.sequence_authority() == sequence_authority); // 4/14
    check(
        passed,
        recovered.activation_epoch() == activation_epoch
            && recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 5/14
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 2); // 6/14
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ),
    ); // 7/14
    drop(sequence);
    let close = close_plan_runtime(root);
    let final_trace = trace.lock().unwrap();
    check(
        passed,
        close.released_static_resources() == plan_resources(plan).len()
            && final_trace.stream_drops == 1
            && final_trace.abandon.is_empty(),
    ); // 8/14
    drop(final_trace);

    // A backend error must restore the attached stream to the registry so a
    // later bounded retry can drain it. Two failures cover the explicit permit
    // path and the first registry recovery attempt independently.
    let (driver, trace) = configured_driver(plan, &[], &[]);
    trace.lock().unwrap().synchronize_failures = 2;
    let root = plan_runtime(plan, driver, "recovery-runtime-retry");
    let sequence = admit_logical_sequence(
        &root,
        "run.recovery-runtime-retry",
        "request.recovery-runtime-retry",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    let failure = match permit.synchronize() {
        Ok(_) => panic!("injected synchronization failure unexpectedly succeeded"),
        Err(failure) => failure,
    };
    check(
        passed,
        matches!(failure.error(), SequenceSynchronizationError::Runtime(_)),
    ); // 9/14
    drop(failure);
    drop(stream);
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Runtime(_))
        ),
    ); // 10/14
    check(
        passed,
        sequence.is_poisoned() && trace.lock().unwrap().runtime_synchronize_calls == 2,
    ); // 11/14
    trace.lock().unwrap().synchronize_failures = 0;
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(
        passed,
        recovered.sequence_authority() == sequence_authority
            && recovered.activation_epoch() == activation_epoch
            && recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 12/14
    let (runtime_synchronize_calls, stream_drops) = {
        let trace = trace.lock().unwrap();
        (trace.runtime_synchronize_calls, trace.stream_drops)
    };
    check(passed, runtime_synchronize_calls == 3 && stream_drops == 1); // 13/14
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ),
    ); // 14/14
    drop(sequence);
    let _ = close_plan_runtime(root);
}

fn abandoned_recovery_unlocks_during_runtime_sync(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "concurrent-abandoned-recovery");
    let sequence = admit_logical_sequence(
        &root,
        "run.concurrent-abandoned-recovery",
        "request.concurrent-abandoned-recovery",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    drop(permit);
    drop(stream);
    check(passed, sequence.is_poisoned()); // 1/7

    let entered = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    let release = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    trace.lock().unwrap().synchronize_block = Some((Arc::clone(&entered), Arc::clone(&release)));

    // Exactly one worker performs the slow recovery; the caller is the only
    // contender. Spawn is completed before either barrier wait. After the
    // entered rendezvous, no assertion or fallible unwrap occurs until the
    // release rendezvous and join have completed, so a failed assertion cannot
    // strand the already-started worker on an unreachable barrier.
    let worker_sequence = Arc::clone(&sequence);
    let (concurrent, worker_result, calls_while_blocked) = std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("vnext-abandoned-recovery-worker".to_owned())
            .spawn_scoped(scope, move || worker_sequence.recover_abandoned_sequence())
            .expect("the single bounded abandoned-recovery worker starts");
        entered.wait();
        let calls_while_blocked = trace.lock().unwrap().runtime_synchronize_calls;
        let concurrent = sequence.recover_abandoned_sequence();
        release.wait();
        let worker_result = worker
            .join()
            .expect("the bounded abandoned-recovery worker does not panic");
        (concurrent, worker_result, calls_while_blocked)
    });
    let recovered = match worker_result {
        Ok(receipt) => receipt,
        Err(_) => panic!("the primary abandoned recovery unexpectedly failed"),
    };

    check(passed, calls_while_blocked == 1); // 2/7
    check(
        passed,
        matches!(concurrent, Err(AbandonedSequenceRecoveryError::Contract(_))),
    ); // 3/7
    check(passed, recovered.sequence_authority() == sequence_authority); // 4/7
    check(passed, recovered.activation_epoch() == activation_epoch); // 5/7
    check(
        passed,
        recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned
            && trace.lock().unwrap().runtime_synchronize_calls == 1,
    ); // 6/7
    check(
        passed,
        matches!(
            sequence.recover_abandoned_sequence(),
            Err(AbandonedSequenceRecoveryError::Contract(_))
        ) && trace.lock().unwrap().stream_drops == 1,
    ); // 7/7

    drop(sequence);
    let _ = close_plan_runtime(root);
}

#[test]
fn sequence_recovery_contracts_are_exhaustive() {
    const EXPECTED_CASES: usize = 48;
    let plan = execution_plan();
    let mut passed = 0;
    logical_sequence_drop_recovery_contract(&plan, &mut passed);
    forgotten_live_permit_fails_closed(&plan, &mut passed);
    abandoned_sequence_recovery_contract(&plan, &mut passed);
    abandoned_recovery_unlocks_during_runtime_sync(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT SEQUENCE RECOVERY PASS: {passed}/{EXPECTED_CASES}");
}
