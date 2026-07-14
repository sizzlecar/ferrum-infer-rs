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

fn abandoned_recovery_retains_stream_on_backend_panic(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "abandoned-recovery-backend-panic");
    let sequence = admit_logical_sequence(
        &root,
        "run.abandoned-recovery-backend-panic",
        "request.abandoned-recovery-backend-panic",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    let activation_epoch = permit.activation_epoch();
    let sequence_authority = permit.sequence_authority();
    drop(permit);
    drop(stream);
    trace.lock().unwrap().panic_on_stream_state = true;

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = sequence.recover_abandoned_sequence();
    }));
    check(passed, panic.is_err()); // 1/5
    check(passed, sequence.is_poisoned()); // 2/5
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 1); // 3/5

    // recover() restores the stream to Attached before resuming the backend
    // panic, so the second attempt must recover the same epoch rather than
    // leaking or replacing the raw stream.
    let recovered = sequence.recover_abandoned_sequence().unwrap();
    check(
        passed,
        recovered.sequence_authority() == sequence_authority
            && recovered.activation_epoch() == activation_epoch
            && recovered.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 4/5
    let (runtime_synchronize_calls, stream_drops) = {
        let trace = trace.lock().unwrap();
        (trace.runtime_synchronize_calls, trace.stream_drops)
    };
    check(
        passed,
        runtime_synchronize_calls == 2
            && stream_drops == 1
            && matches!(
                sequence.recover_abandoned_sequence(),
                Err(AbandonedSequenceRecoveryError::Contract(_))
            ),
    ); // 5/5

    drop(sequence);
    let _ = close_plan_runtime(root);
}

fn unresolved_owner_drop_retains_backend_lifetimes(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);
    let second_release = failure_key("release", &resources[1]);
    let (driver, trace) = configured_driver(plan, &[(second_release.as_str(), 1)], &[]);
    let runtime = Arc::downgrade(driver.runtime());
    trace.lock().unwrap().retain_ownership = true;

    let root = plan_runtime(plan, driver, "unresolved-close-owner-drop");
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected plan runtime close failure unexpectedly succeeded"),
    };
    check(passed, close_failure.failure().code() == "release_failed"); // C1

    let drops_before_owner_drop = trace.lock().unwrap().buffer_drops;
    drop(close_failure);
    let owner_drop_recorded = {
        let trace = trace.lock().unwrap();
        trace.abandon.len() == 1 && trace.durable_ownership.len() == 1
    };
    check(passed, owner_drop_recorded); // C2

    let ownership = trace
        .lock()
        .unwrap()
        .durable_ownership
        .pop()
        .expect("failed close drop transfers unresolved ownership to the driver");
    check(
        passed,
        ownership.reason() == ResourceOwnershipReason::Abandon,
    ); // C3
    check(
        passed,
        ownership.buffers().len() == 1
            && ownership.claimed_bytes()
                == ownership
                    .buffers()
                    .iter()
                    .map(|buffer| buffer.expected_descriptor().size_bytes)
                    .sum::<u64>(),
    ); // C4
    check(
        passed,
        runtime.upgrade().is_some()
            && trace.lock().unwrap().buffer_drops == drops_before_owner_drop,
    ); // C5

    drop(ownership);
    let (buffer_drops, buffer_drops_after_backend) = {
        let trace = trace.lock().unwrap();
        (trace.buffer_drops, trace.buffer_drops_after_backend)
    };
    check(
        passed,
        runtime.upgrade().is_none()
            && buffer_drops == drops_before_owner_drop + 1
            && buffer_drops_after_backend == 0,
    ); // C6
}

fn abort_receipt_authority_and_cleanup_retry_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let resources = plan_resources(plan);

    let (source_driver, source_trace) = configured_driver(plan, &[], &[]);
    let source_root = plan_runtime(plan, source_driver, "logical-abort-source");
    let source_run = "run.logical-abort.source";
    let source_request = "request.logical-abort.source";
    let source_sequence = admit_logical_sequence(&source_root, source_run, source_request);
    let source_coordinator = source_sequence.coordinator_id();
    let source_authority = source_sequence.sequence_authority();
    let mut source_stream = source_sequence.create_execution_stream().unwrap();
    let source_active = source_sequence.activate(&mut source_stream).unwrap();
    let source_epoch = source_active.activation_epoch();
    let source_abort = source_active.synchronize().unwrap().abort().unwrap();

    check(
        passed,
        source_abort.plan().coordinator_id() == source_coordinator,
    ); // C1
    check(
        passed,
        source_abort.sequence_authority() == source_authority,
    ); // C2
    check(passed, source_abort.run_id().as_str() == source_run); // C3
    check(passed, source_abort.request_id().as_str() == source_request); // C4
    check(passed, source_abort.activation_epoch() == source_epoch); // C5
    check(
        passed,
        source_abort.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C6
    check(
        passed,
        source_abort.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // C7

    let second_release = failure_key("release", &resources[1]);
    let (target_driver, target_trace) =
        configured_driver(plan, &[(second_release.as_str(), 1)], &[]);
    let target_root = plan_runtime(plan, target_driver, "logical-abort-target");
    let target_run = "run.logical-abort.target";
    let target_request = "request.logical-abort.target";
    let target_sequence = admit_logical_sequence(&target_root, target_run, target_request);
    let target_coordinator = target_sequence.coordinator_id();
    let target_authority = target_sequence.sequence_authority();
    check(
        passed,
        source_abort.plan().coordinator_id() != target_coordinator
            && (
                source_abort.plan().coordinator_id(),
                source_abort.sequence_authority(),
            ) != (target_coordinator, target_authority),
    ); // C8

    drop(source_sequence);
    drop(source_stream);
    let source_close = close_plan_runtime(source_root);
    let source_cleanup_is_clean = {
        let trace = source_trace.lock().unwrap();
        trace.quarantine_sizes.is_empty() && trace.abandon.is_empty()
    };
    check(
        passed,
        source_close.released_static_resources() == resources.len() && source_cleanup_is_clean,
    ); // C9

    let mut target_stream = target_sequence.create_execution_stream().unwrap();
    let target_active = target_sequence.activate(&mut target_stream).unwrap();
    let target_abort = target_active.synchronize().unwrap().abort().unwrap();
    check(
        passed,
        target_abort.plan().coordinator_id() == target_coordinator
            && target_abort.sequence_authority() == target_authority
            && target_abort.plan().coordinator_id() != source_abort.plan().coordinator_id(),
    ); // C10

    drop(target_sequence);
    drop(target_stream);
    let close_failure = match PlanRuntimeResources::close(target_root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected target cleanup failure unexpectedly succeeded"),
    };
    check(
        passed,
        close_failure.failure().code() == "release_failed"
            && target_trace.lock().unwrap().quarantine_sizes.is_empty(),
    ); // C11
    let target_close = match close_failure.retry() {
        Ok(receipt) => receipt,
        Err(_) => panic!("target cleanup retry unexpectedly failed"),
    };
    let target_cleanup_is_clean = {
        let trace = target_trace.lock().unwrap();
        trace.quarantine_sizes.is_empty() && trace.abandon.is_empty()
    };
    check(
        passed,
        target_close.released_static_resources() == resources.len() && target_cleanup_is_clean,
    ); // C12
}

fn bound_stream_sequence_authority_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "bound-logical-sequence-stream");
    let source = admit_logical_sequence(
        &root,
        "run.bound-sequence.source",
        "request.bound-sequence.source",
    );
    let target = admit_logical_sequence(
        &root,
        "run.bound-sequence.target",
        "request.bound-sequence.target",
    );

    check(
        passed,
        source.coordinator_id() == target.coordinator_id()
            && source.sequence_authority() != target.sequence_authority(),
    ); // C1
    check(
        passed,
        source.request_authority() != target.request_authority(),
    ); // C2

    let mut source_stream = source.create_execution_stream().unwrap();
    check(passed, target.activate(&mut source_stream).is_err()); // C3
    check(passed, !target.is_poisoned()); // C4

    let source_active = source.activate(&mut source_stream).unwrap();
    check(
        passed,
        source_active.sequence_authority() == source.sequence_authority()
            && source_active.coordinator_id() == source.coordinator_id(),
    ); // C5
    let source_completion = source_active.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        source_completion.sequence_authority() == source.sequence_authority()
            && source_completion.plan().coordinator_id() == source.coordinator_id(),
    ); // C6

    check(
        passed,
        target.activate(&mut source_stream).is_err() && !target.is_poisoned(),
    ); // C7

    let mut target_stream = target.create_execution_stream().unwrap();
    let target_completion = target
        .activate(&mut target_stream)
        .unwrap()
        .synchronize()
        .unwrap()
        .complete()
        .unwrap();
    check(
        passed,
        target_completion.sequence_authority() == target.sequence_authority()
            && target_completion.plan().coordinator_id() == target.coordinator_id(),
    ); // C8

    drop(source);
    drop(target);
    drop(source_stream);
    drop(target_stream);
    drop(source_completion);
    drop(target_completion);
    let _ = close_plan_runtime(root);
}

// Call this once from `runtime_implementation_authority_contract`, or call it
// as a sibling from the exhaustive test. It contributes exactly six checks.
fn runtime_implementation_authority_root_extension(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "runtime-authority-root");
    let binding = root.trusted_runtime_binding().unwrap();
    check(
        passed,
        binding.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C1
    check(
        passed,
        binding.evidence().runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C2

    let request = admit_logical_request(
        &root,
        "run.runtime-authority.request",
        "request.runtime-authority.request",
    );
    check(
        passed,
        request.plan_evidence().runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C3

    let sequence = admit_logical_sequence(
        &root,
        "run.runtime-authority.sequence",
        "request.runtime-authority.sequence",
    );
    check(
        passed,
        sequence
            .plan_evidence()
            .runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C4

    let mut stream = sequence.create_execution_stream().unwrap();
    let active = sequence.activate(&mut stream).unwrap();
    check(
        passed,
        active.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // C5
    let completion = active.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        completion.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint()
            && completion.plan().coordinator_id() == sequence.coordinator_id(),
    ); // C6

    drop(binding);
    drop(request);
    drop(sequence);
    drop(stream);
    drop(completion);
    let _ = close_plan_runtime(root);
}

fn deferred_static_cannot_mint_binding(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, _) = configured_driver(plan, &[], &[]);
    let mut committed = transaction(plan, driver, "deferred-root-handoff")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    committed.defer_all().unwrap();
    let handoff_failure = match committed.into_plan_runtime() {
        Err(failure) => failure,
        Ok(_) => panic!("deferred static resources minted an owning runtime root"),
    };
    check(
        passed,
        matches!(
            handoff_failure.error(),
            VNextError::InvalidExecutionPlan { reason }
                if reason == "plan runtime handoff requires one complete active committed transaction"
        ),
    ); // C1

    let mut committed = handoff_failure.into_transaction();
    committed.resume_all().unwrap();
    let _ = committed.release().unwrap();
}

#[test]
fn recovery_authority_contracts_are_exhaustive() {
    const EXPECTED_CASES: usize = 38;
    let plan = execution_plan();
    let mut passed = 0;
    abandoned_recovery_retains_stream_on_backend_panic(&plan, &mut passed);
    unresolved_owner_drop_retains_backend_lifetimes(&plan, &mut passed);
    abort_receipt_authority_and_cleanup_retry_contract(&plan, &mut passed);
    bound_stream_sequence_authority_contract(&plan, &mut passed);
    runtime_implementation_authority_root_extension(&plan, &mut passed);
    deferred_static_cannot_mint_binding(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT RECOVERY AUTHORITY PASS: {passed}/{EXPECTED_CASES}");
}
