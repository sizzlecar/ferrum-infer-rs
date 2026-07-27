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

fn logical_sequence_activation_completion_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "logical-activation-completion");
    let run_id = "run.logical.activation-completion";
    let request_id = "request.logical.activation-completion";
    let resources = admit_logical_sequence(&root, run_id, request_id);
    let request_resources = resources.request_resources();
    let expected_request_slices = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| descriptor.lifetime() == AllocationLifetime::Request)
        .count();
    let expected_sequence_slices = plan
        .payload()
        .memory()
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| descriptor.lifetime() == AllocationLifetime::Sequence)
        .count();

    check(passed, resources.run_id() == &id::<RunId>(run_id)); // 1
    check(
        passed,
        resources.request_id() == &id::<RequestIdentity>(request_id),
    ); // 2
    check(passed, request_resources.run_id() == resources.run_id()); // 3
    check(
        passed,
        request_resources.request_id() == resources.request_id(),
    ); // 4
    check(
        passed,
        resources.request_authority() == request_resources.request_authority(),
    ); // 5
    check(
        passed,
        resources.coordinator_id() == request_resources.coordinator_id(),
    ); // 6
    check(passed, resources.static_provisioning().is_some()); // 7
    check(
        passed,
        request_resources.backing_slices().len() == expected_request_slices,
    ); // 8
    check(passed, resources.backing_generation().unwrap().get() == 1); // 9

    let evidence = resources.plan_evidence();
    check(passed, evidence.plan_id() == plan.payload().plan_id()); // 10
    check(passed, evidence.plan_hash() == plan.plan_hash()); // 11
    check(passed, evidence.device_id() == plan.payload().device_id()); // 12
    check(
        passed,
        evidence.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 13

    let mut stream = resources.create_execution_stream().unwrap();
    let mut competing_stream = resources.create_execution_stream().unwrap();
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 0); // 14
    let permit = resources.activate(&mut stream).unwrap();
    let first_epoch = permit.activation_epoch();
    check(passed, std::ptr::eq(permit.resources(), resources.as_ref())); // 15
    check(passed, permit.run_id() == resources.run_id()); // 16
    check(passed, permit.request_id() == resources.request_id()); // 17
    check(
        passed,
        permit.sequence_authority() == resources.sequence_authority(),
    ); // 18
    check(
        passed,
        permit.coordinator_id() == resources.coordinator_id(),
    ); // 19
    check(
        passed,
        permit.backing_slices().len() == expected_sequence_slices,
    ); // 20
    check(
        passed,
        permit.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 21
    check(passed, first_epoch == 1); // 22
    check(passed, resources.activate(&mut competing_stream).is_err()); // 23

    let first_receipt = permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        first_receipt.sequence_authority() == resources.sequence_authority()
            && first_receipt.activation_epoch() == first_epoch
            && first_receipt.run_id() == resources.run_id()
            && first_receipt.request_id() == resources.request_id()
            && first_receipt.plan().plan_id() == plan.payload().plan_id()
            && first_receipt.runtime_implementation_fingerprint()
                == plan.payload().device_runtime_implementation_fingerprint()
            && trace.lock().unwrap().runtime_synchronize_calls == 1
            && !resources.is_poisoned(),
    ); // 24

    let second_permit = resources.activate(&mut stream).unwrap();
    let second_epoch = second_permit.activation_epoch();
    let second_receipt = second_permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        second_epoch > first_epoch
            && second_receipt.activation_epoch() == second_epoch
            && second_receipt.sequence_authority() == resources.sequence_authority()
            && trace.lock().unwrap().runtime_synchronize_calls == 2
            && !resources.is_poisoned(),
    ); // 25

    drop(competing_stream);
    drop(stream);
    drop(resources);
    let _close = close_plan_runtime(root);
}

fn logical_sequence_synchronization_retry_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    trace.lock().unwrap().synchronize_failures = 1;
    let root = plan_runtime(plan, driver, "logical-synchronization-retry");
    let resources = admit_logical_sequence(
        &root,
        "run.logical.synchronization-retry",
        "request.logical.synchronization-retry",
    );
    let mut stream = resources.create_execution_stream().unwrap();
    let permit = resources.activate(&mut stream).unwrap();
    let epoch = permit.activation_epoch();
    let failure = match permit.synchronize() {
        Ok(_) => panic!("injected synchronization failure unexpectedly succeeded"),
        Err(failure) => failure,
    };
    check(
        passed,
        matches!(failure.error(), SequenceSynchronizationError::Runtime(_)),
    ); // 1
    check(
        passed,
        trace.lock().unwrap().runtime_synchronize_calls == 1 && !resources.is_poisoned(),
    ); // 2
    let receipt = failure.retry().unwrap().complete().unwrap();
    check(
        passed,
        receipt.sequence_authority() == resources.sequence_authority()
            && receipt.activation_epoch() == epoch
            && trace.lock().unwrap().runtime_synchronize_calls == 2
            && !resources.is_poisoned(),
    ); // 3

    drop(stream);
    drop(resources);
    let _close = close_plan_runtime(root);
}

fn deferred_admission_has_no_execution_authority(_plan: &ExecutionPlan, passed: &mut usize) {
    let constrained_plan = execution_plan_with_policy(policy_with_memory(4096, 128, 1));
    let (driver, _) = configured_driver(&constrained_plan, &[], &[]);
    let root = plan_runtime(&constrained_plan, driver, "logical-deferred-no-authority");
    let request = admit_logical_request(
        &root,
        "run.logical.deferred-no-authority",
        "request.logical.deferred-no-authority",
    );
    let first = admit_logical_child_sequence(&root, &request);
    let admission = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut deferred = None;
    for attempt in 0..=3 {
        match request.try_admit_sequence(admission.clone()).unwrap() {
            SequenceResourceAdmissionDecision::Deferred(decision) => {
                deferred = Some(decision);
                break;
            }
            SequenceResourceAdmissionDecision::BackingDeferred(backing) if attempt < 3 => {
                backing.maintain().unwrap();
            }
            SequenceResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("deferred admission backing did not converge after bounded maintenance")
            }
            SequenceResourceAdmissionDecision::Admitted(_) => {
                panic!("sequence ceiling produced executable authority")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("sequence ceiling was treated as permanent rejection")
            }
        }
    }
    let deferred = deferred.expect("bounded admission reaches logical deferral");
    // The Deferred variant carries only retry evidence. In particular, this
    // branch never yields Arc<AdmittedSequenceResources<_>>, so no stream or
    // ActiveSequencePermit can be minted from the rejected attempt.
    check(
        passed,
        deferred.action() == DeferredAction::WaitForRelease
            && deferred.available().active_sequences() == 1
            && deferred.available().maximum_active_sequences() == 1
            && deferred.blockers().iter().any(|blocker| {
                blocker.kind() == CapacityShortfallKind::ActiveSequenceCeiling
                    && blocker.domain().is_none()
            }),
    ); // 1

    drop(deferred);
    drop(first);
    drop(request);
    let _close = close_plan_runtime(root);
}

fn logical_sequence_explicit_abort_is_exact_contract(plan: &ExecutionPlan, passed: &mut usize) {
    let (driver, trace) = configured_driver(plan, &[], &[]);
    let root = plan_runtime(plan, driver, "logical-explicit-abort-exact");
    let run_id = "run.logical.explicit-abort-exact";
    let request_id = "request.logical.explicit-abort-exact";
    let request = admit_logical_request(&root, run_id, request_id);
    let aborted_resources = admit_logical_child_sequence(&root, &request);
    let sibling_resources = admit_logical_child_sequence(&root, &request);

    check(
        passed,
        aborted_resources.request_authority() == sibling_resources.request_authority(),
    ); // 1
    check(
        passed,
        aborted_resources.sequence_authority() != sibling_resources.sequence_authority(),
    ); // 2
    check(
        passed,
        aborted_resources.coordinator_id() == sibling_resources.coordinator_id(),
    ); // 3
    check(
        passed,
        Arc::ptr_eq(
            aborted_resources.request_resources(),
            sibling_resources.request_resources(),
        ),
    ); // 4
    check(
        passed,
        !aborted_resources.is_poisoned() && !sibling_resources.is_poisoned(),
    ); // 5

    let mut aborted_stream = aborted_resources.create_execution_stream().unwrap();
    let mut sibling_stream = sibling_resources.create_execution_stream().unwrap();
    let aborted_permit = aborted_resources.activate(&mut aborted_stream).unwrap();
    let sibling_permit = sibling_resources.activate(&mut sibling_stream).unwrap();
    let aborted_epoch = aborted_permit.activation_epoch();
    let sibling_epoch = sibling_permit.activation_epoch();
    check(
        passed,
        aborted_permit.sequence_authority() == aborted_resources.sequence_authority(),
    ); // 6
    check(
        passed,
        sibling_permit.sequence_authority() == sibling_resources.sequence_authority(),
    ); // 7
    check(passed, aborted_epoch == 1 && sibling_epoch == 1); // 8
    check(
        passed,
        aborted_permit.sequence_authority() != sibling_permit.sequence_authority(),
    ); // 9

    let abort = aborted_permit.synchronize().unwrap().abort().unwrap();
    check(passed, trace.lock().unwrap().runtime_synchronize_calls == 1); // 10
    check(
        passed,
        abort.disposition() == ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    ); // 11
    check(
        passed,
        abort.sequence_authority() == aborted_resources.sequence_authority(),
    ); // 12
    check(passed, abort.run_id() == &id::<RunId>(run_id)); // 13
    check(
        passed,
        abort.request_id() == &id::<RequestIdentity>(request_id),
    ); // 14
    check(passed, abort.activation_epoch() == aborted_epoch); // 15
    check(
        passed,
        abort.plan().coordinator_id() == aborted_resources.coordinator_id(),
    ); // 16
    check(
        passed,
        abort.runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    ); // 17
    check(passed, aborted_resources.is_poisoned()); // 18
    check(
        passed,
        matches!(
            aborted_resources.create_execution_stream(),
            Err(ExecutionStreamCreationError::Contract(_))
        ),
    ); // 19
    check(
        passed,
        aborted_resources.activate(&mut aborted_stream).is_err(),
    ); // 20
    check(passed, !sibling_resources.is_poisoned()); // 21

    let sibling_completion = sibling_permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        sibling_completion.sequence_authority() == sibling_resources.sequence_authority()
            && sibling_completion.activation_epoch() == sibling_epoch
            && trace.lock().unwrap().runtime_synchronize_calls == 2
            && !sibling_resources.is_poisoned(),
    ); // 22

    let third_resources = admit_logical_child_sequence(&root, &request);
    let mut third_stream = third_resources.create_execution_stream().unwrap();
    let third_permit = third_resources.activate(&mut third_stream).unwrap();
    let third_completion = third_permit.synchronize().unwrap().complete().unwrap();
    check(
        passed,
        third_completion.sequence_authority() == third_resources.sequence_authority()
            && third_resources.sequence_authority() != aborted_resources.sequence_authority()
            && !third_resources.is_poisoned()
            && trace.lock().unwrap().runtime_synchronize_calls == 3,
    ); // 23

    drop(third_stream);
    drop(third_resources);
    drop(sibling_stream);
    drop(sibling_resources);
    drop(aborted_stream);
    drop(aborted_resources);
    drop(request);
    let close = close_plan_runtime(root);
    let trace = trace.lock().unwrap();
    check(
        passed,
        close.released_static_resources() == plan_resources(plan).len()
            && trace.quarantine_sizes.is_empty()
            && trace.abandon.is_empty(),
    ); // 24
}

#[test]
fn sequence_activation_contracts_are_exhaustive() {
    const EXPECTED_CASES: usize = 53;
    let plan = execution_plan();
    let mut passed = 0;
    logical_sequence_activation_completion_contract(&plan, &mut passed);
    logical_sequence_synchronization_retry_contract(&plan, &mut passed);
    deferred_admission_has_no_execution_authority(&plan, &mut passed);
    logical_sequence_explicit_abort_is_exact_contract(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT SEQUENCE ACTIVATION PASS: {passed}/{EXPECTED_CASES}");
}
