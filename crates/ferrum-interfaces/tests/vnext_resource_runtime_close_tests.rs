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

#[test]
fn plan_runtime_close_recovery_is_ownership_safe() {
    const EXPECTED_CLOSE_CASES: usize = 18;
    let plan = execution_plan();
    let resources = plan_resources(&plan);
    assert_eq!(resources.len(), 2);
    let second_release = failure_key("release", &resources[1]);
    let mut passed = 0;

    let (driver, trace) = configured_driver(&plan, &[(second_release.as_str(), 1)], &[]);
    let committed = transaction(&plan, driver, "plan-close-retry")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let root = match committed.into_plan_runtime() {
        Ok(root) => root,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    };
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected close failure unexpectedly succeeded"),
    };
    check(
        &mut passed,
        close_failure.failure().code() == "release_failed",
    );
    check(
        &mut passed,
        calls(&trace, "release:")
            == [
                failure_key("release", &resources[0]),
                failure_key("release", &resources[1]),
            ],
    );
    let retry_receipt = match close_failure.retry() {
        Ok(receipt) => receipt,
        Err(_) => panic!("plan runtime close retry unexpectedly failed"),
    };
    check(
        &mut passed,
        retry_receipt.released_static_resources() == resources.len(),
    );
    check(
        &mut passed,
        trace.lock().unwrap().quarantine_sizes.is_empty(),
    );
    check(&mut passed, trace.lock().unwrap().abandon.is_empty());

    let (driver, trace) = configured_driver(&plan, &[(second_release.as_str(), 1)], &[]);
    trace.lock().unwrap().retain_ownership = true;
    let committed = transaction(&plan, driver, "plan-close-quarantine")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let root = match committed.into_plan_runtime() {
        Ok(root) => root,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    };
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected close failure unexpectedly succeeded"),
    };
    check(
        &mut passed,
        close_failure.failure().code() == "release_failed",
    );
    let quarantine_receipt = match close_failure.quarantine() {
        Ok(receipt) => receipt,
        Err(_) => panic!("plan runtime quarantine unexpectedly failed"),
    };
    check(
        &mut passed,
        quarantine_receipt.released_static_resources() == 1,
    );
    check(
        &mut passed,
        quarantine_receipt.quarantined_static_resources() == 1,
    );
    check(&mut passed, trace.lock().unwrap().quarantine_sizes == [1]);
    check(
        &mut passed,
        trace.lock().unwrap().quarantine_actual_mismatch == [false],
    );
    check(
        &mut passed,
        trace.lock().unwrap().durable_ownership.len() == 1,
    );
    check(
        &mut passed,
        trace.lock().unwrap().durable_ownership[0].buffers().len() == 1,
    );
    check(&mut passed, trace.lock().unwrap().abandon.is_empty());
    let retained_ownership = {
        let mut trace = trace.lock().unwrap();
        std::mem::take(&mut trace.durable_ownership)
    };
    drop(retained_ownership);

    let failures = [(second_release.as_str(), 1), ("quarantine", 1)];
    let (driver, trace) = configured_driver(&plan, &failures, &[]);
    let committed = transaction(&plan, driver, "plan-close-quarantine-retry")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let root = match committed.into_plan_runtime() {
        Ok(root) => root,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    };
    let close_failure = match PlanRuntimeResources::close(root) {
        Err(failure) => failure,
        Ok(_) => panic!("injected close failure unexpectedly succeeded"),
    };
    let quarantine_failure = match close_failure.quarantine() {
        Err(failure) => failure,
        Ok(_) => panic!("injected quarantine failure unexpectedly succeeded"),
    };
    check(
        &mut passed,
        quarantine_failure.failure().code() == "quarantine_failed",
    );
    check(&mut passed, trace.lock().unwrap().quarantine_sizes == [1]);
    let retry_receipt = match quarantine_failure.retry() {
        Ok(receipt) => receipt,
        Err(_) => panic!("close retry after quarantine failure unexpectedly failed"),
    };
    check(
        &mut passed,
        retry_receipt.released_static_resources() == resources.len(),
    );
    check(
        &mut passed,
        calls(&trace, "release:")
            == [
                failure_key("release", &resources[0]),
                failure_key("release", &resources[1]),
                failure_key("release", &resources[1]),
            ],
    );
    check(&mut passed, trace.lock().unwrap().abandon.is_empty());

    assert_eq!(passed, EXPECTED_CLOSE_CASES);
    println!("\nVNEXT PLAN RUNTIME CLOSE PASS: {passed}/{EXPECTED_CLOSE_CASES}");
}

fn closing_error(error: &VNextError) -> bool {
    error.to_string().contains("closing plan runtime")
}

fn begin_single_participant_step(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let lane = root.create_execution_lane().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![one_token_span()]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), &lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => return step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            StepResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("step backing did not converge after bounded maintenance")
            }
            StepResourceAdmissionDecision::Deferred(_) => {
                panic!("single-participant step unexpectedly deferred")
            }
            StepResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("single-participant step unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded step admission loop always returns or panics")
}

#[test]
fn poisoned_bound_stream_retains_sequence_until_stream_drop() {
    let plan = execution_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let root = plan_runtime(&plan, driver, "poisoned-stream-sequence-hold");
    let sequence = admit_logical_sequence(
        &root,
        "run.poisoned-stream-sequence-hold",
        "request.poisoned-stream-sequence-hold",
    );
    let weak_sequence = Arc::downgrade(&sequence);
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    drop(permit);
    assert!(sequence.is_poisoned());

    drop(sequence);
    assert!(weak_sequence.upgrade().is_some());
    assert_eq!(trace.lock().unwrap().stream_drops, 0);
    let root = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("poisoned bound stream released its sequence/root hold too early")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };

    trace.lock().unwrap().synchronize_failures = 1;
    drop(stream);
    assert!(weak_sequence.upgrade().is_none());
    let pending = root.deferred_cleanup_status();
    assert_eq!(pending.pending(), 1);
    assert_eq!(trace.lock().unwrap().runtime_synchronize_calls, 0);
    let first_cleanup = root.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(first_cleanup.retryable(), 1);
    assert_eq!(first_cleanup.status_after().pending(), 1);
    assert_eq!(trace.lock().unwrap().stream_drops, 0);
    trace.lock().unwrap().synchronize_failures = 0;
    let second_cleanup = root.maintain_deferred_cleanups(1).unwrap();
    assert_eq!(second_cleanup.completed(), 1);
    assert_eq!(second_cleanup.status_after().pending(), 0);
    let (synchronize_calls, stream_drops) = {
        let trace = trace.lock().unwrap();
        (trace.runtime_synchronize_calls, trace.stream_drops)
    };
    assert_eq!(synchronize_calls, 2);
    assert_eq!(stream_drops, 1);
    let _ = close_plan_runtime(root);
}

#[test]
fn sequence_owner_drop_defers_blocking_backend_recovery() {
    let plan = execution_plan();
    let (driver, trace) = configured_driver(&plan, &[], &[]);
    let root = plan_runtime(&plan, driver, "blocking-sequence-owner-drop");
    let sequence = admit_logical_sequence(
        &root,
        "run.blocking-sequence-owner-drop",
        "request.blocking-sequence-owner-drop",
    );
    let mut stream = sequence.create_execution_stream().unwrap();
    let permit = sequence.activate(&mut stream).unwrap();
    drop(permit);

    let entered = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    let release = Arc::new(Barrier::new(ABANDONED_RECOVERY_CONCURRENT_WORKERS + 1));
    trace.lock().unwrap().synchronize_block = Some((Arc::clone(&entered), Arc::clone(&release)));
    drop(sequence);

    let (dropped_tx, dropped_rx) = mpsc::sync_channel(1);
    let drop_returned = std::thread::scope(|scope| {
        let drop_worker = std::thread::Builder::new()
            .name("vnext-blocking-sequence-drop".to_owned())
            .spawn_scoped(scope, move || {
                drop(stream);
                let _ = dropped_tx.send(());
            })
            .expect("the single bounded sequence-drop worker starts");
        let drop_returned = dropped_rx.recv_timeout(Duration::from_millis(250)).is_ok();
        drop_worker
            .join()
            .expect("the bounded sequence-drop worker does not panic");
        drop_returned
    });
    assert!(drop_returned, "sequence Drop waited on the backend");
    assert_eq!(root.deferred_cleanup_status().pending(), 1);
    assert_eq!(trace.lock().unwrap().runtime_synchronize_calls, 0);

    let close = PlanRuntimeResources::close(root);

    let (root, strong_count, deferred_cleanup) = match close {
        Ok(PlanRuntimeCloseOutcome::Referenced {
            resources,
            strong_count,
            deferred_cleanup,
        }) => (resources, strong_count, deferred_cleanup),
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("blocked cleanup released its plan root before quiescence")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };
    assert!(
        strong_count >= 2,
        "cleanup registry did not retain the plan root"
    );
    assert_eq!(deferred_cleanup.pending(), 1);

    let maintenance_root = Arc::clone(&root);
    let (cleanup, calls_while_blocked, stream_drops_while_blocked) = std::thread::scope(|scope| {
        let cleanup_worker = std::thread::Builder::new()
            .name("vnext-sequence-cleanup-recovery".to_owned())
            .spawn_scoped(scope, move || {
                maintenance_root.maintain_deferred_cleanups(1)
            })
            .expect("the single bounded sequence cleanup worker starts");
        entered.wait();
        let (calls_while_blocked, stream_drops_while_blocked) = {
            let trace = trace.lock().unwrap();
            (trace.runtime_synchronize_calls, trace.stream_drops)
        };
        release.wait();
        let cleanup = cleanup_worker
            .join()
            .expect("the bounded sequence cleanup worker does not panic")
            .expect("the bounded sequence cleanup call is valid");
        (cleanup, calls_while_blocked, stream_drops_while_blocked)
    });
    assert_eq!(calls_while_blocked, 1);
    assert_eq!(stream_drops_while_blocked, 0);
    assert_eq!(cleanup.completed(), 1);
    assert_eq!(cleanup.status_after().pending(), 0);
    assert_eq!(Arc::strong_count(&root), 1);
    let final_trace = trace.lock().unwrap();
    assert_eq!(final_trace.runtime_synchronize_calls, 1);
    assert_eq!(final_trace.stream_drops, 1);
    drop(final_trace);
    let _ = close_plan_runtime(root);
}

#[test]
fn closing_root_rejects_every_parent_to_child_derivation() {
    let plan = execution_plan();
    let (driver, _trace) = configured_driver(&plan, &[], &[]);
    let root = plan_runtime(&plan, driver, "closing-child-derivation");
    let sequence = admit_logical_sequence(
        &root,
        "run.closing-sequence-child",
        "request.closing-sequence-child",
    );
    let request = Arc::clone(sequence.request_resources());
    let mut existing_stream = sequence.create_execution_stream().unwrap();
    let session = sequence.open_session().unwrap();
    let existing_batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let existing_step = begin_single_participant_step(&root, &existing_batch);

    let root = match PlanRuntimeResources::close(root) {
        Ok(PlanRuntimeCloseOutcome::Referenced { resources, .. }) => resources,
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {
            panic!("live resource parents unexpectedly allowed root close")
        }
        Err(failure) => panic!("referenced root close failed: {:?}", failure.failure()),
    };
    assert!(root.is_closing());

    let sequence_request = SequenceResourceAdmissionRequest::new(
        one_token_work(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(matches!(
        request.try_admit_sequence(sequence_request),
        Err(error) if closing_error(&error)
    ));
    assert!(matches!(
        sequence.open_session(),
        Err(error) if closing_error(&error)
    ));
    assert!(matches!(
        sequence.create_execution_stream(),
        Err(ExecutionStreamCreationError::Contract(error)) if closing_error(&error)
    ));
    assert!(matches!(
        sequence.activate(&mut existing_stream),
        Err(error) if closing_error(&error)
    ));
    assert!(matches!(
        ExecutionBatchParticipants::new(vec![Arc::clone(&session)]),
        Err(error) if closing_error(&error)
    ));

    let step_request = StepResourceAdmissionRequest::new(
        existing_batch
            .bind_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(matches!(
        existing_batch.try_begin_step(step_request, existing_step.execution_lane()),
        Err(error) if closing_error(&error)
    ));
    let invocation_request = InvocationResourceAdmissionRequest::for_all_step_participants(
        id("node.main"),
        existing_step
            .bind_all_invocation_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(matches!(
        existing_step.try_admit_invocation(invocation_request),
        Err(error) if closing_error(&error)
    ));

    drop(existing_stream);
    drop(existing_step);
    drop(existing_batch);
    drop(session);
    drop(sequence);
    drop(request);
    let _ = close_plan_runtime(root);
}
