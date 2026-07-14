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

fn runtime_implementation_authority_contract(plan: &ExecutionPlan, passed: &mut usize) {
    check(
        passed,
        plan.payload().device_runtime_implementation_fingerprint() == sha('d'),
    );
    let propagation = admit_resources(plan, id("request.runtime-fingerprint.propagation")).unwrap();
    check(
        passed,
        propagation
            .binding()
            .device_runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    );
    check(
        passed,
        propagation
            .binding()
            .pool_identity()
            .device_runtime_implementation_fingerprint()
            == plan.payload().device_runtime_implementation_fingerprint(),
    );
    drop(propagation);

    let mut wrong_runtime = sequence_runtime(plan);
    let wrong_trace = Arc::clone(&wrong_runtime.trace);
    Arc::get_mut(&mut wrong_runtime)
        .unwrap()
        .descriptor
        .runtime_implementation_fingerprint = sha('f');
    check(
        passed,
        required_static(
            plan,
            wrong_runtime,
            id("request.runtime-fingerprint.wrong-admission"),
        )
        .is_err(),
    );
    check(
        passed,
        wrong_trace.lock().unwrap().runtime_allocate_calls == 0,
    );

    let admitted_runtime = sequence_runtime(plan);
    let exact = required_static(
        plan,
        Arc::clone(&admitted_runtime),
        id("request.runtime-instance.exact-admission"),
    )
    .unwrap();
    let exact_identity = ResourceTransactionIdentity::for_admission(
        exact.binding(),
        id("run.runtime-instance.impostor-driver"),
        id("transaction.runtime-instance.impostor-driver"),
    );
    let (impostor_driver, impostor_trace) = configured_driver(plan, &[], &[]);
    check(
        passed,
        ResourceTransaction::begin(impostor_driver, exact_identity, exact).is_err(),
    );
    check(passed, impostor_trace.lock().unwrap().calls.is_empty());

    let (drifting_driver, drifting_trace) = configured_driver(plan, &[], &[]);
    drifting_trace.lock().unwrap().drift_on_allocate = true;
    let reserved = transaction(plan, drifting_driver, "descriptor-drift-allocation")
        .reserve()
        .unwrap();
    let rejected_commit = reserved.commit();
    check(passed, rejected_commit.is_err());
    drop(rejected_commit);
    check(
        passed,
        drifting_trace.lock().unwrap().runtime_allocate_calls == 1,
    );

    let alternate_plan = execution_plan_with_policy_and_runtime_fingerprint(policy(), sha('f'));
    check(
        passed,
        alternate_plan
            .payload()
            .device_runtime_implementation_fingerprint()
            == sha('f'),
    );
    let anchor = admit_resources(plan, id("request.runtime-fingerprint.account-anchor")).unwrap();
    check(
        passed,
        admit_resources(
            &alternate_plan,
            id("request.runtime-fingerprint.account-conflict"),
        )
        .is_err(),
    );
    drop(anchor);
    check(
        passed,
        admit_resources(
            &alternate_plan,
            id("request.runtime-fingerprint.account-after-drop"),
        )
        .is_ok(),
    );

    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let policy = policy();
    let registry = operation_registry(&catalog);
    let planning = registry.planning();
    let resolution = node_resolution(&family, &catalog, &policy, &planning);
    let mut wire = serde_json::to_value(plan).unwrap();
    wire["payload"]["device_runtime_implementation_fingerprint"] = json!(sha('f'));
    rehash_plan_json(&mut wire);
    check(
        passed,
        ExecutionPlan::from_json_validated(
            &serde_json::to_vec(&wire).unwrap(),
            &family,
            &catalog,
            &policy,
            vec![resolution],
        )
        .is_err(),
    );
}

fn device_global_capacity_contract(base_plan: &ExecutionPlan, passed: &mut usize) {
    let expected_peak = base_plan.payload().memory().static_bytes();
    let scaled = |factor: u64| {
        expected_peak
            .checked_mul(factor)
            .expect("bounded resource test capacity does not overflow")
    };
    let plan = execution_plan_with_policy(policy_with_memory_id(
        "runtime-policy.resource-test.capacity-primary",
        scaled(20),
        scaled(4),
        1_000,
    ));
    let second_plan = execution_plan_with_policy(policy_with_memory_id(
        "runtime-policy.resource-test.second-plan",
        scaled(20),
        scaled(3),
        1_000,
    ));
    let peak = plan.payload().memory().static_bytes();
    assert_eq!(peak, expected_peak);
    let raw_capacity = plan.payload().memory().device_capacity_bytes();
    let effective_usable_capacity = plan
        .payload()
        .memory()
        .usable_capacity_bytes()
        .min(second_plan.payload().memory().usable_capacity_bytes());
    let maximum_claims = (effective_usable_capacity / peak) as usize;
    assert_eq!(maximum_claims, RESOURCE_CAPACITY_TEST_MAXIMUM_CLAIMS);
    check(passed, maximum_claims >= 2);
    check(passed, effective_usable_capacity < raw_capacity);
    check(
        passed,
        second_plan.payload().device_id() == plan.payload().device_id()
            && second_plan.payload().memory().usable_capacity_bytes()
                != plan.payload().memory().usable_capacity_bytes(),
    );
    check(
        passed,
        try_execution_plan_with_policy_and_runtime_fingerprint(
            policy_with_memory_id(
                "runtime-policy.resource-test.own-usable-reject",
                scaled(2),
                scaled(2) - 1,
                1_000,
            ),
            sha('d'),
        )
        .is_err(),
    );

    let mut permits = Vec::new();
    for index in 0..maximum_claims {
        let source = if index % 2 == 0 { &plan } else { &second_plan };
        permits.push(
            admit_resources(source, id(format!("request.capacity.multi-plan.{index}"))).unwrap(),
        );
    }
    check(
        passed,
        admit_resources(&second_plan, id("request.capacity.over-limit")).is_err(),
    );
    check(
        passed,
        permits
            .windows(2)
            .all(|pair| pair[0].binding().pool_id() != pair[1].binding().pool_id()),
    );
    drop(permits.pop());
    let replacement =
        admit_resources(&second_plan, id("request.capacity.after-permit-drop")).unwrap();
    check(passed, replacement.binding().admitted_bytes() == peak);
    drop(replacement);
    drop(permits);

    let held = admit_resources(&plan, id("request.capacity.metadata-anchor")).unwrap();
    let different_capacity = execution_plan_with_policy(policy_with_memory_id(
        "runtime-policy.resource-test.different-capacity",
        scaled(18),
        scaled(3),
        1_000,
    ));
    let different_held = admit_resources(
        &different_capacity,
        id("request.capacity.different-usable-a-then-b"),
    )
    .unwrap();
    check(passed, different_held.binding().admitted_bytes() == peak);
    drop(different_held);
    drop(held);
    let reverse_anchor = admit_resources(
        &different_capacity,
        id("request.capacity.different-usable-b-anchor"),
    )
    .unwrap();
    check(
        passed,
        admit_resources(&plan, id("request.capacity.different-usable-b-then-a")).is_ok(),
    );
    drop(reverse_anchor);

    let permit = admit_resources(&plan, id("request.capacity.begin-failure")).unwrap();
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        id("run.capacity.begin-failure"),
        id("transaction.capacity.begin-failure"),
    );
    let (mut wrong_driver, _) = configured_driver(&plan, &[], &[]);
    wrong_driver.device_capacity_bytes -= 1;
    check(
        passed,
        ResourceTransaction::begin(wrong_driver, identity, permit).is_err(),
    );
    let mut after_begin_failure = Vec::new();
    for index in 0..maximum_claims {
        after_begin_failure.push(
            admit_resources(&plan, id(format!("request.capacity.begin-return.{index}"))).unwrap(),
        );
    }
    check(passed, after_begin_failure.len() == maximum_claims);
    drop(after_begin_failure);

    let (driver, _) = configured_driver(&plan, &[], &[]);
    let rolled_back = transaction(&plan, driver, "capacity-terminal-rollback")
        .reserve()
        .unwrap()
        .rollback()
        .unwrap();
    let mut while_terminal_held = Vec::new();
    for index in 0..maximum_claims {
        while_terminal_held.push(
            admit_resources(
                &plan,
                id(format!("request.capacity.terminal-return.{index}")),
            )
            .unwrap(),
        );
    }
    check(passed, while_terminal_held.len() == maximum_claims);
    drop(while_terminal_held);
    drop(rolled_back);

    let first_release = failure_key("release", &plan_resources(&plan)[0]);
    let (driver, trace) = configured_driver(&plan, &[(first_release.as_str(), 1)], &[]);
    trace.lock().unwrap().retain_ownership = true;
    let committed = transaction(&plan, driver, "capacity-durable-quarantine")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let quarantined = expect_err(committed.release()).quarantine().unwrap();
    check(passed, trace.lock().unwrap().durable_ownership.len() == 1);
    let quarantine_owned_shape = {
        let trace = trace.lock().unwrap();
        trace.durable_ownership[0].claimed_bytes() == peak
            && trace.durable_ownership[0].buffers().len()
                == plan.payload().memory().static_allocations().len()
    };
    check(passed, quarantine_owned_shape);
    let mut under_quarantine = Vec::new();
    for index in 0..(maximum_claims - 1) {
        under_quarantine.push(
            admit_resources(&plan, id(format!("request.capacity.quarantined.{index}"))).unwrap(),
        );
    }
    check(
        passed,
        admit_resources(&plan, id("request.capacity.quarantine-still-claimed")).is_err(),
    );
    drop(under_quarantine);
    let cleaned = std::mem::take(&mut trace.lock().unwrap().durable_ownership);
    drop(cleaned);
    check(
        passed,
        admit_resources(&plan, id("request.capacity.quarantine-cleaned")).is_ok(),
    );
    drop(quarantined);

    let (driver, trace) = configured_driver(&plan, &[], &[]);
    trace.lock().unwrap().retain_ownership = true;
    let committed = transaction(&plan, driver, "capacity-durable-abandon")
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    drop(committed);
    let abandon_owned_shape = {
        let trace = trace.lock().unwrap();
        trace.durable_ownership.len() == 1 && trace.abandon_claimed_bytes == [peak]
    };
    check(passed, abandon_owned_shape);
    check(
        passed,
        trace.lock().unwrap().durable_ownership[0].reason() == ResourceOwnershipReason::Abandon,
    );
    let cleaned = std::mem::take(&mut trace.lock().unwrap().durable_ownership);
    drop(cleaned);

    // Capacity cardinality is not an OS-thread cardinality. Fill every slot but
    // one serially, then use one bounded worker plus the calling test thread to
    // prove that exactly one contender can atomically take the final slot.
    let mut concurrent_prefill = Vec::with_capacity(maximum_claims - 1);
    for index in 0..(maximum_claims - 1) {
        let source = if index % 2 == 0 { &plan } else { &second_plan };
        concurrent_prefill.push(
            admit_resources(
                source,
                id(format!("request.capacity.concurrent-prefill.{index}")),
            )
            .unwrap(),
        );
    }
    let barrier = Arc::new(Barrier::new(RESOURCE_CAPACITY_CONCURRENT_WORKERS + 1));
    let results = std::thread::scope(|scope| {
        let worker_barrier = Arc::clone(&barrier);
        let worker = std::thread::Builder::new()
            .name("vnext-resource-capacity-contender".to_owned())
            .spawn_scoped(scope, move || {
                worker_barrier.wait();
                admit_resources(&second_plan, id("request.capacity.concurrent.worker"))
            })
            .expect("bounded resource-capacity worker starts");
        barrier.wait();
        let caller = admit_resources(&plan, id("request.capacity.concurrent.caller"));
        vec![
            caller,
            worker
                .join()
                .expect("bounded resource-capacity worker does not panic"),
        ]
    });
    let successful = results.iter().filter(|result| result.is_ok()).count();
    check(passed, successful == 1);
    check(passed, results.len() - successful == 1);
    drop(results);
    drop(concurrent_prefill);
}

#[test]
fn resource_capacity_concurrency_is_bounded() {
    let plan = execution_plan();
    let mut passed = 0;
    device_global_capacity_contract(&plan, &mut passed);
    assert_eq!(passed, DEVICE_GLOBAL_CAPACITY_CASES);
    println!(
        "\nVNEXT RESOURCE CAPACITY THREAD BOUND PASS: {passed}/{DEVICE_GLOBAL_CAPACITY_CASES}"
    );
}

#[test]
fn runtime_implementation_authority_is_exact() {
    const EXPECTED_CASES: usize = 13;
    let plan = execution_plan();
    let mut passed = 0;
    runtime_implementation_authority_contract(&plan, &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT RUNTIME IMPLEMENTATION AUTHORITY PASS: {passed}/{EXPECTED_CASES}");
}
