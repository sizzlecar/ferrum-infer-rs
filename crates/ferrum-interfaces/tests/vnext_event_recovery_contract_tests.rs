mod vnext_event_contract;

use vnext_event_contract::*;

#[test]
fn vnext_event_recovery_contract() {
    const EXPECTED_CASES: usize = 20;
    let mut passed = 0_usize;
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("recovery", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    let ProvisionedRuntimePool {
        resources: plan_resources,
        runtime: _,
        evidence: pool_evidence,
        journal: _,
        committed_snapshot,
    } = provision_runtime_pool(&plan, &topology, "recovery");
    let ProvisionedRuntimePool {
        resources: abort_plan_resources,
        runtime: _,
        evidence: abort_pool_evidence,
        journal: _,
        committed_snapshot: abort_committed_snapshot,
    } = provision_runtime_pool(&plan, &topology, "recovery-abort");
    let close_receipt = close_plan_runtime(plan_resources);
    let abort_close_receipt = close_plan_runtime(abort_plan_resources);
    let (mut lease_committed, _, _, _) = provision_pool(&plan, &topology, "recovery-lease");
    lease_committed.defer_all().unwrap();

    let (failure_evidence, failure_opened, failure_reserved, failure_event, anchor, recovery) =
        failure_recovery_pair(&plan, &topology, "one");
    let (_, _, _, _, _, wrong_recovery) = failure_recovery_pair(&plan, &topology, "two");
    let mut failure_cursor = ResourcePoolEventCursor::new(failure_evidence.clone());
    failure_cursor.observe(&failure_opened).unwrap();
    failure_cursor.observe(&failure_reserved).unwrap();
    failure_cursor.observe(&failure_event).unwrap();
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&failure_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &anchor,
            ))
            .unwrap()
            == failure_event,
    );
    check(
        &mut passed,
        ResourcePoolEvent::recovery_completed(4, pool_timestamp(4), &failure_evidence, &anchor)
            .is_err(),
    );
    let mut mislabeled_failure = serde_json::to_value(&failure_event).unwrap();
    mislabeled_failure["kind"] = json!("resource_recovery_completed");
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&mislabeled_failure).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &anchor,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::failed(3, pool_timestamp(3), &pool_evidence, &anchor).is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&failure_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &pool_evidence,
                &anchor,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::recovery_completed(
            4,
            pool_timestamp(4),
            &failure_evidence,
            &wrong_recovery,
        )
        .is_err()
            && failure_cursor.last_sequence() == 3,
    );
    check(
        &mut passed,
        wrong_recovery
            .validate_recovery_continuation(&anchor)
            .is_err(),
    );
    let recovery_event =
        ResourcePoolEvent::recovery_completed(4, pool_timestamp(4), &failure_evidence, &recovery)
            .unwrap();
    failure_cursor.observe(&recovery_event).unwrap();
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&recovery_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &recovery,
            ))
            .unwrap()
            == recovery_event,
    );
    check(
        &mut passed,
        ResourcePoolEvent::failed(4, pool_timestamp(4), &failure_evidence, &recovery).is_err(),
    );
    let mut mislabeled_recovery = serde_json::to_value(&recovery_event).unwrap();
    mislabeled_recovery["kind"] = json!("resource_failed");
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&mislabeled_recovery).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &failure_evidence,
                &recovery,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::recovery_completed(4, pool_timestamp(4), &pool_evidence, &recovery)
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&recovery_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::failure(
                &pool_evidence,
                &recovery,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        recovery.failure_id() == anchor.failure_id()
            && recovery.recovery_complete()
            && recovery.validate_recovery_continuation(&anchor).is_ok(),
    );

    check(
        &mut passed,
        ResourcePoolEvent::closed(5, pool_timestamp(5), &pool_evidence, &committed_snapshot)
            .is_err(),
    );
    check(
        &mut passed,
        close_receipt.released_static_resources() == committed_snapshot.entries().len(),
    );
    check(
        &mut passed,
        close_receipt.evidence().static_pool_identity()
            == Some(pool_evidence.admission().pool_identity()),
    );
    check(
        &mut passed,
        close_receipt.evidence().static_provisioning_identity()
            == Some(pool_evidence.provisioning_identity()),
    );
    check(
        &mut passed,
        close_receipt.evidence().plan_hash() == pool_evidence.admission().plan_hash(),
    );

    check(
        &mut passed,
        abort_close_receipt.released_static_resources() == abort_committed_snapshot.entries().len(),
    );
    check(
        &mut passed,
        abort_close_receipt.evidence().static_pool_identity()
            == Some(abort_pool_evidence.admission().pool_identity()),
    );
    lease_committed.resume_all().unwrap();
    let _lease_released = lease_committed.release().unwrap();

    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT EVENT RECOVERY PASS: {passed}/{EXPECTED_CASES}");
}
