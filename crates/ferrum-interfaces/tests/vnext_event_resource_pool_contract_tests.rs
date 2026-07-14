mod vnext_event_contract;

use vnext_event_contract::*;

#[test]
fn vnext_event_resource_pool_contract() {
    const EXPECTED_CASES: usize = 27;
    let mut passed = 0_usize;
    let runtime_catalog = catalog();
    let operation_registry = make_operation_registry(&runtime_catalog);
    let plan = execution_plan("resource-pool", &operation_registry);
    let topology = TrustedExecutionTopology::from_plan(&plan).unwrap();
    let resolved = resolved_model_plan(&plan, "resource-pool", &operation_registry);
    let ProvisionedRuntimePool {
        resources: plan_resources,
        runtime: plan_runtime,
        evidence: pool_evidence,
        journal: pool_journal,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "resource-pool");
    let SequenceEvidence {
        active,
        completed,
        submissions,
        completions,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.resource-pool.one",
        "request.resource-pool.one",
        2,
    );
    let journal = request_journal(&plan, &active, &completed, &submissions, &completions, 2);
    let cursor = observe_journal(
        &journal,
        &topology,
        &active,
        &completed,
        &submissions,
        &completions,
    )
    .unwrap();
    let SequenceEvidence {
        active: active_two,
        completed: completed_two,
        submissions: submissions_two,
        completions: completions_two,
    } = execute_sequence(
        &plan_resources,
        &plan_runtime,
        &resolved,
        &operation_registry,
        "run.resource-pool.two",
        "request.resource-pool.two",
        1,
    );
    let ProvisionedRuntimePool {
        resources: abort_plan_resources,
        runtime: _,
        evidence: abort_pool_evidence,
        journal: _,
        committed_snapshot: _,
    } = provision_runtime_pool(&plan, &topology, "resource-pool-foreign");

    let mut pool_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    for event in &pool_journal {
        pool_cursor.observe(event).unwrap();
    }
    check(
        &mut passed,
        pool_cursor.is_open() && pool_cursor.last_sequence() == 3,
    );
    check(&mut passed, cursor.is_terminal() && pool_cursor.is_open());

    let opened_wire =
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[0]).unwrap())
            .unwrap();
    check(
        &mut passed,
        opened_wire
            .revalidate(&TrustedResourcePoolEventContext::opened(&pool_evidence))
            .unwrap()
            == pool_journal[0],
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[0]).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::opened(
                &abort_pool_evidence,
            ))
            .is_err(),
    );
    let (reserve_receipt, reserve_context) = match pool_journal[1].detail() {
        ResourcePoolEventDetail::Transition { receipt, context } => (receipt, context),
        _ => unreachable!(),
    };
    let transition_wire =
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[1]).unwrap())
            .unwrap();
    check(
        &mut passed,
        transition_wire
            .revalidate(&TrustedResourcePoolEventContext::transition(
                &pool_evidence,
                reserve_context,
            ))
            .unwrap()
            == pool_journal[1],
    );
    check(
        &mut passed,
        ResourcePoolEvent::transition(
            2,
            pool_timestamp(2),
            &abort_pool_evidence,
            reserve_receipt,
            reserve_context,
        )
        .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_journal[1]).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::transition(
                &abort_pool_evidence,
                reserve_context,
            ))
            .is_err(),
    );
    let transition_value = serde_json::to_value(&pool_journal[1]).unwrap();
    let mut pool_unknown_top = transition_value.clone();
    pool_unknown_top["unknown_top"] = json!(true);
    let mut pool_unknown_identity = transition_value.clone();
    pool_unknown_identity["identity"]["unknown_identity"] = json!(true);
    let mut pool_unknown_detail = transition_value;
    pool_unknown_detail["detail"]["transition"]["receipt"]["unknown_nested"] = json!(true);
    for wire in [pool_unknown_top, pool_unknown_identity, pool_unknown_detail] {
        check(
            &mut passed,
            ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap()).is_err(),
        );
    }
    let mut pool_unknown_variant = serde_json::to_value(&pool_journal[1]).unwrap();
    pool_unknown_variant["detail"]["transition"]["extra"] = json!(true);
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&pool_unknown_variant).unwrap())
            .is_err(),
    );
    for (path, replacement) in [
        ("pool_id", json!(999_998_u64)),
        ("pool_identity_fingerprint", json!(sha('0'))),
        ("transaction_id", json!("transaction.wire-tampered")),
    ] {
        let mut wire = serde_json::to_value(&pool_journal[1]).unwrap();
        wire["identity"][path] = replacement;
        check(
            &mut passed,
            ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&wire).unwrap())
                .unwrap()
                .revalidate(&TrustedResourcePoolEventContext::transition(
                    &pool_evidence,
                    reserve_context,
                ))
                .is_err(),
        );
    }
    let mut context_tamper = serde_json::to_value(&pool_journal[1]).unwrap();
    context_tamper["detail"]["transition"]["context"]["action"] = json!("commit");
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&context_tamper).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::transition(
                &pool_evidence,
                reserve_context,
            ))
            .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&vec![b' '; MAX_RESOURCE_POOL_EVENT_WIRE_BYTES + 1])
            .is_err(),
    );
    let resource_pool_source = include_str!("../src/vnext/event/resource_pool.rs");
    check(
        &mut passed,
        resource_pool_source.contains(
            "#[derive(Debug, Clone, PartialEq, Eq, Serialize)]\npub struct ResourcePoolEvent",
        ) && resource_pool_source.contains("pub struct UnvalidatedResourcePoolEvent"),
    );

    let skipped = ResourcePoolEvent::transition(
        3,
        pool_timestamp(3),
        &pool_evidence,
        reserve_receipt,
        reserve_context,
    )
    .unwrap();
    let mut skipped_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    skipped_cursor.observe(&pool_journal[0]).unwrap();
    check(
        &mut passed,
        skipped_cursor.observe(&skipped).is_err() && skipped_cursor.last_sequence() == 1,
    );
    let reused_reserve_receipt = ResourcePoolEvent::transition(
        3,
        pool_timestamp(3),
        &pool_evidence,
        reserve_receipt,
        reserve_context,
    )
    .unwrap();
    let mut receipt_reuse_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    receipt_reuse_cursor.observe(&pool_journal[0]).unwrap();
    receipt_reuse_cursor.observe(&pool_journal[1]).unwrap();
    check(
        &mut passed,
        receipt_reuse_cursor
            .observe(&reused_reserve_receipt)
            .is_err()
            && receipt_reuse_cursor.last_sequence() == 2,
    );
    let timestamp_rewind = ResourcePoolEvent::transition(
        2,
        pool_timestamp(1),
        &pool_evidence,
        reserve_receipt,
        reserve_context,
    )
    .unwrap();
    let mut timestamp_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
    timestamp_cursor.observe(&pool_journal[0]).unwrap();
    check(
        &mut passed,
        timestamp_cursor.observe(&timestamp_rewind).is_err()
            && timestamp_cursor.last_sequence() == 1,
    );

    let (mut lease_committed, _, lease_evidence, _) =
        provision_pool(&plan, &topology, "lease-binding");
    let lease_receipt = lease_committed.defer_all().unwrap();
    let lease_context = lease_committed.latest_lease_validation_context().unwrap();
    let lease_event = ResourcePoolEvent::lease_transition(
        4,
        pool_timestamp(4),
        &lease_evidence,
        &lease_receipt,
        lease_context,
    )
    .unwrap();
    check(
        &mut passed,
        lease_event.kind() == ResourcePoolEventKind::ResourceLeaseTransition,
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&lease_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::lease_transition(
                &lease_evidence,
                lease_context,
            ))
            .unwrap()
            == lease_event,
    );
    check(
        &mut passed,
        ResourcePoolEvent::lease_transition(
            4,
            pool_timestamp(4),
            &abort_pool_evidence,
            &lease_receipt,
            lease_context,
        )
        .is_err(),
    );
    check(
        &mut passed,
        ResourcePoolEvent::decode_untrusted(&serde_json::to_vec(&lease_event).unwrap())
            .unwrap()
            .revalidate(&TrustedResourcePoolEventContext::lease_transition(
                &abort_pool_evidence,
                lease_context,
            ))
            .is_err(),
    );

    let journal_two = request_journal(
        &plan,
        &active_two,
        &completed_two,
        &submissions_two,
        &completions_two,
        1,
    );
    check(
        &mut passed,
        observe_journal(
            &journal_two,
            &topology,
            &active_two,
            &completed_two,
            &submissions_two,
            &completions_two,
        )
        .unwrap()
        .is_terminal(),
    );
    check(
        &mut passed,
        pool_cursor.is_open() && !pool_cursor.is_closed(),
    );
    check(
        &mut passed,
        active.sequence_authority() != active_two.sequence_authority()
            && active.activation_epoch() == active_two.activation_epoch()
            && active.fingerprint() != active_two.fingerprint()
            && active.static_pool_id() == active_two.static_pool_id(),
    );

    lease_committed.resume_all().unwrap();
    let _lease_released = lease_committed.release().unwrap();
    close_plan_runtime(plan_resources);
    close_plan_runtime(abort_plan_resources);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT EVENT RESOURCE POOL PASS: {passed}/{EXPECTED_CASES}");
}
