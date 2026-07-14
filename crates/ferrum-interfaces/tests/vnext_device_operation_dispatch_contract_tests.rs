mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn descriptor_and_registry_contract(passed: &mut usize) {
    let catalog = catalog();
    let mut zero_capacity = catalog.device().clone();
    zero_capacity.total_memory_bytes = 0;
    check(passed, zero_capacity.validate().is_err());

    let mut invalid_runtime = catalog.device().clone();
    invalid_runtime.runtime_implementation_fingerprint = "not-a-sha".to_owned();
    check(passed, invalid_runtime.validate().is_err());

    let behavior = Arc::new(Mutex::new(ProviderBehavior::Success));
    let trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let descriptor = catalog.providers_for(&id("operation.main")).unwrap()[0].clone();
    let duplicate = OperationRuntimeRegistry::<TestRuntime>::new(
        vec![Box::new(TestOperationContract {
            descriptor: operation(),
        })],
        vec![
            Box::new(TestProvider {
                descriptor: descriptor.clone(),
                behavior: Arc::clone(&behavior),
                trace: Arc::clone(&trace),
            }),
            Box::new(TestProvider {
                descriptor,
                behavior,
                trace,
            }),
        ],
    );
    check(passed, duplicate.is_err());
}

fn device_failure_contract(
    runtime: &TestRuntime,
    plan: &ExecutionPlan,
    operation_identity: &ExecutionIdentityEnvelope,
    passed: &mut usize,
) {
    let descriptor = runtime.descriptor();
    let device_only = ExecutionIdentityEnvelope::new(device_identity_parts(
        "run.device-only",
        "request.device-only",
        descriptor.id.clone(),
        descriptor.runtime_implementation_fingerprint.clone(),
    ))
    .unwrap();
    let failure =
        classify_device_error(runtime, device_only, &TestRuntimeError("device-only")).unwrap();
    check(passed, failure.failure().domain() == FailureDomain::Device);

    let missing_runtime = {
        let mut parts = device_identity_parts(
            "run.device-missing-runtime",
            "request.device-missing-runtime",
            descriptor.id.clone(),
            descriptor.runtime_implementation_fingerprint.clone(),
        );
        parts.runtime_implementation_fingerprint = None;
        ExecutionIdentityEnvelope::new(parts)
    };
    check(passed, missing_runtime.is_err());

    let wrong_runtime = ExecutionIdentityEnvelope::new(device_identity_parts(
        "run.device-wrong-runtime",
        "request.device-wrong-runtime",
        descriptor.id.clone(),
        sha('f'),
    ))
    .unwrap();
    check(
        passed,
        classify_device_error(runtime, wrong_runtime, &TestRuntimeError("wrong-runtime")).is_err(),
    );

    let wrong_device = ExecutionIdentityEnvelope::new(device_identity_parts(
        "run.device-wrong-device",
        "request.device-wrong-device",
        id("device.other"),
        descriptor.runtime_implementation_fingerprint.clone(),
    ))
    .unwrap();
    check(
        passed,
        classify_device_error(runtime, wrong_device, &TestRuntimeError("wrong-device")).is_err(),
    );

    let mut plan_parts = device_identity_parts(
        "run.device-plan",
        "request.device-plan",
        descriptor.id.clone(),
        descriptor.runtime_implementation_fingerprint.clone(),
    );
    plan_parts.plan_id = Some(plan.payload().plan_id().clone());
    plan_parts.plan_hash = Some(plan.plan_hash().clone());
    let plan_identity = ExecutionIdentityEnvelope::new(plan_parts).unwrap();
    check(
        passed,
        classify_device_error(runtime, plan_identity, &TestRuntimeError("plan")).is_ok(),
    );
    check(
        passed,
        classify_device_error(
            runtime,
            operation_identity.clone(),
            &TestRuntimeError("active-operation"),
        )
        .is_ok(),
    );

    let mut resource_parts = operation_identity.parts().clone();
    resource_parts.frame_id = None;
    resource_parts.node_invocation_id = None;
    resource_parts.node_id = None;
    resource_parts.operation_id = None;
    resource_parts.provider_id = None;
    resource_parts.active_sequence_slot = None;
    resource_parts.admission_generation = None;
    resource_parts.activation_epoch = None;
    resource_parts.active_sequence_fingerprint = None;
    resource_parts.resource_id = Some(id("resource.input"));
    resource_parts.resource_generation = Some(1);
    let resource_identity = ExecutionIdentityEnvelope::new(resource_parts).unwrap();
    check(
        passed,
        classify_device_error(runtime, resource_identity, &TestRuntimeError("resource")).is_ok(),
    );
}

fn operation_dispatch_contract(fixture: Fixture, passed: &mut usize) {
    let Fixture {
        registry,
        impostor_registry,
        resolved,
        plan,
        impostor_plan_hash,
        runtime,
        runtime_trace,
        provider_behavior,
        provider_trace,
        plan_resources,
    } = fixture;
    let node = &plan.payload().nodes()[0];
    check(passed, plan.plan_hash() == &impostor_plan_hash);
    let plan_bytes = plan.to_json().unwrap();
    let revalidated = revalidate_plan_for_registry(&plan_bytes, &registry);
    check(passed, revalidated == plan);
    let revalidated_impostor = revalidate_plan_for_registry(&plan_bytes, &impostor_registry);
    check(
        passed,
        revalidated_impostor.plan_hash() == plan.plan_hash()
            && revalidated_impostor.to_json().unwrap() == plan.to_json().unwrap()
            && revalidated_impostor != plan,
    );
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    let impostor_resolution =
        node_resolution(&family, &catalog, &runtime_policy, &impostor_registry);
    check(
        passed,
        plan.validate_against(&family, &catalog, &runtime_policy, &[impostor_resolution])
            .is_err(),
    );
    check(passed, registry.bind(&resolved, node.id()).is_ok());
    check(
        passed,
        impostor_registry.bind(&resolved, node.id()).is_err(),
    );
    let provider = registry.bind(&resolved, node.id()).unwrap();

    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.execute",
        "request.device-operation.execute",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    device_failure_contract(&runtime, &plan, &identity, passed);
    assert!(step.try_retire_normal().is_ok());

    type IdentityMutation = fn(&mut ExecutionIdentityParts);
    let mutations: [IdentityMutation; 13] = [
        |parts| parts.plan_id = Some(id(format!("plan/sha256/{}", sha('f')))),
        |parts| parts.plan_hash = Some(serde_json::from_value(json!(sha('f'))).unwrap()),
        |parts| {
            parts.resource_pool_id =
                Some(ResourcePoolId::try_from(parts.resource_pool_id.unwrap().get() + 1).unwrap())
        },
        |parts| parts.resource_pool_identity_fingerprint = Some(sha('f')),
        |parts| parts.activation_epoch = Some(parts.activation_epoch.unwrap() + 1),
        |parts| parts.active_sequence_slot = Some(parts.active_sequence_slot.unwrap() + 1),
        |parts| parts.request_id = id("request.device-operation.wrong"),
        |parts| parts.run_id = id("run.device-operation.wrong"),
        |parts| {
            parts.frame_id =
                Some(ExecutionFrameId::try_from(parts.frame_id.unwrap().get() + 1).unwrap())
        },
        |parts| {
            parts.node_invocation_id = Some(
                NodeInvocationId::try_from(parts.node_invocation_id.unwrap().get() + 1).unwrap(),
            )
        },
        |parts| parts.provider_id = Some(id("provider.operation.wrong")),
        |parts| parts.node_id = Some(id("node.wrong")),
        |parts| parts.runtime_implementation_fingerprint = Some(sha('f')),
    ];

    for mutate in mutations {
        let step = begin_single_participant_step(&plan_resources, &batch);
        let frame_id = step.participant_frames().next().unwrap().frame_id();
        let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
        let mut parts = operation_identity(&plan, &active, frame_id, invocation_id)
            .parts()
            .clone();
        mutate(&mut parts);
        let wrong = ExecutionIdentityEnvelope::new(parts).unwrap();
        check(
            passed,
            matches!(
                encode_and_submit_single(
                    &provider,
                    &resolved,
                    &wrong,
                    &frame_id,
                    &invocation_id,
                    node.id(),
                    &active,
                    admit_single_participant_invocation(&plan_resources, &step, node.id()),
                    &lane,
                    &reaper,
                ),
                Err(OperationDispatchError::Contract(_))
            ),
        );
        assert!(step.try_retire_normal().is_ok());
    }
    for completed in [true, false] {
        let step = begin_single_participant_step(&plan_resources, &batch);
        let frame_id = step.participant_frames().next().unwrap().frame_id();
        let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
        let mut parts = operation_identity(&plan, &active, frame_id, invocation_id)
            .parts()
            .clone();
        if completed {
            parts.completed_sequence_fingerprint = Some(sha('c'));
        } else {
            parts.aborted_sequence_fingerprint = Some(sha('a'));
        }
        let terminal = ExecutionIdentityEnvelope::new(parts).unwrap();
        let failure_rejected = OperationFailure::new(
            terminal.clone(),
            ProfilePhase::Decode,
            "operation_failed",
            "terminal sequence cannot report an active operation failure",
            false,
        )
        .is_err();
        let dispatch_rejected = matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &terminal,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        );
        check(passed, failure_rejected && dispatch_rejected);
        assert!(step.try_retire_normal().is_ok());
    }
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);

    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let mut resource_item = operation_identity(&plan, &active, frame_id, invocation_id)
        .parts()
        .clone();
    resource_item.resource_id = Some(id("resource.input"));
    resource_item.resource_generation = Some(active.sequence_authority().generation());
    let resource_item = ExecutionIdentityEnvelope::new(resource_item).unwrap();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &resource_item,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    assert!(step.try_retire_normal().is_ok());

    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let mut resource_batch = operation_identity(&plan, &active, frame_id, invocation_id)
        .parts()
        .clone();
    resource_batch.resource_batch_fingerprint = Some(sha('f'));
    let resource_batch = ExecutionIdentityEnvelope::new(resource_batch).unwrap();
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &resource_batch,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    assert!(step.try_retire_normal().is_ok());
    check(passed, provider_trace.lock().unwrap().encode_calls == 0);

    *provider_behavior.lock().unwrap() = ProviderBehavior::WrongIdentity;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    assert!(step.try_retire_normal().is_ok());

    *provider_behavior.lock().unwrap() = ProviderBehavior::WrongPhase;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(passed, runtime_trace.lock().unwrap().submit_calls == 0);
    assert!(step.try_retire_normal().is_ok());

    *provider_behavior.lock().unwrap() = ProviderBehavior::Success;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let receipt = encode_and_submit_single(
        &provider,
        &resolved,
        &identity,
        &frame_id,
        &invocation_id,
        node.id(),
        &active,
        admit_single_participant_invocation(&plan_resources, &step, node.id()),
        &lane,
        &reaper,
    )
    .unwrap();
    check(
        passed,
        receipt.batch_identity().participants()[0].identity() == &identity,
    );
    check(
        passed,
        matches!(receipt.poll(), Ok(CompletionObservation::Terminal(_))),
    );
    check(passed, provider_trace.lock().unwrap().encode_calls == 3);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 1);
    check(
        passed,
        provider_trace.lock().unwrap().component_resources
            == BTreeSet::from([id("resource.weight.left"), id("resource.weight.right")]),
    );
    check(
        passed,
        provider_trace.lock().unwrap().view_resources
            == BTreeSet::from([
                id("resource.input"),
                id("resource.intermediate"),
                id("resource.weight.left"),
                id("resource.weight.right"),
            ]),
    );
    let step_receipt = step.try_retire_normal().unwrap();
    check(
        passed,
        step_receipt.participants()[0]
            .assignment()
            .sequence_authority()
            == active.sequence_authority(),
    );
    drop(receipt);

    let encode_before_tamper = provider_trace.lock().unwrap().encode_calls;
    let submit_before_tamper = runtime_trace.lock().unwrap().submit_calls;
    runtime_trace.lock().unwrap().tamper_buffer_descriptor = true;
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(frame_id.get()).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    check(
        passed,
        matches!(
            encode_and_submit_single(
                &provider,
                &resolved,
                &identity,
                &frame_id,
                &invocation_id,
                node.id(),
                &active,
                admit_single_participant_invocation(&plan_resources, &step, node.id()),
                &lane,
                &reaper,
            ),
            Err(OperationDispatchError::Contract(_))
        ),
    );
    check(
        passed,
        provider_trace.lock().unwrap().encode_calls == encode_before_tamper,
    );
    check(
        passed,
        runtime_trace.lock().unwrap().submit_calls == submit_before_tamper,
    );
    runtime_trace.lock().unwrap().tamper_buffer_descriptor = false;
    assert!(step.try_retire_normal().is_ok());

    wire_limit_contract(&plan, &identity, passed);
    check(passed, session.try_complete().is_ok());
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    close_plan_runtime(plan_resources, passed);
}

fn wire_limit_contract(
    plan: &ExecutionPlan,
    identity: &ExecutionIdentityEnvelope,
    passed: &mut usize,
) {
    let plan_bytes = plan.to_json().unwrap();
    check(passed, ExecutionPlan::decode_untrusted(&plan_bytes).is_ok());
    let at_plan_limit = vec![b' '; MAX_EXECUTION_PLAN_WIRE_BYTES];
    let message =
        serialization_message(ExecutionPlan::decode_untrusted(&at_plan_limit).unwrap_err());
    check(passed, !message.contains("exceeds limit"));
    let over_plan_limit = vec![b' '; MAX_EXECUTION_PLAN_WIRE_BYTES + 1];
    let message =
        serialization_message(ExecutionPlan::decode_untrusted(&over_plan_limit).unwrap_err());
    check(passed, message.contains("exceeds limit"));

    let failure = OperationFailure::new(
        identity.clone(),
        ProfilePhase::Decode,
        "operation_failed",
        "operation failed",
        false,
    )
    .unwrap();
    let failure_bytes = serde_json::to_vec(&failure).unwrap();
    let unvalidated = OperationFailure::decode_untrusted(&failure_bytes).unwrap();
    check(
        passed,
        unvalidated
            .revalidate(identity, ProfilePhase::Decode)
            .is_ok(),
    );
    let at_failure_limit = vec![b' '; MAX_OPERATION_FAILURE_WIRE_BYTES];
    let message =
        serialization_message(OperationFailure::decode_untrusted(&at_failure_limit).unwrap_err());
    check(passed, !message.contains("exceeds limit"));
    let over_failure_limit = vec![b' '; MAX_OPERATION_FAILURE_WIRE_BYTES + 1];
    let message =
        serialization_message(OperationFailure::decode_untrusted(&over_failure_limit).unwrap_err());
    check(passed, message.contains("exceeds limit"));
}

fn submit_descriptor_drift_contract(fixture: Fixture, passed: &mut usize) {
    let Fixture {
        registry,
        resolved,
        plan,
        runtime,
        runtime_trace,
        provider_trace,
        plan_resources,
        ..
    } = fixture;
    let node = &plan.payload().nodes()[0];
    let provider = registry.bind(&resolved, node.id()).unwrap();
    let resources = logical_resources(
        &plan_resources,
        "run.device-operation.submit-drift",
        "request.device-operation.submit-drift",
    );
    let session = resources.open_session().unwrap();
    let active = TrustedActiveSequenceBinding::from_session(&session).unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = begin_single_participant_step(&plan_resources, &batch);
    let frame_id = step.participant_frames().next().unwrap().frame_id();
    let invocation_id = NodeInvocationId::try_from(1).unwrap();
    let identity = operation_identity(&plan, &active, frame_id, invocation_id);
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let reaper = CompletionReaper::new();
    runtime_trace.lock().unwrap().drift_on_submit = true;
    let completion = match encode_and_submit_single(
        &provider,
        &resolved,
        &identity,
        &frame_id,
        &invocation_id,
        node.id(),
        &active,
        admit_single_participant_invocation(&plan_resources, &step, node.id()),
        &lane,
        &reaper,
    ) {
        Err(OperationDispatchError::PostSubmitContract { completion, .. }) => completion,
        _ => panic!("submit descriptor drift did not retain a completion handle"),
    };
    check(passed, reaper.retained_count() == 1);
    check(passed, provider_trace.lock().unwrap().encode_calls == 1);
    check(passed, runtime_trace.lock().unwrap().submit_calls == 1);
    check(passed, lane.is_fail_closed());
    let step = match step.try_retire_normal() {
        Err(failure) => failure.into_step(),
        Ok(_) => panic!("pending drift fence released invocation resources"),
    };
    let terminal = match completion.poll().unwrap() {
        CompletionObservation::Terminal(receipt) => receipt,
        other => panic!("descriptor drift did not produce a quiescent terminal: {other:?}"),
    };
    check(
        passed,
        matches!(
            terminal.disposition(),
            OperationCompletionDisposition::ContractFailedButQuiescent(failure)
                if failure.reason().contains("descriptor")
        ),
    );
    runtime
        .use_alternate_descriptor
        .store(false, Ordering::Release);
    runtime_trace.lock().unwrap().drift_on_submit = false;
    check(passed, reaper.retained_count() == 0);
    check(passed, lane.in_flight_count() == 0);
    check(passed, step.try_retire_normal().is_ok());
    check(passed, session.try_complete().is_ok());
    drop(completion);
    drop(reaper);
    drop(lane);
    drop(batch);
    drop(session);
    drop(resources);
    close_plan_runtime(plan_resources, passed);
}

fn wrong_runtime_admission_contract(passed: &mut usize) {
    let catalog = catalog();
    let behavior = Arc::new(Mutex::new(ProviderBehavior::Success));
    let provider_trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let registry = operation_registry(&catalog, behavior, provider_trace);
    let plan = plan_for_registry(&registry);
    let (runtime, runtime_trace) = runtime(&catalog);
    runtime
        .use_alternate_descriptor
        .store(true, Ordering::Release);
    check(
        passed,
        plan.provision_static(runtime, id("request.device-operation.wrong-runtime"))
            .is_err(),
    );
    check(passed, runtime_trace.lock().unwrap().allocation_calls == 0);
}

#[test]
fn device_operation_dispatch_contract_is_exhaustive() {
    const EXPECTED_CASES: usize = 70;
    let mut passed = 0;
    descriptor_and_registry_contract(&mut passed);
    wrong_runtime_admission_contract(&mut passed);
    operation_dispatch_contract(fixture(), &mut passed);
    submit_descriptor_drift_contract(fixture(), &mut passed);
    assert_eq!(passed, EXPECTED_CASES);
    println!("\nVNEXT DEVICE OPERATION DISPATCH PASS: {passed}/{EXPECTED_CASES}");
}
