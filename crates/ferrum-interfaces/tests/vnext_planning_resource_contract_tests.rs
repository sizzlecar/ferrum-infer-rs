mod vnext_core_contract;

use vnext_core_contract::*;

#[test]
fn operation_resource_contract_requires_explicit_presence_and_alignment() {
    assert!(ResourcePresenceRequirement::Required.accepts(true));
    assert!(!ResourcePresenceRequirement::Required.accepts(false));
    assert!(ResourcePresenceRequirement::Optional.accepts(true));
    assert!(ResourcePresenceRequirement::Optional.accepts(false));
    assert!(ResourcePresenceRequirement::Forbidden.accepts(false));
    assert!(!ResourcePresenceRequirement::Forbidden.accepts(true));

    let mut invalid_alignment = operation();
    invalid_alignment.resources.minimum_value_alignment_bytes = 3;
    assert!(invalid_alignment.validate().is_err());
    assert!(OperationProviderDescriptor::new(
        id("provider.invalid-estimator"),
        operation().id,
        operation().fingerprint().unwrap(),
        sha('f'),
        ContractVersion::new(1, 0),
        id("device.reference.0"),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation()),
        "resource estimator with spaces",
        ContractVersion::new(1, 0),
        sha('e'),
    )
    .is_err());
    assert!(EngineProviderDescriptor::new(
        id("provider.engine.invalid"),
        ContractVersion::new(1, 0),
        "not-a-sha256",
        id("device.reference.0"),
        BTreeSet::new(),
    )
    .is_err());
    assert!(OperationProviderDescriptor::new(
        id("provider.invalid-implementation"),
        operation().id,
        operation().fingerprint().unwrap(),
        "not-a-sha256",
        ContractVersion::new(1, 0),
        id("device.reference.0"),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation()),
        "resource-estimator.reference",
        ContractVersion::new(1, 0),
        sha('e'),
    )
    .is_err());
}

#[test]
fn execution_memory_is_core_owned_and_exact() {
    let fixture = plan_fixture(0);
    let memory = fixture.plan.payload().memory();
    assert_eq!(memory.device_capacity_bytes(), 1 << 20);
    assert_eq!(memory.policy_capacity_bytes(), 4096);
    assert_eq!(memory.reserve_bytes(), 128);
    assert_eq!(memory.usable_capacity_bytes(), 3968);
    assert_eq!(memory.maximum_active_sequences(), 3);
    assert_eq!(memory.static_bytes(), 48);
    assert_eq!(memory.minimum_runnable_request_bytes(), 112);
    assert_eq!(memory.theoretical_ceiling_bytes(), 384);
    assert_eq!(memory.static_allocations().len(), 2);
    assert_eq!(memory.dynamic_descriptors().len(), 4);
    assert!(!memory.dynamic_pools().is_empty());
    assert!(memory
        .dynamic_pools()
        .windows(2)
        .all(|pair| pair[0].pool_id() < pair[1].pool_id()));

    let pooled_resource_ids = memory
        .dynamic_pools()
        .iter()
        .flat_map(|pool| pool.resource_ids().iter().cloned())
        .collect::<BTreeSet<_>>();
    let dynamic_resource_ids = memory
        .dynamic_descriptors()
        .iter()
        .map(|descriptor| descriptor.base_resource_id().clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(pooled_resource_ids, dynamic_resource_ids);
    for pool in memory.dynamic_pools() {
        assert_eq!(
            pool.provisioning().mode(),
            DynamicPoolProvisioningMode::DemandDrivenElastic
        );
        assert!(
            pool.provisioning().minimum_resident_bytes()
                <= pool.provisioning().maximum_resident_bytes()
        );
        assert!(
            pool.provisioning().maximum_resident_bytes() as u128
                <= pool.theoretical_ceiling_bytes()
        );
    }
    for descriptor in memory.dynamic_descriptors() {
        let pool = memory
            .dynamic_pools()
            .iter()
            .find(|pool| pool.pool_id() == descriptor.pool_id())
            .unwrap();
        assert!(pool.resource_ids().contains(descriptor.base_resource_id()));
        assert_eq!(
            pool.compatibility().profile(),
            descriptor.storage().profile()
        );
    }

    let scratch = memory
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
        .collect::<Vec<_>>();
    let persistent = memory
        .static_allocations()
        .iter()
        .filter(|allocation| matches!(allocation.kind(), AllocationKind::Persistent { .. }))
        .collect::<Vec<_>>();
    assert_eq!(scratch.len(), 1);
    assert!(scratch.iter().all(|descriptor| {
        descriptor.minimum_request_bytes().unwrap() == 64
            && descriptor.alignment_bytes() == 16
            && descriptor.usage() == BufferUsage::Scratch
            && descriptor.lifetime() == AllocationLifetime::Invocation
            && descriptor.theoretical_maximum_instances() == 3
    }));
    assert_eq!(persistent.len(), 1);
    assert!(persistent.iter().all(|allocation| {
        allocation.size_bytes() == 32
            && allocation.usage() == BufferUsage::Persistent
            && allocation.lifetime() == AllocationLifetime::Plan
            && allocation.storage().profile() == contiguous_storage_profile()
    }));
    let state = memory
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id().as_str() == "resource.state.0")
        .unwrap();
    assert_eq!(state.minimum_request_bytes().unwrap(), 16);
    assert_eq!(state.theoretical_maximum_instances(), 3);
    assert_eq!(state.lifetime(), AllocationLifetime::Sequence);
    assert_eq!(memory.static_buffer_requests().unwrap().len(), 2);

    let small_policy = policy(511);
    let small_plan = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &small_policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(
        small_plan.payload().memory().theoretical_ceiling_bytes()
            > u128::from(small_plan.payload().memory().usable_capacity_bytes())
    );
}

#[test]
fn reusable_execution_workspace_is_core_derived_plan_data() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let reusable_execution = ReusableExecutionPolicy::new(
        2,
        vec![
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("execution.test-token").unwrap(),
                ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
            )
            .unwrap(),
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new("execution.test-token").unwrap(),
                ReusableExecutionCapacity::new(1, 64, 1).unwrap(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let policy = ResolvedRuntimePolicy::new(
        "runtime-policy.reusable-test",
        ContractVersion::new(2, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 4096,
            reserve_bytes: 128,
            maximum_active_sequences: 3,
            dynamic_storage_profile_order: vec![contiguous_storage_profile()],
        },
        serde_json::from_value(json!({
            "maximum_queue_depth": 8,
            "maximum_scheduled_tokens": 4096,
            "sequence_fit_policy": "immediate_only",
            "allow_defer": true,
            "cancellation_check_interval_steps": 1
        }))
        .unwrap(),
        Some(reusable_execution),
    )
    .unwrap();
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let node_resolutions = vec![node_resolution(&family, &catalog, &policy, 0, &planning)];
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, node_resolutions.clone()).unwrap(),
    )
    .unwrap();
    let memory = plan.payload().memory();
    let reusable = memory
        .reusable_execution()
        .expect("reusable execution memory plan");

    assert_eq!(reusable.maximum_reusable_lanes(), 2);
    assert_eq!(reusable.buckets().len(), 2);
    assert_eq!(
        reusable.maximum_device_executables(),
        u64::try_from(plan.payload().nodes().len()).unwrap() * 2 * 2
    );
    assert!(reusable
        .buckets()
        .iter()
        .all(|bucket| !bucket.pool_budgets().is_empty()));

    let reusable_bytes = memory
        .dynamic_pools()
        .iter()
        .map(|pool| u128::from(pool.reusable_workspace_ceiling_bytes()))
        .sum::<u128>();
    let live_dynamic_bytes = memory
        .dynamic_pools()
        .iter()
        .map(DynamicBackingPoolSpec::theoretical_ceiling_bytes)
        .sum::<u128>();
    assert!(reusable_bytes > 0);
    assert_eq!(
        memory.theoretical_ceiling_bytes(),
        u128::from(memory.static_bytes()) + live_dynamic_bytes + reusable_bytes
    );
    for pool in memory.dynamic_pools() {
        assert!(
            u128::from(pool.provisioning().maximum_resident_bytes())
                <= pool.theoretical_ceiling_bytes()
                    + u128::from(pool.reusable_workspace_ceiling_bytes())
        );
    }

    let mut tampered = serde_json::to_value(&plan).unwrap();
    tampered["payload"]["memory"]["reusable_execution"]["maximum_device_executables"] =
        json!(reusable.maximum_device_executables() + 1);
    rehash_plan_json(&mut tampered);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&tampered).unwrap(),
        &family,
        &catalog,
        &policy,
        node_resolutions,
    )
    .is_err());
}

#[test]
fn minimum_runnable_sums_lifetime_minima_and_sequential_invocation_peak() {
    let registration = TypedFamilyRegistration::new(SequentialScratchFamily);
    let family = registration.prepare(&json!({"width": 4})).unwrap();
    let catalog = catalog();
    let policy = policy(4096);
    let descriptor = catalog.providers_for(&id("operation.main")).unwrap()[0].clone();
    let runtime_registry = OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: operation(),
            calls: Arc::new(AtomicUsize::new(0)),
            reject_signature: false,
        }) as Box<dyn OperationContract>],
        vec![Box::new(SequentialScratchEstimator { descriptor })
            as Box<dyn OperationProvider<PlanningTestRuntime>>],
    )
    .unwrap();
    let planning = runtime_registry.planning();
    let first = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        id("node.first"),
        sequential_resolved_values(
            "value.input",
            "resource.sequential.input",
            "value.intermediate",
            "resource.sequential.intermediate",
        ),
        BTreeSet::new(),
        None,
    )
    .unwrap();
    let second = PlanNodeResolution::resolve(
        &family,
        &catalog,
        &policy,
        &planning,
        id("node.second"),
        sequential_resolved_values(
            "value.intermediate",
            "resource.sequential.intermediate",
            "value.output",
            "resource.sequential.output",
        ),
        BTreeSet::new(),
        None,
    )
    .unwrap();
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![first, second]).unwrap(),
    )
    .unwrap();
    let memory = plan.payload().memory();
    let scratch_sum = memory
        .dynamic_descriptors()
        .iter()
        .filter(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
        .map(|descriptor| descriptor.minimum_request_bytes().unwrap())
        .sum::<u64>();
    let intermediate = memory
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| {
            descriptor.base_resource_id().as_str() == "resource.sequential.intermediate"
        })
        .unwrap();
    assert_eq!(scratch_sum, 160);
    assert_eq!(intermediate.lifetime(), AllocationLifetime::Step);
    assert_eq!(intermediate.minimum_request_bytes().unwrap(), 16);
    assert_eq!(
        intermediate.theoretical_maximum_request_bytes().unwrap(),
        16
    );
    assert_eq!(memory.minimum_request_bytes(), 32);
    assert_eq!(memory.minimum_sequence_bytes(), 16);
    assert_eq!(memory.minimum_step_bytes(), 16);
    assert_eq!(memory.minimum_invocation_peak_bytes(), 96);
    assert_eq!(memory.minimum_runnable_request_bytes(), 160);
}

#[test]
fn provider_formula_is_policy_invariant_and_core_binds_token_ceiling() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let mut provider_evidence = None;
    let mut bounded_maxima = Vec::new();

    for maximum_scheduled_tokens in [32, 4096] {
        let policy = policy_with_tokens(1 << 20, 128, 3, maximum_scheduled_tokens).unwrap();
        let planning =
            TestPlanningRegistry::new(&catalog, 7, 32, EstimateBehavior::TokenScaledScratch);
        let resolution = node_resolution(&family, &catalog, &policy, 0, &planning);
        let resources = &resolution.provider_resource_candidates()[0];
        let evidence = (
            resources.estimator_input_fingerprint().to_owned(),
            resources.estimate_fingerprint().to_owned(),
            resources.scratch().unwrap().size_formula().clone(),
        );
        assert_eq!(
            provider_evidence.get_or_insert_with(|| evidence.clone()),
            &evidence
        );

        let plan = ExecutionPlan::build(
            PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            plan.payload().maximum_scheduled_tokens(),
            maximum_scheduled_tokens
        );
        let scratch = plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .iter()
            .find(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
            .unwrap();
        bounded_maxima.push(scratch.theoretical_maximum_request_bytes().unwrap());
    }

    assert_eq!(bounded_maxima, vec![7 * 32, 7 * 4096]);
}

#[test]
fn runtime_capacity_reserve_and_concurrency_are_typed_planning_inputs() {
    assert!(policy_with(4096, 4096, 3).is_err());
    assert!(policy_with(4096, 128, 0).is_err());

    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let oversized = policy_with((1 << 20) + 1, 128, 3).unwrap();
    let oversized_resolution = vec![node_resolution(&family, &catalog, &oversized, 0, &planning)];
    assert!(ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &oversized, oversized_resolution).unwrap(),
    )
    .is_err());

    let four_sequences = policy_with(4096, 256, 4).unwrap();
    let resolution = vec![node_resolution(
        &family,
        &catalog,
        &four_sequences,
        0,
        &planning,
    )];
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &four_sequences, resolution).unwrap(),
    )
    .unwrap();
    let memory = plan.payload().memory();
    assert_eq!(memory.maximum_active_sequences(), 4);
    assert_eq!(memory.usable_capacity_bytes(), 3840);
    assert_eq!(memory.theoretical_ceiling_bytes(), 496);
    assert!(plan.payload().nodes()[0].scratch_resource().is_some());
    let state = memory
        .dynamic_descriptors()
        .iter()
        .find(|descriptor| descriptor.base_resource_id().as_str() == "resource.state.0")
        .unwrap();
    assert_eq!(state.theoretical_maximum_instances(), 4);
}

#[test]
fn maximum_active_sequence_ceiling_is_nonzero_and_o_graph() {
    assert!(policy_with(1 << 20, 128, 0).is_err());
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog();
    let rejected_planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let malicious = AdversarialRuntimePolicy {
        maximum_active_sequences: 0,
        maximum_scheduled_tokens: 4096,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    let planning = rejected_planning.planning();
    assert!(PlanNodeResolution::resolve(
        &family,
        &catalog,
        &malicious,
        &planning,
        id("node.main"),
        resolved_values(0),
        BTreeSet::new(),
        None,
    )
    .is_err());

    let zero_tokens = AdversarialRuntimePolicy {
        maximum_active_sequences: 1,
        maximum_scheduled_tokens: 0,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    assert!(PlanNodeResolution::resolve(
        &family,
        &catalog,
        &zero_tokens,
        &planning,
        id("node.main"),
        resolved_values(0),
        BTreeSet::new(),
        None,
    )
    .is_err());
    assert_eq!(rejected_planning.estimator_calls.load(Ordering::SeqCst), 0);
    assert_eq!(rejected_planning.contract_calls.load(Ordering::SeqCst), 0);

    let mut expected_rows = None;
    let mut expected_provider_formula = None;
    for maximum_active_sequences in [1, 32, u32::MAX] {
        let policy = policy_with(1 << 20, 128, maximum_active_sequences).unwrap();
        let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
        let resolution = node_resolution(&family, &catalog, &policy, 0, &planning);
        let plan = ExecutionPlan::build(
            PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution.clone()]).unwrap(),
        )
        .unwrap();
        let rows = (
            plan.payload().nodes().len(),
            plan.payload().nodes()[0].resources().len(),
            plan.payload().memory().static_allocations().len(),
            plan.payload().memory().dynamic_descriptors().len(),
        );
        assert_eq!(*expected_rows.get_or_insert(rows), rows);
        let resources = &resolution.provider_resource_candidates()[0];
        let formula_evidence = (
            resources.estimator_input_fingerprint().to_owned(),
            resources.estimate_fingerprint().to_owned(),
            resources.scratch().unwrap().size_formula().clone(),
            resources.scratch().unwrap().minimum_bytes().unwrap(),
            plan.payload()
                .memory()
                .dynamic_descriptors()
                .iter()
                .find(|descriptor| matches!(descriptor.kind(), AllocationKind::Scratch { .. }))
                .unwrap()
                .demand()
                .clone(),
        );
        assert_eq!(
            expected_provider_formula.get_or_insert_with(|| formula_evidence.clone()),
            &formula_evidence
        );
        assert_eq!(
            plan.payload().nodes()[0]
                .scratch_resource()
                .into_iter()
                .count(),
            1
        );
        if maximum_active_sequences == u32::MAX {
            let policy_wire = serde_json::to_vec(&policy).unwrap();
            let restored: ResolvedRuntimePolicy = serde_json::from_slice(&policy_wire).unwrap();
            assert_eq!(restored.memory().maximum_active_sequences, u32::MAX);
            let restored_plan = ExecutionPlan::from_json_validated(
                &plan.to_json().unwrap(),
                &family,
                &catalog,
                &policy,
                vec![resolution],
            )
            .unwrap();
            assert_eq!(restored_plan.plan_hash(), plan.plan_hash());
        }
    }
}

#[test]
fn theoretical_ceiling_over_u64_is_canonical_evidence_not_capacity_policy() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog_with_memory(u64::MAX);
    let policy = policy_with(u64::MAX, 1, u32::MAX).unwrap();
    let planning = TestPlanningRegistry::new(
        &catalog,
        8 * 1024 * 1024 * 1024,
        32,
        EstimateBehavior::Correct,
    );
    let resolution = node_resolution(&family, &catalog, &policy, 0, &planning);
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution.clone()]).unwrap(),
    )
    .unwrap();
    assert!(plan.payload().memory().theoretical_ceiling_bytes() > u128::from(u64::MAX));
    assert!(
        u128::from(plan.payload().memory().static_bytes())
            + u128::from(plan.payload().memory().minimum_runnable_request_bytes())
            <= u128::from(plan.payload().memory().usable_capacity_bytes())
    );
    let wire = plan.to_json().unwrap();
    let value: Value = serde_json::from_slice(&wire).unwrap();
    let encoded = value["payload"]["memory"]["theoretical_ceiling_bytes"]
        .as_str()
        .unwrap();
    assert_eq!(
        encoded,
        plan.payload()
            .memory()
            .theoretical_ceiling_bytes()
            .to_string()
    );
    let restored =
        ExecutionPlan::from_json_validated(&wire, &family, &catalog, &policy, vec![resolution])
            .unwrap();
    assert_eq!(restored.plan_hash(), plan.plan_hash());
}

#[test]
fn state_capacity_demand_is_explicit_checked_and_wire_closed() {
    let state = StateSpec {
        id: id("state.scaled"),
        value_id: id("value.scaled"),
        tensor: ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::U8,
            layout: ResolvedTensorLayout::Contiguous,
        },
        lifetime: StateLifetime::Sequence,
        capacity_demand: StateCapacityDemand::TokenScaled {
            bytes_per_token: 4,
            maximum_tokens: 128,
        },
        initialization: StateInitialization::Zero,
    };
    let restored: StateSpec =
        serde_json::from_value(serde_json::to_value(&state).unwrap()).unwrap();
    assert_eq!(restored, state);
    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "token_scaled": {"bytes_per_token": 0, "maximum_tokens": 128}
    }))
    .is_err());
    let mut undersized = serde_json::to_value(&state).unwrap();
    undersized["capacity_demand"]["token_scaled"]["bytes_per_token"] = json!(1);
    assert!(serde_json::from_value::<StateSpec>(undersized).is_err());
    assert_eq!(state.capacity_demand.theoretical_bytes(4).unwrap(), 512);
    assert!(serde_json::from_value::<StateCapacityDemand>(json!("plan_static")).is_err());
    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "fixed_per_scope": {"unexpected": true}
    }))
    .is_err());
    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "page_scaled": {"bytes_per_page": 4096, "maximum_pages": 1024}
    }))
    .is_err());
}

#[test]
fn provider_workspace_formulas_are_actual_shape_checked_and_wire_closed() {
    let shape = resource_work(&[3, 3, 3, 3]);
    assert_eq!(
        DynamicResourceDemand::fixed(13)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        13
    );
    assert_eq!(
        DynamicResourceDemand::actual_sequences(7, 8)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        28
    );
    assert_eq!(
        DynamicResourceDemand::tokens(3, 32)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        36
    );
    assert_eq!(
        DynamicResourceDemand::affine(5, 7, 8, 3, 32)
            .unwrap()
            .evaluate_bytes(&shape)
            .unwrap(),
        69
    );
    assert_eq!(
        DynamicResourceDemand::pages(11, 8)
            .unwrap()
            .evaluate_bytes(&shape)
            .is_err(),
        true
    );

    let buckets = DynamicResourceDemand::bounded_shape_buckets(vec![
        DynamicResourceShapeBucket::new(1, 8, 2, 64).unwrap(),
        DynamicResourceShapeBucket::new(4, 32, 8, 128).unwrap(),
    ])
    .unwrap();
    assert_eq!(buckets.evaluate_bytes(&shape).unwrap(), 128);
    assert!(buckets
        .evaluate_bytes(&resource_work(&[3, 3, 2, 2, 2]))
        .is_err());

    let aligned = ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::actual_sequences(7).unwrap(),
        16,
        ProviderWorkspaceScope::Invocation,
        contiguous_storage_requirement(),
    )
    .unwrap();
    assert_eq!(aligned.evaluate_bytes(&shape).unwrap(), 32);
    let affine = ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::affine(5, 7, 3).unwrap(),
        16,
        ProviderWorkspaceScope::Invocation,
        contiguous_storage_requirement(),
    )
    .unwrap();
    assert_eq!(affine.evaluate_bytes(&shape).unwrap(), 80);
    assert!(ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::tokens(4).unwrap(),
        16,
        ProviderWorkspaceScope::Plan,
        contiguous_storage_requirement(),
    )
    .is_err());
    assert!(ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::actual_sequences(4).unwrap(),
        16,
        ProviderWorkspaceScope::Sequence,
        contiguous_storage_requirement(),
    )
    .is_err());
    assert!(ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::fixed(u64::MAX).unwrap(),
        16,
        ProviderWorkspaceScope::Invocation,
        contiguous_storage_requirement(),
    )
    .is_err());

    assert!(DynamicResourceDemand::tokens(u64::MAX, 2).is_err());
    assert!(DynamicResourceDemand::actual_sequences(u64::MAX, 2).is_err());
    assert!(DynamicResourceDemand::affine(1, u64::MAX, 2, 1, 2).is_err());
    assert!(ProviderWorkspaceSizeFormula::affine(1, 0, 0).is_err());
    assert!(ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::affine(1, 1, 1).unwrap(),
        16,
        ProviderWorkspaceScope::Sequence,
        contiguous_storage_requirement(),
    )
    .is_err());
    assert!(
        serde_json::from_value::<ProviderWorkspaceSizeFormula>(json!({
            "actual_sequences": {
                "bytes_per_sequence": u64::MAX,
                "maximum_sequences": 2
            }
        }))
        .is_err()
    );
    assert!(
        serde_json::from_value::<ProviderWorkspaceSizeFormula>(json!({
            "tokens": {"bytes_per_token": 4, "maximum_tokens": 32, "unknown": 1}
        }))
        .is_err()
    );
    assert!(DynamicResourceDemand::bounded_shape_buckets(vec![
        DynamicResourceShapeBucket::new(4, 32, 8, 128).unwrap(),
        DynamicResourceShapeBucket::new(2, 64, 16, 256).unwrap(),
    ])
    .is_err());
    assert!(DynamicResourceDemand::bounded_shape_buckets(vec![
        DynamicResourceShapeBucket::new(1, 8, 2, 128).unwrap(),
        DynamicResourceShapeBucket::new(4, 32, 8, 64).unwrap(),
    ])
    .is_err());
    assert!(DynamicResourceDemand::bounded_shape_buckets(
        (1..=MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS + 1)
            .map(|index| {
                DynamicResourceShapeBucket::new(
                    index as u32,
                    index as u64,
                    index as u64,
                    index as u64,
                )
                .unwrap()
            })
            .collect(),
    )
    .is_err());

    let wire = serde_json::to_value(&aligned).unwrap();
    let restored: ProviderWorkspaceRequirement = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(restored, aligned);
    let mut unknown = wire;
    unknown["unknown"] = json!(1);
    assert!(serde_json::from_value::<ProviderWorkspaceRequirement>(unknown).is_err());
    assert!(
        serde_json::from_value::<ProviderWorkspaceSizeFormula>(json!({
            "bounded_shape_buckets": {"buckets": [
                {"maximum_sequences": 4, "maximum_tokens": 32, "maximum_pages": 8, "bytes": 128},
                {"maximum_sequences": 2, "maximum_tokens": 64, "maximum_pages": 16, "bytes": 256}
            ]}
        }))
        .is_err()
    );
}
