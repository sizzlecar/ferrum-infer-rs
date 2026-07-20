mod vnext_core_contract;

use vnext_core_contract::*;

#[test]
fn dynamic_descriptor_and_memory_plan_standalone_wire_are_checked() {
    let fixture = plan_fixture(0);
    let mut descriptor = serde_json::to_value(
        fixture
            .plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .first()
            .unwrap(),
    )
    .unwrap();
    descriptor["alignment_bytes"] = json!(3);
    assert!(serde_json::from_value::<DynamicResourceDescriptor>(descriptor).is_err());

    let mut descriptor = serde_json::to_value(
        fixture
            .plan
            .payload()
            .memory()
            .dynamic_descriptors()
            .first()
            .unwrap(),
    )
    .unwrap();
    descriptor["usage"] = json!("weights");
    assert!(serde_json::from_value::<DynamicResourceDescriptor>(descriptor).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    memory["minimum_invocation_peak_bytes"] = json!(u64::MAX);
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    memory["dynamic_descriptors"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    assert!(memory["dynamic_pools"].as_array().unwrap().len() > 1);
    memory["dynamic_pools"].as_array_mut().unwrap().reverse();
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());

    let mut memory = serde_json::to_value(fixture.plan.payload().memory()).unwrap();
    let persistent = memory["static_allocations"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|allocation| allocation["usage"] == json!("persistent"))
        .unwrap();
    persistent["storage"]["profile"] = serde_json::to_value(paged_storage_profile(4096)).unwrap();
    assert!(serde_json::from_value::<MemoryPlan>(memory).is_err());
}

#[test]
fn execution_plan_is_deterministic_100_of_100() {
    for variant in 0..100 {
        let left = plan_fixture(variant).plan;
        let right = plan_fixture(variant).plan;
        assert_eq!(left.plan_hash(), right.plan_hash());
        assert_eq!(left.to_json().unwrap(), right.to_json().unwrap());
    }
    println!("\nVNEXT PLAN DETERMINISM PASS: 100/100");
}

#[test]
fn execution_plan_schema_round_trip_100_of_100() {
    for variant in 0..100 {
        let fixture = plan_fixture(variant);
        let restored = ExecutionPlan::from_json_validated(
            &fixture.plan.to_json().unwrap(),
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap();
        assert_eq!(fixture.plan, restored);
    }
    println!("\nVNEXT PLAN ROUNDTRIP PASS: 100/100");
}

#[test]
fn breaking_schema_versions_are_rejected_100_of_100() {
    for variant in 0..100 {
        let fixture = plan_fixture(variant);
        let mut value = serde_json::to_value(&fixture.plan).unwrap();
        value["payload"]["schema"]["major"] = json!(EXECUTION_PLAN_SCHEMA.major + 1);
        rehash_plan_json(&mut value);
        assert!(ExecutionPlan::from_json_validated(
            &serde_json::to_vec(&value).unwrap(),
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .is_err());
    }
    println!("\nVNEXT BREAKING VERSION REJECT PASS: 100/100");
}

#[test]
fn legacy_schema_is_rejected_before_v5_nested_binding_validation() {
    let fixture = plan_fixture(0);
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    value["payload"]["schema"] = json!({"major": 3, "minor": 0});
    let weight = value["payload"]["nodes"][0]["values"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|binding| binding["usage"] == json!("weights"))
        .unwrap();
    weight.as_object_mut().unwrap().remove("weight");

    let error = ExecutionPlan::decode_untrusted(&serde_json::to_vec(&value).unwrap()).unwrap_err();
    assert!(matches!(
        error,
        VNextError::UnsupportedPlanSchema {
            expected_major: 5,
            expected_minor: 0,
            actual_major: 3,
            actual_minor: 0,
        }
    ));
}

#[test]
fn forged_self_hashed_plan_is_rejected_by_semantic_rebuild() {
    let fixture = plan_fixture(0);
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    value["payload"]["memory"]["static_allocations"][0]["size_bytes"] = json!(1024);
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn resolved_weight_layout_cannot_be_stripped_from_plan_wire() {
    let fixture = plan_fixture(0);
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    let weight = value["payload"]["nodes"][0]["values"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|binding| binding["usage"] == json!("weights"))
        .unwrap();
    weight.as_object_mut().unwrap().remove("weight");
    rehash_plan_json(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn externally_trusted_node_resolution_cannot_be_replaced_by_wire_data() {
    let fixture = plan_fixture(0);
    let different_resolution = vec![node_resolution(
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        1,
        &fixture.planning,
    )];
    assert!(ExecutionPlan::from_json_validated(
        &fixture.plan.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        different_resolution,
    )
    .is_err());
}

#[test]
fn self_consistent_wire_resource_estimate_and_memory_mutation_is_rejected() {
    let fixture = plan_fixture(0);
    let alternate_planning =
        TestPlanningRegistry::new(&fixture.catalog, 96, 48, EstimateBehavior::Correct);
    let alternate_resolution = vec![node_resolution_with_registry(
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        0,
        &alternate_planning,
        None,
    )];
    let alternate = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            alternate_resolution,
        )
        .unwrap(),
    )
    .unwrap();
    assert_ne!(
        alternate.payload().memory().theoretical_ceiling_bytes(),
        fixture.plan.payload().memory().theoretical_ceiling_bytes()
    );
    assert!(ExecutionPlan::from_json_validated(
        &alternate.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn self_consistent_wire_provider_selection_is_rejected() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let catalog = catalog_with_secondary_provider();
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let original_resolution = vec![node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some("provider.operation.reference"),
    )];
    let alternate_resolution = vec![node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some("provider.operation.secondary"),
    )];
    let alternate = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, alternate_resolution).unwrap(),
    )
    .unwrap();
    assert_eq!(
        alternate.payload().nodes()[0]
            .selection()
            .selected_provider()
            .as_str(),
        "provider.operation.secondary"
    );
    assert!(ExecutionPlan::from_json_validated(
        &alternate.to_json().unwrap(),
        &family,
        &catalog,
        &policy,
        original_resolution,
    )
    .is_err());
}

#[test]
fn typed_planning_registry_invokes_real_contract_and_estimator_once() {
    let fixture = plan_fixture(0);
    assert_eq!(fixture.planning.contract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.planning.estimator_calls.load(Ordering::SeqCst), 1);
    let resources = &fixture.node_resolutions[0].provider_resource_candidates()[0];
    assert_eq!(
        resources.provider_id().as_str(),
        "provider.operation.reference"
    );
    assert_eq!(resources.estimator_id(), "resource-estimator.reference");
    let node = &fixture.plan.payload().nodes()[0];
    assert_eq!(node.provider_implementation_fingerprint(), sha('f'));
    assert_ne!(
        node.provider_implementation_fingerprint(),
        resources.estimator_implementation_fingerprint()
    );
    assert_eq!(resources.scratch().unwrap().minimum_bytes().unwrap(), 64);
    assert_eq!(resources.persistent().unwrap().minimum_bytes().unwrap(), 32);

    let _rebuilt = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(fixture.planning.contract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.planning.estimator_calls.load(Ordering::SeqCst), 1);
}
