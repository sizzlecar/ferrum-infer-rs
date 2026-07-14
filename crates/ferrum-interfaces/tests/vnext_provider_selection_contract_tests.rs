mod vnext_core_contract;

use vnext_core_contract::*;

#[test]
fn provider_implementation_fingerprint_is_plan_hashed_and_revalidated() {
    let fixture = plan_fixture(0);
    let original_hash = fixture.plan.plan_hash().as_str().to_owned();
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    assert_eq!(
        value["payload"]["nodes"][0]["provider_implementation_fingerprint"],
        json!(sha('f'))
    );
    value["payload"]["nodes"][0]["provider_implementation_fingerprint"] = json!(sha('0'));
    rehash_plan_json(&mut value);
    assert_ne!(value["plan_hash"], json!(original_hash));
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
fn planning_registry_missing_duplicate_and_mismatched_entries_fail_before_plan() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);

    let mut missing_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let missing_contract_calls = missing_contract.contract_calls.clone();
    let missing_estimator_calls = missing_contract.estimator_calls.clone();
    missing_contract.contracts.clear();
    assert!(missing_contract.build().is_err());
    assert_eq!(missing_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(missing_estimator_calls.load(Ordering::SeqCst), 0);

    let mut duplicate_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let duplicate_contract_calls = duplicate_contract.contract_calls.clone();
    let duplicate_estimator_calls = duplicate_contract.estimator_calls.clone();
    duplicate_contract.contracts.push(TestOperationContract {
        descriptor: duplicate_contract.contracts[0].descriptor.clone(),
        calls: duplicate_contract_calls.clone(),
        reject_signature: false,
    });
    assert!(duplicate_contract.build().is_err());
    assert_eq!(duplicate_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(duplicate_estimator_calls.load(Ordering::SeqCst), 0);

    let mut mismatched_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let mismatched_contract_calls = mismatched_contract.contract_calls.clone();
    let mismatched_estimator_calls = mismatched_contract.estimator_calls.clone();
    mismatched_contract.contracts[0].descriptor.version = ContractVersion::new(1, 1);
    assert!(mismatched_contract.build().is_err());
    assert_eq!(mismatched_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(mismatched_estimator_calls.load(Ordering::SeqCst), 0);

    let mut rejecting_contract =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    rejecting_contract.contracts[0].reject_signature = true;
    let rejecting_contract = rejecting_contract.build().unwrap();
    assert!(try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &rejecting_contract,
        None,
    )
    .is_err());
    assert_eq!(rejecting_contract.contract_calls.load(Ordering::SeqCst), 1);
    assert_eq!(rejecting_contract.estimator_calls.load(Ordering::SeqCst), 0);

    let mut missing_estimator =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let missing_contract_calls = missing_estimator.contract_calls.clone();
    let missing_estimator_calls = missing_estimator.estimator_calls.clone();
    missing_estimator.estimators.clear();
    assert!(missing_estimator.build().is_err());
    assert_eq!(missing_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(missing_estimator_calls.load(Ordering::SeqCst), 0);

    let mut duplicate_estimator =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let duplicate_contract_calls = duplicate_estimator.contract_calls.clone();
    let duplicate_estimator_calls = duplicate_estimator.estimator_calls.clone();
    duplicate_estimator.estimators.push(TestEstimator {
        descriptor: duplicate_estimator.estimators[0].descriptor.clone(),
        calls: duplicate_estimator_calls.clone(),
        scratch_bytes: 64,
        persistent_bytes: 32,
        behavior: EstimateBehavior::Correct,
    });
    assert!(duplicate_estimator.build().is_err());
    assert_eq!(duplicate_contract_calls.load(Ordering::SeqCst), 0);
    assert_eq!(duplicate_estimator_calls.load(Ordering::SeqCst), 0);

    let mut mismatched_estimator =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let descriptor = mismatched_estimator.estimators[0].descriptor.clone();
    mismatched_estimator.estimators[0].descriptor = OperationProviderDescriptor::new(
        descriptor.provider_id().clone(),
        descriptor.operation_id().clone(),
        descriptor.operation_fingerprint(),
        descriptor.provider_implementation_fingerprint(),
        descriptor.version(),
        descriptor.device_id().clone(),
        descriptor.capabilities().clone(),
        descriptor.accepted_weight_formats().clone(),
        descriptor.accepted_quantization_formats().clone(),
        descriptor.dynamic_storage_bindings().to_vec(),
        "resource-estimator.mismatch",
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
    )
    .unwrap();
    let mismatched_estimator = mismatched_estimator.build().unwrap();
    assert!(try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &mismatched_estimator,
        None,
    )
    .is_err());
    assert_eq!(
        mismatched_estimator.contract_calls.load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        mismatched_estimator.estimator_calls.load(Ordering::SeqCst),
        0
    );

    let mut mismatched_provider_implementation =
        TestPlanningEntries::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let descriptor = mismatched_provider_implementation.estimators[0]
        .descriptor
        .clone();
    mismatched_provider_implementation.estimators[0].descriptor = OperationProviderDescriptor::new(
        descriptor.provider_id().clone(),
        descriptor.operation_id().clone(),
        descriptor.operation_fingerprint(),
        sha('0'),
        descriptor.version(),
        descriptor.device_id().clone(),
        descriptor.capabilities().clone(),
        descriptor.accepted_weight_formats().clone(),
        descriptor.accepted_quantization_formats().clone(),
        descriptor.dynamic_storage_bindings().to_vec(),
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
    )
    .unwrap();
    let mismatched_provider_implementation = mismatched_provider_implementation.build().unwrap();
    assert!(try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &mismatched_provider_implementation,
        None,
    )
    .is_err());
    assert_eq!(
        mismatched_provider_implementation
            .contract_calls
            .load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        mismatched_provider_implementation
            .estimator_calls
            .load(Ordering::SeqCst),
        0
    );
}

#[test]
fn provider_raw_estimate_identity_input_and_output_are_revalidated_by_core() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    for behavior in [
        EstimateBehavior::WrongEstimatorId,
        EstimateBehavior::WrongEstimatorVersion,
        EstimateBehavior::WrongImplementation,
        EstimateBehavior::WrongInput,
        EstimateBehavior::InvalidAlignment,
        EstimateBehavior::MissingScratch,
    ] {
        let planning = TestPlanningRegistry::new(&catalog, 64, 32, behavior);
        assert!(
            try_node_resolution_with_registry(&family, &catalog, &policy, 0, &planning, None,)
                .is_err()
        );
        assert_eq!(planning.contract_calls.load(Ordering::SeqCst), 1);
        assert_eq!(planning.estimator_calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn preferred_provider_is_only_a_core_validated_preference() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog();
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let resolution = try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some("provider.operation.unregistered"),
    )
    .unwrap();
    assert_eq!(
        resolution.provider_resource_candidates()[0]
            .provider_id()
            .as_str(),
        "provider.operation.reference"
    );
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
    )
    .unwrap();
    let selection = plan.payload().nodes()[0].selection();
    assert_eq!(
        selection.selected_provider().as_str(),
        "provider.operation.reference"
    );
    assert_eq!(
        selection.selection_reason(),
        ProviderSelectionReason::FallbackFromPreferred
    );
}

#[test]
fn storage_incompatible_preference_falls_back_with_canonical_evidence() {
    let family = TestRegistry::new().prepare();
    let catalog = catalog_with_secondary_provider_storage(
        paged_storage_requirement(4096),
        contiguous_storage_requirement(),
    );
    let policy = policy(4096);
    let planning = TestPlanningRegistry::new(&catalog, 64, 32, EstimateBehavior::Correct);
    let preferred: ProviderId = id("provider.operation.reference");
    let resolution = try_node_resolution_with_registry(
        &family,
        &catalog,
        &policy,
        0,
        &planning,
        Some(preferred.as_str()),
    )
    .unwrap();

    assert_eq!(resolution.provider_resource_candidates().len(), 1);
    assert_eq!(
        resolution.provider_resource_candidates()[0]
            .provider_id()
            .as_str(),
        "provider.operation.secondary"
    );
    let rejected_resource_ids = match resolution
        .provider_resolution_rejections()
        .get(&preferred)
        .unwrap()
    {
        PlanProviderRejectReason::StorageIncompatible { resource_ids } => resource_ids.clone(),
        other => panic!("unexpected storage rejection: {other:?}"),
    };
    assert!(!rejected_resource_ids.is_empty());
    assert!(rejected_resource_ids
        .windows(2)
        .all(|pair| pair[0] < pair[1]));

    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &policy, vec![resolution]).unwrap(),
    )
    .unwrap();
    let selection = plan.payload().nodes()[0].selection();
    assert_eq!(
        selection.selected_provider().as_str(),
        "provider.operation.secondary"
    );
    assert_eq!(
        selection.selection_reason(),
        ProviderSelectionReason::FallbackFromPreferred
    );
    let rejection = selection
        .rejected_providers()
        .iter()
        .find(|rejection| rejection.provider_id() == &preferred)
        .unwrap();
    assert_eq!(
        rejection.reasons(),
        &PlanProviderRejectReason::StorageIncompatible {
            resource_ids: rejected_resource_ids,
        }
    );
}
