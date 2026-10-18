mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

fn close_fixture(fixture: Fixture) {
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}

#[test]
fn live_catalog_and_resolved_plan_form_one_exact_coverage_registry() {
    let fixture =
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding);
    let node = &fixture.resolved.execution_plan().payload().nodes()[0];
    let operation = fixture
        .resolved
        .parts()
        .capabilities
        .operation(node.operation_id())
        .unwrap();
    assert_eq!(node.operation_version(), ContractVersion::new(1, 0));
    assert_eq!(operation.version, ContractVersion::new(1, 1));
    assert!(operation.version.satisfies(node.operation_version()));
    let mut registry =
        ExecutionDeterminismCoverageRegistry::from_catalog(&fixture.resolved.parts().capabilities)
            .unwrap();
    registry
        .try_add_resolved_model_plan("M1", &fixture.resolved)
        .unwrap();

    assert_eq!(registry.models().len(), 1);
    assert_eq!(registry.models()[0].model_key(), "M1");
    assert_eq!(
        registry.models()[0].resolved_plan_fingerprint(),
        fixture.resolved.fingerprint()
    );
    assert_eq!(
        registry.models()[0].node_ids(),
        fixture
            .plan
            .payload()
            .nodes()
            .iter()
            .map(|node| node.id().clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(registry.unselected_provider_requirements().count(), 0);
    assert!(registry.provider_requirements().iter().all(|requirement| {
        requirement.required_comparisons()
            == [
                ExecutionDeterminismComparisonKind::EagerEager,
                ExecutionDeterminismComparisonKind::ReplayReplay,
                ExecutionDeterminismComparisonKind::EagerReplay,
            ]
            && requirement.model_selections().len() == 1
    }));

    let encoded = registry.to_json().unwrap();
    assert_eq!(
        ExecutionDeterminismCoverageRegistry::decode_untrusted(&encoded).unwrap(),
        registry
    );
    assert_eq!(registry.fingerprint().unwrap().len(), 64);
    close_fixture(fixture);
}

#[test]
fn live_catalog_and_resolved_plans_form_one_typed_witness_denominator() {
    let fixture =
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding);
    let denominator = ExecutionDeterminismEvidenceDenominator::from_catalog_and_resolved_plans(
        &fixture.resolved.parts().capabilities,
        &[("M1", &fixture.resolved)],
    )
    .unwrap();

    assert_eq!(
        denominator.provider_coverage(),
        ExecutionDeterminismProviderCoverage::AllCatalogProviders
    );
    assert_eq!(denominator.coverage().models().len(), 1);
    assert_eq!(
        denominator.provider_evidence().len(),
        denominator.coverage().provider_requirements().len()
    );
    assert!(denominator.provider_evidence().iter().all(|evidence| {
        evidence.model_key() == "M1"
            && !evidence.node_ids().is_empty()
            && !evidence.witness_plan().witnesses().is_empty()
            && evidence.witness_plan_fingerprint().len() == 64
            && evidence.required_comparisons()
                == [
                    ExecutionDeterminismComparisonKind::EagerEager,
                    ExecutionDeterminismComparisonKind::ReplayReplay,
                    ExecutionDeterminismComparisonKind::EagerReplay,
                ]
    }));

    let encoded = denominator.to_json().unwrap();
    assert_eq!(
        ExecutionDeterminismEvidenceDenominator::decode_untrusted(&encoded).unwrap(),
        denominator
    );
    assert_eq!(denominator.fingerprint().unwrap().len(), 64);

    let mut missing_witness: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    missing_witness["provider_evidence"][0]["witness_plan"]["witnesses"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(ExecutionDeterminismEvidenceDenominator::decode_untrusted(
        &serde_json::to_vec(&missing_witness).unwrap()
    )
    .is_err());

    let mut stale_provider: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    stale_provider["provider_evidence"][0]["provider_implementation_fingerprint"] =
        serde_json::json!("0".repeat(64));
    assert!(ExecutionDeterminismEvidenceDenominator::decode_untrusted(
        &serde_json::to_vec(&stale_provider).unwrap()
    )
    .is_err());

    close_fixture(fixture);
}

#[test]
fn focused_denominator_allows_unselected_catalog_rows_without_weakening_full_coverage() {
    let fixture =
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding);
    let denominator = ExecutionDeterminismEvidenceDenominator::from_catalog_and_resolved_plans_with_provider_coverage(
        &fixture.resolved.parts().capabilities,
        &[("M1", &fixture.resolved)],
        ExecutionDeterminismProviderCoverage::SelectedPlanProviders,
    )
    .unwrap();
    let mut encoded: serde_json::Value =
        serde_json::from_slice(&denominator.to_json().unwrap()).unwrap();
    let mut unselected = encoded["coverage"]["provider_requirements"][0].clone();
    unselected["operation_id"] = serde_json::json!("operation.zzz-unselected");
    unselected["provider_id"] = serde_json::json!("provider.zzz-unselected");
    unselected["model_selections"] = serde_json::json!([]);
    encoded["coverage"]["provider_requirements"]
        .as_array_mut()
        .unwrap()
        .push(unselected);
    let focused = serde_json::to_vec(&encoded).unwrap();
    assert!(ExecutionDeterminismEvidenceDenominator::decode_untrusted(&focused).is_ok());

    encoded["provider_coverage"] = serde_json::json!("all_catalog_providers");
    assert!(ExecutionDeterminismEvidenceDenominator::decode_untrusted(
        &serde_json::to_vec(&encoded).unwrap()
    )
    .is_err());
    close_fixture(fixture);
}

#[test]
fn coverage_registry_rejects_reused_models_and_foreign_catalogs() {
    let fixture = fixture_with_determinism_provider_behavior(false, ProviderBehavior::Success);
    let mut registry =
        ExecutionDeterminismCoverageRegistry::from_catalog(&fixture.resolved.parts().capabilities)
            .unwrap();
    registry
        .try_add_resolved_model_plan("M1", &fixture.resolved)
        .unwrap();
    assert!(registry
        .try_add_resolved_model_plan("M2", &fixture.resolved)
        .is_err());

    let foreign = fixture_with_determinism_provider_behavior(
        true,
        ProviderBehavior::ProgramBindingFirstNodeEagerBoundary,
    );
    let mut foreign_registry =
        ExecutionDeterminismCoverageRegistry::from_catalog(&foreign.resolved.parts().capabilities)
            .unwrap();
    assert!(foreign_registry
        .try_add_resolved_model_plan("M1", &fixture.resolved)
        .is_err());

    close_fixture(foreign);
    close_fixture(fixture);
}

#[test]
fn untrusted_coverage_registry_rejects_denominator_and_comparison_mutations() {
    let fixture =
        fixture_with_determinism_provider_behavior(false, ProviderBehavior::ProgramBinding);
    let mut registry =
        ExecutionDeterminismCoverageRegistry::from_catalog(&fixture.resolved.parts().capabilities)
            .unwrap();
    registry
        .try_add_resolved_model_plan("M1", &fixture.resolved)
        .unwrap();
    let encoded = registry.to_json().unwrap();

    let mut missing_node: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    missing_node["provider_requirements"][0]["model_selections"][0]["node_ids"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(ExecutionDeterminismCoverageRegistry::decode_untrusted(
        &serde_json::to_vec(&missing_node).unwrap()
    )
    .is_err());

    let mut weakened_comparisons: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    weakened_comparisons["provider_requirements"][0]["required_comparisons"] =
        serde_json::json!(["eager_eager"]);
    assert!(ExecutionDeterminismCoverageRegistry::decode_untrusted(
        &serde_json::to_vec(&weakened_comparisons).unwrap()
    )
    .is_err());

    let mut duplicate_model: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    let duplicate = duplicate_model["models"][0].clone();
    duplicate_model["models"]
        .as_array_mut()
        .unwrap()
        .push(duplicate);
    assert!(ExecutionDeterminismCoverageRegistry::decode_untrusted(
        &serde_json::to_vec(&duplicate_model).unwrap()
    )
    .is_err());

    close_fixture(fixture);
}
