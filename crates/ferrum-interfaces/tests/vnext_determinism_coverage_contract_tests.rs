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
        ProviderBehavior::ProgramBindingIneligible,
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
