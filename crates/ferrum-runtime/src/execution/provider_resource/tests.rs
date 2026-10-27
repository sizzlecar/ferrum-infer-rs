use super::*;
use crate::{
    DeviceId, DynamicStorageRequirement, OperationId, ProviderExecutionSemantics,
    ProviderStorageBindingRequirement, ResolvedValueRole, UnvalidatedProviderResourcePlan,
};
use std::collections::BTreeSet;
use std::sync::Arc;

#[test]
fn prepared_implementation_lives_through_compilation_and_cloned_plans_only() {
    let native = Arc::new(());
    let weak = Arc::downgrade(&native);
    let estimated = estimate()
        .with_prepared_implementation_fingerprint("1".repeat(64))
        .unwrap()
        .with_prepared_implementation_retention(native)
        .unwrap();
    let mut node = crate::PlanNode::resource_test_node(crate::NodeId::new("node.native").unwrap());
    node.provider_resources = plan(estimated);
    let cloned = node.clone();
    let wire = serde_json::to_value(node.provider_resources()).unwrap();
    assert!(wire.get("prepared_implementation_retention").is_none());
    let decoded: UnvalidatedProviderResourcePlan = serde_json::from_value(wire).unwrap();
    drop(node);
    assert!(weak.upgrade().is_some());
    drop(cloned);
    assert!(weak.upgrade().is_none());
    // An untrusted wire description never retains process-local native state.
    drop(decoded);
}

#[test]
fn rejected_estimate_releases_prepared_owner_and_identity_cannot_change() {
    let native = Arc::new(());
    let weak = Arc::downgrade(&native);
    let estimated = estimate()
        .with_prepared_implementation_fingerprint("1".repeat(64))
        .unwrap()
        .with_prepared_implementation_retention(native)
        .unwrap();
    assert!(estimated
        .clone()
        .with_prepared_implementation_fingerprint("2".repeat(64))
        .is_err());
    assert!(
        ProviderResourcePlan::from_provider_output(&descriptor(), "wrong-input", estimated)
            .is_err()
    );
    assert!(weak.upgrade().is_none());
    assert!(estimate()
        .with_prepared_implementation_retention(Arc::new(()))
        .is_err());
}

#[test]
fn prepared_retention_preserves_wire_identity_and_semantic_rebuild() {
    let make = || {
        estimate()
            .with_prepared_implementation_fingerprint("1".repeat(64))
            .unwrap()
    };
    let plain = plan(make());
    let retained = plan(
        make()
            .with_prepared_implementation_retention(Arc::new(()))
            .unwrap(),
    );
    let rebuilt = plan(
        make()
            .with_prepared_implementation_retention(Arc::new(()))
            .unwrap(),
    );
    assert_eq!(retained, rebuilt);
    assert_eq!(
        plain.estimate_fingerprint(),
        retained.estimate_fingerprint()
    );
    assert_eq!(
        serde_json::to_value(plain).unwrap(),
        serde_json::to_value(retained).unwrap()
    );
}

fn descriptor() -> OperationProviderDescriptor {
    OperationProviderDescriptor::new(
        ProviderId::new("provider.prepared-test").unwrap(),
        OperationId::new("operation.prepared-test").unwrap(),
        "a".repeat(64),
        "b".repeat(64),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        DeviceId::new("device.prepared-test").unwrap(),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        vec![ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            0,
            DynamicStorageRequirement::contiguous(),
        )],
        "estimator.prepared-test",
        ContractVersion::new(1, 0),
        "c".repeat(64),
    )
    .unwrap()
}

fn estimate() -> OperationResourceEstimate {
    OperationResourceEstimate::new(
        "estimator.prepared-test",
        ContractVersion::new(1, 0),
        "c".repeat(64),
        "d".repeat(64),
        16,
        None,
        None,
    )
}

fn plan(estimate: OperationResourceEstimate) -> ProviderResourcePlan {
    ProviderResourcePlan::from_provider_output(&descriptor(), &"d".repeat(64), estimate).unwrap()
}

#[test]
fn absent_prepared_identity_preserves_legacy_wire_and_estimate_hash() {
    let plan = plan(estimate());
    let wire = serde_json::to_value(&plan).unwrap();
    assert!(wire.get("prepared_implementation_fingerprint").is_none());
    assert!(wire.get("projection_numerics").is_none());
    // Independently reconstruct the pre-extension hash material. Omission,
    // rather than serializing null, preserves existing executable identities.
    let legacy = serde_json::json!({
        "provider_id":"provider.prepared-test", "estimator_id":"estimator.prepared-test",
        "estimator_version":{"major":1,"minor":0},
        "estimator_implementation_fingerprint":"c".repeat(64),
        "estimator_input_fingerprint":"d".repeat(64),
        "value_alignment_bytes":16,"scratch":null,"binding":null,"persistent":null
    });
    assert_eq!(
        plan.estimate_fingerprint(),
        canonical_fingerprint(&legacy, "legacy estimate test").unwrap()
    );
    let unvalidated: UnvalidatedProviderResourcePlan =
        serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(unvalidated.prepared_implementation_fingerprint, None);
    assert_eq!(serde_json::to_value(unvalidated).unwrap(), wire);
}

#[test]
fn actual_prepared_choice_changes_resource_and_node_identity_at_equal_workspace() {
    let a = plan(
        estimate()
            .with_prepared_implementation_fingerprint("1".repeat(64))
            .unwrap(),
    );
    let b = plan(
        estimate()
            .with_prepared_implementation_fingerprint("2".repeat(64))
            .unwrap(),
    );
    assert_eq!(a.scratch(), b.scratch());
    assert_eq!(
        a.estimator_input_fingerprint(),
        b.estimator_input_fingerprint()
    );
    assert_ne!(a.estimate_fingerprint(), b.estimate_fingerprint());
    assert_eq!(
        a.prepared_implementation_fingerprint(),
        Some("1".repeat(64).as_str())
    );
    let wire = serde_json::to_value(&a).unwrap();
    let decoded: UnvalidatedProviderResourcePlan = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(
        decoded.prepared_implementation_fingerprint,
        Some("1".repeat(64))
    );
    assert_eq!(serde_json::to_value(decoded).unwrap(), wire);
    let mut node = crate::PlanNode::resource_test_node(crate::NodeId::new("node.test").unwrap());
    node.provider_resources = a;
    let before = canonical_fingerprint(&node, "prepared node test").unwrap();
    node.provider_resources = b;
    assert_ne!(
        before,
        canonical_fingerprint(&node, "prepared node test").unwrap()
    );
}

#[test]
fn invalid_digest_and_stale_resource_fingerprint_are_rejected() {
    for invalid in [
        String::new(),
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        assert!(estimate()
            .with_prepared_implementation_fingerprint(invalid.clone())
            .is_err());
        let mut p = plan(estimate());
        p.prepared_implementation_fingerprint = Some(invalid);
        assert!(p.validate_fields().is_err());
    }
    let mut p = plan(
        estimate()
            .with_prepared_implementation_fingerprint("1".repeat(64))
            .unwrap(),
    );
    p.prepared_implementation_fingerprint = Some("2".repeat(64));
    assert!(p.validate_shape().is_err());
    p.estimate_fingerprint = p.compute_estimate_fingerprint().unwrap();
    p.validate_shape().unwrap();
}
