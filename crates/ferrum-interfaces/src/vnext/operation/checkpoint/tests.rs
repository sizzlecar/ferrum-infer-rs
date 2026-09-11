use super::*;
use crate::vnext::{
    DeviceId, DynamicStorageRequirement, OperationId, OperationProviderDescriptor,
    ProviderExecutionRepeatability, ProviderExecutionSemantics, ProviderId,
    ProviderReplayEquivalence, ProviderStorageBindingRequirement, ResolvedValueRole,
};
use serde_json::json;
use std::collections::BTreeSet;

fn nonzero(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn provider() -> OperationProviderDescriptor {
    provider_with_fingerprint(&"a".repeat(64))
}

fn provider_with_fingerprint(fingerprint: &str) -> OperationProviderDescriptor {
    OperationProviderDescriptor::new(
        ProviderId::new("provider.fixture").unwrap(),
        OperationId::new("operation.fixture").unwrap(),
        fingerprint,
        "b".repeat(64),
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
        ContractVersion::new(1, 0),
        DeviceId::new("device.fixture").unwrap(),
        BTreeSet::new(),
        BTreeSet::new(),
        BTreeSet::new(),
        [ResolvedValueRole::Input, ResolvedValueRole::Output]
            .into_iter()
            .map(|role| {
                ProviderStorageBindingRequirement::new(
                    role,
                    0,
                    DynamicStorageRequirement::contiguous(),
                )
            })
            .collect(),
        "estimator.fixture",
        ContractVersion::new(1, 0),
        "c".repeat(64),
    )
    .unwrap()
}

fn supported(partition_numerics: CheckpointPartitionNumerics) -> ProviderCheckpointCapability {
    ProviderCheckpointCapability::CompletedBoundary(ProviderCheckpointContract::new(
        CheckpointInputDependency::ExactTokenPrefix,
        CheckpointBoundaryConstraint::any_positive(),
        partition_numerics,
    ))
}

#[test]
fn checkpoint_boundary_checks_prefix_and_nonempty_suffix_independently() {
    let constraint = CheckpointBoundaryConstraint::new(
        CheckpointTokenSpanConstraint::new(nonzero(3), nonzero(4)).unwrap(),
        CheckpointTokenSpanConstraint::new(nonzero(2), nonzero(3)).unwrap(),
    )
    .unwrap();
    assert!(constraint.permits(4, 7));
    assert!(constraint.permits(8, 14));
    assert!(!constraint.permits(4, 6)); // prefix alignment alone is insufficient
    assert!(!constraint.permits(3, 6));
    assert!(!constraint.permits(0, 3));
    assert!(!constraint.permits(4, 4));
    assert!(!constraint.permits(8, 7));
    // A restored position changes the actual first span. Absolute N is not
    // an alignment condition in this contract.
    assert!(constraint.permits_from(3, 7, 10));
    assert!(!constraint.permits_from(3, 8, 11));
    assert!(!constraint.permits_from(7, 7, 10));
    assert!(!constraint.permits_from(8, 7, 10));
    assert!(!constraint.permits_from(3, 7, 7));
    assert!(!constraint.permits_from(3, 7, 6));
    let any = CheckpointBoundaryConstraint::any_positive();
    assert!(any.permits(u64::MAX - 1, u64::MAX));
    assert!(!any.permits(u64::MAX, 1));
}

#[test]
fn checkpoint_constraints_reject_zero_and_unreachable_lengths() {
    for wire in [
        json!({"minimum_tokens": 0, "alignment": 1}),
        json!({"minimum_tokens": 1, "alignment": 0}),
        json!({"minimum_tokens": u64::MAX, "alignment": 2}),
    ] {
        assert!(serde_json::from_value::<CheckpointTokenSpanConstraint>(wire).is_err());
    }
    assert!(CheckpointTokenSpanConstraint::new(nonzero(u64::MAX), nonzero(2)).is_err());
    let maximal = CheckpointTokenSpanConstraint::new(nonzero(u64::MAX), nonzero(1)).unwrap();
    assert!(CheckpointBoundaryConstraint::new(
        maximal,
        CheckpointTokenSpanConstraint::any_positive()
    )
    .is_err());
    assert!(
        serde_json::from_value::<CheckpointBoundaryConstraint>(json!({
            "prefix": maximal,
            "suffix": CheckpointTokenSpanConstraint::any_positive(),
        }))
        .is_err()
    );
    let valid = CheckpointBoundaryConstraint::any_positive();
    assert_eq!(
        serde_json::from_value::<CheckpointBoundaryConstraint>(
            serde_json::to_value(valid).unwrap()
        )
        .unwrap(),
        valid
    );
}

#[test]
fn device_replay_does_not_implicitly_authorize_checkpoint_restore() {
    let provider = provider();
    assert_eq!(
        provider.execution_semantics().replay_equivalence(),
        ProviderReplayEquivalence::BitwiseEagerEquivalent
    );
    assert!(provider.checkpoint_capability().is_unsupported());
    let wire = serde_json::to_value(&provider).unwrap();
    assert!(wire.get("checkpoint").is_none());
    let restored: OperationProviderDescriptor = serde_json::from_value(wire).unwrap();
    assert_eq!(restored, provider);
    assert_eq!(
        serde_json::to_vec(&restored).unwrap(),
        serde_json::to_vec(&provider).unwrap()
    );

    for numerics in [
        CheckpointPartitionNumerics::SamePartitionOnly,
        CheckpointPartitionNumerics::BitwiseEquivalent,
        CheckpointPartitionNumerics::OperationOracle,
    ] {
        let supported = provider
            .clone()
            .with_checkpoint_capability(supported(numerics));
        assert_ne!(
            serde_json::to_vec(&supported).unwrap(),
            serde_json::to_vec(&provider).unwrap()
        );
        assert_eq!(
            supported.execution_semantics().repeatability(),
            ProviderExecutionRepeatability::BitwiseSameRuntime
        );
        let restored: OperationProviderDescriptor =
            serde_json::from_value(serde_json::to_value(&supported).unwrap()).unwrap();
        assert_eq!(restored, supported);
    }
}

#[test]
fn provider_checkpoint_wire_rejects_unknown_versions_and_incomplete_declarations() {
    let provider = provider()
        .with_checkpoint_capability(supported(CheckpointPartitionNumerics::OperationOracle));
    let wire = serde_json::to_value(provider).unwrap();
    for version in [ContractVersion::new(1, 1), ContractVersion::new(2, 0)] {
        let mut changed = wire.clone();
        changed["checkpoint"]["completed_boundary"]["contract_version"] =
            serde_json::to_value(version).unwrap();
        assert!(serde_json::from_value::<OperationProviderDescriptor>(changed).is_err());
    }
    let mut changed = wire.clone();
    changed["checkpoint"]["completed_boundary"]
        .as_object_mut()
        .unwrap()
        .remove("partition_numerics");
    assert!(serde_json::from_value::<OperationProviderDescriptor>(changed).is_err());
    let mut changed = wire.clone();
    changed["checkpoint"]["completed_boundary"]["tolerance"] = json!(1);
    assert!(serde_json::from_value::<OperationProviderDescriptor>(changed).is_err());
    let mut changed = wire;
    changed["checkpoint"] = serde_json::Value::Null;
    assert!(serde_json::from_value::<OperationProviderDescriptor>(changed).is_err());
}

mod catalog;

#[test]
fn checkpoint_state_ports_are_explicit_canonical_abi_declarations() {
    use crate::vnext::{DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView};
    let profile = DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .unwrap();
    let port = ProviderCheckpointStatePort::new(
        ResolvedValueRole::Input,
        2,
        profile,
        ProviderCheckpointStateLayout::TokenMajorPrefix,
    );
    let contract = ProviderCheckpointContract::new(
        CheckpointInputDependency::ExactTokenPrefix,
        CheckpointBoundaryConstraint::any_positive(),
        CheckpointPartitionNumerics::BitwiseEquivalent,
    );
    assert!(serde_json::to_value(&contract)
        .unwrap()
        .get("state_ports")
        .is_none());
    assert!(contract
        .clone()
        .with_state_ports(vec![port.clone(), port.clone()])
        .is_err());
    let declared = contract.with_state_ports(vec![port]).unwrap();
    assert_eq!(
        serde_json::from_value::<ProviderCheckpointContract>(
            serde_json::to_value(&declared).unwrap()
        )
        .unwrap(),
        declared
    );
    let mut bad = serde_json::to_value(declared).unwrap();
    bad["state_ports"][0]["layout"] = json!({"token_major_prefix": {"bytes_per_token": 4}});
    assert!(serde_json::from_value::<ProviderCheckpointContract>(bad).is_err());
}
