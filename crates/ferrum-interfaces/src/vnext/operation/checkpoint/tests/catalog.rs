use super::*;
use crate::vnext::{
    AliasPolicy, AttributeSchema, CapabilityCatalog, DeviceClass, DeviceDescriptor,
    DimensionConstraint, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView,
    ElementType, EngineProviderDescriptor, LayoutConstraint, OperationDescriptor, OracleSpec,
    ProfilePhase, ProviderRequirement, ResourcePresenceRequirement, ResourceRequirements,
    TensorAccess, TensorContract,
};
use std::collections::BTreeMap;

fn catalog(capability: ProviderCheckpointCapability) -> CapabilityCatalog {
    let tensor = |access| {
        TensorContract::new(
            vec![DimensionConstraint::Exact(4)],
            BTreeSet::from([ElementType::F32]),
            vec![LayoutConstraint::Contiguous],
            access,
            AliasPolicy::NoAlias,
        )
        .unwrap()
    };
    let operation = OperationDescriptor {
        id: OperationId::new("operation.fixture").unwrap(),
        version: ContractVersion::new(1, 0),
        inputs: vec![tensor(TensorAccess::Read)],
        outputs: vec![tensor(TensorAccess::Write)],
        attributes: AttributeSchema::empty(),
        resources: ResourceRequirements {
            minimum_value_alignment_bytes: 4,
            scratch: ResourcePresenceRequirement::Forbidden,
            binding: ResourcePresenceRequirement::Forbidden,
            persistent: ResourcePresenceRequirement::Forbidden,
        },
        oracle: OracleSpec::Exact,
        provider: ProviderRequirement {
            minimum_version: ContractVersion::new(1, 0),
            required_capabilities: BTreeSet::new(),
        },
        profile_phase: ProfilePhase::Forward,
    };
    let provider = provider_with_fingerprint(&operation.fingerprint().unwrap())
        .with_checkpoint_capability(capability);
    let device_id = provider.device_id().clone();
    CapabilityCatalog::new(
        DeviceDescriptor {
            id: device_id.clone(),
            class: DeviceClass::Reference,
            ordinal: 0,
            total_memory_bytes: 1024,
            runtime_implementation_fingerprint: "d".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        },
        vec![operation.clone()],
        BTreeMap::from([(operation.id, vec![provider])]),
        vec![EngineProviderDescriptor::new(
            ProviderId::new("provider.engine.fixture").unwrap(),
            ContractVersion::new(1, 0),
            "e".repeat(64),
            device_id,
            BTreeSet::new(),
        )
        .unwrap()],
    )
    .unwrap()
}

#[test]
fn catalog_identity_includes_explicit_checkpoint_semantics_and_numerics() {
    let unsupported = catalog(ProviderCheckpointCapability::Unsupported);
    let wire = serde_json::to_vec(&unsupported).unwrap();
    let round_trip: CapabilityCatalog = serde_json::from_slice(&wire).unwrap();
    assert_eq!(
        unsupported.fingerprint().unwrap(),
        round_trip.fingerprint().unwrap()
    );
    assert_eq!(wire, serde_json::to_vec(&round_trip).unwrap());

    let bitwise = catalog(supported(CheckpointPartitionNumerics::BitwiseEquivalent));
    let oracle = catalog(supported(CheckpointPartitionNumerics::OperationOracle));
    let same_partition = catalog(supported(CheckpointPartitionNumerics::SamePartitionOnly));
    let captured = catalog(supported(
        CheckpointPartitionNumerics::CapturedExecutionContinuation,
    ));
    let conditioned = catalog(ProviderCheckpointCapability::CompletedBoundary(
        ProviderCheckpointContract::new(
            CheckpointInputDependency::EntireTokenInput,
            CheckpointBoundaryConstraint::any_positive(),
            CheckpointPartitionNumerics::BitwiseEquivalent,
        ),
    ));
    let aligned = catalog(ProviderCheckpointCapability::CompletedBoundary(
        ProviderCheckpointContract::new(
            CheckpointInputDependency::ExactTokenPrefix,
            CheckpointBoundaryConstraint::new(
                CheckpointTokenSpanConstraint::new(nonzero(4), nonzero(4)).unwrap(),
                CheckpointTokenSpanConstraint::any_positive(),
            )
            .unwrap(),
            CheckpointPartitionNumerics::BitwiseEquivalent,
        ),
    ));
    assert_ne!(
        unsupported.fingerprint().unwrap(),
        bitwise.fingerprint().unwrap()
    );
    assert_ne!(
        bitwise.fingerprint().unwrap(),
        oracle.fingerprint().unwrap()
    );
    assert_ne!(
        same_partition.fingerprint().unwrap(),
        bitwise.fingerprint().unwrap()
    );
    for other in [&unsupported, &bitwise, &oracle, &same_partition] {
        assert_ne!(
            captured.fingerprint().unwrap(),
            other.fingerprint().unwrap()
        );
    }
    assert_ne!(
        conditioned.fingerprint().unwrap(),
        bitwise.fingerprint().unwrap()
    );
    assert_ne!(
        aligned.fingerprint().unwrap(),
        bitwise.fingerprint().unwrap()
    );
    let round_trip: CapabilityCatalog =
        serde_json::from_slice(&serde_json::to_vec(&bitwise).unwrap()).unwrap();
    assert_eq!(bitwise, round_trip);
}
