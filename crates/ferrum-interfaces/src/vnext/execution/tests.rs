use super::*;
use crate::vnext::{DynamicStorageAllocator, DynamicStorageView};

fn linear_profile() -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        DynamicStorageView::Contiguous,
    )
    .expect("valid linear profile")
}

fn paged_profile() -> DynamicStorageProfile {
    fixed_block_profile(4096)
}

fn fixed_block_profile(block_bytes: u64) -> DynamicStorageProfile {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena { block_bytes },
        DynamicStorageView::PagedRegions { block_bytes },
    )
    .expect("valid paged profile")
}

fn provider_resources(provider: &str) -> ProviderResourcePlan {
    ProviderResourcePlan {
        provider_id: ProviderId::new(provider).expect("valid provider id"),
        estimator_id: "test-estimator".to_owned(),
        estimator_version: ContractVersion::new(1, 0),
        estimator_implementation_fingerprint: "1".repeat(64),
        estimator_input_fingerprint: "2".repeat(64),
        estimate_fingerprint: "3".repeat(64),
        value_alignment_bytes: 16,
        scratch: None,
        persistent: None,
    }
}

fn joint_candidate(
    provider: &str,
    resource: &ResourceId,
    profiles: &[DynamicStorageProfile],
) -> JointProviderCandidate {
    JointProviderCandidate {
        resources: provider_resources(provider),
        allowed_profiles: BTreeMap::from([(resource.clone(), profiles.iter().copied().collect())]),
        is_preferred: provider == "provider/preferred",
    }
}

#[test]
fn joint_storage_solver_falls_back_when_preferred_breaks_shared_intersection() {
    let resource = ResourceId::new("resource/shared-state").expect("valid resource id");
    let linear = linear_profile();
    let paged = paged_profile();
    let candidate_sets = vec![
        vec![
            joint_candidate("provider/preferred", &resource, &[linear]),
            joint_candidate("provider/fallback", &resource, &[paged]),
        ],
        vec![joint_candidate("provider/consumer", &resource, &[paged])],
    ];

    let (chosen, profiles) =
        ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
            .expect("fallback should satisfy the shared resource");

    assert_eq!(chosen[0].provider_id().as_str(), "provider/fallback");
    assert_eq!(chosen[1].provider_id().as_str(), "provider/consumer");
    assert_eq!(profiles.get(&resource), Some(&paged));
}

#[test]
fn joint_storage_solver_fails_closed_without_shared_intersection() {
    let resource = ResourceId::new("resource/shared-state").expect("valid resource id");
    let linear = linear_profile();
    let paged = paged_profile();
    let candidate_sets = vec![
        vec![joint_candidate("provider/linear", &resource, &[linear])],
        vec![joint_candidate("provider/paged", &resource, &[paged])],
    ];

    let error = ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
        .expect_err("incompatible shared storage must be rejected");

    assert!(error
        .to_string()
        .contains("no joint provider/storage assignment"));
}

#[test]
fn joint_storage_profile_order_precedes_provider_id_tie_break() {
    let resource = ResourceId::new("resource/state").expect("valid resource id");
    let linear = linear_profile();
    let paged = paged_profile();
    let candidate_sets = vec![vec![
        joint_candidate("provider/a", &resource, &[paged]),
        joint_candidate("provider/z", &resource, &[linear]),
    ]];

    let (chosen, profiles) =
        ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
            .expect("one candidate satisfies each profile");

    assert_eq!(chosen[0].provider_id().as_str(), "provider/z");
    assert_eq!(profiles.get(&resource), Some(&linear));
}

#[test]
fn joint_storage_solver_decomposes_independent_resource_components() {
    let first = ResourceId::new("resource/first").expect("valid resource id");
    let second = ResourceId::new("resource/second").expect("valid resource id");
    let linear = linear_profile();
    let paged = paged_profile();
    let candidate_sets = vec![
        vec![
            joint_candidate("provider/a", &first, &[paged]),
            joint_candidate("provider/b", &first, &[linear]),
        ],
        vec![
            joint_candidate("provider/c", &second, &[paged]),
            joint_candidate("provider/d", &second, &[linear]),
        ],
    ];

    let components = joint_candidate_components(&candidate_sets);
    assert_eq!(components, vec![vec![0], vec![1]]);
    let (chosen, profiles) =
        ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
            .expect("independent components should solve independently");
    assert_eq!(chosen[0].provider_id().as_str(), "provider/b");
    assert_eq!(chosen[1].provider_id().as_str(), "provider/d");
    assert_eq!(profiles.get(&first), Some(&linear));
    assert_eq!(profiles.get(&second), Some(&linear));
}

#[test]
fn tensor_layout_fingerprint_excludes_shape_but_preserves_interpretation() {
    let first = ResolvedTensorLayout::Blocked {
        block: vec![16, 16],
        axis_order: vec![1, 0],
        padding: BlockedTensorPadding::ZeroFill {
            physical_dimensions: vec![32, 48],
        },
    };
    let second = ResolvedTensorLayout::Blocked {
        block: vec![16, 16],
        axis_order: vec![1, 0],
        padding: BlockedTensorPadding::ZeroFill {
            physical_dimensions: vec![64, 80],
        },
    };
    let changed_layout = ResolvedTensorLayout::Blocked {
        block: vec![32, 16],
        axis_order: vec![1, 0],
        padding: BlockedTensorPadding::ZeroFill {
            physical_dimensions: vec![64, 80],
        },
    };

    assert_eq!(
        tensor_storage_layout_fingerprint(&first).expect("fingerprint"),
        tensor_storage_layout_fingerprint(&second).expect("fingerprint")
    );
    assert_ne!(
        tensor_storage_layout_fingerprint(&first).expect("fingerprint"),
        tensor_storage_layout_fingerprint(&changed_layout).expect("fingerprint")
    );
}

fn dynamic_value_descriptor(
    resource: &str,
    storage: DynamicStorageContract,
) -> DynamicResourceDescriptor {
    DynamicResourceDescriptor::new(
        ResourceId::new(resource).expect("valid resource id"),
        DynamicResourceDemand::fixed(64).expect("valid demand"),
        16,
        BufferUsage::State,
        ElementType::F16,
        AllocationLifetime::Sequence,
        AllocationKind::Value,
        storage,
        1024,
    )
    .expect("valid descriptor")
}

fn invocation_descriptor(
    resource: &str,
    node: &str,
    bytes: u64,
    storage: DynamicStorageContract,
) -> DynamicResourceDescriptor {
    DynamicResourceDescriptor::new(
        ResourceId::new(resource).expect("valid resource id"),
        DynamicResourceDemand::fixed(bytes).expect("valid demand"),
        16,
        BufferUsage::Scratch,
        ElementType::U8,
        AllocationLifetime::Invocation,
        AllocationKind::Scratch {
            node_id: NodeId::new(node).expect("valid node id"),
        },
        storage,
        1024,
    )
    .expect("valid invocation descriptor")
}

fn step_descriptor(
    resource: &str,
    bytes_per_token: u64,
    usage: BufferUsage,
    storage: DynamicStorageContract,
) -> DynamicResourceDescriptor {
    DynamicResourceDescriptor::new(
        ResourceId::new(resource).expect("valid resource id"),
        DynamicResourceDemand::tokens(bytes_per_token, 4096).expect("valid demand"),
        16,
        usage,
        ElementType::F16,
        AllocationLifetime::Step,
        AllocationKind::Value,
        storage,
        1024,
    )
    .expect("valid step descriptor")
}

fn plan_node(id: &str, dependencies: &[&str], resources: &[&str]) -> PlanNode {
    let provider_resources = provider_resources("provider/test");
    let selected_provider = provider_resources.provider_id().clone();
    let mut resources = resources
        .iter()
        .map(|resource| ResourceId::new(*resource).expect("valid resource id"))
        .collect::<Vec<_>>();
    resources.sort();
    PlanNode {
        id: NodeId::new(id).expect("valid node id"),
        dependencies: dependencies
            .iter()
            .map(|dependency| NodeId::new(*dependency).expect("valid dependency id"))
            .collect(),
        operation_id: OperationId::new(format!("operation/{id}")).expect("operation id"),
        operation_version: ContractVersion::new(1, 0),
        operation_fingerprint: "4".repeat(64),
        provider_implementation_fingerprint: "5".repeat(64),
        required_capabilities: BTreeSet::new(),
        attributes: BTreeMap::new(),
        selection: ProviderSelection {
            requested_provider: None,
            selected_provider,
            selection_reason: ProviderSelectionReason::DeterministicCompatible,
            rejected_providers: Vec::new(),
        },
        provider_resources,
        values: Vec::new(),
        exact_aliases: Vec::new(),
        state_effects: Vec::new(),
        scratch_resource: None,
        persistent_resource: None,
        resources,
    }
}

#[test]
fn fixed_block_storage_quantizes_logical_demand_and_maxima() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(fixed_block_profile(4096), layout).expect("storage");
    let descriptor = DynamicResourceDescriptor::new(
        ResourceId::new("resource/quantized").expect("resource id"),
        DynamicResourceDemand::tokens(1, 4097).expect("demand"),
        16,
        BufferUsage::State,
        ElementType::U8,
        AllocationLifetime::Sequence,
        AllocationKind::Value,
        storage,
        1024,
    )
    .expect("descriptor");

    assert_eq!(
        descriptor
            .evaluate_logical_request_bytes_for_shape(
                DynamicResourceShape::new(1, 1, 1).expect("shape"),
            )
            .expect("logical"),
        1
    );
    assert_eq!(descriptor.physical_allocation_quantum_bytes(), 4096);
    assert_eq!(descriptor.minimum_request_bytes().expect("minimum"), 4096);
    assert_eq!(
        descriptor
            .theoretical_maximum_request_bytes()
            .expect("maximum"),
        8192
    );
}

#[test]
fn fixed_block_storage_uses_larger_allocator_geometry_and_rejects_overflow() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage =
        DynamicStorageContract::new(fixed_block_profile(8192), layout.clone()).expect("storage");
    let descriptor = dynamic_value_descriptor("resource/8k-state", storage.clone());
    assert_eq!(descriptor.physical_allocation_quantum_bytes(), 8192);
    assert_eq!(descriptor.minimum_request_bytes().expect("minimum"), 8192);

    let error = DynamicResourceDescriptor::new(
        ResourceId::new("resource/overflow").expect("resource id"),
        DynamicResourceDemand::fixed(u64::MAX).expect("logical demand is representable"),
        16,
        BufferUsage::State,
        ElementType::U8,
        AllocationLifetime::Sequence,
        AllocationKind::Value,
        storage,
        1024,
    )
    .expect_err("physical quantum rounding must reject overflow");
    assert!(error.to_string().contains("overflows u64"));
}

#[test]
fn per_pool_total_order_uses_maximum_invocation_row() {
    let layout = canonical_fingerprint(&"opaque_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let first = invocation_descriptor("resource/invocation-a", "node/a", 64, storage.clone());
    let second = invocation_descriptor("resource/invocation-b", "node/b", 128, storage);
    let nodes = vec![
        plan_node("node/a", &[], &["resource/invocation-a"]),
        plan_node("node/middle", &["node/a"], &[]),
        plan_node("node/b", &["node/middle"], &["resource/invocation-b"]),
    ];

    let pools =
        MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20).expect("derive pools");
    assert_eq!(pools.len(), 1);
    assert_eq!(
        pools[0].invocation_liveness_mode(),
        InvocationLivenessMode::TotalOrderReuse
    );
    assert_eq!(pools[0].minimum_invocation_peak_bytes(), 128);
    assert_eq!(pools[0].provisioning().minimum_resident_bytes(), 128);
}

#[test]
fn per_pool_unordered_invocations_use_conservative_sum() {
    let layout = canonical_fingerprint(&"opaque_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let first = invocation_descriptor("resource/invocation-a", "node/a", 64, storage.clone());
    let second = invocation_descriptor("resource/invocation-b", "node/b", 128, storage);
    let nodes = vec![
        plan_node("node/a", &[], &["resource/invocation-a"]),
        plan_node("node/b", &[], &["resource/invocation-b"]),
    ];

    let pools =
        MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20).expect("derive pools");
    assert_eq!(
        pools[0].invocation_liveness_mode(),
        InvocationLivenessMode::ConservativeConcurrent
    );
    assert_eq!(pools[0].minimum_invocation_peak_bytes(), 192);
}

#[test]
fn ordered_step_activations_share_one_single_fence_slot() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let first = step_descriptor(
        "resource/activation-a",
        64,
        BufferUsage::Activations,
        storage.clone(),
    );
    let second = step_descriptor(
        "resource/activation-b",
        128,
        BufferUsage::Activations,
        storage,
    );
    let nodes = vec![
        plan_node("node/a", &[], &["resource/activation-a"]),
        plan_node("node/middle", &["node/a"], &[]),
        plan_node("node/b", &["node/middle"], &["resource/activation-b"]),
    ];

    let pools =
        MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20).expect("derive pools");
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].minimum_step_bytes(), 128);
    assert_eq!(pools[0].step_resource_slots().len(), 1);
    assert_eq!(
        pools[0].step_resource_slots()[0].kind(),
        StepResourceSlotKind::OrderedSingleFenceStepWave
    );
}

#[test]
fn overlapping_step_activations_keep_distinct_physical_slots() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let first = step_descriptor(
        "resource/activation-a",
        64,
        BufferUsage::Activations,
        storage.clone(),
    );
    let second = step_descriptor(
        "resource/activation-b",
        128,
        BufferUsage::Activations,
        storage,
    );
    let nodes = vec![
        plan_node("node/a", &[], &["resource/activation-a"]),
        plan_node(
            "node/b",
            &["node/a"],
            &["resource/activation-a", "resource/activation-b"],
        ),
    ];

    let pools =
        MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20).expect("derive pools");
    assert_eq!(pools[0].minimum_step_bytes(), 192);
    assert_eq!(pools[0].step_resource_slots().len(), 2);
    assert!(pools[0]
        .step_resource_slots()
        .iter()
        .all(|slot| slot.kind() == StepResourceSlotKind::Dedicated));
}

#[test]
fn ordered_step_state_never_consumes_activation_reuse_proof() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let first = step_descriptor("resource/state-a", 64, BufferUsage::State, storage.clone());
    let second = step_descriptor("resource/state-b", 128, BufferUsage::State, storage);
    let nodes = vec![
        plan_node("node/a", &[], &["resource/state-a"]),
        plan_node("node/b", &["node/a"], &["resource/state-b"]),
    ];

    let pools =
        MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20).expect("derive pools");
    assert_eq!(pools[0].minimum_step_bytes(), 192);
    assert_eq!(pools[0].step_resource_slots().len(), 2);
    assert!(pools[0]
        .step_resource_slots()
        .iter()
        .all(|slot| slot.kind() == StepResourceSlotKind::Dedicated));
}

#[test]
fn dynamic_pool_derivation_is_order_independent_and_compatibility_hashed() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let first = dynamic_value_descriptor("resource/state-a", storage.clone());
    let second = dynamic_value_descriptor("resource/state-b", storage.clone());

    let forward = MemoryPlan::derive_dynamic_pools(&[first.clone(), second.clone()], &[], 1 << 20)
        .expect("derive pools");
    let reverse =
        MemoryPlan::derive_dynamic_pools(&[second, first], &[], 1 << 20).expect("derive pools");

    assert_eq!(forward, reverse);
    assert_eq!(forward.len(), 1);

    let changed_layout = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"strided_v1", "test layout").expect("fingerprint"),
    )
    .expect("storage");
    let changed = dynamic_value_descriptor("resource/state-c", changed_layout);
    assert_ne!(changed.pool_id(), forward[0].pool_id());
}

#[test]
fn dynamic_pool_identity_covers_every_physical_compatibility_dimension() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let changed_layout = canonical_fingerprint(&"strided_v1", "test layout").expect("fingerprint");
    let linear = DynamicStorageContract::new(linear_profile(), layout.clone()).expect("storage");
    let paged = DynamicStorageContract::new(paged_profile(), layout).expect("storage");
    let changed_layout =
        DynamicStorageContract::new(linear_profile(), changed_layout).expect("storage");

    let ids = [
        PoolCompatibilityKey::new(&linear, BufferUsage::State, ElementType::F16, 16),
        PoolCompatibilityKey::new(&paged, BufferUsage::State, ElementType::F16, 16),
        PoolCompatibilityKey::new(&linear, BufferUsage::Scratch, ElementType::F16, 16),
        PoolCompatibilityKey::new(&linear, BufferUsage::State, ElementType::U8, 16),
        PoolCompatibilityKey::new(&changed_layout, BufferUsage::State, ElementType::F16, 16),
        PoolCompatibilityKey::new(&linear, BufferUsage::State, ElementType::F16, 32),
    ]
    .into_iter()
    .map(|key| {
        DynamicBackingPoolId::from_compatibility(&key.expect("compatibility key")).expect("pool id")
    })
    .collect::<BTreeSet<_>>();

    assert_eq!(ids.len(), 6);
}

#[test]
fn memory_plan_wire_rejects_missing_core_derived_pool() {
    let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
    let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
    let descriptor = dynamic_value_descriptor("resource/state-a", storage);
    let plan = MemoryPlan::from_core(1 << 20, 1 << 20, 4096, 1024, vec![], vec![descriptor], &[])
        .expect("valid memory plan");
    let encoded = serde_json::to_vec(&plan).expect("serialize memory plan");
    let decoded: MemoryPlan = serde_json::from_slice(&encoded).expect("round trip memory plan");
    assert_eq!(decoded, plan);

    let mut tampered = serde_json::to_value(&plan).expect("serialize memory plan");
    tampered["dynamic_pools"] = serde_json::json!([]);
    let error = serde_json::from_value::<MemoryPlan>(tampered)
        .expect_err("missing derived pool must be rejected");
    assert!(error
        .to_string()
        .contains("dynamic backing pool count is not derived"));
}

#[test]
fn plan_scope_allocation_round_trips_selected_storage_and_physical_quantum() {
    let storage = DynamicStorageContract::new(
        fixed_block_profile(4096),
        workspace_storage_layout_fingerprint().expect("workspace layout"),
    )
    .expect("storage");
    let allocation = ResourceAllocation::new(
        ResourceId::new("resource/plan-workspace").expect("resource id"),
        1,
        16,
        BufferUsage::Persistent,
        ElementType::U8,
        AllocationKind::Persistent {
            node_id: NodeId::new("node/plan-workspace").expect("node id"),
        },
        storage.clone(),
    )
    .expect("allocation");
    assert_eq!(allocation.per_instance_bytes(), 1);
    assert_eq!(allocation.instance_stride_bytes(), 4096);
    assert_eq!(allocation.storage(), &storage);

    let plan = MemoryPlan::from_core(1 << 20, 1 << 20, 4096, 1024, vec![allocation], vec![], &[])
        .expect("memory plan");
    let encoded = serde_json::to_vec(&plan).expect("serialize");
    let decoded: MemoryPlan = serde_json::from_slice(&encoded).expect("round trip");
    assert_eq!(decoded, plan);
    assert_eq!(
        decoded.static_allocations()[0].storage().profile(),
        fixed_block_profile(4096)
    );

    let mut tampered = serde_json::to_value(&plan).expect("serialize");
    tampered["static_allocations"][0]["storage"]["logical_layout_fingerprint"] =
        serde_json::json!("not-a-fingerprint");
    assert!(serde_json::from_value::<MemoryPlan>(tampered).is_err());
}

#[test]
fn storage_fallback_evidence_is_nonempty_canonical_and_matches_conflicts() {
    let first = ResourceId::new("resource/a").expect("resource id");
    let second = ResourceId::new("resource/b").expect("resource id");
    let linear = linear_profile();
    let paged = paged_profile();
    let candidate = JointProviderCandidate {
        resources: provider_resources("provider/preferred"),
        allowed_profiles: BTreeMap::from([
            (first.clone(), BTreeSet::from([linear])),
            (second.clone(), BTreeSet::from([paged])),
        ]),
        is_preferred: true,
    };
    let selected_profiles = BTreeMap::from([(first, paged), (second.clone(), linear)]);
    let resource_ids = storage_incompatible_resource_ids(&candidate, &selected_profiles);
    assert_eq!(
        resource_ids,
        vec![ResourceId::new("resource/a").expect("resource id"), second]
    );

    let selection = ProviderSelection {
        requested_provider: Some(ProviderId::new("provider/preferred").expect("provider id")),
        selected_provider: ProviderId::new("provider/fallback").expect("provider id"),
        selection_reason: ProviderSelectionReason::FallbackFromPreferred,
        rejected_providers: vec![RejectedProvider {
            provider_id: ProviderId::new("provider/preferred").expect("provider id"),
            reasons: PlanProviderRejectReason::StorageIncompatible {
                resource_ids: resource_ids.clone(),
            },
        }],
    };
    ExecutionPlan::validate_provider_selection_evidence(&selection)
        .expect("canonical fallback evidence");
    let encoded = serde_json::to_vec(&selection).expect("serialize selection");
    let decoded: ProviderSelection =
        serde_json::from_slice(&encoded).expect("round trip selection");
    assert_eq!(decoded, selection);

    let empty = ProviderSelection {
        rejected_providers: vec![RejectedProvider {
            provider_id: ProviderId::new("provider/preferred").expect("provider id"),
            reasons: PlanProviderRejectReason::StorageIncompatible {
                resource_ids: Vec::new(),
            },
        }],
        ..selection
    };
    assert!(ExecutionPlan::validate_provider_selection_evidence(&empty).is_err());
}
