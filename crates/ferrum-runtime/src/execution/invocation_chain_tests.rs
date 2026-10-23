use super::*;

fn graph() -> (Vec<DynamicResourceDescriptor>, Vec<PlanNode>) {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"opaque_v1", "chain scratch layout").unwrap(),
    )
    .unwrap();
    // Two successive forks. A completed sibling can share a later slot, but
    // a/b and c/d each need disjoint backing within their respective fork.
    let specs: [(&str, &[&str], u64); 4] = [
        ("node/a", &[], 16),
        ("node/b", &[], 32),
        ("node/c", &["node/a", "node/b"], 48),
        ("node/d", &["node/a", "node/b"], 64),
    ];
    let mut descriptors = Vec::new();
    let mut nodes = Vec::new();
    for (id, dependencies, bytes) in specs {
        let resource = format!("scratch/{id}");
        descriptors.push(invocation_descriptor(&resource, id, bytes, storage.clone()));
        let mut node = plan_node(id, dependencies, &[&resource]);
        node.scratch_resource = Some(ResourceId::new(resource).unwrap());
        node.provider_resources.scratch = Some(
            ProviderWorkspaceRequirement::new(
                bytes,
                16,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )
            .unwrap(),
        );
        nodes.push(node);
    }
    (descriptors, nodes)
}

fn memory(
    descriptors: Vec<DynamicResourceDescriptor>,
    nodes: &[PlanNode],
    budget: u64,
) -> Result<MemoryPlan, VNextError> {
    MemoryPlan::from_core(budget, budget, 0, 1, vec![], descriptors, nodes, None)
}

#[test]
fn invocation_chain_forks_share_only_completed_rows_and_fit_exact_budget() {
    let (descriptors, nodes) = graph();
    let memory = memory(descriptors.clone(), &nodes, 112).unwrap();
    assert_eq!(memory.minimum_runnable_request_bytes(), 112);
    let pool = &memory.dynamic_pools()[0];
    assert_eq!(
        pool.invocation_liveness_mode(),
        InvocationLivenessMode::DependencyChainReuse
    );
    assert_eq!(pool.minimum_invocation_peak_bytes(), 112);
    let slots = pool
        .invocation_liveness()
        .iter()
        .map(|row| row.chain_slot())
        .collect::<Vec<_>>();
    assert_eq!(slots, [Some(0), Some(1), Some(0), Some(1)]);
    let layout = pool
        .invocation_liveness_mode()
        .layout(&slots, &[16, 32, 48, 64])
        .unwrap();
    assert_eq!(layout.bytes, 112);
    assert_eq!(layout.row_offsets, [0, 48, 0, 48]);
    assert!(matches!(
        super::invocation_chain::memory(descriptors, &nodes, 111),
        Err(VNextError::PlanCapacityExceeded {
            required_bytes: 112,
            usable_capacity_bytes: 111
        })
    ));
}

#[test]
fn invocation_chain_excludes_unproven_initialization_and_foreign_scratch_ownership() {
    let (descriptors, nodes) = graph();
    for mutation in 0..3 {
        let mut nodes = nodes.clone();
        match mutation {
            0 => {
                nodes[0]
                    .provider_resources
                    .scratch
                    .as_mut()
                    .unwrap()
                    .reuse_policy = ProviderWorkspaceReusePolicy::ZeroBeforeUse
            }
            1 => nodes[0].provider_resources.scratch = None,
            _ => nodes[0].scratch_resource = nodes[1].scratch_resource.clone(),
        }
        let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 1024).unwrap();
        assert_eq!(
            pools[0].invocation_liveness_mode(),
            InvocationLivenessMode::ConservativeConcurrent
        );
        assert_eq!(pools[0].minimum_invocation_peak_bytes(), 160);
        assert!(pools[0]
            .invocation_liveness()
            .iter()
            .all(|row| row.chain_slot().is_none()));
    }
}

#[test]
fn invocation_chain_token_shape_and_reusable_capacity_use_slot_peaks() {
    let (mut descriptors, nodes) = graph();
    for descriptor in &mut descriptors {
        let bytes = descriptor.minimum_request_bytes().unwrap();
        descriptor.demand = DynamicResourceDemand::tokens(bytes, 16).unwrap();
    }
    let policy = ReusableExecutionPolicy::new(
        1,
        vec![ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("class/chain").unwrap(),
            ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
        )
        .unwrap()],
    )
    .unwrap();
    let memory =
        MemoryPlan::from_core(4096, 4096, 0, 1, vec![], descriptors, &nodes, Some(&policy))
            .unwrap();
    assert_eq!(memory.minimum_invocation_peak_bytes(), 112);
    assert_eq!(
        memory.reusable_execution().unwrap().buckets()[0].pool_budgets()[0].invocation_bytes(),
        448
    );
    assert_eq!(memory.startup_peak_bytes(1, 1, 1).unwrap(), 448);
    assert_eq!(memory.startup_peak_bytes(1, 1, 5).unwrap(), 560);
}

#[test]
fn invocation_chain_rehydration_preserves_legacy_conservative_layout_and_wire() {
    let (descriptors, nodes) = graph();
    // This is the layout emitted before dependency-chain optimization. Keep
    // its old charge and wire bytes even though the current graph proves reuse.
    let mut old_nodes = nodes.clone();
    for node in &mut old_nodes {
        node.provider_resources.scratch = None;
    }
    let legacy = memory(descriptors.clone(), &old_nodes, 1024).unwrap();
    let wire = serde_json::to_vec(&legacy).unwrap();
    assert!(!String::from_utf8(wire.clone())
        .unwrap()
        .contains("chain_slot"));
    let restored: MemoryPlan = serde_json::from_slice(&wire).unwrap();
    assert_eq!(serde_json::to_vec(&restored).unwrap(), wire);
    let mut expected = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 1024).unwrap();
    MemoryPlan::preserve_legacy_invocation_layouts(
        &mut expected,
        restored.dynamic_pools(),
        &descriptors,
        1024,
    )
    .unwrap();
    assert_eq!(expected, restored.dynamic_pools());
    assert_eq!(expected[0].minimum_invocation_peak_bytes(), 160);
    let layout = expected[0]
        .invocation_liveness_mode()
        .layout(&[None; 4], &[16, 32, 48, 64])
        .unwrap();
    assert_eq!(layout.row_offsets, [0, 16, 48, 96]);
}

#[test]
fn invocation_chain_wire_requires_dense_complete_slots_and_dependency_evidence() {
    let (descriptors, nodes) = graph();
    let original = memory(descriptors.clone(), &nodes, 1024).unwrap();
    let wire = serde_json::to_value(&original).unwrap();
    for bad_slot in [
        serde_json::Value::Null,
        serde_json::json!(4),
        serde_json::json!(2),
    ] {
        let mut bad = wire.clone();
        bad["dynamic_pools"][0]["invocation_liveness"][1]["chain_slot"] = bad_slot.clone();
        bad["dynamic_pools"][0]["invocation_liveness"][3]["chain_slot"] = bad_slot;
        assert!(serde_json::from_value::<MemoryPlan>(bad).is_err());
    }
    let restored: MemoryPlan = serde_json::from_value(wire).unwrap();
    assert_eq!(restored, original);
    let mut changed = nodes;
    changed[2].dependencies.clear();
    let rederived = MemoryPlan::derive_dynamic_pools(&descriptors, &changed, 1024).unwrap();
    // ExecutionPlan rehydration compares these complete derived pools, so old
    // slot evidence cannot survive removal of its completion dependency.
    assert_ne!(rederived, restored.dynamic_pools());
    let mut tampered = restored.dynamic_pools().to_vec();
    tampered[0].invocation_liveness[1].chain_slot = Some(0);
    assert_ne!(tampered, original.dynamic_pools());
}

#[test]
fn invocation_chain_duplicate_resources_are_not_a_new_legacy_wire_restriction() {
    let (descriptors, _) = graph();
    let descriptor = &descriptors[0];
    let id = descriptor.base_resource_id().clone();
    let rows = vec![
        InvocationResourceLiveness {
            node_id: NodeId::new("node/a").unwrap(),
            resource_ids: vec![id.clone()],
            chain_slot: None,
        },
        InvocationResourceLiveness {
            node_id: NodeId::new("node/b").unwrap(),
            resource_ids: vec![id.clone()],
            chain_slot: None,
        },
    ];
    let ids = BTreeSet::from([id.clone()]);
    let descriptors = BTreeMap::from([(id, descriptor)]);
    assert_eq!(
        validate_pool_liveness_rows(
            InvocationLivenessMode::ConservativeConcurrent,
            &rows,
            &ids,
            &descriptors
        )
        .unwrap(),
        32
    );
    assert_eq!(
        validate_pool_liveness_rows(
            InvocationLivenessMode::TotalOrderReuse,
            &rows,
            &ids,
            &descriptors
        )
        .unwrap(),
        16
    );
    let mut chain = rows;
    for row in &mut chain {
        row.chain_slot = Some(0);
    }
    assert!(validate_pool_liveness_rows(
        InvocationLivenessMode::DependencyChainReuse,
        &chain,
        &ids,
        &descriptors
    )
    .is_err());
}

#[test]
fn invocation_chain_layout_rejects_overflow_and_legacy_slot_injection() {
    assert!(InvocationLivenessMode::DependencyChainReuse
        .layout(&[Some(0), Some(1)], &[u64::MAX, 1])
        .is_err());
    assert!(InvocationLivenessMode::ConservativeConcurrent
        .layout(&[Some(0)], &[16])
        .is_err());
    assert!(InvocationLivenessMode::TotalOrderReuse
        .layout(&[Some(0)], &[16])
        .is_err());
    let old = InvocationLivenessMode::TotalOrderReuse
        .layout(&[None; 2], &[16, 32])
        .unwrap();
    assert_eq!((old.bytes, old.row_offsets), (32, vec![0, 0]));
}
