use super::*;
use crate::{
    DynamicStorageContract, DynamicStorageRequirement, MemoryPlan, NodeId,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ResolvedReusableExecutionBucket, ReusableExecutionBucketSpec, ReusableExecutionCapacity,
    ReusableExecutionClassId, ReusablePoolWorkspaceBudget,
};

fn fixture() -> (DynamicPoolDomainSpec, Vec<PlanNode>) {
    let profile = DynamicStorageProfile::new(
        DynamicStorageAllocator::LinearArena,
        crate::DynamicStorageView::Contiguous,
    )
    .unwrap();
    let storage = DynamicStorageContract::resource_test_contract(profile, "1".repeat(64)).unwrap();
    let mut nodes = Vec::new();
    let mut descriptors = Vec::new();
    for (index, bytes) in [16, 32, 48, 64].into_iter().enumerate() {
        let id = NodeId::new(format!("node/{index}")).unwrap();
        let resource = ResourceId::new(format!("scratch/{index}")).unwrap();
        let dependencies = if index >= 2 {
            vec![
                NodeId::new("node/0").unwrap(),
                NodeId::new("node/1").unwrap(),
            ]
        } else {
            Vec::new()
        };
        nodes.push(PlanNode::resource_test_node_with_scratch(
            id.clone(),
            resource.clone(),
            ProviderWorkspaceRequirement::new(
                bytes,
                16,
                ProviderWorkspaceScope::Invocation,
                ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                DynamicStorageRequirement::contiguous(),
            )
            .unwrap(),
            dependencies,
        ));
        descriptors.push(
            DynamicResourceDescriptor::resource_test_scratch(
                resource,
                DynamicResourceDemand::tokens(bytes, 8).unwrap(),
                16,
                id,
                storage.clone(),
                1,
            )
            .unwrap(),
        );
    }
    let pool = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 4096)
        .unwrap()
        .remove(0);
    (
        DynamicPoolDomainSpec {
            domain_id: CapacityDomainId::new(1).unwrap(),
            pool,
            descriptors,
        },
        nodes,
    )
}

#[test]
fn invocation_chain_real_projection_offsets_preserve_forks_and_capacity_bucket() {
    let (domain, nodes) = fixture();
    let layout = compile_submission_wave_domain_layout(&domain, &nodes)
        .unwrap()
        .unwrap();
    let immediate = layout
        .physical_layout_for_shape(&domain, DynamicResourceShape::new(1, 1, 1).unwrap())
        .unwrap();
    assert_eq!(immediate.bytes, 112);
    assert_eq!(immediate.row_offsets, [0, 48, 0, 48]);
    let fit = layout
        .physical_layout_for_shape(&domain, DynamicResourceShape::new(1, 3, 1).unwrap())
        .unwrap();
    assert_eq!(fit.bytes, 336);
    assert_eq!(fit.row_offsets, [0, 144, 0, 144]);

    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("class/chain").unwrap(),
        ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
    )
    .unwrap();
    let bucket_id = bucket.bucket_id().clone();
    let reusable = ReusableExecutionMemoryPlan::new(
        1,
        1,
        vec![ResolvedReusableExecutionBucket::new(
            bucket,
            vec![ReusablePoolWorkspaceBudget::new(domain.pool.pool_id().clone(), 0, 448).unwrap()],
        )
        .unwrap()],
    )
    .unwrap();
    let layouts = compile_submission_wave_reusable_capacity_layouts(
        std::slice::from_ref(&domain),
        &[Some(layout)],
        Some(&reusable),
    )
    .unwrap();
    let physical = layouts[&bucket_id][0].as_ref().unwrap();
    assert_eq!(physical.physical_size_bytes, 448);
    assert_eq!(
        physical
            .projections
            .iter()
            .map(|p| (p.physical_offset_bytes, p.capacity_size_bytes))
            .collect::<Vec<_>>(),
        [(0, 64), (192, 128), (0, 192), (192, 256)]
    );
    assert!(physical
        .projections
        .iter()
        .all(|p| p.physical_offset_bytes + p.capacity_size_bytes <= 448));
    // A sparse submission cannot rebuild a smaller authority that aliases the
    // omitted sibling's slot. The full plan owns this layout and its extent.
    assert!(compile_submission_wave_domain_layout(&domain, &nodes[1..]).is_err());
}

#[test]
fn invocation_chain_rejects_duplicate_or_missing_projection_members() {
    let (domain, nodes) = fixture();
    let wire = serde_json::to_value(&domain.pool).unwrap();
    for duplicate in [true, false] {
        let mut bad = wire.clone();
        bad["invocation_liveness"][1]["resource_ids"] = if duplicate {
            serde_json::json!(["scratch/0"])
        } else {
            serde_json::json!(["scratch/absent"])
        };
        let bad = DynamicPoolDomainSpec {
            domain_id: domain.domain_id,
            pool: serde_json::from_value(bad).unwrap(),
            descriptors: domain.descriptors.clone(),
        };
        assert!(compile_submission_wave_domain_layout(&bad, &nodes).is_err());
    }
}
