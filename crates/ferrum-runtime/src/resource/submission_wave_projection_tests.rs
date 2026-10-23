use super::*;
use crate::{
    DynamicStorageContract, DynamicStorageRequirement, MemoryPlan, NodeId,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
};

fn wave_harness(mode: InvocationLivenessMode) -> (Harness, ReusableExecutionBucketSpec) {
    let profile = linear_profile();
    let storage = DynamicStorageContract::resource_test_contract(profile, "5".repeat(64)).unwrap();
    let mut nodes = Vec::new();
    let mut descriptors = Vec::new();
    for (index, bytes) in [16, 32, 48, 64].into_iter().enumerate() {
        let node = NodeId::new(format!("node/wave-{index}")).unwrap();
        let resource = ResourceId::new(format!("scratch/wave-{index}")).unwrap();
        let dependencies = match mode {
            InvocationLivenessMode::TotalOrderReuse if index > 0 => {
                vec![NodeId::new(format!("node/wave-{}", index - 1)).unwrap()]
            }
            InvocationLivenessMode::DependencyChainReuse if index >= 2 => vec![
                NodeId::new("node/wave-0").unwrap(),
                NodeId::new("node/wave-1").unwrap(),
            ],
            _ => vec![],
        };
        nodes.push(PlanNode::resource_test_node_with_scratch(
            node.clone(),
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
                DynamicResourceDemand::tokens(bytes, 4).unwrap(),
                16,
                node,
                storage.clone(),
                1,
            )
            .unwrap(),
        );
    }
    let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 4096).unwrap();
    assert_eq!(pools.len(), 1);
    assert_eq!(pools[0].invocation_liveness_mode(), mode);
    let capacity_bytes = match mode {
        InvocationLivenessMode::TotalOrderReuse => 256,
        InvocationLivenessMode::DependencyChainReuse => 448,
        InvocationLivenessMode::ConservativeConcurrent => 640,
        InvocationLivenessMode::NoInvocationResources => unreachable!(),
    };
    let catalog = PoolCatalog {
        pool_id: pools[0].pool_id().clone(),
        pools,
        descriptors,
        profile,
    };
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("class/wave-projection").unwrap(),
        ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
    )
    .unwrap();
    let reusable = ReusableExecutionMemoryPlan::new(
        1,
        1,
        vec![ResolvedReusableExecutionBucket::new(
            bucket.clone(),
            vec![
                ReusablePoolWorkspaceBudget::new(catalog.pool_id.clone(), 0, capacity_bytes)
                    .unwrap(),
            ],
        )
        .unwrap()],
    )
    .unwrap();
    let runtime = new_runtime(&catalog, 4096);
    let harness = harness_with_nodes_and_reusable(
        runtime,
        catalog,
        4096,
        false,
        nodes.into(),
        Some(reusable),
    );
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], capacity_bytes)
        .unwrap();
    (harness, bucket)
}

#[test]
fn compiled_wave_projection_keeps_dynamic_demand_and_reusable_offsets_in_all_modes() {
    for (mode, immediate_bytes, capacity_bytes, offsets) in [
        (
            InvocationLivenessMode::TotalOrderReuse,
            64,
            256,
            [0, 0, 0, 0],
        ),
        (
            InvocationLivenessMode::DependencyChainReuse,
            112,
            448,
            [0, 192, 0, 192],
        ),
        (
            InvocationLivenessMode::ConservativeConcurrent,
            160,
            640,
            [0, 64, 192, 384],
        ),
    ] {
        let (h, bucket) = wave_harness(mode);
        let lane = h.root.create_execution_lane().unwrap();
        let binding = h.root.trusted_runtime_binding().unwrap();
        let allocations = h.runtime.allocate_calls();
        let mut certificate = None;
        for tokens in [1, 2] {
            let (demand, requests) = binding
                .submission_wave_demand(
                    shape(tokens),
                    shape(3),
                    Some(&bucket),
                    AdmissionFitPolicy::FullInputMustFit,
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap();
            assert_eq!(
                demand.immediate_claim().entries()[0].units().get(),
                immediate_bytes * tokens
            );
            assert_eq!(
                demand.fit_requirement().entries()[0].units().get(),
                immediate_bytes * 3
            );
            assert_eq!(requests[0].capacity_size_bytes, capacity_bytes);
            assert_eq!(
                requests[0]
                    .projections
                    .iter()
                    .map(|p| p.physical_offset_bytes)
                    .collect::<Vec<_>>(),
                offsets
            );
            let LaneBackingPrepareDecision::Prepared(prepared) = h
                .root
                .dynamic_pools
                .prepare_lane_stable_claim(&lane, &requests)
                .unwrap()
            else {
                panic!("resident wave must prepare");
            };
            let committed = prepared.commit();
            if let Some(first) = &certificate {
                assert!(Arc::ptr_eq(first, committed.certificate()));
            } else {
                certificate = Some(Arc::clone(committed.certificate()));
            }
            let (slices, slot) = committed.into_parts();
            for (index, slice) in slices.iter().enumerate() {
                assert_eq!(
                    slice.evidence().size_bytes(),
                    (index as u64 + 1) * 16 * tokens
                );
                assert_eq!(
                    slice.evidence().capacity_size_bytes(),
                    (index as u64 + 1) * 64
                );
            }
            drop(slices);
            drop(slot);
        }
        assert_eq!(h.runtime.allocate_calls(), allocations);
        drop(binding);
        drop(lane);
        close_dynamic_test_root(h.root);
    }
}

#[test]
fn compiled_wave_projection_rejects_malformed_members_and_layout_before_warm_reuse() {
    let (h, bucket) = wave_harness(InvocationLivenessMode::DependencyChainReuse);
    let lane = h.root.create_execution_lane().unwrap();
    let binding = h.root.trusted_runtime_binding().unwrap();
    let (_, requests) = binding
        .submission_wave_demand(
            shape(1),
            shape(1),
            Some(&bucket),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
    let LaneBackingPrepareDecision::Prepared(prepared) = h
        .root
        .dynamic_pools
        .prepare_lane_stable_claim(&lane, &requests)
        .unwrap()
    else {
        panic!("resident wave must prepare");
    };
    let committed = prepared.commit();
    let certificate = Arc::clone(committed.certificate());
    let (slices, slot) = committed.into_parts();
    drop(slices);
    drop(slot);
    let status = h.root.maintenance_controller.status().unwrap();
    let allocations = h.runtime.allocate_calls();
    let reject = |request: EvaluatedBackingRequest<'_>| {
        assert!(h
            .root
            .dynamic_pools
            .prepare_claim(std::slice::from_ref(&request))
            .is_err());
        assert!(h
            .root
            .dynamic_pools
            .prepare_lane_stable_claim(&lane, std::slice::from_ref(&request))
            .is_err());
        assert_eq!(h.root.maintenance_controller.status().unwrap(), status);
        assert_eq!(h.runtime.allocate_calls(), allocations);
    };
    let original = &requests[0];
    let mut bad = original.clone();
    bad.projections[0].descriptor_index = usize::MAX;
    reject(bad);
    let mut bad = original.clone();
    bad.projections.swap(0, 1);
    reject(bad);
    let mut bad = original.clone();
    bad.projections[1] = bad.projections[0].clone();
    reject(bad);
    let mut bad = original.clone();
    bad.projections.pop();
    bad.claim_identity = PhysicalBackingClaimIdentity::new(
        bad.domain.pool_id().clone(),
        bad.projections
            .iter()
            .map(|p| p.descriptor.base_resource_id().clone())
            .collect(),
    )
    .unwrap();
    reject(bad);
    let mut bad = original.clone();
    bad.projections[0].physical_offset_bytes = 16;
    reject(bad);
    let mut bad = original.clone();
    bad.capacity_size_bytes += 16;
    reject(bad);
    let mut bad = original.clone();
    bad.projections[0].capacity_size_bytes += 16;
    reject(bad);
    let mut bad = original.clone();
    bad.projections
        .iter_mut()
        .for_each(|p| p.capacity_size_bytes = u64::MAX);
    reject(bad);
    let mut wrong_domain = original.domain.clone();
    wrong_domain.domain_id = CapacityDomainId::new(99).unwrap();
    let mut bad = original.clone();
    bad.domain = &wrong_domain;
    reject(bad);
    // The cached key and every projection remain identical. A forged liveness
    // declaration in the domain must not be accepted merely because IDs match.
    let mut wrong_layout_domain = original.domain.clone();
    let mut wire = serde_json::to_value(&wrong_layout_domain.pool).unwrap();
    wire["invocation_liveness"][0]["chain_slot"] = serde_json::json!(1);
    wrong_layout_domain.pool = serde_json::from_value(wire).unwrap();
    let mut bad = original.clone();
    bad.domain = &wrong_layout_domain;
    reject(bad);
    let foreign_bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("class/foreign-wave").unwrap(),
        ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
    )
    .unwrap();
    let mut bad = original.clone();
    bad.reusable_execution_bucket_id = Some(foreign_bucket.bucket_id().clone());
    reject(bad);
    let equivalent_domain = original.domain.clone();
    let mut equivalent_request = original.clone();
    equivalent_request.domain = &equivalent_domain;
    let LaneBackingPrepareDecision::Prepared(prepared) = h
        .root
        .dynamic_pools
        .prepare_lane_stable_claim(&lane, &[equivalent_request])
        .unwrap()
    else {
        panic!("bad requests must leave the idle slot reusable");
    };
    let committed = prepared.commit();
    assert!(Arc::ptr_eq(&certificate, committed.certificate()));
    drop(committed);
    drop(binding);
    drop(lane);
    close_dynamic_test_root(h.root);
}

#[test]
fn compiled_wave_projection_leaves_dedicated_step_and_non_invocation_claims_valid() {
    for lifetime in [AllocationLifetime::Step, AllocationLifetime::Request] {
        let catalog = pool_catalog(linear_profile(), lifetime, '9', 2, 256, TestDemand::Fixed);
        let runtime = new_runtime(&catalog, 256);
        let h = harness(runtime, catalog, 256, false);
        h.root
            .maintenance_controller
            .grow_pool(&h.pool_ids[0], 256)
            .unwrap();
        let binding = h.root.trusted_runtime_binding().unwrap();
        let (_, requests) = binding
            .scoped_demand(
                lifetime,
                None,
                shape(1),
                shape(1),
                None,
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests
            .iter()
            .all(|request| request.projections.len() == 1));
        let BackingPrepareDecision::Prepared(prepared) =
            h.root.dynamic_pools.prepare_claim(&requests).unwrap()
        else {
            panic!("dedicated and non-Invocation claims must still prepare");
        };
        let claims = prepared.commit();
        assert_eq!(claims.len(), 2);
        assert!(claims
            .iter()
            .all(|claim| claim.evidence().size_bytes() == 64));
        drop(claims);
        drop(binding);
        close_dynamic_test_root(h.root);
    }
}
