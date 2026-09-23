//! Multiple logical views must count their shared physical claim only once.
use super::*;
use crate::vnext::{DynamicStorageContract, MemoryPlan};

#[test]
fn planning_coalesced_invocation_workspace_keeps_distinct_projection_ranges() {
    // These are real conservative program-binding resources, not overlapping
    // zero-offset Step views: [0, 64) and [64, 192) share one physical lease.
    let profile = linear_profile();
    let storage = DynamicStorageContract::resource_test_contract(profile, "d".repeat(64)).unwrap();
    let descriptors: Vec<_> = [("first", 64), ("second", 128)]
        .into_iter()
        .map(|(name, bytes)| {
            DynamicResourceDescriptor::resource_test_binding(
                ResourceId::new(format!("resource/planning-binding-{name}")).unwrap(),
                DynamicResourceDemand::fixed(bytes).unwrap(),
                16,
                NodeId::new(format!("node/planning-binding-{name}")).unwrap(),
                storage.clone(),
                8,
            )
            .unwrap()
        })
        .collect();
    let nodes: Vec<_> = ["first", "second"]
        .into_iter()
        .zip(&descriptors)
        .map(|(name, descriptor)| {
            PlanNode::resource_test_node_with_binding(
                NodeId::new(format!("node/planning-binding-{name}")).unwrap(),
                descriptor.base_resource_id().clone(),
            )
        })
        .collect();
    let pools = MemoryPlan::derive_dynamic_pools(&descriptors, &nodes, 192).unwrap();
    assert_eq!(pools.len(), 1);
    let pool_id = descriptors[0].pool_id().clone();
    let catalog = PoolCatalog {
        pools,
        descriptors,
        pool_id: pool_id.clone(),
        profile,
    };
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("test.distinct-binding-ranges").unwrap(),
        ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
    )
    .unwrap();
    let memory = ReusableExecutionMemoryPlan::new(
        1,
        1,
        vec![ResolvedReusableExecutionBucket::new(
            bucket.clone(),
            vec![ReusablePoolWorkspaceBudget::new(pool_id, 0, 192).unwrap()],
        )
        .unwrap()],
    )
    .unwrap();
    let harness = harness_with_nodes_and_reusable(
        new_runtime(&catalog, 192),
        catalog,
        192,
        false,
        Arc::from(nodes),
        Some(memory),
    );
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 192)
        .unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "distinct-binding-ranges", 4);
    let session = sequence.open_session().unwrap();
    let cold = lane_view(&harness.root, &session, &lane);
    let projected = known(harness.root.project_resource_wave_with_bucket(
        &cold,
        &cold.initial_state(),
        &[row(0, 0, 1)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let shape = DynamicResourceShape::from_validated(1, 1, 0);
    let (_, requests) = binding
        .submission_wave_demand(
            shape,
            shape,
            Some(&bucket),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
    let claim = || match harness
        .root
        .dynamic_pools
        .prepare_lane_stable_claim(&lane, &requests)
        .unwrap()
    {
        LaneBackingPrepareDecision::Prepared(claim) => claim.commit().into_parts(),
        _ => panic!("resident binding arena must admit its exact physical claim"),
    };
    let allocations = harness.runtime.allocate_calls();
    let (slices, slot) = claim();
    assert_eq!(slices.len(), 2);
    let first = slices[0].evidence();
    let second = slices[1].evidence();
    assert_eq!(
        first.physical_claim_identity(),
        second.physical_claim_identity()
    );
    assert_eq!(
        (first.physical_offset_bytes(), first.capacity_size_bytes()),
        (0, 64)
    );
    assert_eq!(
        (second.physical_offset_bytes(), second.capacity_size_bytes()),
        (64, 128)
    );
    assert_eq!(first.physical_size_bytes(), 192);
    assert_ne!(first.segments(), second.segments());
    assert!(Arc::ptr_eq(
        &slices[0].segment_lease,
        &slices[1].segment_lease
    ));
    assert_eq!(
        slices[0]
            .segment_lease
            .segments
            .iter()
            .map(BackingSegment::length_bytes)
            .sum::<u64>(),
        192
    );
    let first_ranges: Vec<_> = slices
        .iter()
        .map(|slice| slice.evidence().segments().to_vec())
        .collect();
    let slot_id = slot.as_ref().unwrap().identity();
    drop(slices);
    drop(slot);
    // The old capture compared these distinct projections and returned
    // InvalidDemand after the first real decode had retained its binding slot.
    let hot = lane_view(&harness.root, &session, &lane);
    let next = known(harness.root.project_resource_wave_with_bucket(
        &hot,
        &hot.initial_state(),
        &[row(0, 1, 1)],
        Some(bucket.bucket_id()),
        &mut || true,
    ));
    assert_eq!(next.domains, projected.domains);
    let (slices, slot) = claim();
    assert_eq!(slot.as_ref().unwrap().identity(), slot_id);
    for (slice, original) in slices.iter().zip(&first_ranges) {
        assert_eq!(slice.evidence().segments(), original);
    }
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    drop(slices);
    drop(slot);
    drop(requests);
    drop(binding);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
    assert_eq!(
        hot.participants().len(),
        1,
        "the retained numeric view pins no authority"
    );
}

#[test]
fn planning_coalesced_workspace_two_buckets_remain_readable_and_reusable() {
    let catalog = pool_catalog_with_options(
        linear_profile(),
        AllocationLifetime::Step,
        'c',
        64,
        768,
        TestDemand::Tokens,
        "activations",
        true,
        StateInitialization::None,
    );
    let buckets: Vec<_> = ["test.coalesced.a", "test.coalesced.b", "test.coalesced.c"]
        .into_iter()
        .map(|name| {
            ReusableExecutionBucketSpec::new(
                ReusableExecutionClassId::new(name).unwrap(),
                ReusableExecutionCapacity::new(1, 4, 1).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let resolved = buckets
        .iter()
        .map(|bucket| {
            ResolvedReusableExecutionBucket::new(
                bucket.clone(),
                vec![ReusablePoolWorkspaceBudget::new(catalog.pool_id.clone(), 256, 0).unwrap()],
            )
            .unwrap()
        })
        .collect();
    let memory = ReusableExecutionMemoryPlan::new(1, 3, resolved).unwrap();
    let harness = harness_with_reusable(new_runtime(&catalog, 768), catalog, 768, memory);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 768)
        .unwrap();
    let lane = harness.root.create_execution_lane().unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, "coalesced-slots", 4);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let allocations = harness.runtime.allocate_calls();
    let limits = ResourcePlanningLimits {
        maximum_descriptors: 128,
        ..Default::default()
    };
    let cold = lane_view_with_limits(&harness.root, &session, &lane, limits);
    let mut state = cold.initial_state();
    let mut identities = Vec::new();
    for bucket in &buckets[..2] {
        let projected = known(harness.root.project_resource_wave_with_bucket(
            &cold,
            &state,
            &[row(0, 0, 1)],
            Some(bucket.bucket_id()),
            &mut || true,
        ));
        let actual = step(&batch, &lane, 1, Some(bucket));
        assert_eq!(actual.backing_slices().len(), 64);
        assert_eq!(
            actual.backing_slices()[0]
                .evidence()
                .physical_claim_identity()
                .resource_ids()
                .len(),
            64,
            "this exercises one actual coalesced allocation with many views"
        );
        let identity = actual
            .claimed_backing()
            .lane_stable_slot_identity()
            .unwrap();
        assert_eq!(projected.selected_step_slot(), Some(&identity));
        identities.push(identity);
        actual.try_retire_normal().unwrap();
        // With the old duplicated identity accounting, the first slot used
        // 64 * 64 entries and a second bucket exceeded the default 4096 bound.
        let _default_view = lane_view(&harness.root, &session, &lane);
        state = projected.state;
    }
    let hot = lane_view_with_limits(&harness.root, &session, &lane, limits);
    let projected = known(harness.root.project_resource_wave_with_bucket(
        &hot,
        &hot.initial_state(),
        &[row(0, 1, 1)],
        Some(buckets[1].bucket_id()),
        &mut || true,
    ));
    assert_eq!(projected.selected_step_slot(), Some(&identities[1]));
    let actual = step(&batch, &lane, 1, Some(&buckets[1]));
    assert_eq!(
        actual
            .claimed_backing()
            .lane_stable_slot_identity()
            .as_ref(),
        Some(&identities[1])
    );
    actual.try_retire_normal().unwrap();
    assert_eq!(harness.runtime.allocate_calls(), allocations);
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &hot,
            &hot.initial_state(),
            &[row(0, 1, 1)],
            Some(buckets[2].bucket_id()),
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::LimitExceeded)
    ));
    // The real allocator can create the third slot. The stricter read-view
    // limit must still reject its genuine 192 logical projections.
    step(&batch, &lane, 1, Some(&buckets[2]))
        .try_retire_normal()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match harness
            .root
            .resource_planning_view_on_lane(&[&session], &lane, limits, &mut || true)
        {
            ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::ReadUnavailable(_))
                if std::time::Instant::now() < deadline =>
            {
                std::thread::yield_now()
            }
            result => {
                assert!(
                    matches!(
                        result,
                        ResourcePlanningAvailability::Unknown(
                            ResourcePlanningUnknown::LimitExceeded
                        )
                    ),
                    "{result:?}"
                );
                break;
            }
        }
    }
    session.try_abort_if_quiescent().unwrap();
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
    // Keeping all forecasts alive across close cannot retain physical leases.
    assert_eq!(state.projected_waves(), 2);
    assert_eq!(hot.participants().len(), 1);
}
