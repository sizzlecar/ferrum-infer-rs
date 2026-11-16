use super::*;

fn aligned_fixture(large_bytes: u64) -> (Vec<PlanNode>, Vec<DynamicResourceDescriptor>) {
    let (mut nodes, mut descriptors) = fixture();
    for index in [1, 2] {
        descriptors[index].demand = DynamicResourceDemand::fixed(large_bytes).unwrap();
    }
    let large = &mut descriptors[1];
    large.alignment_bytes = 256;
    large.pool_id = DynamicBackingPoolId::from_compatibility(
        &PoolCompatibilityKey::new(&large.storage, large.usage, large.element_type, 256).unwrap(),
    )
    .unwrap();
    nodes[1].values[1] = binding("large", ResolvedValueRole::Output, 0, large_bytes);
    nodes[2].values[0] = binding("large", ResolvedValueRole::Input, 0, large_bytes);
    nodes[3].values[0] = binding("later", ResolvedValueRole::Output, 0, large_bytes);
    nodes[4].values[0] = binding("later", ResolvedValueRole::Input, 0, large_bytes);
    (nodes, descriptors)
}

#[test]
fn stronger_alignment_reuses_disjoint_values_at_the_capacity_boundary() {
    let (nodes, descriptors) = aligned_fixture(4096);
    // Two overlapping tensors; the retained output reuses the completed small one.
    // The low-alignment pool otherwise keeps a second 4096-byte slot alive.
    let required = 4096 + 256;
    let plan = memory(&nodes, &descriptors, required, &[]).unwrap();
    assert_eq!(plan.minimum_step_bytes(), required);
    for original in &descriptors {
        let selected = plan
            .dynamic_descriptors()
            .iter()
            .find(|d| d.base_resource_id() == original.base_resource_id())
            .unwrap();
        assert_eq!(selected.demand(), original.demand());
        assert_eq!(selected.alignment_bytes(), 256);
        assert_eq!(selected.alignment_bytes() % original.alignment_bytes(), 0);
    }
    assert_eq!(
        serde_json::from_slice::<MemoryPlan>(&serde_json::to_vec(&plan).unwrap()).unwrap(),
        plan
    );
    assert!(matches!(
        memory(&nodes, &descriptors, required - 1, &[]),
        Err(VNextError::PlanCapacityExceeded { required_bytes, usable_capacity_bytes })
            if required_bytes == required && usable_capacity_bytes == required - 1
    ));
}

#[test]
fn alignment_padding_cannot_increase_the_complete_plan() {
    let (mut nodes, descriptors) = aligned_fixture(256);
    nodes[3].dependencies.clear();
    // Unification would round the 64-byte live tensor and 16-byte retained
    // output to 256 bytes each, costing more than the reuse saves.
    let required = 64 + 256 + 256 + 16;
    let plan = memory(&nodes, &descriptors, required, &[]).unwrap();
    assert_eq!(plan.minimum_step_bytes(), required);
    for original in &descriptors {
        assert!(plan.dynamic_descriptors().contains(original));
    }
}

#[test]
fn coalescing_preserves_completion_order_and_retention() {
    let (nodes, descriptors) = aligned_fixture(4096);
    let retained = memory(&nodes, &descriptors, 16384, &["large"]).unwrap();
    let retained_id = ResourceId::new("resource/large").unwrap();
    let retained_slot = retained
        .dynamic_pools()
        .iter()
        .flat_map(|p| p.step_resource_slots())
        .find(|slot| slot.resource_ids().contains(&retained_id))
        .unwrap();
    assert_eq!(retained_slot.resource_ids(), &[retained_id]);
    let mut independent_nodes = nodes;
    independent_nodes[3].dependencies.clear();
    let independent = memory(&independent_nodes, &descriptors, 16384, &[]).unwrap();
    let large = ResourceId::new("resource/large").unwrap();
    let later = ResourceId::new("resource/later").unwrap();
    for slot in independent
        .dynamic_pools()
        .iter()
        .flat_map(|p| p.step_resource_slots())
    {
        assert!(!(slot.resource_ids().contains(&large) && slot.resource_ids().contains(&later)));
    }
}

#[test]
fn variable_demands_retain_their_original_alignment_contracts() {
    let (nodes, mut descriptors) = aligned_fixture(4096);
    descriptors[2].demand = DynamicResourceDemand::tokens(4096, 32).unwrap();
    let plan = memory(&nodes, &descriptors, 16384, &[]).unwrap();
    for original in &descriptors {
        assert!(plan.dynamic_descriptors().contains(original));
    }
}
