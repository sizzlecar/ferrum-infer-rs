use super::*;

fn memory_with_ids(ids: &[&str]) -> Result<MemoryPlan, VNextError> {
    let storage = DynamicStorageContract::new(
        linear_profile(),
        canonical_fingerprint(&"descriptor-lookup", "test layout").unwrap(),
    )
    .unwrap();
    MemoryPlan::from_core(
        1 << 24,
        1 << 24,
        0,
        16,
        vec![],
        ids.iter()
            .map(|id| dynamic_value_descriptor(id, storage.clone()))
            .collect(),
        &[],
        None,
    )
}

fn check_lookup(plan: &MemoryPlan, present: &[&str]) {
    for name in present {
        let id = ResourceId::new(*name).unwrap();
        let expected = plan
            .dynamic_descriptors()
            .iter()
            .find(|descriptor| descriptor.base_resource_id() == &id)
            .unwrap();
        assert!(
            std::ptr::eq(plan.dynamic_descriptor(&id).unwrap(), expected),
            "lookup must borrow the exact descriptor from this immutable plan"
        );
    }
    // Cover lower/upper misses, sparse interior keys and string ordering.
    for name in ["resource/00", "resource/11", "resource/3", "resource/zz"] {
        assert!(plan
            .dynamic_descriptor(&ResourceId::new(name).unwrap())
            .is_none());
    }
}

#[test]
fn memory_dynamic_descriptor_lookup_preserves_identity_for_constructed_and_loaded_plans() {
    for ids in [
        &[][..],
        &["resource/2"][..],
        &[
            "resource/z",
            "resource/2",
            "resource/10",
            "resource/01",
            "resource/a",
        ][..],
    ] {
        // The constructor canonicalizes unordered producer input; loading
        // validates that same immutable contract without resorting wire rows.
        let plan = memory_with_ids(ids).unwrap();
        check_lookup(&plan, ids);
        let bytes = serde_json::to_vec(&plan).unwrap();
        let loaded: MemoryPlan = serde_json::from_slice(&bytes).unwrap();
        check_lookup(&loaded, ids);
        assert_eq!(serde_json::to_vec(&loaded).unwrap(), bytes);
    }
}

#[test]
fn memory_dynamic_descriptor_lookup_requires_canonical_unique_wire_and_constructor_rows() {
    assert!(memory_with_ids(&["resource/2", "resource/2"]).is_err());
    let plan = memory_with_ids(&["resource/z", "resource/2", "resource/10"]).unwrap();
    let wire = serde_json::to_value(&plan).unwrap();
    let mut unsorted = wire.clone();
    unsorted["dynamic_descriptors"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(serde_json::from_value::<MemoryPlan>(unsorted).is_err());
    let mut duplicate = wire;
    duplicate["dynamic_descriptors"][1] = duplicate["dynamic_descriptors"][0].clone();
    assert!(serde_json::from_value::<MemoryPlan>(duplicate).is_err());
}
