use super::*;

#[test]
fn pool_cow_failed_allocation_preserves_original_search_counter_and_layout() {
    let original = state();
    let mut branch = original.clone();
    let before = original.pools[0].allocator.search_probes;
    let mut independent = (*original.pools[0].allocator).clone();
    let id = branch.pools[0].id.clone();
    let expected = independent.allocate_contiguous(&id, 256).unwrap();
    let actual = Arc::make_mut(&mut branch.pools[0].allocator)
        .allocate_contiguous(&id, 256)
        .unwrap();
    assert_eq!(actual, expected);
    assert!(actual.is_none());
    assert_eq!(original.pools[0].allocator.search_probes, before);
    assert_eq!(
        branch.pools[0].allocator.search_probes,
        independent.search_probes
    );
    assert_eq!(branch.pools[0].allocator.search_probes, before + 1);
    assert_eq!(branch.pools[0].allocator.by_offset, independent.by_offset);
    assert_eq!(branch.pools[0].allocator.by_size, independent.by_size);
    assert!(!Arc::ptr_eq(
        &original.pools[0].allocator,
        &branch.pools[0].allocator
    ));
}

fn state() -> ResourcePlanningState {
    let id = serde_json::from_value(serde_json::json!(format!(
        "dynamic-pool/sha256/{}",
        "a".repeat(64)
    )))
    .unwrap();
    let mut allocator = FreeExtentIndex::default();
    allocator.insert_extent(1, 1, 0, 64).unwrap();
    allocator.insert_extent(1, 1, 128, 64).unwrap();
    ResourcePlanningState {
        fence: Arc::new(()),
        pools: vec![PoolReadView {
            id,
            instance: 1,
            next_extent_generation: 2,
            resident_bytes: 256,
            allocator: Arc::new(allocator),
        }],
        workspace: None,
        logical_available: BTreeMap::from([(CapacityDomainId::new(1).unwrap(), 128)]),
        covered: vec![DynamicResourceShape::from_validated(1, 1, 0)],
        sequence_ranges: vec![Arc::new(BTreeMap::new())],
        checkpoint_retained_bytes: None,
        checkpoint_tokens: Vec::new(),
        waves: 1,
    }
}

#[test]
fn successor_equality_keeps_allocator_layout_and_capture_authority() {
    let original = state();
    assert!(original
        .same_future_state(&original.clone(), &mut || true)
        .unwrap());
    let mut rearranged = original.clone();
    let mut allocator = FreeExtentIndex::default();
    allocator.insert_extent(1, 1, 64, 64).unwrap();
    allocator.insert_extent(1, 1, 192, 64).unwrap();
    rearranged.pools[0].allocator = Arc::new(allocator);
    assert_eq!(
        original.pools[0].allocator.free_bytes,
        rearranged.pools[0].allocator.free_bytes
    );
    assert!(!original
        .same_future_state(&rearranged, &mut || true)
        .unwrap());
    let mut recaptured = original.clone();
    recaptured.fence = Arc::new(());
    assert!(!original
        .same_future_state(&recaptured, &mut || true)
        .unwrap());
    let mut changed_generation = original.clone();
    changed_generation.pools[0].next_extent_generation += 1;
    assert!(!original
        .same_future_state(&changed_generation, &mut || true)
        .unwrap());
    let mut changed_capacity = original.clone();
    *changed_capacity
        .logical_available
        .values_mut()
        .next()
        .unwrap() -= 1;
    assert!(!original
        .same_future_state(&changed_capacity, &mut || true)
        .unwrap());
    let mut changed_coverage = original.clone();
    changed_coverage.covered[0] = DynamicResourceShape::from_validated(1, 2, 0);
    assert!(!original
        .same_future_state(&changed_coverage, &mut || true)
        .unwrap());
    let mut changed_ranges = original.clone();
    Arc::make_mut(&mut changed_ranges.sequence_ranges[0]).insert(
        ResourceId::new("sequence.window").unwrap(),
        Arc::new(vec![BackingSegment::from_chunk(
            &original.pools[0].id,
            1,
            1,
            64,
            64,
        )
        .unwrap()]),
    );
    assert!(!original
        .same_future_state(&changed_ranges, &mut || true)
        .unwrap());
}

#[test]
fn successor_equality_polls_inside_allocator_and_sequence_collections() {
    let original = state();
    let mut polls = 0;
    assert_eq!(
        original.same_future_state(&original.clone(), &mut || {
            polls += 1;
            polls < 5
        }),
        Err(ResourcePlanningUnknown::BudgetExhausted)
    );
    // Independent immutable maps take the content path, without pointer reuse.
    let mut copied = original.clone();
    copied.sequence_ranges = original
        .sequence_ranges
        .iter()
        .map(|row| Arc::new((**row).clone()))
        .collect();
    assert!(original.same_future_state(&copied, &mut || true).unwrap());
}
