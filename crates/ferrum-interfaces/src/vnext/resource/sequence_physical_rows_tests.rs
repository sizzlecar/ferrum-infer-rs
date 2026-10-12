//! Real sequence admission and the legacy full-permit oracle.
use super::*;
use crate::vnext::operation::DisabledSubmissionWaveDispatchTimingSink;
use crate::vnext::{
    ApprovedSequenceBindingPart, SegmentBindingOwnerBlock, SegmentBindingOwnerSources,
    SegmentBindingPageAddressLayout, SegmentBindingPageTemplate, SegmentBindingPageTemplateKey,
    SegmentBindingTemplatePage,
};

fn sequence_row_harness(
    demand: TestDemand,
    suffix: &str,
    resident_bytes: u64,
) -> (Harness, Arc<AdmittedSequenceResources<TestRuntime>>) {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        demand,
    );
    let runtime = new_runtime(&catalog, 256);
    let harness = harness(runtime, catalog, 256, false);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], resident_bytes)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&harness.root, suffix, 3);
    (harness, sequence)
}

fn expectation(
    snapshot: &SequenceBackingSnapshot<TestRuntime>,
    resource: &ResourceId,
    view_bytes: u64,
) -> SegmentBackingExpectation {
    let first = &snapshot.backing_slices_for(resource)[0].evidence;
    SegmentBackingExpectation {
        logical_bytes: view_bytes,
        logical_size_rule: SegmentLogicalSizeRule::AtLeast,
        usage: first.usage,
        storage_profile: first.storage_profile,
        element_type: first.element_type,
        alignment_bytes: first.alignment_bytes,
    }
}

fn window(expected: SegmentBackingExpectation) -> SegmentBackingWindow {
    SegmentBackingWindow {
        resource_index: 0,
        offset_bytes: 0,
        length_bytes: expected.logical_bytes,
        element_type: expected.element_type,
        alignment_bytes: expected.alignment_bytes,
    }
}

fn cached_batch(
    pools: &DynamicPoolSet<TestRuntime>,
    snapshot: &SequenceBackingSnapshot<TestRuntime>,
    resource: &ResourceId,
    expected: SegmentBackingExpectation,
    windows: &[SegmentBackingWindow],
) -> Result<SegmentBackingBatch<<TestRuntime as DeviceRuntime>::Buffer>, VNextError> {
    let source = snapshot.segment_physical_row(resource).unwrap();
    pools.segment_backing_batch_with_sources(
        &[SegmentBackingSource::Sequence(source)],
        &[expected],
        SegmentBackingWindowInput::Slice(windows),
        &DisabledSubmissionWaveDispatchTimingSink,
    )
}

fn assert_same_batch(
    a: &SegmentBackingBatch<<TestRuntime as DeviceRuntime>::Buffer>,
    b: &SegmentBackingBatch<<TestRuntime as DeviceRuntime>::Buffer>,
    windows: usize,
) {
    assert_eq!(a.resources().len(), b.resources().len());
    for (a, b) in a.resources().iter().zip(b.resources()) {
        assert_eq!(
            (
                a.logical_size_bytes(),
                a.capacity_size_bytes(),
                a.view_size_bytes(),
                a.alignment_bytes(),
                a.usage(),
                a.element_type(),
                a.storage_profile()
            ),
            (
                b.logical_size_bytes(),
                b.capacity_size_bytes(),
                b.view_size_bytes(),
                b.alignment_bytes(),
                b.usage(),
                b.element_type(),
                b.storage_profile()
            )
        );
    }
    assert_eq!(a.bindings().len(), b.bindings().len());
    for (a, b) in a.bindings().iter().zip(b.bindings().iter()) {
        assert_eq!(a.segment(), b.segment());
        assert!(std::ptr::eq(a.buffer(), b.buffer()));
        assert!(a.retention_ref().same_owners(b.retention_ref()));
    }
    for index in 0..windows {
        let a = a.window(index).unwrap();
        let b = b.window(index).unwrap();
        assert_eq!(a.region_count(), b.region_count());
        for (a, b) in a.physical_regions().zip(b.physical_regions()) {
            let (ab, ar, ao) = a.borrowed_buffer_and_physical_range();
            let (bb, br, bo) = b.borrowed_buffer_and_physical_range();
            assert!(std::ptr::eq(ab, bb));
            assert_eq!(ar, br);
            assert!(ao.same_owners(bo));
            assert_eq!(a.logical_offset_bytes(), b.logical_offset_bytes());
            assert_eq!(a.length_bytes(), b.length_bytes());
            assert_eq!(a.descriptor(), b.descriptor());
        }
    }
}

fn paired_batch(
    pools: &DynamicPoolSet<TestRuntime>,
    snapshot: &SequenceBackingSnapshot<TestRuntime>,
    resource: &ResourceId,
    view_bytes: u64,
) -> SegmentBackingBatch<<TestRuntime as DeviceRuntime>::Buffer> {
    let expected = expectation(snapshot, resource, view_bytes);
    let windows = [window(expected)];
    let legacy = pools
        .segment_backing_batch(
            &[snapshot.backing_slices_for(resource)],
            &[expected],
            &windows,
        )
        .unwrap();
    let candidate = cached_batch(pools, snapshot, resource, expected, &windows).unwrap();
    assert_same_batch(&legacy, &candidate, windows.len());
    candidate
}

// These deterministic host addresses exercise only Core's scalar geometry.
// The CUDA cold path separately checks actual native address derivation.
fn scalar_template(
    part: &ApprovedSequenceBindingPart<'_, TestBuffer>,
    key: SegmentBindingPageTemplateKey,
) -> SegmentBindingPageTemplate {
    let mut pages = Vec::new();
    let mut addresses = Vec::new();
    for local_owner in 0..part.owner_count() {
        let range = part.physical_range(local_owner).unwrap();
        let mut offset = range.start;
        while offset < range.end {
            let address = 0x1000 + pages.len() as u64 * 0x100;
            pages.push(SegmentBindingTemplatePage {
                local_owner,
                physical_offset_bytes: offset,
                length_bytes: key.physical_page_bytes,
                address,
            });
            match key.address_layout {
                SegmentBindingPageAddressLayout::Flat => addresses.push(address),
                SegmentBindingPageAddressLayout::Blocked {
                    combined_block_bytes,
                    blocks_per_page,
                } => {
                    addresses.extend(
                        (0..blocks_per_page).map(|block| address + block * combined_block_bytes),
                    );
                }
            }
            offset += key.physical_page_bytes;
        }
    }
    SegmentBindingPageTemplate {
        key,
        pages,
        addresses,
        physical_bytes: part.physical_bytes(),
    }
}

#[test]
fn sequence_physical_rows_page_memo_is_scalar_sealed_and_shared_with_exact_owners() {
    let (harness, sequence) = sequence_row_harness(TestDemand::Tokens, "physical-row-memo", 64);
    let initial = sequence.backing_snapshot().unwrap();
    let resource = initial.backing_slices()[0].resource_id().clone();
    let pools = &harness.root.dynamic_pools;
    let first = paired_batch(pools, &initial, &resource, 64);
    let approved = first.window(0).unwrap().shared_sequence_window(2).unwrap();
    let part = approved.parts().next().unwrap();
    let key = SegmentBindingPageTemplateKey {
        physical_page_bytes: 16,
        element_type: expectation(&initial, &resource, 64).element_type,
        address_layout: SegmentBindingPageAddressLayout::Blocked {
            combined_block_bytes: 8,
            blocks_per_page: 2,
        },
        runtime_instance: 1,
        provider_fingerprint: [3; 32],
        numerical_profile: [5; 32],
    };
    let mutations: [fn(&mut SegmentBindingPageTemplate); 10] = [
        |template: &mut SegmentBindingPageTemplate| {
            template.pages[0].local_owner += 1;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.pages[0].physical_offset_bytes += 16;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.pages[0].length_bytes -= 1;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.addresses[1] += 1;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.physical_bytes -= 16;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.pages.pop();
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.key.physical_page_bytes = 0;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.key.element_type = if template.key.element_type == ElementType::F32 {
                ElementType::U8
            } else {
                ElementType::F32
            };
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.pages[0].address = u64::MAX;
        },
        |template: &mut SegmentBindingPageTemplate| {
            template.key.address_layout = SegmentBindingPageAddressLayout::Blocked {
                combined_block_bytes: u64::MAX,
                blocks_per_page: 2,
            };
        },
    ];
    for mutate in mutations {
        let mut invalid = scalar_template(&part, key);
        mutate(&mut invalid);
        assert!(!part.publish_page_template(Arc::new(invalid)));
        assert!(part.page_template(&key).is_none());
    }
    let template = Arc::new(scalar_template(&part, key));
    assert!(part.publish_page_template(Arc::clone(&template)));
    assert!(part.same_page_template(&template));
    assert!(part.matches_page_template(&template));
    assert!(!part.matches_page_template(&scalar_template(&part, key)));
    assert!(!part.publish_page_template(Arc::new(scalar_template(&part, key))));
    assert!(Arc::ptr_eq(&part.page_template(&key).unwrap(), &template));
    for changed in [
        SegmentBindingPageTemplateKey {
            runtime_instance: 2,
            ..key
        },
        SegmentBindingPageTemplateKey {
            provider_fingerprint: [4; 32],
            ..key
        },
        SegmentBindingPageTemplateKey {
            numerical_profile: [6; 32],
            ..key
        },
        SegmentBindingPageTemplateKey {
            address_layout: SegmentBindingPageAddressLayout::Flat,
            ..key
        },
        SegmentBindingPageTemplateKey {
            physical_page_bytes: 32,
            ..key
        },
    ] {
        assert!(part.page_template(&changed).is_none());
    }
    let lease = &initial.backing_slices()[0].segment_lease;
    let probe = Arc::downgrade(lease);
    let counts = (
        Arc::strong_count(lease),
        Arc::strong_count(&first.bindings()[0].chunk),
    );
    let mut sources = SegmentBindingOwnerSources::new();
    sources
        .push(SegmentBindingOwnerBlock::Borrowed(vec![approved
            .shared_block()
            .owner(0)
            .unwrap()]))
        .unwrap();
    sources
        .push(SegmentBindingOwnerBlock::Borrowed(Vec::new()))
        .unwrap();
    sources
        .push(SegmentBindingOwnerBlock::SharedSequence(
            approved.shared_block().clone(),
        ))
        .unwrap();
    assert_eq!(sources.block_count(), 3);
    assert_eq!(sources.resolve(0).unwrap().block(), 0);
    assert_eq!(
        (
            sources.resolve(1).unwrap().block(),
            sources.resolve(1).unwrap().local()
        ),
        (2, 0)
    );
    let copied = sources.clone();
    let (_, retained) = copied.get(1).unwrap().buffer_and_retention();
    assert!(retained.same_owners(first.bindings()[0].retention_ref()));
    assert_eq!(
        (
            Arc::strong_count(lease),
            Arc::strong_count(&first.bindings()[0].chunk)
        ),
        counts
    );
    drop(copied);
    drop(sources);
    harness
        .root
        .maintenance_controller
        .grow_pool(&harness.pool_ids[0], 64)
        .unwrap();
    let session = sequence.open_session().unwrap();
    let extended = match session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap(),
        )
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
        _ => panic!("real append must add a physical tail"),
    };
    let next = paired_batch(pools, &extended, &resource, 128);
    let current = next.window(0).unwrap().shared_sequence_window(0).unwrap();
    let mut parts = current.parts();
    let prefix = parts.next().unwrap();
    let tail = parts.next().unwrap();
    assert!(prefix.same_page_template(&template));
    assert!(tail.page_template(&key).is_none());
    assert!(!tail.matches_page_template(&template));
    assert!(!prefix
        .shared_block()
        .same_owner_block(approved.shared_block()));
    assert!(prefix
        .owner(0)
        .unwrap()
        .buffer_and_retention()
        .1
        .same_owners(
            approved
                .shared_block()
                .owner(0)
                .unwrap()
                .buffer_and_retention()
                .1,
        ));
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop(parts);
    drop(next);
    drop(first);
    drop(extended);
    drop(initial);
    drop(session);
    drop(sequence);
    assert!(
        probe.upgrade().is_none(),
        "retained scalar template must not pin a lease"
    );
    assert_eq!(template.pages.len(), 4);
    close_dynamic_test_root(harness.root);
}

#[test]
fn sequence_physical_rows_reuse_bindings_with_fresh_logical_windows_and_release_owners() {
    let (harness, sequence) = sequence_row_harness(TestDemand::Tokens, "physical-row-windows", 64);
    let snapshot = sequence.backing_snapshot().unwrap();
    let resource = snapshot.backing_slices()[0].resource_id().clone();
    let pools = &harness.root.dynamic_pools;
    assert_eq!(
        snapshot
            .segment_physical_row(&resource)
            .unwrap()
            .cached_binding_count(),
        None
    );
    let first = paired_batch(pools, &snapshot, &resource, 64);
    assert_eq!(
        snapshot
            .segment_physical_row(&resource)
            .unwrap()
            .cached_binding_count(),
        Some(1)
    );
    let lease = &snapshot.backing_slices()[0].segment_lease;
    let lease_probe = Arc::downgrade(lease);
    let chunk_probe = Arc::downgrade(&first.bindings()[0].chunk);
    let counts = (
        Arc::strong_count(lease),
        Arc::strong_count(&first.bindings()[0].chunk),
    );
    let short = paired_batch(pools, &snapshot, &resource, 32);
    assert!(std::ptr::eq(&first.bindings()[0], &short.bindings()[0]));
    assert_eq!(
        (
            Arc::strong_count(lease),
            Arc::strong_count(&first.bindings()[0].chunk)
        ),
        counts
    );
    let region = short.window(0).unwrap().physical_region(0).unwrap();
    let (_, range, _) = region.borrowed_buffer_and_physical_range();
    assert_eq!(range.end - range.start, 32);
    assert_eq!(
        (
            Arc::strong_count(lease),
            Arc::strong_count(&first.bindings()[0].chunk)
        ),
        counts
    );
    let longer = paired_batch(pools, &snapshot, &resource, 48);
    assert!(std::ptr::eq(&first.bindings()[0], &longer.bindings()[0]));
    let narrow = expectation(&snapshot, &resource, 32);
    let exact = SegmentBackingExpectation {
        logical_size_rule: SegmentLogicalSizeRule::Exact,
        ..narrow
    };
    assert!(cached_batch(pools, &snapshot, &resource, exact, &[window(exact)]).is_err());
    let overflow = SegmentBackingWindow {
        offset_bytes: u64::MAX,
        ..window(narrow)
    };
    let misaligned = SegmentBackingWindow {
        alignment_bytes: 3,
        ..window(narrow)
    };
    assert!(cached_batch(pools, &snapshot, &resource, narrow, &[overflow]).is_err());
    assert!(cached_batch(pools, &snapshot, &resource, narrow, &[misaligned]).is_err());
    let expected = [narrow, expectation(&snapshot, &resource, 48)];
    let windows = [
        window(expected[0]),
        SegmentBackingWindow {
            resource_index: 1,
            offset_bytes: 16,
            length_bytes: 16,
            ..window(expected[1])
        },
    ];
    let groups = [snapshot.backing_slices_for(&resource); 2];
    let oracle = pools
        .segment_backing_batch(&groups, &expected, &windows)
        .unwrap();
    let sources = [
        SegmentBackingSource::Sequence(snapshot.segment_physical_row(&resource).unwrap()),
        SegmentBackingSource::Sequence(snapshot.segment_physical_row(&resource).unwrap()),
    ];
    let duplicate = pools
        .segment_backing_batch_with_sources(
            &sources,
            &expected,
            SegmentBackingWindowInput::Slice(&windows),
            &DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap();
    assert_same_batch(&oracle, &duplicate, windows.len());
    assert_eq!(
        (duplicate.resources().len(), duplicate.bindings().len()),
        (2, 1)
    );
    assert!(std::ptr::eq(&first.bindings()[0], &duplicate.bindings()[0]));
    assert_eq!(
        (
            duplicate.resource(0).unwrap().view_size_bytes(),
            duplicate.resource(1).unwrap().view_size_bytes()
        ),
        (32, 48)
    );
    drop(duplicate);
    drop(oracle);
    drop(longer);
    drop(short);
    drop(first);
    drop(sequence);
    drop(snapshot);
    assert!(
        lease_probe.upgrade().is_none(),
        "snapshot cache must not cycle back to its owner"
    );
    let released = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    assert!(!released.poisoned());
    assert_eq!(
        (
            released.active_child_claims(),
            released.active_sequences(),
            released.active_requests()
        ),
        (0, 0, 0)
    );
    assert!(released.domains().iter().all(|d| d.used().get() == 0));
    close_dynamic_test_root(harness.root);
    assert!(chunk_probe.upgrade().is_none());
}

#[test]
fn sequence_physical_rows_preserve_prefix_across_real_and_empty_extensions() {
    for (demand, added_extent) in [(TestDemand::Tokens, true), (TestDemand::Fixed, false)] {
        let (harness, sequence) = sequence_row_harness(demand, "physical-row-extension", 64);
        let initial = sequence.backing_snapshot().unwrap();
        let resource = initial.backing_slices()[0].resource_id().clone();
        let pools = &harness.root.dynamic_pools;
        let first = paired_batch(pools, &initial, &resource, 64);
        if added_extent {
            harness
                .root
                .maintenance_controller
                .grow_pool(&harness.pool_ids[0], 64)
                .unwrap();
        }
        let session = sequence.open_session().unwrap();
        let extended = match session
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    work(2),
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
            )
            .unwrap()
        {
            SequenceResourceExtensionDecision::Extended(snapshot) => snapshot,
            _ => panic!("actual resident sequence backing must advance"),
        };
        assert_eq!(
            extended
                .segment_physical_row(&resource)
                .unwrap()
                .cached_binding_count(),
            if added_extent { None } else { Some(1) }
        );
        let pool = &pools.pools[&harness.pool_ids[0]];
        if added_extent {
            let tail = extended.backing_slices()[1].evidence.segments[0].chunk_ordinal();
            assert_ne!(
                tail,
                initial.backing_slices()[0].evidence.segments[0].chunk_ordinal()
            );
            let removed = pool.state.lock().unwrap().chunks.remove(&tail).unwrap();
            let expected = expectation(&extended, &resource, 128);
            assert!(
                cached_batch(pools, &extended, &resource, expected, &[window(expected)]).is_err()
            );
            assert_eq!(
                extended
                    .segment_physical_row(&resource)
                    .unwrap()
                    .cached_binding_count(),
                None,
                "a rejected tail must not publish a partial row"
            );
            pool.state.lock().unwrap().chunks.insert(tail, removed);
        }
        let bytes = if added_extent { 128 } else { 64 };
        let next = paired_batch(pools, &extended, &resource, bytes);
        assert!(std::ptr::eq(&first.bindings()[0], &next.bindings()[0]));
        assert_eq!(next.bindings().len(), if added_extent { 2 } else { 1 });
        if added_extent {
            let expected = expectation(&extended, &resource, 128);
            let windows = [(0, 32), (80, 32), (48, 32), (16, 96)].map(|(offset, length)| {
                SegmentBackingWindow {
                    offset_bytes: offset,
                    length_bytes: length,
                    ..window(expected)
                }
            });
            let legacy = pools
                .segment_backing_batch(
                    &[extended.backing_slices_for(&resource)],
                    &[expected],
                    &windows,
                )
                .unwrap();
            let selected = cached_batch(pools, &extended, &resource, expected, &windows).unwrap();
            assert_same_batch(&legacy, &selected, windows.len());
            let parts = |index| {
                selected
                    .window(index)
                    .unwrap()
                    .shared_sequence_window(2)
                    .unwrap()
                    .parts()
                    .map(|part| {
                        (
                            part.owner_block(),
                            part.owner_base(),
                            part.logical_start_in_part(),
                            part.length_bytes(),
                            part.logical_offset_in_window(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            // Legal windows may leave an entire physical part before or after
            // the selected interval; neither excluded part may be evaluated.
            assert_eq!(parts(0), vec![(2, 0, 0, 32, 0)]);
            assert_eq!(parts(1), vec![(2, 1, 16, 32, 0)]);
            assert_eq!(parts(2), vec![(2, 0, 48, 16, 0), (2, 1, 0, 16, 16)]);
            assert_eq!(parts(3), vec![(2, 0, 16, 48, 0), (2, 1, 0, 48, 48)]);
        }
        let current = match session
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    work(1),
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap(),
            )
            .unwrap()
        {
            SequenceResourceExtensionDecision::Current(snapshot) => snapshot,
            _ => panic!("narrow execution must retain the current snapshot"),
        };
        assert!(Arc::ptr_eq(&current, &extended));
        let narrow = paired_batch(pools, &current, &resource, 32);
        assert!(std::ptr::eq(&next.bindings()[0], &narrow.bindings()[0]));
        // This resource-only fixture has submitted no model frame. Cancel the
        // idle session through its normal protocol instead of claiming completion.
        session.request_cancel().unwrap();
        session.try_abort().unwrap();
        drop(narrow);
        drop(next);
        drop(first);
        drop(current);
        drop(extended);
        drop(initial);
        drop(session);
        drop(sequence);
        close_dynamic_test_root(harness.root);
    }
}

#[test]
fn sequence_physical_rows_recheck_live_chunks_poison_and_window_bounds() {
    let (harness, sequence) = sequence_row_harness(TestDemand::Tokens, "physical-row-fresh", 256);
    let (donor, donor_sequence) =
        sequence_row_harness(TestDemand::Tokens, "physical-row-donor", 256);
    let snapshot = sequence.backing_snapshot().unwrap();
    let resource = snapshot.backing_slices()[0].resource_id().clone();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let donor_pool = &donor.root.dynamic_pools.pools[&donor.pool_ids[0]];
    let expected = expectation(&snapshot, &resource, 64);
    let windows = [window(expected)];
    let mut invalid = windows;
    invalid[0].length_bytes += 1;
    assert!(cached_batch(pools, &snapshot, &resource, expected, &invalid).is_err());
    assert_eq!(
        snapshot
            .segment_physical_row(&resource)
            .unwrap()
            .cached_binding_count(),
        None
    );
    let first = paired_batch(pools, &snapshot, &resource, 64);
    pool.state.lock().unwrap().poisoned = true;
    assert!(cached_batch(pools, &snapshot, &resource, expected, &windows).is_err());
    pool.state.lock().unwrap().poisoned = false;
    let ordinal = snapshot.backing_slices()[0].evidence.segments[0].chunk_ordinal();
    let original = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    assert!(cached_batch(pools, &snapshot, &resource, expected, &windows).is_err());
    let mut replacement = donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .remove(&ordinal)
        .unwrap();
    let donor_identity = replacement.backing.identity.clone();
    let donor_descriptor = replacement.backing.descriptor.clone();
    let backing = Arc::get_mut(&mut replacement.backing).unwrap();
    backing.identity = original.backing.identity.clone();
    backing.descriptor = original.backing.descriptor.clone();
    backing.descriptor.size_bytes = 32;
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    assert!(cached_batch(pools, &snapshot, &resource, expected, &windows).is_err());
    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    Arc::get_mut(&mut replacement.backing).unwrap().descriptor =
        original.backing.descriptor.clone();
    pool.state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    let fresh = paired_batch(pools, &snapshot, &resource, 64);
    assert!(fresh.window(0).unwrap().shared_sequence_window(0).is_none());
    assert!(!Arc::ptr_eq(
        &first.bindings()[0].chunk,
        &fresh.bindings()[0].chunk
    ));
    assert!(!std::ptr::eq(&first.bindings()[0], &fresh.bindings()[0]));
    // An independent lease on the same installed chunk is never the cached row's owner.
    let independent = admitted_sequence(&harness.root, "physical-row-independent");
    let independent_snapshot = independent.backing_snapshot().unwrap();
    let independent_view = paired_batch(pools, &independent_snapshot, &resource, 64);
    assert!(std::ptr::eq(
        fresh.bindings()[0].buffer(),
        independent_view.bindings()[0].buffer()
    ));
    assert!(!fresh.bindings()[0]
        .retention_ref()
        .same_owners(independent_view.bindings()[0].retention_ref()));
    drop(independent_view);
    drop(independent_snapshot);
    drop(independent);
    drop(fresh);
    drop(first);
    let mut replacement = pool.state.lock().unwrap().chunks.remove(&ordinal).unwrap();
    pool.state.lock().unwrap().chunks.insert(ordinal, original);
    drop(snapshot);
    drop(sequence);
    // Drop cache ownership before restoring the donor's original descriptor.
    let backing = Arc::get_mut(&mut replacement.backing).unwrap();
    backing.identity = donor_identity;
    backing.descriptor = donor_descriptor;
    donor_pool
        .state
        .lock()
        .unwrap()
        .chunks
        .insert(ordinal, replacement);
    drop(donor_sequence);
    close_dynamic_test_root(harness.root);
    close_dynamic_test_root(donor.root);
}

#[test]
fn sequence_physical_rows_publish_only_after_other_sources_and_all_windows_validate() {
    let (harness, sequence) = sequence_row_harness(TestDemand::Tokens, "physical-row-partial", 256);
    let snapshot = sequence.backing_snapshot().unwrap();
    let resource = snapshot.backing_slices()[0].resource_id().clone();
    let pools = &harness.root.dynamic_pools;
    let pool = &pools.pools[&harness.pool_ids[0]];
    let original = &snapshot.backing_slices()[0];
    let a = &original.evidence.allocation;
    let mut malformed = original.retained();
    // Keep the actual lease. Only its projection is corrupted: this shifted
    // segment still fits the real chunk but does not belong to this lease.
    let segment = &a.segments[0];
    malformed.evidence.allocation = Arc::new(LogicalBackingSliceAllocationEvidence {
        domain_id: a.domain_id,
        pool_id: a.pool_id.clone(),
        resource_id: a.resource_id.clone(),
        pool_instance_id: a.pool_instance_id,
        physical_claim_identity: a.physical_claim_identity.clone(),
        reusable_execution_bucket_id: a.reusable_execution_bucket_id.clone(),
        segment_generation: a.segment_generation,
        segments: vec![BackingSegment::new(
            segment.chunk().clone(),
            segment.offset_bytes() + 16,
            segment.length_bytes(),
        )
        .unwrap()],
        physical_offset_bytes: a.physical_offset_bytes,
        capacity_size_bytes: a.capacity_size_bytes,
        physical_size_bytes: a.physical_size_bytes,
        alignment_bytes: a.alignment_bytes,
        usage: a.usage,
        element_type: a.element_type,
        storage_profile: a.storage_profile,
        initialization: a.initialization,
        fingerprint: a.fingerprint.clone(),
    });
    let expected = expectation(&snapshot, &resource, 64);
    let windows = [
        window(expected),
        SegmentBackingWindow {
            resource_index: 1,
            ..window(expected)
        },
    ];
    let sources = [
        SegmentBackingSource::Sequence(snapshot.segment_physical_row(&resource).unwrap()),
        SegmentBackingSource::Authorities(std::slice::from_ref(&malformed)),
    ];
    assert!(pools
        .segment_backing_batch_with_sources(
            &sources,
            &[expected; 2],
            SegmentBackingWindowInput::Slice(&windows),
            &DisabledSubmissionWaveDispatchTimingSink
        )
        .is_err());
    assert!(pool.state.try_lock().is_ok());
    assert_eq!(
        snapshot
            .segment_physical_row(&resource)
            .unwrap()
            .cached_binding_count(),
        None
    );
    drop(sources);
    let valid = paired_batch(pools, &snapshot, &resource, 64);
    let sources = [
        SegmentBackingSource::Sequence(snapshot.segment_physical_row(&resource).unwrap()),
        SegmentBackingSource::Authorities(std::slice::from_ref(&malformed)),
    ];
    assert!(pools
        .segment_backing_batch_with_sources(
            &sources,
            &[expected; 2],
            SegmentBackingWindowInput::Slice(&windows),
            &DisabledSubmissionWaveDispatchTimingSink
        )
        .is_err());
    assert_eq!(
        snapshot
            .segment_physical_row(&resource)
            .unwrap()
            .cached_binding_count(),
        Some(1)
    );
    assert!(pool.state.try_lock().is_ok());
    drop(sources);
    drop(valid);
    drop(malformed);
    drop(snapshot);
    drop(sequence);
    close_dynamic_test_root(harness.root);
}
