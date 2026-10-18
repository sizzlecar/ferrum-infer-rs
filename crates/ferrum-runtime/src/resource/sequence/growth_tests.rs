use super::*;
use std::num::NonZeroU32;

fn grow(
    session: &Arc<SequenceSession<TestRuntime>>,
    tokens: usize,
    grouped: bool,
) -> Arc<SequenceBackingSnapshot<TestRuntime>> {
    let request = SequenceResourceExtensionRequest::new(
        work(tokens),
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let request = if grouped {
        request.with_paged_growth_group(NonZeroU32::new(4).unwrap())
    } else {
        request
    };
    match session.try_ensure_backing_covers(request).unwrap() {
        SequenceResourceExtensionDecision::Extended(snapshot)
        | SequenceResourceExtensionDecision::Current(snapshot) => snapshot,
        _ => panic!("required resident capacity must fit"),
    }
}

fn owned(snapshot: &SequenceBackingSnapshot<TestRuntime>) -> u64 {
    snapshot
        .backing_slices()
        .iter()
        .map(|s| s.size_bytes())
        .sum()
}

fn grow_only_request(tokens: usize) -> SequenceResourceExtensionRequest {
    SequenceResourceExtensionRequest::new(work(tokens), AdmissionPressureAction::WaitForRelease)
        .unwrap()
        .with_paged_growth_group_grow_only(NonZeroU32::new(4).unwrap())
}

#[test]
fn paged_growth_grow_only_reserves_a_cold_pool_without_advancing_actual_work() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let h = harness(runtime, catalog, 256, false);
    h.root
        .maintenance_controller
        .initialize_pool(&h.pool_ids[0])
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "cold-grow-only", 4);
    let session = sequence.open_session().unwrap();
    let allocations = h.runtime.allocate_calls();
    let old = session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(2), AdmissionPressureAction::WaitForRelease)
                .unwrap()
                .with_paged_growth_group(NonZeroU32::new(4).unwrap()),
        )
        .unwrap();
    assert!(matches!(
        old,
        SequenceResourceExtensionDecision::BackingDeferred(_)
    ));
    drop(old);
    assert_eq!(h.runtime.allocate_calls(), allocations);
    let next = match session
        .try_ensure_backing_covers(grow_only_request(2))
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(next) => next,
        _ => panic!("budgeted page-group growth must admit from a cold pool"),
    };
    assert_eq!(next.committed_tokens(), 2);
    assert_eq!(owned(&next), 256);
    assert_eq!(h.runtime.allocate_calls(), allocations + 1);
    h.runtime
        .physical_binding_tables_enabled
        .store(true, Ordering::Relaxed);
    let table = next
        .physical_binding_table(
            &h.root.dynamic_pools,
            h.runtime.as_ref(),
            &crate::DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap()
        .unwrap();
    let last = match session
        .try_ensure_backing_covers(grow_only_request(4))
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(next) => next,
        _ => panic!("owned surplus must cover later real work"),
    };
    assert_eq!(last.committed_tokens(), 4);
    assert!(next.shares_physical_catalog(&last));
    let last_table = last
        .physical_binding_table(
            &h.root.dynamic_pools,
            h.runtime.as_ref(),
            &crate::DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&table, &last_table));
    assert_eq!(h.runtime.allocate_calls(), allocations + 1);
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop((last_table, table, last, next, session, sequence));
    close_dynamic_test_root(h.root);
}

#[test]
fn paged_growth_grow_only_budget_denial_preserves_idle_chunks_and_uses_exact() {
    let sequence_catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let idle_catalog = pool_catalog(
        linear_profile(),
        AllocationLifetime::Step,
        'b',
        1,
        128,
        TestDemand::Fixed,
    );
    let catalog = combine_catalogs(&[sequence_catalog.clone(), idle_catalog.clone()]);
    let runtime = new_runtime(&catalog, 256);
    let h = harness(runtime, catalog, 256, false);
    for pool_id in [&sequence_catalog.pool_id, &idle_catalog.pool_id] {
        h.root
            .maintenance_controller
            .grow_pool(pool_id, 128)
            .unwrap();
    }
    let sequence = admitted_sequence_with_ceiling(&h.root, "budgeted-surplus", 4);
    let session = sequence.open_session().unwrap();
    let before = h.root.dynamic_pool_status().unwrap();
    let allocations = h.runtime.allocate_calls();
    let next = match session
        .try_ensure_backing_covers(grow_only_request(2))
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(next) => next,
        _ => panic!("required work must fit without reclaiming optional capacity"),
    };
    assert_eq!(next.committed_tokens(), 2);
    assert_eq!(owned(&next), 128);
    assert_eq!(h.runtime.allocate_calls(), allocations);
    let after = h.root.dynamic_pool_status().unwrap();
    for original in before.pools() {
        let current = after
            .pools()
            .iter()
            .find(|p| p.pool_id() == original.pool_id())
            .unwrap();
        assert_eq!(current.resident_bytes(), original.resident_bytes());
        assert_eq!(current.resident_chunks(), original.resident_chunks());
        assert_eq!(current.pending_growth_bytes(), 0);
        if original.pool_id() == &idle_catalog.pool_id {
            assert_eq!(current, original);
        }
    }
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop((next, session, sequence));
    close_dynamic_test_root(h.root);
}

#[test]
fn paged_growth_grow_only_pool_ceiling_denial_uses_exact_without_allocating() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        128,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let h = harness(runtime, catalog, 256, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 128)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "pool-ceiling-surplus", 4);
    let session = sequence.open_session().unwrap();
    let allocations = h.runtime.allocate_calls();
    let next = match session
        .try_ensure_backing_covers(grow_only_request(2))
        .unwrap()
    {
        SequenceResourceExtensionDecision::Extended(next) => next,
        _ => panic!("an optional pool-ceiling denial must preserve required work"),
    };
    assert_eq!(owned(&next), 128);
    assert_eq!(h.runtime.allocate_calls(), allocations);
    let status = h.root.maintenance_controller.status().unwrap();
    assert_eq!(status.budget_claimed_bytes(), 128);
    assert_eq!(status.pools()[0].pending_growth_bytes(), 0);
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop((next, session, sequence));
    close_dynamic_test_root(h.root);
}

#[test]
fn paged_growth_grow_only_allocation_error_is_not_hidden_by_exact_fallback() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let h = harness(runtime, catalog, 256, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 128)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "allocation-error-surplus", 4);
    let session = sequence.open_session().unwrap();
    let before = sequence.backing_snapshot().unwrap();
    h.runtime.fail_on_call(h.runtime.allocate_calls() + 1);
    assert!(session
        .try_ensure_backing_covers(grow_only_request(2))
        .is_err());
    let after = sequence.backing_snapshot().unwrap();
    assert!(Arc::ptr_eq(&before, &after));
    let status = h.root.maintenance_controller.status().unwrap();
    assert_eq!(status.budget_claimed_bytes(), 128);
    assert_eq!(status.pools()[0].pending_growth_bytes(), 0);
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop((after, before, session, sequence));
    close_dynamic_test_root(h.root);
}

#[test]
fn paged_growth_keeps_actual_frontier_and_reuses_owned_capacity() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let h = harness(runtime, catalog, 256, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 256)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "grouped-growth", 4);
    let session = sequence.open_session().unwrap();
    let initial = sequence.backing_snapshot().unwrap();
    let reserved = grow(&session, 2, true);
    assert_eq!(reserved.committed_tokens(), 2);
    assert_eq!(owned(&reserved), 256);
    assert!(!reserved.shares_physical_catalog(&initial));
    h.runtime
        .physical_binding_tables_enabled
        .store(true, Ordering::Relaxed);
    let table = reserved
        .physical_binding_table(
            &h.root.dynamic_pools,
            h.runtime.as_ref(),
            &crate::DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap()
        .unwrap();
    let lane = h.root.create_execution_lane().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let step = match batch
        .try_begin_step(
            StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![token_span(2)]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            &lane,
        )
        .unwrap()
    {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident Sequence capacity must admit its fresh frame"),
    };
    assert!(matches!(
        session
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    work(3),
                    AdmissionPressureAction::WaitForRelease
                )
                .unwrap()
                .with_paged_growth_group(NonZeroU32::new(4).unwrap()),
            )
            .unwrap(),
        SequenceResourceExtensionDecision::RetryRequired(_)
    ));
    step.try_rollback_unsubmitted().unwrap();
    drop(batch);
    // Exact requests also consume existing owned capacity rather than claiming
    // the same bytes a second time after the opt-in reservation has succeeded.
    let third = grow(&session, 3, false);
    let fourth = grow(&session, 4, true);
    assert_eq!(third.committed_tokens(), 3);
    assert_eq!(fourth.committed_tokens(), 4);
    assert_eq!(owned(&fourth), 256);
    assert!(reserved.shares_physical_catalog(&third));
    assert!(reserved.shares_physical_catalog(&fourth));
    let last_table = fourth
        .physical_binding_table(
            &h.root.dynamic_pools,
            h.runtime.as_ref(),
            &crate::DisabledSubmissionWaveDispatchTimingSink,
        )
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&table, &last_table));
    session.request_cancel().unwrap();
    assert!(session
        .try_ensure_backing_covers(
            SequenceResourceExtensionRequest::new(work(4), AdmissionPressureAction::WaitForRelease)
                .unwrap()
                .with_paged_growth_group(NonZeroU32::new(4).unwrap()),
        )
        .is_err());
    session.try_abort().unwrap();
    drop((
        last_table, table, fourth, third, reserved, initial, session, sequence, lane,
    ));
    assert!(h
        .root
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .all(|p| p.free_bytes() == p.resident_bytes()));
    close_dynamic_test_root(h.root);
}

#[test]
fn paged_growth_surplus_shortage_falls_back_without_pool_growth() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        256,
        TestDemand::Tokens,
    );
    let runtime = new_runtime(&catalog, 256);
    let h = harness(Arc::clone(&runtime), catalog, 256, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 128)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "grouped-shortage", 4);
    let session = sequence.open_session().unwrap();
    let allocations = runtime.allocate_calls.load(Ordering::Acquire);
    let next = grow(&session, 2, true);
    assert_eq!(next.committed_tokens(), 2);
    assert_eq!(owned(&next), 128);
    assert_eq!(runtime.allocate_calls.load(Ordering::Acquire), allocations);
    assert_eq!(
        h.root.dynamic_pool_status().unwrap().pools()[0].resident_bytes(),
        128
    );
    session.request_cancel().unwrap();
    session.try_abort().unwrap();
    drop((next, session, sequence));
    close_dynamic_test_root(h.root);
}

#[test]
fn paged_growth_clamps_to_parent_and_exact_remains_default() {
    for (grouped, ceiling, expected_bytes) in [(false, 4, 128), (true, 3, 192)] {
        let catalog = pool_catalog(
            paged_profile(),
            AllocationLifetime::Sequence,
            'a',
            1,
            256,
            TestDemand::Tokens,
        );
        let runtime = new_runtime(&catalog, 256);
        let h = harness(runtime, catalog, 256, false);
        h.root
            .maintenance_controller
            .grow_pool(&h.pool_ids[0], 256)
            .unwrap();
        let sequence = admitted_sequence_with_ceiling(&h.root, "grouped-ceiling", ceiling);
        let session = sequence.open_session().unwrap();
        let next = grow(&session, 2, grouped);
        assert_eq!(next.committed_tokens(), 2);
        assert_eq!(owned(&next), expected_bytes);
        assert!(session
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    work(ceiling + 1),
                    AdmissionPressureAction::WaitForRelease
                )
                .unwrap()
                .with_paged_growth_group(NonZeroU32::new(4).unwrap()),
            )
            .is_err());
        session.request_cancel().unwrap();
        session.try_abort().unwrap();
        drop((next, session, sequence));
        close_dynamic_test_root(h.root);
    }
}
