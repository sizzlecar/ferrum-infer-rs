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
            &crate::vnext::DisabledSubmissionWaveDispatchTimingSink,
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
            &crate::vnext::DisabledSubmissionWaveDispatchTimingSink,
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
