//! CPU differential enumeration uses the actual resource allocator and mask
//! ledger transitions; projection performs no live allocation or provider execution.
use super::*;
use crate::execution_cost::ActualWaveGraphState;
use crate::vnext::{
    ExecutionCostRouteAvailability, ExecutionCostRouteState, ExecutionCostRouteView,
    ProductTokenMaskContent,
};

fn route_view(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    session: &SequenceSession<TestRuntime>,
    lane: &ExecutionLane<TestRuntime>,
    frontier: u64,
) -> ExecutionCostRouteView {
    let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match root.execution_cost_route_view(
            &[session],
            &[frontier],
            lane,
            ResourcePlanningLimits::default(),
            &mut || true,
        ) {
            ExecutionCostRouteAvailability::Known(view) => return view,
            ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
                ResourcePlanningUnknown::ReadUnavailable(_),
            )) if std::time::Instant::now() < until => std::thread::yield_now(),
            other => panic!("route snapshot: {other:?}"),
        }
    }
}

fn advance(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    view: &ExecutionCostRouteView,
    state: &ExecutionCostRouteState,
    content: ProductTokenMaskContent,
) -> ExecutionCostRouteState {
    let projection = match root.project_resource_wave(
        view.resource_view(),
        &state.resources,
        &[row(0, state.frontiers[0], 1)],
        &mut || true,
    ) {
        ResourcePlanningAvailability::Known(projection) => projection,
        ResourcePlanningAvailability::Unknown(reason) => panic!(
            "resource rollout at wave {} frontier {}: {reason:?}",
            state.projected_waves(),
            state.frontiers[0]
        ),
    };
    let mut next = state.clone();
    next.frontiers[0] += 1;
    next.initialized[0] = true;
    next.last_token_mask_uploads = Some(
        next.token_masks
            .as_mut()
            .unwrap()
            .project_contents(projection.selected_step_slot(), 5, &[content], &mut || true)
            .unwrap(),
    );
    next.resources = projection.state;
    next
}

#[test]
fn future_successor_compaction_matches_unmerged_ten_wave_resource_and_mask_rollout() {
    let catalog = pool_catalog(
        paged_profile(),
        AllocationLifetime::Sequence,
        'a',
        1,
        2048,
        TestDemand::TokensThrough(12),
    );
    let runtime = new_runtime(&catalog, 2048);
    let h = harness(Arc::clone(&runtime), catalog, 2048, false);
    h.root
        .maintenance_controller
        .grow_pool(&h.pool_ids[0], 2048)
        .unwrap();
    let sequence = admitted_sequence_with_ceiling(&h.root, "future-equivalence", 12);
    let session = sequence.open_session().unwrap();
    let lane = h.root.create_execution_lane().unwrap();
    let view = route_view(&h.root, &session, &lane, 1).with_token_mask_residency(
        ProductTokenMaskResidencySnapshot::new(false, 4, vec![], &mut || true).unwrap(),
    );
    let allocations = runtime.allocate_calls();
    let mask: Arc<[i8]> = Arc::from([0, 1, 0, 1, 1]);
    let modes = [
        ProductTokenMaskContent::selection(5, 7, &mask),
        ProductTokenMaskContent::AllValid { vocabulary_size: 5 },
    ];
    let mut full = vec![view.initial_state()];
    let mut compact = vec![view.initial_state()];
    for _ in 0..10 {
        let mut next_full = Vec::new();
        for state in &full {
            for mode in &modes {
                next_full.push(advance(&h.root, &view, state, mode.clone()));
            }
        }
        let mut next_compact: Vec<ExecutionCostRouteState> = Vec::new();
        for state in &compact {
            for mode in &modes {
                let next = advance(&h.root, &view, state, mode.clone());
                if !next_compact
                    .iter()
                    .any(|old| old.same_future_state(&next, &mut || true).unwrap())
                {
                    next_compact.push(next);
                }
            }
        }
        // Both inclusions compare whole successors, not just token totals or
        // matching state counts. Every subsequent transition is repeated from
        // both populations, including the next actual allocator extension.
        for state in &next_full {
            assert!(next_compact
                .iter()
                .any(|other| other.same_future_state(state, &mut || true).unwrap()));
        }
        for state in &next_compact {
            assert!(next_full
                .iter()
                .any(|other| other.same_future_state(state, &mut || true).unwrap()));
        }
        assert_eq!(next_compact.len(), 1);
        full = next_full;
        compact = next_compact;
    }
    assert!(full.len() > compact.len());
    assert_eq!(compact[0].resources.covered_tokens(0), Some(11));
    assert_eq!(runtime.allocate_calls(), allocations);
    assert_eq!(view.initial_state().frontiers, [1]);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(h.root);
}

#[test]
fn future_state_equality_preserves_frontiers_initialization_graph_and_weak_mask_identity() {
    let (h, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&h.root, "state-identity", 4);
    let session = sequence.open_session().unwrap();
    let view = route_view(&h.root, &session, &lane, 0);
    let original = view.initial_state();
    let mut changed = original.clone();
    changed.frontiers[0] += 1;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut changed = original.clone();
    changed.initialized[0] = true;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let mut changed = original.clone();
    changed.projected_graph_state = ActualWaveGraphState::Warm;
    assert!(!original.same_future_state(&changed, &mut || true).unwrap());
    let recaptured = route_view(&h.root, &session, &lane, 0).initial_state();
    assert!(!original
        .same_future_state(&recaptured, &mut || true)
        .unwrap());

    let slot = real_mask_step_identity();
    let other_slot = real_mask_step_identity();
    let a: Arc<[i8]> = Arc::from([0, 1, 0, 1, 1]);
    let same_bytes: Arc<[i8]> = Arc::from(a.as_ref());
    let masks = |slot, source: &Arc<[i8]>| {
        ProductTokenMaskResidencySnapshot::new(
            true,
            4,
            vec![ProductTokenMaskResidencyEntry::with_content(
                slot,
                0,
                ProductTokenMaskContent::selection(5, 7, source),
            )],
            &mut || true,
        )
        .unwrap()
    };
    let mut a_state = original.clone();
    a_state.token_masks = Some(masks(slot.clone(), &a));
    let mut independent_source = a_state.clone();
    independent_source.token_masks = Some(masks(slot.clone(), &same_bytes));
    assert!(!a_state
        .same_future_state(&independent_source, &mut || true)
        .unwrap());
    let mut independent_slot = a_state.clone();
    independent_slot.token_masks = Some(masks(other_slot, &a));
    assert!(!a_state
        .same_future_state(&independent_slot, &mut || true)
        .unwrap());
    let weak = Arc::downgrade(&a);
    let clone = a_state.clone();
    let strong = Arc::strong_count(&a);
    assert!(a_state.same_future_state(&clone, &mut || true).unwrap());
    assert_eq!(Arc::strong_count(&a), strong);
    drop(a);
    assert!(weak.upgrade().is_none());
    assert!(a_state.same_future_state(&clone, &mut || true).unwrap());
    // Equality did not keep the mask source alive or suppress the existing
    // stale-source rule when a subsequent wave actually tries to read it.
    assert_eq!(
        a_state.token_masks.as_mut().unwrap().project_contents(
            Some(&slot),
            5,
            &[ProductTokenMaskContent::selection(5, 7, &same_bytes)],
            &mut || true,
        ),
        Err(ExecutionCostRouteUnknown::StaleView)
    );
    assert!(matches!(
        original.same_future_state(&original, &mut || false),
        Err(ExecutionCostRouteUnknown::BudgetExhausted)
    ));
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(h.root);
}
