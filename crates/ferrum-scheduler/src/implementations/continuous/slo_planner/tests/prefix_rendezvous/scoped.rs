//! Original controller classification survives later prefix search/replay clocks.
use super::*;
use std::time::{Duration, Instant};

fn planning_origin(snapshot: &SchedulerSnapshot) -> PlanningTimeOrigin {
    let start = Instant::now();
    PlanningTimeOrigin::from_origin(start, start + Duration::from_nanos(snapshot.observed_at_ns))
        .unwrap()
}
fn window() -> PlanningBudgetWindow {
    PlanningBudgetWindow {
        started_at_ns: 0,
        deadline_ns: 10_000,
    }
}
fn scope(snapshot: &SchedulerSnapshot) -> Arc<PlanningObligationSet> {
    Arc::new(PlanningObligationSet::capture(snapshot, snapshot.observed_at_ns).unwrap())
}

#[test]
fn prefix_scoped_continuation_keeps_original_arc_after_clock_advance() {
    let (mut snapshot, offer) = setup();
    snapshot.requests[1].readiness = RequestReadiness::StateBlocked;
    let protection = scope(&snapshot);
    let origin = planning_origin(&snapshot);
    let fixture = Fixture::default();
    let decision = origin
        .continue_prefix_rendezvous_scoped_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            PrefixContinuationPhase::HeldAwaitingProducer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            Some(protection.clone()),
            window(),
            || origin.observed_at() + Duration::from_nanos(1),
        )
        .unwrap();
    let PrefixContinuationDecision::Ready { continuation, .. } = decision else {
        panic!("original scope should survive the clock advance: {decision:?}");
    };
    assert!(Arc::ptr_eq(continuation.protection(), &protection));
    assert_eq!(
        continuation.protection().classified_at_ns(),
        snapshot.observed_at_ns
    );
    let PrefixContinuationAction::Wave(selected) = continuation.action() else {
        panic!("the held producer still requires a real wave");
    };
    assert!(Arc::ptr_eq(
        selected.protection.as_ref().unwrap(),
        &protection
    ));
    let roots = fixture.continuation_roots.borrow();
    assert_eq!(roots.len(), 2);
    assert_ne!(
        roots[0].0, roots[1].0,
        "fresh replay must not reuse search's physical root"
    );
    assert!(fixture
        .seen_owner_counts
        .borrow()
        .iter()
        .all(|n| *n == snapshot.requests.len()));
}

#[test]
fn prefix_scoped_ready_and_capture_preserve_the_exact_controller_scope() {
    let (mut snapshot, old) = setup();
    snapshot.requests.remove(0);
    let ready = ReadyPrefixRestoreOffer {
        identity: old.identity,
        based_on_generation: snapshot.generation,
        target: old.target,
        boundary_tokens: old.boundary_tokens,
        expires_at_ns: old.expires_at_ns,
    };
    let protection = scope(&snapshot);
    let origin = planning_origin(&snapshot);
    let fixture = Fixture::default();
    let decision = origin
        .plan_ready_prefix_restore_scoped_with_execution_budget_window(
            &planner(6),
            &snapshot,
            &ready,
            ReadyPrefixPhase::Ready,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            Some(protection.clone()),
            window(),
            || origin.observed_at() + Duration::from_nanos(1),
        )
        .unwrap();
    let ReadyPrefixDecision::Ready { evidence, .. } = decision else {
        panic!("ready restore scope mismatch: {decision:?}");
    };
    assert!(Arc::ptr_eq(evidence.protection(), &protection));
    let roots = fixture.ready_bindings.borrow();
    assert_eq!(roots.len(), 2);
    assert_ne!(roots[0].0, roots[1].0);
    drop(roots);

    let (mut snapshot, old) = setup();
    let source = &mut snapshot.requests[0];
    let RequestPhaseView::Prefill(progress) = &mut source.phase else {
        unreachable!()
    };
    progress.offset = old.boundary_tokens.get();
    progress.logical_high_water = progress.offset;
    source.context_tokens = progress.offset;
    source.timing.budgets.ttft_ns = n64(150);
    let offer = PrefixCacheCaptureOffer {
        identity: old.identity,
        based_on_generation: snapshot.generation,
        source: old.producer,
        capture_span_start: 8,
        boundary_tokens: old.boundary_tokens,
        expires_at_ns: snapshot.scope.horizon_end_ns,
    };
    let protection = scope(&snapshot);
    let capture_origin = planning_origin(&snapshot);
    let fixture = Fixture {
        cache_capture_mode: true,
        ..Default::default()
    };
    let decision = capture_origin
        .plan_prefix_cache_capture_in_phase_scoped_with_execution_budget_window(
            &planner(6),
            &snapshot,
            &offer,
            PrefixCacheCapturePhase::AtBoundary,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            Some(protection.clone()),
            window(),
            || capture_origin.observed_at() + Duration::from_nanos(1),
        )
        .unwrap();
    let PrefixCacheCaptureDecision::Known { evidence, .. } = decision else {
        panic!("capture scope mismatch: {decision:?}");
    };
    assert!(Arc::ptr_eq(evidence.protection(), &protection));
    assert_eq!(evidence.protection().rows().len(), snapshot.requests.len());
    let roots = fixture.cache_capture_bindings.borrow();
    assert_eq!(roots.len(), 2);
    assert_ne!(roots[0], roots[1]);
}

#[test]
fn prefix_scoped_comparison_uses_original_budget_and_rejects_stale_generation() {
    let (snapshot, offer) = setup();
    let original = scope(&snapshot);
    let origin = planning_origin(&snapshot);
    let fixture = Fixture::default();
    let result = origin
        .compare_prefix_rendezvous_scoped_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            Some(original),
            window(),
            || origin.observed_at() + Duration::from_nanos(1),
        )
        .unwrap();
    assert!(
        matches!(result, PrefixRendezvousDecision::Compared { .. }),
        "{result:?}"
    );
    assert!(
        fixture.roots.get() >= 4,
        "both trajectories have independent replay"
    );
    let mut stale = snapshot.clone();
    stale.generation += 1;
    let fixture = Fixture::default();
    let rejected = origin
        .compare_prefix_rendezvous_scoped_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            Some(scope(&stale)),
            window(),
            || origin.observed_at() + Duration::from_nanos(1),
        )
        .unwrap();
    assert!(
        matches!(
            rejected,
            PrefixRendezvousDecision::Unknown {
                reason: PlanningUnknownReason::InvalidSnapshot,
                ..
            }
        ),
        "{rejected:?}"
    );
    assert_eq!(fixture.roots.get(), 0);
}

#[test]
fn prefix_scoped_scope_rejects_future_classification_and_changed_owner_before_projection() {
    let (mut snapshot, offer) = setup();
    snapshot.requests[1].readiness = RequestReadiness::StateBlocked;
    let future =
        Arc::new(PlanningObligationSet::capture(&snapshot, snapshot.observed_at_ns + 2).unwrap());
    let mut foreign = snapshot.clone();
    foreign.requests[2].key.incarnation += 1;
    for protection in [future, scope(&foreign)] {
        let fixture = Fixture::default();
        let decision = planner(12).continue_prefix_rendezvous_scoped(
            &snapshot,
            &offer,
            PrefixContinuationPhase::HeldAwaitingProducer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            Some(protection),
            &mut Clock(snapshot.observed_at_ns + 1),
        );
        assert!(
            matches!(
                decision,
                PrefixContinuationDecision::Unknown {
                    reason: PlanningUnknownReason::InvalidSnapshot,
                    ..
                }
            ),
            "{decision:?}"
        );
        assert_eq!(
            fixture.roots.get(),
            0,
            "foreign scope never reaches a physical provider"
        );
    }
}
