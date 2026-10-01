use super::*;
use std::time::{Duration, Instant};

fn origin(snapshot: &SchedulerSnapshot) -> PlanningTimeOrigin {
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

#[test]
fn prefix_clock_checked_window_preserves_both_independent_replays() {
    let (snapshot, offer) = setup();
    let origin = origin(&snapshot);
    let fixture = Fixture::default();
    let compared = origin
        .compare_prefix_rendezvous_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            window(),
            || origin.observed_at(),
        )
        .unwrap();
    assert!(
        matches!(compared, PrefixRendezvousDecision::Compared { .. }),
        "{compared:?}"
    );
    assert!(
        fixture.roots.get() >= 4,
        "both alternatives receive fresh replay roots"
    );

    let mut held = snapshot.clone();
    held.requests[1].readiness = RequestReadiness::StateBlocked;
    let fixture = Fixture::default();
    let continued = origin
        .continue_prefix_rendezvous_with_execution_budget_window(
            &planner(12),
            &held,
            &offer,
            PrefixContinuationPhase::HeldAwaitingProducer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            window(),
            || origin.observed_at(),
        )
        .unwrap();
    assert!(
        matches!(continued, PrefixContinuationDecision::Ready { .. }),
        "{continued:?}"
    );
    assert_eq!(fixture.continuation_roots.borrow().len(), 2);
}

#[test]
fn prefix_clock_original_start_and_phase_deadline_are_not_renewed() {
    let (snapshot, offer) = setup();
    let origin = origin(&snapshot);
    let fixture = Fixture::default();
    let phase = PlanningPhaseBudget {
        window: window(),
        planner_deadline_ns: Some(snapshot.observed_at_ns),
    };
    let decision = origin
        .compare_prefix_rendezvous_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            phase,
            || origin.observed_at(),
        )
        .unwrap();
    assert!(matches!(
        decision,
        PrefixRendezvousDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    assert_eq!(fixture.roots.get(), 0);
    let mut held = snapshot.clone();
    held.requests[1].readiness = RequestReadiness::StateBlocked;
    let result = origin.continue_prefix_rendezvous_with_execution_budget_window(
        &planner(12),
        &held,
        &offer,
        PrefixContinuationPhase::HeldAwaitingProducer,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        PlanningBudgetWindow {
            started_at_ns: snapshot.observed_at_ns + 1,
            ..window()
        },
        || origin.observed_at(),
    );
    assert!(matches!(
        result,
        Err(PlanningTimeError::SnapshotOriginMismatch)
    ));
    assert_eq!(fixture.roots.get(), 0);
}

#[test]
fn prefix_clock_sticky_backwards_read_overrides_unknown_decision() {
    let (snapshot, mut offer) = setup();
    offer.based_on_generation += 1;
    let origin = origin(&snapshot);
    let fixture = Fixture::default();
    let reads = Cell::new(0);
    let result = origin.compare_prefix_rendezvous_with_execution_budget_window(
        &planner(12),
        &snapshot,
        &offer,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        window(),
        || {
            let n = reads.get();
            reads.set(n + 1);
            origin.observed_at() - Duration::from_nanos(u64::from(n > 0))
        },
    );
    assert!(matches!(
        result,
        Err(PlanningTimeError::ClockMovedBackwards)
    ));
}

#[test]
fn prefix_clock_final_read_cannot_publish_late_continuation() {
    let (mut snapshot, offer) = setup();
    snapshot.requests[1].readiness = RequestReadiness::StateBlocked;
    let origin = origin(&snapshot);
    let reads = Cell::new(0);
    let first = origin
        .continue_prefix_rendezvous_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            PrefixContinuationPhase::HeldAwaitingProducer,
            &Model(infer),
            &Maintenance::default(),
            &Fixture::default(),
            window(),
            || {
                reads.set(reads.get() + 1);
                origin.observed_at()
            },
        )
        .unwrap();
    assert!(
        matches!(first, PrefixContinuationDecision::Ready { .. }),
        "{first:?}"
    );
    let final_read = reads.get();
    reads.set(0);
    let second = origin
        .continue_prefix_rendezvous_with_execution_budget_window(
            &planner(12),
            &snapshot,
            &offer,
            PrefixContinuationPhase::HeldAwaitingProducer,
            &Model(infer),
            &Maintenance::default(),
            &Fixture::default(),
            window(),
            || {
                reads.set(reads.get() + 1);
                if reads.get() == final_read {
                    origin.instant_at_ns(window().deadline_ns).unwrap()
                } else {
                    origin.observed_at()
                }
            },
        )
        .unwrap();
    assert!(matches!(second, PrefixContinuationDecision::Unknown {
        reason: PlanningUnknownReason::ComputeBudgetExhausted, search
    } if search == match first { PrefixContinuationDecision::Ready { search, .. } => search, _ => unreachable!() }));
}
