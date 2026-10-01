use super::*;
use crate::implementations::continuous::slo_planner::{candidates::FrontierCursor, simulation};

fn row(id: u128, total: u32, offset: u32, work: &[u64]) -> RequestSchedulingView {
    let mut row = prefill(id);
    row.context_tokens = offset;
    row.timing.budgets.ttft_ns = n64(150);
    let RequestPhaseView::Prefill(p) = &mut row.phase else {
        unreachable!()
    };
    p.total_prompt_tokens = n32(total);
    p.offset = offset;
    p.logical_high_water = offset;
    p.executable_until = total;
    p.reference = Arc::new(PrefillReferenceWork {
        evaluation: Default::default(),
        version: 1,
        points: work
            .iter()
            .enumerate()
            .map(|(i, &ns)| ReferenceWorkPoint {
                prompt_tokens: i as u32 * 2,
                cumulative_work_ns: ns,
            })
            .collect(),
    });
    row
}

fn pair() -> (SchedulerSnapshot, PrefixRendezvousOffer) {
    let rows = vec![
        row(1, 8, 0, &[0, 20, 40, 60, 80]),
        row(2, 8, 0, &[0, 20, 40, 60, 80]),
        decode(3),
    ];
    let mut snapshot = snapshot(rows);
    snapshot.capabilities.prefill_alignment = n32(2);
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(2)];
    snapshot.capabilities.prefill_batch_sizes = vec![nz(1), nz(2)];
    snapshot.capabilities.max_prefill_tokens_per_wave = n64(4);
    snapshot.capabilities.native_mixed = false;
    let offer = PrefixRendezvousOffer {
        identity: [42; 32],
        based_on_generation: snapshot.generation,
        producer: snapshot.requests[0].key.clone(),
        target: snapshot.requests[1].key.clone(),
        boundary_tokens: n32(4),
        expires_at_ns: 490,
    };
    (snapshot, offer)
}

// Follow just the original opening path, using the real cursor, projection,
// cost/output simulation and common witness predicate. This isolates the deep
// branch that otherwise has to be backtracked; it does not replace the planner.
fn opening_path(
    snapshot: &SchedulerSnapshot,
    horizon: usize,
    goal_order: bool,
) -> (bool, Vec<Vec<CandidateWork>>) {
    let fixture = Fixture::default();
    let protection = PlanningObligationSet::capture(snapshot, 100).unwrap();
    let model = Model(|_: &WaveExecutionShape| Some(1));
    let mut state = simulation::begin(snapshot, &fixture, &mut || Ok(()), 100).unwrap();
    let mut waves = Vec::new();
    for depth in 0..horizon {
        let mut cursor = if goal_order {
            FrontierCursor::with_prefill_goal(
                snapshot,
                &state.requests,
                state.now_ns,
                64,
                Some(&protection),
                Some(nz(horizon - depth)),
                Some(&snapshot.requests[0].key),
                &mut || Ok(()),
            )
        } else {
            FrontierCursor::with_remaining_waves(
                snapshot,
                &state.requests,
                state.now_ns,
                64,
                Some(&protection),
                Some(nz(horizon - depth)),
                &mut || Ok(()),
            )
        }
        .unwrap();
        let Some(work) = cursor
            .next(
                snapshot,
                &state.requests,
                &mut 0,
                usize::MAX,
                &mut || Ok(()),
            )
            .unwrap()
        else {
            break;
        };
        let transition = simulation::advance(
            snapshot,
            &state,
            &work,
            &model,
            true,
            &mut || Ok(()),
            true,
            Some(&protection),
        )
        .unwrap_or_else(|_| panic!("legal opening must project"));
        state = transition.state;
        waves.push(work);
        if simulation::ready_witness(snapshot, &state, true, Some(&protection)).unwrap() {
            return (true, waves);
        }
    }
    (false, waves)
}

#[test]
fn prefix_goal_order_closes_hard_h_complete_queue_and_independently_replays() {
    let (snapshot, offer) = pair();
    let original = snapshot.clone();
    // Four required prefill visits per owner, plus the old decoder's service.
    // After that decoder, a singleton leaves its peer needing four waves with
    // only three left. Four declared two-row waves close the SAME obligations.
    let (old_closed, old) = opening_path(&snapshot, 5, false);
    let (new_closed, new) = opening_path(&snapshot, 5, true);
    assert!(!old_closed);
    assert!(new_closed);
    assert_eq!(old.len(), 5);
    assert_eq!(new.len(), 5);
    assert!(matches!(new[0][0].action, WaveAction::Decode));
    assert!(new[1..].iter().all(|work| work.len() == 2));

    let fixture = Fixture::default();
    let protection = Arc::new(PlanningObligationSet::capture(&snapshot, 100).unwrap());
    let result = comparison(planner(5).compare_prefix_rendezvous_scoped(
        &snapshot,
        &offer,
        &Model(|_: &WaveExecutionShape| Some(1)),
        &Maintenance::default(),
        &fixture,
        Some(Arc::clone(&protection)),
        &mut Clock(100),
    ));
    for path in [result.direct(), result.waiting()] {
        assert!(path
            .steps()
            .iter()
            .any(|step| matches!(step, PrefixPathStep::Wave(w)
            if w.work.iter().any(|row| row.key == snapshot.requests[2].key))));
        assert!(
            path.steps()
                .iter()
                .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
                .count()
                <= 5
        );
    }
    assert!(
        fixture.roots.get() >= 4,
        "both routes need independent fresh replay"
    );
    assert!(fixture
        .seen_owner_counts
        .borrow()
        .iter()
        .all(|&n| n == snapshot.requests.len()));
    assert_eq!(snapshot, original);
}

#[test]
fn prefix_goal_order_preserves_faster_singleton_with_optional_expensive_peer() {
    let (mut snapshot, offer) = setup();
    snapshot.capabilities.prefill_batch_sizes = vec![nz(1), nz(2)];
    let RequestPhaseView::Prefill(p) = &mut snapshot.requests[0].phase else {
        unreachable!()
    };
    p.offset = 0;
    p.logical_high_water = 0;
    snapshot.requests[0].context_tokens = 0;
    let model = Model(|shape: &WaveExecutionShape| {
        if shape.prefill_chunks.len() > 1 {
            Some(1_000)
        } else {
            infer(shape)
        }
    });
    let result = comparison(planner(12).compare_prefix_rendezvous(
        &snapshot,
        &offer,
        &model,
        &Maintenance::default(),
        &Fixture::default(),
        &mut Clock(100),
    ));
    assert_eq!(result.direct().first_commit_at_ns(), 166);
    assert!(
        !result.should_hold(),
        "declared wide batches must not inflate the direct baseline"
    );
    assert!(result.direct().steps().iter().all(|step| !matches!(step,
        PrefixPathStep::Wave(w) if w.work.iter().filter(|r| matches!(r.action, WaveAction::Prefill { .. })).count() > 1)));
}

#[test]
fn prefix_goal_order_unknown_cohort_keeps_other_endpoints_and_fresh_replay() {
    let mut snapshot = snapshot(vec![
        row(1, 12, 8, &[0, 20, 40, 60, 80, 100, 120]),
        row(
            2,
            20,
            0,
            &[0, 100, 110, 115, 120, 150, 200, 220, 250, 280, 300],
        ),
    ]);
    snapshot.capabilities.prefill_alignment = n32(2);
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(2), n32(4), n32(8)];
    snapshot.capabilities.prefill_batch_sizes = vec![nz(1), nz(2)];
    snapshot.capabilities.max_prefill_tokens_per_wave = n64(16);
    snapshot.capabilities.native_mixed = false;
    let offer = PrefixRendezvousOffer {
        identity: [42; 32],
        based_on_generation: snapshot.generation,
        producer: snapshot.requests[0].key.clone(),
        target: snapshot.requests[1].key.clone(),
        boundary_tokens: n32(10),
        expires_at_ns: 490,
    };
    let fixture = Fixture {
        rejected_prefill_counts: vec![4, 4],
        ..Default::default()
    };
    let decision = planner(3).compare_prefix_rendezvous(
        &snapshot,
        &offer,
        &Model(|_: &WaveExecutionShape| Some(1)),
        &Maintenance::default(),
        &fixture,
        &mut Clock(100),
    );
    let result = comparison(decision);
    assert!(
        fixture.shape_rejections.get() > 0,
        "preferred route must really be Unknown"
    );
    assert!(result.direct().steps().iter().any(|step| matches!(step,
        PrefixPathStep::Wave(w) if w.work.len() == 1)));
    assert!(fixture.roots.get() >= 4);
    assert!(fixture.seen_owner_counts.borrow().iter().all(|&n| n == 2));
}

#[test]
fn prefix_goal_order_preserves_total_work_envelope_and_required_recovery_service() {
    let (mut snapshot, _) = pair();
    snapshot.requests.pop();
    snapshot.capabilities.max_prefill_tokens_per_wave = n64(2);
    let protection = PlanningObligationSet::capture(&snapshot, 100).unwrap();
    let mut cursor = FrontierCursor::with_prefill_goal(
        &snapshot,
        &snapshot.requests,
        100,
        64,
        Some(&protection),
        Some(nz(4)),
        Some(&snapshot.requests[0].key),
        &mut || Ok(()),
    )
    .unwrap();
    let mut attempts = 0;
    while let Some(work) = cursor
        .next(
            &snapshot,
            &snapshot.requests,
            &mut attempts,
            usize::MAX,
            &mut || Ok(()),
        )
        .unwrap()
    {
        assert_eq!(
            work.len(),
            1,
            "largest declared cohort exceeds original token cap"
        );
    }
    snapshot.capabilities.max_prefill_tokens_per_wave = n64(4);
    snapshot.requests[1].timing.budgets.ttft_ns = n64(50);
    snapshot.requests[1].recovery_service = RecoveryServiceDebt::new(nz(1));
    snapshot.requests[1].recovery_service.bypass();
    let protection = PlanningObligationSet::capture(&snapshot, 100).unwrap();
    let mut cursor = FrontierCursor::with_prefill_goal(
        &snapshot,
        &snapshot.requests,
        100,
        64,
        Some(&protection),
        Some(nz(4)),
        Some(&snapshot.requests[0].key),
        &mut || Ok(()),
    )
    .unwrap();
    let first = cursor
        .next(
            &snapshot,
            &snapshot.requests,
            &mut 0,
            usize::MAX,
            &mut || Ok(()),
        )
        .unwrap()
        .unwrap();
    assert!(first.iter().any(|row| row.key == snapshot.requests[1].key));
}
