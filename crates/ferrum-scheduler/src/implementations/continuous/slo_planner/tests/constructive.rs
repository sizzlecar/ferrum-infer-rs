//! Small exact cost tables exercise the common-plan contract. They do not
//! qualify an empirical backend model or replace the independent A0 oracle.
use super::*;
use std::cell::Cell;

struct TransactionClock {
    now: u64,
    origin: u64,
}
impl PlanningClock for TransactionClock {
    fn now_ns(&mut self) -> u64 {
        self.now
    }
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        Some(PlanningBudgetWindow {
            started_at_ns: self.origin,
            deadline_ns: self.origin + 2_000_000,
        })
    }
}

#[test]
fn first_logical_action_is_available_before_optional_siblings_are_built() {
    let mut s = snapshot(vec![decode(1)]);
    s.capabilities.decode_batch_sizes = (1..=64).map(nz).collect();
    s.capabilities.max_wave_rows = nz(64);
    let mut cursor =
        candidates::FrontierCursor::new(&s, &s.requests, 100, 16, None, &mut || Ok(())).unwrap();
    let mut attempts = 0;
    let first = cursor
        .next(&s, &s.requests, &mut attempts, 128, &mut || Ok(()))
        .unwrap()
        .unwrap();
    assert_eq!(
        first,
        vec![CandidateWork {
            key: s.requests[0].key.clone(),
            action: WaveAction::Decode
        }]
    );
    assert_eq!(
        attempts, 1,
        "return the action before constructing siblings"
    );
    assert!(cursor.may_have_more());
    assert!(cursor
        .next(&s, &s.requests, &mut attempts, 128, &mut || Ok(()))
        .unwrap()
        .is_none());
    assert!(
        !cursor.truncated,
        "impossible widths are not a truncated search"
    );
    assert!(attempts < s.capabilities.decode_batch_sizes.len());
}

#[test]
fn ranking_recomputes_old_and_new_plans_with_one_cpu_origin() {
    let mut row = decode(1);
    row.timing.maximum_output_tokens = n32(4);
    let s = snapshot(vec![row]);
    // Same finite horizon is discharged after the first service. Improvement
    // may choose three commits, but only with a better inclusive work rate.
    let model = Model(
        |shape: &WaveExecutionShape| match shape.decode_kv_tokens.as_slice() {
            [10] => Some(1),
            [11] | [12] => Some(2),
            _ => None,
        },
    );
    let (_, immediate, _) = feasible(planner(3).propose(
        &s,
        &model,
        &TestResolver,
        &mut TransactionClock {
            now: 100,
            origin: 100,
        },
    ));
    let (_, with_capture, _) = feasible(planner(3).propose(
        &s,
        &model,
        &TestResolver,
        &mut TransactionClock {
            now: 104,
            origin: 100,
        },
    ));
    assert_eq!(immediate.waves, 1);
    assert_eq!(immediate.predicted_output_tokens, 1);
    assert_eq!(with_capture.waves, 3);
    assert_eq!(with_capture.predicted_output_tokens, 3);
    assert!((with_capture.proxy_score - 3.0 / 9.0).abs() < f64::EPSILON);
    assert_eq!(
        immediate.validated_through_ns,
        with_capture.validated_through_ns
    );
    assert_eq!(s.requests[0].timing.last_commit_at_ns, Some(90));
}

#[test]
fn prefill_debt_uses_common_completion_time_without_advancing_real_obligations() {
    let s = snapshot(vec![prefill(1)]);
    let before = s.clone();
    let settings = planner(2).settings;
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let parent =
        simulation::begin(&s, &context, &mut || Ok(()), 100).unwrap_or_else(|_| panic!("begin"));
    let work = [CandidateWork {
        key: s.requests[0].key.clone(),
        action: WaveAction::Prefill {
            offset: 0,
            count: n32(4),
        },
    }];
    let state = simulation::advance(
        &s,
        &parent,
        &work,
        &Model(|_: &WaveExecutionShape| Some(5)),
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|_| panic!("partial"))
    .state;
    let (_, initial_debt) =
        simulation::score_at(&s, &state, &settings, 100, 100, &mut || Ok(())).unwrap();
    let (_, delayed_debt) =
        simulation::score_at(&s, &state, &settings, 100, 150, &mut || Ok(())).unwrap();
    assert!(delayed_debt > initial_debt);
    assert_eq!(state.now_ns, 105, "ranking must not advance simulation");
    assert_eq!(state.requests[0].timing.ingress_at_ns, 0);
    assert_eq!(s, before);
    assert_eq!(
        simulation::score_at(&s, &state, &settings, 101, 100, &mut || Ok(())),
        Err(PlanningUnknownReason::ClockMovedBackwards)
    );
}

#[test]
fn separately_feasible_owners_cannot_supply_a_common_incumbent() {
    let mut rows = vec![decode(1), decode(2)];
    for row in &mut rows {
        row.timing.budgets.itl_ns = n64(16); // last90 -> due106
        row.timing.budgets.tpot_ns = n64(16);
    }
    let model = Model(|_: &WaveExecutionShape| Some(4));
    for row in &rows {
        let mut s = snapshot(vec![row.clone()]);
        s.scope.horizon_end_ns = 110;
        feasible(planner(2).propose(&s, &model, &TestResolver, &mut Clock(100)));
    }
    let mut s = snapshot(rows);
    s.capabilities.decode_batch_sizes = vec![nz(1)];
    s.scope.horizon_end_ns = 110;
    assert!(matches!(
        planner(2).propose(&s, &model, &TestResolver, &mut Clock(100)),
        PlanningDecision::Unknown { .. }
    ));
}

#[test]
fn unknown_generated_decode_tail_is_not_an_incumbent() {
    let mut row = prefill(1);
    row.timing.maximum_output_tokens = n32(3);
    row.timing.budgets.ttft_ns = n64(120);
    row.timing.budgets.itl_ns = n64(10);
    row.timing.budgets.tpot_ns = n64(10);
    let mut s = snapshot(vec![row]);
    s.capabilities.prefill_chunk_sizes = vec![n32(8)];
    s.scope.horizon_end_ns = 150;
    let first_decode_seen = Cell::new(false);
    let missing_tail_seen = Cell::new(false);
    let model = Model(
        |shape: &WaveExecutionShape| match shape.decode_kv_tokens.as_slice() {
            [] => Some(5),
            [8] => {
                first_decode_seen.set(true);
                Some(5)
            }
            [9] => {
                missing_tail_seen.set(true);
                None
            }
            _ => None,
        },
    );
    let decision = planner(3).propose(&s, &model, &TestResolver, &mut Clock(100));
    assert!(first_decode_seen.get() && missing_tail_seen.get());
    assert!(matches!(
        decision,
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::CostUnavailable,
            ..
        }
    ));
}

#[test]
fn blocked_owner_is_not_an_action_but_keeps_its_deadline_obligation() {
    let mut blocked = prefill(2);
    blocked.timing.budgets.ttft_ns = n64(120);
    blocked.readiness = RequestReadiness::OutputBlocked;
    let s = snapshot(vec![decode(1), blocked]);
    let initial = s.clone();
    let mut cursor =
        candidates::FrontierCursor::new(&s, &s.requests, 100, 16, None, &mut || Ok(())).unwrap();
    let mut attempts = 0;
    while let Some(work) = cursor
        .next(&s, &s.requests, &mut attempts, 128, &mut || Ok(()))
        .unwrap()
    {
        assert_eq!(work.len(), 1);
        assert_eq!(work[0].key, s.requests[0].key);
    }
    assert!(matches!(
        planner(3).propose(
            &s,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown { .. }
    ));
    assert_eq!(s, initial);
}

#[test]
fn common_incumbent_is_replayed_after_cpu_delay_and_can_expire() {
    struct FinalDelay<'a> {
        inner: execution::ReplayContext<'a>,
        begun: Cell<bool>,
        now: &'a Cell<u64>,
        final_now: u64,
    }
    impl PlanningExecutionContext for FinalDelay<'_> {
        fn begin<'epoch>(
            &'epoch self,
            snapshot: &'epoch SchedulerSnapshot,
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason>
        {
            if self.begun.replace(true) {
                self.now.set(self.final_now);
            }
            self.inner.begin(snapshot, poll)
        }
    }
    struct ClockRef<'a>(&'a Cell<u64>);
    impl PlanningClock for ClockRef<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    for (final_now, allowed) in [(105, true), (106, false)] {
        let mut row = decode(1);
        row.timing.budgets.itl_ns = n64(20); // due110 inclusive
        row.timing.budgets.tpot_ns = n64(20);
        let s = snapshot(vec![row]);
        let now = Cell::new(100);
        let context = FinalDelay {
            inner: execution::ReplayContext {
                resolver: &TestResolver,
                resources: None,
            },
            begun: Cell::new(false),
            now: &now,
            final_now,
        };
        let decision = planner(1).propose_with_execution(
            &s,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &context,
            &mut ClockRef(&now),
        );
        if allowed {
            let (first, witness, _) = feasible(decision);
            assert_eq!(first.planning_observed_at_ns, 105);
            assert_eq!(witness.completion_at_ns, 110);
        } else {
            assert!(
                matches!(decision, PlanningDecision::Unknown { .. }),
                "{decision:?}"
            );
        }
        assert_eq!(now.get(), final_now, "final begin must actually run again");
    }
}

#[test]
fn later_mixed_plan_discharges_the_unselected_prefill_milestone_in_same_tail() {
    let mut p = prefill(2);
    p.timing.budgets.ttft_ns = n64(108);
    let RequestPhaseView::Prefill(progress) = &mut p.phase else {
        unreachable!()
    };
    progress.milestones = Arc::from([PrefillMilestone {
        at_ns: 103,
        required_reference_work_ns: 40,
    }]);
    let mut s = snapshot(vec![decode(1), p]);
    s.capabilities.prefill_chunk_sizes = vec![n32(4)];
    s.scope.horizon_end_ns = 120;
    let before = s.clone();
    let model = Model(|shape: &WaveExecutionShape| {
        Some(match shape.kind {
            WaveKind::Decode => 4,
            WaveKind::Mixed => 3,
            WaveKind::Prefill if shape.prefill_chunks[0].offset == 4 => 1,
            WaveKind::Prefill => 3,
            WaveKind::Restore | WaveKind::Maintenance => return None,
        })
    });
    let (first, witness, _) =
        feasible(planner(3).propose(&s, &model, &TestResolver, &mut Clock(100)));
    assert!(first
        .candidate
        .work
        .iter()
        .any(|row| row.action == WaveAction::Decode));
    assert!(first
        .candidate
        .work
        .iter()
        .any(|row| matches!(row.action, WaveAction::Prefill { offset: 0, .. })));
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(witness.waves, 2);
    assert_eq!(s, before);
}
