//! Virtual time establishes phase behavior; no machine timing thresholds.
use super::*;
use ferrum_interfaces::execution_cost::CanonicalWaveCostShape;
use std::cell::Cell;

struct TransactionClock<'a> {
    now: &'a Cell<u64>,
    window: PlanningBudgetWindow,
    cap: Option<u64>,
}
impl PlanningClock for TransactionClock<'_> {
    fn now_ns(&mut self) -> u64 {
        self.now.get()
    }
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        Some(self.window)
    }
    fn planning_phase_deadline_ns(&self) -> Option<u64> {
        self.cap
    }
}

struct ExpensiveOptionalBranch<'a> {
    now: &'a Cell<u64>,
    complete_cost_seen: &'a Cell<bool>,
    final_projection_seen: Cell<bool>,
    optional_calls: Cell<usize>,
    final_read: Option<u64>,
}
impl PlanningShapeResolver for ExpensiveOptionalBranch<'_> {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        if !query.prior_waves.is_empty() {
            self.optional_calls.set(self.optional_calls.get() + 1);
            self.now.set(1_200); // optional A -> B exploration would exhaust the transaction
        } else if query.rows.len() == 2 {
            // The successful full-wave cost lookup moves the clock past the
            // optional cutoff. A fresh root projection after that event belongs
            // to final replay, irrespective of callback/clock-poll counts.
            if self.complete_cost_seen.get() {
                self.final_projection_seen.set(true);
                if let Some(at) = self.final_read {
                    self.now.set(at);
                }
            }
        }
        TestResolver.resolve(query, poll)
    }
}
fn workload() -> SchedulerSnapshot {
    let mut s = snapshot(vec![decode(1), decode(2)]);
    for row in &mut s.requests {
        row.timing.budgets.tpot_ns = n64(100_000);
        row.timing.budgets.itl_ns = n64(100_000);
    }
    s.scope.horizon_end_ns = 10_000;
    s
}
fn run(final_read: Option<u64>, recovery: bool) -> (PlanningDecision, bool, usize) {
    run_capped(final_read, recovery, None)
}

fn run_capped(
    final_read: Option<u64>,
    recovery: bool,
    cap: Option<u64>,
) -> (PlanningDecision, bool, usize) {
    let mut s = workload();
    if recovery {
        s.requests[0].timing.slo_failed = true;
    }
    let now = Cell::new(100);
    let complete_cost_seen = Cell::new(false);
    let resolver = ExpensiveOptionalBranch {
        now: &now,
        complete_cost_seen: &complete_cost_seen,
        final_projection_seen: Cell::new(false),
        optional_calls: Cell::new(0),
        final_read,
    };
    let mut clock = TransactionClock {
        now: &now,
        cap,
        window: PlanningBudgetWindow {
            started_at_ns: 0,
            deadline_ns: 1_000,
        },
    };
    let mut planner = planner(2);
    planner.settings.search.max_planning_us = n64(1);
    let model = Model(|shape: &WaveExecutionShape| {
        // Both owners have one remaining token, so this actual whole-wave
        // lookup discharges the common horizon. No prefix replay count is used.
        if shape.decode_kv_tokens.len() == 2 && !complete_cost_seen.replace(true) {
            now.set(650);
        }
        Some(5)
    });
    let decision = if recovery {
        planner.propose_recovery(
            &s,
            Arc::new(PlanningObligationSet::capture(&s, 100).unwrap()),
            &model,
            &resolver,
            None,
            &mut clock,
        )
    } else {
        planner.propose(&s, &model, &resolver, &mut clock)
    };
    (
        decision,
        resolver.final_projection_seen.get(),
        resolver.optional_calls.get(),
    )
}

#[test]
fn complete_common_witness_stops_optional_branch_and_is_replayed() {
    let (decision, final_projection_seen, optional_calls) = run(None, false);
    let (first, witness, search) = feasible(decision);
    assert_eq!(first.candidate.work.len(), 2);
    assert_eq!(witness.waves, 1);
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(search.search_soft_stops, 1);
    assert!(final_projection_seen, "the saved wave must be reprojected");
    assert_eq!(optional_calls, 0);
}

#[test]
fn recovery_uses_same_phase_budget_without_reclassifying_scope() {
    let (decision, final_projection_seen, optional_calls) = run(None, true);
    let PlanningDecision::ProtectedWithinHorizon {
        protection, search, ..
    } = decision
    else {
        panic!("expected recovered witness, got {decision:?}");
    };
    assert!(protection.rows()[0].historical_violation);
    assert_eq!(search.search_soft_stops, 1);
    assert!(final_projection_seen);
    assert_eq!(optional_calls, 0);
}

#[test]
fn final_replay_phase_or_hard_timeout_never_returns_saved_witness() {
    for at in [800, 1_000, 1_200] {
        let (decision, final_projection_seen, _) = run(Some(at), false);
        assert!(final_projection_seen);
        assert!(
            matches!(
                decision,
                PlanningDecision::Unknown {
                    reason: PlanningUnknownReason::ComputeBudgetExhausted,
                    ..
                }
            ),
            "{at}: {decision:?}"
        );
    }
}

#[test]
fn final_replay_backward_clock_never_returns_saved_witness() {
    let (decision, _, _) = run(Some(50), false);
    assert!(matches!(
        decision,
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ClockMovedBackwards,
            ..
        }
    ));
}

#[test]
fn soft_deadline_cannot_convert_partial_service_into_shared_witness() {
    let mut s = workload();
    s.capabilities.decode_batch_sizes = vec![nz(1)];
    let now = Cell::new(650);
    let mut clock = TransactionClock {
        now: &now,
        cap: None,
        window: PlanningBudgetWindow {
            started_at_ns: 0,
            deadline_ns: 1_000,
        },
    };
    let result = planner(1).propose(
        &s,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut clock,
    );
    let PlanningDecision::Unknown { search, .. } = result else {
        panic!("{result:?}")
    };
    assert_eq!(search.search_soft_stops, 0);
}

#[test]
fn capture_consumes_original_window_and_empty_rounded_phases_are_unknown() {
    for (started_at_ns, deadline_ns, observed, now) in [(0, 1_000, 700, 800), (100, 101, 100, 100)]
    {
        let mut s = workload();
        s.observed_at_ns = observed;
        let now = Cell::new(now);
        let mut clock = TransactionClock {
            now: &now,
            cap: None,
            window: PlanningBudgetWindow {
                started_at_ns,
                deadline_ns,
            },
        };
        let result = planner(2).propose(
            &s,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut clock,
        );
        assert!(matches!(
            result,
            PlanningDecision::Unknown {
                reason: PlanningUnknownReason::ComputeBudgetExhausted,
                ..
            }
        ));
    }
}

#[test]
fn outer_phase_cap_preserves_percentages_and_order_before_search_endpoint() {
    let mut settings = ferrum_types::SloPlannerConfig::default();
    settings.max_planning_us = n64(2);
    let window = PlanningBudgetWindow {
        started_at_ns: 0,
        deadline_ns: 2_000,
    };
    assert_eq!(window.phase_deadlines(&settings), Ok((1_200, 1_600)));
    let capped = |cap| PlanningPhaseBudget {
        window,
        planner_deadline_ns: Some(cap),
    };
    assert_eq!(capped(1_100).phase_deadlines(&settings), Ok((700, 1_100)));
    assert_eq!(capped(399).phase_deadlines(&settings), Ok((0, 399)));
    assert_eq!(capped(3_000).phase_deadlines(&settings), Ok((1_200, 1_600)));
    assert_eq!(
        capped(0).phase_deadlines(&settings),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    );
}

#[test]
fn completion_reserve_allows_a_complete_witness_but_rejects_late_final_replay() {
    let (decision, replayed, _) = run_capped(None, false, Some(700));
    let (_, witness, _) = feasible(decision);
    assert_eq!(witness.predicted_output_tokens, 2);
    assert!(replayed);
    let (decision, replayed, _) = run_capped(Some(750), false, Some(700));
    assert!(replayed);
    assert!(matches!(
        decision,
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
}

#[test]
fn checked_clock_bridge_obeys_cap_without_rescaling_window() {
    use std::time::{Duration, Instant};
    let mut s = workload();
    s.observed_at_ns = 600;
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at + Duration::from_nanos(600)).unwrap();
    for (now, succeeds) in [(650, true), (700, false)] {
        let decision = origin
            .propose_scoped_with_budget_window(
                &planner(2),
                &s,
                &Model(|_: &WaveExecutionShape| Some(5)),
                &TestResolver,
                None,
                None,
                PlanningPhaseBudget {
                    window: PlanningBudgetWindow {
                        started_at_ns: 0,
                        deadline_ns: 1_000,
                    },
                    planner_deadline_ns: Some(700),
                },
                || at + Duration::from_nanos(now),
            )
            .unwrap();
        assert_eq!(
            matches!(decision, PlanningDecision::FeasibleWithinHorizon { .. }),
            succeeds,
            "{decision:?}"
        );
    }
}

#[test]
fn clock_adapter_preserves_capture_consumption_without_reset() {
    use std::time::{Duration, Instant};
    let mut s = workload();
    s.observed_at_ns = 700;
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at + Duration::from_nanos(700)).unwrap();
    let predictions = Cell::new(0);
    let model = Model(|_: &WaveExecutionShape| {
        predictions.set(predictions.get() + 1);
        Some(5)
    });
    let result = origin
        .propose_scoped_with_budget_window(
            &planner(2),
            &s,
            &model,
            &TestResolver,
            None,
            None,
            PlanningBudgetWindow {
                started_at_ns: 0,
                deadline_ns: 1_000,
            },
            || at + Duration::from_nanos(800),
        )
        .unwrap();
    assert!(matches!(
        result,
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    assert_eq!(predictions.get(), 0);
}
