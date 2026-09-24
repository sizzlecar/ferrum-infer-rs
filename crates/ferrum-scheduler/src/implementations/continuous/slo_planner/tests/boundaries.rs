use super::*;
use std::{cell::Cell, rc::Rc};

struct SharedClock(Rc<Cell<u64>>);
impl PlanningClock for SharedClock {
    fn now_ns(&mut self) -> u64 {
        self.0.get()
    }
}

struct ExpiringModel {
    now: Rc<Cell<u64>>,
    calls: Cell<usize>,
    advance_on_call: usize,
    advance_to_ns: u64,
    expires_at_ns: u64,
}

impl PlanningCostModel for ExpiringModel {
    fn model_version(&self) -> u64 {
        7
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        _: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        let valid_for_ns = self.expires_at_ns.checked_sub(now_ns)?;
        let count = self.calls.get() + 1;
        self.calls.set(count);
        // Emulate time spent after a lookup established its evidence window.
        if count == self.advance_on_call {
            self.now.set(self.advance_to_ns);
        }
        Some(PlanningCost {
            typical_ns: 1,
            planning_ns: 1,
            model_version: 7,
            valid_for_ns,
        })
    }
}

fn loose_snapshot() -> SchedulerSnapshot {
    let mut request = decode(1);
    request.timing.budgets.tpot_ns = n64(5000);
    request.timing.budgets.itl_ns = n64(5000);
    let mut snapshot = snapshot(vec![request]);
    snapshot.observed_at_ns = 999;
    snapshot.scope.horizon_end_ns = 2000;
    snapshot.capabilities.decode_batch_sizes = vec![nz(1)];
    snapshot
}

#[test]
fn a_final_clock_past_the_declared_horizon_cannot_return_a_witness() {
    let mut snapshot = snapshot(vec![decode(1)]);
    snapshot.scope.horizon_end_ns = 101;
    let now = Rc::new(Cell::new(100));
    let model = ExpiringModel {
        now: now.clone(),
        calls: Cell::new(0),
        advance_on_call: 2,
        advance_to_ns: 110,
        expires_at_ns: 10_000,
    };
    assert!(matches!(
        planner(1).propose(&snapshot, &model, &TestResolver, &mut SharedClock(now)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::HorizonInsufficient,
            ..
        }
    ));
}

#[test]
fn final_lookup_time_must_fit_the_inclusive_cost_freshness_window() {
    for (final_now, should_pass) in [(1000, true), (1001, false)] {
        let snapshot = loose_snapshot();
        let now = Rc::new(Cell::new(999));
        let model = ExpiringModel {
            now: now.clone(),
            calls: Cell::new(0),
            advance_on_call: 2,
            advance_to_ns: final_now,
            expires_at_ns: 1000,
        };
        let result = planner(1).propose(&snapshot, &model, &TestResolver, &mut SharedClock(now));
        if should_pass {
            let (first, _, _) = feasible(result);
            assert_eq!(first.planning_observed_at_ns, 1000);
            assert_eq!(first.witness_valid_for_ns, 0);
        } else {
            assert!(matches!(
                result,
                PlanningDecision::Unknown {
                    reason: PlanningUnknownReason::CostUnavailable,
                    ..
                }
            ));
        }
    }
}

#[test]
fn freshness_uses_the_tightest_lookup_across_the_complete_common_sequence() {
    // Only a new root marks independent final replay. Incremental search may
    // project any number of siblings without advancing this phase marker.
    struct MarkFinalReplay<'a> {
        inner: super::super::execution::ReplayContext<'a>,
        begun: Cell<bool>,
        final_replay: Rc<Cell<bool>>,
    }
    impl PlanningExecutionContext for MarkFinalReplay<'_> {
        fn begin<'epoch>(
            &'epoch self,
            snapshot: &'epoch SchedulerSnapshot,
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason>
        {
            self.final_replay.set(self.begun.replace(true));
            self.inner.begin(snapshot, poll)
        }
    }
    struct TightestLookup {
        now: Rc<Cell<u64>>,
        final_replay: Rc<Cell<bool>>,
        final_tight_lookup_seen: Cell<bool>,
    }
    impl PlanningCostModel for TightestLookup {
        fn model_version(&self) -> u64 {
            7
        }
        fn predict(
            &self,
            _: &ExecutionFingerprint,
            _: &WaveExecutionShape,
            at_ns: u64,
        ) -> Option<PlanningCost> {
            let valid_for_ns = 1000u64.checked_sub(at_ns)?;
            // With two sequential 1ns waves, the second final lookup starts
            // at 1000 and has zero freshness left. One actual ns spent in
            // that lookup must invalidate the complete saved sequence.
            if self.final_replay.get() && at_ns == 1000 {
                self.final_tight_lookup_seen.set(true);
                self.now.set(1000);
            }
            Some(PlanningCost {
                typical_ns: 1,
                planning_ns: 1,
                model_version: 7,
                valid_for_ns,
            })
        }
    }
    let mut snapshot = loose_snapshot();
    let mut second = snapshot.requests[0].clone();
    second.key = decode(2).key;
    second.context_tokens = 20;
    second.fairness_rank = 2;
    snapshot.requests.push(second);
    let now = Rc::new(Cell::new(999));
    let final_replay = Rc::new(Cell::new(false));
    let context = MarkFinalReplay {
        inner: super::super::execution::ReplayContext {
            resolver: &TestResolver,
            resources: None,
        },
        begun: Cell::new(false),
        final_replay: final_replay.clone(),
    };
    let model = TightestLookup {
        now: now.clone(),
        final_replay,
        final_tight_lookup_seen: Cell::new(false),
    };
    assert!(matches!(
        planner(2).propose_with_execution(&snapshot, &model, &context, &mut SharedClock(now)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::CostUnavailable,
            ..
        }
    ));
    assert!(model.final_tight_lookup_seen.get());
}

#[test]
fn invalid_and_duplicate_shapes_consume_the_independent_attempt_budget() {
    let mut prefill = prefill(2);
    prefill.output_policy_signature = [4; 32];
    let mut snapshot = snapshot(vec![decode(1), prefill]);
    snapshot.capabilities.decode_batch_sizes = vec![nz(1); 64];
    snapshot.capabilities.prefill_batch_sizes = vec![nz(1); 64];
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(4); 64];
    let set = candidates::enumerate(
        &snapshot,
        &snapshot.requests,
        100,
        16,
        None,
        &TestResolver,
        &mut || Ok(()),
    )
    .unwrap();
    assert!(set.truncated);
    assert_eq!(set.attempts, 8 * 16);
    // Distinct row policies are legal when the explicit resolver covers them.
    assert_eq!(set.waves.len(), 3);
    // Public validation also rejects ambiguous duplicate capability entries.
    assert!(matches!(
        planner(1).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(1)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
}

#[test]
fn unique_capability_combinations_are_also_bounded_and_poll_the_clock_inside() {
    let mut snapshot = snapshot(vec![decode(1), prefill(2)]);
    snapshot.capabilities.decode_batch_sizes = (1..=64).map(nz).collect();
    snapshot.capabilities.prefill_batch_sizes = (1..=64).map(nz).collect();
    snapshot.capabilities.prefill_chunk_sizes = (1..=64).map(n32).collect();
    snapshot.capabilities.max_wave_rows = nz(128);
    let set = candidates::enumerate(
        &snapshot,
        &snapshot.requests,
        100,
        16,
        None,
        &TestResolver,
        &mut || Ok(()),
    )
    .unwrap();
    assert!(set.truncated);
    assert_eq!(set.attempts, 128);
    // A checked callback may poll once more on return to detect an overrun.
    // The safety property is that no ordering/resolution work starts after
    // exhaustion, rather than an exact count of clock reads.
    struct BudgetChecked<'a>(&'a std::cell::Cell<bool>);
    impl PlanningShapeResolver for BudgetChecked<'_> {
        fn order_work(
            &self,
            _: &SchedulerSnapshot,
            _: &mut [CandidateWork],
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<(), PlanningUnknownReason> {
            assert!(!self.0.get(), "ordering continued after budget exhaustion");
            poll()
        }
        fn resolve(
            &self,
            query: &PlanningShapeQuery<'_>,
            poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
        ) -> Result<
            Option<ferrum_interfaces::execution_cost::CanonicalWaveCostShape>,
            PlanningUnknownReason,
        > {
            assert!(
                !self.0.get(),
                "resolution continued after budget exhaustion"
            );
            TestResolver.resolve(query, poll)
        }
    }
    let exhausted = std::cell::Cell::new(false);
    let mut polls = 0;
    let result = candidates::enumerate(
        &snapshot,
        &snapshot.requests,
        100,
        16,
        None,
        &BudgetChecked(&exhausted),
        &mut || {
            polls += 1;
            if polls == 10 {
                exhausted.set(true);
                Err(PlanningUnknownReason::ComputeBudgetExhausted)
            } else {
                Ok(())
            }
        },
    );
    assert!(matches!(
        result,
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    assert!(exhausted.get());
}

#[test]
fn contradictory_single_token_timestamps_are_not_a_feasible_snapshot() {
    let mut request = decode(1);
    request.timing.last_commit_at_ns = Some(95);
    assert!(matches!(
        planner(1).propose(
            &snapshot(vec![request]),
            &Model(|_: &WaveExecutionShape| Some(1)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
}

#[test]
fn visible_endpoints_expose_past_ttft_and_tpot_violations_without_the_flag() {
    let mut ttft = decode(1);
    ttft.timing.first_commit_at_ns = Some(110);
    ttft.timing.last_commit_at_ns = Some(110);
    let mut tpot = decode(2);
    tpot.timing.first_commit_at_ns = Some(10);
    tpot.timing.last_commit_at_ns = Some(111);
    tpot.timing.committed_tokens = 2;
    tpot.timing.maximum_output_tokens = n32(3);
    tpot.timing.budgets.itl_ns = n64(1000);
    for request in [ttft, tpot] {
        assert!(!request.timing.slo_failed);
        let mut snapshot = snapshot(vec![request]);
        snapshot.observed_at_ns = 120;
        assert!(matches!(
            planner(1).propose(
                &snapshot,
                &Model(|_: &WaveExecutionShape| Some(5)),
                &TestResolver,
                &mut Clock(120)
            ),
            PlanningDecision::ProvenImpossibleUnderModel {
                reason: PlanningImpossibleReason::HistoricalViolation { .. },
                ..
            }
        ));
    }
}
