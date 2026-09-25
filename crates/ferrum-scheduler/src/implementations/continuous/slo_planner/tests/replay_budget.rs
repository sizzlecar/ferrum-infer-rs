//! Production Construct/Improve/replay under virtual callback time. Canonical
//! fixtures are synthetic; they do not qualify backend or empirical coverage.
use super::*;
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy, Default)]
enum FinalFault {
    #[default]
    None,
    Timeout,
    BackwardClock,
    ChangedRoute,
}
struct TimedContext {
    now: Cell<u64>,
    begins: Cell<usize>,
    begin_ns: u64,
    edge_ns: u64,
    final_fault: FinalFault,
    seen: RefCell<Vec<(usize, u32, usize)>>,
}
impl TimedContext {
    fn new(edge_ns: u64) -> Self {
        Self {
            now: Cell::new(300_000), // Original transaction includes capture.
            begins: Cell::new(0),
            begin_ns: 50_000,
            edge_ns,
            final_fault: FinalFault::None,
            seen: RefCell::new(Vec::new()),
        }
    }
}
struct TimedState<'a> {
    context: &'a TimedContext,
    snapshot: &'a SchedulerSnapshot,
    epoch: usize,
}
impl PlanningExecutionContext for TimedContext {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>, PlanningUnknownReason> {
        poll()?;
        let epoch = self.begins.get() + 1;
        self.begins.set(epoch);
        self.now.set(self.now.get() + self.begin_ns);
        poll()?;
        Ok(Arc::new(TimedState {
            context: self,
            snapshot,
            epoch,
        }))
    }
}
impl<'epoch> PlanningExecutionState<'epoch> for TimedState<'epoch> {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'epoch>>, PlanningUnknownReason> {
        self.context
            .now
            .set(self.context.now.get() + self.context.edge_ns);
        let mut canonical = TestResolver
            .resolve(
                &PlanningShapeQuery {
                    snapshot: self.snapshot,
                    prior_waves: &[],
                    kind: input.kind,
                    rows: input.rows,
                    recurrent_state_bytes: input.recurrent_state_bytes,
                },
                poll,
            )?
            .unwrap();
        self.context.seen.borrow_mut().push((
            self.epoch,
            input.rows[0].request.context_tokens,
            canonical.rows.as_ptr() as usize,
        ));
        if self.epoch > 1 {
            match self.context.final_fault {
                FinalFault::None => {}
                FinalFault::Timeout => self.context.now.set(2_000_001),
                FinalFault::BackwardClock => self.context.now.set(100),
                FinalFault::ChangedRoute => canonical.provider_signature[0] ^= 1,
            }
        }
        // checked() must catch a fault even if this callback returns a shape.
        Ok(Some(ProjectedExecution {
            host_content_forecasts: None,
            statistical_evidence: None,
            ordered_work: input.work.to_vec(),
            canonical_domain: PlanningShapeDomain::Exact(canonical),
            successor: Arc::new(TimedState {
                context: self.context,
                snapshot: self.snapshot,
                epoch: self.epoch,
            }),
        }))
    }
}
struct VirtualClock<'a>(&'a Cell<u64>);
impl PlanningClock for VirtualClock<'_> {
    fn now_ns(&mut self) -> u64 {
        self.0.get()
    }
    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        Some(PlanningBudgetWindow {
            started_at_ns: 0,
            deadline_ns: 2_000_000,
        })
    }
}
fn workload() -> SchedulerSnapshot {
    let mut row = decode(1);
    row.timing.maximum_output_tokens = n32(5);
    row.timing.first_commit_at_ns = Some(100_000);
    row.timing.last_commit_at_ns = Some(100_000);
    row.timing.budgets = PlannerLatencyBudgets {
        ttft_ns: n64(10_000_000),
        tpot_ns: n64(10_000_000),
        itl_ns: n64(10_000_000),
    };
    let mut s = snapshot(vec![row]);
    s.observed_at_ns = 300_000;
    s.scope.horizon_end_ns = 5_000_000;
    s
}
fn run(s: &SchedulerSnapshot, context: &TimedContext) -> PlanningDecision {
    planner(3).propose_with_execution(
        s,
        &Model(|_: &WaveExecutionShape| Some(100_000)),
        context,
        &mut VirtualClock(&context.now),
    )
}

#[test]
fn replay_reserve_cheap_complete_plan_still_improves_with_one_global_budget() {
    let s = workload();
    let context = TimedContext::new(50_000);
    let (selected, witness, stats) = feasible(run(&s, &context));
    assert!(
        context
            .seen
            .borrow()
            .iter()
            .any(|&(epoch, kv, _)| epoch == 1 && kv > 10),
        "cheap optional improvement must run after the first complete plan"
    );
    assert!(witness.waves > 1);
    assert_eq!(context.begins.get(), 2);
    assert!(selected.replayed_first_wave(&s).is_some());
    assert_eq!(stats.measured_replay_work_ns, 200_000);
    assert_eq!(stats.replay_reserve_ns, 600_000);
    assert_eq!(stats.replay_reserve_stops, 0);
    assert!(context.now.get() < 1_600_000);
}

#[test]
fn replay_reserve_expensive_complete_plan_stops_improve_and_delivers_fresh_proof() {
    let s = workload();
    let before = s.clone();
    let context = TimedContext::new(500_000);
    let (selected, witness, stats) = feasible(run(&s, &context));
    assert_eq!(witness.waves, 1);
    let seen = context.seen.borrow();
    assert_eq!(
        seen.iter().map(|&(epoch, _, _)| epoch).collect::<Vec<_>>(),
        vec![1, 2]
    );
    assert_eq!(context.begins.get(), 2);
    assert_eq!(context.now.get(), 1_400_000);
    assert_eq!(stats.search_soft_stops, 1);
    assert_eq!(stats.replay_reserve_stops, 1);
    assert_eq!(stats.measured_replay_work_ns, 550_000);
    assert_eq!(stats.replay_reserve_ns, 950_000);
    assert_eq!(
        selected.replayed_first_wave(&s).unwrap().rows.as_ptr() as usize,
        seen[1].2
    );
    assert_ne!(
        seen[0].2, seen[1].2,
        "the search allocation is not the final receipt"
    );
    assert_eq!(s, before);
}

#[test]
fn replay_reserve_cannot_save_a_witness_whose_fresh_replay_does_not_fit() {
    let mut s = workload();
    s.observed_at_ns = 900_000;
    let context = TimedContext::new(500_000);
    context.now.set(900_000);
    let decision = run(&s, &context);
    let PlanningDecision::Unknown { reason, search } = decision else {
        panic!("{decision:?}")
    };
    assert_eq!(reason, PlanningUnknownReason::ComputeBudgetExhausted);
    assert_eq!(context.begins.get(), 2);
    assert_eq!(search.replay_reserve_ns, 950_000);
    assert_eq!(search.phase, PlanningSearchPhase::Finalization);
    assert_eq!(search.search_soft_stops, 1);
}

#[test]
fn replay_reserve_keeps_after_callback_clock_and_changed_route_rejections() {
    for fault in [
        FinalFault::Timeout,
        FinalFault::BackwardClock,
        FinalFault::ChangedRoute,
    ] {
        let s = workload();
        let mut context = TimedContext::new(500_000);
        context.final_fault = fault;
        let decision = run(&s, &context);
        let PlanningDecision::Unknown { reason, .. } = decision else {
            panic!("{decision:?}")
        };
        match fault {
            FinalFault::Timeout => {
                assert_eq!(reason, PlanningUnknownReason::ComputeBudgetExhausted)
            }
            FinalFault::BackwardClock => {
                assert_eq!(reason, PlanningUnknownReason::ClockMovedBackwards)
            }
            FinalFault::ChangedRoute => {
                assert_ne!(reason, PlanningUnknownReason::ComputeBudgetExhausted)
            }
            FinalFault::None => unreachable!(),
        }
        assert_eq!(context.begins.get(), 2);
    }
}

#[test]
fn replay_reserve_unknown_tail_never_installs_a_reserve_or_partial_witness() {
    let mut s = workload();
    s.requests[0].timing.maximum_output_tokens = n32(3);
    s.requests[0].timing.budgets.itl_ns = n64(500_000);
    s.requests[0].timing.budgets.tpot_ns = n64(500_000);
    s.scope.horizon_end_ns = 2_000_000;
    let context = TimedContext::new(50_000);
    let decision = planner(3).propose_with_execution(
        &s,
        &Model(|shape: &WaveExecutionShape| (shape.decode_kv_tokens == [10]).then_some(100_000)),
        &context,
        &mut VirtualClock(&context.now),
    );
    let PlanningDecision::Unknown { search, .. } = decision else {
        panic!("{decision:?}")
    };
    assert_eq!(search.replay_reserve_ns, 0);
    assert_eq!(search.measured_replay_work_ns, 0);
    assert_eq!(search.search_soft_stops, 0);
    assert_eq!(context.begins.get(), 1);
    assert!(context.seen.borrow().iter().any(|&(_, kv, _)| kv == 11));
}

#[test]
fn replay_reserve_without_a_complete_plan_keeps_the_original_construct_deadline() {
    let s = workload();
    let context = TimedContext::new(1_300_000);
    let decision = run(&s, &context);
    let PlanningDecision::Unknown { reason, search } = decision else {
        panic!("{decision:?}")
    };
    assert_eq!(reason, PlanningUnknownReason::ComputeBudgetExhausted);
    assert_eq!(context.begins.get(), 1, "no saved partial plan is replayed");
    assert_eq!(search.phase, PlanningSearchPhase::Construct);
    assert_eq!(search.measured_replay_work_ns, 0);
    assert_eq!(search.replay_reserve_ns, 0);
    assert_eq!(search.search_soft_stops, 0);
    assert_eq!(search.replay_reserve_stops, 0);
}

#[test]
fn replay_reserve_recovery_and_admission_use_the_same_fresh_replay_policy() {
    let mut s = workload();
    s.requests[0].timing.slo_failed = true;
    let context = TimedContext::new(500_000);
    let decision = planner(3).propose_recovery_with_execution(
        &s,
        Arc::new(PlanningObligationSet::capture(&s, 300_000).unwrap()),
        &Model(|_: &WaveExecutionShape| Some(100_000)),
        &context,
        &mut VirtualClock(&context.now),
    );
    let PlanningDecision::ProtectedWithinHorizon {
        first_wave, search, ..
    } = decision
    else {
        panic!("{decision:?}")
    };
    assert_eq!(search.replay_reserve_stops, 1);
    assert!(first_wave.replayed_first_wave(&s).is_some());
    assert_eq!(context.begins.get(), 2);

    let mut p = prefill(1);
    p.timing.maximum_output_tokens = n32(3);
    p.timing.budgets = workload().requests[0].timing.budgets;
    let mut s = snapshot(vec![p]);
    s.observed_at_ns = 300_000;
    s.scope.horizon_end_ns = 5_000_000;
    s.capabilities.prefill_chunk_sizes = vec![n32(8)];
    let context = TimedContext::new(250_000);
    let (selected, witness, stats) = feasible(planner(3).propose_admission_with_execution(
        &s,
        &s.requests[0].key,
        &Model(|_: &WaveExecutionShape| Some(100_000)),
        &context,
        &mut VirtualClock(&context.now),
    ));
    assert_eq!(
        witness.waves, 2,
        "a new first token alone must not activate the reserve"
    );
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(stats.measured_replay_work_ns, 550_000);
    assert_eq!(stats.replay_reserve_stops, 1);
    assert_eq!(context.begins.get(), 2);
    assert!(selected.replayed_first_wave(&s).is_some());
}
