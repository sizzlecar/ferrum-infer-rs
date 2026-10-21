use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountedSnapshot {
    inner: Arc<dyn CostSnapshot>,
    fail_cost: bool,
    scalar: AtomicUsize,
    cost: AtomicUsize,
    unknown_cost: AtomicUsize,
    stopping: AtomicUsize,
    completion: AtomicUsize,
}

impl CostSnapshot for CountedSnapshot {
    fn revision(&self) -> u64 {
        self.inner.revision()
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        self.inner.query_limits()
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.scalar.fetch_add(1, Ordering::Relaxed);
        self.inner.predict(query, now)
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.cost.fetch_add(1, Ordering::Relaxed);
        let prediction = if self.fail_cost {
            CostScenarioPrediction::Unknown(CostUnknownReason::PlanningBudget)
        } else {
            self.inner.cost_scenarios(query, now, count)
        };
        if matches!(prediction, CostScenarioPrediction::Unknown(_)) {
            self.unknown_cost.fetch_add(1, Ordering::Relaxed);
        }
        prediction
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.stopping.fetch_add(1, Ordering::Relaxed);
        self.inner.stopping_forecast(query)
    }
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.completion.fetch_add(1, Ordering::Relaxed);
        self.inner.completion_forecast(query)
    }
}

fn count_snapshot(round: &mut TimePlanningRound, fail_cost: bool) -> Arc<CountedSnapshot> {
    let counted = Arc::new(CountedSnapshot {
        inner: round.snapshot.clone(),
        fail_cost,
        scalar: AtomicUsize::new(0),
        cost: AtomicUsize::new(0),
        unknown_cost: AtomicUsize::new(0),
        stopping: AtomicUsize::new(0),
        completion: AtomicUsize::new(0),
    });
    round.snapshot = counted.clone();
    counted
}

fn assert_actual_query_count(
    round: &TimePlanningRound,
    source: &CountedSnapshot,
    known_costs: usize,
    unknown_costs: usize,
) {
    let costs = known_costs + unknown_costs;
    assert_eq!(source.scalar.load(Ordering::Relaxed), 0);
    assert_eq!(source.cost.load(Ordering::Relaxed), costs);
    assert_eq!(source.unknown_cost.load(Ordering::Relaxed), unknown_costs);
    assert_eq!(source.stopping.load(Ordering::Relaxed), 2);
    assert_eq!(source.completion.load(Ordering::Relaxed), 2);
    assert_eq!(round.query_count, costs + 4);
    assert_eq!(
        round.statistics.joint_evaluation.dispatch.provider_calls,
        costs as u64
    );
}

fn observed_next(
    scheduler: &ContinuousBatchScheduler,
    round: TimePlanningRound,
) -> (BatchPlan, TimePlanningRound) {
    // Use the public token selection entrypoint inside the same owner scope as
    // Engine. The private fixture's `next` helper intentionally does not merge
    // global health; reads before scope completion must not finalize it.
    let mut observation = scheduler.observe_planning_round(Some(round));
    let intent = scheduler.prepare_token_batch_selection(hint());
    let batch = scheduler
        .next_batch_with_token_admissions(intent, None, &mut |_| {}, observation.as_mut())
        .unwrap()
        .batch
        .unwrap();
    assert_eq!(scheduler.slo_snapshot().unwrap().planning_rounds, 0);
    let round = observation.finish().unwrap();
    let health = scheduler.slo_snapshot().unwrap();
    assert_eq!(health.planning_rounds, 1);
    assert_eq!(health.planning_rounds_before_selection, 0);
    (batch, round)
}

fn assert_applied_rescue(alternative_step: Duration, ttft_limit: Duration) {
    let (scheduler, mut round, prefill, decoder, _) =
        partial_fixture_with_timing(false, alternative_step, ttft_limit);
    let source = count_snapshot(&mut round, false);
    let (batch, round) = observed_next(&scheduler, round);
    assert_eq!(tokens(&batch, &prefill), 1);
    assert_eq!(tokens(&batch, &decoder), 1);
    assert_eq!(round.statistics.joint_interventions, 1);
    assert_eq!(round.statistics.joint_decode_only_interventions, 0);
    // Known: seed plus three prefill-limit groups, then DecodeOnly's singleton
    // decoder and full-prefill group with the advanced decoder. Its final
    // prefiller decode reuses the cache. Unknown: static tail once per rank;
    // Unknown distributions are intentionally not cached.
    assert_actual_query_count(&round, &source, 6, 2);
    let decision = round.statistics.joint_decision.dispatch;
    assert_eq!(decision.computed, 1);
    assert_eq!(decision.returned, 1);
    assert_eq!(decision.consumed, 1);
    assert_eq!(decision.post_finish_time_rejected, 0);
    assert_eq!(decision.outcomes.alternative, 1);
    assert_eq!(decision.candidates.joint.feasible, 1);
    assert_eq!(decision.candidates.joint.infeasible, 1);
    assert_eq!(decision.candidates.gates.eligible, 1);
    assert_eq!(decision.unresolved.decisions, 0);
    assert!(decision.latest_unresolved.is_none());
    // Reading and serializing cold health must not replay the decision.
    for _ in 0..2 {
        let health = scheduler.slo_snapshot().unwrap();
        assert_eq!(health.selections.joint_decision.dispatch.consumed, 1);
        assert_eq!(
            health
                .selections
                .joint_decision
                .dispatch
                .outcomes
                .alternative,
            1
        );
        assert_eq!(health.selections.joint_interventions, 1);
        assert_eq!(health.selections.joint_decode_only_interventions, 0);
        serde_json::to_value(health).unwrap();
    }
    assert_actual_query_count(&round, &source, 6, 2);
}

#[test]
fn actual_dispatch_records_one_applied_rescue_without_more_provider_queries() {
    assert_applied_rescue(Duration::from_secs(10), Duration::from_secs(50));
}

#[test]
fn actual_dispatch_applies_complete_rescue_later_than_the_static_prefix_bound() {
    // Preserve the 100:120 counterexample in simulated seconds so the real
    // one-second planning allowance cannot dominate its SLO margins. Static
    // TTFT 100 exceeds 90; candidate TTFT 80 and TPOT/ITL 40 meet the frozen
    // fixture targets even though full completion 120 exceeds prefix 100.
    assert_applied_rescue(Duration::from_secs(40), Duration::from_secs(90));
}

#[test]
fn actual_dispatch_records_unresolved_candidate_and_keeps_static_work() {
    let (scheduler, mut round, prefill, decoder, _) = partial_fixture(true);
    let source = count_snapshot(&mut round, false);
    let (batch, round) = observed_next(&scheduler, round);
    assert_eq!(tokens(&batch, &prefill), 2);
    assert_eq!(tokens(&batch, &decoder), 1);
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(round.statistics.joint_decode_only_interventions, 0);
    // Known: seed plus two prefill-limit groups. The static tail, prefill-limit
    // tail, and DecodeOnly's first singleton group are Unknown. Each is queried
    // again in the second rank because Unknown distributions are not cached.
    assert_actual_query_count(&round, &source, 3, 6);
    let decision = round.statistics.joint_decision.dispatch;
    assert_eq!(decision.computed, 1);
    assert_eq!(decision.returned, 1);
    assert_eq!(decision.consumed, 1);
    assert_eq!(decision.outcomes.feasibility_unresolved, 1);
    assert_eq!(decision.unresolved.decisions, 1);
    assert_eq!(decision.unresolved.no_alternatives, 0);
    assert_eq!(decision.unresolved.candidates.joint.unresolved, 2);
    let witness = decision.latest_unresolved.unwrap();
    assert_eq!(witness.partial_static_ranks, 2);
    assert_eq!(witness.alternatives_total, 2);
    assert_eq!(
        witness.alternatives[0].unwrap().unfinished_request_mass,
        2.0
    );
    let health = scheduler.slo_snapshot().unwrap();
    assert_eq!(
        health
            .selections
            .joint_decision
            .dispatch
            .outcomes
            .feasibility_unresolved,
        1
    );
    assert_eq!(
        health
            .selections
            .joint_decision
            .dispatch
            .unresolved
            .decisions,
        1
    );
    assert_actual_query_count(&round, &source, 3, 6);
}

#[test]
fn hard_provider_budget_failure_has_no_fabricated_finished_decision() {
    let (scheduler, mut round, prefill, decoder, _) = partial_fixture(false);
    let source = count_snapshot(&mut round, true);
    let (batch, round) = observed_next(&scheduler, round);
    assert_eq!(tokens(&batch, &prefill), 2);
    assert_eq!(tokens(&batch, &decoder), 1);
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_actual_query_count(&round, &source, 0, 1);
    assert_eq!(round.statistics.joint_evaluation.dispatch.evaluations, 1);
    assert_eq!(
        round.statistics.joint_evaluation.dispatch.failed_traversals,
        1
    );
    let decision = round.statistics.joint_decision.dispatch;
    assert_eq!(decision.computed, 0);
    assert_eq!(decision.returned, 0);
    assert_eq!(decision.consumed, 0);
    assert_eq!(decision.unresolved.decisions, 0);
    assert!(decision.latest_unresolved.is_none());
    assert!(round.query_budget_exhausted);
    let health = scheduler.slo_snapshot().unwrap();
    assert_eq!(health.selections.joint_decision.dispatch.computed, 0);
    assert_eq!(
        health
            .selections
            .joint_evaluation
            .dispatch
            .failed_traversals,
        1
    );
    assert_actual_query_count(&round, &source, 0, 1);
}
