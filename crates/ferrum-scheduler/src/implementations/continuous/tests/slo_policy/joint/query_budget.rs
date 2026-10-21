use super::*;
use crate::joint_policy::FailureOrigin;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
enum Fault {
    Unknown,
    Panic,
    UnspecifiedBudget,
}

struct Counted {
    base: ConditionalSnapshot,
    fault: Option<(usize, Fault)>,
    cost: AtomicUsize,
    stopping: AtomicUsize,
    completion: AtomicUsize,
}

impl Counted {
    fn calls(&self) -> usize {
        self.cost.load(Ordering::Relaxed)
            + self.stopping.load(Ordering::Relaxed)
            + self.completion.load(Ordering::Relaxed)
    }
}

impl CostSnapshot for Counted {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        self.cost.fetch_add(1, Ordering::Relaxed);
        self.base.predict(query, now)
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        max_scenarios: usize,
    ) -> CostScenarioPrediction {
        let call = self.cost.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some((at, fault)) = self.fault {
            if call == at {
                return match fault {
                    Fault::Unknown => {
                        CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedDomain)
                    }
                    Fault::Panic => panic!("original cost provider panic"),
                    Fault::UnspecifiedBudget => {
                        CostScenarioPrediction::Unknown(CostUnknownReason::PlanningBudget)
                    }
                };
            }
        }
        self.base.cost_scenarios(query, now, max_scenarios)
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.stopping.fetch_add(1, Ordering::Relaxed);
        self.base.stopping_forecast(query)
    }
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.completion.fetch_add(1, Ordering::Relaxed);
        self.base.completion_forecast(query)
    }
}

fn waiting_fixture(
    fault: Option<(usize, Fault)>,
) -> (
    ContinuousBatchScheduler,
    TimePlanningRound,
    Arc<Counted>,
    RequestId,
    Instant,
) {
    let (scheduler, mut round, prefill, decoder, now) =
        fixture(CostExecution::SequentialPrefillThenDecode, true, true);
    let mut waiting = create_test_request_with_prompt_tokens(Priority::Normal, 16);
    waiting.sampling_params.max_tokens = 2;
    let waiting_id = waiting.id.clone();
    enqueue_waiting(&scheduler, waiting);
    let mut facts = round.request_facts().unwrap().to_vec();
    let mut clocks = round.request_clocks().unwrap().to_vec();
    let mut fact = facts
        .iter()
        .find(|(id, _)| id == &prefill)
        .unwrap()
        .1
        .clone();
    fact.lifecycle.admission = 3;
    let clock = clocks.iter().find(|(id, _)| id == &prefill).unwrap().1;
    facts.push((waiting_id.clone(), fact));
    clocks.push((waiting_id, clock));
    assert!(round.set_request_facts(facts));
    assert!(round.set_request_clocks(clocks));
    let source = Arc::new(Counted {
        base: ConditionalSnapshot {
            at: now - Duration::from_millis(1),
            stopping_known: true,
            completion_known: true,
        },
        fault,
        cost: AtomicUsize::new(0),
        stopping: AtomicUsize::new(0),
        completion: AtomicUsize::new(0),
    });
    round.snapshot = source.clone();
    // Keep these correctness checks independent of host scheduling latency.
    round.config.planning.max_elapsed_us = 1_000_000;
    (scheduler, round, source, decoder, now)
}

fn admit_at(
    scheduler: &ContinuousBatchScheduler,
    round: &mut TimePlanningRound,
    now: Instant,
) -> usize {
    let mut scan = scheduler.begin_token_admission(8, None).unwrap();
    let mut admitted = 0;
    while let Some(candidate) = scan.next_at(&mut |_| {}, Some(&mut *round), now).unwrap() {
        let request_id = candidate.request().id.clone();
        assert!(matches!(
            scan.apply(
                candidate,
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt { request_id })
            ),
            TokenAdmissionApply::Applied { accepted: Some(_) }
        ));
        admitted += 1;
    }
    scan.finish().unwrap();
    admitted
}

#[test]
fn joint_query_charge_failed_admission_preserves_real_dispatch_allowance() {
    for fault in [(1, Fault::Unknown), (1, Fault::Panic), (2, Fault::Panic)] {
        let (scheduler, mut round, source, decoder, now) = waiting_fixture(Some(fault));
        scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
        assert_eq!(round.statistics.joint_evaluation.admission.evaluations, 1);
        assert_eq!(
            round
                .statistics
                .joint_evaluation
                .admission
                .failed_traversals,
            1
        );
        assert_eq!(source.cost.load(Ordering::Relaxed), fault.0);
        assert_eq!(round.query_count, source.calls());
        assert!(!round.query_budget_exhausted);
        let spent = round.query_count;
        let elapsed = round.prediction_elapsed;
        assert!(spent < round.config.planning.max_queries);

        // Consume the ordinary prepared admission verdict and actual dispatch
        // path, retaining the same immutable snapshot and planning round.
        assert_eq!(admit_at(&scheduler, &mut round, now), 1);
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &decoder), 1);
        assert!(round.statistics.joint_evaluation.dispatch.provider_calls > 0);
        assert!(round.query_count > spent);
        assert_eq!(round.query_count, source.calls());
        assert!(round.query_count <= round.config.planning.max_queries);
        assert!(round.prediction_elapsed >= elapsed);
    }
}

#[test]
fn joint_query_charge_true_exhaustion_and_unspecified_provider_budget_stay_sticky() {
    for (cache, fault, expected_origin) in [
        (0, None, FailureOrigin::ProviderQueryBudget),
        (1 << 20, None, FailureOrigin::SimulationQueryBudget),
        (
            1 << 20,
            Some((1, Fault::UnspecifiedBudget)),
            FailureOrigin::Provider,
        ),
    ] {
        let (scheduler, mut round, source, decoder, now) = waiting_fixture(fault);
        round.config.planning.max_cost_cache_bytes = cache;
        // Three cohort members each consume stopping + completion; one cost
        // call remains. A cache hit must not manufacture another allowance.
        round.config.planning.max_queries = if fault.is_some() { 128 } else { 7 };
        scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
        assert!(round.query_budget_exhausted);
        assert_eq!(round.query_count, source.calls());
        assert_eq!(source.cost.load(Ordering::Relaxed), 1);
        if fault.is_some() {
            assert!(round.query_count < round.config.planning.max_queries);
            assert_eq!(round.remaining_queries(), 0);
        }
        assert_eq!(
            round
                .statistics
                .joint_evaluation
                .admission
                .latest_terminal_failure
                .unwrap()
                .origin,
            expected_origin
        );
        let calls = source.calls();
        assert_eq!(admit_at(&scheduler, &mut round, now), 1);
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &decoder), 1);
        assert_eq!(source.calls(), calls);
        assert_eq!(round.query_count, calls);
        assert_eq!(round.statistics.joint_evaluation.dispatch.evaluations, 0);
    }
}

#[test]
fn joint_query_charge_zero_balance_and_elapsed_exhaustion_do_not_call_provider() {
    for time_exhausted in [false, true] {
        let (scheduler, mut round, source, decoder, now) = waiting_fixture(None);
        if time_exhausted {
            round.prediction_elapsed = Duration::from_micros(round.config.planning.max_elapsed_us);
        } else {
            round.query_count = round.config.planning.max_queries;
        }
        let elapsed = round.prediction_elapsed;
        scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
        assert_eq!(admit_at(&scheduler, &mut round, now), 1);
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &decoder), 1);
        assert_eq!(source.calls(), 0);
        assert!(round.prediction_elapsed >= elapsed);
    }
}

#[test]
fn joint_query_charge_cached_ranks_count_only_original_provider_calls() {
    let (scheduler, mut round, source, decoder, now) = waiting_fixture(None);
    // This fixture permits up to one second of planning to avoid host-speed
    // dependence; the policy also charges that full allowance to simulated
    // request age. Allow that declared delay plus the 360ms work trajectory so
    // admission is feasible and both consumers traverse every rank.
    round.config.ttft_ms = 2000.0;
    round.config.itl_ms = 2000.0;
    let clocks = round
        .request_clocks()
        .unwrap()
        .iter()
        .map(|(id, clock)| {
            let mut clock = *clock;
            clock.itl_threshold = Some(Duration::from_secs(2));
            (id.clone(), clock)
        })
        .collect::<Vec<_>>();
    assert!(round.set_request_clocks(clocks));
    scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
    assert_eq!(admit_at(&scheduler, &mut round, now), 1);
    let batch = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&batch, &decoder), 1);
    assert_eq!(round.query_count, source.calls());
    let costs = source.cost.load(Ordering::Relaxed) as u64;
    let diagnostics = round.statistics.joint_evaluation;
    for consumer in [diagnostics.admission, diagnostics.dispatch] {
        assert_eq!(consumer.completed_traversals, 1);
        assert_eq!(consumer.failed_traversals, 0);
        assert!(consumer.rank_completed >= 2);
    }
    assert_eq!(
        diagnostics.admission.provider_calls + diagnostics.dispatch.provider_calls,
        costs
    );
    assert!(diagnostics.admission.cache_hits + diagnostics.dispatch.cache_hits > 0);
    assert!(
        diagnostics.admission.simulation_queries + diagnostics.dispatch.simulation_queries > costs
    );
    assert_eq!(round.statistics.joint_cost_snapshot_queries, costs);
}

#[test]
fn joint_query_charge_report_boundaries_block_later_forecast_calls() {
    // u64::MAX also exercises a non-representable count on narrower targets.
    for reported in [2, 3, u64::MAX] {
        let (_, mut round, source, _, now) = waiting_fixture(None);
        round.query_count = round.config.planning.max_queries - 2;
        let before = round.query_count;
        let accepted = round.charge_scenario_queries(&crate::joint_policy::ScenarioDiagnostics {
            provider_calls: reported,
            ..Default::default()
        });
        assert_eq!(accepted, reported == 2);
        assert_eq!(round.query_budget_exhausted, reported != 2);
        assert_eq!(round.query_count, before + usize::from(accepted) * 2);
        assert_eq!(round.remaining_queries(), 0);

        let fact = round.request_facts().unwrap()[0].1.clone();
        assert_eq!(
            round
                .stopping_forecast(&StoppingQuery {
                    rules: &fact.rules,
                    verified_survival_through: fact.verified_survival_through,
                    cap: fact.cap,
                    horizon: fact.cap,
                    now,
                })
                .unresolved_mass,
            1.0
        );
        assert_eq!(
            round
                .completion_forecast(&CompletionQuery {
                    rules: &fact.rules,
                    cap: fact.cap,
                    maturity: Duration::from_secs(1),
                    now,
                })
                .unresolved_mass,
            1.0
        );
        assert_eq!(source.calls(), 0);
    }
}
