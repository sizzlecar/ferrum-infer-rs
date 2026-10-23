use super::*;

#[test]
fn catchup_wave_limit_overflow_is_not_wrapped_or_silently_unbounded() {
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut config = config();
    config.planning.max_catchup_rounds = usize::MAX;
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config,
        &[prefill(100, 64)],
        &adapter,
        &[Action::SmallPrefillPhase],
        EvaluationHorizon::BoundedRounds(1),
    ))
    .unwrap();
    assert!(matches!(
        comparison.alternatives[0],
        Err(Unsupported::Arithmetic)
    ));
    assert_eq!(comparison.static_result.complete().unwrap().queries, 1);
    assert_eq!(snapshot.call_count(), 1);
}

#[test]
fn no_extra_waves_is_a_local_unknown_and_does_not_skip_later_candidates() {
    let snapshot = Snapshot::new();
    let adapter = Adapter::new();
    let mut config = config();
    config.planning.max_catchup_rounds = 0;
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config,
        &[prefill(100, 64)],
        &adapter,
        &[Action::SmallPrefillPhase, Action::Ordinary],
        EvaluationHorizon::BoundedRounds(2),
    ))
    .unwrap();
    assert!(matches!(
        comparison.alternatives[0],
        Err(Unsupported::BoundedHorizonExhausted)
    ));
    let later = comparison.alternatives[1].as_ref().unwrap();
    assert_eq!(later.queries, 2);
    assert_eq!(later.requests[0].prefill_tokens, 16);
    assert_eq!(snapshot.call_count(), 6);
    assert_eq!(adapter.waves.load(Ordering::Relaxed), 6);
}

#[test]
fn bounded_catchup_applies_the_entire_split_wave_before_returning() {
    let snapshot = Snapshot::new();
    let mut adapter = Adapter::new();
    adapter.split = true;
    let mut config = config();
    config.planning.max_catchup_rounds = 3;
    let requests = [prefill(100, 64), decode(64)];
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config,
        &requests,
        &adapter,
        &[Action::SmallPrefillPhase],
        EvaluationHorizon::BoundedRounds(1),
    ))
    .unwrap();
    let candidate = comparison.alternatives[0].as_ref().unwrap();
    // The fourth prefill group first reaches the static prefill frontier. Its
    // already planned decode group must still run; returning there loses work.
    assert_eq!(candidate.requests[0].prefill_tokens, 8);
    assert_eq!(candidate.requests[1].output_tokens, 5);
    assert_eq!(candidate.target_reached_at, Duration::from_millis(7));
    assert_eq!(candidate.central_elapsed, Duration::from_millis(8));
    let timing = candidate.bounded_timing.as_ref().unwrap();
    assert_eq!(timing.reference_reached_at, Some(Duration::from_millis(8)));
    let prefill_reference = timing.requests[0].reference.unwrap();
    assert_eq!(
        prefill_reference.progress_reached_at,
        Some(Duration::from_millis(7))
    );
    assert_eq!(
        prefill_reference.progress_at_reach,
        Some(comparison.targets[0])
    );
    assert!(prefill_reference.output_initially_satisfied);
    assert_eq!(
        prefill_reference.central.unwrap().projection.first_output,
        None
    );
    let decode_reference = timing.requests[1].reference.unwrap();
    assert!(!decode_reference.output_initially_satisfied);
    assert_eq!(decode_reference.target.output_tokens, 2);
    // The reference ordinal arrived in wave one. The final decode group and
    // later catch-up waves retain their actual work without rewriting it.
    assert_eq!(
        decode_reference.central.unwrap().projection.last_output,
        Some(Duration::from_millis(7))
    );
    assert_eq!(
        decode_reference.following_central_output_at,
        Some(Duration::from_millis(9))
    );
    assert_eq!(
        timing.requests[1].central.projection.last_output,
        Some(Duration::from_millis(13))
    );
    assert_eq!(candidate.queries, 8);
    assert_eq!(snapshot.call_count(), 10);
    assert_eq!(adapter.phase_waves.load(Ordering::Relaxed), 3);
    assert_eq!(
        candidate.requests[1].central_clock,
        requests[1].central_clock
    );
    assert!(!candidate.has_window_debt(&comparison.targets));
}

#[test]
fn catchup_limit_support_and_query_refusals_are_distinct() {
    for extra in [5, 6] {
        let snapshot = Snapshot::new();
        let adapter = Adapter::new();
        let mut config = config();
        config.planning.max_catchup_rounds = extra;
        let comparison = SloPolicy::evaluate(input(
            &snapshot,
            &config,
            &[prefill(100, 64)],
            &adapter,
            &[Action::SmallPrefillPhase],
            EvaluationHorizon::BoundedRounds(2),
        ))
        .unwrap();
        if extra == 5 {
            assert!(matches!(
                comparison.alternatives[0],
                Err(Unsupported::BoundedHorizonExhausted)
            ));
        } else {
            assert_eq!(comparison.alternatives[0].as_ref().unwrap().queries, 8);
        }
        assert_eq!(snapshot.call_count(), 2 + 2 + extra);
    }

    let mut snapshot = Snapshot::new();
    snapshot.unknown_call = Some(4);
    let adapter = Adapter::new();
    let comparison = SloPolicy::evaluate(input(
        &snapshot,
        &config(),
        &[prefill(100, 64)],
        &adapter,
        &[Action::SmallPrefillPhase],
        EvaluationHorizon::BoundedRounds(2),
    ))
    .unwrap();
    assert!(matches!(
        comparison.alternatives[0],
        Err(Unsupported::Cost(CostUnknownReason::NoCoverage))
    ));
    assert_eq!(snapshot.call_count(), 4);

    let snapshot = Snapshot::new();
    let mut limited = config();
    limited.planning.max_queries = 4;
    assert!(matches!(
        SloPolicy::evaluate(input(
            &snapshot,
            &limited,
            &[prefill(100, 64)],
            &adapter,
            &[Action::SmallPrefillPhase, Action::Ordinary],
            EvaluationHorizon::BoundedRounds(2),
        )),
        Err(Unsupported::QueryBudget)
    ));
    assert_eq!(snapshot.call_count(), 4);
}

struct RankCosts(Instant);

impl CostSnapshot for RankCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("use the actual conditional-scenario path")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        _: usize,
    ) -> CostScenarioPrediction {
        assert_eq!(query.work.len(), 1);
        let larger = query.work[0].requested.as_range().len() == 16;
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(2),
            scenarios: (0..2)
                .map(|rank| CostScenario {
                    rank,
                    weight: 0.5,
                    service_time: Duration::from_millis(if larger {
                        if rank == 0 {
                            1
                        } else {
                            5
                        }
                    } else {
                        2
                    }),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ExactObservedWork,
            samples: 2,
            observed_at: self.0,
            expires_at: self.0 + Duration::from_secs(60),
        }
    }
}

struct LargerFirst(Adapter);

impl TrajectoryAdapter<Action> for LargerFirst {
    fn plan(
        &self,
        action: Option<&Action>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.0.wave(
            if matches!(action, Some(Action::SmallPrefillPhase)) {
                16
            } else {
                8
            },
            requests,
            plan,
        )
    }
}

#[test]
fn each_rank_checks_its_own_wall_and_natural_completion_without_pausing() {
    for prompt in [1_000, 16] {
        let snapshot = RankCosts(Instant::now());
        let adapter = LargerFirst(Adapter::new());
        let mut config = config();
        config.planning.max_scenarios = 2;
        config.planning.max_catchup_rounds = 1;
        config.planning.max_cost_cache_bytes = 0;
        let requests = [prefill(prompt, 1)];
        let mut visits = 0;
        let summary = SloPolicy::evaluate_scenarios_with_requests(
            EvaluationInput {
                snapshot: &snapshot,
                now: snapshot.0,
                config: &config,
                execution: CostExecution::SingleInvocation,
                requests: &requests,
                resources: ResourceSupport {
                    waiting_requests: 0,
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &adapter,
                static_action: &Action::Ordinary,
                alternatives: &[Action::SmallPrefillPhase],
                horizon: EvaluationHorizon::BoundedRounds(2),
            },
            |_, _| Ok(()),
            |rank, comparison| {
                visits += 1;
                let baseline = comparison.static_result.complete().unwrap();
                let candidate = comparison.alternatives[0].as_ref().unwrap();
                assert_eq!(baseline.central_elapsed, Duration::from_millis(4));
                if prompt == 1_000 && rank.index == 0 {
                    assert_eq!(candidate.requests[0].prefill_tokens, 32);
                    assert_eq!(candidate.queries, 3);
                    assert_eq!(candidate.central_elapsed, Duration::from_millis(5));
                    assert_eq!(candidate.target_reached_at, Duration::from_millis(1));
                    let timing = candidate.bounded_timing.as_ref().unwrap();
                    assert_eq!(timing.reference_reached_at, Some(Duration::from_millis(1)));
                    let reference = timing.requests[0].reference.unwrap();
                    assert_eq!(reference.target.prefill_tokens, 16);
                    assert_eq!(
                        reference.progress_reached_at,
                        Some(Duration::from_millis(1))
                    );
                    assert_eq!(reference.progress_at_reach.unwrap().prefill_tokens, 16);
                    assert!(reference.output_initially_satisfied);
                    assert_eq!(reference.central.unwrap().projection.first_output, None);
                    assert_eq!(reference.following_central_output_at, None);
                } else {
                    assert_eq!(candidate.requests[0].prefill_tokens, 16);
                    assert_eq!(candidate.queries, 1);
                    assert_eq!(
                        candidate.central_elapsed,
                        Duration::from_millis(if rank.index == 0 { 1 } else { 5 })
                    );
                }
                assert!(!candidate.has_window_debt(&comparison.targets));
                assert_eq!(
                    candidate.requests[0].central_clock,
                    requests[0].central_clock
                );
                if prompt == 16 {
                    assert!(candidate.requests[0].finished());
                    assert!(candidate.bounded_timing.as_ref().unwrap().requests[0]
                        .central
                        .projection
                        .completed
                        .is_some());
                }
            },
        )
        .unwrap();
        assert_eq!(visits, 2);
        assert_eq!(summary.queries, if prompt == 1_000 { 9 } else { 7 });
    }
}
