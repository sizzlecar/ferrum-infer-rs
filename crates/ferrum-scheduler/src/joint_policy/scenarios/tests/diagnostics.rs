use super::*;

fn config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    config
}

fn run(
    snapshot: &dyn CostSnapshot,
    now: Instant,
    config: &SchedulerSloConfig,
    requests: &[SimulatedRequest],
    alternatives: &[usize],
    assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
) -> (Result<ScenarioEvaluation, Unsupported>, ScenarioDiagnostics) {
    let mut report = ScenarioDiagnostics::default();
    let result = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot,
            now,
            config,
            execution: CostExecution::SequentialPrefillThenDecode,
            requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &Split,
            static_action: &128,
            alternatives,
            static_window_rounds: 64,
        },
        assign,
        |_, _| {},
        &mut report,
    );
    (result, report)
}

#[test]
fn failed_simulation_retains_actual_provider_and_cached_query_counts() {
    let snapshot = Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let mut config = config();
    config.planning.max_queries = 2;
    let (result, report) = run(
        &snapshot,
        snapshot.at,
        &config,
        &[request()],
        &[],
        |_, _| Ok(()),
    );
    assert!(matches!(result, Err(Unsupported::QueryBudget)));
    assert_eq!(
        report.provider_calls,
        snapshot.calls.load(Ordering::Relaxed) as u64
    );
    assert_eq!(report.provider_calls, 2);
    assert_eq!(report.simulation_queries, 2);
    assert_eq!(report.cache_hits, 1);
    assert_eq!(report.rank_attempts, 1);
    assert_eq!(report.rank_completed, 0);
    let failure = report.terminal_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::SimulationQueryBudget);
    assert_eq!(failure.point.rank, Some(0));
    assert_eq!(failure.point.candidate, Some(0));
    assert_eq!(failure.point.round, Some(2));
    assert_eq!(failure.point.group, None);
    assert_eq!(failure.rank_simulation_queries, 2);
    assert_eq!(failure.query_limit, 2);
    assert_eq!(report.first_failure, report.terminal_failure);
}

#[test]
fn failed_uncached_provider_budget_is_distinct_from_simulation_budget() {
    let snapshot = Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let mut config = config();
    config.planning.max_queries = 2;
    config.planning.max_cost_cache_bytes = 0;
    let (result, report) = run(
        &snapshot,
        snapshot.at,
        &config,
        &[request()],
        &[],
        |_, _| Ok(()),
    );
    assert!(matches!(
        result,
        Err(Unsupported::Cost(CostUnknownReason::PlanningBudget))
    ));
    assert_eq!(report.provider_calls, 2);
    assert_eq!(
        report.provider_calls,
        snapshot.calls.load(Ordering::Relaxed) as u64
    );
    assert_eq!(report.simulation_queries, 2);
    assert_eq!(report.cache_hits, 0);
    let failure = report.terminal_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::ProviderQueryBudget);
    assert_eq!(failure.point.round, Some(1));
    assert_eq!(failure.point.group, Some(0));
}

struct Reject {
    base: Snapshot,
    reject_seed: bool,
    reject_decode: bool,
}

impl CostSnapshot for Reject {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("must use complete distributions")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        if self.reject_seed
            || (self.reject_decode
                && query
                    .work
                    .iter()
                    .any(|work| work.kind == AttemptWorkKind::Decode))
            || query.work.iter().any(|work| {
                work.kind == AttemptWorkKind::Prefill && work.requested.as_range().len() == 64
            })
        {
            self.base.calls.fetch_add(1, Ordering::Relaxed);
            CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedDomain)
        } else {
            self.base.cost_scenarios(query, now, count)
        }
    }
}

#[test]
fn rejected_seed_counts_real_call_without_inventing_simulated_queries() {
    let snapshot = Reject {
        base: Snapshot {
            at: Instant::now(),
            calls: AtomicUsize::new(0),
        },
        reject_seed: true,
        reject_decode: false,
    };
    let (result, report) = run(
        &snapshot,
        snapshot.base.at,
        &config(),
        &[request()],
        &[],
        |_, _| Ok(()),
    );
    assert!(matches!(
        result,
        Err(Unsupported::Cost(CostUnknownReason::UnsupportedDomain))
    ));
    assert_eq!(report.provider_calls, 1);
    assert_eq!(report.simulation_queries, 0);
    assert_eq!(report.rank_attempts, 0);
    let failure = report.terminal_failure.unwrap();
    assert_eq!(failure.point.stage, EvaluationStage::Seed);
    assert_eq!(failure.point.rank, None);
    assert_eq!(failure.point.group, Some(0));
    assert_eq!(failure.origin, FailureOrigin::Provider);
}

#[test]
fn unsupported_alternative_is_observed_without_marking_successful_traversal_failed() {
    let snapshot = Reject {
        base: Snapshot {
            at: Instant::now(),
            calls: AtomicUsize::new(0),
        },
        reject_seed: false,
        reject_decode: false,
    };
    let mut request = request();
    request.prompt_tokens = 128;
    request.prefill_tokens = 0;
    request.output_tokens = 0;
    request.output_cap = 1;
    request.scenario_end_tokens = 1;
    request.central_clock = RequestClock::default();
    request.risk_clock = RequestClock::default();
    let (result, report) = run(
        &snapshot,
        snapshot.base.at,
        &config(),
        &[request],
        &[64],
        |_, _| Ok(()),
    );
    let result = result.unwrap();
    assert_eq!(report.provider_calls, result.queries as u64);
    assert_eq!(
        report.provider_calls,
        snapshot.base.calls.load(Ordering::Relaxed) as u64
    );
    assert_eq!(report.cache_hits, result.cache_hits as u64);
    assert_eq!(report.rank_completed, 2);
    assert_eq!(report.failure_count, 2);
    assert!(report.terminal_failure.is_none());
    let failure = report.first_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::Provider);
    assert_eq!(failure.point.rank, Some(0));
    assert_eq!(failure.point.candidate, Some(1));
    assert_eq!(failure.point.round, Some(0));
    assert_eq!(failure.point.group, Some(0));
}

#[test]
fn unresolved_assignment_preserves_mass_and_reports_no_simulation() {
    let snapshot = Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let (result, report) = run(
        &snapshot,
        snapshot.at,
        &config(),
        &[request()],
        &[],
        |_, _| Err(Unsupported::StoppingCoverage),
    );
    let result = result.unwrap();
    assert_eq!(result.unresolved_mass, 1.0);
    assert_eq!(report.rank_attempts, 2);
    assert_eq!(report.rank_unresolved, 2);
    assert_eq!(report.rank_completed, 0);
    assert_eq!(report.provider_calls, 1);
    assert_eq!(report.simulation_queries, 0);
    assert_eq!(
        report.first_failure.unwrap().origin,
        FailureOrigin::RankAssignment
    );
    assert!(report.terminal_failure.is_none());
}

#[test]
fn first_alternative_failure_and_later_terminal_static_failure_keep_their_own_points() {
    let snapshot = Reject {
        base: Snapshot {
            at: Instant::now(),
            calls: AtomicUsize::new(0),
        },
        reject_seed: false,
        reject_decode: true,
    };
    let mut request = request();
    request.prompt_tokens = 128;
    request.prefill_tokens = 0;
    request.output_tokens = 0;
    request.output_cap = 2;
    request.scenario_end_tokens = 2;
    request.central_clock = RequestClock::default();
    request.risk_clock = RequestClock::default();
    let (result, report) = run(
        &snapshot,
        snapshot.base.at,
        &config(),
        &[request],
        &[64],
        |rank, requests| {
            requests[0].scenario_end_tokens = rank.index + 1;
            Ok(())
        },
    );
    assert!(matches!(
        result,
        Err(Unsupported::Cost(CostUnknownReason::UnsupportedDomain))
    ));
    assert_eq!(report.failure_count, 2);
    let first = report.first_failure.unwrap();
    let terminal = report.terminal_failure.unwrap();
    assert_eq!(
        (first.point.rank, first.point.candidate, first.point.round),
        (Some(0), Some(1), Some(0))
    );
    assert_eq!(
        (
            terminal.point.rank,
            terminal.point.candidate,
            terminal.point.round
        ),
        (Some(1), Some(0), Some(1))
    );
    assert_eq!(terminal.provider_calls, 3);
    assert_eq!(terminal.simulation_queries, 4);
    assert_eq!(
        report.provider_calls,
        snapshot.base.calls.load(Ordering::Relaxed) as u64
    );
    assert!(first.provider_calls < terminal.provider_calls);
}

#[test]
fn post_provider_elapsed_boundary_is_budget_not_unsupported_domain() {
    let dependence = CostScenarioDependence::SharedResidualRank;
    let scenarios = [CostScenario {
        rank: 0,
        weight: 1.0,
        service_time: Duration::from_millis(1),
    }];
    let limit = Duration::from_millis(10);
    assert_eq!(
        conditional_distribution_failure(
            limit - Duration::from_nanos(1),
            limit,
            dependence,
            dependence,
            &scenarios,
            &[1.0]
        ),
        None
    );
    assert_eq!(
        conditional_distribution_failure(limit, limit, dependence, dependence, &scenarios, &[1.0]),
        Some((
            CostUnknownReason::PlanningBudget,
            FailureOrigin::PostQueryTime
        ))
    );
    // A genuine coupling mismatch remains distinct when time is available.
    assert_eq!(
        conditional_distribution_failure(
            Duration::ZERO,
            limit,
            dependence,
            dependence,
            &scenarios,
            &[0.5, 0.5]
        ),
        Some((
            CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::InvalidScenarioDistribution
            ),
            FailureOrigin::ScenarioContract
        ))
    );
    // Time exhaustion is still explicit when a malformed distribution arrives late.
    assert_eq!(
        conditional_distribution_failure(limit, limit, dependence, dependence, &scenarios, &[]),
        Some((
            CostUnknownReason::PlanningBudget,
            FailureOrigin::PostQueryTime
        ))
    );
}

#[derive(Debug, Clone, Copy)]
enum InvalidKnown {
    ZeroCentral,
    ZeroUpper,
    ZeroSamples,
    FutureObserved,
    Expired,
    Quantile(f64),
    LimitsPanic,
}

struct MalformedSnapshot {
    at: Instant,
    invalid: InvalidKnown,
}

impl CostSnapshot for MalformedSnapshot {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        assert!(
            !matches!(self.invalid, InvalidKnown::LimitsPanic),
            "provider limits panic"
        );
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Known {
            central_service_time: if matches!(self.invalid, InvalidKnown::ZeroCentral) {
                Duration::ZERO
            } else {
                Duration::from_millis(1)
            },
            upper_service_time: if matches!(self.invalid, InvalidKnown::ZeroUpper) {
                Duration::ZERO
            } else {
                Duration::from_millis(1)
            },
            risk: CostRisk::CalibratedQuantile {
                quantile: match self.invalid {
                    InvalidKnown::Quantile(value) => value,
                    _ => 0.95,
                },
                approximation: CostRiskApproximation::EmpiricalConditional,
            },
            support: CostSupport::ValidatedFeatureDomain,
            samples: if matches!(self.invalid, InvalidKnown::ZeroSamples) {
                0
            } else {
                1
            },
            observed_at: self.at
                + if matches!(self.invalid, InvalidKnown::FutureObserved) {
                    Duration::from_millis(1)
                } else {
                    Duration::ZERO
                },
            expires_at: self.at
                + if matches!(self.invalid, InvalidKnown::Expired) {
                    Duration::ZERO
                } else {
                    Duration::from_secs(1)
                },
        }
    }
}

fn malformed_simulation(
    invalid: InvalidKnown,
) -> (Result<Comparison, Unsupported>, ScenarioDiagnostics) {
    let snapshot = MalformedSnapshot {
        at: Instant::now(),
        invalid,
    };
    let config = config();
    let recorder = crate::joint_policy::diagnostics::Recorder::new(config.planning.max_queries);
    let result = SloPolicy::evaluate_recorded(
        EvaluationInput {
            snapshot: &snapshot,
            now: snapshot.at,
            config: &config,
            execution: CostExecution::SequentialPrefillThenDecode,
            requests: &[request()],
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &Split,
            static_action: &128,
            alternatives: &[],
            static_window_rounds: 1,
        },
        &recorder,
        None,
    );
    let report = recorder.finish(result.as_ref().err().copied());
    (result, report)
}

#[test]
fn invalid_known_timing_and_quantiles_are_scenario_contract_failures() {
    for invalid in [
        InvalidKnown::ZeroCentral,
        InvalidKnown::ZeroUpper,
        InvalidKnown::ZeroSamples,
        InvalidKnown::FutureObserved,
        InvalidKnown::Expired,
        InvalidKnown::Quantile(f64::NAN),
        InvalidKnown::Quantile(0.0),
        InvalidKnown::Quantile(1.0),
    ] {
        let (result, report) = malformed_simulation(invalid);
        let reason = Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
            CostUnsupportedDetail::InvalidScenarioDistribution,
        ));
        assert!(
            matches!(result, Err(found) if found == reason),
            "{invalid:?}"
        );
        let failure = report.terminal_failure.unwrap();
        assert_eq!(failure.reason, reason, "{invalid:?}");
        assert_eq!(
            failure.origin,
            FailureOrigin::ScenarioContract,
            "{invalid:?}"
        );
        assert_eq!(failure.point.group, Some(0));
        assert_eq!(report.simulation_queries, 1);
        assert_eq!(report.failure_count, 1);
    }
}

#[test]
fn query_limits_panics_identify_provider_at_both_real_evaluation_boundaries() {
    let snapshot = MalformedSnapshot {
        at: Instant::now(),
        invalid: InvalidKnown::LimitsPanic,
    };
    let (result, report) = run(
        &snapshot,
        snapshot.at,
        &config(),
        &[request()],
        &[],
        |_, _| Ok(()),
    );
    assert!(matches!(
        result,
        Err(Unsupported::Cost(CostUnknownReason::UnsupportedDomain))
    ));
    let failure = report.terminal_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::ProviderPanic);
    assert_eq!(failure.point.stage, EvaluationStage::Preparation);
    assert_eq!(report.provider_calls, 0);
    assert_eq!(report.simulation_queries, 0);
    assert_eq!(report.failure_count, 1);

    let (result, report) = malformed_simulation(InvalidKnown::LimitsPanic);
    assert!(matches!(
        result,
        Err(Unsupported::Cost(CostUnknownReason::UnsupportedDomain))
    ));
    let failure = report.terminal_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::ProviderPanic);
    assert_eq!(failure.point.stage, EvaluationStage::Simulation);
    assert_eq!(failure.point.candidate, Some(0));
    assert_eq!(failure.point.round, Some(0));
    assert_eq!(failure.point.group, None);
    assert_eq!(report.simulation_queries, 0);
    assert_eq!(report.failure_count, 1);
}
