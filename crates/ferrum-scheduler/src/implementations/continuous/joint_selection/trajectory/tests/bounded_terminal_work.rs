//! Finite-prefix counterexamples using the production token/slot trajectory.
//! These are generated-service opportunities, not predicted visible SSE events.
//! The full-tail oracle uses declared cap endpoints, not natural EOS evidence.
//! The production comparison shares its query/time budget across every rank
//! and action; the complete tail below is a separate small correctness oracle.

use super::*;
use ferrum_types::{CostQueryLimits, SchedulerSloConfig};
use std::time::Instant;

const K: usize = 8;

struct ServiceDomain {
    at: Instant,
    unknown_prefill: bool,
}

impl ServiceDomain {
    fn cost(&self, query: &CostQuery<'_>) -> Result<Duration, CostUnknownReason> {
        assert!(query.work_is_exact);
        assert_eq!(query.execution, CostExecution::SingleInvocation);
        assert!(!query.work.is_empty() && query.work.len() <= 16);
        let kind = query.work[0].kind;
        assert!(query.work.iter().all(|work| work.kind == kind));
        for work in query.work {
            assert_eq!(work.prefix_restore, AttemptPrefixRestore::NoneObserved);
            assert!(work.requested.as_range().end <= work.context_tokens);
            assert!(work.context_tokens <= 2048);
        }
        match kind {
            AttemptWorkKind::Prefill if self.unknown_prefill => Err(CostUnknownReason::NoCoverage),
            AttemptWorkKind::Prefill => {
                // A fixed invocation cost plus actual work. For three 128-token
                // members: 19ms + 384/384 ms = 20ms. A smaller first chunk is
                // still costly; no zero/negative cost or model-name exception.
                let tokens: usize = query
                    .work
                    .iter()
                    .map(|work| work.requested.as_range().len())
                    .sum();
                Ok(Duration::from_millis(19)
                    + Duration::from_nanos(u64::try_from(tokens).unwrap() * 1_000_000 / 384))
            }
            AttemptWorkKind::Decode => {
                assert!(query
                    .work
                    .iter()
                    .all(|work| work.requested.as_range().len() == 1));
                Ok(Duration::from_millis(1))
            }
        }
    }
}

impl CostSnapshot for ServiceDomain {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }

    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        match self.cost(query) {
            Ok(time) => CostPrediction::Known {
                central_service_time: time,
                upper_service_time: time,
                risk: CostRisk::EngineeringMargin,
                support: CostSupport::ExactObservedWork,
                samples: 32,
                observed_at: self.at,
                expires_at: self.at + Duration::from_secs(60),
            },
            Err(reason) => CostPrediction::Unknown(reason),
        }
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        _: usize,
    ) -> CostScenarioPrediction {
        match self.cost(query) {
            Ok(time) => CostScenarioPrediction::Known {
                central_service_time: time,
                scenarios: (0..2)
                    .map(|rank| CostScenario {
                        rank,
                        weight: 0.5,
                        service_time: time,
                    })
                    .collect(),
                dependence: CostScenarioDependence::SharedResidualRank,
                support: CostSupport::ExactObservedWork,
                samples: 32,
                observed_at: self.at,
                expires_at: self.at + Duration::from_secs(60),
            },
            Err(reason) => CostScenarioPrediction::Unknown(reason),
        }
    }
}

fn fixture(decoder_outputs: usize) -> (Vec<SimulatedRequest>, CohortTrajectory) {
    assert!((1..64).contains(&decoder_outputs));
    let history = decoder_outputs - 1;
    let age = Duration::from_millis(u64::try_from(history).unwrap() + 10);
    let mut requests = (0..4)
        .map(|_| SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 9 * 128,
            prefill_tokens: 0,
            output_tokens: 0,
            output_cap: 64,
            scenario_end_tokens: 64,
            age,
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        })
        .collect::<Vec<_>>();
    let decoder = &mut requests[0];
    decoder.prompt_tokens = 1;
    decoder.prefill_tokens = 1;
    decoder.output_tokens = decoder_outputs;
    decoder.central_clock.first_output = Some(Duration::ZERO);
    decoder.central_clock.last_output = Some(Duration::from_millis(history as u64));
    decoder.central_clock.interval_count = history;
    decoder.central_clock.maximum_output_interval = if history == 0 {
        Duration::ZERO
    } else {
        Duration::from_millis(1)
    };
    decoder.risk_clock = decoder.central_clock;
    let mut adapter = CohortTrajectory {
        first_wave: Vec::new(),
        cap_commit_targets: Vec::new(),
        shapes: vec![shape(None); requests.len()],
        admission: AdmissionSlotPlan::new(&[InitialAdmission::Resident; 4], 16, 16).unwrap(),
        execution: CostExecution::SequentialPrefillThenDecode,
        shape_policy: PrefillShapePolicy::Identity,
        max_batch: 16,
        max_tokens: 512,
        prefill_step_chunk: Some(128),
        active_decode_prefill_chunk: Some(128),
        active_decode_prefill_budget: None,
        decode_pressure_threshold: 16,
        default_active_chunk: 128,
        chunking: true,
    };
    adapter.first_wave = adapter.natural_members(&requests).unwrap();
    assert!(adapter.allows_decode_only(&adapter.first_wave));
    assert_eq!(adapter.first_wave.len(), 4);
    assert_eq!(adapter.first_wave[0].work.kind, AttemptWorkKind::Decode);
    assert!(adapter.first_wave[1..].iter().all(|member| {
        member.work.kind == AttemptWorkKind::Prefill
            && member.work.requested.as_range().len() == 128
    }));
    (requests, adapter)
}

fn policy(tpot_ms: f64) -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:500,tpot:30,itl:15".parse().unwrap();
    config.tpot_ms = tpot_ms;
    // Correctness oracle tolerance, not a 10ms production performance result.
    // Query count, work, clocks and endpoints establish the assertions below.
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_catchup_rounds = 1;
    assert_eq!(config.planning.max_queries, 4096);
    config
}

fn input<'a>(
    requests: &'a [SimulatedRequest],
    adapter: &'a CohortTrajectory,
    domain: &'a ServiceDomain,
    config: &'a SchedulerSloConfig,
    action: &'a FirstAction,
    alternatives: &'a [FirstAction],
    horizon: EvaluationHorizon,
) -> EvaluationInput<'a, FirstAction, CohortTrajectory> {
    EvaluationInput {
        snapshot: domain,
        now: domain.at,
        config,
        execution: adapter.execution,
        requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: action,
        alternatives,
        horizon,
    }
}

struct PrefixRun {
    decision: BoundedDecision,
    ranks: Vec<(TrajectoryResult, Vec<Result<TrajectoryResult, Unsupported>>)>,
}

fn prefix(
    requests: &[SimulatedRequest],
    adapter: &CohortTrajectory,
    domain: &ServiceDomain,
    config: &SchedulerSloConfig,
) -> PrefixRun {
    let alternatives = actions();
    let mut aggregate = BoundedAccumulator::new(requests, alternatives.len(), config).unwrap();
    let mut ranks = Vec::new();
    let mut diagnostics = ScenarioDiagnostics::default();
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        input(
            requests,
            adapter,
            domain,
            config,
            &FirstAction::Static,
            &alternatives,
            EvaluationHorizon::BoundedRounds(K),
        ),
        |_, _| Ok(()),
        |rank, comparison| {
            let actual = comparison.static_result.complete().unwrap();
            // 71 separates immutable observed text from simulated service.
            for result in std::iter::once(actual).chain(
                comparison
                    .alternatives
                    .iter()
                    .filter_map(|r| r.as_ref().ok()),
            ) {
                for (before, after) in requests.iter().zip(&result.requests) {
                    assert_eq!(before.central_clock, after.central_clock);
                    assert_eq!(before.risk_clock, after.risk_clock);
                }
            }
            // The real Split adapter has at most two invocation groups/wave.
            assert!(actual.queries <= 2 * K);
            for result in comparison
                .alternatives
                .iter()
                .filter_map(|r| r.as_ref().ok())
            {
                assert!(result.queries <= 2 * (K + config.planning.max_catchup_rounds));
                assert!(result
                    .requests
                    .iter()
                    .zip(&actual.requests)
                    .all(|(after, target)| {
                        after.prefill_tokens >= target.prefill_tokens
                            && after.output_tokens >= target.output_tokens
                    }));
                assert!(
                    result.central_elapsed >= actual.central_elapsed
                        || result.requests.iter().all(SimulatedRequest::finished)
                );
            }
            ranks.push((actual.clone(), comparison.alternatives.clone()));
            aggregate.observe(rank, comparison).unwrap();
        },
        &mut diagnostics,
    )
    .unwrap();
    assert_eq!(summary.ranks, 2);
    assert_eq!(summary.resolved_mass, 1.0);
    assert_eq!(summary.unresolved_mass, 0.0);
    assert_eq!(summary.queries as u64, diagnostics.provider_calls);
    assert!(summary.queries <= config.planning.max_queries);
    assert!(!diagnostics.query_budget_exhausted);
    let exhausted: usize = ranks
        .iter()
        .map(|(_, alternatives)| {
            alternatives
                .iter()
                .filter(|result| matches!(result, Err(Unsupported::BoundedHorizonExhausted)))
                .count()
        })
        .sum();
    assert_eq!(diagnostics.candidate_horizon_exhaustions, exhausted as u64);
    if config.planning.max_catchup_rounds == 0 {
        assert_eq!(exhausted, alternatives.len() * summary.ranks);
    }
    let maximum_queries =
        summary.ranks * 2 * (K + alternatives.len() * (K + config.planning.max_catchup_rounds));
    assert!(diagnostics.simulation_queries <= maximum_queries as u64);
    if ranks
        .iter()
        .all(|(_, alternatives)| alternatives.iter().all(Result::is_ok))
    {
        let actual_queries: usize = ranks
            .iter()
            .map(|(baseline, alternatives)| {
                baseline.queries
                    + alternatives
                        .iter()
                        .map(|r| r.as_ref().unwrap().queries)
                        .sum::<usize>()
            })
            .sum();
        assert_eq!(diagnostics.simulation_queries, actual_queries as u64);
    }
    PrefixRun {
        decision: aggregate.finish(summary).unwrap(),
        ranks,
    }
}

fn full_tail(
    requests: &[SimulatedRequest],
    adapter: &CohortTrajectory,
    domain: &ServiceDomain,
    config: &SchedulerSloConfig,
) -> Comparison {
    let comparison = SloPolicy::evaluate(input(
        requests,
        adapter,
        domain,
        config,
        &FirstAction::Static,
        &[FirstAction::DecodeOnly],
        EvaluationHorizon::ScenarioCompletion,
    ))
    .unwrap();
    for result in [
        comparison.static_result.complete().unwrap(),
        comparison.alternatives[0].as_ref().unwrap(),
    ] {
        assert!(result.bounded_timing.is_none());
        for (before, after) in requests.iter().zip(&result.requests) {
            assert_eq!(after.id, before.id);
            assert_eq!(after.prefill_tokens, before.prompt_tokens);
            assert_eq!(after.output_tokens, before.scenario_end_tokens);
            assert_eq!(after.central_clock.interval_count, 63);
            assert!(after.central_clock.completed.is_some());
        }
    }
    comparison
}

fn bad_intervals(result: &TrajectoryResult) -> usize {
    result
        .requests
        .iter()
        .map(|request| request.central_clock.violating_interval_count)
        .sum()
}

fn actions() -> [FirstAction; 3] {
    [
        FirstAction::PrefillLimit(NonZeroUsize::new(64).unwrap()),
        FirstAction::PrefillLimit(NonZeroUsize::new(32).unwrap()),
        FirstAction::DecodeOnly,
    ]
}

#[test]
fn bounded_common_targets_reject_window_shift_and_extra_token_tpot_credit() {
    let (requests, adapter) = fixture(1);
    let domain = ServiceDomain {
        at: Instant::now(),
        unknown_prefill: false,
    };
    // The second threshold exposes an independent defect in equal-target-only
    // scoring: the extra decoder output used to erase TPOT debt. Both variants
    // now use ONE real production comparison, including all three alternatives.
    for tpot_ms in [30.0, 20.0] {
        let config = policy(tpot_ms);
        let bounded = prefix(&requests, &adapter, &domain, &config);
        assert_eq!(bounded.decision.selected, None);
        assert_eq!(
            bounded.decision.static_evidence.score.itl_pressure,
            8.0 / 4.0
        );
        assert_eq!(
            bounded.decision.alternatives[2].score.itl_pressure,
            8.0 / 4.0
        );
        assert_eq!(bounded.decision.static_evidence.score.ttft_pressure, 0.0);
        assert_eq!(bounded.decision.alternatives[2].score.ttft_pressure, 0.0);
        let (baseline, alternatives) = &bounded.ranks[0];
        let deferred = alternatives[2].as_ref().unwrap();
        assert_eq!(baseline.central_elapsed, Duration::from_millis(168));
        assert_eq!(deferred.central_elapsed, Duration::from_millis(169));
        assert_eq!(baseline.queries, 2 * K);
        assert_eq!(deferred.queries, 1 + 2 * K);
        assert!(baseline.requests[1..]
            .iter()
            .all(|r| r.prefill_tokens == 1024 && r.output_tokens == 0));
        assert!(deferred.requests[1..]
            .iter()
            .all(|r| r.prefill_tokens == 1024 && r.output_tokens == 0));
        // This decoder is still active throughout: its actual increment counts
        // complete waves. Catch-up executes K+1, not an unbounded target chase.
        assert_eq!(
            baseline.requests[0].output_tokens - requests[0].output_tokens,
            K
        );
        assert_eq!(
            deferred.requests[0].output_tokens - requests[0].output_tokens,
            K + 1
        );
        assert_eq!(bounded.decision.alternatives[2].extra_output_tokens, 1.0);
        if tpot_ms == 20.0 {
            assert!(bounded.decision.static_evidence.score.tpot_time_debt > 0.0);
            assert!(
                bounded.decision.alternatives[2].score.tpot_time_debt
                    > bounded.decision.static_evidence.score.tpot_time_debt
            );
        }
    }
    let config = policy(30.0);
    let full = full_tail(&requests, &adapter, &domain, &config);
    let baseline = full.static_result.complete().unwrap();
    let deferred = full.alternatives[0].as_ref().unwrap();
    assert_eq!(bad_intervals(baseline), 9);
    assert_eq!(bad_intervals(deferred), 9);
    assert_eq!(baseline.central_elapsed, Duration::from_millis(252));
    assert_eq!(deferred.central_elapsed, Duration::from_millis(253));

    let mut no_catchup = config;
    no_catchup.planning.max_catchup_rounds = 0;
    let exhausted = prefix(&requests, &adapter, &domain, &no_catchup);
    assert_eq!(exhausted.decision.selected, None);
    assert!(exhausted.ranks.iter().all(|(_, alternatives)| alternatives
        .iter()
        .all(|result| matches!(result, Err(Unsupported::BoundedHorizonExhausted)))));
    assert!(exhausted
        .decision
        .alternatives
        .iter()
        .all(|evidence| evidence.unsupported_mass == 1.0));

    // Unknown cost is not a zero terminal debt or a known catch-up prefix.
    let unknown = ServiceDomain {
        at: domain.at,
        unknown_prefill: true,
    };
    assert!(matches!(
        SloPolicy::evaluate_scenarios(
            input(
                &requests,
                &adapter,
                &unknown,
                &config,
                &FirstAction::Static,
                &actions(),
                EvaluationHorizon::BoundedRounds(K)
            ),
            |_, _, _| panic!("an unknown seed cannot publish a scored comparison"),
        ),
        Err(Unsupported::Cost(CostUnknownReason::NoCoverage))
    ));
}

#[test]
fn bounded_do_can_genuinely_remove_a_bad_interval_when_decoder_finishes_before_the_tail() {
    // Seven remaining DECLARED decode opportunities, not a forecast of EOS.
    let (requests, adapter) = fixture(57);
    let domain = ServiceDomain {
        at: Instant::now(),
        unknown_prefill: false,
    };
    let config = policy(30.0);
    let bounded = prefix(&requests, &adapter, &domain, &config);
    assert_eq!(bounded.decision.selected, Some(2));
    assert_eq!(
        bounded.decision.static_evidence.score.itl_pressure,
        7.0 / 4.0
    );
    assert_eq!(
        bounded.decision.alternatives[2].score.itl_pressure,
        6.0 / 4.0
    );
    let (baseline, alternatives) = &bounded.ranks[0];
    let deferred = alternatives[2].as_ref().unwrap();
    assert_eq!(baseline.requests[0].output_tokens, 64);
    assert_eq!(deferred.requests[0].output_tokens, 64);
    assert_eq!(baseline.central_elapsed, Duration::from_millis(167));
    assert_eq!(deferred.central_elapsed, baseline.central_elapsed);
    // Seven mixed static waves plus one prefill wave; DO executes its first
    // decode, six mixed waves, and two prefill waves to reach the same targets.
    assert_eq!(baseline.queries, 2 * 7 + 1);
    assert_eq!(deferred.queries, 1 + 2 * 6 + 2);
    assert!(deferred.requests[1..]
        .iter()
        .all(|r| r.prefill_tokens == 1024 && r.output_tokens == 0));
    assert_eq!(bounded.decision.alternatives[2].extra_output_tokens, 0.0);

    let full = full_tail(&requests, &adapter, &domain, &config);
    let baseline = full.static_result.complete().unwrap();
    let deferred = full.alternatives[0].as_ref().unwrap();
    assert_eq!(bad_intervals(baseline), 7);
    assert_eq!(bad_intervals(deferred), 6);
    assert_eq!(baseline.central_elapsed, Duration::from_millis(250));
    assert_eq!(deferred.central_elapsed, baseline.central_elapsed);
}
