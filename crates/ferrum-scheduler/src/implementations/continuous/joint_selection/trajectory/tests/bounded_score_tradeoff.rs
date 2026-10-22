//! A policy-objective counterexample, not a new desired selection rule.
//! All alternatives finish the same declared endpoints within K + C. No bad
//! interval is hidden beyond the horizon and no candidate earns extra tokens.
//! These are exact generated-service clocks, not a forecast of visible SSEs.

use super::*;
use ferrum_types::{CostQueryLimits, SchedulerSloConfig};
use std::sync::Mutex;
use std::time::Instant;

struct ExactCosts {
    at: Instant,
    calls: Mutex<Vec<Vec<CostWork>>>,
}

impl CostSnapshot for ExactCosts {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("scenario evaluation must not issue an extra scalar query")
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        assert!(query.work_is_exact);
        assert_eq!(query.execution, CostExecution::SingleInvocation);
        assert!(!query.work.is_empty() && query.work.len() <= 2);
        let kind = query.work[0].kind;
        assert!(query.work.iter().all(|work| {
            work.kind == kind
                && work.prefix_restore == AttemptPrefixRestore::NoneObserved
                && work.requested.end() <= work.context_tokens
        }));
        let service_time = match kind {
            AttemptWorkKind::Prefill => {
                assert_eq!(query.work.len(), 1);
                // Fixed invocation overhead plus positive work. Thus P4=20ms,
                // P2=11ms, P1=6.5ms and the remaining P3=15.5ms.
                let tokens = query.work[0].requested.as_range().len();
                Duration::from_micros(2_000 + 4_500 * tokens as u64)
            }
            AttemptWorkKind::Decode => {
                assert!(query
                    .work
                    .iter()
                    .all(|work| work.requested.as_range().len() == 1));
                Duration::from_millis(1)
            }
        };
        self.calls.lock().unwrap().push(query.work.to_vec());
        CostScenarioPrediction::Known {
            central_service_time: service_time,
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time,
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ExactObservedWork,
            samples: count,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

fn fixture() -> (Vec<SimulatedRequest>, CohortTrajectory) {
    let mut decoder = request(1, 61);
    decoder.output_cap = 63;
    decoder.scenario_end_tokens = 63;
    decoder.age = Duration::from_millis(61);
    decoder.central_clock = RequestClock {
        first_output: Some(Duration::from_millis(1)),
        last_output: Some(Duration::from_millis(61)),
        interval_count: 60,
        maximum_output_interval: Duration::from_millis(1),
        ..RequestClock::default()
    };
    decoder.risk_clock = decoder.central_clock;
    let mut prefiller = request(4, 0);
    prefiller.output_cap = 2;
    prefiller.scenario_end_tokens = 2;
    let requests = vec![decoder, prefiller];
    let mut adapter = CohortTrajectory {
        first_wave: Vec::new(),
        cap_commit_targets: Vec::new(),
        shapes: vec![shape(None), shape(None)],
        admission: AdmissionSlotPlan::new(&[InitialAdmission::Resident; 2], 2, 2).unwrap(),
        execution: CostExecution::SequentialPrefillThenDecode,
        shape_policy: PrefillShapePolicy::Identity,
        max_batch: 2,
        max_tokens: 8,
        prefill_step_chunk: Some(4),
        active_decode_prefill_chunk: Some(4),
        active_decode_prefill_budget: None,
        decode_pressure_threshold: 2,
        default_active_chunk: 4,
        chunking: true,
    };
    adapter.first_wave = adapter.natural_members(&requests).unwrap();
    assert_eq!(
        adapter
            .first_wave
            .iter()
            .map(|member| (member.request, member.work.requested.as_range().len()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (1, 4)]
    );
    assert!(adapter.allows_decode_only(&adapter.first_wave));
    (requests, adapter)
}

#[test]
fn bounded_equal_weight_proxy_can_trade_ttft_and_completion_for_one_itl_violation() {
    let (requests, adapter) = fixture();
    let domain = ExactCosts {
        at: Instant::now(),
        calls: Mutex::new(Vec::new()),
    };
    let mut config: SchedulerSloConfig = "ttft:21,tpot:30,itl:15".parse().unwrap();
    config.strategy = ferrum_types::SloStrategy::Bounded;
    config.planning.max_scenarios = 2;
    config.planning.max_lookahead_rounds = 2;
    config.planning.max_catchup_rounds = 1;
    // A CPU correctness allowance, not a measured planning-performance gate.
    config.planning.max_elapsed_us = 1_000_000;
    // Exactly the current automatic bounded candidates for natural P4.
    let alternatives = [
        FirstAction::PrefillLimit(NonZeroUsize::new(2).unwrap()),
        FirstAction::PrefillLimit(NonZeroUsize::new(1).unwrap()),
        FirstAction::DecodeOnly,
    ];
    let mut accumulator = BoundedAccumulator::new(&requests, alternatives.len(), &config).unwrap();
    let mut diagnostics = ScenarioDiagnostics::default();
    let mut observed_ranks = 0;
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot: &domain,
            now: domain.at,
            config: &config,
            execution: adapter.execution,
            requests: &requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &adapter,
            static_action: &FirstAction::Static,
            alternatives: &alternatives,
            horizon: EvaluationHorizon::BoundedRounds(config.planning.max_lookahead_rounds),
        },
        |_, _| Ok(()),
        |rank, comparison| {
            observed_ranks += 1;
            let baseline = comparison.static_result.complete().unwrap();
            let results = std::iter::once(baseline).chain(
                comparison
                    .alternatives
                    .iter()
                    .map(|result| result.as_ref().unwrap()),
            );
            // Static: P4,D | (D,P). Half: P2,D | P2,D | P.
            // Quarter: P1,D | P3,D | P. DecodeOnly: D | P4,D | P.
            // Each '|': a complete wave; same endpoints, no omitted tail.
            for (index, result) in results.enumerate() {
                assert_eq!(result.central_elapsed, result.risk_elapsed);
                assert_eq!(
                    result.central_elapsed,
                    Duration::from_millis([22, 25, 25, 23][index])
                );
                assert_eq!(result.queries, [3, 5, 5, 4][index]);
                assert!(result.queries <= 2 * (2 + config.planning.max_catchup_rounds));
                assert!(result.central_elapsed >= baseline.central_elapsed);
                assert_eq!(result.window_progress, baseline.window_progress);
                for (before, after) in requests.iter().zip(&result.requests) {
                    assert_eq!(after.prefill_tokens, before.prompt_tokens);
                    assert_eq!(after.output_tokens, before.scenario_end_tokens);
                    assert!(after.finished());
                    // Bounded simulation never invents visible text history.
                    assert_eq!(after.central_clock, before.central_clock);
                    assert_eq!(after.risk_clock, before.risk_clock);
                }
                let timing = result.bounded_timing.as_ref().unwrap();
                let decoder = timing.service_clock(0).unwrap();
                let prefiller = timing.service_clock(1).unwrap();
                assert_eq!(
                    prefiller.first_output,
                    Some(Duration::from_millis([20, 23, 23, 21][index]))
                );
                assert_eq!(
                    decoder.last_output,
                    Some(Duration::from_millis(61 + [22, 24, 24, 22][index]))
                );
                assert_eq!(decoder.violating_interval_count, [1, 0, 1, 1][index]);
                assert_eq!(decoder.interval_count, 62);
                assert_eq!(prefiller.interval_count, 1);
                assert_eq!(prefiller.violating_interval_count, 0);
                assert!(decoder.completed.is_some() && prefiller.completed.is_some());
            }
            accumulator.observe(rank, comparison).unwrap();
        },
        &mut diagnostics,
    )
    .unwrap();
    assert_eq!(observed_ranks, 2);
    assert_eq!(summary.queries, domain.calls.lock().unwrap().len());
    assert_eq!(summary.queries as u64, diagnostics.provider_calls);
    assert_eq!(diagnostics.simulation_queries, 2 * (3 + 5 + 5 + 4));
    assert_eq!(diagnostics.candidate_horizon_exhaustions, 0);
    assert!(!diagnostics.query_budget_exhausted);
    let decision = accumulator.finish(summary).unwrap();
    assert_eq!(decision.selected, Some(0));
    assert_eq!(decision.static_score, 0.5);
    for (evidence, expected_score) in
        decision
            .alternatives
            .iter()
            .zip([1.0 / 21.0, 0.5 + 1.0 / 21.0, 0.5])
    {
        assert!((evidence.score.total() - expected_score).abs() < 1e-12);
        assert_eq!(evidence.score.tpot_time_debt, 0.0);
        assert_eq!(evidence.extra_output_tokens, 0.0);
        assert_eq!(evidence.unfinished_requests, 0.0);
        assert_eq!(evidence.unsupported_mass, 0.0);
    }
    let selected = &decision.alternatives[0];
    assert!(selected.score.ttft_pressure > decision.static_evidence.score.ttft_pressure);
    assert!(selected.score.itl_pressure < decision.static_evidence.score.itl_pressure);
    assert_eq!(decision.historical_bad_visible_intervals, 0);
    assert_eq!(decision.historical_visible_intervals, 60);
    // P's first service opportunity crosses 21ms (20 -> 23); the same four
    // new output tokens finish at 22 -> 25ms. No throughput reward appears in
    // this score. This test records the present proxy tradeoff, not an SLO win.
}
