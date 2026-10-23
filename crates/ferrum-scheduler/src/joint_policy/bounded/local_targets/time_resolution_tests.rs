use super::*;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};
use std::sync::atomic::{AtomicUsize, Ordering};

fn evidence(elapsed: Duration, groups: usize) -> BoundedActionEvidence {
    let within = BoundedMetricTarget::from_counts(20, 0, 0, 0.95).unwrap();
    BoundedActionEvidence {
        local_targets: BoundedLocalTargets {
            ttft: within,
            tpot: within,
            itl: within,
        },
        reference_time_seconds: elapsed.as_secs_f64(),
        reference_time_bounds: BoundedReferenceTimeBounds::from_duration(elapsed, groups).unwrap(),
        supported_mass: 1.0,
        ..Default::default()
    }
}

#[test]
fn bounded_time_resolution_keeps_fractional_nanosecond_rank_gains_as_ties() {
    // Real integer-nanosecond Duration values produce fractional-ns weighted
    // means. This is a numerical reproducer, not a replay of saved model ranks.
    for improved_ranks in [2, 13] {
        let mut baseline = BoundedActionEvidence::default();
        let mut candidate = BoundedActionEvidence::default();
        for rank in 0..32 {
            add_evidence(
                &mut baseline,
                evidence(Duration::from_secs(1), 2),
                1.0 / 32.0,
            )
            .unwrap();
            add_evidence(
                &mut candidate,
                evidence(
                    Duration::from_nanos(1_000_000_000 - u64::from(rank < improved_ranks)),
                    3,
                ),
                1.0 / 32.0,
            )
            .unwrap();
        }
        assert!(candidate.reference_time_seconds < baseline.reference_time_seconds);
        assert_eq!(
            select(&baseline, &[candidate]),
            (None, BoundedSelectionReason::StaticRetained)
        );
    }
}

#[test]
fn bounded_time_resolution_handles_group_rounding_and_preserves_real_choices() {
    // Equal unrounded total work can have different rounded group sums.
    let half = Duration::try_from_secs_f64(1.6e-9).unwrap();
    let joined = Duration::try_from_secs_f64(3.2e-9).unwrap();
    let split = evidence(half.checked_add(half).unwrap(), 2);
    let together = evidence(joined, 1);
    assert!(together.reference_time_seconds < split.reference_time_seconds);
    assert_eq!(select(&split, &[together]).0, None);

    // Touching numerical intervals are a tie too, with no arbitrary epsilon.
    let baseline = evidence(Duration::from_nanos(100), 1);
    assert_eq!(
        select(&baseline, &[evidence(Duration::from_nanos(98), 1)]).0,
        None
    );
    let faster = evidence(Duration::from_nanos(90), 1);
    assert_eq!(
        select(&baseline, &[faster]),
        (Some(0), BoundedSelectionReason::WithinTargetsFaster)
    );

    let mut outside = baseline;
    outside.local_targets.itl = BoundedMetricTarget::from_counts(20, 0, 2, 0.95).unwrap();
    let slower_recovery = evidence(Duration::from_nanos(110), 1);
    assert_eq!(
        select(&outside, &[slower_recovery]),
        (Some(0), BoundedSelectionReason::WithinTargetsRecovery)
    );

    // Nonuniform positive weights and a large integral Duration remain enclosed
    // through integer conversion, division, multiplication and accumulation.
    let elapsed = Duration::new(1_u64 << 54, 123);
    let mut total = BoundedReferenceTimeBounds::default();
    let value = BoundedReferenceTimeBounds::from_duration(elapsed, 3).unwrap();
    total.add_weighted(value, 0.25).unwrap();
    total.add_weighted(value, 0.75).unwrap();
    assert!(total.lower_seconds < elapsed.as_secs_f64());
    assert!(total.upper_seconds > elapsed.as_secs_f64());
}

struct Costs {
    now: Instant,
    calls: AtomicUsize,
}

impl CostSnapshot for Costs {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("scenario evaluation must not add scalar queries")
    }

    fn cost_scenarios(
        &self,
        _: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(1),
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time: Duration::from_millis(1),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ExactObservedWork,
            samples: count,
            observed_at: self.now,
            expires_at: self.now + Duration::from_secs(60),
        }
    }
}

struct Decode;

impl TrajectoryAdapter<()> for Decode {
    fn plan(
        &self,
        _: Option<&()>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let context = requests[0].prompt_tokens + requests[0].output_tokens;
        plan.push_group(&[GroupMember {
            request: 0,
            work: CostWork {
                kind: AttemptWorkKind::Decode,
                requested: ExecutionRange::new(context - 1, 1).unwrap(),
                context_tokens: context,
                prefix_restore: AttemptPrefixRestore::NoneObserved,
                internally_reused_tokens: None,
            },
        }])
    }
}

fn compare_with_cache(max_cache_bytes: usize) -> (BoundedDecision, ScenarioEvaluation) {
    let costs = Costs {
        now: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let clock = RequestClock {
        first_output: Some(Duration::from_millis(1)),
        last_output: Some(Duration::from_millis(1)),
        ..Default::default()
    };
    let requests = [SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 1,
        output_tokens: 1,
        output_cap: 4,
        scenario_end_tokens: 4,
        age: Duration::from_millis(1),
        central_clock: clock,
        risk_clock: clock,
    }];
    let mut config: SchedulerSloConfig = "ttft:100,tpot:100,itl:100".parse().unwrap();
    config.planning.max_cost_cache_bytes = max_cache_bytes;
    config.planning.max_scenarios = 2;
    config.planning.max_lookahead_rounds = 3;
    config.planning.max_catchup_rounds = 0;
    config.planning.max_elapsed_us = 1_000_000;
    let mut accumulator = BoundedAccumulator::new(&requests, 1, &config).unwrap();
    let mut diagnostics = ScenarioDiagnostics::default();
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot: &costs,
            now: costs.now,
            config: &config,
            execution: CostExecution::SingleInvocation,
            requests: &requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &Decode,
            static_action: &(),
            alternatives: &[()],
            horizon: EvaluationHorizon::BoundedRounds(3),
        },
        |_, _| Ok(()),
        |rank, comparison| accumulator.observe(rank, comparison).unwrap(),
        &mut diagnostics,
    )
    .unwrap();
    assert_eq!(summary.queries, costs.calls.load(Ordering::Relaxed));
    assert_eq!(summary.queries as u64, diagnostics.provider_calls);
    // One distribution seed precedes the simulated groups. Every later known
    // group is either a provider call or a cache hit, without extra scoring I/O.
    assert_eq!(
        diagnostics.simulation_queries + 1,
        diagnostics.provider_calls + diagnostics.cache_hits
    );
    let decision = accumulator.finish(summary).unwrap();
    (decision, summary)
}

#[test]
fn bounded_time_resolution_uses_simulated_groups_even_on_cache_hits() {
    let (cached, cache_summary) = compare_with_cache(2 * 1024 * 1024);
    let (uncached, plain_summary) = compare_with_cache(0);
    assert!(cache_summary.cache_hits > 0);
    assert_eq!(plain_summary.cache_hits, 0);
    assert!(cache_summary.queries < plain_summary.queries);
    for (cached, uncached) in std::iter::once(&cached.static_evidence)
        .chain(&cached.alternatives)
        .zip(std::iter::once(&uncached.static_evidence).chain(&uncached.alternatives))
    {
        assert_eq!(
            cached.reference_time_seconds,
            uncached.reference_time_seconds
        );
        assert_eq!(
            cached.reference_time_bounds.lower_seconds,
            uncached.reference_time_bounds.lower_seconds
        );
        assert_eq!(
            cached.reference_time_bounds.upper_seconds,
            uncached.reference_time_bounds.upper_seconds
        );
        assert!(cached.reference_time_bounds.lower_seconds < cached.reference_time_seconds);
        assert!(cached.reference_time_bounds.upper_seconds > cached.reference_time_seconds);
    }
    assert_eq!(cached.selected, None);
    assert_eq!(uncached.selected, None);
}
