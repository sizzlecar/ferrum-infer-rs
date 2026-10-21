use super::*;
use ferrum_execution::{execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange};

mod budget_selection;
mod diagnostics;
mod partial_support;

struct Snapshot {
    at: Instant,
    calls: AtomicUsize,
}
impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("scenario evaluation must consume complete distributions")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let prefill = query.work[0].kind == AttemptWorkKind::Prefill;
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(1),
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time: Duration::from_micros(1000 + rank as u64),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: count,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(if prefill { 2 } else { 1 }),
        }
    }
}

struct Split;

fn request() -> SimulatedRequest {
    let clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::ZERO),
        ..Default::default()
    };
    SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 1,
        prefill_tokens: 1,
        output_tokens: 1,
        output_cap: 64,
        scenario_end_tokens: 64,
        age: Duration::ZERO,
        central_clock: clock,
        risk_clock: clock,
    }
}
impl TrajectoryAdapter<usize> for Split {
    fn plan(
        &self,
        action: Option<&usize>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
            let members = requests
                .iter()
                .enumerate()
                .filter(|(_, request)| {
                    !request.finished()
                        && (request.output_tokens == 0) == (kind == AttemptWorkKind::Prefill)
                })
                .map(|(index, request)| {
                    let context = request.prompt_tokens + request.output_tokens;
                    let requested = if kind == AttemptWorkKind::Prefill {
                        ExecutionRange::new(
                            request.prefill_tokens,
                            (request.prompt_tokens - request.prefill_tokens)
                                .min(action.copied().unwrap_or(128)),
                        )
                        .unwrap()
                    } else {
                        ExecutionRange::new(context - 1, 1).unwrap()
                    };
                    GroupMember {
                        request: index,
                        work: CostWork {
                            kind,
                            requested,
                            context_tokens: context,
                            prefix_restore: AttemptPrefixRestore::NoneObserved,
                            internally_reused_tokens: None,
                        },
                    }
                })
                .collect::<Vec<_>>();
            if !members.is_empty() {
                plan.push_group(&members)?;
            }
        }
        Ok(())
    }
}

#[test]
fn complete_scenario_cache_preserves_all_ranks_under_realistic_decode_horizon() {
    let snapshot = Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    // This is a query/memory correctness boundary, not a debug-build CPU benchmark.
    config.planning.max_elapsed_us = 1_000_000;
    let requests = (0..16)
        .map(|index| {
            let mut request = request();
            request.output_cap = 64;
            request.scenario_end_tokens = 64;
            if index < 8 {
                request.prompt_tokens = 256;
                request.prefill_tokens = 0;
                request.output_tokens = 0;
                request.central_clock = RequestClock::default();
                request.risk_clock = RequestClock::default();
            }
            request
        })
        .collect::<Vec<_>>();
    let run = |config: &SchedulerSloConfig| {
        SloPolicy::evaluate_scenarios(
            EvaluationInput {
                snapshot: &snapshot,
                now: snapshot.at,
                config,
                execution: CostExecution::SequentialPrefillThenDecode,
                requests: &requests,
                resources: ResourceSupport {
                    waiting_requests: 0,
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &Split,
                static_action: &128,
                alternatives: &[64, 32],
                horizon: EvaluationHorizon::StaticRounds(16 * (256 + 64)),
            },
            |_, _, comparison| {
                if config.planning.max_cost_cache_bytes > 0 {
                    assert!(comparison.alternatives.iter().all(Result::is_ok));
                    assert!(comparison
                        .static_result
                        .complete()
                        .unwrap()
                        .requests
                        .iter()
                        .all(SimulatedRequest::finished));
                }
            },
        )
    };
    let summary = run(&config).unwrap();
    assert_eq!(summary.ranks, config.planning.max_scenarios);
    assert_eq!(summary.resolved_mass, 1.0);
    assert_eq!(summary.unresolved_mass, 0.0);
    assert!(summary.queries < config.planning.max_queries);
    assert!(summary.cache_hits > summary.queries);
    assert!(summary.cache_retained_bytes <= config.planning.max_cost_cache_bytes);
    assert_eq!(summary.queries, snapshot.calls.load(Ordering::Relaxed));
    assert_eq!(summary.valid_until, snapshot.at + Duration::from_secs(1));
    config.planning.max_cost_cache_bytes = 0;
    // A separate uncached comparison must not inherit this snapshot's prior
    // calls or an earlier comparison's retained predictions.
    assert!(run(&config).is_err());
}

#[test]
fn scenario_completion_matches_the_previous_remaining_progress_bound() {
    let snapshot = Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    let requests = (0..3)
        .map(|index| {
            let mut request = request();
            request.output_cap = 3 + index;
            request.scenario_end_tokens = request.output_cap;
            if index == 0 {
                request.prompt_tokens = 4;
                request.prefill_tokens = 0;
                request.output_tokens = 0;
                request.central_clock = RequestClock::default();
                request.risk_clock = RequestClock::default();
            }
            request
        })
        .collect::<Vec<_>>();
    // This was the production adapter's cap-based round bound. Every legal
    // wave consumes at least one unit of remaining prompt/output progress.
    let bound = requests
        .iter()
        .map(|request| {
            request.prompt_tokens - request.prefill_tokens + request.output_cap
                - request.output_tokens
        })
        .sum();
    let run = |horizon| {
        let mut observed = Vec::new();
        let summary = SloPolicy::evaluate_scenarios(
            EvaluationInput {
                snapshot: &snapshot,
                now: snapshot.at,
                config: &config,
                execution: CostExecution::SequentialPrefillThenDecode,
                requests: &requests,
                resources: ResourceSupport {
                    waiting_requests: 0,
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &Split,
                static_action: &4,
                alternatives: &[2, 1],
                horizon,
            },
            |_, _, comparison| {
                let static_result = comparison.static_result.complete().unwrap();
                assert!(static_result
                    .requests
                    .iter()
                    .all(SimulatedRequest::finished));
                observed.push((
                    comparison.window,
                    comparison.targets.clone(),
                    static_result
                        .requests
                        .iter()
                        .map(|request| request.central_clock)
                        .collect::<Vec<_>>(),
                    comparison
                        .alternatives
                        .iter()
                        .map(|result| {
                            let result = result.as_ref().unwrap();
                            (
                                result.target_reached_at,
                                result.window_progress.clone(),
                                result
                                    .requests
                                    .iter()
                                    .map(|request| request.central_clock)
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                ));
            },
        )
        .unwrap();
        (summary, observed)
    };
    let (bounded, bounded_observed) = run(EvaluationHorizon::StaticRounds(bound));
    let (complete, complete_observed) = run(EvaluationHorizon::ScenarioCompletion);
    assert_eq!(bounded_observed, complete_observed);
    assert_eq!(bounded.queries, complete.queries);
    assert_eq!(bounded.resolved_mass, complete.resolved_mass);
    assert_eq!(complete.partial_static_ranks, 0);
}

#[test]
fn cache_keys_include_exact_work_identity_and_time_and_respect_memory_cap() {
    let snapshot = Snapshot {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
    };
    let mut work = [CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(1, 1).unwrap(),
        context_tokens: 2,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }];
    let query = CostQuery {
        work: &work,
        work_is_exact: true,
        execution: CostExecution::SingleInvocation,
    };
    let distribution = snapshot
        .cost_scenarios(&query, snapshot.at, 2)
        .validated(snapshot.at, 2);
    let mut cache = ScenarioCache::new(snapshot.at, 1024, 1);
    cache.insert(&query, distribution.clone());
    assert!(cache.get(&query, snapshot.at, 1).is_some());
    assert!(cache
        .get(&query, snapshot.at + Duration::from_nanos(1), 1)
        .is_none());
    assert!(cache
        .get(
            &CostQuery {
                work_is_exact: false,
                ..query
            },
            snapshot.at,
            1
        )
        .is_none());
    assert!(cache
        .get(
            &CostQuery {
                execution: CostExecution::SequentialPrefillThenDecode,
                ..query
            },
            snapshot.at,
            1
        )
        .is_none());
    work[0].internally_reused_tokens = Some(0);
    let changed = CostQuery {
        work: &work,
        work_is_exact: true,
        execution: CostExecution::SingleInvocation,
    };
    assert!(cache.get(&changed, snapshot.at, 1).is_none());
    assert!(cache.retained_bytes() <= 1024);
    let mut tiny = ScenarioCache::new(snapshot.at, 1, 100);
    tiny.insert(&changed, distribution);
    assert!(tiny.get(&changed, snapshot.at, 1).is_none());
    assert!(tiny.retained_bytes() <= 1);
}
