use super::*;

#[derive(Clone, Copy)]
enum Reply {
    Known,
    Unknown,
    Expired,
    InvalidKnown,
    DifferentWeights,
}

struct Provider {
    base: Snapshot,
    reply: Reply,
}

impl CostSnapshot for Provider {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("must use scenario distributions")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        now: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        let mut result = self.base.cost_scenarios(query, now, count);
        match self.reply {
            Reply::Unknown => {
                return CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage)
            }
            Reply::Known => {}
            _ => {
                if let CostScenarioPrediction::Known {
                    central_service_time,
                    scenarios,
                    expires_at,
                    ..
                } = &mut result
                {
                    match self.reply {
                        Reply::Expired => *expires_at = now,
                        Reply::InvalidKnown => *central_service_time = Duration::ZERO,
                        Reply::DifferentWeights => {
                            scenarios[0].weight = 0.25;
                            scenarios[1].weight = 0.75;
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        result
    }
}

struct Harness {
    original: Provider,
    config: SchedulerSloConfig,
    cache: parking_lot::Mutex<ScenarioCache>,
    queries: AtomicUsize,
    valid_until: parking_lot::Mutex<Instant>,
    diagnostics: crate::joint_policy::diagnostics::Recorder,
}

impl Harness {
    fn new(reply: Reply, cache_bytes: usize, max_queries: usize) -> Self {
        let at = Instant::now();
        let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
        config.planning.max_elapsed_us = 1_000_000;
        config.planning.max_scenarios = 2;
        config.planning.max_queries = max_queries;
        config.planning.max_cost_cache_bytes = cache_bytes;
        Self {
            original: Provider {
                base: Snapshot {
                    at,
                    calls: AtomicUsize::new(0),
                },
                reply,
            },
            config,
            cache: parking_lot::Mutex::new(ScenarioCache::new(at, cache_bytes, max_queries)),
            queries: AtomicUsize::new(0),
            valid_until: parking_lot::Mutex::new(at + Duration::from_secs(60)),
            diagnostics: crate::joint_policy::diagnostics::Recorder::new(max_queries),
        }
    }

    fn predict(&self, work: &[CostWork], rank: usize) -> CostPrediction {
        ConditionalSnapshot {
            original: &self.original,
            rank,
            weights: &[0.5, 0.5],
            dependence: CostScenarioDependence::SharedResidualRank,
            query_limits: CostQueryLimits::default(),
            config: &self.config,
            started: self.original.base.at,
            queries: &self.queries,
            cache: &self.cache,
            valid_until: &self.valid_until,
            diagnostics: &self.diagnostics,
        }
        .predict(
            &CostQuery {
                work,
                work_is_exact: true,
                execution: CostExecution::SingleInvocation,
            },
            self.original.base.at,
        )
    }
}

fn decode_work() -> CostWork {
    CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(1, 1).unwrap(),
        context_tokens: 2,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }
}

#[test]
fn provider_unknown_expiry_invalid_distribution_and_coupling_never_enter_cache() {
    let invalid = CostUnknownReason::UnsupportedWithDetail(
        CostUnsupportedDetail::InvalidScenarioDistribution,
    );
    for (reply, expected) in [
        (Reply::Unknown, CostUnknownReason::NoCoverage),
        (Reply::Expired, CostUnknownReason::Stale),
        (Reply::InvalidKnown, invalid),
        (Reply::DifferentWeights, invalid),
    ] {
        let harness = Harness::new(reply, 8192, 8);
        for _ in 0..2 {
            assert_eq!(
                harness.predict(&[decode_work()], 0),
                CostPrediction::Unknown(expected)
            );
        }
        let report = harness.diagnostics.finish(None);
        assert_eq!(harness.original.base.calls.load(Ordering::Relaxed), 2);
        assert_eq!(report.provider_calls, 2);
        assert_eq!(report.cache_hits, 0);
        assert_eq!(harness.queries.load(Ordering::Relaxed), 2);
        assert_eq!(
            *harness.valid_until.lock(),
            harness.original.base.at + Duration::from_secs(60)
        );
    }
}

#[test]
fn absent_insertion_token_still_validates_predicts_and_charges_actual_budget() {
    for bytes in [0, 1] {
        let harness = Harness::new(Reply::Known, bytes, 2);
        for rank in 0..2 {
            assert!(matches!(
                harness.predict(&[decode_work()], rank),
                CostPrediction::Known { .. }
            ));
        }
        let report = harness.diagnostics.finish(None);
        assert_eq!(report.provider_calls, 2);
        assert_eq!(report.cache_hits, 0);
        assert_eq!(harness.cache.lock().retained_bytes(), 0);
        assert_eq!(
            *harness.valid_until.lock(),
            harness.original.base.at + Duration::from_secs(1)
        );
        assert_eq!(
            harness.predict(&[decode_work()], 0),
            CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
        );
        let report = harness.diagnostics.finish(None);
        assert!(report.query_budget_exhausted);
        assert_eq!(report.provider_calls, 2);
        assert_eq!(harness.original.base.calls.load(Ordering::Relaxed), 2);
    }
    let harness = Harness::new(Reply::Known, 8192, 1);
    assert!(matches!(
        harness.predict(&[decode_work()], 0),
        CostPrediction::Known { .. }
    ));
    // The original quota is exactly used; a genuine cache hit still succeeds.
    assert!(matches!(
        harness.predict(&[decode_work()], 1),
        CostPrediction::Known { .. }
    ));
    let report = harness.diagnostics.finish(None);
    assert_eq!((report.provider_calls, report.cache_hits), (1, 1));
    assert!(!report.query_budget_exhausted);
    let changed = CostWork {
        context_tokens: 3,
        ..decode_work()
    };
    assert_eq!(
        harness.predict(&[changed], 0),
        CostPrediction::Unknown(CostUnknownReason::PlanningBudget)
    );
    assert!(harness.diagnostics.finish(None).query_budget_exhausted);
    assert_eq!(harness.original.base.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn cached_and_uncached_complete_scenarios_have_identical_endpoints_risks_and_choice() {
    let at = Instant::now();
    let mut requests = vec![request(), request()];
    requests[0].prompt_tokens = 4;
    requests[0].prefill_tokens = 0;
    requests[0].output_tokens = 0;
    requests[0].central_clock = RequestClock::default();
    requests[0].risk_clock = RequestClock::default();
    for request in &mut requests {
        request.output_cap = 3;
        request.scenario_end_tokens = 3;
    }
    let run = |cache_bytes| {
        let snapshot = Snapshot {
            at,
            calls: AtomicUsize::new(0),
        };
        let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
        config.planning.max_elapsed_us = 1_000_000;
        config.planning.max_scenarios = 2;
        config.planning.max_cost_cache_bytes = cache_bytes;
        let alternatives = [2, 1];
        let mut accumulator = ScenarioRiskAccumulator::new(
            &requests,
            alternatives.len(),
            &config,
            &[Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0,
            }); 2],
        )
        .unwrap();
        let mut observed = Vec::new();
        let mut diagnostics = ScenarioDiagnostics::default();
        let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
            EvaluationInput {
                snapshot: &snapshot,
                now: at,
                config: &config,
                execution: CostExecution::SequentialPrefillThenDecode,
                requests: &requests,
                resources: ResourceSupport {
                    waiting_requests: 0,
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &Split,
                static_action: &4,
                alternatives: &alternatives,
                horizon: EvaluationHorizon::ScenarioCompletion,
            },
            |_, _| Ok(()),
            |rank, comparison| {
                accumulator.observe(rank, comparison).unwrap();
                let trajectories = std::iter::once(comparison.static_result.complete().unwrap())
                    .chain(comparison.alternatives.iter().map(|r| r.as_ref().unwrap()))
                    .map(|result| {
                        (
                            result.target_reached_at,
                            result.central_elapsed,
                            result.risk_elapsed,
                            result.window_progress.clone(),
                            result
                                .requests
                                .iter()
                                .map(|request| {
                                    (
                                        request.id.clone(),
                                        request.prefill_tokens,
                                        request.output_tokens,
                                        request.scenario_end_tokens,
                                        request.central_clock,
                                        request.risk_clock,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                observed.push((
                    rank.index,
                    rank.weight,
                    comparison.window,
                    comparison.targets.clone(),
                    trajectories,
                ));
            },
            &mut diagnostics,
        )
        .unwrap();
        assert_eq!(summary.queries as u64, diagnostics.provider_calls);
        assert_eq!(summary.queries, snapshot.calls.load(Ordering::Relaxed));
        assert_eq!(summary.cache_hits as u64, diagnostics.cache_hits);
        assert!(!diagnostics.query_budget_exhausted);
        let decision = accumulator.finish(summary, &[None, None]).unwrap();
        (summary, decision, observed)
    };
    let (cached, cached_decision, cached_trace) = run(8192);
    let (uncached, uncached_decision, uncached_trace) = run(0);
    assert_eq!(cached_trace, uncached_trace);
    assert_eq!(cached.valid_until, uncached.valid_until);
    assert_eq!(cached.resolved_mass, 1.0);
    assert_eq!(uncached.resolved_mass, 1.0);
    assert!(cached.cache_hits > 0);
    assert_eq!(uncached.cache_hits, 0);
    assert_eq!(cached_decision.choice, uncached_decision.choice);
    assert_eq!(
        cached_decision.static_target_time,
        uncached_decision.static_target_time
    );
    assert_eq!(
        cached_decision.alternative_target_times,
        uncached_decision.alternative_target_times
    );
    assert_eq!(
        cached_decision.any_window_debt,
        uncached_decision.any_window_debt
    );
    for (cached, uncached) in std::iter::once(cached_decision.static_risk)
        .chain(cached_decision.alternative_risks)
        .zip(
            std::iter::once(uncached_decision.static_risk)
                .chain(uncached_decision.alternative_risks),
        )
    {
        assert_eq!(cached.ttft, uncached.ttft);
        assert_eq!(cached.tpot, uncached.tpot);
        assert_eq!(cached.itl, uncached.itl);
        assert_eq!(cached.completion_failure, uncached.completion_failure);
        assert_eq!(cached.tpot_undefined_mass, uncached.tpot_undefined_mass);
        assert_eq!(cached.request_count, uncached.request_count);
    }
}
