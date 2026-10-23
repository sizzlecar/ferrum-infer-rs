//! Completing at a query limit differs from actually refusing more work.

use super::*;

struct KnownCosts {
    at: Instant,
    calls: AtomicUsize,
    reject: Option<(usize, CostUnknownReason)>,
}

impl CostSnapshot for KnownCosts {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        Default::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("the scenario evaluator must request distributions")
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        let call = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some((rejected_call, reason)) = self.reject {
            if call == rejected_call {
                return CostScenarioPrediction::Unknown(reason);
            }
        }
        let full_prefill = query.work.iter().any(|work| {
            work.kind == AttemptWorkKind::Prefill && work.requested.as_range().len() == 2
        });
        let service_time = Duration::from_millis(if full_prefill { 100 } else { 10 });
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
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

struct Mixed {
    third_action_error: Option<Unsupported>,
}

impl TrajectoryAdapter<usize> for Mixed {
    fn plan(
        &self,
        first_action: Option<&usize>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        if first_action == Some(&3) {
            if let Some(reason) = self.third_action_error {
                return Err(reason);
            }
        }
        let mut members = Vec::new();
        for (index, request) in requests.iter().enumerate() {
            if request.finished() {
                continue;
            }
            let (kind, requested, context_tokens) = if request.output_tokens == 0 {
                (
                    AttemptWorkKind::Prefill,
                    ExecutionRange::new(
                        request.prefill_tokens,
                        (request.prompt_tokens - request.prefill_tokens)
                            .min(first_action.copied().unwrap_or(2)),
                    )
                    .unwrap(),
                    request.prompt_tokens,
                )
            } else {
                let context = request.prompt_tokens + request.output_tokens;
                (
                    AttemptWorkKind::Decode,
                    ExecutionRange::new(context - 1, 1).unwrap(),
                    context,
                )
            };
            members.push(GroupMember {
                request: index,
                work: CostWork {
                    kind,
                    requested,
                    context_tokens,
                    prefix_restore: AttemptPrefixRestore::NoneObserved,
                    internally_reused_tokens: None,
                },
            });
        }
        plan.push_group(&members)
    }
}

fn evaluate(
    alternatives: &[usize],
) -> (
    Result<ScenarioPolicyDecision, Unsupported>,
    ScenarioDiagnostics,
) {
    evaluate_with(alternatives, Case::default())
}

struct Case {
    max_queries: usize,
    cache_bytes: Option<usize>,
    adapter_error: Option<Unsupported>,
    provider_error: Option<(usize, CostUnknownReason)>,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            max_queries: 5,
            cache_bytes: None,
            adapter_error: None,
            provider_error: None,
        }
    }
}

fn evaluate_with(
    alternatives: &[usize],
    case: Case,
) -> (
    Result<ScenarioPolicyDecision, Unsupported>,
    ScenarioDiagnostics,
) {
    let source = KnownCosts {
        at: Instant::now(),
        calls: AtomicUsize::new(0),
        reject: case.provider_error,
    };
    let mut config: SchedulerSloConfig = "ttft:50,tpot:50,itl:50".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_queries = case.max_queries;
    if let Some(bytes) = case.cache_bytes {
        config.planning.max_cost_cache_bytes = bytes;
    }
    config.planning.max_candidates = 4;
    config.planning.max_scenarios = 1;
    config.joint.min_completion_rate = 1.0;
    config.joint.request_deadline_ms = 1_000;
    let mut decoder = request();
    decoder.output_cap = 3;
    decoder.scenario_end_tokens = 3;
    let requests = [
        decoder,
        SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 2,
            prefill_tokens: 0,
            output_tokens: 0,
            output_cap: 2,
            scenario_end_tokens: 2,
            age: Duration::ZERO,
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        },
    ];
    let success = [Some(ProbabilityInterval {
        lower: 1.0,
        upper: 1.0,
    }); 2];
    let mut aggregate =
        ScenarioRiskAccumulator::new(&requests, alternatives.len(), &config, &success).unwrap();
    let mut aggregate_error = None;
    let mut diagnostics = ScenarioDiagnostics::default();
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot: &source,
            now: source.at,
            config: &config,
            execution: CostExecution::SingleInvocation,
            requests: &requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &Mixed {
                third_action_error: case.adapter_error,
            },
            static_action: &2,
            alternatives,
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        |_, _| Ok(()),
        |rank, comparison| {
            assert!(comparison.static_result.complete().is_some());
            aggregate_error = aggregate.observe(rank, comparison).err();
        },
        &mut diagnostics,
    );
    assert_eq!(
        source.calls.load(Ordering::Relaxed) as u64,
        diagnostics.provider_calls
    );
    assert_eq!(diagnostics.rank_partial_static, 0);
    let decision = summary.and_then(|summary| {
        if let Some(error) = aggregate_error {
            return Err(error);
        }
        aggregate.finish(summary, &vec![None; alternatives.len()])
    });
    (decision, diagnostics)
}

#[test]
fn completing_the_comparison_at_the_exact_query_limit_can_choose_a_feasible_alternative() {
    let (decision, diagnostics) = evaluate(&[1]);
    let decision = decision.unwrap();
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    assert_eq!(
        decision.static_target_time,
        Some(Duration::from_millis(110))
    );
    assert_eq!(
        decision.alternative_target_times,
        vec![Some(Duration::from_millis(30))]
    );
    assert!(!diagnostics.query_budget_exhausted);
    assert_eq!(diagnostics.provider_calls, 5);
    assert_eq!(diagnostics.rank_completed, 1);
}

#[test]
fn later_real_query_exhaustion_cannot_select_an_earlier_feasible_alternative() {
    // Static and the first alternative have used five real provider calls.
    // Unlike the positive control, a second alternative now needs more work
    // and is actually refused by the simulation's query guard.
    let (decision, diagnostics) = evaluate(&[1, 2]);
    assert!(matches!(
        decision,
        Err(Unsupported::QueryBudget | Unsupported::Cost(CostUnknownReason::PlanningBudget))
    ));
    assert!(diagnostics.query_budget_exhausted);
    assert_eq!(diagnostics.provider_calls, 5);
    assert_eq!(diagnostics.simulation_queries, 5);
    let failure = diagnostics.terminal_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::SimulationQueryBudget);
    assert_eq!(failure.point.candidate, Some(2));
}

#[test]
fn typed_adapter_budget_refusal_stops_before_visiting_any_later_candidate() {
    for reason in [
        Unsupported::ParticipantBudget,
        Unsupported::CandidateBudget,
        Unsupported::TimeBudget,
    ] {
        let (decision, diagnostics) = evaluate_with(
            &[1, 3, 2],
            Case {
                max_queries: 32,
                adapter_error: Some(reason),
                ..Case::default()
            },
        );
        assert!(matches!(decision, Err(found) if found == reason));
        assert_eq!(diagnostics.provider_calls, 5);
        assert_eq!(diagnostics.simulation_queries, 5);
        assert_eq!(diagnostics.rank_completed, 0);
        assert!(!diagnostics.query_budget_exhausted);
        let failure = diagnostics.terminal_failure.unwrap();
        assert_eq!(failure.origin, FailureOrigin::Simulation);
        assert_eq!(failure.point.candidate, Some(2));
        assert_eq!(failure.point.group, None);
    }
}

#[test]
fn provider_planning_budget_refusal_invalidates_a_previously_complete_candidate() {
    let (decision, diagnostics) = evaluate_with(
        &[1, 2],
        Case {
            max_queries: 32,
            cache_bytes: Some(0),
            // Seed + static two calls + first candidate's three calls have
            // completed before the next candidate's original-provider call.
            provider_error: Some((7, CostUnknownReason::PlanningBudget)),
            ..Case::default()
        },
    );
    assert!(matches!(
        decision,
        Err(Unsupported::Cost(CostUnknownReason::PlanningBudget))
    ));
    assert_eq!(diagnostics.provider_calls, 7);
    assert_eq!(diagnostics.simulation_queries, 6);
    assert_eq!(diagnostics.rank_completed, 0);
    assert!(diagnostics.query_budget_exhausted);
    let failure = diagnostics.terminal_failure.unwrap();
    assert_eq!(failure.origin, FailureOrigin::Provider);
    assert_eq!(failure.point.candidate, Some(2));
}

#[test]
fn ordinary_support_unknown_remains_local_to_the_rejected_candidate() {
    for reason in [
        CostUnknownReason::NoCoverage,
        CostUnknownReason::QueryLimit,
        CostUnknownReason::UnsupportedWithDetail(CostUnsupportedDetail::OutsideObservedSupport),
    ] {
        let (decision, diagnostics) = evaluate_with(
            &[1, 2],
            Case {
                max_queries: 32,
                cache_bytes: Some(0),
                provider_error: Some((7, reason)),
                ..Case::default()
            },
        );
        let decision = decision.unwrap();
        assert_eq!(decision.choice, PolicyChoice::Alternative(0));
        assert_eq!(
            decision.alternative_target_times[0],
            Some(Duration::from_millis(30))
        );
        assert!(decision.alternative_target_times[1].is_none());
        assert_eq!(diagnostics.provider_calls, 7);
        assert_eq!(diagnostics.rank_completed, 1);
        assert!(!diagnostics.query_budget_exhausted);
        assert!(diagnostics.terminal_failure.is_none());
        let failure = diagnostics.first_failure.unwrap();
        assert_eq!(failure.reason, Unsupported::Cost(reason));
        assert_eq!(failure.origin, FailureOrigin::Provider);
        assert_eq!(failure.point.candidate, Some(2));
    }
}
