//! Cost support gaps must preserve evidence without inventing a completion.

use super::*;

mod decision_diagnostics;

#[derive(Clone, Copy)]
enum RejectAt {
    NoGap { static_tail_ms: u64 },
    StaticContinuation,
    StaticAndSingletonContext(usize),
    EveryDecodeContinuation,
    Seed,
    FirstGroupAfterAssignment,
}

struct PartialSupportSnapshot {
    at: Instant,
    calls: AtomicUsize,
    reject_at: RejectAt,
    reason: CostUnknownReason,
    static_first_ms: u64,
    alternative_step_ms: u64,
}

impl PartialSupportSnapshot {
    fn new(reject_at: RejectAt) -> Self {
        Self {
            at: Instant::now(),
            calls: AtomicUsize::new(0),
            reject_at,
            reason: CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport,
            ),
            static_first_ms: 100,
            alternative_step_ms: 10,
        }
    }
}

impl CostSnapshot for PartialSupportSnapshot {
    fn revision(&self) -> u64 {
        11
    }

    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("the scenario adapter must query one complete distribution")
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert!(query.work_is_exact);
        assert_eq!(query.execution, CostExecution::SingleInvocation);
        let full_prefill = query.work.iter().any(|work| {
            work.kind == AttemptWorkKind::Prefill && work.requested.as_range().len() == 2
        });
        let decode_only = query
            .work
            .iter()
            .all(|work| work.kind == AttemptWorkKind::Decode);
        let rejected = match self.reject_at {
            RejectAt::NoGap { .. } => false,
            RejectAt::StaticContinuation => decode_only && query.work.len() == 2,
            RejectAt::StaticAndSingletonContext(context) => {
                decode_only
                    && (query.work.len() == 2
                        || query.work.iter().any(|work| work.context_tokens == context))
            }
            RejectAt::EveryDecodeContinuation => decode_only,
            RejectAt::Seed => full_prefill,
            RejectAt::FirstGroupAfterAssignment => full_prefill && query.work.len() == 1,
        };
        if rejected {
            return CostScenarioPrediction::Unknown(self.reason);
        }
        let static_tail = match self.reject_at {
            RejectAt::NoGap { static_tail_ms } if decode_only && query.work.len() == 2 => {
                Some(static_tail_ms)
            }
            _ => None,
        };
        let duration = Duration::from_millis(if let Some(static_tail_ms) = static_tail {
            static_tail_ms
        } else if full_prefill {
            self.static_first_ms
        } else {
            self.alternative_step_ms
        });
        CostScenarioPrediction::Known {
            central_service_time: duration,
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time: duration,
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: count,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

/// Both actions keep the same participants. Only the first prefill is halved;
/// all later waves follow the same decode-first continuation.
struct MixedContinuation;

impl TrajectoryAdapter<usize> for MixedContinuation {
    fn plan(
        &self,
        action: Option<&usize>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let mut members = Vec::new();
        for kind in [AttemptWorkKind::Decode, AttemptWorkKind::Prefill] {
            for (index, request) in requests.iter().enumerate() {
                if request.finished()
                    || (request.output_tokens == 0) != (kind == AttemptWorkKind::Prefill)
                {
                    continue;
                }
                let (requested, context_tokens) = if kind == AttemptWorkKind::Prefill {
                    (
                        ExecutionRange::new(
                            request.prefill_tokens,
                            (request.prompt_tokens - request.prefill_tokens)
                                .min(action.copied().unwrap_or(2)),
                        )
                        .unwrap(),
                        request.prompt_tokens,
                    )
                } else {
                    let context = request.prompt_tokens + request.output_tokens;
                    (ExecutionRange::new(context - 1, 1).unwrap(), context)
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
        }
        plan.push_group(&members)
    }
}

fn partial_requests() -> [SimulatedRequest; 2] {
    let mut decoder = request();
    decoder.output_cap = 3;
    decoder.scenario_end_tokens = 3;
    let prefill = SimulatedRequest {
        id: RequestId::new(),
        prompt_tokens: 2,
        prefill_tokens: 0,
        output_tokens: 0,
        output_cap: 2,
        scenario_end_tokens: 2,
        age: Duration::ZERO,
        central_clock: RequestClock::default(),
        risk_clock: RequestClock::default(),
    };
    [decoder, prefill]
}

fn partial_config() -> SchedulerSloConfig {
    let mut config: SchedulerSloConfig = "ttft:50,tpot:50,itl:50".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    config.joint.min_completion_rate = 1.0;
    config.joint.request_deadline_ms = 1_000;
    config
}

fn evaluate_partial(
    snapshot: &PartialSupportSnapshot,
    config: &SchedulerSloConfig,
) -> (
    Result<ScenarioPolicyDecision, Unsupported>,
    ScenarioDiagnostics,
) {
    let requests = partial_requests();
    evaluate_partial_with(
        snapshot,
        config,
        &requests,
        EvaluationHorizon::ScenarioCompletion,
        |_, _| Ok(()),
    )
}

fn evaluate_partial_with(
    snapshot: &PartialSupportSnapshot,
    config: &SchedulerSloConfig,
    requests: &[SimulatedRequest],
    horizon: EvaluationHorizon,
    assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
) -> (
    Result<ScenarioPolicyDecision, Unsupported>,
    ScenarioDiagnostics,
) {
    evaluate_partial_with_actions(snapshot, config, requests, horizon, &[1], assign)
}

fn evaluate_partial_with_actions(
    snapshot: &PartialSupportSnapshot,
    config: &SchedulerSloConfig,
    requests: &[SimulatedRequest],
    horizon: EvaluationHorizon,
    alternatives: &[usize],
    assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
) -> (
    Result<ScenarioPolicyDecision, Unsupported>,
    ScenarioDiagnostics,
) {
    let success = vec![
        Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0,
        });
        requests.len()
    ];
    let mut aggregate =
        ScenarioRiskAccumulator::new(requests, alternatives.len(), config, &success).unwrap();
    let mut aggregate_error = None;
    let mut diagnostics = ScenarioDiagnostics::default();
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot,
            now: snapshot.at,
            config,
            execution: CostExecution::SingleInvocation,
            requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &MixedContinuation,
            static_action: &2,
            alternatives,
            horizon,
        },
        assign,
        |rank, comparison| {
            if aggregate_error.is_none() {
                aggregate_error = aggregate.observe(rank, comparison).err();
            }
        },
        &mut diagnostics,
    );
    let decision = summary.and_then(|summary| {
        if let Some(error) = aggregate_error {
            return Err(error);
        }
        aggregate.finish(summary, &vec![None; alternatives.len()])
    });
    assert_eq!(
        diagnostics.provider_calls,
        snapshot.calls.load(Ordering::Relaxed) as u64
    );
    assert!(diagnostics.provider_calls <= config.planning.max_queries as u64);
    (decision, diagnostics)
}

#[test]
fn violated_static_prefix_can_be_replaced_by_a_fully_known_complete_trajectory() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    let config = partial_config();
    let (decision, diagnostics) = evaluate_partial(&snapshot, &config);
    let decision = decision.unwrap();
    assert_eq!(diagnostics.rank_partial_static, 2);
    assert_eq!(diagnostics.rank_completed, 0);
    assert_eq!(diagnostics.rank_unresolved, 0);
    // Static first wave proves B's TTFT violated. Its next invocation is
    // unsupported; the completed candidate needs different, supported groups.
    assert_eq!(decision.static_risk.ttft.lower, 0.5);
    assert_eq!(
        decision.static_risk.classify(&config),
        JointFeasibility::Infeasible
    );
    assert_eq!(
        decision.alternative_risks[0].classify(&config),
        JointFeasibility::Feasible
    );
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    // 30 ms covers A's third and B's second token. Reusing the short static
    // prefix as the target would report the earlier 20 ms crossing instead.
    assert_eq!(
        decision.alternative_target_times,
        vec![Some(Duration::from_millis(30))]
    );
    assert!(decision.static_target_time.is_none());
    assert_eq!(decision.any_window_debt, vec![false]);
}

#[test]
fn an_unknown_tail_needed_by_both_actions_cannot_satisfy_full_completion() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::EveryDecodeContinuation);
    let config = partial_config();
    let (decision, _) = evaluate_partial(&snapshot, &config);
    let decision = decision.unwrap();
    assert_eq!(
        decision.static_risk.classify(&config),
        JointFeasibility::Infeasible
    );
    assert!(decision.alternative_risks[0].completion_failure.upper > 0.0);
    assert_ne!(
        decision.alternative_risks[0].classify(&config),
        JointFeasibility::Feasible
    );
    assert!(decision.alternative_target_times[0].is_none());
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
}

#[test]
fn a_feasible_alternative_does_not_rescue_an_unresolved_static_prefix() {
    let mut snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    snapshot.static_first_ms = 10;
    snapshot.alternative_step_ms = 1;
    let config = partial_config();
    let (decision, _) = evaluate_partial(&snapshot, &config);
    let decision = decision.unwrap();
    assert_eq!(
        decision.static_risk.classify(&config),
        JointFeasibility::Unresolved
    );
    assert_eq!(
        decision.alternative_risks[0].classify(&config),
        JointFeasibility::Feasible
    );
    assert_eq!(
        decision.alternative_target_times[0],
        Some(Duration::from_millis(3))
    );
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
}

#[test]
fn partial_rescue_can_complete_after_the_static_prefix_without_inventing_a_window() {
    let mut snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    snapshot.alternative_step_ms = 40;
    let mut config = partial_config();
    config.ttft_ms = 90.0;
    let (decision, _) = evaluate_partial(&snapshot, &config);
    let decision = decision.unwrap();
    assert_eq!(
        decision.static_risk.classify(&config),
        JointFeasibility::Infeasible
    );
    assert_eq!(
        decision.alternative_risks[0].classify(&config),
        JointFeasibility::Feasible
    );
    assert_eq!(
        decision.alternative_target_times[0],
        Some(Duration::from_millis(120))
    );
    assert!(decision.static_target_time.is_none());
    // Actual static completion might be before or after 120 ms. Rescue of an
    // already violated static policy does not claim a common-window speedup.
    assert_eq!(decision.any_window_debt, vec![true]);
    assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    assert_eq!(decision.observation.candidates.gates.eligible, 1);
}

#[test]
fn partial_rescue_matches_complete_static_rescue_on_both_sides_of_candidate_time() {
    for static_tail_ms in [1, 100] {
        let mut snapshot = PartialSupportSnapshot::new(RejectAt::NoGap { static_tail_ms });
        snapshot.alternative_step_ms = 40;
        let mut config = partial_config();
        config.ttft_ms = 90.0;
        let (decision, diagnostics) = evaluate_partial(&snapshot, &config);
        let decision = decision.unwrap();
        assert_eq!(diagnostics.rank_partial_static, 0);
        assert_eq!(
            decision.static_risk.classify(&config),
            JointFeasibility::Infeasible
        );
        assert_eq!(
            decision.alternative_risks[0].classify(&config),
            JointFeasibility::Feasible
        );
        assert_eq!(
            decision.static_target_time,
            Some(Duration::from_millis(100 + static_tail_ms))
        );
        assert_eq!(
            decision.alternative_target_times[0],
            Some(Duration::from_millis(120))
        );
        assert_eq!(decision.any_window_debt, vec![static_tail_ms == 1]);
        assert_eq!(decision.choice, PolicyChoice::Alternative(0));
    }
}

#[test]
fn partial_rescue_direct_choose_requires_final_endpoints_and_completion_clocks() {
    let mut snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    snapshot.alternative_step_ms = 40;
    let mut config = partial_config();
    config.ttft_ms = 90.0;
    let requests = partial_requests();
    let mut diagnostics = ScenarioDiagnostics::default();
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot: &snapshot,
            now: snapshot.at,
            config: &config,
            execution: CostExecution::SingleInvocation,
            requests: &requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &MixedContinuation,
            static_action: &2,
            alternatives: &[1],
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        |_, _| Ok(()),
        |_, comparison| {
            let choose = |value: &Comparison, static_status| {
                SloPolicy::choose(value, static_status, &[JointFeasibility::Feasible], &[None])
            };
            assert_eq!(
                choose(comparison, JointFeasibility::Infeasible),
                PolicyChoice::Alternative(0)
            );
            assert_eq!(
                choose(comparison, JointFeasibility::Unresolved),
                PolicyChoice::Unresolved
            );
            let StaticTrajectory::OutsideSupportPrefix(prefix) = &comparison.static_result else {
                panic!("expected the actual static support gap");
            };
            let candidate = comparison.alternatives[0].as_ref().unwrap();
            assert!(candidate.has_completed_endpoints());
            assert_eq!(candidate.central_elapsed, Duration::from_millis(120));
            assert!(!comparison.partial_window_covered(candidate));
            for missing in 0..4 {
                let mut candidate = candidate.clone();
                match missing {
                    0 => candidate.requests[0].prefill_tokens -= 1,
                    1 => candidate.requests[0].output_tokens -= 1,
                    2 => candidate.requests[0].central_clock.completed = None,
                    _ => candidate.window_progress[0].output_tokens -= 1,
                }
                // A caller-supplied Feasible risk cannot substitute for the
                // endpoint evidence required by this public selection API.
                let incomplete = Comparison {
                    revision: comparison.revision,
                    window: comparison.window,
                    targets: comparison.targets.clone(),
                    static_result: StaticTrajectory::OutsideSupportPrefix(TrajectoryPrefix {
                        requests: prefix.requests.clone(),
                        central_elapsed: prefix.central_elapsed,
                        risk_elapsed: prefix.risk_elapsed,
                        queries: prefix.queries,
                        cost_risk: prefix.cost_risk,
                        unsupported: prefix.unsupported,
                    }),
                    alternatives: vec![Ok(candidate)],
                };
                assert_eq!(
                    choose(&incomplete, JointFeasibility::Infeasible),
                    PolicyChoice::Unresolved
                );
            }
        },
        &mut diagnostics,
    )
    .unwrap();
    assert_eq!(summary.partial_static_ranks, 2);
    assert_eq!(diagnostics.rank_partial_static, 2);
}

#[test]
fn one_unsupported_candidate_rank_retains_its_completion_failure_mass() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticAndSingletonContext(4));
    let config = partial_config();
    let mut requests = partial_requests();
    requests[1].output_cap = 3;
    requests[1].scenario_end_tokens = 3;
    let (decision, _) = evaluate_partial_with(
        &snapshot,
        &config,
        &requests,
        EvaluationHorizon::ScenarioCompletion,
        |rank, requests| {
            requests[1].scenario_end_tokens = if rank.index == 0 { 2 } else { 3 };
            Ok(())
        },
    );
    let decision = decision.unwrap();
    assert_eq!(
        decision.static_risk.classify(&config),
        JointFeasibility::Infeasible
    );
    assert!(decision.alternative_risks[0].completion_failure.upper >= 0.5);
    assert!(decision.alternative_target_times[0].is_none());
    assert_eq!(decision.choice, PolicyChoice::Unresolved);
}

#[test]
fn finite_round_horizon_cannot_replace_unknown_static_progress_with_full_endpoints() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    let (decision, _) = evaluate_partial_with(
        &snapshot,
        &partial_config(),
        &partial_requests(),
        EvaluationHorizon::StaticRounds(2),
        |_, _| Ok(()),
    );
    assert!(matches!(decision, Err(Unsupported::Cost(reason)) if reason == snapshot.reason));
}

#[test]
fn only_explicit_outside_support_can_become_a_partial_static_trajectory() {
    for reason in [
        CostUnknownReason::NoCoverage,
        CostUnknownReason::Stale,
        CostUnknownReason::InexactWork,
        CostUnknownReason::UnsupportedDomain,
        CostUnknownReason::QueryLimit,
        CostUnknownReason::PlanningBudget,
        CostUnknownReason::UnsupportedWithDetail(CostUnsupportedDetail::ExecutionIdentityMismatch),
        CostUnknownReason::UnsupportedWithDetail(CostUnsupportedDetail::InvalidWorkProjection),
        CostUnknownReason::UnsupportedWithDetail(
            CostUnsupportedDetail::InvalidScenarioDistribution,
        ),
    ] {
        let mut snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
        snapshot.reason = reason;
        let (decision, _) = evaluate_partial(&snapshot, &partial_config());
        assert!(
            matches!(decision, Err(Unsupported::Cost(found)) if found == reason),
            "{reason:?}"
        );
    }
}

#[test]
fn missing_seed_distribution_cannot_invent_shared_rank_weights() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::Seed);
    let (decision, diagnostics) = evaluate_partial(&snapshot, &partial_config());
    assert!(matches!(decision, Err(Unsupported::Cost(reason)) if reason == snapshot.reason));
    assert_eq!(diagnostics.provider_calls, 1);
    assert_eq!(diagnostics.rank_attempts, 0);
}

#[test]
fn known_seed_does_not_invent_a_prefix_when_the_first_assigned_group_is_unknown() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::FirstGroupAfterAssignment);
    let (decision, diagnostics) = evaluate_partial_with(
        &snapshot,
        &partial_config(),
        &partial_requests(),
        EvaluationHorizon::ScenarioCompletion,
        |_, requests| {
            // The seed mixed invocation was Known. Once A's actual assigned
            // endpoint is its existing one token, the first simulated group
            // contains only B and has no supported elapsed prefix.
            requests[0].scenario_end_tokens = requests[0].output_tokens;
            Ok(())
        },
    );
    assert!(matches!(decision, Err(Unsupported::Cost(reason)) if reason == snapshot.reason));
    assert_eq!(diagnostics.provider_calls, 2);
    assert_eq!(diagnostics.rank_partial_static, 0);
    assert_eq!(diagnostics.rank_completed, 0);
}

#[test]
fn query_exhaustion_is_not_a_supported_partial_cost_tail() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    let mut config = partial_config();
    config.planning.max_queries = 1;
    let (decision, diagnostics) = evaluate_partial(&snapshot, &config);
    assert!(matches!(
        decision,
        Err(Unsupported::QueryBudget | Unsupported::Cost(CostUnknownReason::PlanningBudget))
    ));
    assert!(diagnostics.query_budget_exhausted);
}

#[test]
fn later_query_exhaustion_invalidates_an_already_feasible_partial_rescue() {
    let snapshot = PartialSupportSnapshot::new(RejectAt::StaticContinuation);
    let mut config = partial_config();
    // Two static invocations (one unknown), then the first alternative's three
    // supported invocations. This completes that alternative at 30 ms before
    // the budget rejects the next alternative's first invocation.
    config.planning.max_queries = 5;
    let (decision, diagnostics) = evaluate_partial_with_actions(
        &snapshot,
        &config,
        &partial_requests(),
        EvaluationHorizon::ScenarioCompletion,
        &[1, 2],
        |_, _| Ok(()),
    );
    assert!(matches!(
        decision,
        Err(Unsupported::QueryBudget | Unsupported::Cost(CostUnknownReason::PlanningBudget))
    ));
    assert_eq!(diagnostics.provider_calls, 5);
    assert!(diagnostics.query_budget_exhausted);
}
