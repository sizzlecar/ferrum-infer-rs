//! Model-relative counterexamples, not executable production action proposals.
//! All projection, continuation, simulation and risk aggregation use production
//! code. The two multi-wave prefixes below exist only in this test adapter.

use super::*;
use ferrum_execution::execution_attempt::AttemptWorkKind;
use ferrum_execution::model_executor::PrefillShapePolicy;
use ferrum_types::SchedulerSloConfig;
use std::cell::{Cell, RefCell};

mod future_waiter;

struct ExactCosts(Instant);

impl ExactCosts {
    fn time(query: &CostQuery<'_>) -> Duration {
        let prefill = query
            .work
            .iter()
            .filter(|work| work.kind == AttemptWorkKind::Prefill)
            .map(|work| work.requested.as_range().len() as u64)
            .sum::<u64>();
        if prefill > 0 {
            Duration::from_millis(10 * prefill)
        } else {
            // One full decode invocation, not one charge per participant.
            Duration::from_micros(100)
        }
    }
}

impl CostSnapshot for ExactCosts {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("conditional scenarios must not make another scalar prediction")
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        let time = Self::time(query);
        CostScenarioPrediction::Known {
            central_service_time: time,
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time: time,
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.0,
            expires_at: self.0 + Duration::from_secs(60),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ProbeAction {
    Current(FirstAction),
    /// Freeze the original decoder's endpoint, at most63 additional steps.
    DrainOriginalDecoder,
    /// Apply a positive cap until the original prefill completes; at most32
    /// steps. Later arrivals are absent from this closed-cohort counterexample.
    OriginalPrefillPhase(NonZeroUsize),
}

#[derive(Debug)]
struct PrefixTrace {
    action: ProbeAction,
    /// Actual simulator frontiers at successive phase calls.
    states: Vec<(usize, usize)>,
    continuation_entered: bool,
}

struct PhaseAdapter {
    original: CohortTrajectory,
    action: Cell<ProbeAction>,
    traces: RefCell<Vec<PrefixTrace>>,
}

impl TrajectoryAdapter<ProbeAction> for PhaseAdapter {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        Some(&self.original.admission)
    }

    fn plan(
        &self,
        first_action: Option<&ProbeAction>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> std::result::Result<(), Unsupported> {
        if let Some(action) = first_action {
            self.action.set(*action);
            self.traces.borrow_mut().push(PrefixTrace {
                action: *action,
                states: Vec::new(),
                continuation_entered: false,
            });
        }
        let action = self.action.get();
        let prefix = match action {
            ProbeAction::Current(action) => first_action.map(|_| action),
            ProbeAction::DrainOriginalDecoder if requests[0].output_tokens < 64 => {
                Some(FirstAction::DecodeOnly)
            }
            ProbeAction::OriginalPrefillPhase(limit)
                if requests[1].prefill_tokens < requests[1].prompt_tokens =>
            {
                Some(FirstAction::PrefillLimit(limit))
            }
            _ => None,
        };
        if !matches!(action, ProbeAction::Current(_)) {
            let mut traces = self.traces.borrow_mut();
            let trace = traces.last_mut().unwrap();
            if prefix.is_some() {
                assert!(
                    !trace.continuation_entered,
                    "a finished phase cannot restart"
                );
                trace
                    .states
                    .push((requests[0].output_tokens, requests[1].prefill_tokens));
                assert!(trace.states.len() <= 63);
            } else {
                trace.continuation_entered = true;
            }
        }
        // Empty first_wave deliberately selects the real natural members at
        // each frontier. No initial work ranges are replayed in later waves.
        self.original.plan(prefix.as_ref(), requests, plan)
    }
}

fn fixture() -> (SchedulerSloConfig, Vec<SimulatedRequest>, PhaseAdapter) {
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:50,itl:15".parse().unwrap();
    config.joint.request_deadline_ms = 2000;
    config.joint.min_completion_rate = 1.0;
    config.planning.max_scenarios = 2;
    // Host-time resource guard only: this direct policy fixture supplies its
    // clocks explicitly and does not assert an Engine control-delay model.
    config.planning.max_elapsed_us = 5_000_000;
    let decoder_clock = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::ZERO),
        ..Default::default()
    };
    let requests = vec![
        SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 1,
            prefill_tokens: 1,
            output_tokens: 1,
            output_cap: 64,
            scenario_end_tokens: 64,
            age: Duration::ZERO,
            central_clock: decoder_clock,
            risk_clock: decoder_clock,
        },
        SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 32,
            prefill_tokens: 0,
            output_tokens: 0,
            output_cap: 64,
            scenario_end_tokens: 64,
            age: Duration::ZERO,
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        },
    ];
    let original = CohortTrajectory {
        first_wave: Vec::new(),
        cap_commit_targets: Vec::new(),
        shapes: vec![
            RequestShape {
                restored_tokens: 0,
                execution_ceiling: None,
                phase_cap: None,
                decode_only_allowed: true,
            };
            2
        ],
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
    (
        config,
        requests,
        PhaseAdapter {
            original,
            action: Cell::new(ProbeAction::Current(FirstAction::Static)),
            traces: RefCell::new(Vec::new()),
        },
    )
}

fn current_alternatives() -> Vec<ProbeAction> {
    vec![
        ProbeAction::Current(FirstAction::PrefillLimit(NonZeroUsize::new(2).unwrap())),
        ProbeAction::Current(FirstAction::PrefillLimit(NonZeroUsize::new(1).unwrap())),
        ProbeAction::Current(FirstAction::DecodeOnly),
    ]
}

fn evaluate(
    config: &SchedulerSloConfig,
    requests: &[SimulatedRequest],
    adapter: &PhaseAdapter,
    alternatives: &[ProbeAction],
) -> (ScenarioPolicyDecision, Vec<Vec<TrajectoryResult>>) {
    let now = Instant::now();
    let snapshot = ExactCosts(now);
    let mut aggregate = ScenarioRiskAccumulator::new(
        requests,
        alternatives.len(),
        config,
        &vec![
            Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0
            });
            requests.len()
        ],
    )
    .unwrap();
    let mut trajectories = Vec::new();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
        EvaluationInput {
            snapshot: &snapshot,
            now,
            config,
            execution: CostExecution::SequentialPrefillThenDecode,
            requests,
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter,
            static_action: &ProbeAction::Current(FirstAction::Static),
            alternatives,
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        // Both ranks have explicit, certain64-token endpoints. This is a
        // typed stopping fixture, never filling actual Unknown mass at cap.
        |_, assigned| {
            assert!(assigned
                .iter()
                .all(|request| request.scenario_end_tokens == 64));
            Ok(())
        },
        |rank, comparison| {
            let results = std::iter::once(comparison.static_result.complete().unwrap())
                .chain(
                    comparison
                        .alternatives
                        .iter()
                        .map(|result| result.as_ref().unwrap()),
                )
                .cloned()
                .collect::<Vec<_>>();
            for result in &results {
                for (actual, initial) in result.requests.iter().zip(requests) {
                    assert_eq!(actual.id, initial.id);
                    assert_eq!(actual.prefill_tokens, initial.prompt_tokens);
                    assert_eq!(actual.output_tokens, initial.scenario_end_tokens);
                    assert!(actual.central_clock.completed.is_some());
                    assert_eq!(actual.central_clock.interval_count, 63);
                }
            }
            aggregate.observe(rank, comparison).unwrap();
            trajectories.push(results);
        },
    )
    .unwrap();
    assert_eq!(summary.partial_static_ranks, 0);
    assert_eq!(summary.unresolved_mass, 0.0);
    let decision = aggregate
        .finish(summary, &vec![None; alternatives.len()])
        .unwrap();
    (decision, trajectories)
}

#[test]
fn cap64_first_wave_edits_can_all_fail_while_finite_positive_prefill_phase_succeeds() {
    let (config, requests, adapter) = fixture();
    let mut alternatives = current_alternatives();
    let (current, _) = evaluate(&config, &requests, &adapter, &alternatives);
    assert_eq!(current.choice, PolicyChoice::AllKnownInfeasible);
    for risk in std::iter::once(&current.static_risk).chain(&current.alternative_risks) {
        assert!(risk.itl.unwrap().lower >= 8.0 / 126.0);
        assert_eq!(risk.ttft.upper, 0.0);
        assert_eq!(risk.tpot.unwrap().upper, 0.0);
        assert_eq!(risk.completion_failure.upper, 0.0);
    }
    adapter.traces.borrow_mut().clear();
    alternatives.push(ProbeAction::DrainOriginalDecoder);
    alternatives.push(ProbeAction::OriginalPrefillPhase(
        NonZeroUsize::new(1).unwrap(),
    ));
    let (expanded, results) = evaluate(&config, &requests, &adapter, &alternatives);
    // Prefix chunk1 finishes sooner than draining the decoder first. Both
    // are feasible with full endpoints, even though static is slightly faster.
    assert_eq!(expanded.choice, PolicyChoice::Alternative(4));
    for index in [3, 4] {
        assert_eq!(
            expanded.alternative_risks[index].classify(&config),
            JointFeasibility::Feasible
        );
        assert!(
            expanded.alternative_target_times[index].unwrap()
                > expanded.static_target_time.unwrap()
        );
        assert!(expanded.any_window_debt[index]);
    }
    for rank in results {
        assert_eq!(rank[0].central_elapsed, Duration::from_micros(327_100));
        assert_eq!(rank[4].central_elapsed, Duration::from_micros(332_600));
        assert_eq!(rank[5].central_elapsed, Duration::from_micros(329_500));
        assert_eq!(
            rank[4].requests[1].central_clock.first_output,
            Some(Duration::from_micros(326_300))
        );
        assert_eq!(
            rank[5].requests[1].central_clock.first_output,
            Some(Duration::from_micros(323_100))
        );
    }
    for trace in adapter.traces.borrow().iter() {
        match trace.action {
            ProbeAction::DrainOriginalDecoder => {
                assert!(trace.continuation_entered);
                assert_eq!(
                    trace.states,
                    (1..64).map(|output| (output, 0)).collect::<Vec<_>>()
                );
            }
            ProbeAction::OriginalPrefillPhase(_) => {
                assert!(trace.continuation_entered);
                // Actual simulator states prove positive prefill progress in
                // every phase wave; no paused or silently discarded work.
                assert_eq!(
                    trace.states,
                    (0..32)
                        .map(|prefill| (prefill + 1, prefill))
                        .collect::<Vec<_>>()
                );
            }
            ProbeAction::Current(_) => {}
        }
    }
}

#[test]
fn historical_ttft_violation_cannot_be_repaired_even_by_feasible_future_phase_actions() {
    let (config, mut requests, adapter) = fixture();
    let mut alternatives = current_alternatives();
    alternatives.push(ProbeAction::DrainOriginalDecoder);
    alternatives.push(ProbeAction::OriginalPrefillPhase(
        NonZeroUsize::new(1).unwrap(),
    ));
    let (without_debt, _) = evaluate(&config, &requests, &adapter, &alternatives);
    assert_eq!(without_debt.choice, PolicyChoice::Alternative(4));
    let late = Duration::from_millis(1100);
    requests[0].age = late;
    requests[0].central_clock.first_output = Some(late);
    requests[0].central_clock.last_output = Some(late);
    requests[0].risk_clock = requests[0].central_clock;
    let (with_debt, _) = evaluate(&config, &requests, &adapter, &alternatives);
    assert_eq!(with_debt.choice, PolicyChoice::AllKnownInfeasible);
    for risk in std::iter::once(&with_debt.static_risk).chain(&with_debt.alternative_risks) {
        assert_eq!(
            risk.ttft,
            ProbabilityInterval {
                lower: 0.5,
                upper: 0.5
            }
        );
        assert_eq!(risk.completion_failure.upper, 0.0);
        assert_eq!(risk.request_count, 2);
    }
    // The two finite prefixes still satisfy every other metric. They cannot
    // undo an already late first text event by accelerating future work.
    for index in [3, 4] {
        assert_eq!(with_debt.alternative_risks[index].tpot.unwrap().upper, 0.0);
        assert_eq!(with_debt.alternative_risks[index].itl.unwrap().upper, 0.0);
    }
}
