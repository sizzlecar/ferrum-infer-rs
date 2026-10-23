//! A real learner's current-step support is not complete trajectory support.
//! Synthetic eligible execution facts exercise Rust contracts, not GPU timing.
//! Stopping endpoints and product success are declared conditional inputs here;
//! this test does not claim that the stopping estimator learned either fact.

use super::*;
use ferrum_execution::model_executor::PrefillChunk;
use ferrum_scheduler::joint_policy::*;
use ferrum_types::SchedulerSloConfig;

fn decode(context: usize) -> CostWork {
    CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(context - 1, 1).unwrap(),
        context_tokens: context,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }
}

fn prefill(start: usize, tokens: usize) -> CostWork {
    CostWork {
        kind: AttemptWorkKind::Prefill,
        requested: ExecutionRange::new(start, tokens).unwrap(),
        context_tokens: 2,
        ..decode(2)
    }
}

fn completed(work: &[CostWork], at: Instant) -> ExecutionAttemptSummary {
    ExecutionAttemptSummary {
        started_at: at - Duration::from_millis(2),
        completed_at: at,
        eligibility: ExecutionAttemptEligibility::Completed,
        participants: work
            .iter()
            .map(|work| ExecutionAttemptParticipant {
                request_id: RequestId::new(),
                kind: work.kind,
                requested: work.requested,
                context_tokens: work.context_tokens,
                prefix_restore: work.prefix_restore,
                internally_reused_tokens: work.internally_reused_tokens,
                product_failed: false,
                outcome: AttemptParticipantOutcome::Completed {
                    work: match work.kind {
                        AttemptWorkKind::Prefill => ExecutedTokenWork::Prefill {
                            chunk: PrefillChunk::new(
                                work.requested.start(),
                                work.requested.as_range().len(),
                                work.context_tokens,
                            )
                            .unwrap(),
                            capacity_probe_count: 0,
                        },
                        AttemptWorkKind::Decode => ExecutedTokenWork::Decode {
                            position: work.requested.start(),
                        },
                    },
                    frontier_committed: true,
                },
            })
            .collect(),
    }
}

/// Each fit uses 6 earlier training, 3 subsequent validation, and 3 still later
/// calibration records. Every partition sees all three shapes. No label from
/// the calibration suffix enters fitting or parameter selection.
fn observe_partitioned_shapes(
    learner: &mut Learner,
    shared: &Shared,
    clock: &mut Instant,
    shapes: &[Vec<CostWork>; 3],
) {
    for index in 0..12 {
        *clock += Duration::from_millis(10);
        let sample = completed(&shapes[index % shapes.len()], *clock);
        learner.apply(shared, &sample, *clock);
    }
}

fn predict(snapshot: &snapshot::Snapshot, work: &[CostWork], at: Instant) -> CostPrediction {
    snapshot.predict(
        &CostQuery {
            work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        },
        at,
    )
}

fn outside() -> CostPrediction {
    CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
        CostUnsupportedDetail::OutsideObservedSupport,
    ))
}

/// Same prefill-then-decode invocation boundaries as the production split
/// executor. Only the first prefill quota changes; continuation is identical.
struct SplitContinuation;

impl TrajectoryAdapter<usize> for SplitContinuation {
    fn plan(
        &self,
        action: Option<&usize>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
            let mut members = Vec::new();
            for (index, request) in requests.iter().enumerate() {
                if request.finished()
                    || (request.output_tokens == 0) != (kind == AttemptWorkKind::Prefill)
                {
                    continue;
                }
                let work = if kind == AttemptWorkKind::Prefill {
                    prefill(
                        request.prefill_tokens,
                        (request.prompt_tokens - request.prefill_tokens)
                            .min(action.copied().unwrap_or(2)),
                    )
                } else {
                    decode(request.prompt_tokens + request.output_tokens)
                };
                members.push(GroupMember {
                    request: index,
                    work,
                });
            }
            if !members.is_empty() {
                plan.push_group(&members)?;
            }
        }
        Ok(())
    }
}

fn requests() -> [SimulatedRequest; 2] {
    let prior_token = RequestClock {
        first_output: Some(Duration::ZERO),
        last_output: Some(Duration::ZERO),
        ..Default::default()
    };
    [
        SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 1,
            prefill_tokens: 1,
            output_tokens: 1,
            output_cap: 3,
            scenario_end_tokens: 3,
            age: Duration::ZERO,
            central_clock: prior_token,
            risk_clock: prior_token,
        },
        SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 2,
            prefill_tokens: 0,
            output_tokens: 0,
            output_cap: 3,
            scenario_end_tokens: 3,
            age: Duration::ZERO,
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        },
    ]
}

fn evaluate(
    snapshot: &snapshot::Snapshot,
    requests: &[SimulatedRequest],
    config: &SchedulerSloConfig,
    now: Instant,
) -> (ScenarioPolicyDecision, ScenarioEvaluation) {
    let mut aggregate = ScenarioRiskAccumulator::new(
        requests,
        1,
        config,
        &[Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0,
        }); 2],
    )
    .unwrap();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
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
            adapter: &SplitContinuation,
            static_action: &2,
            alternatives: &[1],
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        |_, _| Ok(()),
        |rank, comparison| aggregate.observe(rank, comparison).unwrap(),
    )
    .unwrap();
    (aggregate.finish(summary, &[None]).unwrap(), summary)
}

#[test]
fn trajectory_support_needs_prior_long_context_evidence_in_both_real_learners() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let config = LearningConfig {
            training_samples: 6,
            validation_samples: 3,
            calibration_samples: 3,
            min_calibration_samples: 2,
            retrain_every: 12,
            window_samples: 12,
            work_bucket_boundaries: vec![4],
            ..config()
        };
        config.validate().unwrap();
        let shared = shared(&config);
        let mut learner = Learner::new(config, identity(), family);
        let mut clock = Instant::now();
        observe_partitioned_shapes(
            &mut learner,
            &shared,
            &mut clock,
            &[
                vec![prefill(0, 2)],
                vec![prefill(0, 1)],
                vec![prefill(1, 1)],
            ],
        );
        observe_partitioned_shapes(
            &mut learner,
            &shared,
            &mut clock,
            &[vec![decode(2)], vec![decode(3), decode(3)], vec![decode(3)]],
        );
        let old = learner.snapshot(clock).unwrap();
        let old_models: Vec<_> = old
            .domains
            .iter()
            .map(|domain| serde_json::to_value(&*domain.model).unwrap())
            .collect();
        assert_eq!(shared.stats().model_fits, 2);

        // Genuine completed evidence before the compared cohort's query. This
        // changes the initialization history, not the query, cap, or support rule.
        observe_partitioned_shapes(
            &mut learner,
            &shared,
            &mut clock,
            &[vec![decode(2)], vec![decode(3), decode(3)], vec![decode(4)]],
        );
        let initialized = learner.snapshot(clock).unwrap();
        assert_eq!(shared.stats().model_fits, 3);
        assert_eq!(shared.stats().invalid_attempts, 0);
        assert_eq!(shared.stats().excluded_attempts, 0);
        assert_eq!(old.domains.len(), 2);
        assert_eq!(initialized.domains.len(), 2);

        let cohort = requests();
        let mut policy: SchedulerSloConfig = "ttft:1000,tpot:1000,itl:1000".parse().unwrap();
        policy.planning.max_elapsed_us = 1_000_000;
        policy.planning.max_scenarios = 2;
        policy.joint.min_completion_rate = 1.0;
        policy.joint.request_deadline_ms = 1_000;

        // Both actual first invocation groups are already Known. Context4 is
        // inside old width/sum bounds but outside its max-context coordinate.
        for work in [vec![prefill(0, 2)], vec![decode(2)]] {
            assert!(matches!(
                predict(&old, &work, clock),
                CostPrediction::Known { .. }
            ));
        }
        let decode_domain = old
            .domains
            .iter()
            .find(|domain| domain.model.domain().phase == crate::model::Phase::Decode)
            .unwrap();
        let (minimum, maximum) = decode_domain.model.support_bounds();
        let future = WorkFeatures::extract(&[decode(4)], 2).unwrap();
        for (axis, value) in future.values().iter().enumerate() {
            assert_eq!(
                *value < minimum[axis] || *value > maximum[axis],
                axis == 8,
                "only max_context is missing in the cold fixture"
            );
        }
        assert_eq!(predict(&old, &[decode(4)], clock), outside());
        let (cold, cold_summary) = evaluate(&old, &cohort, &policy, clock);
        assert_eq!(cold_summary.partial_static_ranks, cold_summary.ranks);
        assert_eq!(cold.alternative_target_times, vec![None]);
        assert_eq!(
            cold.alternative_risks[0].classify(&policy),
            JointFeasibility::Unresolved
        );
        assert_eq!(cold.choice, PolicyChoice::Unresolved);

        // Real trained/calibrated support now covers every invocation in both
        // complete trajectories. Generous fixed SLOs establish conditional risk
        // feasibility, not throughput benefit or measured client compliance.
        let (covered, covered_summary) = evaluate(&initialized, &cohort, &policy, clock);
        assert_eq!(covered_summary.partial_static_ranks, 0);
        assert_eq!(covered_summary.unresolved_mass, 0.0);
        assert!(covered.static_target_time.is_some());
        assert!(covered.alternative_target_times[0].is_some());
        assert_eq!(
            covered.static_risk.classify(&policy),
            JointFeasibility::Feasible
        );
        assert_eq!(
            covered.alternative_risks[0].classify(&policy),
            JointFeasibility::Feasible
        );
        assert_eq!(covered.choice, PolicyChoice::Static);

        assert_eq!(predict(&old, &[decode(4)], clock), outside());
        assert_eq!(predict(&initialized, &[decode(5)], clock), outside());
        assert_eq!(
            predict(&initialized, &[prefill(0, 1), decode(2)], clock),
            CostPrediction::Unknown(CostUnknownReason::NoCoverage),
            "separate phase observations must not invent a mixed invocation model"
        );
        assert_eq!(
            predict(&initialized, &[decode(4)], clock + Duration::from_secs(31)),
            CostPrediction::Unknown(CostUnknownReason::Stale),
            "prior support does not renew calibration TTL"
        );
        assert_eq!(
            old.domains
                .iter()
                .map(|domain| serde_json::to_value(&*domain.model).unwrap())
                .collect::<Vec<_>>(),
            old_models,
        );
    }
}
