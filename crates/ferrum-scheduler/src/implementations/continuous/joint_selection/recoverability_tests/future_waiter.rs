//! A model-relative opportunity outside the current first-wave action set.
//! The counterfactual freezes a cap in a test adapter, never in the scheduler.
//! It does not grant admission, shorten endpoints, or predict actual close time.

use super::*;
use ferrum_execution::model_executor::{
    PersistentCapacityMember, PersistentCapacityQuery, PersistentCapacityState,
    PersistentCohortCapacitySnapshot,
};
use ferrum_execution::token_execution::ExecutorSequenceIssuer;
use ferrum_interfaces::request_observation::*;
use ferrum_types::{ModelId, SamplingParams};

mod deferred_commit;

struct ScaledCosts(Instant);

impl CostSnapshot for ScaledCosts {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("conditional scenarios must not query the scalar predictor")
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        // The same 10:0.1 ratio as the original recoverability fixture, in
        // seconds. Keep the real one-second planning allowance without a
        // machine-speed-dependent ITL margin.
        let time = ExactCosts::time(query).checked_mul(1000).unwrap();
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
            expires_at: self.0 + Duration::from_secs(3600),
        }
    }

    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        assert_eq!(query.cap, 64);
        StoppingForecast {
            stops: vec![StoppingMass {
                generated_tokens: 64,
                mass: 1.0,
            }],
            known_survival_mass: 0.0,
            unresolved_mass: 0.0,
            tpot_undefined_mass: 0.0,
        }
    }

    fn completion_forecast(&self, _: &CompletionQuery<'_>) -> CompletionForecast {
        CompletionForecast {
            succeeded_mass: 1.0,
            failed_mass: 0.0,
            unresolved_mass: 0.0,
            samples: 32,
            approximation: Some(CompletionApproximation::EmpiricalMaturedCohort),
        }
    }
}

struct FutureFixture {
    scheduler: ContinuousBatchScheduler,
    config: SchedulerSloConfig,
    source: Arc<ScaledCosts>,
    now: Instant,
    requests: Vec<SimulatedRequest>,
}

impl FutureFixture {
    fn new(historical_ttft_failure: bool) -> Self {
        let now = Instant::now();
        let mut config: SchedulerSloConfig = "ttft:1000000,tpot:50000,itl:15000".parse().unwrap();
        config.strategy = ferrum_types::SloStrategy::Joint;
        config.planning.max_elapsed_us = 1_000_000;
        config.planning.max_scenarios = 2;
        config.joint.request_deadline_ms = 2_000_000;
        config.joint.min_completion_rate = 1.0;
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 2,
            active_decode_prefill_chunk: Some(4),
            prefill_step_chunk: Some(4),
            slo: Some(config),
            ..Default::default()
        });
        let (_, original, _) = fixture();
        let mut retiring = original[0].clone();
        retiring.id = RequestId::new();
        retiring.output_tokens = 63;
        let late = if historical_ttft_failure {
            Duration::from_secs(1100)
        } else {
            Duration::ZERO
        };
        retiring.age = late + Duration::from_millis(6200);
        retiring.central_clock = RequestClock {
            first_output: Some(late),
            last_output: Some(retiring.age),
            maximum_output_interval: Duration::from_millis(100),
            interval_count: 62,
            ..Default::default()
        };
        retiring.risk_clock = retiring.central_clock;
        let mut decoder = original[0].clone();
        decoder.age = Duration::from_millis(6200);
        decoder.central_clock.first_output = Some(decoder.age);
        decoder.central_clock.last_output = Some(decoder.age);
        decoder.risk_clock = decoder.central_clock;
        let mut waiting = original[1].clone();
        waiting.prompt_tokens = 64;
        waiting.age = Duration::from_millis(6200);
        let requests = vec![retiring, decoder, waiting];
        for request in &requests[..2] {
            scheduler.submit_request(Self::input(request)).unwrap();
        }
        // Original admission/probe consumer establishes exactly two residents.
        let mut scan = scheduler.begin_token_admission(2, None).unwrap();
        while let Some(candidate) = scan.next(&mut |_| {}, None).unwrap() {
            let request_id = candidate.request().id.clone();
            assert!(matches!(
                scan.apply(
                    candidate,
                    AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt { request_id })
                ),
                TokenAdmissionApply::Applied { accepted: Some(_) }
            ));
        }
        scan.finish().unwrap();
        for request in &requests[..2] {
            scheduler.mark_prefill_complete(&request.id, 1);
            scheduler.update_decode_progress(&request.id, request.output_tokens);
        }
        scheduler.submit_request(Self::input(&requests[2])).unwrap();
        Self {
            scheduler,
            config,
            source: Arc::new(ScaledCosts(now)),
            now,
            requests,
        }
    }

    fn input(request: &SimulatedRequest) -> InferenceRequest {
        InferenceRequest {
            id: request.id.clone(),
            prompt: "test".to_owned(),
            model_id: ModelId::new("test-model"),
            sampling_params: SamplingParams {
                max_tokens: 64,
                ..Default::default()
            },
            stream: true,
            priority: Priority::Normal,
            client_id: None,
            session_id: None,
            created_at: chrono::Utc::now(),
            ingress: None,
            api_request: None,
            evidence_request: Default::default(),
            metadata: HashMap::from([(
                PROMPT_TOKENS_METADATA_KEY.to_owned(),
                serde_json::Value::from(request.prompt_tokens),
            )]),
        }
    }

    fn hint() -> BatchHint {
        BatchHint {
            max_batch_size: 2,
            max_tokens: 8,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        }
    }

    fn round(&self) -> TimePlanningRound {
        let mut round = TimePlanningRound::new(
            self.source.clone(),
            self.config,
            PrefillShapePolicy::Identity,
            CostExecution::SequentialPrefillThenDecode,
        );
        let rules =
            Arc::new(EffectiveStoppingRules::new(vec![1], StoppingMechanism::CapOnly, 16).unwrap());
        let tickets = self
            .scheduler
            .decode_queue
            .read()
            .requests
            .values()
            .chain(self.scheduler.waiting_queue.read().iter())
            .map(|request| {
                (
                    request.inner.request.id.clone(),
                    request.waiting_admission_ticket.unwrap().get(),
                )
            })
            .collect::<HashMap<_, _>>();
        assert!(round.set_request_facts(self.requests.iter().map(|request| (
            request.id.clone(),
            RequestStoppingObservation {
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission: tickets[&request.id],
                },
                admitted_at: self.now,
                observed_at: self.now,
                revision: request.output_tokens as u64 + 1,
                rules: rules.clone(),
                cap: 64,
                generated_tokens: request.output_tokens,
                verified_survival_through: request.output_tokens,
                terminal: None,
                completion: None,
            },
        ))));
        assert!(
            round.set_request_clocks(self.requests.iter().map(|request| {
                (
                    request.id.clone(),
                    RequestClockFacts {
                        captured_at: self.now,
                        age: request.age,
                        first_text: request.central_clock.first_output,
                        last_text: request.central_clock.last_output,
                        max_text_interval: (request.output_tokens > 1)
                            .then_some(request.central_clock.maximum_output_interval),
                        nonempty_text_events: request.output_tokens as u64,
                        violating_text_intervals: request.central_clock.violating_interval_count
                            as u64,
                        itl_threshold: Some(Duration::from_secs(15)),
                        generated_tokens: request.output_tokens,
                        streaming: true,
                    },
                )
            }))
        );
        // As in persistent_capacity.rs, a test resource owner attests the
        // complete cohort. This checks the scheduler's real contract consumer;
        // it is not a claim that this test allocated GPU pools or proved close
        // latency. No fictitious paged free-block count or third active slot.
        let issuer = ExecutorSequenceIssuer::new();
        let members = round
            .request_facts()
            .unwrap()
            .iter()
            .zip(&self.requests)
            .map(|((id, fact), request)| PersistentCapacityMember {
                request_id: id.clone(),
                lifecycle: fact.lifecycle,
                state: if self
                    .scheduler
                    .waiting_queue
                    .read()
                    .iter()
                    .any(|waiting| waiting.inner.request.id == *id)
                {
                    PersistentCapacityState::Waiting
                } else {
                    PersistentCapacityState::Resident {
                        sequence: issuer.issue_sequence().unwrap(),
                    }
                },
                through_tokens: request.prompt_tokens + 64 - 1,
            })
            .collect::<Vec<_>>();
        let query = PersistentCapacityQuery {
            members: &members,
            max_live_sequences: 2,
            max_wave_sequences: 2,
            inspection_budget: Default::default(),
        };
        let proof = PersistentCohortCapacitySnapshot::from_verified_query(&query).unwrap();
        round.set_persistent_capacity(&query, Some(proof));
        assert!(round.capacity().is_none());
        assert!(round.persistent_capacity().is_some());
        round
    }
}

/// A fixed counterfactual policy for the already captured waiter. Its cap is
/// inert until the original slot plan makes that waiter eligible. No mutable
/// action cell, phase-progress exception, or new arrival is involved.
struct FutureCapAdapter {
    original: CohortTrajectory,
    retiring: usize,
    waiting: usize,
    saw_waiting_work: Cell<bool>,
}

impl TrajectoryAdapter<FirstAction> for FutureCapAdapter {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        Some(&self.original.admission)
    }

    fn plan(
        &self,
        action: Option<&FirstAction>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> std::result::Result<(), Unsupported> {
        let active = self.original.admission.active_mask(requests)?;
        assert!(active.iter().filter(|active| **active).count() <= 2);
        if !requests[self.retiring].finished() {
            assert!(!active[self.waiting]);
            assert_eq!(requests[self.waiting].prefill_tokens, 0);
            assert_eq!(requests[self.waiting].output_tokens, 0);
        }
        let members = self.original.natural_members(requests)?;
        if action.is_some() {
            assert_eq!(members.len(), 2);
            assert!(members
                .iter()
                .all(|m| m.work.kind == AttemptWorkKind::Decode));
        }
        if members.iter().any(|member| member.request == self.waiting) {
            assert!(requests[self.retiring].finished());
            self.saw_waiting_work.set(true);
        }
        self.original.plan(action, requests, plan)
    }
}

fn complete_counterfactual(
    fixture: &FutureFixture,
    future_cap: Option<NonZeroUsize>,
) -> ScenarioPolicyDecision {
    let mut round = fixture.round();
    let mut cohort = fixture
        .scheduler
        .joint_cohort(
            &FutureFixture::hint(),
            &[],
            &mut round,
            fixture.now,
            AdmissionCapMode::Propose,
            0,
        )
        .unwrap();
    let retiring = cohort
        .requests
        .iter()
        .position(|request| request.id == fixture.requests[0].id)
        .unwrap();
    let waiting = cohort
        .requests
        .iter()
        .position(|request| request.id == fixture.requests[2].id)
        .unwrap();
    assert!(
        !cohort
            .trajectory
            .admission
            .active_mask(&cohort.requests)
            .unwrap()[waiting]
    );
    // Match the production comparison's maximum declared control-delay charge.
    // These independent model-relative comparisons do not replenish one round.
    for request in &mut cohort.requests {
        request.age += Duration::from_micros(fixture.config.planning.max_elapsed_us);
    }
    cohort.trajectory.shapes[waiting].phase_cap = future_cap;
    let adapter = FutureCapAdapter {
        original: cohort.trajectory,
        retiring,
        waiting,
        saw_waiting_work: Cell::new(false),
    };
    let success = vec![
        Some(ProbabilityInterval {
            lower: 1.0,
            upper: 1.0
        });
        3
    ];
    let mut aggregate =
        ScenarioRiskAccumulator::new(&cohort.requests, 0, &fixture.config, &success).unwrap();
    let summary = SloPolicy::evaluate_scenarios_with_requests(
        EvaluationInput {
            snapshot: fixture.source.as_ref(),
            now: fixture.now,
            config: &fixture.config,
            execution: CostExecution::SequentialPrefillThenDecode,
            requests: &cohort.requests,
            resources: ResourceSupport {
                waiting_requests: 1,
                kv_reserved_through_scenario_end: true,
            },
            adapter: &adapter,
            static_action: &FirstAction::Static,
            alternatives: &[],
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        |_, requests| {
            assert!(requests
                .iter()
                .all(|request| request.scenario_end_tokens == 64));
            Ok(())
        },
        |rank, comparison| {
            let complete = comparison.static_result.complete().unwrap();
            assert_eq!(complete.requests.len(), 3);
            for request in &complete.requests {
                assert_eq!(request.prefill_tokens, request.prompt_tokens);
                assert_eq!(request.output_tokens, 64);
                assert_eq!(request.central_clock.interval_count, 63);
                assert!(request.central_clock.completed.is_some());
            }
            aggregate.observe(rank, comparison).unwrap();
        },
    )
    .unwrap();
    assert!(adapter.saw_waiting_work.get());
    assert_eq!(summary.partial_static_ranks, 0);
    assert_eq!(summary.unresolved_mass, 0.0);
    aggregate.finish(summary, &[]).unwrap()
}

#[test]
fn future_waiter_cap64_counterfactual_preserves_the_first_wave_only_baseline_gap() {
    let fixture = FutureFixture::new(false);
    let mut round = fixture.round();
    let current = fixture
        .scheduler
        .evaluate_joint_admission(
            &FutureFixture::hint(),
            &mut round,
            fixture.now,
            // ExistingOnly freezes the old baseline: no new cap is proposed
            // and this decode-only first wave has no ordinary alternatives.
            // The deferred_commit tests separately exercise the new Propose.
            AdmissionCapMode::ExistingOnly,
            0,
        )
        .unwrap();
    assert_eq!(current.decision.choice, PolicyChoice::AllKnownInfeasible);
    assert!(current.decision.alternative_risks.is_empty());
    assert!(current.prefill_cap.is_none());
    assert_eq!(current.decision.static_risk.request_count, 3);
    assert!((current.decision.static_risk.itl.unwrap().lower - 16.0 / 189.0).abs() < 1e-12);

    assert_eq!(fixture.scheduler.active_count(), 2);
    assert_eq!(fixture.scheduler.waiting_count(), 1);
    assert!(fixture
        .scheduler
        .waiting_queue
        .read()
        .iter()
        .next()
        .unwrap()
        .joint_prefill_phase_cap
        .is_none());

    let baseline = complete_counterfactual(&fixture, None);
    let capped = complete_counterfactual(&fixture, NonZeroUsize::new(1));
    assert_eq!(baseline.choice, PolicyChoice::AllKnownInfeasible);
    assert_eq!(capped.choice, PolicyChoice::Static);
    assert_eq!(
        capped.static_risk.classify(&fixture.config),
        JointFeasibility::Feasible
    );
    assert_eq!(capped.static_risk.itl.unwrap().upper, 0.0);
    assert_eq!(capped.static_risk.completion_failure.upper, 0.0);
    assert!(capped.static_target_time.unwrap() > baseline.static_target_time.unwrap());
    assert!(
        fixture
            .scheduler
            .waiting_queue
            .read()
            .iter()
            .next()
            .unwrap()
            .joint_prefill_phase_cap
            .is_none(),
        "the adapter is only a counterfactual, not an installed production action"
    );

    let mut unsupported = fixture.round();
    unsupported.set_capacity(None); // Also invalidates the complete owner proof.
    assert_eq!(
        fixture
            .scheduler
            .evaluate_joint_admission(
                &FutureFixture::hint(),
                &mut unsupported,
                fixture.now,
                AdmissionCapMode::Propose,
                0
            )
            .unwrap_err(),
        Unsupported::MissingCapacity
    );
}

#[test]
fn future_waiter_cap_does_not_erase_an_original_historical_ttft_violation() {
    let fixture = FutureFixture::new(true);
    let capped = complete_counterfactual(&fixture, NonZeroUsize::new(1));
    assert_eq!(capped.choice, PolicyChoice::AllKnownInfeasible);
    assert_eq!(
        capped.static_risk.ttft,
        ProbabilityInterval {
            lower: 1.0 / 3.0,
            upper: 1.0 / 3.0
        }
    );
    assert_eq!(capped.static_risk.itl.unwrap().upper, 0.0);
    assert_eq!(capped.static_risk.tpot.unwrap().upper, 0.0);
    assert_eq!(capped.static_risk.completion_failure.upper, 0.0);
    assert_eq!(capped.static_risk.request_count, 3);
}
