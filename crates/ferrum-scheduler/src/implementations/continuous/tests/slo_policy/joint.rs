use super::*;
use crate::cost_prediction::{
    CostScenario, CostScenarioDependence, CostScenarioPrediction, PlanningCapacity,
    RequestClockFacts,
};
use ferrum_interfaces::request_observation::*;

struct ConditionalSnapshot {
    at: Instant,
    stopping_known: bool,
    completion_known: bool,
}

impl ConditionalSnapshot {
    fn cost(&self, query: &CostQuery<'_>) -> Duration {
        Duration::from_millis(
            query
                .work
                .iter()
                .map(|work| work.requested.as_range().len() as u64 * 10)
                .sum(),
        )
    }
}

impl CostSnapshot for ConditionalSnapshot {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Known {
            central_service_time: self.cost(query),
            upper_service_time: self.cost(query),
            risk: CostRisk::CalibratedQuantile {
                quantile: 0.95,
                approximation: crate::cost_prediction::CostRiskApproximation::EmpiricalConditional,
            },
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        _: usize,
    ) -> CostScenarioPrediction {
        CostScenarioPrediction::Known {
            central_service_time: self.cost(query),
            scenarios: (0..2)
                .map(|rank| CostScenario {
                    rank,
                    weight: 0.5,
                    service_time: self.cost(query),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        if !self.stopping_known {
            return StoppingForecast::unknown();
        }
        StoppingForecast {
            stops: vec![StoppingMass {
                generated_tokens: query.cap,
                mass: 1.0,
            }],
            known_survival_mass: 0.0,
            unresolved_mass: 0.0,
            tpot_undefined_mass: f64::from(query.cap < 2),
        }
    }
    fn completion_forecast(&self, _: &CompletionQuery<'_>) -> CompletionForecast {
        if !self.completion_known {
            return CompletionForecast::unknown();
        }
        CompletionForecast {
            succeeded_mass: 1.0,
            failed_mass: 0.0,
            unresolved_mass: 0.0,
            samples: 32,
            approximation: Some(CompletionApproximation::EmpiricalMaturedCohort),
        }
    }
}

fn fixture(
    execution: CostExecution,
    stopping_known: bool,
    completion_known: bool,
) -> (
    ContinuousBatchScheduler,
    TimePlanningRound,
    RequestId,
    RequestId,
    Instant,
) {
    let mut config = targets();
    config.strategy = ferrum_types::SloStrategy::Joint;
    config.itl_ms = 100.0;
    config.tpot_ms = 500.0;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        active_decode_prefill_chunk: Some(8),
        slo: Some(config),
        ..SchedulerConfig::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let now = ingress.instant() + Duration::from_millis(50);
    let mut decoder =
        create_test_request_with_prompt_tokens(Priority::Normal, 1).with_ingress(ingress);
    decoder.sampling_params.max_tokens = 4;
    let decoder_id = decoder.id.clone();
    let mut prefiller =
        create_test_request_with_prompt_tokens(Priority::Normal, 16).with_ingress(ingress);
    prefiller.sampling_params.max_tokens = 2;
    let prefill_id = prefiller.id.clone();
    enqueue_waiting(&scheduler, decoder);
    enqueue_waiting(&scheduler, prefiller);
    admit(&scheduler, None);
    scheduler.mark_prefill_complete(&decoder_id, 1);
    scheduler.update_decode_progress(&decoder_id, 2);
    let source = Arc::new(ConditionalSnapshot {
        at: now - Duration::from_millis(1),
        stopping_known,
        completion_known,
    });
    let mut round = TimePlanningRound::new(source, config, PrefillShapePolicy::Identity, execution);
    let rules =
        Arc::new(EffectiveStoppingRules::new(vec![1], StoppingMechanism::CapOnly, 16).unwrap());
    let members = [(decoder_id.clone(), 4, 2), (prefill_id.clone(), 2, 0)];
    assert!(round.set_request_facts(members.iter().enumerate().map(
        |(index, (id, cap, generated))| {
            (
                id.clone(),
                RequestStoppingObservation {
                    lifecycle: RequestLifecycleId {
                        producer: 1,
                        admission: index as u64 + 1,
                    },
                    admitted_at: ingress.instant(),
                    observed_at: now,
                    revision: 1,
                    rules: rules.clone(),
                    cap: *cap,
                    generated_tokens: *generated,
                    verified_survival_through: *generated,
                    terminal: None,
                    completion: None,
                },
            )
        }
    )));
    assert!(
        round.set_request_clocks(members.iter().map(|(id, _, generated)| {
            let decoded = *generated > 0;
            (
                id.clone(),
                RequestClockFacts {
                    captured_at: now,
                    age: Duration::from_millis(50),
                    first_text: decoded.then_some(Duration::from_millis(5)),
                    last_text: decoded.then_some(Duration::from_millis(10)),
                    max_text_interval: decoded.then_some(Duration::from_millis(5)),
                    nonempty_text_events: *generated as u64,
                    violating_text_intervals: 0,
                    itl_threshold: Some(Duration::from_millis(100)),
                    generated_tokens: *generated,
                    streaming: true,
                },
            )
        }))
    );
    round.set_capacity(Some(PlanningCapacity {
        block_size: 16,
        total_blocks: 100,
        free_blocks: 100,
        admission_coverage:
            ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                max_new_sequences: 8,
                max_sequence_tokens: 4096,
                max_batch_sequences: 8,
            },
    }));
    (scheduler, round, prefill_id, decoder_id, now)
}

fn next(
    scheduler: &ContinuousBatchScheduler,
    round: &mut TimePlanningRound,
    now: Instant,
) -> BatchPlan {
    scheduler
        .create_iteration_batch_with_time(
            hint(),
            WaitingAdmissionMode::Prepared {
                initial_priority: None,
                wake: None,
                observer: &mut |_| {},
            },
            now,
            Some(round),
        )
        .unwrap()
        .unwrap()
}

fn tokens(batch: &BatchPlan, id: &RequestId) -> usize {
    batch
        .requests
        .iter()
        .find(|request| &request.request.id == id)
        .unwrap()
        .tokens_to_process
        .unwrap()
}

#[test]
fn joint_dispatch_changes_only_supported_first_prefill_and_keeps_decoder() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        let (scheduler, mut round, prefill, decoder, now) = fixture(execution, true, true);
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &prefill), 4);
        assert_eq!(tokens(&batch, &decoder), 1);
        assert_eq!(round.statistics.joint_interventions, 1);
        assert!(round.statistics.joint_cost_snapshot_queries > 0);
        assert!(round.statistics.joint_cost_cache_hits > 0);
        assert!(
            round.statistics.joint_cost_cache_peak_bytes
                <= round.config.planning.max_cost_cache_bytes as u64
        );
        assert_eq!(batch.resource_requirements.gpu_memory, 5 * 16);
        assert!(round.selection_is_exact(batch.size()));
        assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 16, 4));
        assert_eq!(
            scheduler
                .prefill_queue
                .read()
                .iter()
                .find(|request| request.inner.request.id == prefill)
                .unwrap()
                .prefill_chunk_offset,
            4
        );
    }
}

#[test]
fn joint_dispatch_keeps_static_when_stopping_completion_or_capacity_is_unknown() {
    for (stopping, completion, capacity) in [
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        let (scheduler, mut round, prefill, decoder, now) =
            fixture(CostExecution::SingleInvocation, stopping, completion);
        if !capacity {
            round.set_capacity(None);
        }
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &prefill), 8);
        assert_eq!(tokens(&batch, &decoder), 1);
        assert_eq!(round.statistics.joint_interventions, 0);
    }
}

#[test]
fn joint_dispatch_preserves_nominal_static_budget_before_prompt_tail_projection() {
    let (scheduler, mut round, prefill, _, now) =
        fixture(CostExecution::SingleInvocation, true, true);
    round.set_capacity(None);
    round.prefill_shape_policy = prefill_shape::tail(12);
    let batch = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&batch, &prefill), 8);
    let index = batch
        .requests
        .iter()
        .position(|request| request.request.id == prefill)
        .unwrap();
    assert_eq!(round.selected[index].requested.as_range().len(), 4);
    assert!(round.cost_selection_is_exact(batch.size()));
    assert!(!round.selection_is_exact(batch.size()));
}

fn waiting_fixture(
    count: usize,
    slots: usize,
    ttft_ms: f64,
) -> (ContinuousBatchScheduler, TimePlanningRound, Instant) {
    let mut config = targets();
    config.strategy = ferrum_types::SloStrategy::Joint;
    config.ttft_ms = ttft_ms;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: slots,
        slo: Some(config),
        ..Default::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let now = ingress.instant();
    let rules =
        Arc::new(EffectiveStoppingRules::new(vec![1], StoppingMechanism::CapOnly, 16).unwrap());
    let mut facts = Vec::new();
    let mut clocks = Vec::new();
    for index in 0..count {
        let mut request =
            create_test_request_with_prompt_tokens(Priority::Normal, 1).with_ingress(ingress);
        request.stream = true;
        request.sampling_params.max_tokens = 2;
        let id = request.id.clone();
        enqueue_waiting(&scheduler, request);
        facts.push((
            id.clone(),
            RequestStoppingObservation {
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission: index as u64 + 1,
                },
                admitted_at: now,
                observed_at: now,
                revision: 1,
                rules: rules.clone(),
                cap: 2,
                generated_tokens: 0,
                verified_survival_through: 0,
                terminal: None,
                completion: None,
            },
        ));
        clocks.push((
            id,
            RequestClockFacts {
                captured_at: now,
                age: Duration::ZERO,
                first_text: None,
                last_text: None,
                max_text_interval: None,
                nonempty_text_events: 0,
                violating_text_intervals: 0,
                itl_threshold: Some(Duration::from_millis(config.itl_ms as u64)),
                generated_tokens: 0,
                streaming: true,
            },
        ));
    }
    let mut round = TimePlanningRound::new(
        Arc::new(ConditionalSnapshot {
            at: now - Duration::from_millis(1),
            stopping_known: true,
            completion_known: true,
        }),
        config,
        PrefillShapePolicy::Identity,
        CostExecution::SequentialPrefillThenDecode,
    );
    assert!(round.set_request_facts(facts));
    assert!(round.set_request_clocks(clocks));
    round.set_capacity(Some(PlanningCapacity {
        block_size: 16,
        total_blocks: 100,
        free_blocks: 100,
        admission_coverage:
            ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                max_new_sequences: count,
                max_sequence_tokens: 4096,
                max_batch_sequences: 8,
            },
    }));
    (scheduler, round, now)
}

#[test]
fn joint_admission_accounts_for_waiting_ttft_and_rejects_before_executor_probe() {
    let (scheduler, mut round, now) = waiting_fixture(2, 1, 25.0);
    let decision = scheduler
        .evaluate_joint_admission(&hint(), &mut round, now)
        .unwrap();
    assert_eq!(
        decision.choice,
        crate::joint_policy::PolicyChoice::AllKnownInfeasible
    );
    assert_eq!(decision.static_risk.request_count, 2);
    assert_eq!(decision.static_risk.ttft.lower, 0.5);
    scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
    let mut scan = scheduler.begin_token_admission(8, None).unwrap();
    assert!(scan
        .next_at(&mut |_| {}, Some(&mut round), now)
        .unwrap()
        .is_none());
    assert_eq!(round.statistics.rejected_candidates, 1);
    assert_eq!(scheduler.waiting_count(), 1);
}

#[test]
fn joint_admission_feasible_waiting_retains_all_arrival_denominators() {
    let (scheduler, mut round, now) = waiting_fixture(3, 1, 100.0);
    let decision = scheduler
        .evaluate_joint_admission(&hint(), &mut round, now)
        .unwrap();
    assert_eq!(decision.choice, crate::joint_policy::PolicyChoice::Static);
    assert_eq!(decision.static_risk.request_count, 3);
    assert_eq!(decision.static_risk.completion_failure.upper, 0.0);
    scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
    let mut scan = scheduler.begin_token_admission(8, None).unwrap();
    assert!(scan
        .next_at(&mut |_| {}, Some(&mut round), now)
        .unwrap()
        .is_some());
}

#[test]
fn joint_admission_does_not_recredit_metadata_after_scan_promotions() {
    let (scheduler, mut round, now) = waiting_fixture(3, 2, 100.0);
    let mut capacity = round.capacity().unwrap();
    capacity.admission_coverage =
        ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
            max_new_sequences: 2,
            max_sequence_tokens: 4096,
            max_batch_sequences: 8,
        };
    round.set_capacity(Some(capacity));
    scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
    assert_eq!(round.capacity_initial_new_sequences, Some(3));
    admit(&scheduler, None);
    assert_eq!(scheduler.waiting_count(), 1);
    assert!(matches!(
        scheduler.evaluate_joint_admission(&hint(), &mut round, now),
        Err(crate::joint_policy::Unsupported::UnknownKvCapacity)
    ));
    capacity.admission_coverage = ferrum_execution::model_executor::PagedAdmissionCoverage::Unknown;
    round.set_capacity(Some(capacity));
    assert!(matches!(
        scheduler.evaluate_joint_admission(&hint(), &mut round, now),
        Err(crate::joint_policy::Unsupported::UnknownKvCapacity)
    ));
}

#[test]
fn joint_dispatch_does_not_interpret_coalesced_or_nonstreaming_history_as_token_events() {
    for coalesced in [false, true] {
        let (scheduler, mut round, prefill, decoder, now) =
            fixture(CostExecution::SingleInvocation, true, true);
        let mut clocks = round.request_clocks().unwrap().to_vec();
        let changed_id = if coalesced { &decoder } else { &prefill };
        let clock = &mut clocks
            .iter_mut()
            .find(|(id, _)| id == changed_id)
            .unwrap()
            .1;
        if coalesced {
            clock.nonempty_text_events = 1;
            clock.last_text = clock.first_text;
            clock.max_text_interval = None;
        } else {
            clock.streaming = false;
        }
        assert!(round.set_request_clocks(clocks));
        let batch = next(&scheduler, &mut round, now);
        assert_eq!(tokens(&batch, &prefill), 8);
        assert_eq!(round.statistics.joint_interventions, 0);
        assert_eq!(round.statistics.joint_unresolved, 1);
    }
}
