use super::*;
use crate::cost_prediction::CostUnsupportedDetail;

mod decision_diagnostics;

struct FixedGapCost {
    base: ConditionalSnapshot,
    reject_singletons: bool,
}

impl FixedGapCost {
    fn service_time(&self, query: &CostQuery<'_>) -> Option<Duration> {
        let decode_only = query
            .work
            .iter()
            .all(|work| work.kind == AttemptWorkKind::Decode);
        if decode_only && (query.work.len() == 2 || self.reject_singletons) {
            return None;
        }
        let full_prefill = query.work.iter().any(|work| {
            work.kind == AttemptWorkKind::Prefill && work.requested.as_range().len() == 2
        });
        Some(Duration::from_secs(if full_prefill { 100 } else { 10 }))
    }
}

impl CostSnapshot for FixedGapCost {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }

    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        let Some(service_time) = self.service_time(query) else {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport,
            ));
        };
        CostPrediction::Known {
            central_service_time: service_time,
            upper_service_time: service_time,
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.base.at,
            expires_at: self.base.at + Duration::from_secs(60),
        }
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        let Some(service_time) = self.service_time(query) else {
            return CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport,
            ));
        };
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
            observed_at: self.base.at,
            expires_at: self.base.at + Duration::from_secs(60),
        }
    }

    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.base.stopping_forecast(query)
    }

    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.base.completion_forecast(query)
    }
}

fn partial_fixture(
    reject_singletons: bool,
) -> (
    ContinuousBatchScheduler,
    TimePlanningRound,
    RequestId,
    RequestId,
    Instant,
) {
    // These are simulated service durations, not host timing requirements.
    // The one-second planning allowance is charged to both trajectories and
    // remains small relative to the declared fifty-second latency limits.
    let mut config: SchedulerSloConfig = "ttft:50000,tpot:50000,itl:50000".parse().unwrap();
    config.strategy = ferrum_types::SloStrategy::Joint;
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    config.joint.min_completion_rate = 1.0;
    config.joint.request_deadline_ms = 1_000_000;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 2,
        active_decode_prefill_chunk: Some(2),
        slo: Some(config),
        ..Default::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let now = ingress.instant();
    let mut decoder =
        create_test_request_with_prompt_tokens(Priority::Normal, 1).with_ingress(ingress);
    decoder.sampling_params.max_tokens = 3;
    decoder.stream = true;
    let decoder_id = decoder.id.clone();
    let mut prefiller =
        create_test_request_with_prompt_tokens(Priority::Normal, 2).with_ingress(ingress);
    prefiller.sampling_params.max_tokens = 2;
    prefiller.stream = true;
    let prefill_id = prefiller.id.clone();
    enqueue_waiting(&scheduler, decoder);
    enqueue_waiting(&scheduler, prefiller);
    admit(&scheduler, None);
    scheduler.mark_prefill_complete(&decoder_id, 1);
    scheduler.update_decode_progress(&decoder_id, 1);
    let source = Arc::new(FixedGapCost {
        base: ConditionalSnapshot {
            at: now,
            stopping_known: true,
            completion_known: true,
        },
        reject_singletons,
    });
    let mut round = TimePlanningRound::new(
        source,
        config,
        PrefillShapePolicy::Identity,
        CostExecution::SingleInvocation,
    );
    let rules =
        Arc::new(EffectiveStoppingRules::new(vec![1], StoppingMechanism::CapOnly, 16).unwrap());
    let members = [(decoder_id.clone(), 3, 1), (prefill_id.clone(), 2, 0)];
    assert!(round.set_request_facts(members.iter().enumerate().map(
        |(index, (id, cap, generated))| (
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
                cap: *cap,
                generated_tokens: *generated,
                verified_survival_through: *generated,
                terminal: None,
                completion: None,
            },
        )
    )));
    assert!(
        round.set_request_clocks(members.iter().map(|(id, _, generated)| (
            id.clone(),
            RequestClockFacts {
                captured_at: now,
                age: Duration::ZERO,
                first_text: (*generated > 0).then_some(Duration::ZERO),
                last_text: (*generated > 0).then_some(Duration::ZERO),
                max_text_interval: None,
                nonempty_text_events: *generated as u64,
                violating_text_intervals: 0,
                itl_threshold: Some(Duration::from_secs(50)),
                generated_tokens: *generated,
                streaming: true,
            },
        )))
    );
    round.set_capacity(Some(PlanningCapacity {
        block_size: 16,
        total_blocks: 100,
        free_blocks: 100,
        admission_coverage:
            ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                max_new_sequences: 2,
                max_sequence_tokens: 4096,
                max_batch_sequences: 2,
            },
    }));
    (scheduler, round, prefill_id, decoder_id, now)
}

#[test]
fn actual_joint_dispatch_replaces_a_violated_partial_static_with_complete_endpoints() {
    let (scheduler, mut round, prefill, decoder, now) = partial_fixture(false);
    let batch = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&batch, &prefill), 1);
    assert_eq!(tokens(&batch, &decoder), 1);
    assert_eq!(batch.size(), 2);
    assert_eq!(round.statistics.joint_interventions, 1);
    assert_eq!(
        round
            .statistics
            .joint_evaluation
            .dispatch
            .rank_partial_static,
        2
    );
    assert_eq!(
        round.statistics.joint_evaluation.dispatch.failed_traversals,
        0
    );
    assert!(round.selection_is_exact(batch.size()));
    assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 2, 1));
    assert_eq!(scheduler.prefill_queue.read()[0].prefill_chunk_offset, 1);
}

#[test]
fn actual_joint_dispatch_keeps_static_when_the_alternative_also_has_an_unknown_tail() {
    let (scheduler, mut round, prefill, decoder, now) = partial_fixture(true);
    let batch = next(&scheduler, &mut round, now);
    assert_eq!(tokens(&batch, &prefill), 2);
    assert_eq!(tokens(&batch, &decoder), 1);
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(
        round
            .statistics
            .joint_evaluation
            .dispatch
            .rank_partial_static,
        2
    );
    assert_eq!(
        serde_json::to_value(round.statistics.joint_unsupported).unwrap()["feasibility_unresolved"],
        1
    );
}
