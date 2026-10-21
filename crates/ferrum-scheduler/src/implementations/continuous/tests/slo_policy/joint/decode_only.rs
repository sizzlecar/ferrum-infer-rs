use super::*;
use std::sync::atomic::AtomicUsize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blocker {
    None,
    CostUnknown,
    StoppingUnknown,
    CompletionUnknown,
    NoCapacity,
    ProjectionUnknown,
    QueryBudget,
    TimeBudget,
}

struct MixedCost {
    base: ConditionalSnapshot,
    unknown: bool,
    scalar_calls: AtomicUsize,
    calls: Mutex<Vec<Vec<CostWork>>>,
}

impl MixedCost {
    fn service_time(&self, query: &CostQuery<'_>) -> Duration {
        // Simulated seconds preserve the 10:1 service-cost counterexample
        // without turning its ITL margin into a millisecond host-speed test.
        Duration::from_secs(
            query
                .work
                .iter()
                .map(|work| match work.kind {
                    AttemptWorkKind::Prefill => work.requested.as_range().len() as u64 * 10,
                    AttemptWorkKind::Decode => 1,
                })
                .sum(),
        )
    }
}

impl CostSnapshot for MixedCost {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        self.scalar_calls.fetch_add(1, Ordering::Relaxed);
        if self.unknown {
            return CostPrediction::Unknown(CostUnknownReason::NoCoverage);
        }
        let time = self.service_time(query);
        CostPrediction::Known {
            central_service_time: time,
            upper_service_time: time,
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.base.at,
            expires_at: self.base.at + Duration::from_secs(180),
        }
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.lock().push(query.work.to_vec());
        if self.unknown {
            return CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage);
        }
        let time = self.service_time(query);
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
            observed_at: self.base.at,
            expires_at: self.base.at + Duration::from_secs(180),
        }
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.base.stopping_forecast(query)
    }
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.base.completion_forecast(query)
    }
}

struct DecodeOnlyFixture {
    scheduler: ContinuousBatchScheduler,
    round: TimePlanningRound,
    source: Arc<MixedCost>,
    prefill: Option<RequestId>,
    decoder: Option<RequestId>,
}

fn decode_only_fixture(
    prefill: bool,
    decode: bool,
    execution: CostExecution,
    blocker: Blocker,
) -> DecodeOnlyFixture {
    let mut config: SchedulerSloConfig = "ttft:100000,tpot:50000,itl:5000".parse().unwrap();
    config.strategy = ferrum_types::SloStrategy::Joint;
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    config.joint.min_completion_rate = 1.0;
    config.joint.request_deadline_ms = 120_000;
    let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 2,
        active_decode_prefill_chunk: Some(4),
        slo: Some(config),
        ..Default::default()
    });
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let now = ingress.instant();
    let mut members = Vec::new();
    let decoder = decode.then(|| {
        let mut request =
            create_test_request_with_prompt_tokens(Priority::Normal, 1).with_ingress(ingress);
        request.sampling_params.max_tokens = 2;
        request.stream = true;
        let id = request.id.clone();
        enqueue_waiting(&scheduler, request);
        members.push((id.clone(), 1));
        id
    });
    let prefill = prefill.then(|| {
        let mut request =
            create_test_request_with_prompt_tokens(Priority::Normal, 4).with_ingress(ingress);
        request.sampling_params.max_tokens = 2;
        request.stream = true;
        let id = request.id.clone();
        enqueue_waiting(&scheduler, request);
        members.push((id.clone(), 0));
        id
    });
    admit(&scheduler, None);
    if let Some(id) = &decoder {
        scheduler.mark_prefill_complete(id, 1);
        scheduler.update_decode_progress(id, 1);
    }
    let source = Arc::new(MixedCost {
        base: ConditionalSnapshot {
            at: now,
            stopping_known: blocker != Blocker::StoppingUnknown,
            completion_known: blocker != Blocker::CompletionUnknown,
        },
        unknown: blocker == Blocker::CostUnknown,
        scalar_calls: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    });
    let mut round = TimePlanningRound::new(
        source.clone(),
        config,
        PrefillShapePolicy::Identity,
        execution,
    );
    let rules =
        Arc::new(EffectiveStoppingRules::new(vec![1], StoppingMechanism::CapOnly, 16).unwrap());
    assert!(
        round.set_request_facts(members.iter().enumerate().map(|(index, (id, generated))| (
            id.clone(),
            RequestStoppingObservation {
                lifecycle: RequestLifecycleId {
                    producer: 1,
                    admission: index as u64 + 1
                },
                admitted_at: now,
                observed_at: now,
                revision: 1,
                rules: rules.clone(),
                cap: 2,
                generated_tokens: *generated,
                verified_survival_through: *generated,
                terminal: None,
                completion: None,
            },
        )))
    );
    assert!(
        round.set_request_clocks(members.iter().map(|(id, generated)| (
            id.clone(),
            RequestClockFacts {
                captured_at: now,
                age: Duration::ZERO,
                first_text: (*generated > 0).then_some(Duration::ZERO),
                last_text: (*generated > 0).then_some(Duration::ZERO),
                max_text_interval: None,
                nonempty_text_events: *generated as u64,
                violating_text_intervals: 0,
                itl_threshold: Some(Duration::from_secs(5)),
                generated_tokens: *generated,
                streaming: true,
            },
        )))
    );
    round.set_capacity(Some(PlanningCapacity {
        block_size: 16,
        total_blocks: 16,
        free_blocks: 16,
        admission_coverage:
            ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                max_new_sequences: 2,
                max_sequence_tokens: 128,
                max_batch_sequences: 2,
            },
    }));
    match blocker {
        Blocker::NoCapacity => round.set_capacity(None),
        Blocker::ProjectionUnknown => round.prefill_shape_policy = PrefillShapePolicy::Unknown,
        Blocker::QueryBudget => round.config.planning.max_queries = 3,
        Blocker::TimeBudget => round.prediction_elapsed = Duration::from_secs(1),
        _ => {}
    }
    DecodeOnlyFixture {
        scheduler,
        round,
        source,
        prefill,
        decoder,
    }
}

fn observed_selection(
    scheduler: &ContinuousBatchScheduler,
    round: TimePlanningRound,
) -> (BatchPlan, TimePlanningRound) {
    let previous = scheduler
        .slo_snapshot()
        .map_or(0, |health| health.planning_rounds);
    let mut scope = scheduler.observe_planning_round(Some(round));
    let intent = scheduler.prepare_token_batch_selection(hint());
    let batch = scheduler
        .next_batch_with_token_admissions(intent, None, &mut |_| {}, scope.as_mut())
        .unwrap()
        .batch
        .unwrap();
    assert_eq!(scheduler.slo_snapshot().unwrap().planning_rounds, previous);
    let round = scope.finish().unwrap();
    assert_eq!(
        scheduler.slo_snapshot().unwrap().planning_rounds,
        previous + 1
    );
    (batch, round)
}

#[test]
fn joint_decode_only_wave_is_feasible_and_defers_without_losing_prefill_work() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        let DecodeOnlyFixture {
            scheduler,
            round,
            source,
            prefill,
            decoder,
        } = decode_only_fixture(true, true, execution, Blocker::None);
        let prefill = prefill.unwrap();
        let decoder = decoder.unwrap();
        let before = scheduler.prefill_queue.read()[0]
            .logical_work_frontier
            .clone();
        let (batch, round) = observed_selection(&scheduler, round);
        // The static/half/quarter first waves delay the decoder by41/21/11
        // simulated seconds. Decode-only costs1, then the SAME continuation
        // runs prefill40 and decode1. The charged control delay is at most1s.
        assert_eq!(
            batch.size(),
            1,
            "only the already admitted decoder may run first"
        );
        assert_eq!(tokens(&batch, &decoder), 1);
        assert!(batch.requests.iter().all(|item| item.request.id != prefill));
        assert_eq!(round.selected.len(), 1);
        assert_eq!(round.selected[0].kind, AttemptWorkKind::Decode);
        assert!(round.selection_is_exact(batch.size()));
        assert_eq!(batch.resource_requirements.gpu_memory, 16);
        assert_eq!(scheduler.active_count(), 2);
        assert_eq!(scheduler.waiting_count(), 0);
        {
            let queue = scheduler.prefill_queue.read();
            let pending = queue
                .iter()
                .find(|item| item.inner.request.id == prefill)
                .unwrap();
            assert_eq!(pending.prefill_chunk_offset, 0);
            assert_eq!(pending.decode_tokens, 0);
            assert_eq!(
                pending.logical_work_frontier, before,
                "withdraw only the uncommitted scheduled frontier"
            );
            assert!(pending.logical_work_frontier.can_restore_at(0));
        }
        let decision = round.statistics.joint_decision.dispatch;
        assert_eq!(round.statistics.joint_interventions, 1);
        assert_eq!(round.statistics.joint_decode_only_interventions, 1);
        assert_eq!(decision.computed, 1);
        assert_eq!(decision.returned, 1);
        assert_eq!(decision.consumed, 1);
        assert_eq!(decision.outcomes.alternative, 1);
        assert_eq!(decision.candidates.joint.infeasible, 2);
        assert_eq!(decision.candidates.joint.feasible, 1);
        assert_eq!(decision.candidates.metrics.itl.infeasible, 2);
        assert_eq!(decision.candidates.gates.eligible, 1);
        assert_eq!(round.statistics.joint_evaluation.dispatch.rank_completed, 2);
        assert_eq!(
            round
                .statistics
                .joint_evaluation
                .dispatch
                .rank_partial_static,
            0
        );
        let calls = source.calls.lock().len();
        assert_eq!(source.scalar_calls.load(Ordering::Relaxed), 0);
        assert_eq!(round.query_count, calls + 4);
        assert_eq!(
            round.statistics.joint_evaluation.dispatch.provider_calls,
            calls as u64
        );
        let health = scheduler.slo_snapshot().unwrap();
        assert_eq!(health.selections.joint_interventions, 1);
        assert_eq!(health.selections.joint_decode_only_interventions, 1);
        serde_json::to_value(health).unwrap();
        assert_eq!(source.calls.lock().len(), calls);

        // Commit only the emitted decoder token, retire that completed request,
        // then freeze the remaining cohort anew. The deferred request must
        // execute its original0..4 prompt, not disappear or skip its work.
        scheduler.update_decode_progress(&decoder, 2);
        let mut response = super::super::super::capacity_release::response(&decoder);
        response.usage = ferrum_types::TokenUsage::new(1, 2);
        scheduler.complete_request(decoder, &response).unwrap();
        let mut next_round = TimePlanningRound::new(
            source.clone(),
            round.config,
            round.prefill_shape_policy,
            execution,
        );
        assert!(next_round.set_request_facts(
            round
                .request_facts()
                .unwrap()
                .iter()
                .filter(|(id, _)| id == &prefill)
                .cloned()
        ));
        assert!(next_round.set_request_clocks(
            round
                .request_clocks()
                .unwrap()
                .iter()
                .filter(|(id, _)| id == &prefill)
                .cloned()
        ));
        next_round.set_capacity(round.capacity());
        let (next_batch, next_round) = observed_selection(&scheduler, next_round);
        assert_eq!(next_batch.size(), 1);
        assert_eq!(tokens(&next_batch, &prefill), 4);
        assert_eq!(next_batch.requests[0].tokens_processed, 0);
        assert!(next_round.selection_is_exact(1));
        assert_eq!(next_round.statistics.joint_interventions, 0);
        assert_eq!(next_round.statistics.joint_decode_only_interventions, 0);
        assert_eq!(
            scheduler
                .slo_snapshot()
                .unwrap()
                .selections
                .joint_decode_only_interventions,
            1
        );
        assert!(scheduler.mark_prefill_chunk_processed(&prefill, 4, 4));
        assert_eq!(scheduler.prefill_queue.read().len(), 0);
        assert!(scheduler
            .decode_queue
            .read()
            .requests
            .contains_key(&prefill));
    }
}

#[test]
fn joint_decode_only_is_not_an_empty_or_pure_prefill_action() {
    for (prefill, decode, expected_candidates) in [(true, false, 2), (false, true, 0)] {
        let fixture = decode_only_fixture(
            prefill,
            decode,
            CostExecution::SingleInvocation,
            Blocker::None,
        );
        let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);
        assert_eq!(batch.size(), 1);
        assert_eq!(round.statistics.joint_interventions, 0);
        assert_eq!(round.statistics.joint_decode_only_interventions, 0);
        let counts = round.statistics.joint_decision.dispatch.candidates.joint;
        assert_eq!(
            counts.feasible + counts.infeasible + counts.unresolved,
            expected_candidates
        );
        if let Some(id) = fixture.prefill {
            assert_eq!(tokens(&batch, &id), 4);
        }
        if let Some(id) = fixture.decoder {
            assert_eq!(tokens(&batch, &id), 1);
        }
        assert!(batch
            .requests
            .iter()
            .all(|item| item.tokens_to_process.is_some_and(|tokens| tokens > 0)));
    }
}

#[test]
fn joint_decode_only_does_not_bypass_unknown_support_capacity_projection_or_budgets() {
    for blocker in [
        Blocker::CostUnknown,
        Blocker::StoppingUnknown,
        Blocker::CompletionUnknown,
        Blocker::NoCapacity,
        Blocker::ProjectionUnknown,
        Blocker::QueryBudget,
        Blocker::TimeBudget,
    ] {
        let fixture = decode_only_fixture(true, true, CostExecution::SingleInvocation, blocker);
        let (batch, round) = observed_selection(&fixture.scheduler, fixture.round);
        assert_eq!(batch.size(), 2, "{blocker:?}");
        assert_eq!(
            tokens(&batch, fixture.prefill.as_ref().unwrap()),
            4,
            "{blocker:?}"
        );
        assert_eq!(
            tokens(&batch, fixture.decoder.as_ref().unwrap()),
            1,
            "{blocker:?}"
        );
        assert_eq!(round.statistics.joint_interventions, 0, "{blocker:?}");
        assert_eq!(
            round.statistics.joint_decode_only_interventions, 0,
            "{blocker:?}"
        );
        assert_eq!(
            fixture
                .scheduler
                .slo_snapshot()
                .unwrap()
                .selections
                .joint_decode_only_interventions,
            0,
            "{blocker:?}"
        );
        assert_eq!(
            round
                .statistics
                .joint_decision
                .dispatch
                .outcomes
                .alternative,
            0,
            "{blocker:?}"
        );
        assert!(!fixture.scheduler.prefill_queue.read()[0]
            .logical_work_frontier
            .can_restore_at(0));
        if matches!(blocker, Blocker::QueryBudget) {
            // Two requests need four stopping/completion forecasts before any
            // candidate is evaluated. Three queries exhaust this prerequisite
            // independently of the number of first-wave alternatives.
            assert_eq!(round.query_count, 3);
            assert_eq!(round.remaining_queries(), 0);
            assert!(fixture.source.calls.lock().is_empty());
            assert_eq!(
                serde_json::to_value(round.statistics.joint_unsupported).unwrap()["query_budget"],
                1
            );
            let decisions = &round.statistics.joint_decision.dispatch;
            assert_eq!(decisions.computed, 0);
            assert_eq!(decisions.returned, 0);
            assert_eq!(decisions.consumed, 0);
            assert_eq!(round.statistics.joint_evaluation.dispatch.evaluations, 0);
        }
        if matches!(
            blocker,
            Blocker::NoCapacity | Blocker::ProjectionUnknown | Blocker::TimeBudget
        ) {
            assert!(fixture.source.calls.lock().is_empty(), "{blocker:?}");
        }
    }
}

fn recapture_round(previous: &TimePlanningRound) -> TimePlanningRound {
    let now = Instant::now();
    let mut round = TimePlanningRound::new(
        previous.snapshot.clone(),
        previous.config,
        previous.prefill_shape_policy,
        previous.execution,
    );
    assert!(
        round.set_request_facts(previous.request_facts().unwrap().iter().map(|(id, fact)| {
            let mut fact = fact.clone();
            fact.observed_at = now;
            (id.clone(), fact)
        }))
    );
    assert!(
        round.set_request_clocks(
            previous
                .request_clocks()
                .unwrap()
                .iter()
                .map(|(id, clock)| {
                    let mut clock = *clock;
                    clock.age += now.duration_since(clock.captured_at);
                    clock.captured_at = now;
                    (id.clone(), clock)
                })
        )
    );
    round.set_capacity(previous.capacity());
    round
}

#[test]
fn joint_decode_only_deferral_requires_actual_prefill_progress_before_reuse() {
    let DecodeOnlyFixture {
        scheduler,
        mut round,
        source: _,
        prefill,
        decoder,
    } = decode_only_fixture(true, true, CostExecution::SingleInvocation, Blocker::None);
    let prefill = prefill.unwrap();
    let decoder = decoder.unwrap();
    let capacity = round.capacity();
    let original_generation = scheduler.prefill_queue.read()[0]
        .logical_work_frontier
        .progress_generation();

    // An unsupported round must not acquire an action's deferral debt.
    round.set_capacity(None);
    let (unsupported, failed_round) = observed_selection(&scheduler, round);
    assert_eq!(tokens(&unsupported, &prefill), 4);
    assert_eq!(failed_round.statistics.joint_interventions, 0);
    assert_eq!(failed_round.statistics.joint_decode_only_interventions, 0);
    let mut qualified = recapture_round(&failed_round);
    qualified.set_capacity(capacity);
    let (first, first_round) = observed_selection(&scheduler, qualified);
    assert_eq!(first.size(), 1);
    assert_eq!(tokens(&first, &decoder), 1);
    assert_eq!(first_round.statistics.joint_interventions, 1);
    assert_eq!(first_round.statistics.joint_decode_only_interventions, 1);

    // Neither selected work nor a speculative trajectory is a commit. Replan
    // the unchanged resident cohort without submitting either request.
    let (second, second_round) = observed_selection(&scheduler, recapture_round(&first_round));
    assert_eq!(tokens(&second, &prefill), 4);
    assert_eq!(tokens(&second, &decoder), 1);
    assert_eq!(second_round.statistics.joint_interventions, 0);
    assert_eq!(second_round.statistics.joint_decode_only_interventions, 0);
    assert_eq!(scheduler.prefill_queue.read()[0].prefill_chunk_offset, 0);
    assert_eq!(
        scheduler.prefill_queue.read()[0]
            .logical_work_frontier
            .progress_generation(),
        original_generation
    );

    // The second plan calls mark_scheduled(4), which must not clear debt.
    let (third, third_round) = observed_selection(&scheduler, recapture_round(&second_round));
    assert_eq!(tokens(&third, &prefill), 4);
    assert_eq!(tokens(&third, &decoder), 1);
    assert_eq!(third_round.statistics.joint_interventions, 0);
    assert_eq!(third_round.statistics.joint_decode_only_interventions, 0);
    assert_eq!(
        scheduler
            .slo_snapshot()
            .unwrap()
            .selections
            .joint_decode_only_interventions,
        1
    );
    assert_eq!(
        scheduler.prefill_queue.read()[0]
            .logical_work_frontier
            .progress_generation(),
        original_generation
    );

    // Commit a real prefix portion: the remaining three-token prompt still makes
    // static/half/quarter violate decoder ITL. Only this progress re-enables
    // DecodeOnly; the already computed token and new generation are retained.
    assert!(!scheduler.mark_prefill_chunk_processed(&prefill, 4, 1));
    let committed = scheduler.prefill_queue.read()[0]
        .logical_work_frontier
        .clone();
    assert_ne!(committed.progress_generation(), original_generation);
    let (fourth, fourth_round) = observed_selection(&scheduler, recapture_round(&third_round));
    assert_eq!(fourth.size(), 1);
    assert_eq!(tokens(&fourth, &decoder), 1);
    assert_eq!(fourth_round.statistics.joint_interventions, 1);
    assert_eq!(fourth_round.statistics.joint_decode_only_interventions, 1);
    assert_eq!(
        scheduler
            .slo_snapshot()
            .unwrap()
            .selections
            .joint_decode_only_interventions,
        2
    );
    let queue = scheduler.prefill_queue.read();
    assert_eq!(queue[0].prefill_chunk_offset, 1);
    assert_eq!(queue[0].logical_work_frontier, committed);
    assert!(queue[0].logical_work_frontier.can_restore_at(1));
}
