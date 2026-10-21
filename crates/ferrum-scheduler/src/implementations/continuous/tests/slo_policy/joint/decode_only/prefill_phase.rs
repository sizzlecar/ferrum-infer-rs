//! Actual multi-wave scheduling, rather than a simulated phase-only gain.
use super::*;

mod admission_precommit;

struct PhaseCosts {
    base: ConditionalSnapshot,
    unavailable: bool,
}

impl CostSnapshot for PhaseCosts {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("conditional planning must not make a second scalar query")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        if self.unavailable {
            return CostScenarioPrediction::Unknown(CostUnknownReason::Stale);
        }
        // Scale the cap64 counterexample to model seconds so its ITL margin
        // does not depend on host CPU speed. A decode invocation costs100ms.
        let prefill = query
            .work
            .iter()
            .filter(|work| work.kind == AttemptWorkKind::Prefill)
            .map(|work| work.requested.as_range().len() as u64)
            .sum::<u64>();
        let decode = query
            .work
            .iter()
            .any(|work| work.kind == AttemptWorkKind::Decode);
        let time = Duration::from_millis(prefill * 10_000 + u64::from(decode) * 100);
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
            expires_at: self.base.at + Duration::from_secs(3600),
        }
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.base.stopping_forecast(query)
    }
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.base.completion_forecast(query)
    }
}

struct PhaseFixture {
    scheduler: ContinuousBatchScheduler,
    config: SchedulerSloConfig,
    source: Arc<PhaseCosts>,
    decoder: RequestId,
    prefill: RequestId,
    admitted_at: Instant,
}

impl PhaseFixture {
    fn new() -> Self {
        Self::with_capacity(2)
    }

    fn with_capacity(max_running_requests: usize) -> Self {
        Self::with_admission(max_running_requests, false)
    }

    fn with_admission(max_running_requests: usize, waiting_prefill: bool) -> Self {
        let mut config: SchedulerSloConfig = "ttft:1000000,tpot:50000,itl:15000".parse().unwrap();
        config.strategy = ferrum_types::SloStrategy::Joint;
        config.planning.max_elapsed_us = 1_000_000;
        config.planning.max_scenarios = 2;
        config.joint.request_deadline_ms = 2_000_000;
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests,
            active_decode_prefill_chunk: Some(4),
            prefill_step_chunk: Some(4),
            slo: Some(config),
            ..Default::default()
        });
        let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
        let mut decoder =
            create_test_request_with_prompt_tokens(Priority::Normal, 1).with_ingress(ingress);
        decoder.sampling_params.max_tokens = 64;
        decoder.stream = true;
        let decoder_id = decoder.id.clone();
        let mut prefill =
            create_test_request_with_prompt_tokens(Priority::Normal, 32).with_ingress(ingress);
        prefill.sampling_params.max_tokens = 64;
        prefill.stream = true;
        let prefill_id = prefill.id.clone();
        enqueue_waiting(&scheduler, decoder);
        if waiting_prefill {
            admit(&scheduler, None);
        }
        enqueue_waiting(&scheduler, prefill);
        if !waiting_prefill {
            admit(&scheduler, None);
        }
        scheduler.mark_prefill_complete(&decoder_id, 1);
        scheduler.update_decode_progress(&decoder_id, 1);
        Self {
            scheduler,
            config,
            source: Arc::new(PhaseCosts {
                base: ConditionalSnapshot {
                    at: ingress.instant(),
                    stopping_known: true,
                    completion_known: true,
                },
                unavailable: false,
            }),
            decoder: decoder_id,
            prefill: prefill_id,
            admitted_at: ingress.instant(),
        }
    }

    fn round(&self) -> TimePlanningRound {
        let now = Instant::now();
        let members = self
            .scheduler
            .prefill_queue
            .read()
            .iter()
            .chain(self.scheduler.decode_queue.read().requests.values())
            .chain(self.scheduler.waiting_queue.read().iter())
            .map(|request| {
                (
                    request.inner.request.id.clone(),
                    request.decode_tokens,
                    request.waiting_admission_ticket.unwrap().get(),
                    request.inner.request.sampling_params.max_tokens,
                )
            })
            .collect::<Vec<_>>();
        let rules =
            Arc::new(EffectiveStoppingRules::new(vec![1], StoppingMechanism::CapOnly, 16).unwrap());
        let mut round = TimePlanningRound::new(
            self.source.clone(),
            self.config,
            PrefillShapePolicy::Identity,
            CostExecution::SequentialPrefillThenDecode,
        );
        assert!(
            round.set_request_facts(members.iter().map(|(id, generated, ticket, cap)| (
                id.clone(),
                RequestStoppingObservation {
                    lifecycle: RequestLifecycleId {
                        producer: 1,
                        admission: *ticket,
                    },
                    admitted_at: self.admitted_at,
                    observed_at: now,
                    revision: *generated as u64 + 1,
                    rules: rules.clone(),
                    cap: *cap,
                    generated_tokens: *generated,
                    verified_survival_through: *generated,
                    terminal: None,
                    completion: None,
                },
            )))
        );
        assert!(
            round.set_request_clocks(members.iter().map(|(id, generated, _, _)| (
                id.clone(),
                RequestClockFacts {
                    captured_at: now,
                    age: now.duration_since(self.admitted_at),
                    first_text: (*generated > 0).then_some(Duration::ZERO),
                    last_text: (*generated > 0).then_some(Duration::ZERO),
                    max_text_interval: (*generated > 1).then_some(Duration::ZERO),
                    nonempty_text_events: *generated as u64,
                    violating_text_intervals: 0,
                    itl_threshold: Some(Duration::from_secs_f64(self.config.itl_ms / 1000.0)),
                    generated_tokens: *generated,
                    streaming: true,
                },
            )))
        );
        round.set_capacity(Some(PlanningCapacity {
            block_size: 16,
            total_blocks: 32,
            free_blocks: 32,
            admission_coverage:
                ferrum_execution::model_executor::PagedAdmissionCoverage::AllPersistentState {
                    max_new_sequences: self.scheduler.config.max_running_requests,
                    max_sequence_tokens: 128,
                    max_batch_sequences: self.scheduler.config.max_running_requests,
                },
        }));
        round
    }

    fn next(&self) -> (BatchPlan, TimePlanningRound) {
        observed_selection(&self.scheduler, self.round())
    }
}

#[test]
fn positive_prefill_phase_is_executed_until_prompt_completion_even_after_cost_expiry() {
    let mut fixture = PhaseFixture::new();
    let (mut batch, round) = fixture.next();
    assert_eq!(tokens(&batch, &fixture.prefill), 1);
    assert_eq!(tokens(&batch, &fixture.decoder), 1);
    assert_eq!(round.statistics.joint_interventions, 1);
    assert_eq!(round.statistics.joint_decode_only_interventions, 0);
    assert_eq!(round.statistics.joint_prefill_phase_interventions, 1);

    // The commitment must survive a later stale model. Every selected chunk
    // is actually committed before the next scheduler invocation. Replanning
    // alone never supplies the missing prefill progress.
    fixture.source = Arc::new(PhaseCosts {
        base: ConditionalSnapshot {
            at: fixture.admitted_at,
            stopping_known: true,
            completion_known: true,
        },
        unavailable: true,
    });
    let mut ranges = Vec::new();
    for offset in 0..32 {
        let item = batch
            .requests
            .iter()
            .find(|request| request.request.id == fixture.prefill)
            .unwrap();
        assert_eq!(item.tokens_processed, offset);
        assert_eq!(item.tokens_to_process, Some(1));
        assert_eq!(tokens(&batch, &fixture.decoder), 1);
        ranges.push(item.tokens_processed..item.tokens_processed + 1);
        let done = fixture
            .scheduler
            .mark_prefill_chunk_processed(&fixture.prefill, 32, 1);
        assert_eq!(done, offset == 31);
        fixture
            .scheduler
            .update_decode_progress(&fixture.decoder, offset + 2);
        if done {
            fixture
                .scheduler
                .update_decode_progress(&fixture.prefill, 1);
        } else {
            let (next, next_round) = fixture.next();
            assert_eq!(next_round.statistics.joint_interventions, 0);
            assert_eq!(next_round.statistics.joint_decode_only_interventions, 0);
            assert_eq!(next_round.statistics.joint_prefill_phase_interventions, 0);
            batch = next;
        }
    }
    assert_eq!(
        ranges,
        (0..32).map(|offset| offset..offset + 1).collect::<Vec<_>>()
    );
    assert!(fixture.scheduler.prefill_queue.read().is_empty());
    let (decode, round) = fixture.next();
    assert_eq!(decode.size(), 2);
    assert_eq!(tokens(&decode, &fixture.prefill), 1);
    assert_eq!(tokens(&decode, &fixture.decoder), 1);
    assert!(round
        .selected
        .iter()
        .all(|work| work.kind == AttemptWorkKind::Decode));
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(round.statistics.joint_prefill_phase_interventions, 0);
    assert!(
        fixture.scheduler.decode_queue.read().requests[&fixture.prefill]
            .joint_prefill_phase_cap
            .is_none()
    );
    assert_eq!(
        fixture
            .scheduler
            .slo_snapshot()
            .unwrap()
            .selections
            .joint_interventions,
        1
    );
}

#[test]
fn committed_phase_is_budgeted_before_new_arrivals_without_requiring_new_predictions() {
    let fixture = PhaseFixture::new();
    let (first, _) = fixture.next();
    assert_eq!(tokens(&first, &fixture.prefill), 1);
    assert!(!fixture
        .scheduler
        .mark_prefill_chunk_processed(&fixture.prefill, 32, 1));

    let mut newcomer = create_test_request_with_prompt_tokens(Priority::Normal, 32);
    newcomer.sampling_params.max_tokens = 64;
    let newcomer_id = newcomer.id.clone();
    enqueue_waiting(&fixture.scheduler, newcomer);
    assert!(fixture
        .scheduler
        .waiting_queue
        .read()
        .iter()
        .find(|request| request.inner.request.id == newcomer_id)
        .unwrap()
        .joint_prefill_phase_cap
        .is_none());
    assert!(fixture
        .scheduler
        .cancel_request(fixture.decoder.clone())
        .unwrap());
    admit(&fixture.scheduler, None);

    let mut narrow = hint();
    narrow.max_tokens = 5;
    let batch = fixture
        .scheduler
        .next_batch_with_token_admissions(
            fixture.scheduler.prepare_token_batch_selection(narrow),
            None,
            &mut |_| {},
            None,
        )
        .unwrap()
        .batch
        .unwrap();
    assert_eq!(tokens(&batch, &fixture.prefill), 1);
    assert_eq!(
        tokens(&batch, &newcomer_id),
        4,
        "the committed cap must spend one token, not spend four then clip to one"
    );
    let queue = fixture.scheduler.prefill_queue.read();
    assert_eq!(
        queue
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap()
            .prefill_chunk_offset,
        1
    );
    assert!(queue
        .iter()
        .find(|request| request.inner.request.id == newcomer_id)
        .unwrap()
        .joint_prefill_phase_cap
        .is_none());
}

#[test]
fn phase_survives_uncommitted_replanning_and_same_lifecycle_deferral_but_not_id_reuse() {
    let fixture = PhaseFixture::new();
    let (_, round) = fixture.next();
    assert_eq!(round.statistics.joint_prefill_phase_interventions, 1);
    let (ticket, generation) = {
        let queue = fixture.scheduler.prefill_queue.read();
        let request = queue
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap();
        (
            request.waiting_admission_ticket.unwrap(),
            request.logical_work_frontier.progress_generation(),
        )
    };
    let uncommitted = select(&fixture.scheduler, None).batch.unwrap();
    assert_eq!(tokens(&uncommitted, &fixture.prefill), 1);
    {
        let queue = fixture.scheduler.prefill_queue.read();
        let request = queue
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap();
        assert_eq!(request.prefill_chunk_offset, 0);
        assert_eq!(
            request.logical_work_frontier.progress_generation(),
            generation
        );
    }
    assert!(fixture.scheduler.defer_prefill_to_waiting(&fixture.prefill));
    {
        let waiting = fixture.scheduler.waiting_queue.read();
        let request = waiting
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap();
        assert_eq!(request.waiting_admission_ticket, Some(ticket));
        assert_eq!(request.joint_prefill_phase_cap.unwrap().limit.get(), 1);
    }
    assert!(fixture
        .scheduler
        .cancel_request(fixture.prefill.clone())
        .unwrap());
    // The resource owner reports release separately from logical cancellation.
    fixture.scheduler.record_capacity_release_progress();
    let mut replacement = create_test_request_with_prompt_tokens(Priority::Normal, 32);
    replacement.id = fixture.prefill.clone();
    replacement.sampling_params.max_tokens = 64;
    enqueue_waiting(&fixture.scheduler, replacement);
    admit(&fixture.scheduler, None);
    {
        let queue = fixture.scheduler.prefill_queue.read();
        let request = queue
            .iter()
            .find(|request| request.inner.request.id == fixture.prefill)
            .unwrap();
        assert_ne!(request.waiting_admission_ticket, Some(ticket));
        assert!(request.joint_prefill_phase_cap.is_none());
    }
    let batch = select(&fixture.scheduler, None).batch.unwrap();
    assert_eq!(tokens(&batch, &fixture.prefill), 4);
}

#[test]
fn phase_apply_revalidates_all_members_before_installing_any_commitment() {
    for replace_ticket in [false, true] {
        let mut fixture = PhaseFixture::with_capacity(3);
        fixture.config.itl_ms = 25_000.0;
        fixture.config.joint.itl_quantile = 0.99;
        fixture.scheduler.config.slo = Some(fixture.config);
        let mut other = create_test_request_with_prompt_tokens(Priority::Normal, 32);
        other.sampling_params.max_tokens = 64;
        other.stream = true;
        let other_id = other.id.clone();
        enqueue_waiting(&fixture.scheduler, other);
        admit(&fixture.scheduler, None);

        // Freeze the real static wave, evaluate it without applying, then let
        // a real lifecycle/progress transition race that frozen decision.
        let mut batch = select(&fixture.scheduler, None).batch.unwrap();
        let mut round = fixture.round();
        {
            let prefill = fixture.scheduler.prefill_queue.read();
            let decode = fixture.scheduler.decode_queue.read();
            for scheduled in &batch.requests {
                let request = prefill
                    .iter()
                    .chain(decode.requests.values())
                    .find(|request| request.inner.request.id == scheduled.request.id)
                    .unwrap();
                round.selected_work(fixture.scheduler.cost_work(
                    request,
                    scheduled.tokens_to_process.unwrap(),
                    request.phase == RequestPhase::Decoding,
                    round.prefill_shape_policy,
                ));
            }
        }
        let choice = fixture
            .scheduler
            .compare_joint_first_wave(&hint(), &batch.requests, &mut round, Instant::now())
            .unwrap()
            .expect("the phase quarter is a feasible rescue for both prefills");
        if replace_ticket {
            assert!(fixture.scheduler.cancel_request(other_id.clone()).unwrap());
            let mut replacement = create_test_request_with_prompt_tokens(Priority::Normal, 32);
            replacement.id = other_id.clone();
            replacement.sampling_params.max_tokens = 64;
            replacement.stream = true;
            enqueue_waiting(&fixture.scheduler, replacement);
            admit(&fixture.scheduler, None);
        } else {
            assert!(!fixture
                .scheduler
                .mark_prefill_chunk_processed(&other_id, 32, 1));
        }
        assert!(!fixture
            .scheduler
            .apply_joint_first_wave(&mut batch.requests, &mut round, choice));
        assert_eq!(tokens(&batch, &fixture.prefill), 4);
        assert_eq!(tokens(&batch, &other_id), 4);
        assert!(
            fixture
                .scheduler
                .prefill_queue
                .read()
                .iter()
                .all(|request| request.joint_prefill_phase_cap.is_none()),
            "the first valid prefiller must not acquire a cap before a later stale member rejects"
        );
        assert_eq!(round.statistics.joint_interventions, 0);
        assert_eq!(round.statistics.joint_prefill_phase_interventions, 0);
    }
}

#[test]
fn phase_installation_can_change_future_waves_without_shrinking_the_current_chunk() {
    let fixture = PhaseFixture::with_capacity(3);
    let mut short = create_test_request_with_prompt_tokens(Priority::Normal, 1);
    short.sampling_params.max_tokens = 2;
    short.stream = true;
    let short_id = short.id.clone();
    enqueue_waiting(&fixture.scheduler, short);
    admit(&fixture.scheduler, None);
    fixture.scheduler.mark_prefill_complete(&short_id, 1);
    fixture.scheduler.update_decode_progress(&short_id, 1);

    // Two decoders leave only one current prefill token. The short decoder
    // finishes this wave; without a phase the SAME max_tokens=3 continuation
    // would then allow two-token chunks and violate the long decoder's ITL.
    let mut narrow = hint();
    narrow.max_tokens = 3;
    let mut scope = fixture
        .scheduler
        .observe_planning_round(Some(fixture.round()));
    let batch = fixture
        .scheduler
        .next_batch_with_token_admissions(
            fixture.scheduler.prepare_token_batch_selection(narrow),
            None,
            &mut |_| {},
            scope.as_mut(),
        )
        .unwrap()
        .batch
        .unwrap();
    let round = scope.finish().unwrap();
    assert_eq!(tokens(&batch, &fixture.prefill), 1);
    assert_eq!(round.statistics.shrunk_prefills, 0);
    assert_eq!(round.statistics.joint_interventions, 1);
    assert_eq!(round.statistics.joint_prefill_phase_interventions, 1);
    assert_eq!(round.statistics.joint_decode_only_interventions, 0);
    assert!(!fixture
        .scheduler
        .mark_prefill_chunk_processed(&fixture.prefill, 32, 1));
    fixture
        .scheduler
        .update_decode_progress(&fixture.decoder, 2);
    fixture.scheduler.update_decode_progress(&short_id, 2);
    let mut response = super::super::super::super::capacity_release::response(&short_id);
    response.usage = ferrum_types::TokenUsage::new(1, 2);
    fixture
        .scheduler
        .complete_request(short_id, &response)
        .unwrap();
    let next = select(&fixture.scheduler, None).batch.unwrap();
    assert_eq!(tokens(&next, &fixture.prefill), 1);
    assert_eq!(
        next.requests
            .iter()
            .find(|request| request.request.id == fixture.prefill)
            .unwrap()
            .tokens_processed,
        1
    );
    assert_eq!(
        fixture
            .scheduler
            .slo_snapshot()
            .unwrap()
            .selections
            .joint_prefill_phase_interventions,
        1
    );
}
