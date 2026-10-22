//! The former future-waiter counterfactual now crosses the real cap commit
//! boundary. Executor admission and slot release are still separate receipts.
use super::*;

fn complete(scheduler: &ContinuousBatchScheduler, id: &RequestId, prompt: usize) {
    let response = InferenceResponse {
        request_id: id.clone(),
        text: String::new(),
        tokens: Vec::new(),
        finish_reason: ferrum_types::FinishReason::Length,
        usage: ferrum_types::TokenUsage::new(prompt, 64),
        latency_ms: 0,
        created_at: chrono::Utc::now(),
        metadata: Default::default(),
        api_response: None,
        execution_evidence: None,
    };
    scheduler.complete_request(id.clone(), &response).unwrap();
}

fn waiting_cap(fixture: &FutureFixture) -> Option<PrefillPhaseCap> {
    fixture
        .scheduler
        .waiting_queue
        .read()
        .iter()
        .next()
        .unwrap()
        .joint_prefill_phase_cap
}

#[test]
fn deferred_prefill_commit_waits_for_actual_slot_then_preserves_cap_to_full_completion() {
    let mut fixture = FutureFixture::new(false);
    let waiter = fixture.requests[2].id.clone();
    let mut proposal_round = fixture.round();
    let proposal = fixture
        .scheduler
        .evaluate_joint_admission(
            &FutureFixture::hint(),
            &mut proposal_round,
            fixture.now,
            AdmissionCapMode::Propose,
            0,
        )
        .unwrap();
    assert!(matches!(
        proposal.decision.choice,
        PolicyChoice::Alternative(_)
    ));
    let cap = proposal.prefill_cap.unwrap();
    assert_eq!(cap.limit.get(), 1);
    assert_eq!(cap.targets.len(), 1);
    assert_eq!(cap.targets[0].phase, RequestPhase::Waiting);
    assert_eq!(cap.targets[0].prompt_tokens, 64);
    assert!(
        waiting_cap(&fixture).is_none(),
        "simulation does not install the cap"
    );

    let original_frontier = fixture
        .scheduler
        .waiting_queue
        .read()
        .iter()
        .next()
        .unwrap()
        .logical_work_frontier
        .clone();
    let mut round = fixture.round();
    fixture
        .scheduler
        .prepare_joint_admission_gate(&FutureFixture::hint(), &mut round, fixture.now);
    assert_eq!(round.statistics.joint_admission_prefill_phase_commits, 1);
    assert_eq!(round.statistics.joint_interventions, 0);
    let diagnostics = &round.statistics.joint_decision.admission;
    assert_eq!(diagnostics.outcomes.requires_action, 1);
    assert_eq!(diagnostics.outcomes.static_feasible, 1);
    assert_eq!(diagnostics.consumed, 2);
    assert_eq!(waiting_cap(&fixture).unwrap().limit.get(), 1);
    assert_eq!(
        fixture
            .scheduler
            .waiting_queue
            .read()
            .iter()
            .next()
            .unwrap()
            .logical_work_frontier,
        original_frontier
    );
    assert_eq!(fixture.scheduler.active_count(), 2);
    let mut full = fixture.scheduler.begin_token_admission(1, None).unwrap();
    assert!(full.next(&mut |_| {}, Some(&mut round)).unwrap().is_none());
    full.finish().unwrap();
    assert_eq!(
        fixture.scheduler.waiting_count(),
        1,
        "a cap never grants a third physical owner"
    );

    // Execute the original pure-decode wave through ordinary scheduler work,
    // then provide the original terminal/release receipts. No waiter work was
    // smuggled into the retiring wave.
    let batch = fixture
        .scheduler
        .next_batch_with_token_admissions(
            fixture
                .scheduler
                .prepare_token_batch_selection(FutureFixture::hint()),
            None,
            &mut |_| {},
            None,
        )
        .unwrap()
        .batch
        .unwrap();
    assert_eq!(batch.requests.len(), 2);
    assert!(batch
        .requests
        .iter()
        .all(|member| member.request.id != waiter && member.tokens_to_process == Some(1)));
    fixture
        .scheduler
        .update_decode_progress(&fixture.requests[0].id, 64);
    fixture
        .scheduler
        .update_decode_progress(&fixture.requests[1].id, 2);
    complete(&fixture.scheduler, &fixture.requests[0].id, 1);
    fixture.scheduler.record_capacity_release_progress();

    // A changed cohort cannot reuse the old capacity proof or gate. Capture
    // the remaining two members with fresh test-owner capacity evidence.
    fixture.requests.remove(0);
    fixture.requests[0].output_tokens = 2;
    fixture.requests[0].age += Duration::from_millis(100);
    fixture.requests[0].central_clock.last_output = Some(fixture.requests[0].age);
    fixture.requests[0].central_clock.maximum_output_interval = Duration::from_millis(100);
    fixture.requests[0].central_clock.interval_count = 1;
    fixture.requests[0].risk_clock = fixture.requests[0].central_clock;
    fixture.requests[1].age += Duration::from_millis(100);
    let mut fresh = fixture.round();
    fixture
        .scheduler
        .prepare_joint_admission_gate(&FutureFixture::hint(), &mut fresh, fixture.now);
    assert_eq!(fresh.statistics.joint_admission_prefill_phase_commits, 0);
    assert_eq!(
        fresh
            .statistics
            .joint_decision
            .admission
            .outcomes
            .static_feasible,
        1
    );
    let mut scan = fixture.scheduler.begin_token_admission(1, None).unwrap();
    let candidate = scan
        .next(&mut |_| {}, Some(&mut fresh))
        .unwrap()
        .expect("released slot and fresh gate permit the original executor probe");
    assert_eq!(candidate.request().id, waiter);
    assert!(matches!(
        scan.apply(
            candidate,
            AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                request_id: waiter.clone()
            })
        ),
        TokenAdmissionApply::Applied { accepted: Some(_) }
    ));
    scan.finish().unwrap();

    // Existing committed caps do not require another model prediction. Report
    // actual scheduler work completion, including all 64 output tokens; this
    // is a scheduler fixture, not an inference or measured-latency claim.
    let decoder = fixture.requests[0].id.clone();
    let mut outputs = HashMap::from([(decoder.clone(), 2usize), (waiter.clone(), 0usize)]);
    let mut prefilled = 0usize;
    let work_bound = 64 + (64 - 1) + (64 - 2);
    for _ in 0..work_bound {
        if fixture.scheduler.active_count() == 0 {
            break;
        }
        let batch = fixture
            .scheduler
            .next_batch_with_token_admissions(
                fixture
                    .scheduler
                    .prepare_token_batch_selection(FutureFixture::hint()),
                None,
                &mut |_| {},
                None,
            )
            .unwrap()
            .batch
            .unwrap();
        assert!(!batch.requests.is_empty());
        for member in batch.requests {
            let id = &member.request.id;
            assert_eq!(member.tokens_to_process, Some(1));
            let generated = if *id == waiter && prefilled < 64 {
                assert_eq!(member.tokens_processed, prefilled);
                prefilled += 1;
                let done = fixture.scheduler.mark_prefill_chunk_processed(id, 64, 1);
                assert_eq!(done, prefilled == 64);
                if !done {
                    continue;
                }
                1
            } else {
                outputs[id] + 1
            };
            outputs.insert(id.clone(), generated);
            fixture.scheduler.update_decode_progress(id, generated);
            if generated == 64 {
                complete(&fixture.scheduler, id, if *id == waiter { 64 } else { 1 });
            }
        }
    }
    assert_eq!(prefilled, 64);
    assert!(outputs.values().all(|tokens| *tokens == 64));
    assert_eq!(fixture.scheduler.active_count(), 0);
    assert_eq!(fixture.scheduler.waiting_count(), 0);
}

#[test]
fn deferred_prefill_commit_keeps_missing_capacity_and_irreversible_history_boundaries() {
    let missing = FutureFixture::new(false);
    let mut round = missing.round();
    round.set_capacity(None);
    missing
        .scheduler
        .prepare_joint_admission_gate(&FutureFixture::hint(), &mut round, missing.now);
    assert!(waiting_cap(&missing).is_none());
    assert_eq!(round.statistics.joint_admission_prefill_phase_commits, 0);
    let historical = FutureFixture::new(true);
    let mut round = historical.round();
    let decision = historical
        .scheduler
        .evaluate_joint_admission(
            &FutureFixture::hint(),
            &mut round,
            historical.now,
            AdmissionCapMode::Propose,
            0,
        )
        .unwrap();
    assert_eq!(decision.decision.choice, PolicyChoice::AllKnownInfeasible);
    assert!(decision.prefill_cap.is_none());
    assert_eq!(decision.decision.alternative_risks.len(), 2);
    assert!(decision
        .decision
        .alternative_risks
        .iter()
        .all(|risk| risk.ttft.lower == 1.0 / 3.0));
    assert!(waiting_cap(&historical).is_none());
}

#[test]
fn deferred_prefill_commit_fresh_pass_does_not_replenish_exhausted_queries() {
    let fixture = FutureFixture::new(false);
    let mut first = fixture.round();
    first.config.planning.max_cost_cache_bytes = 0;
    let decision = fixture
        .scheduler
        .evaluate_joint_admission(
            &FutureFixture::hint(),
            &mut first,
            fixture.now,
            AdmissionCapMode::Propose,
            0,
        )
        .unwrap();
    assert!(decision.prefill_cap.is_some());
    let mut limited = fixture.round();
    limited.config.planning.max_cost_cache_bytes = 0;
    limited.config.planning.max_queries = first.query_count;
    fixture.scheduler.prepare_joint_admission_gate(
        &FutureFixture::hint(),
        &mut limited,
        fixture.now,
    );
    assert_eq!(limited.statistics.joint_admission_prefill_phase_commits, 1);
    assert_eq!(limited.query_count, first.query_count);
    assert_eq!(
        limited
            .statistics
            .joint_decision
            .admission
            .outcomes
            .static_feasible,
        0
    );
    assert_eq!(
        serde_json::to_value(limited.statistics.joint_unsupported).unwrap()["query_budget"],
        1
    );
    assert_eq!(
        waiting_cap(&fixture).unwrap().limit.get(),
        1,
        "a failed fresh gate does not undo the policy commitment"
    );
    assert_eq!(fixture.scheduler.active_count(), 2);
    assert_eq!(fixture.scheduler.waiting_count(), 1);
}

struct Stale(ScaledCosts);
impl CostSnapshot for Stale {
    fn revision(&self) -> u64 {
        2
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Unknown(CostUnknownReason::Stale)
    }
    fn cost_scenarios(&self, _: &CostQuery<'_>, _: Instant, _: usize) -> CostScenarioPrediction {
        CostScenarioPrediction::Unknown(CostUnknownReason::Stale)
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        self.0.stopping_forecast(query)
    }
    fn completion_forecast(&self, query: &CompletionQuery<'_>) -> CompletionForecast {
        self.0.completion_forecast(query)
    }
}

#[test]
fn deferred_prefill_commit_needs_fresh_prediction_but_staleness_cannot_undo_prior_cap() {
    let fixture = FutureFixture::new(false);
    let mut stale = fixture.round();
    stale.snapshot = Arc::new(Stale(ScaledCosts(fixture.now)));
    assert_eq!(
        fixture
            .scheduler
            .evaluate_joint_admission(
                &FutureFixture::hint(),
                &mut stale,
                fixture.now,
                AdmissionCapMode::Propose,
                0,
            )
            .unwrap_err(),
        Unsupported::Cost(CostUnknownReason::Stale)
    );
    assert!(waiting_cap(&fixture).is_none());

    let mut known = fixture.round();
    fixture
        .scheduler
        .prepare_joint_admission_gate(&FutureFixture::hint(), &mut known, fixture.now);
    assert_eq!(known.statistics.joint_admission_prefill_phase_commits, 1);
    let cap = waiting_cap(&fixture);
    let mut stale = fixture.round();
    stale.snapshot = Arc::new(Stale(ScaledCosts(fixture.now)));
    fixture
        .scheduler
        .prepare_joint_admission_gate(&FutureFixture::hint(), &mut stale, fixture.now);
    assert_eq!(stale.statistics.joint_admission_prefill_phase_commits, 0);
    assert_eq!(
        stale
            .statistics
            .joint_decision
            .admission
            .outcomes
            .static_feasible,
        0
    );
    assert_eq!(waiting_cap(&fixture), cap);
    assert_eq!(fixture.scheduler.active_count(), 2);
    assert_eq!(fixture.scheduler.waiting_count(), 1);
}
