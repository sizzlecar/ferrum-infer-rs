//! Shared product construction and actual controlled executor submissions.
//! These are policy/ownership tests, not native performance evidence.
use super::*;
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_types::{SloExperimentStageV1 as Stage, SloOutputTransport};

async fn pre_cost_engine(
    stage: Stage,
    width: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (tokenizer, executor) = startup_components(width).await;
    executor.plain_wave_protocol.store(true, Ordering::Release);
    let mut config = ferrum_types::EngineConfig::default();
    config.batching.prefill_decode_execution = ferrum_types::PrefillDecodeExecution::Split;
    config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    config.scheduler.slo.experiment_stage = Some(stage);
    config.scheduler.slo.output.transport = stage.output_transport();
    config.scheduler.slo.output.max_queued_events_per_request = NonZeroUsize::new(2).unwrap();
    config.scheduler.slo.planner.max_planning_us = NonZeroU64::new(30_000_000).unwrap();
    config.scheduler.slo.default_service_class = Some("ordinary".into());
    for (id, ttft_ms) in [("ordinary", 10_000), ("urgent", 1_000)] {
        config
            .scheduler
            .slo
            .services
            .push(ferrum_types::ServiceSloConfig {
                id: id.into(),
                server_token_commit: ferrum_types::SloLatencyBudgets {
                    ttft_ms: NonZeroU64::new(ttft_ms).unwrap(),
                    tpot_ms: NonZeroU64::new(10_000).unwrap(),
                    itl_ms: NonZeroU64::new(10_000).unwrap(),
                },
                attainment: Default::default(),
                client_visible: None,
            });
    }
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let mut engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler.clone(),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    // The tests drive the actual iteration explicitly, never a parallel driver.
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .bg_loop_spawned
        .store(true, Ordering::Release);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(engine.inner.prefill_reference_runtime.is_none());
    assert_no_cost_activity(&engine);
    (engine, scheduler, executor)
}

fn request(
    engine: &ContinuousBatchEngine,
    tokens: usize,
    maximum: usize,
) -> ferrum_types::InferenceRequest {
    let mut request = ferrum_types::InferenceRequest::new(
        vec!["test"; tokens].join(" "),
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = maximum;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    request
}

fn assert_no_cost_activity(engine: &ContinuousBatchEngine) {
    let runtime = engine
        .inner
        .cost_runtime
        .as_ref()
        .expect("shared frontier authority");
    assert_eq!(
        runtime.test_activity(),
        (0, 0),
        "actual worker ownership and snapshot reads"
    );
    assert_eq!(runtime.trained_samples(), 0);
    let stats = runtime.sink.stats();
    assert_eq!(stats.preparations_started, 0);
    assert_eq!(stats.calls_started, 0);
    assert_eq!(stats.offered, 0);
    assert_eq!(stats.drained, 0);
    assert!(runtime.profile_receipt().is_none());
}

#[tokio::test]
async fn stage_ablation_adaptive_and_single_wave_differ_on_real_split_work() {
    for stage in [Stage::ControlledAdaptiveBaseline, Stage::SingleWave] {
        let (engine, scheduler, executor) = pre_cost_engine(stage, 2).await;
        let decoder = request(&engine, 1, 3);
        let decoder_id = decoder.id.clone();
        let mut decoder_output = engine.infer_stream(decoder).await.unwrap();
        let hint = ferrum_interfaces::BatchHint::simple(2);
        let batch = scheduler.next_batch(hint.clone()).await.unwrap();
        engine.inner.process_batch(&batch).await.unwrap();
        assert_eq!(
            engine.inner.sequences.read()[&decoder_id]
                .generated_tokens
                .len(),
            1
        );
        let prefill_request = request(&engine, 3, 3);
        let prefill_id = prefill_request.id.clone();
        let mut prefill_output = engine.infer_stream(prefill_request).await.unwrap();
        let batch = scheduler.next_batch(hint.clone()).await.unwrap();
        assert!(batch
            .requests
            .iter()
            .any(|row| row.request.id == decoder_id));
        assert!(batch
            .requests
            .iter()
            .any(|row| row.request.id == prefill_id));
        let physical = executor.physical.load(Ordering::Acquire);
        engine.inner.process_batch(&batch).await.unwrap();
        assert_eq!(
            executor.physical.load(Ordering::Acquire) - physical,
            if stage == Stage::SingleWave { 1 } else { 2 }
        );
        assert_eq!(
            engine.inner.sequences.read()[&prefill_id].prefill_tokens_processed,
            if stage == Stage::SingleWave { 0 } else { 3 }
        );
        assert!(matches!(
            engine.inner.prepare_slo_controller(&hint).unwrap(),
            SloIterationPlan::Legacy
        ));
        bounded(async {
            while !engine.inner.sequences.read().is_empty() {
                if let Some(batch) = scheduler.next_batch(hint.clone()).await {
                    engine.inner.process_batch(&batch).await.unwrap();
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
        for output in [&mut decoder_output, &mut prefill_output] {
            let mut final_usage = None;
            while let Some(chunk) = bounded(output.next()).await {
                let chunk = chunk.unwrap();
                if let Some(usage) = chunk.usage {
                    final_usage = Some(usage);
                }
                assert_ne!(chunk.finish_reason, Some(ferrum_types::FinishReason::Error));
            }
            assert_eq!(final_usage.unwrap().completion_tokens, 3);
        }
        assert_no_cost_activity(&engine);
        engine.shutdown().await.unwrap();
        assert_no_cost_activity(&engine);
    }
}

#[tokio::test]
async fn stage_ablation_single_wave_not_submitted_never_advances_or_retries() {
    use crate::continuous_engine::inner::batch::bounded_wave::{
        PlanRuntimeWaveOutcome, PlanRuntimeWaveSelection,
    };
    let (engine, scheduler, executor) = pre_cost_engine(Stage::SingleWave, 1).await;
    let request = request(&engine, 1, 3);
    let id = request.id.clone();
    let output = engine.infer_stream(request).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    let frontier = engine.inner.sequences.read()[&id].cost_frontier;
    executor.deferrals.capacity(std::slice::from_ref(&id), None);
    let outcome = engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &PlanRuntimeWaveSelection::Prefill {
                request_ids: vec![id.clone()],
            },
        )
        .await
        .unwrap();
    assert!(matches!(outcome, PlanRuntimeWaveOutcome::NotSubmitted(_)));
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(engine.inner.sequences.read()[&id].cost_frontier, frontier);
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        0
    );
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    assert_no_cost_activity(&engine);
    drop(output);
    // This fixture owns the driver. Receiver disconnection must go through
    // the same cancellation path as a normal iteration before shutdown.
    engine.inner.cancel_abandoned_requests().await.unwrap();
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn stage_ablation_deadline_orders_original_ingress_without_prediction() {
    let (engine, _, executor) = pre_cost_engine(Stage::DeadlineOnly, 2).await;
    let ingress = slo_clock_now();
    let ordinary = request(&engine, 1, 3);
    let ordinary_id = ordinary.id.clone();
    let ordinary_output = engine
        .infer_credited_stream(
            ordinary,
            InferenceRequestContext::from_ingress(ingress),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(&engine, &ordinary_id).await;
    let urgent = request(&engine, 1, 3);
    let urgent_id = urgent.id.clone();
    let mut urgent_output = engine
        .infer_credited_stream(
            urgent,
            InferenceRequestContext::from_ingress(ingress).with_service_class("urgent".into()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(&engine, &urgent_id).await;
    prefill::admit(&engine, 2).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        engine.inner.prepare_slo_controller(&hint)
    })
    .await;
    assert!(matches!(
        engine
            .inner
            .slo_controller
            .lock()
            .pending_execution
            .as_ref()
            .unwrap()
            .work
            .expected
            .commitment(),
        WaveCommitment::CompleteRequests(CompletionOnlyReason::DeadlinePolicy)
    ));
    bounded(engine.inner.execute_slo_controller_wave(prepared))
        .await
        .unwrap();
    assert_eq!(
        executor.submitted_requests.lock().last().unwrap(),
        &[urgent_id.clone()]
    );
    assert_eq!(
        engine.inner.sequences.read()[&ordinary_id]
            .generated_tokens
            .len(),
        0
    );
    assert_eq!(
        engine.inner.sequences.read()[&urgent_id]
            .slo
            .as_ref()
            .unwrap()
            .ingress(),
        ingress
    );
    drop(bounded(urgent_output.frames.next()).await.unwrap());
    assert_no_cost_activity(&engine);
    drop(ordinary_output);
    cleanup(engine, urgent_output).await;
}

#[tokio::test]
async fn stage_ablation_output_isolation_uses_credits_without_cost_controller() {
    let (engine, scheduler, executor) = pre_cost_engine(Stage::OutputIsolation, 2).await;
    let (id, mut output) = prefill::request(&engine, 1, 3).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    assert!(matches!(
        engine.inner.prepare_slo_controller(&hint).unwrap(),
        SloIterationPlan::Legacy
    ));
    let frontier = engine.inner.sequences.read()[&id].cost_frontier.unwrap();
    let batch = scheduler.next_batch(hint).await.unwrap();
    engine.inner.process_batch(&batch).await.unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(
        engine.inner.sequences.read()[&id]
            .cost_frontier
            .unwrap()
            .owner_incarnation,
        frontier.owner_incarnation
    );
    drop(bounded(output.frames.next()).await.unwrap());
    ready(&engine, &id).await;
    let (prefill_id, prefill_output) = prefill::request(&engine, 3, 3).await;
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    assert!(batch
        .requests
        .iter()
        .any(|row| row.request.id == prefill_id));
    engine.inner.process_batch(&batch).await.unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    {
        let sequences = engine.inner.sequences.read();
        let unselected = &sequences[&prefill_id];
        assert_eq!(unselected.prefill_tokens_processed, 0);
        assert!(unselected.generated_tokens.is_empty());
        assert!(
            unselected.credited_output.as_ref().unwrap().grant.is_none(),
            "unselected participant returns its unused real output grant"
        );
    }
    drop(bounded(output.frames.next()).await.unwrap());
    assert_no_cost_activity(&engine);
    drop(prefill_output);
    cleanup(engine, output).await;
}

#[tokio::test]
async fn stage_ablation_transport_conflicts_fail_before_acceptance() {
    let (engine, _, _) = pre_cost_engine(Stage::SingleWave, 1).await;
    assert!(engine
        .infer_credited_stream(
            request(&engine, 1, 2),
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text())
        )
        .await
        .is_err());
    assert!(engine.inner.sequences.read().is_empty());
    engine.shutdown().await.unwrap();
    let (engine, _, _) = pre_cost_engine(Stage::OutputIsolation, 1).await;
    assert!(engine.infer_stream(request(&engine, 1, 2)).await.is_err());
    assert_eq!(
        engine.inner.config.scheduler.slo.output.transport,
        SloOutputTransport::Credited
    );
    assert!(engine.inner.sequences.read().is_empty());
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn stage_ablation_cost_candidates_disable_both_time_gates() {
    for stage in [Stage::CostCandidates, Stage::Complete] {
        let (mut engine, scheduler, executor) = fixture_with_width(2).await;
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
        inner.config.scheduler.slo.experiment_stage = Some(stage);
        inner.config.scheduler.slo.output.transport = SloOutputTransport::Credited;
        inner.config.scheduler.slo.admission.max_active_requests = NonZeroUsize::new(1).unwrap();
        let (first_id, first) = prefill::request(&engine, 1, 3).await;
        let (second_id, second) = prefill::request(&engine, 1, 3).await;
        prefill::admit(&engine, 2).await;
        assert_eq!(
            scheduler.active_count(),
            if stage == Stage::Complete { 1 } else { 2 }
        );
        assert_eq!(
            executor.physical.load(Ordering::Acquire),
            0,
            "admission only discovers actual resource capacity"
        );
        let before = engine
            .inner
            .cost_runtime
            .as_ref()
            .unwrap()
            .test_activity()
            .1;
        let captured = prefill::captured(&engine, &executor).await;
        assert!(
            engine
                .inner
                .cost_runtime
                .as_ref()
                .unwrap()
                .test_activity()
                .1
                > before,
            "cost stage captures the actual published predictor"
        );
        let proposal = engine.inner.propose_slo_time_admission(&captured);
        assert_eq!(proposal.is_some(), stage == Stage::Complete);
        if stage == Stage::CostCandidates {
            // Execute the regular candidate algorithm with the same captured
            // model. Missing future evidence may remain Unknown in this CPU
            // fixture; neither the test nor the stage fabricates a witness.
            let _decision = engine.inner.propose_slo_controller(&captured);
            let sequences = engine.inner.sequences.read();
            for id in [&first_id, &second_id] {
                let state = sequences[id].time_admission.as_ref().unwrap();
                state.assert_no_time_admission_work();
            }
        }
        drop(captured);
        drop(second);
        cleanup(engine, first).await;
    }
}

#[tokio::test]
async fn stage_ablation_cost_stage_preserves_output_slot_limit_and_reclaims_slot() {
    let (mut engine, scheduler, executor) = fixture_with_custom_config(2, |config| {
        config.scheduler.max_running_requests = 1;
    })
    .await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.experiment_stage = Some(Stage::CostCandidates);
    inner.config.scheduler.slo.output.transport = SloOutputTransport::Credited;
    inner.config.scheduler.slo.admission.max_active_requests = NonZeroUsize::new(2).unwrap();
    let (first_id, first) = prefill::request(&engine, 1, 3).await;
    let second = request(&engine, 1, 3);
    let second_id = second.id.clone();
    let rejected = engine
        .infer_credited_stream(
            second,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await;
    assert!(matches!(
        rejected,
        Err(FerrumError::ResourceExhausted { .. })
    ));
    assert!(!engine.inner.sequences.read().contains_key(&second_id));
    assert_eq!(engine.inner.sequences.read().len(), 1);
    assert_eq!(pool(&engine).snapshot().limits.max_open_accounts, 1);
    prefill::admit(&engine, 2).await;
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    drop(first);
    engine.inner.cancel_abandoned_requests().await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&first_id));
    assert_eq!(scheduler.active_count(), 0);
    let mut changes = pool(&engine).subscribe();
    bounded(async {
        while pool(&engine).snapshot().retained_accounts != 0 {
            changes.changed().await.unwrap();
        }
    })
    .await;
    let (_, third) = prefill::request(&engine, 1, 3).await;
    prefill::admit(&engine, 2).await;
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(pool(&engine).snapshot().open_accounts, 1);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, third).await;
}

#[tokio::test]
async fn stage_ablation_cost_stage_preserves_real_allocator_capacity_and_release_epoch() {
    let (mut engine, scheduler, executor) = fixture_with_custom_config(2, |config| {
        config.scheduler.max_running_requests = 2;
    })
    .await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.experiment_stage = Some(Stage::CostCandidates);
    inner.config.scheduler.slo.output.transport = SloOutputTransport::Credited;
    inner.config.scheduler.slo.admission.max_active_requests = NonZeroUsize::new(2).unwrap();
    executor.enable_real_admission_capacity(1);
    let (first_id, first) = prefill::request(&engine, 1, 3).await;
    let (second_id, second) = prefill::request(&engine, 1, 3).await;
    assert_eq!(pool(&engine).snapshot().open_accounts, 2);
    let second_ingress = engine.inner.sequences.read()[&second_id]
        .slo
        .as_ref()
        .unwrap()
        .ingress();
    let second_frontier = engine.inner.sequences.read()[&second_id].cost_frontier;
    prefill::admit(&engine, 2).await;
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(executor.retained_admission_owners(), 1);
    assert_eq!(
        scheduler.trace_phase(&first_id),
        Some(RequestPhase::Prefilling)
    );
    assert_eq!(
        scheduler.trace_phase(&second_id),
        Some(RequestPhase::Waiting)
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let held_epoch = executor.execution_capacity_epochs().unwrap().unwrap();
    // Unchanged physical capacity cannot be reinterpreted as a time-policy
    // bypass or another admission opportunity.
    assert!(!engine.inner.prepare_slo_admission_turn(2).await.unwrap());
    assert_eq!(executor.retained_admission_owners(), 1);
    assert_eq!(
        engine.inner.sequences.read()[&second_id].cost_frontier,
        second_frontier
    );

    drop(first);
    engine.inner.cancel_abandoned_requests().await.unwrap();
    assert_eq!(executor.retained_admission_owners(), 0);
    let released_epoch = executor.execution_capacity_epochs().unwrap().unwrap();
    assert_eq!(released_epoch.coordinator_id, held_epoch.coordinator_id);
    assert!(released_epoch.release_epoch > held_epoch.release_epoch);
    prefill::admit(&engine, 2).await;
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(executor.retained_admission_owners(), 1);
    assert_eq!(
        engine.inner.sequences.read()[&second_id]
            .slo
            .as_ref()
            .unwrap()
            .ingress(),
        second_ingress
    );
    assert_eq!(
        engine.inner.sequences.read()[&second_id].cost_frontier,
        second_frontier
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, second).await;
    assert_eq!(executor.retained_admission_owners(), 0);
}

#[tokio::test]
async fn stage_ablation_deadline_zero_submit_keeps_owner_and_complete_requests_fallback() {
    let (engine, _, executor) = pre_cost_engine(Stage::DeadlineOnly, 1).await;
    let (id, output) = prefill::request(&engine, 1, 3).await;
    prefill::admit(&engine, 1).await;
    let frontier = engine.inner.sequences.read()[&id].cost_frontier;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        engine.inner.prepare_slo_controller(&hint)
    })
    .await;
    executor
        .deferrals
        .capacity(std::slice::from_ref(&id), Some(false));
    bounded(engine.inner.execute_slo_controller_wave(prepared))
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(engine.inner.sequences.read()[&id].cost_frontier, frontier);
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    assert_no_cost_activity(&engine);
    cleanup(engine, output).await;

    // Losing timing evidence cannot revoke accepted work in CompleteRequests.
    let (engine, _, executor) = pre_cost_engine(Stage::DeadlineOnly, 1).await;
    let (id, mut output) = prefill::request(&engine, 1, 3).await;
    prefill::admit(&engine, 1).await;
    engine.inner.sequences.write().get_mut(&id).unwrap().slo = None;
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        engine.inner.prepare_slo_controller(&hint)
    })
    .await;
    bounded(engine.inner.execute_slo_controller_wave(prepared))
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    drop(bounded(output.frames.next()).await.unwrap());
    assert_no_cost_activity(&engine);
    cleanup(engine, output).await;
}
