use super::*;
use crate::continuous_engine::inner::batch::bounded_wave::{
    PlanRuntimeWaveOutcome, PlanRuntimeWaveSelection,
};

fn mixed_selection(prefill_id: RequestId, decode_id: RequestId) -> PlanRuntimeWaveSelection {
    PlanRuntimeWaveSelection::Mixed {
        prefill_ids: vec![prefill_id],
        decode_ids: vec![decode_id],
    }
}

#[tokio::test]
async fn bounded_wave_split_returns_before_future_prefill_and_decode_work() {
    let (mut engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture_with_execution(Some(MixedBehavior::Exact), true, None).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    engine.inner.process_batch(&batch).await.unwrap();
    {
        let sequences = engine.inner.sequences.read();
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
        assert_eq!(sequences[&prefill_id].prefill_tokens_processed, 0);
        assert!(!sequences[&prefill_id].prefill_complete);
        assert!(sequences[&prefill_id].generated_tokens.is_empty());
    }
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    // A fresh explicit selection is needed to perform the deferred phase.
    let outcome = engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &PlanRuntimeWaveSelection::Prefill {
                request_ids: vec![prefill_id.clone()],
            },
        )
        .await
        .unwrap();
    let PlanRuntimeWaveOutcome::Completed(receipt) = outcome else {
        panic!("prefill did not complete")
    };
    assert!(receipt.decode_request_ids.is_empty());
    assert_eq!(receipt.prefills.len(), 1);
    assert_eq!(receipt.prefills[0].request_id, prefill_id);
    assert_eq!(receipt.prefills[0].completed_chunk.range(), 0..4);
    assert_eq!(
        engine.inner.sequences.read()[&decode_id]
            .generated_tokens
            .len(),
        2
    );
    assert_eq!(
        engine.inner.sequences.read()[&prefill_id]
            .generated_tokens
            .len(),
        1
    );
}

#[tokio::test]
async fn bounded_wave_receipt_reports_actual_narrowed_progress_before_replan() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::NarrowPrefill), true).await;
    let outcome = engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &mixed_selection(prefill_id.clone(), decode_id.clone()),
        )
        .await
        .unwrap();
    let PlanRuntimeWaveOutcome::Completed(receipt) = outcome else {
        panic!("mixed wave did not complete")
    };
    assert_eq!(receipt.decode_request_ids, vec![decode_id.clone()]);
    let progress = &receipt.prefills[0];
    assert_eq!(progress.request_id, prefill_id);
    assert_eq!(progress.planned_chunk.range(), 0..4);
    assert_eq!(progress.completed_chunk.range(), 0..1);
    assert_eq!(progress.capacity_probe_count, 1);
    assert!(receipt.executor_started_at <= receipt.executor_completed_at);
    assert!(receipt.executor_completed_at <= receipt.commit_completed_at);
    {
        let sequences = engine.inner.sequences.read();
        assert_eq!(sequences[&prefill_id].prefill_tokens_processed, 1);
        assert!(sequences[&prefill_id].generated_tokens.is_empty());
        assert!(!sequences[&prefill_id].prefill_complete);
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
    }
    // Reusing the old planned offset is rejected before any new execution.
    assert!(engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &PlanRuntimeWaveSelection::Prefill {
                request_ids: vec![prefill_id.clone()]
            }
        )
        .await
        .is_err());
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert!(engine.inner.sequences.read().contains_key(&prefill_id));
}

#[tokio::test]
async fn bounded_wave_deferral_and_unsupported_preserve_frontiers_without_fallback() {
    for behavior in [None, Some(MixedBehavior::NotSubmitted)] {
        let (engine, executor, batch, prefill_id, decode_id) = mixed_fixture(behavior, true).await;
        let cache_before = engine.inner.sequences.read()[&decode_id]
            .kv_cache_handle()
            .unwrap()
            .cache_id();
        let outcome = engine
            .inner
            .execute_plan_runtime_wave(
                &batch,
                &mixed_selection(prefill_id.clone(), decode_id.clone()),
            )
            .await
            .unwrap();
        match (behavior, outcome) {
            (None, PlanRuntimeWaveOutcome::Unsupported) => {}
            (Some(MixedBehavior::NotSubmitted), PlanRuntimeWaveOutcome::NotSubmitted(_)) => {}
            _ => panic!("lost typed zero-submission disposition"),
        }
        let sequences = engine.inner.sequences.read();
        assert_eq!(sequences[&prefill_id].prefill_tokens_processed, 0);
        assert!(sequences[&prefill_id].generated_tokens.is_empty());
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 1);
        assert_eq!(
            sequences[&decode_id].kv_cache_handle().unwrap().cache_id(),
            cache_before
        );
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    }
}

#[tokio::test]
async fn bounded_wave_unknown_submission_and_bad_receipts_make_all_participants_terminal() {
    for behavior in [
        MixedBehavior::SubmittedError,
        MixedBehavior::WrongDecodeCache,
        MixedBehavior::ShortPrefill,
    ] {
        let (engine, executor, batch, prefill_id, decode_id) =
            mixed_fixture(Some(behavior), true).await;
        let selection = mixed_selection(prefill_id.clone(), decode_id.clone());
        assert!(engine
            .inner
            .execute_plan_runtime_wave(&batch, &selection)
            .await
            .is_err());
        assert!(!engine.inner.sequences.read().contains_key(&prefill_id));
        assert!(!engine.inner.sequences.read().contains_key(&decode_id));
        assert!(engine
            .inner
            .scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(2))
            .await
            .is_none());
        assert!(engine
            .inner
            .execute_plan_runtime_wave(&batch, &selection)
            .await
            .is_err());
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn bounded_wave_cleanup_failure_still_terminates_other_participants() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::SubmittedError), true).await;
    assert!(engine
        .inner
        .scheduler
        .cancel(prefill_id.clone())
        .await
        .unwrap());
    assert!(engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &mixed_selection(prefill_id.clone(), decode_id.clone())
        )
        .await
        .is_err());
    assert!(!engine.inner.sequences.read().contains_key(&prefill_id));
    assert!(!engine.inner.sequences.read().contains_key(&decode_id));
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn bounded_wave_rejects_duplicate_foreign_and_wrong_phase_selections_before_submit() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::Exact), true).await;
    for selection in [
        PlanRuntimeWaveSelection::Prefill {
            request_ids: vec![prefill_id.clone(), prefill_id.clone()],
        },
        PlanRuntimeWaveSelection::Decode {
            request_ids: vec![RequestId::new()],
        },
        PlanRuntimeWaveSelection::Decode {
            request_ids: vec![prefill_id.clone()],
        },
        PlanRuntimeWaveSelection::Prefill {
            request_ids: vec![decode_id.clone()],
        },
        PlanRuntimeWaveSelection::Mixed {
            prefill_ids: vec![prefill_id.clone()],
            decode_ids: vec![],
        },
    ] {
        assert!(engine
            .inner
            .execute_plan_runtime_wave(&batch, &selection)
            .await
            .is_err());
    }
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        engine.inner.sequences.read()[&prefill_id].prefill_tokens_processed,
        0
    );
    assert_eq!(
        engine.inner.sequences.read()[&decode_id]
            .generated_tokens
            .len(),
        1
    );
}

#[tokio::test]
async fn bounded_wave_observe_keeps_existing_split_execution() {
    let (mut engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture_with_execution(Some(MixedBehavior::Exact), true, None).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Observe;
    engine.inner.process_batch(&batch).await.unwrap();
    let sequences = engine.inner.sequences.read();
    assert_eq!(sequences[&prefill_id].generated_tokens.len(), 1);
    assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn bounded_wave_active_observer_falls_back_once_when_executor_has_no_evidence_capability() {
    use crate::continuous_engine::inner::cost_observation::{CostCallRejection, EngineCostRuntime};
    let (mut engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::Exact), true).await;
    // This fixture publishes frontiers directly. Install the actual observer
    // and initialize those frontiers explicitly, rather than only toggling a
    // config field on an engine constructed with Off.
    let runtime = Arc::new(
        EngineCostRuntime::with_clock(
            executor.execution_cost_identity(),
            Arc::new(super::inner::cost_observation::EngineCostClock::default()),
        )
        .unwrap(),
    );
    {
        let inner = Arc::get_mut(&mut engine.inner).unwrap();
        inner.config.scheduler.slo.mode = ferrum_types::SloMode::Observe;
        inner.cost_runtime = Some(runtime.clone());
    }
    {
        let mut sequences = engine.inner.sequences.write();
        for sequence in sequences.values_mut() {
            engine.inner.initialize_sequence_cost(sequence);
            assert!(sequence.cost_frontier.is_some());
        }
    }
    let outcome = engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &mixed_selection(prefill_id.clone(), decode_id.clone()),
        )
        .await
        .unwrap();
    assert!(matches!(outcome, PlanRuntimeWaveOutcome::Completed(_)));
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    let sequences = engine.inner.sequences.read();
    assert_eq!(sequences[&prefill_id].generated_tokens.len(), 1);
    assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
    drop(sequences);
    runtime.consume_samples();
    let stats = runtime.sink.stats();
    assert_eq!(stats.rejected(CostCallRejection::Unavailable), 1);
    assert_eq!(stats.published, 0);
    assert_eq!(runtime.trained_samples(), 0);
}

#[tokio::test]
async fn bounded_wave_awaits_executor_completion_before_publishing_any_frontier() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::YieldBeforeCompletion), true).await;
    let selection = mixed_selection(prefill_id.clone(), decode_id.clone());
    let mut execution = Box::pin(engine.inner.execute_plan_runtime_wave(&batch, &selection));
    assert!(futures::poll!(execution.as_mut()).is_pending());
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    {
        let sequences = engine.inner.sequences.read();
        assert_eq!(sequences[&prefill_id].prefill_tokens_processed, 0);
        assert!(sequences[&prefill_id].generated_tokens.is_empty());
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 1);
    }
    let PlanRuntimeWaveOutcome::Completed(receipt) = execution.await.unwrap() else {
        panic!("completion did not produce a receipt")
    };
    assert_eq!(receipt.prefills[0].completed_chunk.range(), 0..4);
    assert_eq!(receipt.decode_request_ids, vec![decode_id.clone()]);
    assert_eq!(
        engine.inner.sequences.read()[&prefill_id]
            .generated_tokens
            .len(),
        1
    );
    assert_eq!(
        engine.inner.sequences.read()[&decode_id]
            .generated_tokens
            .len(),
        2
    );
}

#[tokio::test]
async fn bounded_wave_decode_deferral_does_not_split_or_recompute_in_same_call() {
    let (engine, scheduler, executor, tokenizer) = plan_runtime_batch_decode_test_engine(
        PlanRuntimeBatchDecodeBehavior::DeferUntilPeerCacheRelease,
    );
    let (request_ids, initial_tokens, cache_ids) =
        install_plan_runtime_decode_cohort(&engine, &scheduler, tokenizer).await;
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    let outcome = engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &PlanRuntimeWaveSelection::Decode {
                request_ids: request_ids.clone(),
            },
        )
        .await
        .unwrap();
    assert!(matches!(outcome, PlanRuntimeWaveOutcome::NotSubmitted(_)));
    let sequences = engine.inner.sequences.read();
    for ((rid, initial), cache_id) in request_ids.iter().zip(initial_tokens).zip(cache_ids) {
        assert_eq!(sequences[rid].generated_tokens, vec![initial]);
        assert!(sequences[rid].prefill_complete);
        assert_eq!(
            sequences[rid].kv_cache_handle().unwrap().cache_id(),
            cache_id
        );
    }
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.released_cache_count.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.active_count(), 2);
}

#[tokio::test]
async fn bounded_wave_batch_prefill_unsupported_does_not_run_individual_fallbacks() {
    let (engine, scheduler, _executor, tokenizer) =
        plan_runtime_batch_decode_test_engine(PlanRuntimeBatchDecodeBehavior::Exact);
    let mut request_ids = Vec::new();
    for _ in 0..2 {
        let mut request = policy_request();
        request.prompt = "test".to_owned();
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.to_owned(), serde_json::json!(4));
        scheduler.submit(request.clone()).await.unwrap();
        let rid = request.id.clone();
        let sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request,
            vec![TokenId::new(5); 4],
            Some(tokenizer.clone()),
            Some(64),
        );
        engine.inner.sequences.write().insert(rid.clone(), sequence);
        request_ids.push(rid);
    }
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    let outcome = engine
        .inner
        .execute_plan_runtime_wave(
            &batch,
            &PlanRuntimeWaveSelection::Prefill {
                request_ids: request_ids.clone(),
            },
        )
        .await
        .unwrap();
    assert!(matches!(outcome, PlanRuntimeWaveOutcome::Unsupported));
    let sequences = engine.inner.sequences.read();
    for rid in request_ids {
        assert_eq!(sequences[&rid].prefill_tokens_processed, 0);
        assert!(sequences[&rid].generated_tokens.is_empty());
        assert!(sequences[&rid].kv_cache_handle().is_none());
    }
}
