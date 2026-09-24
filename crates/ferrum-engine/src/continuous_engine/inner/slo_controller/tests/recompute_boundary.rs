//! Real product tokenization, capacity yield, exact completion-only work and
//! host receipts. Native physical encoding has separate backend contracts.
use super::*;
use ferrum_interfaces::engine::InferenceEngine;

fn queue(engine: &ContinuousBatchEngine) -> PlanningQueueSnapshot {
    let mut availability = Vec::new();
    let epochs = engine
        .inner
        .model_executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    engine
        .inner
        .scheduler
        .planning_state(
            NonZeroUsize::new(8).unwrap(),
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability,
            ),
        )
        .unwrap()
}

async fn admit(engine: &ContinuousBatchEngine) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    assert!(engine.inner.prepare_slo_admission_turn(1).await.unwrap());
}

async fn step(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    expected_offset: usize,
    tokens: usize,
) {
    let mut hint = ferrum_interfaces::BatchHint::simple(1);
    hint.max_tokens = tokens;
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
    // A successful capture may still race resource/output publication. Use
    // the real retry wake while preserving this budget and every work frontier;
    // the helper rejects permanent evidence failures and any physical submit.
    let prepared = prefill::selected_after_retry(engine, executor, || {
        engine.inner.prepare_completion_controller(
            &hint,
            &budget,
            CompletionOnlyReason::CostUnavailable,
        )
    })
    .await;
    {
        let state = engine.inner.slo_controller.lock();
        let batch = &state.pending_execution.as_ref().unwrap().work.batch;
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(
            (
                batch.requests[0].tokens_processed,
                batch.requests[0].tokens_to_process
            ),
            (expected_offset, Some(tokens))
        );
    }
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
}

async fn consume(
    engine: &ContinuousBatchEngine,
    id: &RequestId,
    session: &mut CreditedOutputSession,
) {
    drop(
        bounded(session.frames.next())
            .await
            .expect("committed output frame"),
    );
    ready(engine, id).await;
}

#[tokio::test]
async fn completion_rebuilds_full_context_after_capacity_yield_and_resumes_decode() {
    let (engine, _, executor) = completion_fixture(1).await;
    let mut request = ferrum_types::InferenceRequest::new(
        "test test test test",
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    // A client hint cannot override actual ingress tokenization.
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.to_owned(), 999.into());
    let id = request.id.clone();
    let mut session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(&engine, &id).await;
    assert_eq!(queue(&engine).requests()[0].prefill_context_tokens, Some(4));
    assert!(engine.inner.prefill_reference_runtime.is_none());
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    admit(&engine).await;
    step(&engine, &executor, 0, 4).await;
    consume(&engine, &id, &mut session).await;
    {
        let state = queue(&engine);
        let row = &state.requests()[0];
        assert_eq!((row.computed_tokens, row.committed_output_tokens), (4, 1));
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.input_tokens.len(), 4);
        assert_eq!(sequence.prefill_context_len(), 5);
        assert_eq!(sequence.sampling_params.max_tokens, 4);
    }
    {
        let _iteration = engine.inner.iteration_lock.lock().await;
        assert!(
            engine
                .inner
                .defer_decode_for_capacity_recompute(&id, 1, None)
                .await
        );
    }
    let waiting = queue(&engine);
    let row = &waiting.requests()[0];
    assert_eq!(row.queue, PlanningQueueKind::Waiting);
    assert_eq!(row.recompute_target_tokens, Some(4));
    assert_eq!(row.prefill_context_tokens, Some(5));
    assert_eq!((row.computed_tokens, row.prefill_offset), (0, 0));
    drop(waiting);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    admit(&engine).await;
    assert_eq!(queue(&engine).requests()[0].prefill_context_tokens, Some(5));
    step(&engine, &executor, 0, 5).await;
    consume(&engine, &id, &mut session).await;
    let rebuilt = queue(&engine);
    let row = &rebuilt.requests()[0];
    assert_eq!(row.queue, PlanningQueueKind::Decode);
    assert_eq!(
        (
            row.computed_tokens,
            row.resident_tokens,
            row.scheduled_tokens
        ),
        (5, 5, 5)
    );
    assert_eq!(row.committed_output_tokens, 2);
    assert_eq!(row.maximum_output_tokens, 4);
    drop(rebuilt);
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 5);
        assert_eq!(sequence.generated_tokens.len(), 2);
        assert_eq!(sequence.kv_cache_handle().unwrap().num_tokens(), 5);
    }
    step(&engine, &executor, 5, 1).await;
    consume(&engine, &id, &mut session).await;
    assert_eq!(
        (
            queue(&engine).requests()[0].computed_tokens,
            engine.inner.sequences.read()[&id].generated_tokens.len()
        ),
        (6, 3)
    );
    // The unchanged four-output request finishes through one further exact
    // decode, with no duplicate output or hidden physical retry.
    step(&engine, &executor, 6, 1).await;
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(executor.physical.load(Ordering::Acquire), 4);
    assert_eq!(
        executor.submitted_requests.lock().as_slice(),
        &[
            vec![id.clone()],
            vec![id.clone()],
            vec![id.clone()],
            vec![id],
        ]
    );
    cleanup(engine, session).await;
}
