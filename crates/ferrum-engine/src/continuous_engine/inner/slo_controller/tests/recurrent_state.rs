//! The controlled executor has real scheduler/owner/output transitions but no
//! legacy recurrent handle, just as VNext stores state in PlanRuntime resources.
//! These are engine capture regressions, not native execution or cost accuracy.
use super::*;

fn memory(fixed: u64, kv: u64, other: u64) -> TypedSequenceStateMemory {
    TypedSequenceStateMemory {
        fixed_bytes_per_sequence: fixed,
        kv_bytes_per_token: kv,
        other_token_scaled_bytes_per_token: other,
    }
}

async fn run_chunk(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    id: &RequestId,
    offset: usize,
    count: usize,
) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    let mut hint = ferrum_interfaces::BatchHint::simple(1);
    hint.max_tokens = count;
    let batch = scheduler.next_batch(hint).await.unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(&batch.requests[0].request.id, id);
    assert_eq!(batch.requests[0].tokens_processed, offset);
    assert_eq!(batch.requests[0].tokens_to_process, Some(count));
    engine.inner.process_batch(&batch).await.unwrap();
}

async fn assert_state(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    id: &RequestId,
    fixed: u64,
    context: u32,
    prefill_offset: Option<u32>,
) {
    assert!(engine.inner.sequences.read()[id].recurrent_state.is_none());
    let before = prefill::UnsubmittedState::capture(engine, executor);
    let captured = prefill::captured(engine, executor).await;
    before.assert_unchanged(engine, executor);
    assert_eq!(captured.snapshot.requests.len(), 1);
    let row = &captured.snapshot.requests[0];
    assert_eq!(&row.key.request_id, id);
    assert_eq!(row.recurrent_state_bytes, fixed);
    assert_eq!(row.context_tokens, context);
    match (&row.phase, prefill_offset) {
        (RequestPhaseView::Prefill(progress), Some(offset)) => {
            assert_eq!(progress.offset, offset);
        }
        (RequestPhaseView::Decode, None) => {}
        other => panic!("unexpected real frontier: {other:?}"),
    }
}

#[tokio::test]
async fn plan_runtime_fixed_state_survives_fresh_partial_final_and_decode_capture() {
    let (engine, scheduler, executor) = fixture().await;
    // Both dimensions are nonzero. KV bytes must never be folded into the
    // recurrent-state identity, and no legacy allocation is used to infer it.
    let fixed = 11_320;
    *executor.typed_sequence_state.lock() = Some(memory(fixed, 128, 0));
    let (id, mut output) = prefill::request(&engine, 4, 3).await;
    prefill::admit(&engine, 1).await;
    assert_state(&engine, &executor, &id, fixed, 0, Some(0)).await;

    run_chunk(&engine, &scheduler, &id, 0, 2).await;
    assert_state(&engine, &executor, &id, fixed, 2, Some(2)).await;
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());

    run_chunk(&engine, &scheduler, &id, 2, 2).await;
    drop(bounded(output.frames.next()).await.unwrap());
    ready(&engine, &id).await;
    assert_state(&engine, &executor, &id, fixed, 4, None).await;
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    cleanup(engine, output).await;
}

#[tokio::test]
async fn typed_kv_only_state_is_not_misclassified_as_unsupported_dynamic_state() {
    let (engine, scheduler, executor) = fixture().await;
    *executor.typed_sequence_state.lock() = Some(memory(0, 4096, 0));
    let (id, output) = prefill::request(&engine, 2, 2).await;
    prefill::admit(&engine, 1).await;
    assert_state(&engine, &executor, &id, 0, 0, Some(0)).await;
    run_chunk(&engine, &scheduler, &id, 0, 1).await;
    assert_state(&engine, &executor, &id, 0, 1, Some(1)).await;
    cleanup(engine, output).await;
}

#[tokio::test]
async fn non_kv_token_scaled_state_remains_unavailable_without_submitting_work() {
    let (engine, _scheduler, executor) = fixture().await;
    *executor.typed_sequence_state.lock() = Some(memory(11_320, 128, 1));
    let (_id, output) = prefill::request(&engine, 4, 3).await;
    prefill::admit(&engine, 1).await;
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    let error = engine
        .inner
        .capture_slo_controller_snapshot(
            &ferrum_interfaces::BatchHint::simple(1),
            ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
        )
        .err()
        .expect("unmodeled non-KV token-scaled state must not become zero");
    assert_eq!(error.reason, "non_kv_token_scaled_state_unsupported");
    before.assert_unchanged(&engine, &executor);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, output).await;
}
