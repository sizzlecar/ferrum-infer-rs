//! Participant timing uses real protocol identity; these products are controlled,
//! unlike the separate tiny CPU ingress numerical tests.
use super::*;

fn decode_completion(
    scope: &ferrum_execution::token_execution::ExecutionBatch<'_>,
    index: usize,
    vocab: usize,
) -> ExecutionCompletion {
    let mut logits = vec![f32::NEG_INFINITY; vocab];
    logits[8] = 1.0;
    scope
        .completion(
            index,
            ExecutedTokenWork::Decode { position: 2 },
            ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)),
        )
        .unwrap()
}

#[tokio::test]
async fn request_clock_interrupted_retry_preserves_decode_scheduling_until_actual_execution() {
    let (engine, executor, mut inputs) = fixture_with_clock(2, true).await;
    allocator_tests::commit_initial_products(&engine, &executor, &mut inputs).await;
    let selected = inputs.iter().collect::<Vec<_>>();
    let works = inputs
        .iter()
        .map(|input| input.as_work().unwrap())
        .collect::<Vec<_>>();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let completion = decode_completion(&scope, 0, executor.info().vocab_size);
    let retry = scope
        .participant(
            1,
            ExecutionProgress::NotSubmitted {
                stage: NoSubmissionStage::Preparation,
            },
            ContinuationDisposition::Retryable(inputs[1].frontier.clone()),
            0,
        )
        .unwrap();
    let mut pool = ferrum_models::common::paged_pool::BlockAllocator::new(1);
    let allocated = pool.allocate().unwrap();
    let deferred = TokenAdmissionCapacityDeferral::new(
        pool.capacity_domain().clone(),
        1,
        ferrum_execution::model_executor::KvSlotCapacitySnapshot {
            block_size: 16,
            total_blocks: pool.capacity() as usize,
            free_blocks: pool.free_count(),
        },
    )
    .unwrap();
    let outcome = scope
        .interrupted(
            vec![completion],
            vec![retry],
            vec![ExecutorExecutionDeferral::AllocatorCapacity(deferred).into()],
        )
        .unwrap();
    let first = TokenExecutionTiming {
        started_at: Instant::now(),
        completed_at: Instant::now(),
    };
    let mut deferred = Vec::new();
    engine
        .inner
        .consume_token_outcome(&selected, &works, outcome, Some(first), &mut deferred)
        .await
        .unwrap();
    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred[0].input_indices, vec![1]);
    assert!(executor.closed.lock().is_empty());
    {
        let state = inputs[1].handle.lock();
        let sequence = state.as_ref().unwrap();
        assert_eq!(
            sequence.execution_frontier.as_ref(),
            Some(&inputs[1].frontier)
        );
        assert_eq!(sequence.generated_tokens, vec![TokenId(7)]);
        assert!(sequence.closes_pending_decode_scheduling());
    }
    let retry_works = [inputs[1].as_work().unwrap()];
    let retry_scope = executor
        .issuer
        .batch(&retry_works, executor.info().vocab_size)
        .unwrap();
    let completion = decode_completion(&retry_scope, 0, executor.info().vocab_size);
    let actual = TokenExecutionTiming {
        started_at: Instant::now(),
        completed_at: Instant::now(),
    };
    engine
        .inner
        .consume_token_outcome(
            &[&inputs[1]],
            &retry_works,
            retry_scope.completed(vec![completion]).unwrap(),
            Some(actual),
            &mut Vec::new(),
        )
        .await
        .unwrap();
    for (input, interval) in inputs.iter().zip([first, actual]) {
        let mut state = input.handle.lock();
        let sequence = state.as_mut().unwrap();
        assert_eq!(sequence.generated_tokens, vec![TokenId(7), TokenId(8)]);
        assert_eq!(
            sequence
                .execution_frontier
                .as_ref()
                .unwrap()
                .consumed_tokens(),
            3
        );
        let ingress = sequence.original_request.ingress.unwrap();
        let timing = sequence
            .take_execution_evidence()
            .unwrap()
            .unwrap()
            .engine_token_timing
            .unwrap();
        timing.validate(2).unwrap();
        assert_eq!(
            timing
                .decode_stage_intervals
                .iter()
                .map(|span| span.stage)
                .collect::<Vec<_>>(),
            [
                EngineDecodeStage::DecodeScheduling,
                EngineDecodeStage::DecodeExecution,
                EngineDecodeStage::DecodePostprocess
            ]
        );
        assert_eq!(
            timing.decode_stage_intervals[0].start_nanos_since_request_start,
            timing.token_commit_nanos_since_request_start[0]
        );
        assert_eq!(
            timing.decode_stage_intervals[0].end_nanos_since_request_start,
            ingress.elapsed_at(interval.started_at).as_nanos() as u64
        );
        assert_eq!(
            timing.decode_stage_intervals[1].end_nanos_since_request_start,
            ingress.elapsed_at(interval.completed_at).as_nanos() as u64
        );
    }
    pool.free(&[allocated]);
    engine.shutdown().await.unwrap();
    assert_eq!(executor.closed.lock().len(), 2);
    assert!(engine.inner.sequences.is_empty());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}
