//! INT8 qualification uses the actual paged model and original worker.
//! The head decorator observes real encoding; it supplies no lifecycle facts.

use super::*;
use ferrum_interfaces::kv_dtype::KvInt8;

fn executor() -> (LlmExecutor, Arc<Control>) {
    observed_executor_with_kv::<KvInt8>(true)
}

#[tokio::test]
#[ignore = "requires actual CUDA INT8 completion and original owner retirement"]
async fn qualification_unwrapped_cuda_int8_completion_and_original_owner_close() {
    let (executor, control) = executor();
    let executor =
        qualified_completion_with_executor(executor, true, ModelTokenExecutionMode::ManagedWorker)
            .await;
    assert!(control.worker_thread.lock().is_some());
    close_original(executor, &control).await;
}

#[tokio::test]
#[ignore = "requires actual CUDA INT8 cold allocation and checked retirement"]
async fn qualification_unwrapped_cuda_int8_cold_owner_closes_without_admission() {
    let (executor, control) = executor();
    assert!(executor.token_worker.get().is_none());
    assert!(control.worker_thread.lock().is_none());
    close_original(executor, &control).await;
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 0);
}

#[tokio::test]
#[ignore = "requires actual CUDA INT8 encoded failure and healthy peer continuation"]
async fn qualification_unwrapped_cuda_int8_encoded_failure_keeps_peer_and_closes_original() {
    let (executor, control) = executor();
    encoded_failure_keeps_peer_and_closes_original(executor, control).await;
}

#[tokio::test]
#[ignore = "requires actual CUDA INT8 encoded cancellation and healthy peer continuation"]
async fn qualification_unwrapped_cuda_int8_encoded_cancellation_preserves_peer() {
    let (executor, control) = executor();
    encoded_cancellation_preserves_peer(executor, control).await;
}

#[tokio::test]
#[ignore = "requires actual CUDA INT8 cancelled owner waiter and original worker retirement"]
async fn qualification_unwrapped_cuda_int8_cancelled_close_waiter_joins_original_worker() {
    let (executor, control) = executor();
    cancelled_close_waiter_joins_original_worker(executor, control).await;
}

#[tokio::test]
#[ignore = "requires actual CUDA INT8 page crossing and partial serial batch failure"]
async fn qualification_unwrapped_cuda_int8_serial_page_crossing_keeps_completed_peer() {
    let (executor, control) = executor();
    let prompts: [Vec<TokenId>; 2] = [
        (1..=16).map(TokenId).collect(),
        (31..=46).map(TokenId).collect(),
    ];
    let requests = [RequestId::new(), RequestId::new()];
    let mut frontiers = Vec::new();
    for (request, prompt) in requests.iter().zip(&prompts) {
        let admitted = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_startup(request, prompt, prompt.len() + 6),
                adapter: None,
            })
            .await
            .unwrap();
        let TokenAdmissionOutcome::Admitted(frontier) = admitted else {
            panic!("the actual INT8 pool must fit both declared requests: {admitted:?}");
        };
        frontiers.push(frontier);
    }

    // Two original participants, with partial and final multi-query prefill.
    // Each logical KV reaches exactly one 16-token page before decode crosses
    // into the second page. The independent CPU model receives the same chunks.
    let mut reference = real_model::<CpuBackend>(false);
    let policy = LogitsReturnPolicy::FullLogits;
    for (start, count) in [(0, 13), (13, 3)] {
        let work: Vec<_> = frontiers
            .iter()
            .map(|frontier| {
                ExecutionWork::new(
                    frontier,
                    TokenWork::Prefill(PrefillChunk::new(start, count, 16).unwrap()),
                    &policy,
                )
                .unwrap()
            })
            .collect();
        let outcome = executor.execute_tokens(&work).await.unwrap();
        let ExecutionBatchOutcome::Completed(batch) = outcome else {
            panic!("INT8 serial prefill must complete both actual participants: {outcome:?}");
        };
        let completions = batch.into_completions();
        assert_eq!(completions.len(), 2);
        for (index, completion) in completions.iter().enumerate() {
            assert_eq!(
                completion.next_frontier().sequence(),
                frontiers[index].sequence()
            );
            assert_eq!(completion.next_frontier().consumed_tokens(), start + count);
            let tokens: Vec<_> = prompts[index][start..start + count]
                .iter()
                .map(|token| token.get())
                .collect();
            let expected = reference.prefill_internal(&format!("peer-{index}"), &tokens);
            if start + count == 16 {
                compare_logits(completion, &expected);
            } else {
                assert!(matches!(
                    completion.product(),
                    ExecutionProduct::Intermediate
                ));
            }
        }
        frontiers = completions
            .into_iter()
            .map(|completion| completion.next_frontier().clone())
            .collect();
    }

    // The first row fails after actual CUDA head encoding. Its checked failure
    // must not erase the second row's real completion or grant first-row replay.
    control.fail_head_once.store(true, Ordering::SeqCst);
    let work: Vec<_> = frontiers
        .iter()
        .zip([17, 47])
        .map(|(frontier, token)| {
            ExecutionWork::new(
                frontier,
                TokenWork::Decode {
                    token: TokenId(token),
                    position: 16,
                },
                &policy,
            )
            .unwrap()
        })
        .collect();
    let outcome = executor.execute_tokens(&work).await.unwrap();
    let ExecutionBatchOutcome::Interrupted(interrupted) = outcome else {
        panic!("the failed INT8 row must remain distinct from its completed peer: {outcome:?}");
    };
    let (mut completed, mut remaining, causes) = interrupted.into_parts();
    assert_eq!(completed.len(), 1);
    assert_eq!(remaining.len(), 1);
    assert_eq!(causes.len(), 1);
    assert!(matches!(
        &causes[0],
        ExecutionInterruptionCause::Error(error)
            if error.to_string() == FerrumError::backend(HEAD_ERROR).to_string()
    ));
    let peer = completed.remove(0);
    assert_eq!(peer.next_frontier().sequence(), frontiers[1].sequence());
    assert_eq!(peer.next_frontier().consumed_tokens(), 17);
    compare_logits(&peer, &reference.decode_internal("peer-1", 47, 16));

    let failed = remaining.remove(0);
    assert_eq!(failed.sequence(), frontiers[0].sequence());
    assert!(matches!(
        failed.progress(),
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Failed,
            ..
        }
    ));
    let (_, _, _, continuation, _) = failed.into_parts();
    let ended = match continuation {
        ContinuationDisposition::Ended(ended) => ended,
        ContinuationDisposition::Held(held) => {
            let terminal = tokio::time::timeout(WAIT, held.wait())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                terminal.error.unwrap().to_string(),
                FerrumError::backend(HEAD_ERROR).to_string()
            );
            terminal.ended
        }
        _ => panic!("an encoded INT8 failure cannot become retryable work"),
    };
    assert_eq!(ended.sequence(), frontiers[0].sequence());
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 6);
    assert_eq!(
        executor.token_sequences.lock().live[frontiers[1].sequence()].frontier,
        *peer.next_frontier()
    );
    let capacity = executor
        .lock_model()
        .unwrap()
        .kv_slot_capacity_snapshot()
        .unwrap();
    assert_eq!(capacity.block_size, 16);
    assert_eq!(capacity.total_blocks, 4);
    assert_eq!(
        capacity.free_blocks, 2,
        "only the failed request's two pages retire"
    );

    let decoded = execute(
        &executor,
        peer.next_frontier(),
        TokenWork::Decode {
            token: TokenId(48),
            position: 17,
        },
    )
    .await;
    compare_logits(&decoded, &reference.decode_internal("peer-1", 48, 17));
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 7);
    terminal(
        executor
            .finish_tokens(
                decoded.next_frontier().sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 16,
                    output_tokens: 3,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    assert_all_slots_free(&executor, true);
    reference.release("peer-0");
    reference.release("peer-1");
    reference.close().unwrap();
    close_original(executor, &control).await;
}
