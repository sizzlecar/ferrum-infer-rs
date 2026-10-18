//! Real CPU embedding/head execution through the public token protocol.
//! This fixture exercises native resource ownership, not an attention KV model.
use super::*;
use ferrum_execution::token_execution::*;

async fn admit_tokens<R: DeviceRuntime>(
    fixture: &Fixture<R>,
    request: &RequestId,
    tokens: &[TokenId],
) -> ExecutionFrontier {
    let input = ExecutionAdmission {
        context: ExecutorPrefillAdmission::for_diagnostic(request, tokens, 16),
        adapter: None,
    };
    loop {
        match fixture.executor.admit_tokens(input).await.unwrap() {
            TokenAdmissionOutcome::Admitted(frontier) => return frontier,
            TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                assert!(matches!(
                    fixture.executor.maintain_prefill_backing(request).unwrap(),
                    ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                ));
            }
            other => panic!("unexpected tiny admission result: {other:?}"),
        }
    }
}

async fn execute_one<R: DeviceRuntime>(
    fixture: &Fixture<R>,
    frontier: &ExecutionFrontier,
    work: TokenWork,
) -> ExecutionCompletion {
    let policy = LogitsReturnPolicy::FullLogits;
    let input = ExecutionWork::new(frontier, work, &policy).unwrap();
    let ExecutionBatchOutcome::Completed(batch) =
        fixture.executor.execute_tokens(&[input]).await.unwrap()
    else {
        panic!("tiny native token work did not complete")
    };
    let mut completed = batch.into_completions();
    assert_eq!(completed.len(), 1);
    completed.pop().unwrap()
}

#[tokio::test]
async fn native_tokens_cpu_partial_final_decode_and_terminal_receipt() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let tokens = [TokenId::new(0), TokenId::new(1)];
    let initial = admit_tokens(&fixture, &request, &tokens).await;
    let repeated = admit_tokens(&fixture, &request, &tokens).await;
    assert_eq!(
        initial, repeated,
        "repeat probe must retain the same session"
    );
    assert_eq!(fixture.executor.sequences.lock().total_len(), 1);
    let first = execute_one(
        &fixture,
        &initial,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap()),
    )
    .await;
    assert!(matches!(first.product(), ExecutionProduct::Intermediate));
    assert_eq!(first.next_frontier().consumed_tokens(), 1);
    assert!(
        fixture
            .executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_diagnostic(&request, &tokens, 16),
                adapter: None,
            })
            .await
            .is_err(),
        "initial admission cannot be replayed after a committed partial prefill"
    );
    let final_prefill = execute_one(
        &fixture,
        first.next_frontier(),
        TokenWork::Prefill(PrefillChunk::new(1, 1, 2).unwrap()),
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        final_prefill.product()
    else {
        panic!("missing final logits")
    };
    assert_eq!(logits, &[3.0, 2.0, 1.0, -4.0]);
    let decoded = execute_one(
        &fixture,
        final_prefill.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(3),
            position: 2,
        },
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) = decoded.product()
    else {
        panic!("missing decode logits")
    };
    assert_eq!(logits, &[1.0, 1.0009765625, -1.0, -2.0]);
    assert_eq!(decoded.next_frontier().consumed_tokens(), 3);
    let ExecutionCloseDisposition::Ended(ended) = fixture
        .executor
        .finish_tokens(
            initial.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: 2,
                output_tokens: 1,
            },
        )
        .await
        .unwrap()
        .disposition
    else {
        panic!("quiescent real session must end")
    };
    assert_eq!(ended.sequence(), initial.sequence());
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert!(fixture.executor.sequences.lock().token_index.is_empty());
    assert_eq!(fixture.executor.reaper.retained_count(), 0);
    assert_eq!(fixture.executor.lane.in_flight_count(), 0);
}

#[tokio::test]
async fn native_tokens_cpu_foreign_stale_and_admission_mutation_rejected() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let other = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let tokens = [TokenId::new(0), TokenId::new(1)];
    let initial = admit_tokens(&fixture, &request, &tokens).await;
    let mut changed = ExecutorPrefillAdmission::for_diagnostic(&request, &tokens, 16);
    changed.request_origin = ExecutorRequestOrigin::Product;
    assert!(fixture
        .executor
        .admit_tokens(ExecutionAdmission {
            context: changed,
            adapter: None
        })
        .await
        .is_err());
    let policy = LogitsReturnPolicy::FullLogits;
    let work = TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap());
    assert!(other
        .executor
        .execute_tokens(&[ExecutionWork::new(&initial, work, &policy).unwrap()])
        .await
        .is_err());
    let first = execute_one(&fixture, &initial, work).await;
    let before = fixture
        .executor
        .metrics
        .prefill_operations
        .load(Ordering::Relaxed);
    assert!(fixture
        .executor
        .execute_tokens(&[ExecutionWork::new(&initial, work, &policy).unwrap()])
        .await
        .is_err());
    assert_eq!(
        fixture
            .executor
            .metrics
            .prefill_operations
            .load(Ordering::Relaxed),
        before
    );
    let sequence = fixture
        .executor
        .sequences
        .lock()
        .token_index
        .get(initial.sequence())
        .unwrap()
        .upgrade()
        .unwrap();
    assert_eq!(
        sequence.prefill_tokens_processed.load(Ordering::Acquire),
        first.next_frontier().consumed_tokens()
    );
    assert!(matches!(
        fixture
            .executor
            .cancel_tokens(initial.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert!(sequence.settlement.lock().terminal.is_some());
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
}

#[tokio::test]
async fn native_tokens_cpu_mixed_preserves_original_order_and_frontiers() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let first_request = RequestId::new();
    let second_request = RequestId::new();
    let first = admit_tokens(&fixture, &first_request, &[TokenId::new(0)]).await;
    let active = execute_one(
        &fixture,
        &first,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    let second = admit_tokens(&fixture, &second_request, &[TokenId::new(1)]).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [
        ExecutionWork::new(
            &second,
            TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
            &policy,
        )
        .unwrap(),
        ExecutionWork::new(
            active.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(3),
                position: 1,
            },
            &policy,
        )
        .unwrap(),
    ];
    let ExecutionBatchOutcome::Completed(batch) =
        fixture.executor.execute_tokens(&inputs).await.unwrap()
    else {
        panic!("mixed native wave did not complete")
    };
    assert_eq!(batch.completions().len(), 2);
    assert_eq!(
        batch.completions()[0].next_frontier().sequence(),
        second.sequence()
    );
    assert_eq!(
        batch.completions()[1].next_frontier().sequence(),
        first.sequence()
    );
    for (completion, expected) in batch
        .completions()
        .iter()
        .zip([[3.0, 2.0, 1.0, -4.0], [1.0, 1.0009765625, -1.0, -2.0]])
    {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
            completion.product()
        else {
            panic!("missing mixed logits")
        };
        assert_eq!(logits, &expected);
    }
    let _ = fixture.executor.cancel_tokens(first.sequence()).unwrap();
    let _ = fixture.executor.cancel_tokens(second.sequence()).unwrap();
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
}

#[tokio::test]
async fn native_tokens_cpu_bad_middle_product_preserves_both_successful_peers() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let requests = [RequestId::new(), RequestId::new(), RequestId::new()];
    let mut frontiers = Vec::with_capacity(requests.len());
    for request in &requests {
        frontiers.push(admit_tokens(&fixture, request, &[TokenId::new(1)]).await);
    }
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
                &policy,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let result = fixture
        .executor
        .execute_native_tokens_inner(
            &inputs,
            Some(|outputs| {
                assert_eq!(outputs.len(), 3);
                outputs[1] = ExecutorSamplingOutput::FullLogits(Vec::new());
            }),
        )
        .await
        .unwrap();
    let ExecutionBatchOutcome::ContractViolation(violation) = result else {
        panic!("bad product must preserve a contract violation")
    };
    assert_eq!(violation.completed.len(), 2);
    assert_eq!(violation.remaining.len(), 1);
    assert_eq!(violation.rejected_completions.len(), 1);
    assert!(violation.unassigned_products.is_empty());
    assert_eq!(violation.error.to_string(), violation.causes[0].to_string());
    let failed = &violation.remaining[0];
    assert_eq!(failed.input_index(), violation.rejected_completions[0].0);
    assert!(matches!(
        failed.progress(),
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Successful,
            ..
        }
    ));
    assert!(matches!(
        failed.continuation(),
        ContinuationDisposition::Ended(_)
    ));
    assert_eq!(fixture.executor.sequences.lock().total_len(), 2);
    for completion in &violation.completed {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
            completion.product()
        else {
            panic!("peer product lost")
        };
        assert_eq!(logits, &[3.0, 2.0, 1.0, -4.0]);
        let next = execute_one(
            &fixture,
            completion.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(3),
                position: 1,
            },
        )
        .await;
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) = next.product()
        else {
            panic!("peer continuation lost")
        };
        assert_eq!(logits, &[1.0, 1.0009765625, -1.0, -2.0]);
        let close = fixture
            .executor
            .cancel_tokens(completion.next_frontier().sequence())
            .unwrap();
        assert!(close.error.is_none());
        assert!(matches!(
            close.disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
}

#[tokio::test]
async fn native_tokens_cpu_extra_output_row_retains_unassigned_products_and_closes_every_owner() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let frontier = admit_tokens(&fixture, &request, &[TokenId::new(1)]).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let input = ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        &policy,
    )
    .unwrap();
    let result = fixture
        .executor
        .execute_native_tokens_inner(
            &[input],
            Some(|outputs| {
                outputs.push(ExecutorSamplingOutput::FullLogits(vec![91.0, 92.0]));
            }),
        )
        .await
        .unwrap();
    let ExecutionBatchOutcome::ContractViolation(violation) = result else {
        panic!("extra row is a contract violation")
    };
    assert_eq!(violation.unassigned_products.len(), 2);
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(original)) =
        &violation.unassigned_products[0]
    else {
        panic!("original row lost")
    };
    assert_eq!(original, &[3.0, 2.0, 1.0, -4.0]);
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(extra)) =
        &violation.unassigned_products[1]
    else {
        panic!("extra row lost")
    };
    assert_eq!(extra, &[91.0, 92.0]);
    assert_eq!(violation.remaining.len(), 1);
    assert!(matches!(
        violation.remaining[0].continuation(),
        ContinuationDisposition::Ended(_)
    ));
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert!(fixture.executor.sequences.lock().token_index.is_empty());
}

#[tokio::test]
async fn native_tokens_cpu_cancel_queued_wave_waits_for_actual_terminal_receipt() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let frontier = admit_tokens(&fixture, &request, &[TokenId::new(0)]).await;
    let active = execute_one(
        &fixture,
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    let sequence = fixture
        .executor
        .sequences
        .lock()
        .token_index
        .get(frontier.sequence())
        .unwrap()
        .upgrade()
        .unwrap();
    let (started, started_rx) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let mut blocker = Box::pin(fixture.executor.completion_worker.execute(
        VNextCompletionTaskKind::PostSubmitDrain,
        move || {
            let _ = started.send(());
            let _ = released.recv();
        },
    ));
    assert!(futures::poll!(blocker.as_mut()).is_pending());
    started_rx.await.unwrap();
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [ExecutionWork::new(
        active.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(1),
            position: 1,
        },
        &policy,
    )
    .unwrap()];
    let mut execution = Box::pin(fixture.executor.execute_tokens(&inputs));
    assert!(futures::poll!(execution.as_mut()).is_pending());
    let close = fixture.executor.cancel_tokens(frontier.sequence()).unwrap();
    assert!(close.error.is_none());
    let ExecutionCloseDisposition::Held(held) = close.disposition else {
        panic!("live native wave cannot be declared ended")
    };
    assert!(sequence.settlement.lock().terminal.is_none());
    let mut terminal = Box::pin(held.wait());
    assert!(futures::poll!(terminal.as_mut()).is_pending());
    drop(execution);
    release.send(()).unwrap();
    blocker.await.unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(3), terminal)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(terminal.ended.sequence(), frontier.sequence());
    assert!(terminal.error.is_none());
    assert!(sequence.settlement.lock().terminal.is_some());
    fixture
        .executor
        .completion_worker
        .execute(VNextCompletionTaskKind::PostSubmitDrain, || ())
        .await
        .unwrap();
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert!(fixture.executor.sequences.lock().token_index.is_empty());
    assert_eq!(fixture.executor.reaper.retained_count(), 0);
    assert_eq!(fixture.executor.lane.in_flight_count(), 0);
}

#[tokio::test]
async fn native_tokens_cpu_cancel_one_shared_participant_preserves_healthy_peer() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let requests = [RequestId::new(), RequestId::new()];
    let mut active = Vec::new();
    for request in &requests {
        let frontier = admit_tokens(&fixture, request, &[TokenId::new(0)]).await;
        active.push(
            execute_one(
                &fixture,
                &frontier,
                TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
            )
            .await,
        );
    }
    let (started, started_rx) = tokio::sync::oneshot::channel();
    let (release, released) = std::sync::mpsc::channel();
    let mut blocker = Box::pin(fixture.executor.completion_worker.execute(
        VNextCompletionTaskKind::PostSubmitDrain,
        move || {
            let _ = started.send(());
            let _ = released.recv();
        },
    ));
    assert!(futures::poll!(blocker.as_mut()).is_pending());
    started_rx.await.unwrap();
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = active
        .iter()
        .map(|completion| {
            ExecutionWork::new(
                completion.next_frontier(),
                TokenWork::Decode {
                    token: TokenId::new(1),
                    position: 1,
                },
                &policy,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut execution = Box::pin(fixture.executor.execute_tokens(&inputs));
    assert!(futures::poll!(execution.as_mut()).is_pending());
    let cancelled = inputs[0].sequence();
    let close = fixture.executor.cancel_tokens(cancelled).unwrap();
    assert!(close.error.is_none());
    let ExecutionCloseDisposition::Held(held) = close.disposition else {
        panic!("the queued shared wave still owns the cancelled participant")
    };
    let mut terminal = Box::pin(held.wait());
    assert!(futures::poll!(terminal.as_mut()).is_pending());
    release.send(()).unwrap();
    blocker.await.unwrap();
    let ExecutionBatchOutcome::Interrupted(interrupted) = execution.await.unwrap() else {
        panic!("one cancelled participant must not erase its healthy peer")
    };
    assert_eq!(interrupted.completed().len(), 1);
    assert_eq!(interrupted.remaining().len(), 1);
    let peer = &interrupted.completed()[0];
    assert_eq!(peer.input_index(), 1);
    assert_eq!(peer.next_frontier().sequence(), inputs[1].sequence());
    assert_eq!(peer.next_frontier().consumed_tokens(), 2);
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) = peer.product()
    else {
        panic!("healthy shared-wave logits were lost")
    };
    assert_eq!(logits, &[3.0, 2.0, 1.0, -4.0]);
    assert_eq!(interrupted.remaining()[0].input_index(), 0);
    assert!(matches!(
        interrupted.remaining()[0].continuation(),
        ContinuationDisposition::Ended(_)
    ));
    let terminal = tokio::time::timeout(Duration::from_secs(3), terminal)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(terminal.ended.sequence(), cancelled);
    assert!(terminal.error.is_none());
    assert_eq!(fixture.executor.sequences.lock().total_len(), 1);
    let next = execute_one(
        &fixture,
        peer.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(3),
            position: 2,
        },
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) = next.product()
    else {
        panic!("healthy continuation lost its logits")
    };
    assert_eq!(logits, &[1.0, 1.0009765625, -1.0, -2.0]);
    assert_eq!(next.next_frontier().consumed_tokens(), 3);
    assert!(matches!(
        fixture
            .executor
            .cancel_tokens(inputs[1].sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert_eq!(fixture.executor.sequences.lock().total_len(), 0);
    assert!(fixture.executor.sequences.lock().token_index.is_empty());
    assert_eq!(fixture.executor.reaper.retained_count(), 0);
    assert_eq!(fixture.executor.lane.in_flight_count(), 0);
}

#[tokio::test]
async fn native_tokens_cpu_deferral_retry_checks_decode_frontier_and_cancellation() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let request = RequestId::new();
    let initial = admit_tokens(&fixture, &request, &[TokenId::new(0)]).await;
    let active = execute_one(
        &fixture,
        &initial,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    let sequence = fixture
        .executor
        .sequences
        .lock()
        .token_index
        .get(initial.sequence())
        .unwrap()
        .upgrade()
        .unwrap();
    let policy = LogitsReturnPolicy::FullLogits;
    let input = ExecutionWork::new(
        active.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(1),
            position: 1,
        },
        &policy,
    )
    .unwrap();
    {
        let _operation = sequence.operation.lock().await;
        fixture
            .executor
            .restore_token_batch_retryable(
                std::slice::from_ref(&input),
                &[Arc::clone(&sequence)],
                &[0],
                &[None],
            )
            .unwrap();
    }
    let next = execute_one(
        &fixture,
        active.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(1),
            position: 1,
        },
    )
    .await;
    let _operation = sequence.operation.lock().await;
    assert!(
        fixture
            .executor
            .restore_token_batch_retryable(
                std::slice::from_ref(&input),
                &[Arc::clone(&sequence)],
                &[0],
                &[None]
            )
            .is_err(),
        "a live but advanced incarnation cannot reuse its stale frontier"
    );
    let input = ExecutionWork::new(
        next.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(3),
            position: 2,
        },
        &policy,
    )
    .unwrap();
    let closed = fixture.executor.cancel_tokens(initial.sequence()).unwrap();
    assert!(closed.error.is_none());
    let ExecutionCloseDisposition::Ended(ended) = closed.disposition else {
        panic!("quiescent decode cancellation must prove terminal closure")
    };
    assert_eq!(ended.sequence(), initial.sequence());
    assert!(sequence.settlement.lock().terminal.is_some());
    assert!(
        fixture
            .executor
            .restore_token_batch_retryable(
                std::slice::from_ref(&input),
                &[Arc::clone(&sequence)],
                &[0],
                &[None]
            )
            .is_err(),
        "zero-submit does not make a cancelled decode frontier retryable"
    );
}
