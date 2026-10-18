//! Completion errors preserve the original model owner and both client results.
//! Shared fixture executes an actual CPU Llama and checks its full prefill logits.

use super::*;

#[derive(Clone, Copy)]
enum CompletionFault {
    Evidence,
    Confirmation,
    Scheduler,
}

async fn completion_failure_closes_known_owners(fault: CompletionFault) {
    let mut fixture = fixture(false, false).await;
    let (frontier, token, prompt) = {
        let states = &fixture.engine.inner.sequences;
        let state_handle = states.get(&fixture.request_id).unwrap();
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        (
            state.execution_frontier.clone().unwrap(),
            state.generated_tokens[0],
            state.input_tokens.clone(),
        )
    };
    match fault {
        CompletionFault::Evidence => {
            // Real timing validation: two commit observations for one token.
            fixture
                .engine
                .inner
                .sequences
                .get(&fixture.request_id)
                .unwrap()
                .lock()
                .as_mut()
                .unwrap()
                .record_generated_token_commit();
        }
        CompletionFault::Confirmation => {
            fixture
                .executor
                .fail_confirmation
                .store(true, Ordering::SeqCst);
        }
        CompletionFault::Scheduler => {
            assert!(fixture
                .engine
                .inner
                .scheduler
                .cancel_request(fixture.request_id.clone())
                .unwrap());
        }
    }
    let error = fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap_err();
    match fault {
        CompletionFault::Evidence => assert!(matches!(&error,
            FerrumError::Internal { message } if message.contains("invalid engine token timing evidence"))),
        CompletionFault::Confirmation => assert!(matches!(&error,
            FerrumError::Backend { message } if message == "injected terminal confirmation failure")),
        CompletionFault::Scheduler => assert!(matches!(&error, FerrumError::Scheduler { .. })),
    }
    assert!(!fixture
        .engine
        .inner
        .sequences
        .get(&fixture.request_id)
        .is_some());
    assert_eq!(
        fixture.engine.inner.scheduler.trace_snapshot().active_len,
        0
    );
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        error.to_string()
    );
    let mut failures = 0;
    while let Ok(chunk) = fixture.stream.try_recv() {
        match chunk {
            Ok(chunk) => assert!(
                chunk.chunk().finish_reason.is_none(),
                "no success terminal after an error"
            ),
            Err(actual) => {
                assert_eq!(actual.to_string(), error.to_string());
                failures += 1;
            }
        }
    }
    assert_eq!(failures, 1);
    let confirmation_failed = matches!(fault, CompletionFault::Confirmation);
    assert_eq!(
        fixture.executor.finishes.load(Ordering::SeqCst),
        usize::from(confirmation_failed)
    );
    assert_eq!(
        fixture.executor.cancels.load(Ordering::SeqCst),
        usize::from(!confirmation_failed)
    );

    if confirmation_failed {
        // Failed observation has no release authority: the exact original
        // executor, KV and request slot must remain owned. Continuing only in
        // this CPU fixture proves that no generic cleanup freed the model.
        assert!(fixture.identity.observed_end().is_none());
        assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 1);
        assert_eq!(slot_events(&fixture, ResourceAction::Release), 0);
        assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
        let policy = LogitsReturnPolicy::FullLogits;
        let work = ExecutionWork::new(
            &frontier,
            TokenWork::Decode {
                token,
                position: frontier.consumed_tokens(),
            },
            &policy,
        )
        .unwrap();
        let ExecutionBatchOutcome::Completed(batch) = fixture
            .executor
            .inner
            .execute_tokens(&[work])
            .await
            .unwrap()
        else {
            panic!("original CPU owner must remain usable after failed confirmation")
        };
        let result = batch.into_completions().pop().unwrap();
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) =
            result.product()
        else {
            panic!("full numerical result required")
        };
        let cfg = TinyLlamaConfig::default();
        let mut reference = tiny_llama_model(&cfg);
        reference.prefill(
            "reference",
            &prompt.iter().map(|token| token.get()).collect::<Vec<_>>(),
        );
        let expected = reference.decode("reference", token.get(), prompt.len() as u32);
        assert_eq!(actual.len(), cfg.vocab_size);
        assert_eq!(expected.len(), cfg.vocab_size);
        for (&actual, expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite() && expected.is_finite());
            assert!((actual - expected).abs() <= 1e-5);
        }
        {
            let observation = fixture.model.lock();
            assert!(observation.releases.is_empty());
            for layer in observation.model.kv_caches.values().next().unwrap() {
                assert_eq!(layer.len, fixture.prompt_len + 1);
            }
        }
        let closed = fixture
            .executor
            .inner
            .cancel_tokens(&fixture.identity)
            .unwrap();
        assert!(matches!(
            closed.disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
        fixture.engine.inner.drain_execution_closes().unwrap();
        assert_eq!(
            fixture.model.lock().releases,
            vec![vec![fixture.prompt_len + 1; fixture.layers]]
        );
    } else {
        assert_eq!(
            fixture.model.lock().releases,
            vec![vec![fixture.prompt_len; fixture.layers]]
        );
    }
    assert!(fixture.identity.observed_end().is_some());
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 1);
    assert_eq!(slot_events(&fixture, ResourceAction::RequestClose), 1);
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    fixture
        .engine
        .inner
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap();
    fixture.engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        fixture.executor.finishes.load(Ordering::SeqCst),
        usize::from(confirmation_failed)
    );
    assert_eq!(
        fixture.executor.cancels.load(Ordering::SeqCst),
        usize::from(!confirmation_failed)
    );
    assert_eq!(fixture.model.lock().releases.len(), 1);
    assert_eq!(slot_events(&fixture, ResourceAction::Release), 1);
}

#[tokio::test]
async fn completion_evidence_error_still_releases_resources_and_notifies_clients() {
    completion_failure_closes_known_owners(CompletionFault::Evidence).await;
}

#[tokio::test]
async fn completion_confirmation_error_preserves_lower_owner_and_closes_clients() {
    completion_failure_closes_known_owners(CompletionFault::Confirmation).await;
}

#[tokio::test]
async fn completion_scheduler_error_still_closes_slot_and_notifies_clients() {
    completion_failure_closes_known_owners(CompletionFault::Scheduler).await;
}
