//! The wait task is synthetic; both executor incarnations and their KV are the
//! existing real CPU fixture. A cancelled wake does not retract a late error.
use super::*;
use crate::continuous_engine::token_completion_tests as actual;
use ferrum_execution::model_executor::{ExecutorSamplingOutput, PrefillChunk};
use ferrum_execution::token_execution::{ExecutionProduct, TokenWork};
use ferrum_models::test_support::{tiny_llama_model, TinyLlamaConfig};
use ferrum_models::DecoderOnlyLLM;

#[derive(Clone, Copy, PartialEq)]
enum Replacement {
    None,
    SameHandle,
    NewHandle,
}

#[tokio::test]
async fn execution_readiness_late_failure_preserves_exact_replacement() {
    for replacement in [
        Replacement::None,
        Replacement::SameHandle,
        Replacement::NewHandle,
    ] {
        let mut fixture = actual::fixture(false, false).await;
        let inner = &fixture
            .engine
            .active_inner()
            .expect("engine execution is open");
        let old_handle = inner.sequences.get(&fixture.request_id).unwrap();
        inner
            .scheduler
            .mark_prefill_complete(&fixture.request_id, fixture.prompt_len);
        let wake = inner
            .scheduler
            .defer_for_execution_readiness(std::slice::from_ref(&fixture.request_id))
            .unwrap()
            .into_wake();
        let (result, observed) = oneshot::channel::<std::result::Result<(), String>>();
        let notify = Arc::new(Notify::new());
        inner
            .execution_readiness_waiters
            .track(
                wake.clone(),
                async move { observed.await.expect("test retains the wait result sender") },
                vec![ExecutionReadinessWaitTarget {
                    request: Arc::downgrade(&old_handle),
                    sequence: fixture.identity.clone(),
                }],
                Arc::clone(&notify),
                inner.execution_readiness_waiters.reserve().await.unwrap(),
            )
            .unwrap();
        assert!(wake.cancel());

        let mut current_handle = Arc::clone(&old_handle);
        let mut current_identity = fixture.identity.clone();
        if replacement != Replacement::None {
            let closed = fixture.executor.cancel_tokens(&fixture.identity).unwrap();
            assert!(closed.error.is_none());
            assert!(matches!(
                closed.disposition,
                ExecutionCloseDisposition::Ended(_)
            ));
            assert!(fixture.model.lock().model.kv_caches.is_empty());
            let (prompt, capacity, request) = {
                let state = old_handle.lock();
                let state = state.as_ref().unwrap();
                (
                    state.input_tokens.clone(),
                    state.model_maximum_sequence_tokens(),
                    state.original_request.clone(),
                )
            };
            let TokenAdmissionOutcome::Admitted(start) = fixture
                .executor
                .admit_tokens(ExecutionAdmission {
                    context: ExecutorPrefillAdmission::for_startup(
                        &fixture.request_id,
                        &prompt,
                        capacity,
                    ),
                    adapter: None,
                })
                .await
                .unwrap()
            else {
                panic!("replacement has real CPU admission");
            };
            let policy = LogitsReturnPolicy::FullLogits;
            let work = ExecutionWork::new(
                &start,
                TokenWork::Prefill(PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap()),
                &policy,
            )
            .unwrap();
            let ExecutionBatchOutcome::Completed(batch) =
                fixture.executor.execute_tokens(&[work]).await.unwrap()
            else {
                panic!("replacement has actual CPU forward");
            };
            let completion = batch.into_completions().pop().unwrap();
            let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
                completion.product()
            else {
                panic!("replacement requires full vocabulary oracle");
            };
            let expected = tiny_llama_model(&TinyLlamaConfig::default()).prefill(
                "readiness-reference",
                &prompt.iter().map(|token| token.get()).collect::<Vec<_>>(),
            );
            assert_eq!(logits, &expected);
            current_identity = completion.next_frontier().sequence().clone();
            assert_ne!(current_identity, fixture.identity);
            {
                let mut state = old_handle.lock();
                let state = state.as_mut().unwrap();
                state.execution_frontier = Some(completion.next_frontier().clone());
                state.generated_tokens[0] = TokenId(
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32,
                );
            }
            if replacement == Replacement::NewHandle {
                let state = inner.sequences.take(&fixture.request_id).unwrap();
                assert!(old_handle.lock().is_none());
                assert!(inner
                    .sequences
                    .insert(fixture.request_id.clone(), state)
                    .is_none());
                current_handle = inner.sequences.get(&fixture.request_id).unwrap();
                assert!(!Arc::ptr_eq(&old_handle, &current_handle));
                assert!(inner
                    .scheduler
                    .cancel_request(fixture.request_id.clone())
                    .unwrap());
                inner.scheduler.submit(request).await.unwrap();
                let selected = inner
                    .scheduler
                    .next_batch(ferrum_interfaces::BatchHint::simple(1))
                    .await
                    .unwrap();
                assert_eq!(selected.requests[0].request.id, fixture.request_id);
                inner
                    .scheduler
                    .mark_prefill_complete(&fixture.request_id, fixture.prompt_len);
            } else {
                assert!(Arc::ptr_eq(&old_handle, &current_handle));
            }
        }
        let phase = inner.scheduler.trace_phase(&fixture.request_id);
        result.send(Err("late readiness failure".into())).unwrap();
        tokio::time::timeout(Duration::from_secs(3), notify.notified())
            .await
            .unwrap();
        {
            let failures = inner.execution_readiness_waiters.failures.lock();
            assert_eq!(
                failures.len(),
                1,
                "cancelled wake still queues a real waiter error"
            );
            assert_eq!(failures[0].requests.len(), 1);
            assert!(failures[0].requests[0]
                .request
                .ptr_eq(&Arc::downgrade(&old_handle)));
            assert_eq!(failures[0].requests[0].sequence, fixture.identity);
        }
        inner.complete_execution_readiness_failures().await.unwrap();
        inner.output_tasks.join_current().await;
        inner.execution_closes.join_observers().await;
        inner.drain_execution_closes().unwrap();

        if replacement == Replacement::None {
            assert!(inner.sequences.get(&fixture.request_id).is_none());
            assert!(fixture
                .response
                .try_recv()
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("late readiness failure"));
            assert!(fixture.identity.observed_end().is_some());
            assert_eq!(fixture.model.lock().releases.len(), 1);
        } else {
            assert!(inner.sequences.is_current(&current_handle));
            {
                let state = current_handle.lock();
                let state = state.as_ref().unwrap();
                assert_eq!(
                    state.execution_frontier.as_ref().unwrap().sequence(),
                    &current_identity
                );
                assert_eq!(
                    state.execution_frontier.as_ref().unwrap().consumed_tokens(),
                    fixture.prompt_len
                );
                assert!(state.request_slot.as_ref().unwrap().armed);
                assert!(state.execution_close.is_none());
            }
            assert_eq!(inner.scheduler.trace_phase(&fixture.request_id), phase);
            assert!(matches!(
                fixture.response.try_recv(),
                Err(oneshot::error::TryRecvError::Empty)
            ));
            assert!(current_identity.observed_end().is_none());
            {
                let model = fixture.model.lock();
                assert_eq!(model.releases.len(), 1);
                assert_eq!(model.model.kv_caches.len(), 1);
                assert!(model
                    .model
                    .kv_caches
                    .values()
                    .next()
                    .unwrap()
                    .iter()
                    .all(|layer| layer.len == fixture.prompt_len));
            }
            assert_eq!(actual::slot_events(&fixture, ResourceAction::Release), 0);
            inner
                .complete_request_for_test(&fixture.request_id, FinishReason::Length)
                .await
                .unwrap();
            inner.execution_closes.join_observers().await;
            inner.drain_execution_closes().unwrap();
            assert!(current_identity.observed_end().is_some());
            assert_eq!(fixture.model.lock().releases.len(), 2);
        }
        assert!(fixture.model.lock().model.kv_caches.is_empty());
        assert_eq!(inner.execution_closes.unreclaimed_count(), 0);
        assert_eq!(actual::slot_events(&fixture, ResourceAction::Release), 1);
        assert_eq!(
            actual::slot_events(&fixture, ResourceAction::RequestClose),
            1
        );
        inner
            .execution_readiness_waiters
            .abort_and_join()
            .await
            .unwrap();
    }
}
