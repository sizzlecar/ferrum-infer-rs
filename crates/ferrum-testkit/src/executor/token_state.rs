//! Host-only bookkeeping for the two synthetic-logit executors. This owns no
//! device/KV state and never adapts the retired tensor execution interface.
use ferrum_execution::{model_executor::ExecutorSamplingOutput, token_execution::*};
use ferrum_types::{FerrumError, RequestId, Result, TokenId};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

struct Entry {
    request: RequestId,
    prompt: Arc<[TokenId]>,
    maximum: usize,
    product_prompt: usize,
    replayed: usize,
    frontier: ExecutionFrontier,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigurableModelExecutor, MockModelExecutor};
    use ferrum_execution::{
        model_executor::{ExecutorPrefillAdmission, LogitsReturnPolicy, PrefillChunk},
        ModelExecutor,
    };

    fn close(executor: &impl ModelExecutor, sequence: &ExecutorSequence) {
        let outcome = executor.cancel_tokens(sequence).unwrap();
        assert!(outcome.error.is_none());
        let ExecutionCloseDisposition::Ended(ended) = outcome.disposition else {
            panic!("synchronous fixture must really end");
        };
        assert_eq!(ended.sequence(), sequence);
    }

    async fn admit(executor: &impl ModelExecutor, request: &RequestId) -> ExecutionFrontier {
        let prompt = [TokenId::new(3), TokenId::new(4), TokenId::new(5)];
        let TokenAdmissionOutcome::Admitted(frontier) = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_diagnostic(request, &prompt, 8),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("fixture admission");
        };
        frontier
    }

    async fn run(
        executor: &impl ModelExecutor,
        frontier: &ExecutionFrontier,
        work: TokenWork,
    ) -> ExecutionCompletion {
        let policy = LogitsReturnPolicy::FullLogits;
        let input = [ExecutionWork::new(frontier, work, &policy).unwrap()];
        let ExecutionBatchOutcome::Completed(output) =
            executor.execute_tokens(&input).await.unwrap()
        else {
            panic!("fixture completion");
        };
        output.into_completions().remove(0)
    }

    fn selected(completion: &ExecutionCompletion) -> usize {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
            completion.product()
        else {
            panic!("full logits");
        };
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    }

    #[tokio::test]
    async fn host_shutdown_counter_observer_does_not_retain_mock_or_configurable_tokens() {
        use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorShutdownPhase};
        let mock = MockModelExecutor::instant(64);
        let observation = mock.observer();
        let frontier = admit(&mock, &RequestId::new()).await;
        let prefill = run(
            &mock,
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        let decode = run(
            &mock,
            prefill.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(42),
                position: 3,
            },
        )
        .await;
        assert_eq!(selected(&decode), 42);
        let owner = ExecutorOwner::new(Box::new(mock));
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            owner.begin_shutdown().wait(),
        )
        .await
        .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        assert!(frontier.sequence().observed_end().is_some());
        assert_eq!(observation.prefill_count(), 1);
        assert_eq!(observation.decode_count(), 1);

        let scripted = ConfigurableModelExecutor::with_token_sequence(64, vec![7, 9]);
        let observation = scripted.observer();
        let frontier = admit(&scripted, &RequestId::new()).await;
        let prefill = run(
            &scripted,
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        let decode = run(
            &scripted,
            prefill.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(7),
                position: 3,
            },
        )
        .await;
        assert_eq!(selected(&decode), 7);
        let owner = ExecutorOwner::new(Box::new(scripted));
        let report = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            owner.begin_shutdown().wait(),
        )
        .await
        .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        assert!(frontier.sequence().observed_end().is_some());
        assert_eq!(observation.decode_count(), 1);
    }

    #[tokio::test]
    async fn host_shutdown_waits_for_actual_borrowed_work_scope_after_waiter_cancel() {
        use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorShutdownPhase};
        use std::future::Future;
        use std::task::Poll;
        use std::time::Duration;
        let executor = MockModelExecutor::new(64, Duration::from_secs(60), Duration::ZERO);
        let observation = executor.observer();
        let frontier = admit(&executor, &RequestId::new()).await;
        let owner = ExecutorOwner::new(Box::new(executor));
        let scope = owner.scope().unwrap();
        let work = [ExecutionWork::new(
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            &LogitsReturnPolicy::FullLogits,
        )
        .unwrap()];
        let mut running = Box::pin(scope.execute_tokens(&work));
        std::future::poll_fn(|cx| {
            assert!(running.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        let closing = owner.begin_shutdown();
        drop(owner);
        assert_eq!(closing.snapshot().phase(), ExecutorShutdownPhase::Draining);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), closing.wait())
                .await
                .is_err()
        );
        assert!(frontier.sequence().observed_end().is_none());
        drop(running);
        drop(scope);
        let report = tokio::time::timeout(Duration::from_secs(5), closing.wait())
            .await
            .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        assert!(frontier.sequence().observed_end().is_some());
        assert_eq!(
            observation.prefill_count(),
            0,
            "cancelled latency never executed"
        );
    }

    #[tokio::test]
    async fn token_fixture_chunks_and_script_positions_are_per_incarnation() {
        let executor = ConfigurableModelExecutor::with_token_sequence(64, vec![7, 9]);
        let a = admit(&executor, &RequestId::new()).await;
        let first = run(
            &executor,
            &a,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
        )
        .await;
        assert!(matches!(first.product(), ExecutionProduct::Intermediate));
        let final_chunk = run(
            &executor,
            first.next_frontier(),
            TokenWork::Prefill(PrefillChunk::new(2, 1, 3).unwrap()),
        )
        .await;
        assert_eq!(selected(&final_chunk), 7);
        let decode = run(
            &executor,
            final_chunk.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(7),
                position: 3,
            },
        )
        .await;
        assert_eq!(selected(&decode), 7);
        let decode = run(
            &executor,
            decode.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(7),
                position: 4,
            },
        )
        .await;
        assert_eq!(selected(&decode), 9);
        let b = admit(&executor, &RequestId::new()).await;
        let full = run(
            &executor,
            &b,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        assert_eq!(selected(&full), 7);
        let next = run(
            &executor,
            full.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(7),
                position: 3,
            },
        )
        .await;
        assert_eq!(
            selected(&next),
            7,
            "one request must not advance another's script"
        );
        close(&executor, a.sequence());
        close(&executor, b.sequence());
    }

    #[tokio::test]
    async fn token_fixture_rejects_foreign_stale_and_mixed_work_without_advancing_peers() {
        let executor = MockModelExecutor::instant(64);
        let id = RequestId::new();
        let a = admit(&executor, &id).await;
        let a_done = run(
            &executor,
            &a,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        let b = admit(&executor, &RequestId::new()).await;
        let policy = LogitsReturnPolicy::FullLogits;
        let mixed = [
            ExecutionWork::new(
                a_done.next_frontier(),
                TokenWork::Decode {
                    token: TokenId::new(42),
                    position: 3,
                },
                &policy,
            )
            .unwrap(),
            ExecutionWork::new(
                &b,
                TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
                &policy,
            )
            .unwrap(),
        ];
        assert!(matches!(
            executor.execute_tokens(&mixed).await.unwrap(),
            ExecutionBatchOutcome::NotSubmitted(_)
        ));
        let foreign_executor = MockModelExecutor::instant(64);
        let foreign = admit(&foreign_executor, &RequestId::new()).await;
        let bad = [ExecutionWork::new(
            &foreign,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            &policy,
        )
        .unwrap()];
        assert!(executor.execute_tokens(&bad).await.is_err());
        close(&executor, a.sequence());
        let replacement = admit(&executor, &id).await;
        assert_ne!(replacement.sequence(), a.sequence());
        close(&executor, a.sequence());
        assert!(replacement.sequence().observed_end().is_none());
        let stale = [ExecutionWork::new(
            &a,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            &policy,
        )
        .unwrap()];
        assert!(executor.execute_tokens(&stale).await.is_err());
        assert_eq!(
            run(
                &executor,
                &replacement,
                TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap())
            )
            .await
            .next_frontier()
            .consumed_tokens(),
            3
        );
        assert_eq!(
            run(
                &executor,
                &b,
                TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap())
            )
            .await
            .next_frontier()
            .consumed_tokens(),
            3
        );
        close(&executor, replacement.sequence());
        close(&executor, b.sequence());
        close(&foreign_executor, foreign.sequence());
    }
}

#[derive(Default)]
pub(crate) struct TokenState {
    issuer: ExecutorSequenceIssuer,
    entries: Mutex<HashMap<ExecutorSequence, Entry>>,
}

impl TokenState {
    pub(crate) fn shutdown(&mut self) -> Result<()> {
        let sequences = self.entries.get_mut().keys().cloned().collect::<Vec<_>>();
        for sequence in sequences {
            let outcome = self.close(&sequence, ExecutionEndReason::Cancelled)?;
            if let Some(error) = outcome.error {
                return Err(error);
            }
            if !matches!(outcome.disposition, ExecutionCloseDisposition::Ended(_)) {
                return Err(FerrumError::internal("synthetic token owner remains live"));
            }
        }
        Ok(())
    }

    pub(crate) fn admit(
        &self,
        input: ExecutionAdmission<'_>,
        max_context: usize,
        vocab: usize,
    ) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        if input.adapter.is_some() {
            return Err(FerrumError::unsupported(
                "synthetic executor has no adapter implementation",
            ));
        }
        let context = input.context;
        if vocab == 0
            || context.maximum_sequence_tokens > max_context
            || context
                .input_tokens
                .iter()
                .any(|token| token.get() as usize >= vocab)
        {
            return Err(FerrumError::request_validation(
                "invalid synthetic executor admission extent/vocabulary",
            ));
        }
        let mut entries = self.entries.lock();
        if let Some(entry) = entries
            .values()
            .find(|entry| &entry.request == context.request_id)
        {
            if entry.frontier.consumed_tokens() == 0
                && entry.prompt.as_ref() == context.input_tokens
                && entry.maximum == context.maximum_sequence_tokens
                && entry.product_prompt == context.product_prompt_tokens
                && entry.replayed == context.replayed_output_tokens
            {
                return Ok(TokenAdmissionOutcome::Admitted(entry.frontier.clone()));
            }
            return Err(FerrumError::request_validation(
                "request already has a different live admission",
            ));
        }
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        entries.insert(
            sequence,
            Entry {
                request: context.request_id.clone(),
                prompt: context.input_tokens.into(),
                maximum: context.maximum_sequence_tokens,
                product_prompt: context.product_prompt_tokens,
                replayed: context.replayed_output_tokens,
                frontier: frontier.clone(),
            },
        );
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }

    /// The callback computes synthetic logits without side effects. All products
    /// are checked before any frontier changes; a rejected shape never executes.
    pub(crate) fn execute(
        &self,
        work: &[ExecutionWork<'_>],
        vocab: usize,
        mut logits: impl FnMut(TokenWork, usize) -> Vec<f32>,
    ) -> Result<ExecutionBatchOutcome> {
        let batch = self.issuer.batch(work, vocab)?;
        let mut entries = self.entries.lock();
        for input in work {
            let entry = entries
                .get(input.sequence())
                .ok_or_else(|| FerrumError::request_validation("unknown synthetic incarnation"))?;
            if &entry.frontier != input.expected_frontier()
                || input.work().range()?.end() > entry.maximum
            {
                return Err(FerrumError::request_validation(
                    "stale or out-of-bounds synthetic frontier",
                ));
            }
            match input.work() {
                TokenWork::Prefill(chunk) if chunk.total_prompt_tokens() == entry.prompt.len() => {}
                TokenWork::Decode { position, .. } if position >= entry.prompt.len() => {}
                _ => {
                    return Err(FerrumError::request_validation(
                        "work does not match admitted prompt",
                    ))
                }
            }
        }
        let prefill = matches!(work[0].work(), TokenWork::Prefill(_));
        if work
            .iter()
            .any(|input| matches!(input.work(), TokenWork::Prefill(_)) != prefill)
        {
            return Ok(batch.not_submitted(NoSubmissionReason::UnsupportedShape));
        }
        let mut completed = Vec::with_capacity(work.len());
        for (index, input) in work.iter().enumerate() {
            let entry = &entries[input.sequence()];
            let executed = match input.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            };
            let product = match input.work() {
                TokenWork::Prefill(chunk) if !chunk.is_final() => ExecutionProduct::Intermediate,
                _ => ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits(
                    input.work(),
                    entry.prompt.len(),
                ))),
            };
            match batch.completion(index, executed, product) {
                Ok(completion) => completed.push(completion),
                Err(error) => {
                    return Ok(
                        batch.not_submitted(NoSubmissionReason::Rejected(error.into_parts().0))
                    )
                }
            }
        }
        for completion in &completed {
            entries
                .get_mut(completion.next_frontier().sequence())
                .expect("validated entry")
                .frontier = completion.next_frontier().clone();
        }
        Ok(batch
            .completed(completed)
            .unwrap_or_else(|error| error.into_outcome()))
    }

    pub(crate) fn close(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.issuer.frontier(sequence, 0)?;
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get(sequence) else {
            return sequence
                .observed_end()
                .map(ExecutionCloseOutcome::ended)
                .ok_or_else(|| {
                    FerrumError::request_validation("unknown synthetic close incarnation")
                });
        };
        if let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        {
            if input_tokens != entry.product_prompt
                || output_tokens < entry.replayed
                || input_tokens
                    .checked_add(output_tokens.saturating_sub(1))
                    .is_none_or(|end| end > entry.maximum)
            {
                return Err(FerrumError::request_validation(
                    "synthetic completion usage does not match admission",
                ));
            }
        }
        entries.remove(sequence);
        Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?))
    }
}
