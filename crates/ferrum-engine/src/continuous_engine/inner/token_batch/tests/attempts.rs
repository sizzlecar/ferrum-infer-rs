//! Consumer protocol oracles use genuine issued identities and retained Helds.
//! Their injected intervals are not model timing measurements; real CPU
//! invocation boundaries are covered by attempt_observation_tests.

use super::*;
use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;
use ferrum_execution::execution_attempt::*;
use ferrum_execution::token_execution::{ExecutionTerminalOutcome, TerminalModelDisposition};

fn timing() -> Option<TokenExecutionTiming> {
    let now = Instant::now();
    Some(TokenExecutionTiming {
        started_at: now,
        completed_at: now,
    })
}

fn plan_full_prompt(inputs: &mut [TokenBatchInput]) {
    for input in inputs {
        input.work = TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap());
        input.scheduled_tokens = 2;
    }
}

fn protocol_capture(
    inner: &EngineInner,
    inputs: &[&TokenBatchInput],
) -> Option<TokenAttemptCapture> {
    let mut capture = inner.prepare_execution_attempt(inputs)?;
    capture.complete_interval(timing().unwrap());
    Some(capture)
}

#[tokio::test]
async fn attempt_rejects_duplicate_terminal_failure_and_unknown_without_losing_healthy_peer() {
    for case in 0..3 {
        let recorder = Arc::new(AttemptRecorder::default());
        let (engine, executor, mut inputs) =
            fixture_observed(2, false, Some(recorder.clone())).await;
        plan_full_prompt(&mut inputs);
        let selected = inputs.iter().collect::<Vec<_>>();
        let works = inputs
            .iter()
            .map(TokenBatchInput::as_work)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let scope = executor
            .issuer
            .batch(&works, executor.info().vocab_size)
            .unwrap();
        let mut sender = None;
        let outcome = if case == 0 {
            let mut completed = completions(&scope);
            completed.insert(
                0,
                scope
                    .completion(
                        0,
                        ExecutedTokenWork::Prefill {
                            chunk: PrefillChunk::new(0, 1, 2).unwrap(),
                            capacity_probe_count: 1,
                        },
                        ExecutionProduct::Intermediate,
                    )
                    .unwrap(),
            );
            scope.completed(completed).unwrap_err().into_outcome()
        } else {
            let (progress, continuation) = if case == 1 {
                (
                    ExecutionProgress::Terminal {
                        disposition: TerminalModelDisposition::Failed,
                        executed_range: None,
                    },
                    ContinuationDisposition::Ended(
                        executor
                            .issuer
                            .ended(inputs[0].frontier.sequence())
                            .unwrap(),
                    ),
                )
            } else {
                let (continuation, original_sender) =
                    held(&executor.issuer, inputs[0].frontier.sequence());
                sender = Some(original_sender);
                (
                    ExecutionProgress::Unknown {
                        last_known: LastKnownExecutionStage::Submitted,
                        known_executed_range: None,
                    },
                    continuation,
                )
            };
            let remaining = scope.participant(0, progress, continuation, 0).unwrap();
            let healthy = completions(&scope).pop().unwrap();
            scope
                .interrupted(
                    vec![healthy],
                    vec![remaining],
                    vec![FerrumError::backend("injected peer failure").into()],
                )
                .unwrap()
        };
        let (original_inner, original_scope, consume_result) = {
            let inner = engine.active_inner().unwrap();
            let capture = protocol_capture(&inner, &selected);
            let result = inner
                .consume_token_outcome(
                    &selected,
                    &works,
                    outcome,
                    timing(),
                    &mut Vec::new(),
                    capture,
                )
                .await;
            (
                Arc::downgrade(&inner),
                Arc::downgrade(&inner.model_executor),
                result,
            )
        };
        assert_healthy(&engine, &inputs[1]);
        let summaries = recorder.take();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(
            summary.eligibility,
            if case == 0 {
                ExecutionAttemptEligibility::InvalidOutcome
            } else {
                ExecutionAttemptEligibility::Interrupted
            }
        );
        assert!(matches!(
            summary.participants[1].outcome,
            AttemptParticipantOutcome::Completed {
                frontier_committed: true,
                ..
            }
        ));
        assert_eq!(summary.participants[1].requested.start(), 0);
        assert_eq!(summary.participants[1].requested.end(), 2);
        assert!(matches!(
            summary.participants[0].outcome,
            AttemptParticipantOutcome::Unverified | AttemptParticipantOutcome::Interrupted(_)
        ));
        if let Some(sender) = sender {
            assert!(!sender.is_closed());
            sender
                .send(Ok(ExecutionTerminalOutcome::ended(
                    executor
                        .issuer
                        .ended(inputs[0].frontier.sequence())
                        .unwrap(),
                )))
                .unwrap();
        }
        engine
            .active_inner()
            .unwrap()
            .execution_closes
            .join_observers()
            .await;
        let close_result = engine.active_inner().unwrap().drain_execution_closes();
        drop(works);
        let shutdown = engine.driver.shutdown_active().unwrap();
        let report = shutdown.wait().await;
        assert!(!report.retains_unproved_owners());
        assert_eq!(
            report.executor_shutdown().unwrap().snapshot().phase(),
            ExecutorShutdownPhase::Retired
        );
        assert!(original_inner.upgrade().is_none());
        assert!(original_scope.upgrade().is_none());
        let errors = [
            consume_result.err(),
            close_result.err(),
            report.result().err(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        if case != 0 {
            assert!(
                !errors.is_empty(),
                "the original participant failure must remain observable"
            );
            assert!(errors
                .iter()
                .all(|error| error.to_string().contains("injected peer failure")));
        }
        assert_eq!(
            executor
                .closed
                .lock()
                .iter()
                .filter(|sequence| *sequence == inputs[1].frontier.sequence())
                .count(),
            1
        );
        assert_eq!(
            executor
                .closed
                .lock()
                .iter()
                .filter(|sequence| *sequence == inputs[0].frontier.sequence())
                .count(),
            usize::from(case == 0)
        );
    }
}

#[tokio::test]
async fn attempt_completed_and_definitely_not_submitted_keep_distinct_work() {
    let recorder = Arc::new(AttemptRecorder::default());
    let (engine, executor, mut inputs) = fixture_observed(2, false, Some(recorder.clone())).await;
    plan_full_prompt(&mut inputs);
    let selected = inputs.iter().collect::<Vec<_>>();
    let works = inputs
        .iter()
        .map(TokenBatchInput::as_work)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let completed = completions(&scope).remove(0);
    let remaining = scope
        .participant(
            1,
            ExecutionProgress::NotSubmitted {
                stage: NoSubmissionStage::Preparation,
            },
            ContinuationDisposition::Retryable(inputs[1].frontier.clone()),
            0,
        )
        .unwrap();
    let shortfall = TokenAdmissionCapacityDeferral::new(
        ferrum_types::AllocatorCapacityDomain::new(),
        1,
        ferrum_execution::model_executor::KvSlotCapacitySnapshot {
            block_size: 16,
            total_blocks: 2,
            free_blocks: 0,
        },
    )
    .unwrap();
    let outcome = scope
        .interrupted(
            vec![completed],
            vec![remaining],
            vec![ExecutorExecutionDeferral::AllocatorCapacity(shortfall).into()],
        )
        .unwrap();
    let mut deferred = Vec::new();
    {
        let inner = engine.active_inner().unwrap();
        let capture = protocol_capture(&inner, &selected);
        inner
            .consume_token_outcome(&selected, &works, outcome, timing(), &mut deferred, capture)
            .await
            .unwrap();
    }
    assert_healthy(&engine, &inputs[0]);
    assert_eq!(
        inputs[1]
            .handle
            .lock()
            .as_ref()
            .unwrap()
            .execution_frontier
            .as_ref(),
        Some(&inputs[1].frontier)
    );
    assert_eq!(deferred[0].input_indices, [1]);
    let summary = recorder.take().remove(0);
    assert_eq!(
        summary.eligibility,
        ExecutionAttemptEligibility::CompletedWithNotSubmitted
    );
    let AttemptParticipantOutcome::Completed {
        work,
        frontier_committed: true,
    } = summary.participants[0].outcome
    else {
        panic!("missing real committed prefix");
    };
    assert_eq!(work.range().unwrap().end(), 1);
    assert_eq!(summary.participants[0].requested.end(), 2);
    assert!(matches!(
        summary.participants[1].outcome,
        AttemptParticipantOutcome::NotSubmitted {
            stage: NoSubmissionStage::Preparation,
            cause: AttemptNoSubmissionCause::Deferred,
        }
    ));
    drop(works);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn attempt_sampling_failure_keeps_both_completed_model_ranges() {
    let recorder = Arc::new(AttemptRecorder::default());
    let (engine, executor, mut inputs) = fixture_observed(2, false, Some(recorder.clone())).await;
    plan_full_prompt(&mut inputs);
    let (failure_sender, failure_receiver) = tokio::sync::oneshot::channel();
    inputs[0].handle.lock().as_mut().unwrap().response_sender = Some(failure_sender);
    let selected = inputs.iter().collect::<Vec<_>>();
    let works = inputs
        .iter()
        .map(TokenBatchInput::as_work)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let completed = (0..2)
        .map(|index| {
            scope
                .completion(
                    index,
                    ExecutedTokenWork::Prefill {
                        chunk: PrefillChunk::new(0, 2, 2).unwrap(),
                        capacity_probe_count: 0,
                    },
                    ExecutionProduct::Sampling(if index == 0 {
                        ExecutorSamplingOutput::FullLogits(vec![
                            f32::NEG_INFINITY;
                            executor.info().vocab_size
                        ])
                    } else {
                        ExecutorSamplingOutput::FullLogits(vec![0.0; executor.info().vocab_size])
                    }),
                )
                .unwrap()
        })
        .collect();
    {
        let inner = engine.active_inner().unwrap();
        let capture = protocol_capture(&inner, &selected);
        let _ = inner
            .consume_token_outcome(
                &selected,
                &works,
                scope.completed(completed).unwrap(),
                timing(),
                &mut Vec::new(),
                capture,
            )
            .await;
    }
    let sampling_error = tokio::time::timeout(Duration::from_secs(3), failure_receiver)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(sampling_error
        .to_string()
        .contains("engine sampling policies have no finite token"));
    let summary = recorder.take().remove(0);
    assert_eq!(summary.eligibility, ExecutionAttemptEligibility::Completed);
    assert!(summary.participants[0].product_failed);
    assert!(!summary.participants[1].product_failed);
    assert_eq!(
        inputs[1]
            .handle
            .lock()
            .as_ref()
            .unwrap()
            .generated_tokens
            .len(),
        1
    );
    for participant in &summary.participants {
        let AttemptParticipantOutcome::Completed {
            work,
            frontier_committed: true,
        } = participant.outcome
        else {
            panic!("sampling cannot erase completed model work");
        };
        assert_eq!(work.range().unwrap().as_range(), 0..2);
    }
    assert_eq!(
        inputs[1]
            .handle
            .lock()
            .as_ref()
            .unwrap()
            .execution_frontier
            .as_ref()
            .unwrap()
            .consumed_tokens(),
        2
    );
    drop(works);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn attempt_prefix_evidence_does_not_cross_a_replacement_incarnation() {
    let recorder = Arc::new(AttemptRecorder::default());
    let (engine, executor, mut inputs) = fixture_observed(1, false, Some(recorder)).await;
    let original = inputs[0].frontier.sequence().clone();
    inputs[0]
        .handle
        .lock()
        .as_mut()
        .unwrap()
        .confirmed_prefix_restore = Some((original.clone(), 1));
    inputs[0].frontier = executor.issuer.frontier(&original, 1).unwrap();
    inputs[0].work = TokenWork::Prefill(PrefillChunk::new(1, 1, 2).unwrap());
    inputs[0].handle.lock().as_mut().unwrap().execution_frontier = Some(inputs[0].frontier.clone());
    let inner = engine.active_inner().unwrap();
    let capture = inner.prepare_execution_attempt(&[&inputs[0]]).unwrap();
    assert_eq!(
        capture.participants[0].prefix_restore,
        AttemptPrefixRestore::Restored { tokens: 1 }
    );
    let replacement = executor.issuer.issue_sequence().unwrap();
    inputs[0].frontier = executor.issuer.frontier(&replacement, 1).unwrap();
    inputs[0].handle.lock().as_mut().unwrap().execution_frontier = Some(inputs[0].frontier.clone());
    let capture = inner.prepare_execution_attempt(&[&inputs[0]]).unwrap();
    assert_eq!(
        capture.participants[0].prefix_restore,
        AttemptPrefixRestore::NoneObserved
    );
    executor.issuer.ended(&original).unwrap();
    drop(inner);
    engine.shutdown().await.unwrap();
}
