//! Projection of already validated product facts; no second execution observer.

use super::*;
use ferrum_execution::execution_attempt::*;
use ferrum_execution::token_execution::ExecutedTokenWork;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub(super) struct TokenAttemptCapture {
    pub(super) participants: Vec<ExecutionAttemptParticipant>,
    interval: Option<TokenExecutionTiming>,
    completion: Option<Box<dyn ExecutionAttemptCompletion>>,
}

impl TokenAttemptCapture {
    pub(super) fn complete_interval(&mut self, interval: TokenExecutionTiming) {
        self.interval = Some(interval);
    }

    pub(super) fn completed(&mut self, index: usize, work: ExecutedTokenWork, committed: bool) {
        self.participants[index].outcome = AttemptParticipantOutcome::Completed {
            work,
            frontier_committed: committed,
        };
    }

    pub(super) fn progress(&mut self, index: usize, progress: ExecutionProgress) {
        self.participants[index].outcome = AttemptParticipantOutcome::Interrupted(progress);
    }

    pub(super) fn deferred(&mut self, index: usize, progress: ExecutionProgress) {
        if let ExecutionProgress::NotSubmitted { stage } = progress {
            self.participants[index].outcome = AttemptParticipantOutcome::NotSubmitted {
                stage,
                cause: AttemptNoSubmissionCause::Deferred,
            };
        }
    }

    pub(super) fn not_submitted(&mut self, reason: &NoSubmissionReason) {
        let cause = match reason {
            NoSubmissionReason::Deferred(_) => AttemptNoSubmissionCause::Deferred,
            NoSubmissionReason::UnsupportedShape => AttemptNoSubmissionCause::UnsupportedShape,
            NoSubmissionReason::Rejected(_) => AttemptNoSubmissionCause::Rejected,
        };
        for participant in &mut self.participants {
            participant.outcome = AttemptParticipantOutcome::NotSubmitted {
                stage: ferrum_execution::token_execution::NoSubmissionStage::BeforeSubmit,
                cause,
            };
        }
    }

    pub(super) fn eligibility(&self, invalid_contract: bool) -> ExecutionAttemptEligibility {
        if invalid_contract
            || self.participants.iter().any(|participant| {
                matches!(
                    participant.outcome,
                    AttemptParticipantOutcome::Unverified
                        | AttemptParticipantOutcome::Completed {
                            frontier_committed: false,
                            ..
                        }
                )
            })
        {
            return ExecutionAttemptEligibility::InvalidOutcome;
        }
        if self.participants.iter().any(|participant| {
            matches!(
                participant.outcome,
                AttemptParticipantOutcome::Interrupted(_)
            )
        }) {
            return ExecutionAttemptEligibility::Interrupted;
        }
        let completed = self
            .participants
            .iter()
            .filter(|participant| {
                matches!(
                    participant.outcome,
                    AttemptParticipantOutcome::Completed { .. }
                )
            })
            .count();
        if completed == self.participants.len() {
            ExecutionAttemptEligibility::Completed
        } else if completed == 0 {
            ExecutionAttemptEligibility::NotSubmitted
        } else {
            ExecutionAttemptEligibility::CompletedWithNotSubmitted
        }
    }
}

impl EngineInner {
    pub(super) fn prepare_execution_attempt(
        &self,
        inputs: &[&TokenBatchInput],
    ) -> Option<TokenAttemptCapture> {
        // Off returns before limit queries, allocation, sequence projection or
        // clocks. Detailed runtime capture remains an independent switch.
        let observer = self.execution_attempt_observer.as_ref()?;
        let limits = catch_unwind(AssertUnwindSafe(|| observer.limits())).ok()?;
        if !limits.accepts(inputs.len()) {
            let _ = catch_unwind(AssertUnwindSafe(|| {
                observer.record_drop(ExecutionAttemptDrop::ProjectionLimit);
            }));
            counter!("ferrum.engine.execution_attempt_dropped_total", "reason" => "projection_limit")
                .increment(1);
            return None;
        }
        let participants: Vec<_> = inputs
            .iter()
            .map(|input| {
                let state = input.handle.lock();
                let prefix_restore = state
                    .as_ref()
                    .filter(|sequence| {
                        self.sequences.is_current(&input.handle)
                            && sequence.execution_frontier.as_ref() == Some(&input.frontier)
                    })
                    .and_then(|sequence| sequence.confirmed_prefix_restore.as_ref())
                    .filter(|(incarnation, tokens)| {
                        incarnation == input.frontier.sequence()
                            && *tokens <= input.frontier.consumed_tokens()
                    })
                    .map_or(AttemptPrefixRestore::NoneObserved, |(_, tokens)| {
                        AttemptPrefixRestore::Restored { tokens: *tokens }
                    });
                ExecutionAttemptParticipant {
                    request_id: input.request_id.clone(),
                    kind: match input.work {
                        TokenWork::Prefill(_) => AttemptWorkKind::Prefill,
                        TokenWork::Decode { .. } => AttemptWorkKind::Decode,
                    },
                    // ExecutionWork was validated before this optional projection.
                    requested: input.work.range().expect("validated execution work range"),
                    context_tokens: input.context_tokens,
                    prefix_restore,
                    internally_reused_tokens: None,
                    outcome: AttemptParticipantOutcome::Unverified,
                    product_failed: false,
                }
            })
            .collect();
        // Every sequence lock used for projection has been released before the
        // optional observer reads its immutable predictor snapshot.
        let completion = catch_unwind(AssertUnwindSafe(|| observer.prepare(&participants)))
            .unwrap_or_else(|_| {
                counter!("ferrum.engine.execution_attempt_dropped_total", "reason" => "prepare_panic")
                    .increment(1);
                None
            });
        Some(TokenAttemptCapture {
            participants,
            interval: None,
            completion,
        })
    }

    pub(super) fn publish_execution_attempt(
        &self,
        capture: Option<TokenAttemptCapture>,
        eligibility: ExecutionAttemptEligibility,
        inputs: &[&TokenBatchInput],
        effects: Option<&TokenBatchEffects>,
    ) {
        let (Some(mut capture), Some(observer)) = (capture, &self.execution_attempt_observer)
        else {
            return;
        };
        let Some(timing) = capture.interval else {
            return;
        };
        if let Some(effects) = effects {
            for (index, participant) in capture.participants.iter_mut().enumerate() {
                participant.product_failed = effects
                    .failed
                    .iter()
                    .any(|(handle, _)| Arc::ptr_eq(handle, &inputs[index].handle));
            }
        }
        let summary = ExecutionAttemptSummary {
            started_at: timing.started_at,
            completed_at: timing.completed_at,
            participants: capture.participants,
            eligibility,
        };
        // The actual frontier and continuation owners are already retained.
        // An optional observer cannot prevent publication or trigger replay.
        match catch_unwind(AssertUnwindSafe(|| match capture.completion {
            Some(completion) => completion.try_record(&summary),
            None => observer.try_record(&summary),
        })) {
            Ok(ExecutionAttemptDelivery::Recorded) => {}
            Ok(ExecutionAttemptDelivery::Dropped(_)) => {
                counter!("ferrum.engine.execution_attempt_dropped_total", "reason" => "observer")
                    .increment(1);
            }
            Err(_) => {
                counter!("ferrum.engine.execution_attempt_dropped_total", "reason" => "panic")
                    .increment(1);
            }
        }
    }
}
