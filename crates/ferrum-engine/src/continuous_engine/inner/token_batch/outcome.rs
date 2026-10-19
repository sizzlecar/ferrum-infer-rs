//! Consume successful peers and retain every close observation on a bad batch.

use super::*;
use ferrum_execution::token_execution::{ExecutionCloseDisposition, ExecutorSequence};

fn cohort_index(inputs: &[&TokenBatchInput], sequence: &ExecutorSequence) -> Option<usize> {
    inputs
        .iter()
        .position(|input| input.frontier.sequence() == sequence)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

fn continuation_sequence(continuation: &ContinuationDisposition) -> &ExecutorSequence {
    match continuation {
        ContinuationDisposition::Retryable(frontier) => frontier.sequence(),
        ContinuationDisposition::Ended(ended) => ended.sequence(),
        ContinuationDisposition::Held(held) => held.key().sequence(),
    }
}

impl EngineInner {
    pub(super) async fn consume_token_outcome(
        &self,
        inputs: &[&TokenBatchInput],
        works: &[ExecutionWork<'_>],
        outcome: ExecutionBatchOutcome,
        timing: Option<TokenExecutionTiming>,
        deferred: &mut Vec<TokenDeferredGroup>,
        mut attempt: Option<TokenAttemptCapture>,
    ) -> Result<()> {
        let vocabulary_size = self.model_executor.info().vocab_size;
        let validated_interruption = matches!(&outcome, ExecutionBatchOutcome::Interrupted(_));
        let malformed = match outcome {
            ExecutionBatchOutcome::Completed(batch) => {
                let error = if batch.completions().len() != inputs.len() {
                    Some(FerrumError::internal(
                        "completed token batch has missing participants",
                    ))
                } else {
                    batch
                        .completions()
                        .iter()
                        .enumerate()
                        .find_map(|(index, output)| {
                            output
                                .validate_for(index, &works[index], vocabulary_size)
                                .err()
                        })
                };
                if let Some(error) = error {
                    ExecutionContractViolation::new(
                        error,
                        (batch.into_completions(), Vec::new(), Vec::new()),
                    )
                } else {
                    let effects = self.commit_token_completions(
                        inputs,
                        batch.into_completions(),
                        timing,
                        &mut attempt,
                    );
                    let eligibility = attempt
                        .as_ref()
                        .map_or(ExecutionAttemptEligibility::Completed, |attempt| {
                            attempt.eligibility(false)
                        });
                    self.publish_execution_attempt(attempt, eligibility, inputs, Some(&effects));
                    return self.publish_token_effects(inputs, effects).await;
                }
            }
            ExecutionBatchOutcome::Interrupted(batch) => {
                let parts = batch.into_parts();
                let error = parts
                    .2
                    .iter()
                    .find_map(ExecutionInterruptionCause::as_error)
                    .cloned()
                    .unwrap_or_else(|| {
                        FerrumError::internal(
                            "interrupted token batch has an invalid participant contract",
                        )
                    });
                ExecutionContractViolation::new(error, parts)
            }
            ExecutionBatchOutcome::ContractViolation(violation) => violation,
            ExecutionBatchOutcome::NotSubmitted(_) => {
                return Err(FerrumError::internal(
                    "zero-submission outcome bypassed admission handling",
                ));
            }
        };
        let ExecutionContractViolation {
            error,
            completed,
            remaining,
            causes,
            rejected_completions,
            rejected_participants,
            unassigned_products,
        } = malformed;
        let mut invalid_contract = !validated_interruption
            || !rejected_completions.is_empty()
            || !rejected_participants.is_empty()
            || !unassigned_products.is_empty();
        let has_execution_error = causes.iter().any(|cause| cause.as_error().is_some());
        // A foreign or otherwise unassignable result may leave every requested
        // peer healthy. Still report the original failure independently; it is
        // not permission to fail or replay those peers.
        if !validated_interruption || causes.iter().any(|cause| cause.as_error().is_some()) {
            warn!(
            error = %error,
            completed = completed.len(),
            remaining = remaining.len(),
            rejected_products = rejected_completions.len(),
            rejected_participants = rejected_participants.len(),
            unassigned_products = unassigned_products.len(),
            "token batch required failure reconciliation"
            );
        }
        // The error path needs a cohort map. The ordered completed fast path
        // above does not allocate this table or clone any sampling product.
        let mut claims = vec![0usize; inputs.len()];
        // An invalid declared index cannot assign an actual incarnation to a
        // different peer. Detect conflicts by the identities the values retain,
        // including an out-of-bounds Held for an otherwise completed peer.
        let mut claim = |sequence: &ExecutorSequence| {
            if let Some(index) = cohort_index(inputs, sequence) {
                claims[index] += 1;
            }
        };
        for completion in &completed {
            claim(completion.next_frontier().sequence());
        }
        for participant in &remaining {
            claim(participant.sequence());
            let continuation = continuation_sequence(participant.continuation());
            if continuation != participant.sequence() {
                claim(continuation);
            }
        }
        for (_, _, continuation, _) in &rejected_participants {
            claim(continuation_sequence(continuation));
        }
        // Rejected completion tuples have no independently retained identity.
        // An in-range metadata claim therefore remains conservatively ambiguous.
        for (index, _, _) in &rejected_completions {
            if let Some(count) = claims.get_mut(*index) {
                *count += 1;
            }
        }
        let mut accepted = Vec::with_capacity(completed.len());
        let mut succeeded = vec![false; inputs.len()];
        let mut deferred_indices = vec![false; inputs.len()];
        let mut failures = vec![None; inputs.len()];
        for completion in completed {
            if let Some(index) = cohort_index(inputs, completion.next_frontier().sequence()) {
                let work = &works[index];
                match completion.validate_for(index, work, vocabulary_size) {
                    Ok(()) if claims[index] == 1 => {
                        succeeded[index] = true;
                        accepted.push(completion);
                    }
                    Err(cause) => {
                        invalid_contract = true;
                        failures[index] = Some(cause);
                    }
                    _ => {
                        invalid_contract = true;
                        failures[index] = Some(error.clone());
                    }
                }
            } else {
                invalid_contract = true;
            }
        }
        let mut closures = (0..inputs.len())
            .map(|_| None)
            .collect::<Vec<Option<ExecutionCloseOutcome>>>();
        let mut executed_failures = Vec::new();
        for participant in remaining {
            let (index, sequence, progress, continuation, cause_index) = participant.into_parts();
            let actual_index = cohort_index(inputs, &sequence);
            let continuation_index = cohort_index(inputs, continuation_sequence(&continuation));
            let matching =
                actual_index == Some(index) && continuation_sequence(&continuation) == &sequence;
            if !matching || claims[index] != 1 || cause_index >= causes.len() {
                invalid_contract = true;
            }
            if validated_interruption
                && matching
                && claims[index] == 1
                && matches!(progress, ExecutionProgress::NotSubmitted { .. })
                && matches!(&continuation, ContinuationDisposition::Retryable(frontier)
                    if frontier == &inputs[index].frontier && frontier.sequence().observed_end().is_none())
            {
                if let Some(ExecutionInterruptionCause::Deferred(deferral)) =
                    causes.get(cause_index)
                {
                    deferred_indices[index] = true;
                    if let Some(attempt) = &mut attempt {
                        attempt.deferred(index, progress);
                    }
                    if let Some(group) = deferred
                        .iter_mut()
                        .find(|group| group.cause_index == cause_index)
                    {
                        group.input_indices.push(index);
                    } else {
                        deferred.push(TokenDeferredGroup {
                            cause_index,
                            input_indices: vec![index],
                            deferral: deferral.clone(),
                        });
                    }
                    continue;
                }
            }
            if validated_interruption && matching && claims[index] == 1 {
                if let Some(attempt) = &mut attempt {
                    attempt.progress(index, progress);
                }
                if !matches!(progress, ExecutionProgress::NotSubmitted { .. }) {
                    executed_failures.push(index);
                }
            }
            let cause = causes
                .get(cause_index)
                .and_then(ExecutionInterruptionCause::as_error)
                .cloned()
                .unwrap_or_else(|| error.clone());
            if let Some(actual_index) = actual_index {
                failures[actual_index] = Some(cause.clone());
            }
            if let Some(continuation_index) = continuation_index {
                failures[continuation_index] = Some(cause.clone());
            }
            self.retain_token_continuation(
                inputs,
                index,
                continuation,
                cause,
                matching,
                &mut closures,
            );
        }
        for (index, _, continuation, cause_index) in rejected_participants {
            let cause = causes
                .get(cause_index)
                .and_then(ExecutionInterruptionCause::as_error)
                .cloned()
                .unwrap_or_else(|| error.clone());
            if let Some(actual_index) = cohort_index(inputs, continuation_sequence(&continuation)) {
                failures[actual_index] = Some(cause.clone());
            }
            // Rejected records have no independently trusted index/identity
            // mapping. Only the continuation's actual identity may match.
            self.retain_token_continuation(inputs, index, continuation, cause, true, &mut closures);
        }
        // Save every healthy model result before any close/output await. Close
        // observations for failed peers move into engine-owned sequence state,
        // so cancelling the caller cannot drop the rest of this batch's Helds.
        let mut effects = self.commit_token_completions(inputs, accepted, timing, &mut attempt);
        // The real healthy frontiers are already saved. Only independently
        // validated submitted/terminal participants consume scheduling time;
        // NotSubmitted retries preserve the still-open interval.
        if let Some(timing) = timing {
            for index in executed_failures {
                let input = inputs[index];
                if !matches!(input.work, TokenWork::Decode { .. }) {
                    continue;
                }
                let mut state = input.handle.lock();
                if let Some(sequence) = state.as_mut().filter(|sequence| {
                    self.sequences.is_current(&input.handle)
                        && sequence.execution_frontier.as_ref() == Some(&input.frontier)
                        && sequence.execution_close.is_none()
                }) {
                    sequence.close_decode_scheduling(timing.started_at);
                    sequence.record_decode_execution(timing.started_at, timing.completed_at);
                }
            }
        }
        for (index, input) in inputs.iter().enumerate() {
            if !succeeded[index] && !deferred_indices[index] {
                effects.failed.push((
                    Arc::clone(&input.handle),
                    failures[index].take().unwrap_or_else(|| error.clone()),
                ));
            }
            if let Some(outcome) = closures[index].take() {
                let mut state = input.handle.lock();
                if let Some(sequence) = state.as_mut().filter(|sequence| {
                    sequence
                        .execution_frontier
                        .as_ref()
                        .is_some_and(|frontier| frontier.sequence() == input.frontier.sequence())
                        && sequence.execution_close.is_none()
                }) {
                    sequence.execution_close = Some(Box::new(Mutex::new(outcome)));
                } else {
                    drop(state);
                    self.retain_unassigned_execution_close(
                        input.request_id.clone(),
                        outcome,
                        error.clone(),
                    );
                }
            }
        }
        let mut eligibility = attempt
            .as_ref()
            .map_or(ExecutionAttemptEligibility::InvalidOutcome, |attempt| {
                attempt.eligibility(invalid_contract)
            });
        if has_execution_error
            && matches!(
                eligibility,
                ExecutionAttemptEligibility::Completed
                    | ExecutionAttemptEligibility::CompletedWithNotSubmitted
            )
        {
            eligibility = ExecutionAttemptEligibility::Interrupted;
        }
        self.publish_execution_attempt(attempt, eligibility, inputs, Some(&effects));
        self.publish_token_effects(inputs, effects).await
    }

    fn retain_token_continuation(
        &self,
        inputs: &[&TokenBatchInput],
        index: usize,
        continuation: ContinuationDisposition,
        cause: FerrumError,
        mapping_matches: bool,
        closures: &mut [Option<ExecutionCloseOutcome>],
    ) {
        let outcome = match continuation {
            ContinuationDisposition::Ended(ended) => ExecutionCloseOutcome::ended(ended),
            ContinuationDisposition::Held(held) => ExecutionCloseOutcome::held(held),
            // Retry is a policy decision, not implied by a model error. This
            // consumer fails the request and closes its original live owner.
            ContinuationDisposition::Retryable(_) => return,
        }
        .with_error(cause.clone());
        let sequence = match &outcome.disposition {
            ExecutionCloseDisposition::Ended(ended) => ended.sequence(),
            ExecutionCloseDisposition::Held(held) => held.key().sequence(),
        };
        if mapping_matches
            && inputs
                .get(index)
                .is_some_and(|input| input.frontier.sequence() == sequence)
            && closures[index].is_none()
        {
            closures[index] = Some(outcome);
        } else if let Some(input) = inputs.get(index).or_else(|| inputs.first()) {
            self.retain_unassigned_execution_close(input.request_id.clone(), outcome, cause);
        }
    }

    pub(super) async fn fail_token_input(
        &self,
        input: &TokenBatchInput,
        error: FerrumError,
        outcome: Option<ExecutionCloseOutcome>,
    ) -> Result<()> {
        self.fail_token_handle_frontier(&input.handle, &input.frontier, error, outcome)
            .await
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) async fn fail_token_frontier(
        &self,
        request_id: &RequestId,
        frontier: &ExecutionFrontier,
        error: FerrumError,
        outcome: Option<ExecutionCloseOutcome>,
    ) -> Result<()> {
        if let Some(handle) = self.sequences.get(request_id) {
            return self
                .fail_token_handle_frontier(&handle, frontier, error, outcome)
                .await;
        }
        self.close_stale_token_frontier(request_id, frontier, error, outcome)
            .await
    }

    pub(in crate::continuous_engine) async fn fail_token_handle_frontier(
        &self,
        handle: &Arc<SequenceHandle>,
        frontier: &ExecutionFrontier,
        error: FerrumError,
        mut outcome: Option<ExecutionCloseOutcome>,
    ) -> Result<()> {
        let current = {
            let mut state = handle.lock();
            let current = state.as_mut().filter(|sequence| {
                self.sequences.is_current(handle)
                    && sequence
                        .execution_frontier
                        .as_ref()
                        .is_some_and(|current| current.sequence() == frontier.sequence())
            });
            if let Some(sequence) = current {
                if sequence.execution_close.is_none() {
                    sequence.execution_close =
                        outcome.take().map(|outcome| Box::new(Mutex::new(outcome)));
                }
                true
            } else {
                false
            }
        };
        if current {
            if let Some(outcome) = outcome {
                self.retain_unassigned_execution_close(
                    handle.request_id().clone(),
                    outcome,
                    error.clone(),
                );
            }
            self.complete_request_exact_frontier_detached(
                handle,
                frontier.sequence(),
                FinishReason::Error,
                Some(error),
            )
        } else {
            self.close_stale_token_frontier(handle.request_id(), frontier, error, outcome)
                .await
        }
    }

    async fn close_stale_token_frontier(
        &self,
        request_id: &RequestId,
        frontier: &ExecutionFrontier,
        error: FerrumError,
        outcome: Option<ExecutionCloseOutcome>,
    ) -> Result<()> {
        // Observing an old incarnation never grants a second product slot or
        // a second capacity-release notification.
        let ticket = self.begin_detached_execution_close(
            request_id.clone(),
            frontier.sequence().clone(),
            outcome,
        )?;
        let result = ticket.initial_result().await;
        let observation = self.drain_execution_closes();
        result.and(observation).and(Err(error))
    }

    async fn publish_token_effects(
        &self,
        inputs: &[&TokenBatchInput],
        effects: TokenBatchEffects,
    ) -> Result<()> {
        let mut first_error = None;
        for (handle, token) in &effects.ready {
            if effects
                .failed
                .iter()
                .any(|(failed, _)| Arc::ptr_eq(failed, handle))
            {
                continue;
            }
            let stop = self.stop_reason_for_sequence(handle);
            if self.should_stream_sequence_token(handle, *token, stop) {
                self.send_stream_update_for_sequence(handle, *token).await;
            }
            if let Some(reason) = stop {
                if let Err(error) = self.complete_request_exact_detached(handle, reason, None) {
                    first_error.get_or_insert(error);
                }
            }
        }
        let mut closed = HashSet::new();
        for (handle, error) in effects.failed {
            if !closed.insert(Arc::as_ptr(&handle) as usize) {
                continue;
            }
            if let Some(input) = inputs
                .iter()
                .find(|input| Arc::ptr_eq(&input.handle, &handle))
            {
                if let Err(close_error) = self.fail_token_input(input, error, None).await {
                    first_error.get_or_insert(close_error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
