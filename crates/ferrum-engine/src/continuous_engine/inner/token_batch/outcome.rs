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
        completed_at: Option<Instant>,
        deferred: &mut Vec<TokenDeferredGroup>,
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
                        completed_at,
                    );
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
                        failures[index] = Some(cause);
                    }
                    _ => {
                        failures[index] = Some(error.clone());
                    }
                }
            }
        }
        let mut closures = (0..inputs.len())
            .map(|_| None)
            .collect::<Vec<Option<ExecutionCloseOutcome>>>();
        for participant in remaining {
            let (index, sequence, progress, continuation, cause_index) = participant.into_parts();
            let actual_index = cohort_index(inputs, &sequence);
            let continuation_index = cohort_index(inputs, continuation_sequence(&continuation));
            let matching =
                actual_index == Some(index) && continuation_sequence(&continuation) == &sequence;
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
        let mut effects = self.commit_token_completions(inputs, accepted, completed_at);
        for (index, input) in inputs.iter().enumerate() {
            if !succeeded[index] && !deferred_indices[index] {
                effects.failed.push((
                    input.request_id.clone(),
                    failures[index].take().unwrap_or_else(|| error.clone()),
                ));
            }
            if let Some(outcome) = closures[index].take() {
                let mut sequences = self.sequences.write();
                if let Some(sequence) = sequences.get_mut(&input.request_id).filter(|sequence| {
                    sequence
                        .execution_frontier
                        .as_ref()
                        .is_some_and(|frontier| frontier.sequence() == input.frontier.sequence())
                        && sequence.execution_close.is_none()
                }) {
                    sequence.execution_close = Some(Box::new(Mutex::new(outcome)));
                } else {
                    drop(sequences);
                    self.retain_unassigned_execution_close(
                        input.request_id.clone(),
                        outcome,
                        error.clone(),
                    );
                }
            }
        }
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
        self.fail_token_frontier(&input.request_id, &input.frontier, error, outcome)
            .await
    }

    pub(in crate::continuous_engine) async fn fail_token_frontier(
        &self,
        request_id: &RequestId,
        frontier: &ExecutionFrontier,
        error: FerrumError,
        outcome: Option<ExecutionCloseOutcome>,
    ) -> Result<()> {
        let current = self
            .sequences
            .read()
            .get(request_id)
            .is_some_and(|sequence| {
                sequence
                    .execution_frontier
                    .as_ref()
                    .is_some_and(|current| current.sequence() == frontier.sequence())
            });
        if current {
            if let Some(outcome) = outcome {
                let mut sequences = self.sequences.write();
                let sequence = sequences
                    .get_mut(request_id)
                    .expect("iteration owner keeps the failed request present");
                if sequence.execution_close.is_none() {
                    sequence.execution_close = Some(Box::new(Mutex::new(outcome)));
                } else {
                    drop(sequences);
                    self.retain_unassigned_execution_close(
                        request_id.clone(),
                        outcome,
                        error.clone(),
                    );
                }
            }
            self.complete_request_with_error(request_id, error).await
        } else {
            // Never cancel a replacement merely because it reused RequestId.
            // The old product's slot/close obligation is no longer ours. Keep
            // observing its exact executor owner without minting another
            // capacity-release notification for a previously drained Ended.
            let ticket = self.begin_detached_execution_close(
                request_id.clone(),
                frontier.sequence().clone(),
                outcome,
            )?;
            let result = ticket.initial_result().await;
            let observation = self.drain_execution_closes();
            result.and(observation).and(Err(error))
        }
    }

    async fn publish_token_effects(
        &self,
        inputs: &[&TokenBatchInput],
        effects: TokenBatchEffects,
    ) -> Result<()> {
        let mut first_error = None;
        let failed_ids = effects
            .failed
            .iter()
            .map(|(id, _)| id)
            .collect::<HashSet<_>>();
        for (request_id, token) in &effects.ready {
            if failed_ids.contains(request_id) {
                continue;
            }
            let stop = self.stop_reason_for_request(request_id);
            if self.should_stream_generated_token(request_id, *token, stop) {
                self.send_stream_update(request_id, *token).await;
            }
            if let Some(reason) = stop {
                if let Err(error) = self.complete_request(request_id, reason).await {
                    first_error.get_or_insert(error);
                }
            }
        }
        let mut closed = HashSet::new();
        for (request_id, error) in effects.failed {
            if !closed.insert(request_id.clone()) {
                continue;
            }
            if let Some(input) = inputs.iter().find(|input| input.request_id == request_id) {
                if let Err(close_error) = self.fail_token_input(input, error, None).await {
                    first_error.get_or_insert(close_error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
