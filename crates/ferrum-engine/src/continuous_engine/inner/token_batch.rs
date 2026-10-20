//! One product token protocol for model-owned and native execution resources.

use super::*;
use ferrum_execution::{
    model_executor::{PrefillChunk, PrefillShapePolicy},
    token_execution::{
        ContinuationDisposition, ExecutionBatchOutcome, ExecutionCloseOutcome,
        ExecutionContractViolation, ExecutionFrontier, ExecutionInterruptionCause,
        ExecutionProgress, ExecutionWork, NoSubmissionReason, TokenAdmissionCapacityDeferral,
        TokenWork,
    },
};

mod attempt;
mod commit;
use attempt::TokenAttemptCapture;
use ferrum_execution::execution_attempt::ExecutionAttemptEligibility;
mod deferral;
mod outcome;
mod prefill;

#[cfg(test)]
mod preparation_tests;

struct TokenBatchInput {
    handle: Arc<SequenceHandle>,
    request_id: RequestId,
    frontier: ExecutionFrontier,
    work: TokenWork,
    logits_policy: LogitsReturnPolicy,
    context_tokens: usize,
}

impl TokenBatchInput {
    fn as_work(&self) -> Result<ExecutionWork<'_>> {
        ExecutionWork::new(&self.frontier, self.work, &self.logits_policy)
    }
}

#[derive(Clone, Copy)]
struct TokenExecutionTiming {
    started_at: Instant,
    completed_at: Instant,
}

struct TokenPreparationFailure {
    handle: Option<Arc<SequenceHandle>>,
    request_id: RequestId,
    // A preparation failure can close only the incarnation actually observed.
    // A missing outer state/frontier provides no authority over a replacement.
    frontier: Option<ExecutionFrontier>,
    error: FerrumError,
}

struct PreparedTokenBatch {
    inputs: Vec<TokenBatchInput>,
    failures: Vec<TokenPreparationFailure>,
}

#[derive(Default)]
struct TokenBatchEffects {
    ready: Vec<(Arc<SequenceHandle>, TokenId)>,
    failed: Vec<(Arc<SequenceHandle>, FerrumError)>,
}

/// Cold-path groups of unchanged work. Indices refer only to the current
/// selected cohort; completed peers never enter the retry stack.
struct TokenDeferredGroup {
    cause_index: usize,
    input_indices: Vec<usize>,
    deferral: ExecutorExecutionDeferral,
}

impl EngineInner {
    fn prepare_token_batch(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
        shape_policy: PrefillShapePolicy,
        already_projected: bool,
    ) -> PreparedTokenBatch {
        let mut inputs = Vec::with_capacity(batch.requests.len());
        let mut failures = Vec::new();
        for scheduled in &batch.requests {
            let request_id = &scheduled.request.id;
            let handle = self.sequences.get(request_id);
            let snapshot = (|| -> Result<_> {
                let handle = handle.as_ref().ok_or_else(|| {
                    FerrumError::internal("scheduled token request has no published handle")
                })?;
                let state = handle.lock();
                let sequence = state.as_ref().ok_or_else(|| {
                    FerrumError::internal("scheduled token request has no published sequence")
                })?;
                let frontier = sequence.execution_frontier.clone().ok_or_else(|| {
                    FerrumError::internal("scheduled token request has no admitted incarnation")
                })?;
                // Once captured, preserve this exact owner even if any later
                // preparation fails. No error can assign a different live owner.
                let prepared = (|| -> Result<_> {
                    // A latched accounting failure still owns this admitted
                    // incarnation, including after a real recompute pause.
                    if let Some(error) = sequence.output_accounting_error() {
                        return Err(error.clone());
                    }
                    if sequence.execution_close.is_some() {
                        return Err(FerrumError::internal(
                            "closing token incarnation was scheduled",
                        ));
                    }
                    let work = if sequence.prefill_complete {
                        TokenWork::Decode {
                            token: *sequence.generated_tokens.last().ok_or_else(|| {
                                FerrumError::internal("decode has no committed input token")
                            })?,
                            position: frontier.consumed_tokens(),
                        }
                    } else {
                        if scheduled.tokens_processed != frontier.consumed_tokens() {
                            return Err(FerrumError::internal(
                                "scheduler prefill position differs from the actual token frontier",
                            ));
                        }
                        TokenWork::Prefill(PrefillChunk::new(
                            frontier.consumed_tokens(),
                            scheduled.tokens_to_process.ok_or_else(|| {
                                FerrumError::scheduler("prefill has no scheduled token budget")
                            })?,
                            sequence.prefill_context_len(),
                        )?)
                    };
                    let policy = if sequence.prefill_complete {
                        sequence.model_decode_logits_policy()
                    } else {
                        LogitsReturnPolicy::FullLogits
                    };
                    Ok((
                        sequence.prefill_context_len(),
                        sequence.generated_tokens.is_empty(),
                        work,
                        policy,
                    ))
                })();
                Ok((frontier, prepared))
            })();
            let (frontier, prepared) = match snapshot {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    failures.push(TokenPreparationFailure {
                        handle: handle.clone(),
                        request_id: request_id.clone(),
                        frontier: None,
                        error,
                    });
                    continue;
                }
            };
            let prepared = prepared.and_then(|(context_tokens, generated_empty, work, policy)| {
                let work = match work {
                    TokenWork::Prefill(chunk) => {
                        TokenWork::Prefill(self.plan_prompt_tail_prefill_chunk(
                            request_id,
                            chunk,
                            generated_empty,
                            shape_policy,
                            already_projected,
                        )?)
                    }
                    decode => decode,
                };
                Ok((context_tokens, work, policy))
            });
            match prepared {
                Ok((context_tokens, work, logits_policy)) => inputs.push(TokenBatchInput {
                    handle: handle
                        .clone()
                        .expect("successful snapshot retained its handle"),
                    request_id: request_id.clone(),
                    frontier,
                    work,
                    logits_policy,
                    context_tokens,
                }),
                Err(error) => failures.push(TokenPreparationFailure {
                    handle,
                    request_id: request_id.clone(),
                    frontier: Some(frontier),
                    error,
                }),
            }
        }
        PreparedTokenBatch { inputs, failures }
    }

    pub(super) async fn process_token_batch(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
        shape_policy: PrefillShapePolicy,
        already_projected: bool,
    ) -> Result<()> {
        let PreparedTokenBatch { inputs, failures } =
            self.prepare_token_batch(batch, shape_policy, already_projected);
        let decode_root_width = inputs
            .iter()
            .filter(|input| matches!(input.work, TokenWork::Decode { .. }))
            .count();
        let mut capacity_pressure_root_width = None;
        let mut decode_root_eligible = true;
        let mut decode_root_completed = false;
        let mut pending = if self.config.batching.prefill_decode_execution
            == ferrum_types::PrefillDecodeExecution::Split
        {
            let (prefills, decodes): (Vec<_>, Vec<_>) = (0..inputs.len())
                .partition(|&index| matches!(inputs[index].work, TokenWork::Prefill(_)));
            // The stack visits prefill first, preserving the declared phase
            // policy while both phases use the same token execution protocol.
            vec![decodes, prefills]
        } else {
            vec![(0..inputs.len()).collect::<Vec<_>>()]
        };
        let mut first_error = None;
        while let Some(mut indices) = pending.pop() {
            // A preceding capacity-yield can retire a different member still
            // waiting in this split stack. Never submit its captured old work.
            indices.retain(|&index| {
                let input = &inputs[index];
                !self
                    .execution_closes
                    .contains_incarnation(input.frontier.sequence())
                    && input.handle.lock().as_ref().is_some_and(|sequence| {
                        sequence.execution_frontier.as_ref() == Some(&input.frontier)
                            && sequence.execution_close.is_none()
                    })
            });
            if indices.is_empty() {
                continue;
            }
            let selected = indices
                .iter()
                .map(|&index| &inputs[index])
                .collect::<Vec<_>>();
            let decode_only = selected
                .iter()
                .all(|input| matches!(input.work, TokenWork::Decode { .. }));
            let works = match selected
                .iter()
                .map(|input| input.as_work())
                .collect::<Result<Vec<_>>>()
            {
                Ok(works) => works,
                Err(error) => {
                    if decode_only {
                        decode_root_eligible = false;
                    }
                    for input in &selected {
                        if let Err(close_error) =
                            self.fail_token_input(input, error.clone(), None).await
                        {
                            first_error.get_or_insert(close_error);
                        }
                    }
                    continue;
                }
            };
            let mut attempt = self.prepare_execution_attempt(&selected);
            // A close may complete while numerical work is in flight. Keep
            // the pre-attempt epoch so installing a later wait cannot lose it.
            let observed_release_epoch = self.scheduler.confirmed_capacity_release_epoch();
            let capture_product_timing = self
                .config
                .runtime
                .profile_detail
                .captures_engine_token_timing();
            let started_at = (capture_product_timing || attempt.is_some()).then(Instant::now);
            let outcome = self.model_executor.execute_tokens(&works).await;
            let completed_at = started_at.map(|_| Instant::now());
            let measured = started_at
                .zip(completed_at)
                .map(|(started_at, completed_at)| TokenExecutionTiming {
                    started_at,
                    completed_at,
                });
            if let (Some(attempt), Some(interval)) = (&mut attempt, measured) {
                attempt.complete_interval(interval);
            }
            // Attempt collection must not enable request timing evidence or
            // consume semantic output quota when the product profile is Off.
            let timing = measured.filter(|_| capture_product_timing);
            // An ordinary error is never a reason to replay the whole cohort.
            let outcome = match outcome {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.publish_execution_attempt(
                        attempt,
                        ExecutionAttemptEligibility::ExecutorError,
                        &selected,
                        None,
                    );
                    if decode_only {
                        decode_root_eligible = false;
                    }
                    for input in &selected {
                        if let Err(close_error) =
                            self.fail_token_input(input, error.clone(), None).await
                        {
                            first_error.get_or_insert(close_error);
                        }
                    }
                    continue;
                }
            };
            if let ExecutionBatchOutcome::NotSubmitted(not_submitted) = outcome {
                if let Some(attempt) = &mut attempt {
                    attempt.not_submitted(not_submitted.reason());
                }
                self.publish_execution_attempt(
                    attempt,
                    ExecutionAttemptEligibility::NotSubmitted,
                    &selected,
                    None,
                );
                if decode_only {
                    decode_root_eligible = false;
                }
                let retry = self
                    .handle_token_deferral(
                        &selected,
                        not_submitted.into_reason(),
                        &mut capacity_pressure_root_width,
                        observed_release_epoch,
                    )
                    .await;
                match retry {
                    Ok(groups) => {
                        for group in groups.into_iter().rev() {
                            pending.push(group.into_iter().map(|local| indices[local]).collect());
                        }
                    }
                    Err(error) => {
                        for input in &selected {
                            if let Err(close_error) =
                                self.fail_token_input(input, error.clone(), None).await
                            {
                                first_error.get_or_insert(close_error);
                            }
                        }
                    }
                }
                continue;
            }
            // A split child or partial result cannot prove that the original
            // decode cohort now fits. The scheduler additionally requires an
            // actual capacity-release epoch before increasing its limit.
            let full_decode_success = decode_only
                && decode_root_eligible
                && selected.len() == decode_root_width
                && matches!(&outcome, ExecutionBatchOutcome::Completed(completed)
                if completed.completions().len() == works.len()
                    && completed.completions().iter().enumerate().all(|(index, completion)| {
                        completion.validate_for(index, &works[index], self.model_executor.info().vocab_size).is_ok()
                    }));
            let mut deferred = Vec::new();
            match self
                .consume_token_outcome(&selected, &works, outcome, timing, &mut deferred, attempt)
                .await
            {
                Ok(()) => {
                    decode_root_completed |= full_decode_success;
                }
                Err(error) => {
                    if decode_only {
                        decode_root_eligible = false;
                    }
                    first_error.get_or_insert(error);
                }
            }
            for group in deferred {
                decode_root_eligible = false;
                let deferred_inputs = group
                    .input_indices
                    .iter()
                    .map(|&index| selected[index])
                    .collect::<Vec<_>>();
                match self
                    .handle_token_deferral(
                        &deferred_inputs,
                        NoSubmissionReason::Deferred(group.deferral),
                        &mut capacity_pressure_root_width,
                        observed_release_epoch,
                    )
                    .await
                {
                    Ok(groups) => {
                        for retry in groups.into_iter().rev() {
                            pending.push(
                                retry
                                    .into_iter()
                                    .map(|local| indices[group.input_indices[local]])
                                    .collect(),
                            );
                        }
                    }
                    Err(error) => {
                        for input in deferred_inputs {
                            if let Err(close_error) =
                                self.fail_token_input(input, error.clone(), None).await
                            {
                                first_error.get_or_insert(close_error);
                            }
                        }
                    }
                }
            }
        }
        if decode_root_eligible && decode_root_completed {
            self.scheduler
                .record_decode_execution_capacity_success(decode_root_width);
        }
        // Healthy work and its products have been committed before any failed
        // preparation is allowed to await physical closure.
        for failure in failures {
            warn!(request_id = %failure.request_id, error = %failure.error, "token preparation failed");
            first_error.get_or_insert_with(|| failure.error.clone());
            if let (Some(handle), Some(frontier)) = (failure.handle, failure.frontier) {
                if let Err(error) = self
                    .fail_token_handle_frontier(&handle, &frontier, failure.error, None)
                    .await
                {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}
