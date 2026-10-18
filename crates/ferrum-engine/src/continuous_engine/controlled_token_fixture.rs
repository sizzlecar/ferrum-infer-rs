//! Common-protocol plumbing for controlled scheduler fixtures only. The fake
//! operations below are host-only and have no in-flight device work. Genuine
//! allocator, model and device lifecycle evidence lives in the real-model tests.
use super::*;
use ferrum_execution::model_executor::{PlanRuntimeMixedBatchOutcome, PlanRuntimePrefillProduct};
use ferrum_execution::token_execution::*;

#[derive(Clone)]
struct Entry {
    request_id: RequestId,
    prompt: Arc<[TokenId]>,
    maximum_tokens: usize,
    frontier: ExecutionFrontier,
    cache: Option<Arc<dyn KvCacheHandle>>,
}

#[derive(Default)]
pub(super) struct ControlledTokenState {
    issuer: ExecutorSequenceIssuer,
    entries: Mutex<HashMap<ExecutorSequence, Entry>>,
}

impl ControlledTokenState {
    pub(super) fn frontier_for_request(&self, request: &RequestId) -> Option<ExecutionFrontier> {
        self.entries
            .lock()
            .values()
            .find(|entry| &entry.request_id == request)
            .map(|entry| entry.frontier.clone())
    }

    /// Called only by a fixture's successful prefix-publication callback.
    pub(super) fn restore_frontier(
        &self,
        sequence: &ExecutorSequence,
        tokens: usize,
    ) -> Result<ExecutionFrontier> {
        let mut entries = self.entries.lock();
        let entry = entries
            .get_mut(sequence)
            .ok_or_else(|| FerrumError::internal("fixture restore target is no longer live"))?;
        if entry.frontier.consumed_tokens() != 0 || tokens == 0 || tokens >= entry.prompt.len() {
            return Err(FerrumError::internal(
                "fixture restore has an invalid target frontier",
            ));
        }
        entry.frontier = self.issuer.frontier(entry.frontier.sequence(), tokens)?;
        entry.cache = Some(Arc::new(ferrum_testkit::MockKvCacheHandle::new(
            entry.request_id.clone(),
            1,
            tokens,
        )));
        Ok(entry.frontier.clone())
    }
}

pub(super) fn admit(
    executor: &impl ModelExecutor,
    state: &ControlledTokenState,
    input: ExecutionAdmission<'_>,
) -> Result<TokenAdmissionOutcome> {
    input.validate()?;
    let decision = executor.try_admit_prefill(input.context)?;
    Ok(match decision {
        ExecutorPrefillAdmissionDecision::Admitted(_) => {
            let sequence = state.issuer.issue_sequence()?;
            let frontier = state.issuer.frontier(&sequence, 0)?;
            let entry = Entry {
                request_id: input.context.request_id.clone(),
                prompt: input.context.input_tokens.into(),
                maximum_tokens: input.context.maximum_sequence_tokens,
                frontier: frontier.clone(),
                cache: None,
            };
            assert!(state.entries.lock().insert(sequence, entry).is_none());
            TokenAdmissionOutcome::Admitted(frontier)
        }
        ExecutorPrefillAdmissionDecision::Deferred(value) => TokenAdmissionOutcome::Deferred(value),
        ExecutorPrefillAdmissionDecision::MaintenanceDeferred(value) => {
            TokenAdmissionOutcome::MaintenanceDeferred(value)
        }
        ExecutorPrefillAdmissionDecision::PermanentRejected(value) => {
            TokenAdmissionOutcome::PermanentRejected(value)
        }
    })
}

pub(super) fn cancel(
    executor: &impl ModelExecutor,
    state: &ControlledTokenState,
    sequence: &ExecutorSequence,
) -> Result<ExecutionCloseOutcome> {
    state.issuer.frontier(sequence, 0)?;
    if let Some(entry) = state.entries.lock().remove(sequence) {
        executor.cancel_prefill_admission(&entry.request_id);
        if let Some(cache) = entry.cache {
            executor.release_cache(&cache.cache_id());
        }
        return Ok(ExecutionCloseOutcome::ended(state.issuer.ended(sequence)?));
    }
    sequence
        .observed_end()
        .map(ExecutionCloseOutcome::ended)
        .ok_or_else(|| FerrumError::internal("fixture close has no exact retained incarnation"))
}

pub(super) async fn finish(
    executor: &impl ModelExecutor,
    state: &ControlledTokenState,
    sequence: &ExecutorSequence,
    reason: ExecutionEndReason,
) -> Result<ExecutionCloseOutcome> {
    let ExecutionEndReason::Completed {
        input_tokens,
        output_tokens,
    } = reason
    else {
        return cancel(executor, state, sequence);
    };
    state.issuer.frontier(sequence, 0)?;
    let entry = state.entries.lock().get(sequence).cloned();
    let Some(entry) = entry else {
        return cancel(executor, state, sequence);
    };
    if let Some(cache) = &entry.cache {
        executor
            .complete_cache(ExecutorSequenceCompletion::new(
                entry.request_id.clone(),
                cache.cache_id(),
                input_tokens,
                output_tokens,
            )?)
            .await?;
    }
    executor.cancel_prefill_admission(&entry.request_id);
    state
        .entries
        .lock()
        .remove(sequence)
        .expect("fixture finish retains the same owner");
    Ok(ExecutionCloseOutcome::ended(state.issuer.ended(sequence)?))
}

pub(super) fn release_sources(
    executor: &impl ModelExecutor,
    state: &ControlledTokenState,
    sequence: &ExecutorSequence,
    sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
) -> Result<bool> {
    state.issuer.frontier(sequence, 0)?;
    sources.clear();
    let Some(entry) = state.entries.lock().get(sequence).cloned() else {
        return Ok(false);
    };
    let cache_id = entry
        .cache
        .as_ref()
        .map(|cache| cache.cache_id())
        .unwrap_or_else(|| entry.request_id.to_string());
    executor.write_execution_capacity_release_sources(
        &ExecutorExecutionCapacityPreemption::new(entry.request_id, cache_id)?,
        sources,
    )
}

struct Row {
    index: usize,
    executed: ExecutedTokenWork,
    product: ExecutionProduct,
    cache: Arc<dyn KvCacheHandle>,
    error: Option<FerrumError>,
}

fn prefill_rows(
    indices: &[usize],
    inputs: &[PlanRuntimePrefillInput],
    outputs: Vec<PlanRuntimePrefillCompletion>,
    vocabulary: usize,
) -> (Vec<Row>, bool) {
    let wrong_length = indices.len() != outputs.len();
    let rows = outputs
        .into_iter()
        .enumerate()
        .map(|(position, completion)| {
            let error = inputs.get(position).and_then(|input| {
                completion
                    .validate_for(&input.request_id, input.chunk, vocabulary)
                    .err()
            });
            let (output, _, chunk, capacity_probe_count) = completion.into_parts();
            let (authority, product) = output.into_parts();
            Row {
                index: indices.get(position).copied().unwrap_or(usize::MAX),
                executed: ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count,
                },
                product: match product {
                    PlanRuntimePrefillProduct::Intermediate => ExecutionProduct::Intermediate,
                    PlanRuntimePrefillProduct::FinalLogits(values) => {
                        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values))
                    }
                },
                cache: authority.into_cache(),
                error,
            }
        })
        .collect();
    (rows, wrong_length)
}

fn decode_rows(
    indices: &[usize],
    inputs: &[PlanRuntimeDecodeInput],
    outputs: Vec<PlanRuntimeDecodeOutput>,
    work: &[ExecutionWork<'_>],
) -> (Vec<Row>, bool) {
    let wrong_length = indices.len() != outputs.len();
    let rows = outputs
        .into_iter()
        .enumerate()
        .map(|(position, output)| {
            let index = indices.get(position).copied().unwrap_or(usize::MAX);
            let token_position = match work.get(index).map(|input| input.work()) {
                Some(TokenWork::Decode { position, .. }) => position,
                _ => 0,
            };
            let error = inputs.get(position).and_then(|input| {
                (input.kv_cache.cache_id() != output.kv_cache.cache_id())
                    .then(|| FerrumError::backend("controlled decode returned a foreign cache"))
            });
            Row {
                index,
                executed: ExecutedTokenWork::Decode {
                    position: token_position,
                },
                product: ExecutionProduct::Sampling(output.sampling_output),
                cache: output.kv_cache,
                error,
            }
        })
        .collect();
    (rows, wrong_length)
}

enum FakeOutcome {
    Rows(Vec<Row>, bool),
    Deferred(ferrum_execution::model_executor::ExecutorExecutionDeferral),
    Unsupported,
}

pub(super) async fn execute(
    executor: &impl ModelExecutor,
    state: &ControlledTokenState,
    work: &[ExecutionWork<'_>],
) -> Result<ExecutionBatchOutcome> {
    let scope = state.issuer.batch(work, executor.info().vocab_size)?;
    let entries = {
        let entries = state.entries.lock();
        work.iter()
            .map(|input| {
                let entry = entries
                    .get(input.sequence())
                    .ok_or_else(|| FerrumError::internal("unknown fixture incarnation"))?;
                if &entry.frontier != input.expected_frontier()
                    || input.work().range()?.end() > entry.maximum_tokens
                {
                    return Err(FerrumError::internal(
                        "stale fixture frontier or admission extent",
                    ));
                }
                Ok(entry.clone())
            })
            .collect::<Result<Vec<_>>>()?
    };
    let mut prefills = Vec::new();
    let mut prefill_indices = Vec::new();
    let mut decodes = Vec::new();
    let mut decode_indices = Vec::new();
    for (index, (input, entry)) in work.iter().zip(&entries).enumerate() {
        match input.work() {
            TokenWork::Prefill(chunk) => {
                prefills.push(PlanRuntimePrefillInput::new(
                    entry.request_id.clone(),
                    entry.prompt.clone(),
                    entry.maximum_tokens,
                    chunk,
                )?);
                prefill_indices.push(index);
            }
            TokenWork::Decode { token, .. } => {
                decodes.push(
                    PlanRuntimeDecodeInput::new(
                        entry.request_id.clone(),
                        token,
                        entry.cache.clone().ok_or_else(|| {
                            FerrumError::internal("fixture decode has no completed prefill")
                        })?,
                    )
                    .with_logits_policy(input.logits_policy().clone()),
                );
                decode_indices.push(index);
            }
        }
    }
    let result: Result<FakeOutcome> = async {
        if decodes.is_empty() {
            if prefills.len() == 1 {
                return Ok(
                    match executor
                        .plan_runtime_prefill_with_capacity(&prefills[0])
                        .await?
                    {
                        PlanRuntimePrefillOutcome::Completed(value) => {
                            let (rows, bad) = prefill_rows(
                                &prefill_indices,
                                &prefills,
                                vec![value],
                                executor.info().vocab_size,
                            );
                            FakeOutcome::Rows(rows, bad)
                        }
                        PlanRuntimePrefillOutcome::Deferred(value) => FakeOutcome::Deferred(value),
                    },
                );
            }
            return Ok(
                match executor
                    .plan_runtime_batch_prefill_with_capacity(&prefills)
                    .await?
                {
                    PlanRuntimeBatchPrefillOutcome::Completed(values) => {
                        let (rows, bad) = prefill_rows(
                            &prefill_indices,
                            &prefills,
                            values,
                            executor.info().vocab_size,
                        );
                        FakeOutcome::Rows(rows, bad)
                    }
                    PlanRuntimeBatchPrefillOutcome::NotSubmitted(value) => {
                        FakeOutcome::Deferred(value)
                    }
                    PlanRuntimeBatchPrefillOutcome::Unsupported => FakeOutcome::Unsupported,
                },
            );
        }
        if prefills.is_empty() {
            return Ok(
                match executor
                    .plan_runtime_batch_decode_with_capacity(&decodes)
                    .await?
                {
                    PlanRuntimeBatchDecodeOutcome::Completed(values) => {
                        let (rows, bad) = decode_rows(&decode_indices, &decodes, values, work);
                        FakeOutcome::Rows(rows, bad)
                    }
                    PlanRuntimeBatchDecodeOutcome::Deferred(value) => FakeOutcome::Deferred(value),
                },
            );
        }
        Ok(
            match executor
                .plan_runtime_mixed_batch_with_capacity(&prefills, &decodes)
                .await?
            {
                PlanRuntimeMixedBatchOutcome::Unsupported => FakeOutcome::Unsupported,
                PlanRuntimeMixedBatchOutcome::NotSubmitted(value) => FakeOutcome::Deferred(value),
                PlanRuntimeMixedBatchOutcome::Completed {
                    prefills: first,
                    decodes: second,
                } => {
                    let (mut rows, a) = prefill_rows(
                        &prefill_indices,
                        &prefills,
                        first,
                        executor.info().vocab_size,
                    );
                    let (tail, b) = decode_rows(&decode_indices, &decodes, second, work);
                    rows.extend(tail);
                    rows.sort_by_key(|row| row.index);
                    FakeOutcome::Rows(rows, a || b)
                }
            },
        )
    }
    .await;
    let (rows, mut violation, mut causes) = match result {
        Ok(FakeOutcome::Deferred(value)) => {
            return Ok(scope.not_submitted(NoSubmissionReason::Deferred(value)))
        }
        Ok(FakeOutcome::Unsupported) => {
            return Ok(scope.not_submitted(NoSubmissionReason::UnsupportedShape))
        }
        Ok(FakeOutcome::Rows(rows, bad)) => (
            rows,
            bad.then(|| {
                ExecutionContractViolation::new(
                    FerrumError::backend("controlled batch returned a wrong row count"),
                    (Vec::new(), Vec::new(), Vec::new()),
                )
            }),
            Vec::new(),
        ),
        Err(error) => (Vec::new(), None, vec![error]),
    };
    let mut completed = Vec::new();
    for row in rows {
        if row.index >= work.len() {
            violation
                .get_or_insert_with(|| {
                    ExecutionContractViolation::new(
                        FerrumError::backend("controlled batch returned an unassigned row"),
                        (Vec::new(), Vec::new(), Vec::new()),
                    )
                })
                .unassigned_products
                .push(row.product);
            continue;
        }
        if let Some(error) = row.error {
            let value = violation.get_or_insert_with(|| {
                ExecutionContractViolation::new(error, (Vec::new(), Vec::new(), Vec::new()))
            });
            value
                .rejected_completions
                .push((row.index, row.executed, row.product));
            continue;
        }
        match scope.completion(row.index, row.executed, row.product) {
            Ok(value) => {
                let mut retained = state.entries.lock();
                let entry = retained
                    .get_mut(work[row.index].sequence())
                    .expect("controlled work owns its incarnation");
                entry.frontier = value.next_frontier().clone();
                entry.cache = Some(row.cache);
                completed.push(value);
            }
            Err(rejected) => {
                let (error, raw) = rejected.into_parts();
                violation
                    .get_or_insert_with(|| {
                        ExecutionContractViolation::new(error, (Vec::new(), Vec::new(), Vec::new()))
                    })
                    .rejected_completions
                    .push(raw);
            }
        }
    }
    if completed.len() == work.len() && violation.is_none() {
        return Ok(scope
            .completed(completed)
            .unwrap_or_else(|rejected| rejected.into_outcome()));
    }
    if causes.is_empty() {
        causes.push(FerrumError::backend(
            "controlled model result was not publishable",
        ));
    }
    let mut remaining = Vec::new();
    for (index, input) in work.iter().enumerate() {
        if completed.iter().any(|value| value.input_index() == index) {
            continue;
        }
        // These controlled fake operations have no outstanding asynchronous
        // work. Closing them proves quiescence, not successful model execution.
        let closed = cancel(executor, state, input.sequence())?;
        let ExecutionCloseDisposition::Ended(ended) = closed.disposition else {
            unreachable!()
        };
        remaining.push(
            scope
                .participant(
                    index,
                    ExecutionProgress::Terminal {
                        disposition: TerminalModelDisposition::UnknownAfterQuiescence,
                        executed_range: None,
                    },
                    ContinuationDisposition::Ended(ended),
                    0,
                )
                .unwrap(),
        );
    }
    if let Some(mut value) = violation {
        value.completed = completed;
        value.remaining = remaining;
        value.causes = causes;
        Ok(ExecutionBatchOutcome::ContractViolation(value))
    } else {
        Ok(scope
            .interrupted(completed, remaining, causes)
            .unwrap_or_else(|rejected| rejected.into_outcome()))
    }
}
