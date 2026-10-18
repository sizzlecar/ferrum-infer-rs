//! Controlled host-only protocol fixtures. Hooks describe fake work and faults;
//! the sole identity/frontier table below is test plumbing, not a second driver.
use super::*;
use ferrum_execution::token_execution::*;
#[path = "controlled_token_fixture/hooks.rs"]
mod hooks;

#[derive(Clone)]
struct Entry {
    request_id: RequestId,
    prompt: Arc<[TokenId]>,
    maximum_tokens: usize,
    frontier: ExecutionFrontier,
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
        entry.frontier = self.issuer.frontier(sequence, tokens)?;
        Ok(entry.frontier.clone())
    }
}

pub(super) struct FixtureInput<'a, 'work> {
    pub index: usize,
    pub request_id: &'a RequestId,
    pub work: &'a ExecutionWork<'work>,
}

pub(super) struct Row {
    pub index: usize,
    pub sequence: ExecutorSequence,
    pub executed: ExecutedTokenWork,
    pub product: ExecutionProduct,
}

impl FixtureInput<'_, '_> {
    pub fn row(&self, executed: ExecutedTokenWork, token: usize, vocabulary: usize) -> Row {
        let product = if matches!(executed, ExecutedTokenWork::Prefill { chunk, .. } if !chunk.is_final())
        {
            ExecutionProduct::Intermediate
        } else {
            let mut logits = vec![0.0; vocabulary];
            logits[token] = 1.0;
            ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits))
        };
        Row {
            index: self.index,
            sequence: self.work.sequence().clone(),
            executed,
            product,
        }
    }

    pub fn exact(&self, token: usize, vocabulary: usize) -> Row {
        self.row(
            match self.work.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            },
            token,
            vocabulary,
        )
    }
}

pub(super) enum FixtureOutcome {
    Rows(Vec<Row>),
    Deferred(ferrum_execution::model_executor::ExecutorExecutionDeferral),
    Unsupported,
}

pub(super) trait ControlledHooks: ModelExecutor {
    fn work(&self, inputs: &[FixtureInput<'_, '_>]) -> Result<FixtureOutcome>;
    fn closed(&self, _request: &RequestId, _reason: ExecutionEndReason) -> Result<()> {
        Ok(())
    }
}

pub(super) fn admit(
    executor: &impl ModelExecutor,
    state: &ControlledTokenState,
    input: ExecutionAdmission<'_>,
) -> Result<TokenAdmissionOutcome> {
    input.validate()?;
    Ok(match executor.try_admit_prefill(input.context)? {
        ExecutorPrefillAdmissionDecision::Admitted(_) => {
            let sequence = state.issuer.issue_sequence()?;
            let frontier = state.issuer.frontier(&sequence, 0)?;
            let entry = Entry {
                request_id: input.context.request_id.clone(),
                prompt: input.context.input_tokens.into(),
                maximum_tokens: input.context.maximum_sequence_tokens,
                frontier: frontier.clone(),
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
    executor: &impl ControlledHooks,
    state: &ControlledTokenState,
    sequence: &ExecutorSequence,
) -> Result<ExecutionCloseOutcome> {
    state.issuer.frontier(sequence, 0)?;
    if let Some(entry) = state.entries.lock().remove(sequence) {
        executor.cancel_prefill_admission(&entry.request_id);
        if entry.frontier.consumed_tokens() > 0 {
            executor.closed(&entry.request_id, ExecutionEndReason::Cancelled)?;
        }
        return Ok(ExecutionCloseOutcome::ended(state.issuer.ended(sequence)?));
    }
    sequence
        .observed_end()
        .map(ExecutionCloseOutcome::ended)
        .ok_or_else(|| FerrumError::internal("fixture close has no exact retained incarnation"))
}

pub(super) async fn finish(
    executor: &impl ControlledHooks,
    state: &ControlledTokenState,
    sequence: &ExecutorSequence,
    reason: ExecutionEndReason,
) -> Result<ExecutionCloseOutcome> {
    if reason == ExecutionEndReason::Cancelled {
        return cancel(executor, state, sequence);
    }
    state.issuer.frontier(sequence, 0)?;
    let Some(entry) = state.entries.lock().get(sequence).cloned() else {
        return cancel(executor, state, sequence);
    };
    if entry.frontier.consumed_tokens() > 0 {
        executor.closed(&entry.request_id, reason)?;
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
    executor.write_execution_capacity_release_sources(
        &ExecutorExecutionCapacityPreemption::new(
            entry.request_id.clone(),
            entry.request_id.to_string(),
        )?,
        sources,
    )
}

pub(super) async fn execute(
    executor: &impl ControlledHooks,
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
    let inputs = work
        .iter()
        .zip(&entries)
        .enumerate()
        .map(|(index, (work, entry))| FixtureInput {
            index,
            request_id: &entry.request_id,
            work,
        })
        .collect::<Vec<_>>();
    let (rows, mut violation, mut causes) = match executor.work(&inputs) {
        Ok(FixtureOutcome::Deferred(value)) => {
            return Ok(scope.not_submitted(NoSubmissionReason::Deferred(value)))
        }
        Ok(FixtureOutcome::Unsupported) => {
            return Ok(scope.not_submitted(NoSubmissionReason::UnsupportedShape))
        }
        Ok(FixtureOutcome::Rows(rows)) => {
            let bad = rows.len() != work.len();
            (
                rows,
                bad.then(|| {
                    ExecutionContractViolation::new(
                        FerrumError::backend("controlled batch returned a wrong row count"),
                        (Vec::new(), Vec::new(), Vec::new()),
                    )
                }),
                Vec::new(),
            )
        }
        Err(error) => (
            Vec::new(),
            None,
            vec![ExecutionInterruptionCause::Error(error)],
        ),
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
        if row.sequence != *work[row.index].sequence() {
            let error = FerrumError::backend("controlled work returned a foreign incarnation");
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
        causes.push(FerrumError::backend("controlled model result was not publishable").into());
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
