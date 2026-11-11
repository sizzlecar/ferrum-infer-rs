//! Submit one original native wave and consume each result by its input index.
use super::*;

/// The original native call's result, independent of per-request extraction or
/// close results. An execution error alone makes no submission claim.
#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::image_generation::p_h3::expansion) enum ExpansionWaveOutcome {
    PreparationFailed,
    ExecutionError,
    Completed,
    Interrupted,
    NotSubmitted,
    ContractViolation,
}

pub(in crate::image_generation::p_h3::expansion) struct ExpansionWave {
    pub(in crate::image_generation::p_h3::expansion) steps:
        Vec<Result<ExpansionStep, ExpansionExecutionFailure>>,
    pub(in crate::image_generation::p_h3::expansion) input_frontiers: Vec<ExecutionFrontier>,
    pub(in crate::image_generation::p_h3::expansion) retained: Option<ExecutionBatchOutcome>,
    pub(in crate::image_generation::p_h3::expansion) fatal: Option<FerrumError>,
    pub(in crate::image_generation::p_h3::expansion) native_outcome: ExpansionWaveOutcome,
}

enum Observation {
    Completed {
        frontier: ExecutionFrontier,
        token: ferrum_types::Result<Option<TokenId>>,
    },
    Failed {
        error: FerrumError,
        ended: Option<ExecutionEnded>,
    },
}

impl ExpansionCursor {
    fn planned_work(&self) -> ferrum_types::Result<TokenWork> {
        let consumed = self.frontier.consumed_tokens();
        if consumed < self.prompt_tokens {
            PrefillChunk::new(
                consumed,
                self.prefill_chunk_limit.min(self.prompt_tokens - consumed),
                self.prompt_tokens,
            )
            .map(TokenWork::Prefill)
        } else {
            self.history
                .ids()
                .last()
                .copied()
                .map(|token| TokenWork::Decode {
                    token: TokenId::new(token),
                    position: consumed,
                })
                .ok_or_else(|| {
                    FerrumError::backend("expansion decode has no actual previous token")
                })
        }
    }

    fn failed(
        self,
        scope: &ExecutorScope,
        reason: FinishReason,
        error: FerrumError,
    ) -> ExpansionExecutionFailure {
        active_failure(scope, self.history, self.frontier, reason, error, None)
    }

    async fn consume_completed(
        mut self,
        scope: &ExecutorScope,
        frontier: ExecutionFrontier,
        token: ferrum_types::Result<Option<TokenId>>,
    ) -> Result<ExpansionStep, ExpansionExecutionFailure> {
        self.frontier = frontier;
        let token = match token {
            Ok(token) => token,
            Err(error) => return Err(self.failed(scope, FinishReason::Error, error)),
        };
        if let Some(token) = token {
            if let Err(error) = self.history.push(token) {
                return Err(self.failed(scope, FinishReason::Error, error));
            }
            let reason = if token.get() == self.generation.eos_token_id {
                Some(FinishReason::EOS)
            } else if self.history.ids().len() == self.generation.max_new_tokens as usize {
                Some(FinishReason::Length)
            } else {
                None
            };
            if let Some(reason) = reason {
                return finish(
                    scope,
                    self.history,
                    self.frontier,
                    self.prompt_tokens,
                    reason,
                )
                .await
                .map(ExpansionStep::Complete);
            }
        }
        Ok(ExpansionStep::Continue(self))
    }
}

fn observations(
    outcome: &ExecutionBatchOutcome,
    inputs: &[ExecutionWork<'_>],
    vocabulary: usize,
) -> ferrum_types::Result<Vec<Observation>> {
    let mut rows: Vec<Option<Observation>> = (0..inputs.len()).map(|_| None).collect();
    let (completed, remaining): (&[ExecutionCompletion], &[InterruptedParticipant]) = match outcome
    {
        ExecutionBatchOutcome::Completed(batch) => (batch.completions(), &[]),
        ExecutionBatchOutcome::Interrupted(batch) => (batch.completed(), batch.remaining()),
        ExecutionBatchOutcome::NotSubmitted(_) => {
            return Ok(inputs
                .iter()
                .enumerate()
                .map(|(index, _)| Observation::Failed {
                    error: incomplete_participant_error(outcome, index),
                    ended: None,
                })
                .collect())
        }
        ExecutionBatchOutcome::ContractViolation(violation) => return Err(violation.error.clone()),
    };
    for completion in completed {
        let index = completion.input_index();
        let input = inputs.get(index).ok_or_else(|| {
            FerrumError::backend("expansion completion has an unknown input index")
        })?;
        completion.validate_for(index, input, vocabulary)?;
        if rows[index].is_some() {
            return Err(FerrumError::backend(
                "expansion wave repeats a completion input",
            ));
        }
        rows[index] = Some(Observation::Completed {
            frontier: completion.next_frontier().clone(),
            token: selected(completion.product()),
        });
    }
    for participant in remaining {
        let index = participant.input_index();
        let input = inputs.get(index).ok_or_else(|| {
            FerrumError::backend("expansion interruption has an unknown input index")
        })?;
        let ExecutionBatchOutcome::Interrupted(batch) = outcome else {
            unreachable!()
        };
        if rows[index].is_some()
            || participant.sequence() != input.sequence()
            || batch.causes().get(participant.cause_index()).is_none()
        {
            return Err(FerrumError::backend(
                "expansion interruption conflicts with its original input or cause",
            ));
        }
        let ended = match participant.continuation() {
            ContinuationDisposition::Retryable(frontier)
                if frontier == input.expected_frontier() =>
            {
                None
            }
            ContinuationDisposition::Ended(ended) if ended.sequence() == input.sequence() => {
                Some(ended.clone())
            }
            ContinuationDisposition::Held(held) if held.key().sequence() == input.sequence() => {
                None
            }
            _ => {
                return Err(FerrumError::backend(
                    "expansion interruption has a foreign continuation",
                ))
            }
        };
        rows[index] = Some(Observation::Failed {
            error: incomplete_participant_error(outcome, index),
            ended,
        });
    }
    rows.into_iter()
        .map(|row| {
            row.ok_or_else(|| {
                FerrumError::backend("expansion wave omitted an original participant")
            })
        })
        .collect()
}

pub(in crate::image_generation::p_h3::expansion) async fn advance_batch(
    scope: &ExecutorScope,
    cursors: Vec<ExpansionCursor>,
) -> ExpansionWave {
    let input_frontiers: Vec<_> = cursors
        .iter()
        .map(|cursor| cursor.frontier.clone())
        .collect();
    let policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let inputs: ferrum_types::Result<Vec<_>> = cursors
        .iter()
        .zip(&input_frontiers)
        .map(|(cursor, frontier)| ExecutionWork::new(frontier, cursor.planned_work()?, &policy))
        .collect();
    let inputs = match inputs {
        Ok(inputs) => inputs,
        Err(error) => {
            return ExpansionWave {
                steps: cursors
                    .into_iter()
                    .map(|cursor| Err(cursor.failed(scope, FinishReason::Error, error.clone())))
                    .collect(),
                input_frontiers,
                retained: None,
                fatal: Some(error),
                native_outcome: ExpansionWaveOutcome::PreparationFailed,
            };
        }
    };
    // Never cancel this original future. Native partial/unknown progress must
    // return as its original typed outcome before another wave is considered.
    let outcome = match scope.execute_tokens(&inputs).await {
        Ok(outcome) => outcome,
        Err(error) => {
            drop(inputs);
            return ExpansionWave {
                steps: cursors
                    .into_iter()
                    .map(|cursor| Err(cursor.failed(scope, FinishReason::Error, error.clone())))
                    .collect(),
                input_frontiers,
                retained: None,
                fatal: None,
                native_outcome: ExpansionWaveOutcome::ExecutionError,
            };
        }
    };
    let native_outcome = match &outcome {
        ExecutionBatchOutcome::Completed(_) => ExpansionWaveOutcome::Completed,
        ExecutionBatchOutcome::Interrupted(_) => ExpansionWaveOutcome::Interrupted,
        ExecutionBatchOutcome::NotSubmitted(_) => ExpansionWaveOutcome::NotSubmitted,
        ExecutionBatchOutcome::ContractViolation(_) => ExpansionWaveOutcome::ContractViolation,
    };
    let observations = observations(&outcome, &inputs, scope.info().vocab_size);
    drop(inputs);
    let observations = match observations {
        Ok(observations) => observations,
        Err(error) => {
            return ExpansionWave {
                steps: cursors
                    .into_iter()
                    .map(|cursor| Err(cursor.failed(scope, FinishReason::Error, error.clone())))
                    .collect(),
                input_frontiers,
                retained: Some(outcome),
                fatal: Some(error),
                native_outcome,
            }
        }
    };
    let mut steps = Vec::with_capacity(cursors.len());
    for (cursor, observation) in cursors.into_iter().zip(observations) {
        steps.push(match observation {
            Observation::Completed { frontier, token } => {
                cursor.consume_completed(scope, frontier, token).await
            }
            Observation::Failed {
                error,
                ended: Some(ended),
            } => {
                let mut failure = ExpansionExecutionFailure::new(
                    error,
                    FinishReason::Error,
                    Some(cursor.history),
                );
                failure.admission_probe = AdmissionProbe::Entered;
                failure.frontier = Some(cursor.frontier);
                failure.close = Some(ExecutionCloseOutcome::ended(ended));
                Err(failure)
            }
            Observation::Failed { error, ended: None } => {
                Err(cursor.failed(scope, FinishReason::Error, error))
            }
        });
    }
    let retain =
        steps.iter().any(Result::is_err) || !matches!(outcome, ExecutionBatchOutcome::Completed(_));
    ExpansionWave {
        steps,
        input_frontiers,
        retained: retain.then_some(outcome),
        fatal: None,
        native_outcome,
    }
}
