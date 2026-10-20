use super::*;

impl TokenWorkerState {
    /// Lost delivery proves neither zero submission nor success. Keep the
    /// original cohort identities and obtain real close observations instead
    /// of returning a retryable-looking model error after possible execution.
    pub(in crate::executor::llm_executor) fn interrupt_lost_delivery(
        &self,
        inputs: &[ExecutionWork<'_>],
        error: FerrumError,
    ) -> Result<ExecutionBatchOutcome> {
        let scope = self.issuer.batch(inputs, self.vocabulary)?;
        let mut remaining = Vec::with_capacity(inputs.len());
        let mut rejected_participants = Vec::new();
        let mut violation = None;
        let mut causes = vec![ExecutionInterruptionCause::Error(error.clone())];
        for (index, input) in inputs.iter().enumerate() {
            let close = {
                let mut sequences = self.sequences.lock();
                if let Some(entry) = sequences.live.get_mut(input.sequence()) {
                    entry.closing = true;
                    if let Some(device) = &mut entry.device {
                        device.operation_error.get_or_insert_with(|| error.clone());
                    }
                    self.observe_close_locked(entry)
                } else if let Some(ended) = input.sequence().observed_end() {
                    Ok(ExecutionCloseOutcome::ended(ended))
                } else {
                    Err(FerrumError::internal(
                        "lost delivery has no original close owner",
                    ))
                }
            };
            let (progress, continuation) = match close {
                Ok(close) => match close.disposition {
                    ExecutionCloseDisposition::Ended(ended) => (
                        ExecutionProgress::Terminal {
                            disposition: TerminalModelDisposition::UnknownAfterQuiescence,
                            executed_range: None,
                        },
                        ContinuationDisposition::Ended(ended),
                    ),
                    ExecutionCloseDisposition::Held(held) => (
                        ExecutionProgress::Unknown {
                            last_known: LastKnownExecutionStage::EnteredSubmit,
                            known_executed_range: None,
                        },
                        ContinuationDisposition::Held(held),
                    ),
                },
                Err(error) => {
                    violation.get_or_insert_with(|| error.clone());
                    causes.push(error.into());
                    continue;
                }
            };
            match scope.participant(index, progress, continuation, 0) {
                Ok(participant) => remaining.push(participant),
                Err(rejected) => {
                    let (error, participant) = rejected.into_parts();
                    violation.get_or_insert(error);
                    rejected_participants.push(participant);
                }
            }
        }
        Ok(if let Some(error) = violation {
            ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation {
                error,
                completed: Vec::new(),
                remaining,
                causes,
                rejected_completions: Vec::new(),
                rejected_participants,
                unassigned_products: Vec::new(),
            })
        } else {
            scope
                .interrupted(Vec::new(), remaining, causes)
                .unwrap_or_else(|rejected| rejected.into_outcome())
        })
    }

    pub(in crate::executor::llm_executor) fn execute_device_batch(
        &self,
        owned: &OwnedTokenBatch,
    ) -> Result<ExecutionBatchOutcome> {
        let inputs: Vec<_> = owned
            .work
            .iter()
            .map(|input| ExecutionWork::new(&input.frontier, input.work, &input.policy))
            .collect::<Result<_>>()?;
        let scope = self.issuer.batch(&inputs, self.vocabulary)?;
        let selected: Vec<_> = {
            let sequences = self.sequences.lock();
            owned
                .work
                .iter()
                .enumerate()
                .filter_map(|(index, input)| {
                    sequences
                        .live
                        .get(input.frontier.sequence())
                        .filter(|entry| {
                            !entry.closing
                                && entry.frontier == input.frontier
                                && entry
                                    .device
                                    .as_ref()
                                    .is_some_and(|device| device.claim == Some(owned.claim))
                        })
                        .map(|_| index)
                })
                .collect()
        };
        let numerical: Vec<_> = selected
            .iter()
            .map(|&index| {
                let input = &owned.work[index];
                ModelTokenInput {
                    cache_id: &input.cache_id,
                    work: match input.work {
                        TokenWork::Prefill(chunk) => ModelTokenWork::Prefill {
                            tokens: &input.prompt[chunk.range()],
                            position: chunk.tokens_processed(),
                        },
                        TokenWork::Decode { token, position } => ModelTokenWork::Decode {
                            token: token.get(),
                            position: position as u32,
                        },
                    },
                    logits_policy: &input.policy,
                    sample: !matches!(input.work, TokenWork::Prefill(chunk) if !chunk.is_final()),
                }
            })
            .collect();

        // The actual model owns the context across this whole call. No caller
        // future, registry lock or Tensor bridge participates in submission.
        let physical = if numerical.is_empty() {
            ModelTokenBatchOutcome::Serial(Vec::new())
        } else {
            lock_live_model(&self.model)?.execute_token_batch(&numerical)
        };
        let mut rows: Vec<Option<ModelTokenOutcome<ModelTokenProduct>>> =
            std::iter::repeat_with(|| None).take(inputs.len()).collect();
        let mut unassigned_products = Vec::new();
        let mut unattributed_causes: Vec<ExecutionInterruptionCause> = Vec::new();
        let mut violation = None;
        let mut cohort_capacity = false;
        let physical_rows = match physical {
            ModelTokenBatchOutcome::Serial(rows) => rows,
            ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::CapacityDeferred(deferred)) => {
                let sequences = self.sequences.lock();
                if selected.len() == inputs.len()
                    && inputs.iter().all(|input| {
                        self.retryable_capacity_frontier(&sequences, input, owned.claim)
                    })
                {
                    return Ok(scope.not_submitted(NoSubmissionReason::Deferred(deferred.into())));
                }
                // A cancelled member can make this a partial outcome even
                // though the selected physical cohort did not execute. Keep
                // its single fit rejection associated with the whole subgroup.
                cohort_capacity = true;
                selected
                    .iter()
                    .map(|_| ModelTokenOutcome::CapacityDeferred(deferred.clone()))
                    .collect()
            }
            ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Terminal(Ok(products))) => products
                .into_iter()
                .map(|product| ModelTokenOutcome::Terminal(Ok(product)))
                .collect(),
            ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::NotSubmitted(error)) => selected
                .iter()
                .map(|_| ModelTokenOutcome::NotSubmitted(error.clone()))
                .collect(),
            ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Terminal(Err(error))) => selected
                .iter()
                .map(|_| ModelTokenOutcome::Terminal(Err(error.clone())))
                .collect(),
            ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Held(error)) => selected
                .iter()
                .map(|_| ModelTokenOutcome::Held(error.clone()))
                .collect(),
        };
        let malformed_count = physical_rows.len() != selected.len();
        if malformed_count {
            violation = Some(FerrumError::internal(
                "model token batch returned a different participant count",
            ));
        }
        for (offset, row) in physical_rows.into_iter().enumerate() {
            if malformed_count {
                // These rows have no identity. A missing middle row cannot
                // be treated as a missing tail: preserve every raw product,
                // but never assign later logits to an earlier participant.
                match row {
                    ModelTokenOutcome::Terminal(Ok(product)) => unassigned_products.push(
                        product.map_or(ExecutionProduct::Intermediate, ExecutionProduct::Sampling),
                    ),
                    ModelTokenOutcome::NotSubmitted(error)
                    | ModelTokenOutcome::Terminal(Err(error))
                    | ModelTokenOutcome::Held(error) => unattributed_causes.push(error.into()),
                    ModelTokenOutcome::CapacityDeferred(deferred) => unattributed_causes
                        .push(ExecutionInterruptionCause::Deferred(deferred.into())),
                }
            } else if let Some(&index) = selected.get(offset) {
                rows[index] = Some(row);
            }
        }

        let mut completed = Vec::with_capacity(inputs.len());
        let mut remaining = Vec::new();
        let mut causes = unattributed_causes;
        let mut cohort_capacity_cause = None;
        let mut rejected_completions = Vec::new();
        let mut rejected_participants = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            let actual = match input.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 0,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            };
            // Already validated before any model call. Keep a missing range
            // conservative if an internal invariant nevertheless changes.
            let range = input.work().range().ok();
            let selected_for_execution = selected.contains(&index);
            let row = rows[index].take().unwrap_or_else(|| {
                if selected_for_execution {
                    ModelTokenOutcome::Held(FerrumError::internal(
                        "model omitted an executed participant result",
                    ))
                } else {
                    ModelTokenOutcome::NotSubmitted(FerrumError::request_validation(
                        "token participant was closed before dispatch",
                    ))
                }
            });
            let (mut progress, error, retain_operation_error) = match row {
                ModelTokenOutcome::CapacityDeferred(deferred) => {
                    let sequences = self.sequences.lock();
                    if self.retryable_capacity_frontier(&sequences, input, owned.claim) {
                        let cause_index = match cohort_capacity_cause.filter(|_| cohort_capacity) {
                            Some(index) => index,
                            None => {
                                let index = causes.len();
                                causes.push(ExecutionInterruptionCause::Deferred(deferred.into()));
                                if cohort_capacity {
                                    cohort_capacity_cause = Some(index);
                                }
                                index
                            }
                        };
                        match scope.participant(
                            index,
                            ExecutionProgress::NotSubmitted {
                                stage: NoSubmissionStage::BeforeSubmit,
                            },
                            ContinuationDisposition::Retryable(input.expected_frontier().clone()),
                            cause_index,
                        ) {
                            Ok(participant) => remaining.push(participant),
                            Err(rejected) => {
                                let (error, participant) = rejected.into_parts();
                                violation.get_or_insert(error);
                                rejected_participants.push(participant);
                            }
                        }
                        continue;
                    }
                    // Cancellation after the fit check does not authorize a
                    // retry, nor turn that ordinary cancellation into a device
                    // error. An unexpected live-frontier mismatch remains an
                    // actual contract failure with its original owner.
                    let cancelled = input.sequence().observed_end().is_some()
                        || sequences
                            .live
                            .get(input.sequence())
                            .is_some_and(|entry| entry.closing);
                    (
                        ExecutionProgress::NotSubmitted {
                            stage: NoSubmissionStage::BeforeSubmit,
                        },
                        FerrumError::request_validation(
                            "capacity-deferred participant no longer has a retryable frontier",
                        ),
                        !cancelled,
                    )
                }
                ModelTokenOutcome::Terminal(Ok(product)) => {
                    let product =
                        product.map_or(ExecutionProduct::Intermediate, ExecutionProduct::Sampling);
                    match scope.completion(index, actual, product) {
                        Ok(completion) => {
                            let mut sequences = self.sequences.lock();
                            if let Some(entry) =
                                sequences.live.get_mut(input.sequence()).filter(|entry| {
                                    !entry.closing
                                        && entry
                                            .device
                                            .as_ref()
                                            .is_some_and(|device| device.claim == Some(owned.claim))
                                })
                            {
                                entry.frontier = completion.next_frontier().clone();
                                completed.push(completion);
                                continue;
                            }
                            // Actual execution succeeded, but this incarnation
                            // was cancelled while the numerical call ran.
                            (
                                ExecutionProgress::Terminal {
                                    disposition: TerminalModelDisposition::Successful,
                                    executed_range: range,
                                },
                                FerrumError::request_validation(
                                    "token participant cancelled during execution",
                                ),
                                false,
                            )
                        }
                        Err(rejected) => {
                            let (error, product) = rejected.into_parts();
                            violation.get_or_insert_with(|| error.clone());
                            rejected_completions.push(product);
                            (
                                ExecutionProgress::Terminal {
                                    disposition: TerminalModelDisposition::Successful,
                                    executed_range: range,
                                },
                                error,
                                true,
                            )
                        }
                    }
                }
                ModelTokenOutcome::Terminal(Err(error)) => (
                    ExecutionProgress::Terminal {
                        disposition: TerminalModelDisposition::Failed,
                        executed_range: None,
                    },
                    error,
                    true,
                ),
                ModelTokenOutcome::NotSubmitted(error) => (
                    ExecutionProgress::NotSubmitted {
                        stage: NoSubmissionStage::BeforeSubmit,
                    },
                    error,
                    selected_for_execution,
                ),
                ModelTokenOutcome::Held(error) => (
                    ExecutionProgress::Unknown {
                        last_known: LastKnownExecutionStage::EnteredSubmit,
                        known_executed_range: None,
                    },
                    error,
                    true,
                ),
            };
            let cause_index = causes.len();
            causes.push(error.clone().into());
            let close = {
                let mut sequences = self.sequences.lock();
                if let Some(entry) = sequences.live.get_mut(input.sequence()) {
                    entry.closing = true;
                    if let Some(device) = &mut entry.device {
                        // Cancellation explains why a successful/undispatched
                        // product cannot continue. It is not a device failure.
                        // Genuine model, validation and unknown-work errors
                        // remain independent of the eventual terminal fact.
                        if retain_operation_error {
                            device.operation_error.get_or_insert(error);
                        }
                        device.needs_recovery |=
                            matches!(progress, ExecutionProgress::Unknown { .. });
                    }
                    self.observe_close_locked(entry)
                } else if let Some(ended) = input.sequence().observed_end() {
                    Ok(ExecutionCloseOutcome::ended(ended))
                } else {
                    Err(FerrumError::internal(
                        "executed participant lost its original close owner",
                    ))
                }
            };
            let continuation = match close {
                Ok(close) => match close.disposition {
                    ExecutionCloseDisposition::Ended(ended) => {
                        if matches!(progress, ExecutionProgress::Unknown { .. }) {
                            progress = ExecutionProgress::Terminal {
                                disposition: TerminalModelDisposition::UnknownAfterQuiescence,
                                executed_range: None,
                            };
                        }
                        ContinuationDisposition::Ended(ended)
                    }
                    ExecutionCloseDisposition::Held(held) => ContinuationDisposition::Held(held),
                },
                Err(error) => {
                    violation.get_or_insert(error.clone());
                    causes.push(error.into());
                    continue;
                }
            };
            match scope.participant(index, progress, continuation, cause_index) {
                Ok(participant) => remaining.push(participant),
                Err(rejected) => {
                    let (error, participant) = rejected.into_parts();
                    violation.get_or_insert(error);
                    rejected_participants.push(participant);
                }
            }
        }
        Ok(if let Some(error) = violation {
            ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation {
                error,
                completed,
                remaining,
                causes,
                rejected_completions,
                rejected_participants,
                unassigned_products,
            })
        } else if remaining.is_empty() {
            scope
                .completed(completed)
                .unwrap_or_else(|rejected| rejected.into_outcome())
        } else {
            scope
                .interrupted(completed, remaining, causes)
                .unwrap_or_else(|rejected| rejected.into_outcome())
        })
    }

    fn retryable_capacity_frontier(
        &self,
        sequences: &TokenSequences,
        input: &ExecutionWork<'_>,
        claim: u64,
    ) -> bool {
        input.sequence().observed_end().is_none()
            && sequences.live.get(input.sequence()).is_some_and(|entry| {
                entry.validate_work(input).is_ok()
                    && entry.device.as_ref().is_some_and(|device| {
                        device.claim == Some(claim)
                            && device.prepared
                            && device.published
                            && !device.needs_recovery
                    })
            })
    }
}
