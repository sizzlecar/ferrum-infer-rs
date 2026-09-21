//! Product ownership for one pre-submitted decode. A cached row is an
//! unconsumed model result, never a token already published by the engine.
use super::*;

pub(super) struct PreparedDecodeSuccessor<R: DeviceRuntime> {
    pub(super) step: Arc<StepResourceLease<R>>,
    pub(super) wave: PreparedStepSubmissionWave<R>,
    pub(super) spans: Vec<TokenSpanWork>,
}

pub(super) struct DispatchedDecodeSuccessor<R: DeviceRuntime> {
    pub(super) step: Arc<StepResourceLease<R>>,
    pub(super) outcome: DispatchOutcome<R>,
    pub(super) readbacks: CompletionReadbackBatchRequest,
}

pub(super) enum DecodePairParent<R: DeviceRuntime> {
    Submitted {
        completion: CompletionHandle<R>,
        readbacks: CompletionReadbackBatchRequest,
        step: Arc<StepResourceLease<R>>,
    },
    Observed(pending_decode::ObservedPairParent<R>),
}

impl<R: DeviceRuntime> DecodePairParent<R> {
    pub(super) fn submit_pair<F>(
        self,
        reservation: VNextCompletionReservation<'_>,
        child: pending_decode::SubmittedPairWave<R>,
        reaper: Arc<CompletionReaper<R>>,
        observer: F,
    ) -> (
        pending_decode::PendingParentReadback<R>,
        Arc<pending_decode::PendingDecodeCohort>,
    )
    where
        F: FnOnce(&Result<CompletionReadbackBatchReceipt>) + Send + 'static,
    {
        match self {
            Self::Submitted {
                completion,
                readbacks,
                step,
            } => pending_decode::submit_pair_with_observer(
                reservation,
                pending_decode::SubmittedPairWave::new(completion, readbacks, step),
                child,
                reaper,
                observer,
            ),
            Self::Observed(parent) => pending_decode::submit_pair_with_observed_parent(
                reservation,
                parent,
                child,
                reaper,
                observer,
            ),
        }
    }

    pub(super) async fn fail(
        self,
        reservation: VNextCompletionReservation<'_>,
        child: Option<(DispatchOutcome<R>, Arc<StepResourceLease<R>>)>,
        reaper: Arc<CompletionReaper<R>>,
        message: String,
    ) -> FerrumError {
        let mut retained = FailedDecodePair {
            parent: Some(self),
            child,
            reaper,
        };
        match reservation
            .submit(VNextCompletionTaskKind::PostSubmitDrain, move || {
                retained.drain()
            })
            .wait()
            .await
        {
            Ok(Ok(())) => FerrumError::backend(message),
            Ok(Err(error)) => FerrumError::backend(format!("{message}; {error}")),
            Err(error) => {
                FerrumError::backend(format!("{message}; cleanup worker failed: {error}"))
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct PendingDecodeRow {
    pub(super) cohort: Arc<pending_decode::PendingDecodeCohort>,
    pub(super) row: usize,
    pub(super) expected_cache_tokens: usize,
    pub(super) expected_input_token: u32,
}

pub(super) enum ObservedVNextStep<R: DeviceRuntime> {
    Ordinary(Arc<StepResourceLease<R>>),
    Paired(pending_decode::ParentRetirementGuard<R>),
}

/// A failed child dispatch cannot abandon its already submitted parent. This
/// owner is transferred to the reservation acquired before either submission.
struct FailedDecodePair<R: DeviceRuntime> {
    parent: Option<DecodePairParent<R>>,
    child: Option<(DispatchOutcome<R>, Arc<StepResourceLease<R>>)>,
    reaper: Arc<CompletionReaper<R>>,
}

impl<R: DeviceRuntime> FailedDecodePair<R> {
    fn drain_handle(&self, handle: &CompletionHandle<R>) -> Result<()> {
        if matches!(handle.wait(), Ok(CompletionObservation::Terminal(_))) {
            return Ok(());
        }
        match self.reaper.recover_slot_by_draining_lane(handle.slot_id()) {
            Ok(CompletionRecoveryOutcome::Drained(_)) => Ok(()),
            Ok(CompletionRecoveryOutcome::Quarantined(_)) => Err(FerrumError::backend(
                "failed decode pair remains quarantined",
            )),
            Err(error) => Err(FerrumError::backend(format!(
                "failed decode pair drain: {error}"
            ))),
        }
    }

    fn drain(&mut self) -> Result<()> {
        let mut errors = Vec::new();
        if let Some(parent) = self.parent.take() {
            match parent {
                DecodePairParent::Submitted {
                    completion, step, ..
                } => {
                    if let Err(error) = self.drain_handle(&completion) {
                        errors.push(error.to_string());
                    }
                    if let Err(failure) = step.try_abort() {
                        errors.push(failure.error().to_string());
                        drop(failure.into_step());
                    }
                }
                DecodePairParent::Observed(parent) => {
                    if let Err(error) = parent.abort() {
                        errors.push(error.to_string());
                    }
                }
            }
        }
        if let Some((outcome, step)) = self.child.take() {
            let result = match outcome {
                DispatchOutcome::Submitted { completion, .. }
                | DispatchOutcome::PostSubmitContract { completion, .. } => {
                    self.drain_handle(&completion)
                }
                DispatchOutcome::SubmissionIndeterminate { recovery, .. } => {
                    match recovery.recover_by_draining_lane() {
                        Ok(CompletionRecoveryOutcome::Drained(_)) => Ok(()),
                        Ok(CompletionRecoveryOutcome::Quarantined(_)) => Err(FerrumError::backend(
                            "successor submission remains quarantined",
                        )),
                        Err(error) => Err(FerrumError::backend(error.to_string())),
                    }
                }
                DispatchOutcome::QuiescentFailure(_) => Ok(()),
            };
            if let Err(error) = result {
                errors.push(error.to_string());
            }
            if let Err(failure) = step.try_abort() {
                errors.push(failure.error().to_string());
                drop(failure.into_step());
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(FerrumError::backend(errors.join("; ")))
        }
    }
}

impl<R: DeviceRuntime> Drop for FailedDecodePair<R> {
    fn drop(&mut self) {
        let _ = self.drain();
    }
}

impl<R: DeviceRuntime> ObservedVNextStep<R> {
    pub(super) fn retire_normal(self) -> Result<StepRetirementReceipt> {
        match self {
            Self::Ordinary(step) => step.try_retire_normal().map_err(|failure| {
                FerrumError::backend(format!("vNext step retirement failed: {}", failure.error()))
            }),
            Self::Paired(guard) => guard.retire_normal(),
        }
    }

    pub(super) fn abort(self) -> Result<StepRetirementReceipt> {
        match self {
            Self::Ordinary(step) => step.try_abort().map_err(|failure| {
                FerrumError::backend(format!("vNext step abort failed: {}", failure.error()))
            }),
            Self::Paired(guard) => guard.abort(),
        }
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn successor_needs_serial_warmup(
        &self,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<bool> {
        if !self.on_demand_reusable_execution_enabled() {
            return Ok(false);
        }
        let Some(id) = OperationDispatch::reusable_execution_program_id_for_wave(
            self.providers.providers(),
            &self.resolved_plan,
            wave,
            &self.lane,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?
        else {
            return Ok(false);
        };
        let catalog = self.reusable_execution_catalog.read();
        let Some(program) = catalog
            .as_ref()
            .filter(|catalog| catalog.lane_epoch == self.lane.reusable_execution_epoch())
            .and_then(|catalog| catalog.programs.get(&id))
        else {
            return Ok(true);
        };
        // Permanent gaps (unsupported capture, memory limits) remain adaptive.
        // Only a missing program or an explicitly transient gap needs a new
        // quiescent warmup of these exact Step and Invocation slots.
        Ok(program.gaps().iter().any(|gap| {
            matches!(
                gap.reason(),
                DeviceReusableExecutionProgramGapReason::WarmupRequired
                    | DeviceReusableExecutionProgramGapReason::QuiescenceDeferred
            )
        }))
    }

    pub(super) fn dispatch_decode_successor(
        &self,
        parent_participants: &[VNextExecutionParticipant<'_, R>],
        prepared: PreparedDecodeSuccessor<R>,
    ) -> Result<DispatchedDecodeSuccessor<R>> {
        let PreparedDecodeSuccessor { step, wave, spans } = prepared;
        // The child's sampling policy is deliberately evaluated on the next
        // host turn, after the parent's token updated UTF-8 and stop/history
        // state. Only its model forward is authorized by the grant.
        let role = VNextParticipantOutputRole::Decode(LogitsReturnPolicy::FullLogits);
        let participants = parent_participants
            .iter()
            .zip(&spans)
            .map(|(parent, span)| VNextExecutionParticipant {
                sequence: parent.sequence,
                tokens: &[],
                span,
                output_role: &role,
            })
            .collect::<Vec<_>>();
        let readbacks = match self.prepare_terminal_readbacks(
            &participants,
            None,
            VNextProductOutputMode::FullLogits,
        ) {
            Ok(VNextTerminalReadbacks::Batch(request)) => request,
            other => {
                drop(wave);
                self.rollback_unsubmitted_step(step, "successor readback preparation")?;
                return Err(match other {
                    Err(error) => error,
                    _ => FerrumError::internal("successor requires one product readback batch"),
                });
            }
        };
        // No residency claim is published ahead of child completion. Upload
        // the neutral masks and invalidate stale slot evidence before reuse.
        let mut masks = VNextProductTokenMaskResidencyTransaction::prepare(
            &self.product_token_mask_residency,
            None,
            participants.iter().map(|_| {
                VNextProductTokenMaskContent::from_policy(
                    None,
                    VNextProductOutputMode::FullLogits,
                    self.io.output_elements,
                )
            }),
        );
        let outcome = self.dispatch_participant_wave(
            &participants,
            wave,
            VNextExecutionWaveKind::Decode,
            VNextProductOutputMode::FullLogits,
            masks.plans(),
            false,
        );
        masks.invalidate_before_slot_release();
        Ok(DispatchedDecodeSuccessor {
            step,
            outcome,
            readbacks,
        })
    }

    /// Admission is optional: pressure on the extra Step must never prevent
    /// the already submitted parent from making ordinary decode progress.
    pub(super) fn prepare_decode_successor(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        completion: &CompletionHandle<R>,
        sources: &CompletionReadbackBatchRequest,
        parent_step: &StepResourceLease<R>,
    ) -> Result<Option<PreparedDecodeSuccessor<R>>> {
        let predecessor = Arc::new(
            completion
                .take_submitted_predecessor()
                .map_err(|error| FerrumError::backend(error.to_string()))?,
        );
        let work = predecessor
            .bind_next_token_work(sources.clone())
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let spans = work
            .participant_work()
            .iter()
            .map(|participant| participant.token_span().clone())
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(
            participants
                .iter()
                .map(|participant| Arc::clone(&participant.sequence.session))
                .collect(),
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let mut request = StepResourceAdmissionRequest::new(
            Arc::new(work),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        if let Some(bucket) = parent_step.reusable_execution_bucket() {
            request = request.with_reusable_execution_bucket(bucket.bucket_id().clone());
        }
        let mut attempts = 0;
        let step = loop {
            let decision = batch
                .try_begin_successor_step(request.clone(), &self.lane, Arc::clone(&predecessor))
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            match decision {
                StepResourceAdmissionDecision::Admitted(step) => break step,
                StepResourceAdmissionDecision::BackingDeferred(deferred)
                    if attempts < MAX_BACKING_MAINTENANCE_ATTEMPTS =>
                {
                    attempts += 1;
                    match deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?
                    {
                        DynamicDeferredMaintenanceOutcome::Maintained(_)
                        | DynamicDeferredMaintenanceOutcome::RetryAdmission { .. } => {}
                        DynamicDeferredMaintenanceOutcome::WaitForRelease { .. } => {
                            return Ok(None)
                        }
                    }
                }
                StepResourceAdmissionDecision::Deferred(_)
                | StepResourceAdmissionDecision::BackingDeferred(_)
                | StepResourceAdmissionDecision::PermanentRejected(_) => return Ok(None),
            }
        };
        let sequences = participants
            .iter()
            .map(|participant| Arc::clone(participant.sequence))
            .collect::<Vec<_>>();
        let prepared = (|| -> Result<Option<PreparedStepSubmissionWave<R>>> {
            let wave = match self.prepare_wave_for_spans_with_capacity(
                &step,
                &sequences,
                &spans,
                VNextExecutionWaveKind::Decode,
            )? {
                VNextExecutionCapacityDecision::Ready(wave) => wave,
                VNextExecutionCapacityDecision::Deferred(_)
                | VNextExecutionCapacityDecision::RequestStateDeferred(_) => return Ok(None),
            };
            let forwards = sources
                .requests()
                .iter()
                .zip(&spans)
                .map(|(source, span)| {
                    let offset = span
                        .immediate_token_range()
                        .start
                        .checked_mul(ElementType::U32.size_bytes())
                        .ok_or_else(|| FerrumError::backend("successor input offset overflow"))?;
                    predecessor
                        .forward_token(
                            source.clone(),
                            self.io.input_node_id.clone(),
                            self.io.input_ordinal,
                            offset,
                        )
                        .map_err(|error| FerrumError::backend(error.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            wave.with_forwarded_inputs(forwards)
                .map(Some)
                .map_err(|error| FerrumError::backend(error.to_string()))
        })();
        match prepared {
            Ok(Some(wave)) => Ok(Some(PreparedDecodeSuccessor { step, wave, spans })),
            other => {
                self.rollback_unsubmitted_step(step, "unused decode successor")?;
                other.map(|_| None)
            }
        }
    }

    pub(super) fn decode_lookahead_eligible(&self, sequences: &[Arc<VNextSequence<R>>]) -> bool {
        !sequences.is_empty()
            && !self.prefix_restore_enabled()
            && self.checkpoint_capture.is_none()
            && self.diagnostic_fault.is_none()
            && sequences.iter().all(|sequence| {
                sequence.events.is_none()
                    && sequence.active.load(Ordering::Acquire)
                    && sequence.pending_decode.lock().is_none()
            })
    }

    pub(super) fn retain_successor_rows(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        outputs: &[ExecutorSamplingOutput],
        cohort: Arc<pending_decode::PendingDecodeCohort>,
    ) -> Result<()> {
        if participants.len() != outputs.len() || cohort.row_count() != participants.len() {
            return Err(FerrumError::internal("lookahead result changed its cohort"));
        }
        let rows = participants
            .iter()
            .zip(outputs)
            .enumerate()
            .map(|(row, (participant, output))| {
                let ExecutorSamplingOutput::GreedyToken(token) = output else {
                    return Err(FerrumError::internal(
                        "lookahead input requires its parent's actual selected token",
                    ));
                };
                if participant.sequence.pending_decode.lock().is_some() {
                    return Err(FerrumError::internal(
                        "lookahead would overwrite an unconsumed result",
                    ));
                }
                Ok(PendingDecodeRow {
                    cohort: Arc::clone(&cohort),
                    row,
                    expected_cache_tokens: usize::try_from(participant.span.full_input_tokens())
                        .map_err(|_| {
                            FerrumError::backend("lookahead input frontier exceeds usize")
                        })?,
                    expected_input_token: token.get(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // The caller holds every sequence's operation guard. Validate the
        // entire cohort before installing any row, preserving caller ordering.
        for (participant, row) in participants.iter().zip(rows) {
            let mut pending = participant.sequence.pending_decode.lock();
            // Synchronous release does not acquire operation. Check under the
            // same pending lock that release uses so it cannot leave a newly
            // installed row behind after marking the sequence inactive.
            if participant.sequence.active.load(Ordering::Acquire) {
                *pending = Some(row);
            } else {
                row.cohort.discard_row(row.row)?;
            }
        }
        Ok(())
    }

    pub(super) fn abort_observed_step(
        &self,
        step: ObservedVNextStep<R>,
        message: impl Into<String>,
    ) -> FerrumError {
        let message = message.into();
        self.metrics.record_failure(message.clone());
        match step.abort() {
            Ok(_) => FerrumError::backend(message),
            Err(error) => FerrumError::backend(format!("{message}; {error}")),
        }
    }
}
