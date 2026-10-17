//! Product integration for a cold decode program and fresh per-wave authority.
use super::*;

pub(super) enum ResidentCache<R: DeviceRuntime> {
    Bootstrap(Arc<ResidentDecodeReservation<R>>),
    Ready(Arc<ResidentDecodeProgram<R>>),
}

impl<R: DeviceRuntime> ResidentCache<R> {
    fn reservation(&self) -> &Arc<ResidentDecodeReservation<R>> {
        match self {
            Self::Bootstrap(reservation) => reservation,
            Self::Ready(program) => program.reservation(),
        }
    }
}

#[derive(Default)]
pub(super) struct ResidentMetrics {
    pub(super) submitted: AtomicU64,
    pub(super) bootstraps: AtomicU64,
    pub(super) installations: AtomicU64,
    pub(super) unavailable: AtomicU64,
}

pub(super) enum PreparedDispatch<R: DeviceRuntime> {
    Ordinary {
        wave: PreparedStepSubmissionWave<R>,
        reservation: Option<Arc<ResidentDecodeReservation<R>>>,
    },
    Resident {
        program: Arc<ResidentDecodeProgram<R>>,
        use_: ResidentDecodeUse<R>,
    },
}

pub(super) enum PreparedRetirement<R: DeviceRuntime> {
    Ordinary(Arc<StepResourceLease<R>>),
    Resident(ResidentFrameRetirement<R>),
}

impl<R: DeviceRuntime> From<Arc<StepResourceLease<R>>> for PreparedRetirement<R> {
    fn from(step: Arc<StepResourceLease<R>>) -> Self {
        Self::Ordinary(step)
    }
}

impl<R: DeviceRuntime> PreparedRetirement<R> {
    pub(super) fn try_retire_normal(self) -> std::result::Result<StepRetirementReceipt, String> {
        match self {
            Self::Ordinary(step) => step.try_retire_normal().map_err(|e| e.error().to_string()),
            Self::Resident(retirement) => retirement
                .try_retire_normal()
                .map_err(|e| e.error().to_string()),
        }
    }
    pub(super) fn try_abort(self) -> std::result::Result<StepRetirementReceipt, String> {
        match self {
            Self::Ordinary(step) => step.try_abort().map_err(|e| e.error().to_string()),
            Self::Resident(retirement) => retirement.try_abort().map_err(|e| e.error().to_string()),
        }
    }
    pub(super) fn try_rollback_unsubmitted(
        self,
    ) -> std::result::Result<StepRetirementReceipt, String> {
        match self {
            Self::Ordinary(step) => step
                .try_rollback_unsubmitted()
                .map_err(|e| e.error().to_string()),
            Self::Resident(retirement) => retirement
                .try_rollback_unsubmitted()
                .map_err(|e| e.error().to_string()),
        }
    }
}

impl<R: DeviceRuntime> PreparedVNextPrefill<R> {
    pub(super) fn ordinary(
        step: Arc<StepResourceLease<R>>,
        wave: PreparedStepSubmissionWave<R>,
    ) -> Self {
        Self {
            retirement: PreparedRetirement::Ordinary(step),
            dispatch: PreparedDispatch::Ordinary {
                wave,
                reservation: None,
            },
        }
    }
}

impl<R: DeviceRuntime> PreparedDispatch<R> {
    pub(super) fn lane_stable_slot_identity(&self) -> Option<LaneStableArenaSlotIdentity> {
        match self {
            Self::Ordinary { wave, .. } => wave
                .step_resources()
                .claimed_backing()
                .lane_stable_slot_identity(),
            Self::Resident { use_, .. } => use_.lane_stable_slot_identity(),
        }
    }
    pub(super) fn set_terminal_readbacks(
        &mut self,
        request: CompletionReadbackBatchRequest,
    ) -> std::result::Result<(), VNextError> {
        match self {
            Self::Ordinary { wave, .. } => wave.set_terminal_readbacks(request),
            Self::Resident { use_, .. } => use_.set_terminal_readbacks(request),
        }
    }
}

pub(super) enum PreparedDispatchResult<R: DeviceRuntime> {
    Outcome(DispatchOutcome<R>),
    // Returned only before enqueue, after dropping the unsubmitted use.
    Unavailable,
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn release_resident_decode(&self) {
        if self.invocation_preparation_strategy
            != InvocationPreparationStrategy::ResidentDecodeProgram
        {
            return;
        }
        if let Some(cached) = self.resident_decode.lock().take() {
            cached.reservation().revoke();
        }
    }

    /// Keep an idle compiled program while Sequence pages are appended. The
    /// next use must validate and refresh its physical snapshot before enqueue.
    /// Bootstrap claims cannot be refreshed and must be released for maintenance.
    pub(super) fn retain_resident_decode_for_growth(&self) -> bool {
        if self.invocation_preparation_strategy
            != InvocationPreparationStrategy::ResidentDecodeProgram
        {
            return false;
        }
        let mut cached = self.resident_decode.lock();
        if matches!(cached.as_ref(), Some(ResidentCache::Ready(program)) if program.is_idle()) {
            return true;
        }
        if let Some(cached) = cached.take() {
            cached.reservation().revoke();
        }
        false
    }

    pub(super) fn try_prepare_resident_decode(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
        greedy: bool,
    ) -> Result<Option<PreparedVNextPrefill<R>>> {
        if self.invocation_preparation_strategy
            != InvocationPreparationStrategy::ResidentDecodeProgram
        {
            return Ok(None);
        }
        if kind != VNextExecutionWaveKind::Decode
            || !greedy
            || self.checkpoint_capture.is_some()
            || !self
                .device_timing_mode()
                .direct_reusable_execution_allowed()
            || spans.iter().any(|span| span.immediate_tokens() != 1)
        {
            self.release_resident_decode();
            return Ok(None);
        }
        // Prefix-cache token evidence belongs to this fresh request/frame, not
        // the cached program. Resident completion and retirement publish its
        // checked frontier just like an ordinary FullPlan wave.
        let request = self.step_request_for_spans(batch, spans, kind)?;
        let cached = self.resident_decode.lock().take();
        let mut reservation = None;
        if let Some(cached) = cached {
            if cached.reservation().is_idle() && cached.reservation().matches_participants(batch) {
                match cached {
                    ResidentCache::Ready(program) => {
                        match program
                            .try_begin_use(batch, &request)
                            .map_err(|e| FerrumError::backend(e.to_string()))?
                        {
                            ResidentDecodeDecision::Ready { use_, retirement } => {
                                *self.resident_decode.lock() =
                                    Some(ResidentCache::Ready(Arc::clone(&program)));
                                return Ok(Some(PreparedVNextPrefill {
                                    retirement: PreparedRetirement::Resident(retirement),
                                    dispatch: PreparedDispatch::Resident { program, use_ },
                                }));
                            }
                            _ => program.revoke(),
                        }
                    }
                    ResidentCache::Bootstrap(value) => reservation = Some(value),
                }
            } else {
                cached.reservation().revoke();
            }
        }
        let reservation = match reservation {
            Some(value) => value,
            None => match batch
                .try_reserve_resident_decode(&request, &self.lane)
                .map_err(|e| FerrumError::backend(e.to_string()))?
            {
                Some(value) => value,
                None => return Ok(None),
            },
        };
        match reservation
            .prepare_bootstrap(&request)
            .map_err(|e| FerrumError::backend(e.to_string()))?
        {
            ResidentBootstrapDecision::Prepared { step, wave } => {
                self.resident_metrics
                    .bootstraps
                    .fetch_add(1, Ordering::Relaxed);
                *self.resident_decode.lock() =
                    Some(ResidentCache::Bootstrap(Arc::clone(&reservation)));
                Ok(Some(PreparedVNextPrefill {
                    retirement: PreparedRetirement::Ordinary(step),
                    dispatch: PreparedDispatch::Ordinary {
                        wave,
                        reservation: Some(reservation),
                    },
                }))
            }
            _ => {
                reservation.revoke();
                Ok(None)
            }
        }
    }

    pub(super) fn install_resident_decode(&self, program: Arc<ResidentDecodeProgram<R>>) {
        let mut cached = self.resident_decode.lock();
        if program.is_idle()
            && cached
                .as_ref()
                .is_some_and(|entry| Arc::ptr_eq(entry.reservation(), program.reservation()))
        {
            *cached = Some(ResidentCache::Ready(program));
            self.resident_metrics
                .installations
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn dispatch_prepared_wave(
        &self,
        participants: &[VNextExecutionParticipant<'_, R>],
        wave: PreparedDispatch<R>,
        kind: VNextExecutionWaveKind,
        output_modes: &ProductReadbackModes,
        mask_plans: &[VNextProductTokenMaskSubmissionPlan],
        cold_wave: Option<&mut reusable_cold_path::WaveObservation<'_>>,
        pending_install: &mut Option<Arc<ResidentDecodeProgram<R>>>,
    ) -> PreparedDispatchResult<R> {
        let (program, mut use_) = match wave {
            PreparedDispatch::Resident { program, use_ } => (program, use_),
            PreparedDispatch::Ordinary { wave, reservation } => {
                return PreparedDispatchResult::Outcome(self.dispatch_participant_wave(
                    participants,
                    wave,
                    kind,
                    output_modes,
                    mask_plans,
                    cold_wave,
                    reservation,
                    pending_install,
                ));
            }
        };
        if output_modes.uniform() != Some(VNextProductOutputMode::GreedyToken) {
            return PreparedDispatchResult::Outcome(DispatchOutcome::QuiescentFailure(
                "resident decode requires uniform greedy readbacks".to_owned(),
            ));
        }
        let uploads =
            match self.prepare_participant_uploads(participants, kind, output_modes, mask_plans) {
                Ok(uploads) => uploads,
                Err(error) => {
                    return PreparedDispatchResult::Outcome(DispatchOutcome::QuiescentFailure(
                        error.to_string(),
                    ))
                }
            };
        let mut retries = 0;
        loop {
            let identity = match OperationDispatch::bind_resident_decode_identity(
                &program,
                participants
                    .iter()
                    .map(|p| p.sequence.active_binding.as_ref()),
                &use_,
                &self.lane,
            ) {
                Ok(identity) => identity,
                Err(error) => {
                    return PreparedDispatchResult::Outcome(DispatchOutcome::QuiescentFailure(
                        error.to_string(),
                    ))
                }
            };
            let timing = VNextWaveTimingSink {
                aggregate: &self.metrics.wave_timing,
                phase: self.metrics.wave_timing_for(kind),
            };
            let submission = if self.host_dispatch_timing_enabled() {
                OperationDispatch::encode_and_submit_resident_decode_with_inputs(
                    &program,
                    &self.resolved_plan,
                    &identity,
                    &uploads,
                    use_,
                    &self.lane,
                    &self.reaper,
                    self.device_timing_mode(),
                    &timing,
                )
            } else {
                OperationDispatch::encode_and_submit_resident_decode_with_inputs(
                    &program,
                    &self.resolved_plan,
                    &identity,
                    &uploads,
                    use_,
                    &self.lane,
                    &self.reaper,
                    self.device_timing_mode(),
                    &invocation_preparation::DisabledTiming,
                )
            };
            let outcome = match submission {
                Ok(submission) => {
                    let (completion, attribution) = submission.into_parts();
                    self.metrics.submitted_waves.fetch_add(1, Ordering::Relaxed);
                    self.resident_metrics
                        .submitted
                        .fetch_add(1, Ordering::Relaxed);
                    DispatchOutcome::Submitted {
                        completion,
                        attribution,
                    }
                }
                Err(ResidentDecodeDispatchError::Unavailable { use_ }) => {
                    drop(use_);
                    self.resident_metrics
                        .unavailable
                        .fetch_add(1, Ordering::Relaxed);
                    return PreparedDispatchResult::Unavailable;
                }
                Err(ResidentDecodeDispatchError::DefinitelyNotSubmitted {
                    use_: retry, ..
                }) if retries < MAX_DEFINITELY_NOT_SUBMITTED_RETRIES => {
                    retries += 1;
                    self.metrics
                        .definitely_not_submitted_retries
                        .fetch_add(1, Ordering::Relaxed);
                    use_ = retry;
                    continue;
                }
                Err(ResidentDecodeDispatchError::SubmissionIndeterminate { recovery }) => {
                    DispatchOutcome::SubmissionIndeterminate {
                        message: "vNext resident submission is indeterminate".to_owned(),
                        recovery,
                    }
                }
                Err(ResidentDecodeDispatchError::PostSubmitContract { error, completion }) => {
                    DispatchOutcome::PostSubmitContract {
                        message: error.to_string(),
                        completion,
                    }
                }
                Err(error) => DispatchOutcome::QuiescentFailure(error.to_string()),
            };
            return PreparedDispatchResult::Outcome(outcome);
        }
    }
}
