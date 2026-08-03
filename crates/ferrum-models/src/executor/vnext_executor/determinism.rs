use super::*;

pub const MAX_VNEXT_DETERMINISM_PARTICIPANTS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VNextDeterminismPhase {
    Prefill,
    Decode,
}

impl VNextDeterminismPhase {
    const fn wave_kind(self) -> VNextExecutionWaveKind {
        match self {
            Self::Prefill => VNextExecutionWaveKind::Prefill,
            Self::Decode => VNextExecutionWaveKind::Decode,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VNextDeterminismInitialState {
    Zero,
    Nonzero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VNextDeterminismWorkspacePoison {
    Zero,
    A5,
}

impl VNextDeterminismWorkspacePoison {
    const fn fill_byte(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::A5 => 0xa5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VNextDeterminismExecutionMode {
    Eager,
    Replayed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VNextDeterminismParticipantSpec {
    token_ids: Vec<u32>,
    immediate_range: Range<usize>,
    maximum_sequence_tokens: usize,
}

impl VNextDeterminismParticipantSpec {
    pub fn new(
        token_ids: Vec<u32>,
        immediate_range: Range<usize>,
        maximum_sequence_tokens: usize,
    ) -> Result<Self> {
        if token_ids.is_empty()
            || immediate_range.start >= immediate_range.end
            || immediate_range.end != token_ids.len()
            || maximum_sequence_tokens < token_ids.len()
        {
            return Err(FerrumError::request_validation(
                "vNext determinism participant requires a non-empty terminal token span covered by its sequence ceiling",
            ));
        }
        Ok(Self {
            token_ids,
            immediate_range,
            maximum_sequence_tokens,
        })
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn immediate_range(&self) -> Range<usize> {
        self.immediate_range.clone()
    }

    pub const fn maximum_sequence_tokens(&self) -> usize {
        self.maximum_sequence_tokens
    }

    fn token_span(&self) -> Result<TokenSpanWork> {
        TokenSpanWork::from_token_ids_with_fit(
            &self.token_ids,
            self.immediate_range.clone(),
            self.maximum_sequence_tokens,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn full_extension(&self) -> Result<ResourceWorkShape> {
        let span = TokenSpanWork::from_token_ids_with_fit(
            &self.token_ids,
            0..self.token_ids.len(),
            self.maximum_sequence_tokens,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        ResourceWorkShape::single(span).map_err(|error| FerrumError::backend(error.to_string()))
    }

    fn immediate_token_ids(&self) -> &[u32] {
        &self.token_ids[self.immediate_range.clone()]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VNextDeterminismExecutionSpec {
    phase: VNextDeterminismPhase,
    participants: Vec<VNextDeterminismParticipantSpec>,
    initial_state: VNextDeterminismInitialState,
    workspace_poison: VNextDeterminismWorkspacePoison,
    mode: VNextDeterminismExecutionMode,
}

impl VNextDeterminismExecutionSpec {
    pub fn new(
        phase: VNextDeterminismPhase,
        participants: Vec<VNextDeterminismParticipantSpec>,
        initial_state: VNextDeterminismInitialState,
        workspace_poison: VNextDeterminismWorkspacePoison,
        mode: VNextDeterminismExecutionMode,
    ) -> Result<Self> {
        if participants.is_empty()
            || participants.len() > MAX_VNEXT_DETERMINISM_PARTICIPANTS
            || (phase == VNextDeterminismPhase::Decode
                && participants
                    .iter()
                    .any(|participant| participant.immediate_range.len() != 1))
        {
            return Err(FerrumError::request_validation(format!(
                "vNext determinism execution requires 1..={MAX_VNEXT_DETERMINISM_PARTICIPANTS} participants and one immediate token per decode participant"
            )));
        }
        Ok(Self {
            phase,
            participants,
            initial_state,
            workspace_poison,
            mode,
        })
    }

    pub const fn phase(&self) -> VNextDeterminismPhase {
        self.phase
    }

    pub fn participants(&self) -> &[VNextDeterminismParticipantSpec] {
        &self.participants
    }

    pub const fn initial_state(&self) -> VNextDeterminismInitialState {
        self.initial_state
    }

    pub const fn workspace_poison(&self) -> VNextDeterminismWorkspacePoison {
        self.workspace_poison
    }

    pub const fn mode(&self) -> VNextDeterminismExecutionMode {
        self.mode
    }
}

struct PendingDeterminismAdmissions<'executor, R: DeviceRuntime> {
    executor: &'executor VNextModelExecutor<R>,
    request_ids: Vec<RequestId>,
    armed: bool,
}

impl<'executor, R: DeviceRuntime> PendingDeterminismAdmissions<'executor, R> {
    fn new(executor: &'executor VNextModelExecutor<R>) -> Self {
        Self {
            executor,
            request_ids: Vec::new(),
            armed: true,
        }
    }

    fn track(&mut self, request_id: RequestId) {
        self.request_ids.push(request_id);
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<R: DeviceRuntime> Drop for PendingDeterminismAdmissions<'_, R> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut sequences = self.executor.sequences.lock();
        for request_id in &self.request_ids {
            sequences.cancel_prefill(request_id);
        }
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    async fn admit_determinism_participants(
        &self,
        spec: &VNextDeterminismExecutionSpec,
    ) -> Result<Vec<(Arc<VNextPrefillSlot<R>>, Arc<VNextSequence<R>>)>> {
        if spec.participants.iter().any(|participant| {
            participant.maximum_sequence_tokens > self.maximum_model_tokens
                || participant.token_ids.iter().any(|token| {
                    usize::try_from(*token).map_or(true, |token| token >= self.info.vocab_size)
                })
        }) {
            return Err(FerrumError::request_validation(
                "vNext determinism participant exceeds the resolved model token or vocabulary bound",
            ));
        }

        let mut pending = PendingDeterminismAdmissions::new(self);
        for participant in &spec.participants {
            let request_id = RequestId::new();
            pending.track(request_id.clone());
            let input_tokens = participant
                .token_ids
                .iter()
                .copied()
                .map(TokenId::new)
                .collect::<Vec<_>>();
            let mut maintenance_attempts = 0_u32;
            loop {
                match self.try_admit_prefill(ExecutorPrefillAdmission::for_diagnostic(
                    &request_id,
                    &input_tokens,
                    participant.maximum_sequence_tokens,
                ))? {
                    ExecutorPrefillAdmissionDecision::Admitted(receipt) => {
                        if receipt.request_id != request_id {
                            return Err(FerrumError::internal(
                                "vNext determinism admission changed request identity",
                            ));
                        }
                        break;
                    }
                    ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                        if maintenance_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                            return Err(FerrumError::resource_exhausted(
                                "vNext determinism admission backing did not converge",
                            ));
                        }
                        maintenance_attempts += 1;
                        match self.maintain_prefill_backing(&request_id)? {
                            ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                            | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. } => continue,
                            ExecutorPrefillMaintenanceOutcome::WaitForRelease { .. } => {
                                return Err(FerrumError::resource_exhausted(
                                    "vNext determinism admission is waiting for capacity release",
                                ))
                            }
                            ExecutorPrefillMaintenanceOutcome::NoLongerPending => {
                                return Err(FerrumError::internal(
                                    "vNext determinism admission lost its retained request",
                                ))
                            }
                        }
                    }
                    ExecutorPrefillAdmissionDecision::Deferred(deferred) => {
                        return Err(Self::deferred(
                            "determinism participant admission",
                            &deferred,
                        ))
                    }
                    ExecutorPrefillAdmissionDecision::PermanentRejected(rejected) => {
                        return Err(FerrumError::resource_exhausted(format!(
                            "vNext determinism participant exceeds its immutable plan: {rejected:?}"
                        )))
                    }
                }
            }
        }

        let request_ids = pending.request_ids.clone();
        let admitted = self
            .sequences
            .lock()
            .begin_prefill_batch_execution(&request_ids)?;
        pending.disarm();
        Ok(admitted)
    }

    fn prepare_determinism_wave(
        &self,
        step: &Arc<StepResourceLease<R>>,
        sequences: &[Arc<VNextSequence<R>>],
        spans: &[TokenSpanWork],
    ) -> Result<PreparedStepSubmissionWave<R>> {
        if sequences.len() != spans.len() {
            return Err(FerrumError::internal(
                "vNext determinism-wave maintenance participants differ from the work spans",
            ));
        }
        Self::validate_step_maintenance_participants(step, sequences)?;
        let work_shape = step
            .shared_all_invocation_work_shape(spans)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let requests = self
            .resolved_plan
            .execution_plan()
            .payload()
            .nodes()
            .iter()
            .map(|node| {
                InvocationResourceAdmissionRequest::for_all_step_participants(
                    node.id().clone(),
                    Arc::clone(&work_shape),
                    AdmissionFitPolicy::ImmediateOnly,
                    AdmissionPressureAction::WaitForRelease,
                )
                .map_err(|error| FerrumError::backend(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut backing_attempts = 0_u32;
        let mut maintenance_receipts = Vec::new();
        loop {
            match step
                .try_prepare_determinism_submission_wave(requests.clone())
                .map_err(|error| FerrumError::backend(error.to_string()))?
            {
                StepSubmissionWaveAdmissionDecision::Prepared(wave) => return Ok(wave),
                StepSubmissionWaveAdmissionDecision::Deferred(deferred) => {
                    if deferred.action() != DeferredAction::AwaitBackingGrowth {
                        return Err(Self::deferred("determinism submission wave", &deferred));
                    }
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        let deferred = ExecutorExecutionCapacityDeferral::from_pending_maintenance(
                            &deferred,
                            ExecutorExecutionCapacityStage::SubmissionWave,
                        )?;
                        return Err(Self::execution_capacity_error(&deferred));
                    }
                    backing_attempts += 1;
                    let outcome = self
                        .plan_resources
                        .maintain_for_admission_deferred(&deferred)
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        outcome,
                        Some(&deferred),
                        sequences.iter().map(Arc::as_ref),
                        &mut maintenance_receipts,
                    )? {
                        return Err(Self::execution_capacity_error(&deferred));
                    }
                }
                StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                    if backing_attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        let deferred = ExecutorExecutionCapacityDeferral::from_backing(
                            deferred.evidence(),
                            ExecutorExecutionCapacityStage::SubmissionWave,
                        )?;
                        return Err(Self::execution_capacity_error(&deferred));
                    }
                    backing_attempts += 1;
                    let outcome = deferred
                        .maintain()
                        .map_err(|error| FerrumError::backend(error.to_string()))?;
                    if let Some(deferred) = self.execution_maintenance_decision(
                        ExecutorExecutionCapacityStage::SubmissionWave,
                        outcome,
                        None,
                        sequences.iter().map(Arc::as_ref),
                        &mut maintenance_receipts,
                    )? {
                        return Err(Self::execution_capacity_error(&deferred));
                    }
                }
                StepSubmissionWaveAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::backend(format!(
                        "vNext determinism wave exceeds its immutable plan: {rejected:?}"
                    )))
                }
                StepSubmissionWaveAdmissionDecision::RequestStateDeferred(deferred) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "vNext determinism wave is waiting for Request-state hazards: {:?}",
                        deferred.blockers()
                    )))
                }
                StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(split) => {
                    return Err(FerrumError::request_validation(format!(
                        "vNext determinism wave requires sibling split for request {:?}: {:?}",
                        split.request(),
                        split.resource_ids()
                    )))
                }
                StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(poison) => {
                    return Err(FerrumError::backend(format!(
                        "vNext determinism Request-state resource is poisoned: {poison:?}"
                    )))
                }
            }
        }
    }

    fn deterministic_scalar_bytes(element_type: ElementType, nonzero: bool) -> &'static [u8] {
        if !nonzero {
            return &[0, 0, 0, 0];
        }
        match element_type {
            ElementType::Bool | ElementType::U8 | ElementType::I8 => &[1],
            ElementType::F16 => &[0x00, 0x3c],
            ElementType::Bf16 => &[0x80, 0x3f],
            ElementType::U32 | ElementType::I32 => &[1, 0, 0, 0],
            ElementType::F32 => &[0x00, 0x00, 0x80, 0x3f],
        }
    }

    fn repeated_deterministic_scalar(
        element_type: ElementType,
        length_bytes: usize,
        nonzero: bool,
    ) -> Result<Vec<u8>> {
        let element_width = usize::try_from(element_type.size_bytes())
            .map_err(|_| FerrumError::internal("determinism element width exceeds usize"))?;
        if length_bytes == 0 || length_bytes % element_width != 0 {
            return Err(FerrumError::internal(
                "determinism payload length differs from its element type",
            ));
        }
        let scalar = Self::deterministic_scalar_bytes(element_type, nonzero);
        let scalar = &scalar[..element_width];
        Ok(scalar.iter().copied().cycle().take(length_bytes).collect())
    }

    fn determinism_external_payload(
        &self,
        participant: &VNextDeterminismParticipantSpec,
        initialization: &ExecutionDeterminismInitializationSpec,
        length_bytes: usize,
    ) -> Result<Vec<u8>> {
        let location = initialization.location();
        let node_id = location.node_id();
        let ordinal = location.ordinal();
        if node_id == &self.io.input_node_id && ordinal == self.io.input_ordinal {
            if location.element_type() != ElementType::U32 {
                return Err(FerrumError::internal(
                    "vNext determinism token input is not U32",
                ));
            }
            let bytes = participant
                .immediate_token_ids()
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>();
            if bytes.len() != length_bytes {
                return Err(FerrumError::internal(
                    "vNext determinism token bytes differ from the prepared input range",
                ));
            }
            return Ok(bytes);
        }
        if node_id == &self.io.token_mask_input_node_id
            && ordinal == self.io.token_mask_input_ordinal
        {
            return Self::repeated_deterministic_scalar(
                location.element_type(),
                length_bytes,
                true,
            );
        }
        if node_id == &self.io.repetition_penalty_input_node_id
            && ordinal == self.io.repetition_penalty_input_ordinal
        {
            if location.element_type() != ElementType::F32 {
                return Err(FerrumError::internal(
                    "vNext determinism repetition penalty is not F32",
                ));
            }
            return Self::repeated_deterministic_scalar(
                location.element_type(),
                length_bytes,
                true,
            );
        }
        Self::repeated_deterministic_scalar(location.element_type(), length_bytes, false)
    }

    fn bind_determinism_restore(
        &self,
        participants: &[&VNextDeterminismParticipantSpec],
        initial_state: VNextDeterminismInitialState,
        identity: &BatchOperationIdentity,
        active_bindings: &[&TrustedActiveSequenceBinding],
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<SubmissionWaveDeterminismRestore> {
        let layout = SubmissionWaveDeterminismRestoreLayout::from_prepared_wave(
            self.runtime.as_ref(),
            self.providers.providers(),
            &self.resolved_plan,
            identity,
            active_bindings.iter().copied(),
            wave,
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        if usize::try_from(layout.participant_count()).ok() != Some(participants.len()) {
            return Err(FerrumError::internal(
                "vNext determinism restore participant count drifted",
            ));
        }
        let payloads = participants
            .iter()
            .enumerate()
            .map(|(participant_index, participant)| {
                let ranges = layout
                    .participant_initialization_ranges(u32::try_from(participant_index).map_err(
                        |_| {
                            FerrumError::internal("vNext determinism participant index exceeds u32")
                        },
                    )?)
                    .ok_or_else(|| {
                        FerrumError::internal("vNext determinism restore lost a participant range")
                    })?;
                layout
                    .witness_plan()
                    .initializations()
                    .iter()
                    .zip(ranges)
                    .map(|(initialization, range)| {
                        let length_bytes = usize::try_from(range.length_bytes()).map_err(|_| {
                            FerrumError::internal("vNext determinism restore range exceeds usize")
                        })?;
                        match initialization.kind() {
                            ExecutionDeterminismInitializationKind::ExternalInput { .. } => self
                                .determinism_external_payload(
                                    participant,
                                    initialization,
                                    length_bytes,
                                ),
                            ExecutionDeterminismInitializationKind::State { .. } => {
                                Self::repeated_deterministic_scalar(
                                    initialization.location().element_type(),
                                    length_bytes,
                                    initial_state == VNextDeterminismInitialState::Nonzero,
                                )
                            }
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        layout
            .bind(payloads)
            .map_err(|error| FerrumError::backend(error.to_string()))
    }

    async fn determinism_dispatch_failure(
        &self,
        step: Arc<StepResourceLease<R>>,
        error: SubmissionWaveDispatchError<R>,
    ) -> FerrumError {
        match error {
            error @ (SubmissionWaveDispatchError::DefinitelyNotSubmitted { .. }
            | SubmissionWaveDispatchError::Contract(_)
            | SubmissionWaveDispatchError::Provider(_)
            | SubmissionWaveDispatchError::Initialization(_)
            | SubmissionWaveDispatchError::InputUpload(_)) => {
                self.abort_unsubmitted_step(step, FerrumError::backend(error.to_string()))
            }
            SubmissionWaveDispatchError::SubmissionIndeterminate { recovery } => {
                let reaper = Arc::clone(&self.reaper);
                let recovered = self
                    .completion_worker
                    .execute(VNextCompletionTaskKind::IndeterminateRecovery, move || {
                        let recovered = recovery.recover_by_draining_lane();
                        drop(reaper);
                        recovered
                    })
                    .await;
                let message = match recovered {
                    Ok(Ok(_)) => "vNext determinism submission was indeterminate".to_owned(),
                    Ok(Err(error)) => format!(
                        "vNext determinism submission was indeterminate and recovery failed: {error}"
                    ),
                    Err(error) => format!(
                        "vNext determinism submission recovery task failed: {error}"
                    ),
                };
                self.abort_step(step, message).await
            }
            SubmissionWaveDispatchError::PostSubmitContract { error, completion } => {
                let message = error.to_string();
                let reaper = Arc::clone(&self.reaper);
                let _ = self
                    .completion_worker
                    .execute(VNextCompletionTaskKind::PostSubmitDrain, move || {
                        let observation = completion.wait();
                        drop(reaper);
                        observation
                    })
                    .await;
                self.abort_step(step, message).await
            }
        }
    }

    pub async fn collect_determinism_execution(
        &self,
        spec: &VNextDeterminismExecutionSpec,
    ) -> Result<SubmissionWaveDeterminismEvidence> {
        if !self.startup_preparation.lock().is_ready() {
            return Err(FerrumError::internal(
                "vNext determinism collection requires completed startup preparation",
            ));
        }

        let admitted = self.admit_determinism_participants(spec).await?;
        let mut execution_guards = admitted
            .iter()
            .map(|(slot, sequence)| {
                VNextPrefillExecutionGuard::new(
                    &self.sequences,
                    Arc::clone(slot),
                    Arc::clone(sequence),
                )
            })
            .collect::<Vec<_>>();
        let mut participant_by_authority = BTreeMap::new();
        for (participant_index, (_, sequence)) in admitted.iter().enumerate() {
            if participant_by_authority
                .insert(sequence.session.sequence_authority(), participant_index)
                .is_some()
            {
                return Err(FerrumError::internal(
                    "vNext determinism admission duplicated sequence authority",
                ));
            }
        }
        let batch = ExecutionBatchParticipants::new(
            admitted
                .iter()
                .map(|(_, sequence)| Arc::clone(&sequence.session))
                .collect(),
        )
        .map_err(|error| FerrumError::backend(error.to_string()))?;
        let canonical_indices = batch
            .sessions()
            .iter()
            .map(|session| {
                participant_by_authority
                    .remove(&session.sequence_authority())
                    .ok_or_else(|| {
                        FerrumError::internal("vNext determinism canonical participant is absent")
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if !participant_by_authority.is_empty() {
            return Err(FerrumError::internal(
                "vNext determinism participant is absent from its canonical batch",
            ));
        }

        let sequences = canonical_indices
            .iter()
            .map(|index| Arc::clone(&admitted[*index].1))
            .collect::<Vec<_>>();
        let participants = canonical_indices
            .iter()
            .map(|index| &spec.participants[*index])
            .collect::<Vec<_>>();
        let mut operation_guards = Vec::with_capacity(sequences.len());
        for sequence in &sequences {
            operation_guards.push(sequence.operation.lock().await);
        }
        for (sequence, participant) in sequences.iter().zip(&participants) {
            self.extend_sequence(sequence, participant.full_extension()?)?;
        }
        let spans = participants
            .iter()
            .map(|participant| participant.token_span())
            .collect::<Result<Vec<_>>>()?;
        let step = match self.begin_step_for_spans_with_capacity(
            &batch,
            &sequences,
            &spans,
            spec.phase.wave_kind(),
        )? {
            VNextExecutionCapacityDecision::Ready(step) => step,
            VNextExecutionCapacityDecision::Deferred(deferred) => {
                return Err(Self::execution_capacity_error(&deferred))
            }
            VNextExecutionCapacityDecision::RequestStateDeferred(_) => {
                return Err(FerrumError::internal(
                    "determinism step admission unexpectedly produced a Request-state deferral",
                ))
            }
        };
        let wave = match self.prepare_determinism_wave(&step, &sequences, &spans) {
            Ok(wave) => wave,
            Err(error) => return Err(self.abort_unsubmitted_step(step, error)),
        };
        let active_bindings = sequences
            .iter()
            .map(|sequence| sequence.active_binding.as_ref())
            .collect::<Vec<_>>();
        let identity = match OperationDispatch::bind_compiled_submission_wave_identity(
            &self.submission_wave_identity,
            active_bindings.iter().copied(),
            &wave,
            &self.lane,
        ) {
            Ok(identity) => identity,
            Err(error) => {
                return Err(
                    self.abort_unsubmitted_step(step, FerrumError::backend(error.to_string()))
                )
            }
        };
        let restore = match self.bind_determinism_restore(
            &participants,
            spec.initial_state,
            &identity,
            &active_bindings,
            &wave,
        ) {
            Ok(restore) => restore,
            Err(error) => return Err(self.abort_unsubmitted_step(step, error)),
        };

        let submission = match spec.mode {
            VNextDeterminismExecutionMode::Eager => {
                OperationDispatch::encode_and_submit_determinism_eager_wave(
                    self.providers.providers(),
                    &self.resolved_plan,
                    &identity,
                    active_bindings.iter().copied(),
                    DeviceTimingMode::Off,
                    &restore,
                    spec.workspace_poison.fill_byte(),
                    wave,
                    &self.lane,
                    &self.reaper,
                )
            }
            VNextDeterminismExecutionMode::Replayed => {
                let program_id = match OperationDispatch::reusable_execution_program_id_for_wave(
                    self.providers.providers(),
                    &self.resolved_plan,
                    &wave,
                    &self.lane,
                ) {
                    Ok(Some(program_id)) => program_id,
                    Ok(None) => {
                        return Err(self.abort_unsubmitted_step(
                            step,
                            FerrumError::backend(
                                "vNext determinism replay has no exact reusable program identity",
                            ),
                        ))
                    }
                    Err(error) => {
                        return Err(self
                            .abort_unsubmitted_step(step, FerrumError::backend(error.to_string())))
                    }
                };
                let catalog = match self.reusable_execution_catalog.get() {
                    Some(catalog) if catalog.lane_epoch == self.lane.reusable_execution_epoch() => {
                        catalog
                    }
                    _ => {
                        return Err(self.abort_unsubmitted_step(
                            step,
                            FerrumError::backend(
                                "vNext determinism replay catalog is absent or stale",
                            ),
                        ))
                    }
                };
                let reusable_program = match catalog.programs.get(&program_id) {
                    Some(program) => program,
                    None => {
                        return Err(self.abort_unsubmitted_step(
                            step,
                            FerrumError::backend(
                                "vNext determinism replay program is absent from the sealed exact catalog",
                            ),
                        ))
                    }
                };
                OperationDispatch::encode_and_submit_determinism_replayed_wave(
                    self.providers.providers(),
                    &self.resolved_plan,
                    &identity,
                    active_bindings.iter().copied(),
                    DeviceTimingMode::Off,
                    &restore,
                    spec.workspace_poison.fill_byte(),
                    reusable_program,
                    wave,
                    &self.lane,
                    &self.reaper,
                )
            }
        };
        let handle = match submission {
            Ok(handle) => handle,
            Err(error) => return Err(self.determinism_dispatch_failure(step, error).await),
        };
        let reaper = Arc::clone(&self.reaper);
        let evidence = self
            .completion_worker
            .execute(VNextCompletionTaskKind::WaveReadback, move || {
                let evidence = handle.wait_into_evidence();
                drop(reaper);
                evidence
            })
            .await
            .map_err(|error| {
                FerrumError::backend(format!("vNext determinism completion task failed: {error}"))
            })?
            .map_err(|error| FerrumError::backend(error.to_string()));
        let evidence = match evidence {
            Ok(evidence) => evidence,
            Err(error) => return Err(self.abort_step(step, error.to_string()).await),
        };
        step.try_retire_normal().map_err(|failure| {
            FerrumError::backend(format!(
                "vNext determinism step retirement failed: {}",
                failure.error()
            ))
        })?;
        drop(operation_guards);
        for guard in &mut execution_guards {
            guard.disarm();
        }
        {
            let mut registry = self.sequences.lock();
            for (slot, sequence) in &admitted {
                registry.finish_prefill_execution(slot, sequence);
            }
        }
        for (_, sequence) in admitted {
            sequence.abort();
        }
        Ok(evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn participant(start: usize, end: usize) -> VNextDeterminismParticipantSpec {
        VNextDeterminismParticipantSpec::new(vec![0; end], start..end, end + 8).unwrap()
    }

    #[test]
    fn participant_spec_requires_one_terminal_immediate_span() {
        assert!(VNextDeterminismParticipantSpec::new(vec![], 0..0, 1).is_err());
        assert!(VNextDeterminismParticipantSpec::new(vec![0, 1], 0..1, 2).is_err());
        assert!(VNextDeterminismParticipantSpec::new(vec![0, 1], 1..2, 2).is_ok());
    }

    #[test]
    fn decode_spec_rejects_multi_token_participants_and_unbounded_width() {
        assert!(VNextDeterminismExecutionSpec::new(
            VNextDeterminismPhase::Decode,
            vec![participant(0, 2)],
            VNextDeterminismInitialState::Zero,
            VNextDeterminismWorkspacePoison::Zero,
            VNextDeterminismExecutionMode::Eager,
        )
        .is_err());
        assert!(VNextDeterminismExecutionSpec::new(
            VNextDeterminismPhase::Decode,
            (0..=MAX_VNEXT_DETERMINISM_PARTICIPANTS)
                .map(|_| participant(0, 1))
                .collect(),
            VNextDeterminismInitialState::Zero,
            VNextDeterminismWorkspacePoison::Zero,
            VNextDeterminismExecutionMode::Eager,
        )
        .is_err());
    }

    #[test]
    fn prefill_spec_accepts_chunk_boundary_shape() {
        let spec = VNextDeterminismExecutionSpec::new(
            VNextDeterminismPhase::Prefill,
            vec![participant(4, 8)],
            VNextDeterminismInitialState::Nonzero,
            VNextDeterminismWorkspacePoison::A5,
            VNextDeterminismExecutionMode::Replayed,
        )
        .unwrap();
        assert_eq!(spec.participants()[0].immediate_range(), 4..8);
        assert_eq!(spec.workspace_poison(), VNextDeterminismWorkspacePoison::A5);
    }
}
