//! Actual Core admissions and complete token waves. The paired cursor owns
//! policy and history; this module owns only the original admission/frame and
//! its pending wait. No maximum-pool preprovisioning or retry loop is hidden here.
use ferrum_execution::{
    CausalComputedWave, CausalProgramSession, CausalRequestInputs, CausalWaveFailure,
};
use ferrum_runtime::*;
use ferrum_types::{FerrumError, Result};
use std::{
    future::Future,
    ops::Range,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

type Wait = Pin<Box<dyn Future<Output = Result<()>> + Send>>;

fn error(e: impl std::fmt::Display) -> FerrumError {
    FerrumError::backend(format!("MTP Core: {e}"))
}

fn token_work(history: &Arc<[u32]>, range: Range<usize>, fit: usize) -> Result<TokenSpanWork> {
    TokenSpanWork::from_token_ids_with_fit(history, range, fit)
        .and_then(|work| work.with_checkpoint_tokens(Arc::clone(history)))
        .map_err(error)
}

fn poll_wait(wait: &mut Option<Wait>, wake: &Waker) -> Result<bool> {
    let Some(future) = wait.as_mut() else {
        return Ok(true);
    };
    match future.as_mut().poll(&mut Context::from_waker(wake)) {
        Poll::Pending => Ok(false),
        Poll::Ready(result) => {
            wait.take();
            result.map(|()| true)
        }
    }
}

fn after_maintenance<R: DeviceRuntime>(
    resources: &Arc<PlanRuntimeResources<R>>,
    outcome: DynamicDeferredMaintenanceOutcome,
    wait: &mut Option<Wait>,
    wake: &Waker,
) -> Result<bool> {
    match outcome {
        DynamicDeferredMaintenanceOutcome::RetryAdmission { .. }
        | DynamicDeferredMaintenanceOutcome::Maintained(_) => Ok(true),
        DynamicDeferredMaintenanceOutcome::WaitForRelease { wait_condition, .. } => {
            let registration = resources
                .register_capacity_waiter(&wait_condition)
                .map_err(error)?;
            *wait = Some(Box::pin(async move {
                registration
                    .wait_for_change()
                    .await
                    .map(|_| ())
                    .map_err(error)
            }));
            // Register the original caller waker now, including an epoch that
            // changed between maintenance and registration. Never spin here.
            poll_wait(wait, wake)
        }
    }
}

pub(super) enum AdmissionPoll<R: DeviceRuntime> {
    Pending { made_progress: bool },
    Ready(Arc<SequenceSession<R>>),
}

#[must_use = "pending admission owns its original request/session until ready or aborted"]
pub(super) struct Admission<R: DeviceRuntime> {
    resources: Arc<PlanRuntimeResources<R>>,
    work: ResourceWorkShape,
    run: RunId,
    request_id: RequestIdentity,
    conditioning: Option<Arc<CausalRequestInputs>>,
    request: Option<Arc<AdmittedRequestResources<R>>>,
    sequence: Option<Arc<AdmittedSequenceResources<R>>>,
    session: Option<Arc<SequenceSession<R>>>,
    wait: Option<Wait>,
    finished: bool,
}

impl<R: DeviceRuntime> Admission<R> {
    pub(super) fn new(
        program: &CausalProgramSession<R>,
        run: RunId,
        request_id: RequestIdentity,
        history: Arc<[u32]>,
        initial_range: Range<usize>,
        fit_tokens: usize,
        conditioning: Option<Arc<CausalRequestInputs>>,
    ) -> Result<Self> {
        if initial_range.start != 0 || program.installation_only() {
            return Err(error(
                "initial admission requires an executable program and first token span",
            ));
        }
        let work = ResourceWorkShape::single(token_work(&history, initial_range, fit_tokens)?)
            .map_err(error)?;
        Ok(Self {
            resources: Arc::clone(program.resources()),
            work,
            run,
            request_id,
            conditioning,
            request: None,
            sequence: None,
            session: None,
            wait: None,
            finished: false,
        })
    }

    pub(super) fn poll(
        &mut self,
        program: &CausalProgramSession<R>,
        wake: &Waker,
    ) -> Result<AdmissionPoll<R>> {
        if self.finished || !Arc::ptr_eq(&self.resources, program.resources()) {
            return Err(error(
                "admission is complete or belongs to another initialized program",
            ));
        }
        if !poll_wait(&mut self.wait, wake)? {
            return Ok(AdmissionPoll::Pending {
                made_progress: false,
            });
        }
        if self.request.is_none() {
            let binding = self.resources.trusted_runtime_binding().map_err(error)?;
            match binding
                .try_admit_request(
                    RequestResourceAdmissionRequest::new(
                        self.work.clone(),
                        AdmissionFitPolicy::ImmediateOnly,
                        AdmissionPressureAction::WaitForRelease,
                    )
                    .map_err(error)?,
                    self.run.clone(),
                    self.request_id.clone(),
                )
                .map_err(error)?
            {
                RequestResourceAdmissionDecision::Admitted(request) => self.request = Some(request),
                RequestResourceAdmissionDecision::BackingDeferred(d) => {
                    let made_progress = after_maintenance(
                        &self.resources,
                        d.maintain().map_err(error)?,
                        &mut self.wait,
                        wake,
                    )?;
                    return Ok(AdmissionPoll::Pending { made_progress });
                }
                RequestResourceAdmissionDecision::Deferred(d) => {
                    let outcome = self
                        .resources
                        .maintain_for_admission_deferred(&d)
                        .map_err(error)?;
                    let made_progress =
                        after_maintenance(&self.resources, outcome, &mut self.wait, wake)?;
                    return Ok(AdmissionPoll::Pending { made_progress });
                }
                RequestResourceAdmissionDecision::PermanentRejected(d) => {
                    return Err(error(format!("request rejected: {d:?}")))
                }
            }
        }
        if self.sequence.is_none() {
            match self
                .request
                .as_ref()
                .expect("admitted request")
                .try_admit_sequence(
                    SequenceResourceAdmissionRequest::new(
                        self.work.clone(),
                        AdmissionFitPolicy::ImmediateOnly,
                        AdmissionPressureAction::WaitForRelease,
                    )
                    .map_err(error)?,
                )
                .map_err(error)?
            {
                SequenceResourceAdmissionDecision::Admitted(sequence) => {
                    self.sequence = Some(sequence)
                }
                SequenceResourceAdmissionDecision::BackingDeferred(d) => {
                    let made_progress = after_maintenance(
                        &self.resources,
                        d.maintain().map_err(error)?,
                        &mut self.wait,
                        wake,
                    )?;
                    return Ok(AdmissionPoll::Pending { made_progress });
                }
                SequenceResourceAdmissionDecision::Deferred(d) => {
                    let outcome = self
                        .resources
                        .maintain_for_admission_deferred(&d)
                        .map_err(error)?;
                    let made_progress =
                        after_maintenance(&self.resources, outcome, &mut self.wait, wake)?;
                    return Ok(AdmissionPoll::Pending { made_progress });
                }
                SequenceResourceAdmissionDecision::PermanentRejected(d) => {
                    return Err(error(format!("sequence rejected: {d:?}")))
                }
            }
        }
        if self.session.is_none() {
            self.session = Some(
                self.sequence
                    .as_ref()
                    .expect("admitted sequence")
                    .open_session()
                    .map_err(error)?,
            );
        }
        // Keep the opened session in self until binding succeeds. A failure
        // here still has a real quiescent session to abort, not a lost lease.
        if let Some(inputs) = &self.conditioning {
            program
                .bind_conditioning_inputs(self.session.as_ref().expect("opened session"), inputs)?;
        }
        let session = self.session.take().expect("bound session");
        self.sequence.take();
        self.request.take();
        self.conditioning.take();
        self.finished = true;
        Ok(AdmissionPoll::Ready(session))
    }

    pub(super) fn try_abort(&mut self) -> Result<bool> {
        self.wait.take();
        if let Some(session) = &self.session {
            if session
                .try_abort_if_quiescent()
                .or_else(|_| session.try_abort())
                .is_err()
            {
                return Ok(false);
            }
        }
        self.session.take();
        self.sequence.take();
        self.request.take();
        self.conditioning.take();
        self.finished = true;
        Ok(true)
    }
}

pub(super) struct ExternalInput {
    pub(super) value: ProgramValueId,
    pub(super) element_type: ElementType,
    pub(super) bytes: ManagedHostBytes,
}

pub(super) struct Member<R: DeviceRuntime> {
    pub(super) session: Arc<SequenceSession<R>>,
    pub(super) history: Arc<[u32]>,
    pub(super) range: Range<usize>,
    pub(super) inputs: Vec<ExternalInput>,
}

pub(super) enum WavePoll<R: DeviceRuntime> {
    Pending { made_progress: bool },
    Computed(CausalComputedWave<R>),
}

/// Before retains pending authority in Wave; Execution moves the actual
/// submitted frame/recovery handles to the returned failure. Keep that error.
#[must_use]
pub(super) enum WaveError<R: DeviceRuntime> {
    Before(FerrumError),
    Execution(CausalWaveFailure<R>),
}
impl<R: DeviceRuntime> From<FerrumError> for WaveError<R> {
    fn from(error: FerrumError) -> Self {
        Self::Before(error)
    }
}

#[must_use = "a wave with a pending frame must be executed or explicitly aborted"]
pub(super) struct Wave<R: DeviceRuntime> {
    resources: Arc<PlanRuntimeResources<R>>,
    batch: Option<ExecutionBatchParticipants<R>>,
    lane: Arc<ExecutionLane<R>>,
    members: Vec<Member<R>>,
    input_indices: Vec<usize>,
    spans: Vec<TokenSpanWork>,
    uploads: Vec<SubmissionWaveInputUpload>,
    extension_index: usize,
    step: Option<Arc<StepResourceLease<R>>>,
    wait: Option<Wait>,
    finished: bool,
}

impl<R: DeviceRuntime> Wave<R> {
    /// `fit_ceiling` is only the configured context ceiling. Each token span
    /// keeps its exact admitted request's own fit claim, even in a mixed batch.
    pub(super) fn new(
        program: &CausalProgramSession<R>,
        members: Vec<Member<R>>,
        fit_ceiling: usize,
    ) -> Result<Self> {
        if program.installation_only() {
            return Err(error("installation-only program cannot form a wave"));
        }
        let batch = ExecutionBatchParticipants::new(
            members.iter().map(|m| Arc::clone(&m.session)).collect(),
        )
        .map_err(error)?;
        let evidence = program
            .resources()
            .trusted_runtime_binding()
            .map_err(error)?
            .evidence();
        if batch
            .sessions()
            .iter()
            .any(|s| s.resources().plan_evidence() != evidence)
        {
            return Err(error(
                "wave participant belongs to another initialized program",
            ));
        }
        // Core sorts by actual authority. Preserve this exact order in work,
        // uploads, readbacks, decisions and the caller's cancellation mapping.
        let input_indices = batch
            .sessions()
            .iter()
            .map(|session| {
                members
                    .iter()
                    .position(|m| Arc::ptr_eq(&m.session, session))
                    .ok_or_else(|| error("Core participant mapping changed"))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut original = members.into_iter().map(Some).collect::<Vec<_>>();
        let members = input_indices
            .iter()
            .map(|&i| original[i].take().expect("unique Core member"))
            .collect::<Vec<_>>();
        let spans = members
            .iter()
            .map(|member| {
                let fit = usize::try_from(
                    member
                        .session
                        .resources()
                        .request_resources()
                        .work_shape()
                        .fit_tokens()
                        .map_err(error)?,
                )
                .map_err(error)?;
                if fit > fit_ceiling {
                    return Err(error("request fit exceeds configured context ceiling"));
                }
                token_work(&member.history, member.range.clone(), fit)
            })
            .collect::<Result<Vec<_>>>()?;
        let uploads = uploads(program, &members)?;
        let lane = program.resources().create_execution_lane().map_err(error)?;
        Ok(Self {
            resources: Arc::clone(program.resources()),
            batch: Some(batch),
            lane,
            members,
            input_indices,
            spans,
            uploads,
            extension_index: 0,
            step: None,
            wait: None,
            finished: false,
        })
    }

    /// Computed/committed outputs use this canonical index -> original Member
    /// index mapping. It remains available after poll returns Computed.
    pub(super) fn input_indices(&self) -> &[usize] {
        &self.input_indices
    }

    /// Only a wave that has not acquired a frame may be discarded to remove
    /// cancelled members. Once a frame exists, keep its exact participants
    /// through compute and let normal settlement discard cancelled outputs.
    pub(super) fn discard_before_frame(&mut self) -> bool {
        if self.step.is_some() || self.finished {
            return false;
        }
        self.wait.take();
        self.batch.take();
        self.uploads.clear();
        self.members.clear();
        self.finished = true;
        true
    }

    #[cfg(test)]
    pub(super) fn has_frame(&self) -> bool {
        self.step.is_some()
    }

    pub(super) fn poll(
        &mut self,
        program: &CausalProgramSession<R>,
        reaper: Arc<CompletionReaper<R>>,
        wake: &Waker,
    ) -> std::result::Result<WavePoll<R>, WaveError<R>> {
        if self.finished || !Arc::ptr_eq(&self.resources, program.resources()) {
            return Err(error("wave is finished or belongs to another initialized program").into());
        }
        if !poll_wait(&mut self.wait, wake)? {
            return Ok(WavePoll::Pending {
                made_progress: false,
            });
        }
        // Required physical coverage grows through the original sequence
        // extension transaction, never through a second state/KV allocation.
        if self.extension_index < self.members.len() {
            let index = self.extension_index;
            let request = SequenceResourceExtensionRequest::new(
                ResourceWorkShape::single(self.spans[index].clone()).map_err(error)?,
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(error)?;
            match self.members[index]
                .session
                .try_ensure_backing_covers(request)
                .map_err(error)?
            {
                SequenceResourceExtensionDecision::Current(_)
                | SequenceResourceExtensionDecision::Extended(_) => self.extension_index += 1,
                SequenceResourceExtensionDecision::RetryRequired(_) => {
                    return Ok(WavePoll::Pending {
                        made_progress: true,
                    })
                }
                SequenceResourceExtensionDecision::BackingDeferred(d) => {
                    let made_progress = after_maintenance(
                        &self.resources,
                        d.maintain().map_err(error)?,
                        &mut self.wait,
                        wake,
                    )?;
                    return Ok(WavePoll::Pending { made_progress });
                }
                SequenceResourceExtensionDecision::Deferred(d) => {
                    let outcome = self
                        .resources
                        .maintain_for_admission_deferred(&d)
                        .map_err(error)?;
                    let made_progress =
                        after_maintenance(&self.resources, outcome, &mut self.wait, wake)?;
                    return Ok(WavePoll::Pending { made_progress });
                }
                SequenceResourceExtensionDecision::PermanentRejected(d) => {
                    return Err(error(format!("sequence extension rejected: {d:?}")).into())
                }
            }
            if self.extension_index < self.members.len() {
                return Ok(WavePoll::Pending {
                    made_progress: true,
                });
            }
        }
        let batch = self.batch.as_ref().expect("unsubmitted batch");
        if self.step.is_none() {
            let request = StepResourceAdmissionRequest::new(
                batch.bind_work_shape(self.spans.clone()).map_err(error)?,
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(error)?;
            match batch.try_begin_step(request, &self.lane).map_err(error)? {
                StepResourceAdmissionDecision::Admitted(step) => {
                    self.step = Some(step);
                    // The exact frame now exists. Give the caller a bounded
                    // cancellation observation point before native submission;
                    // cancelled members still settle on this original frame.
                    return Ok(WavePoll::Pending {
                        made_progress: true,
                    });
                }
                StepResourceAdmissionDecision::BackingDeferred(d) => {
                    let made_progress = after_maintenance(
                        &self.resources,
                        d.maintain().map_err(error)?,
                        &mut self.wait,
                        wake,
                    )?;
                    return Ok(WavePoll::Pending { made_progress });
                }
                StepResourceAdmissionDecision::Deferred(d) => {
                    let outcome = self
                        .resources
                        .maintain_for_admission_deferred(&d)
                        .map_err(error)?;
                    let made_progress =
                        after_maintenance(&self.resources, outcome, &mut self.wait, wake)?;
                    return Ok(WavePoll::Pending { made_progress });
                }
                StepResourceAdmissionDecision::PermanentRejected(d) => {
                    return Err(error(format!("step rejected: {d:?}")).into())
                }
            }
        }
        let step = self.step.as_ref().expect("admitted step");
        let prepared = match step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .map_err(error)?
        {
            StepSubmissionWaveAdmissionDecision::Prepared(prepared) => prepared,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(d) => {
                let made_progress = after_maintenance(
                    &self.resources,
                    d.maintain().map_err(error)?,
                    &mut self.wait,
                    wake,
                )?;
                return Ok(WavePoll::Pending { made_progress });
            }
            StepSubmissionWaveAdmissionDecision::Deferred(d) => {
                let outcome = self
                    .resources
                    .maintain_for_admission_deferred(&d)
                    .map_err(error)?;
                let made_progress =
                    after_maintenance(&self.resources, outcome, &mut self.wait, wake)?;
                return Ok(WavePoll::Pending { made_progress });
            }
            StepSubmissionWaveAdmissionDecision::RequestStateDeferred(d) => {
                let registration = d.register_waiter().map_err(error)?;
                self.wait = Some(Box::pin(async move {
                    registration
                        .wait_for_change()
                        .await
                        .map(|_| ())
                        .map_err(error)
                }));
                let made_progress = poll_wait(&mut self.wait, wake)?;
                return Ok(WavePoll::Pending { made_progress });
            }
            StepSubmissionWaveAdmissionDecision::PermanentRejected(d) => {
                return Err(error(format!("wave rejected: {d:?}")).into())
            }
            StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(d) => {
                return Err(error(format!(
                    "distinct MTP requests require a state split: {d:?}"
                ))
                .into())
            }
            StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(d) => {
                return Err(error(format!("MTP request state poisoned: {d:?}")).into())
            }
        };
        // The CausalComputedWave/failure must be the only wrapper holding the
        // original step. Otherwise selected retirement cannot unwrap its Arc.
        self.step.take();
        let batch = self.batch.take().expect("prepared batch");
        self.finished = true;
        program
            .execute_prepared_wave(&batch, prepared, std::mem::take(&mut self.uploads), reaper)
            .map(WavePoll::Computed)
            .map_err(WaveError::Execution)
    }

    pub(super) fn try_abort(&mut self) -> Result<bool> {
        self.wait.take();
        if let Some(step) = self.step.take() {
            if let Err(failure) = step.try_abort() {
                self.step = Some(failure.into_step());
                return Ok(false);
            }
        }
        self.batch.take();
        self.uploads.clear();
        self.members.clear();
        self.finished = true;
        Ok(true)
    }
}

fn uploads<R: DeviceRuntime>(
    program: &CausalProgramSession<R>,
    members: &[Member<R>],
) -> Result<Vec<SubmissionWaveInputUpload>> {
    let plan = program.plan();
    let external = plan.parts().prepared_family.program().inputs();
    let mut uploads = Vec::new();
    for (participant, member) in members.iter().enumerate() {
        if member.inputs.len() != external.len()
            || member
                .inputs
                .iter()
                .any(|input| !external.contains(&input.value))
            || member
                .inputs
                .iter()
                .enumerate()
                .any(|(i, a)| member.inputs[..i].iter().any(|b| a.value == b.value))
        {
            return Err(error(
                "MTP member inputs differ from the exact compiled external values",
            ));
        }
        let rows = member.range.len();
        for input in &member.inputs {
            if !program.host_account().owns(&input.bytes)
                || input.bytes.class() != HostMemoryClass::InputPreprocess
            {
                return Err(error(
                    "MTP input is not owned by the original Input account",
                ));
            }
            let mut ports = 0usize;
            for node in plan.execution_plan().payload().nodes() {
                for port in node.values().iter().filter(|port| {
                    port.role() == ResolvedValueRole::Input && port.value_id() == &input.value
                }) {
                    let tensor = port.tensor();
                    if tensor.element_type() != input.element_type
                        || tensor.dimensions().is_empty()
                        || member.range.end as u64 > tensor.dimensions()[0]
                        || !matches!(tensor.layout(), ResolvedTensorLayout::Contiguous)
                    {
                        return Err(error(
                            "MTP external input has a different compiled dtype/row extent/layout",
                        ));
                    }
                    let row_bytes = tensor.dimensions()[1..]
                        .iter()
                        .try_fold(input.element_type.size_bytes(), |n, d| n.checked_mul(*d))
                        .ok_or_else(|| error("MTP external row extent overflows"))?;
                    let extent = row_bytes
                        .checked_mul(rows as u64)
                        .ok_or_else(|| error("MTP external extent overflows"))?;
                    if input.bytes.len() as u64 != extent {
                        return Err(error(
                            "MTP input must contain exactly the current participant rows",
                        ));
                    }
                    if node.operation_id().as_str() == TOKEN_EMBEDDING_OPERATION_ID
                        && port.ordinal() == 0
                    {
                        if input.element_type != ElementType::U32
                            || row_bytes != 4
                            || !input
                                .bytes
                                .bytes()
                                .chunks_exact(4)
                                .zip(&member.history[member.range.clone()])
                                .all(|(bytes, token)| bytes == token.to_le_bytes())
                        {
                            return Err(error(
                                "actual embedding IDs differ from admitted token history",
                            ));
                        }
                    }
                    let offset = row_bytes
                        .checked_mul(member.range.start as u64)
                        .ok_or_else(|| error("MTP source offset overflows"))?;
                    uploads.push(
                        SubmissionWaveInputUpload::managed_range(
                            node.id().clone(),
                            participant as u32,
                            port.ordinal(),
                            offset,
                            HostTransferLayout::new(
                                input.element_type,
                                extent / input.element_type.size_bytes(),
                            )
                            .map_err(error)?,
                            input.bytes.clone(),
                        )
                        .map_err(error)?,
                    );
                    ports += 1;
                }
            }
            if ports == 0 {
                return Err(error(
                    "MTP declared external input has no compiled consuming port",
                ));
            }
        }
    }
    Ok(uploads)
}
