//! Step admission and bound execution-stream lifecycle.

use super::{
    acquire_session_frames_with_backing, enter_sequence_dispatch, fmt, invalid_resource,
    issue_batch_step_id, record_step_admission_profile, sequence_dispatch_is_poisoned,
    sequence_slot_active, sequence_slot_is_poisoned, sequence_slot_poisoned_drained,
    sequence_slot_poisoned_undrained, session_frame_capture_candidates,
    step_admission_profile_start, AbandonedSequenceMetadata, AbandonedSequenceRecoveryError,
    ActiveSequenceAbortDisposition, ActiveSequenceAbortReceipt, AdmissionFitPolicy,
    AdmittedSequenceResources, AdmittedStepParticipant, AllocationLifetime, Arc, AtomicU64,
    BatchCapacityClaimDecision, BatchParticipantAuthority, BoundExecutionStream,
    BoundExecutionStreamState, ClaimedBackingTransaction, DeviceRuntime,
    ExecutionBatchParticipants, ExecutionLane, ExecutionStreamCreationError,
    LaneBackingPrepareDecision, LogicalAdmissionCoordinatorId, LogicalBackingSliceAuthority,
    Ordering, RequestIdentity, RunId, SequenceAuthorityId, SequenceBackingSnapshot,
    SequenceExecutionAuthoritySource, Serialize, StepAdmissionBackingDeferral,
    StepResourceAdmissionDecision, StepResourceAdmissionProfilePhase, StepResourceAdmissionRequest,
    StepResourceLease, StreamState, TrustedPlanRuntimeEvidence, VNextError,
    SEQUENCE_DISPATCH_POISONED_BIT,
};
use std::time::Duration;

impl<R> ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    pub fn try_begin_step(
        &self,
        request: StepResourceAdmissionRequest,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<StepResourceAdmissionDecision<R>, VNextError> {
        self.try_begin_step_inner::<false, _>(request, lane, |_, _| {})
    }

    pub fn try_begin_step_profiled<F>(
        &self,
        request: StepResourceAdmissionRequest,
        lane: &Arc<ExecutionLane<R>>,
        observer: F,
    ) -> Result<StepResourceAdmissionDecision<R>, VNextError>
    where
        F: FnMut(StepResourceAdmissionProfilePhase, Duration),
    {
        self.try_begin_step_inner::<true, _>(request, lane, observer)
    }

    fn try_begin_step_inner<const PROFILE: bool, F>(
        &self,
        request: StepResourceAdmissionRequest,
        lane: &Arc<ExecutionLane<R>>,
        mut observer: F,
    ) -> Result<StepResourceAdmissionDecision<R>, VNextError>
    where
        F: FnMut(StepResourceAdmissionProfilePhase, Duration),
    {
        let phase_started = step_admission_profile_start::<PROFILE>();
        let _lifecycle = self.sessions[0]
            .resources()
            .request
            .plan
            .resources
            .read_lifecycle("begin an execution step")?;
        let StepResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
            reusable_execution_bucket_id,
        } = request;
        let work_fingerprint = work_shape.fingerprint().to_owned();
        let expected_participants = self
            .sessions
            .iter()
            .map(|session| {
                BatchParticipantAuthority::new(
                    session.sequence_authority(),
                    session.request_authority(),
                )
            })
            .collect::<Vec<_>>();
        if work_shape.participants() != expected_participants {
            return Err(invalid_resource(
                "step work authority differs from its exact participant set",
            ));
        }
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let plan = &self.sessions[0].resources().request.plan;
        if !Arc::ptr_eq(plan.runtime(), lane.runtime_arc())
            || plan.runtime().descriptor() != lane.descriptor()
            || !lane.is_reusable()
        {
            return Err(invalid_resource(
                "step admission requires the reusable execution lane bound to its plan runtime",
            ));
        }
        let reusable_execution_bucket = reusable_execution_bucket_id
            .as_ref()
            .map(|bucket_id| {
                plan.reusable_execution_bucket(bucket_id)
                    .map(|resolved| resolved.bucket().clone())
                    .ok_or_else(|| {
                        invalid_resource(
                            "step reusable execution bucket is not owned by its immutable plan",
                        )
                    })
            })
            .transpose()?;
        if reusable_execution_bucket.as_ref().is_some_and(|bucket| {
            let capacity = bucket.capacity();
            !capacity.covers(fit_shape.sequences(), fit_shape.tokens(), fit_shape.pages())
        }) {
            return Err(invalid_resource(
                "step work shape exceeds its selected reusable execution bucket",
            ));
        }
        record_step_admission_profile::<PROFILE, _>(
            &mut observer,
            StepResourceAdmissionProfilePhase::AuthorityAndPolicyValidate,
            phase_started,
        );

        let phase_started = step_admission_profile_start::<PROFILE>();
        let (demand, requested_slices) = plan.scoped_demand(
            AllocationLifetime::Step,
            None,
            immediate_shape,
            fit_shape,
            reusable_execution_bucket.as_ref(),
            fit_policy,
            pressure_action,
        )?;
        record_step_admission_profile::<PROFILE, _>(
            &mut observer,
            StepResourceAdmissionProfilePhase::DemandEvaluate,
            phase_started,
        );

        let phase_started = step_admission_profile_start::<PROFILE>();
        let prepared = match plan.prepare_lane_stable_backing_slices(lane, requested_slices)? {
            LaneBackingPrepareDecision::Prepared(prepared) => prepared,
            LaneBackingPrepareDecision::Deferred(deferred) => {
                record_step_admission_profile::<PROFILE, _>(
                    &mut observer,
                    StepResourceAdmissionProfilePhase::BackingClaim,
                    phase_started,
                );
                return Ok(StepResourceAdmissionDecision::BackingDeferred(
                    StepAdmissionBackingDeferral::new(
                        deferred,
                        self.sessions.clone(),
                        work_fingerprint,
                    )?,
                ));
            }
        };
        record_step_admission_profile::<PROFILE, _>(
            &mut observer,
            StepResourceAdmissionProfilePhase::BackingClaim,
            phase_started,
        );

        let phase_started = step_admission_profile_start::<PROFILE>();
        let logical_capacity = if demand.immediate_claim().is_empty() {
            None
        } else {
            let parents = self
                .sessions
                .iter()
                .map(|session| session.resources().logical_lease())
                .collect::<Vec<_>>();
            match plan
                .logical_admission()
                .try_claim_for_sequences(&parents, &demand)?
            {
                BatchCapacityClaimDecision::Claimed(capacity) => {
                    let parents_match = capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .eq(self.sessions.iter().map(|session| {
                            (session.sequence_authority(), session.request_authority())
                        }));
                    if !plan
                        .logical_admission()
                        .owns_batch_capacity_claim(&capacity)
                        || !parents_match
                    {
                        return Err(invalid_resource(
                            "step admission returned capacity for another participant set",
                        ));
                    }
                    Some(capacity)
                }
                BatchCapacityClaimDecision::Deferred(deferred) => {
                    record_step_admission_profile::<PROFILE, _>(
                        &mut observer,
                        StepResourceAdmissionProfilePhase::LogicalCapacityClaim,
                        phase_started,
                    );
                    return Ok(StepResourceAdmissionDecision::Deferred(deferred));
                }
                BatchCapacityClaimDecision::PermanentRejected(rejected) => {
                    record_step_admission_profile::<PROFILE, _>(
                        &mut observer,
                        StepResourceAdmissionProfilePhase::LogicalCapacityClaim,
                        phase_started,
                    );
                    return Ok(StepResourceAdmissionDecision::PermanentRejected(rejected));
                }
            }
        };
        record_step_admission_profile::<PROFILE, _>(
            &mut observer,
            StepResourceAdmissionProfilePhase::LogicalCapacityClaim,
            phase_started,
        );

        let phase_started = step_admission_profile_start::<PROFILE>();
        let committed_backing = prepared.commit();
        let claimed_backing = ClaimedBackingTransaction::new_lane_stable(
            work_shape,
            demand,
            logical_capacity,
            committed_backing,
        )?;
        record_step_admission_profile::<PROFILE, _>(
            &mut observer,
            StepResourceAdmissionProfilePhase::TransactionValidateAndFingerprint,
            phase_started,
        );

        let phase_started = step_admission_profile_start::<PROFILE>();
        let batch_step_id = issue_batch_step_id()?;
        let candidates = session_frame_capture_candidates(&self.sessions);
        let captured_frames = acquire_session_frames_with_backing(&candidates, batch_step_id)?;
        let participants = self
            .sessions
            .iter()
            .cloned()
            .zip(captured_frames)
            .map(|(session, captured)| AdmittedStepParticipant {
                frame: captured.hold,
                backing_snapshot: captured.backing_snapshot,
                session,
            })
            .collect();
        let decision = StepResourceAdmissionDecision::Admitted(Arc::new(StepResourceLease::new(
            participants,
            Arc::clone(lane),
            reusable_execution_bucket,
            batch_step_id,
            claimed_backing,
        )?));
        record_step_admission_profile::<PROFILE, _>(
            &mut observer,
            StepResourceAdmissionProfilePhase::FrameCaptureAndLease,
            phase_started,
        );
        Ok(decision)
    }
}

impl<R> AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn validate_runtime(&self, context: &'static str) -> Result<(), VNextError> {
        let descriptor = self.request.plan.runtime().descriptor();
        descriptor.validate()?;
        if descriptor.id != *self.request.plan.device_id()
            || descriptor.runtime_implementation_fingerprint
                != self.request.plan.runtime_implementation_fingerprint()
        {
            return Err(invalid_resource(format!(
                "{context} runtime differs from the trusted plan/runtime binding"
            )));
        }
        Ok(())
    }

    pub fn create_execution_stream(
        self: &Arc<Self>,
    ) -> Result<BoundExecutionStream<R>, ExecutionStreamCreationError<R::Error>> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("create an execution stream")
            .map_err(ExecutionStreamCreationError::Contract)?;
        if self.is_poisoned() {
            return Err(ExecutionStreamCreationError::Contract(invalid_resource(
                "poisoned logical sequence cannot create an execution stream",
            )));
        }
        self.validate_runtime("execution stream creation preflight")
            .map_err(ExecutionStreamCreationError::Contract)?;
        let stream = self
            .request
            .plan
            .runtime()
            .create_stream()
            .map_err(ExecutionStreamCreationError::Runtime)?;
        self.validate_runtime("execution stream creation completion")
            .map_err(ExecutionStreamCreationError::Contract)?;
        if self.request.plan.runtime().stream_state(&stream) != StreamState::Ready {
            return Err(ExecutionStreamCreationError::Contract(invalid_resource(
                "new execution stream is not ready",
            )));
        }
        Ok(BoundExecutionStream {
            runtime: Arc::clone(self.request.plan.runtime()),
            coordinator_id: self.coordinator_id(),
            sequence_authority: self.sequence_authority(),
            stream: Some(stream),
            state: BoundExecutionStreamState::Ready,
            sequence_recovery: Arc::clone(&self.sequence_recovery),
            sequence_dispatch_gate: Arc::clone(&self.sequence_dispatch_gate),
            abandoned_sequence: None,
            resources: Arc::clone(self),
        })
    }

    pub fn activate<'resources, 'exec>(
        &'resources self,
        stream: &'exec mut BoundExecutionStream<R>,
    ) -> Result<ActiveSequencePermit<'resources, 'exec, R>, VNextError> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("activate an execution stream")?;
        if self.is_poisoned() {
            return Err(invalid_resource(
                "poisoned logical sequence cannot be activated",
            ));
        }
        self.validate_runtime("logical sequence activation")?;
        if !Arc::ptr_eq(self.request.plan.runtime(), &stream.runtime)
            || !std::ptr::eq(self, Arc::as_ref(&stream.resources))
            || stream.coordinator_id != self.coordinator_id()
            || stream.sequence_authority != self.sequence_authority()
            || !Arc::ptr_eq(&self.sequence_recovery, &stream.sequence_recovery)
            || !Arc::ptr_eq(&self.sequence_dispatch_gate, &stream.sequence_dispatch_gate)
        {
            return Err(invalid_resource(
                "execution stream belongs to another logical sequence authority",
            ));
        }
        if stream.state != BoundExecutionStreamState::Ready
            || stream.abandoned_sequence.is_some()
            || self.request.plan.runtime().stream_state(stream.stream()) != StreamState::Ready
        {
            return Err(invalid_resource(
                "logical sequence activation requires one core-ready stream",
            ));
        }
        let mut authority_source = self.lock_authority_source()?;
        let selecting_legacy = match *authority_source {
            SequenceExecutionAuthoritySource::Unselected => true,
            SequenceExecutionAuthoritySource::LegacyStream => false,
            SequenceExecutionAuthoritySource::SequenceSession => {
                return Err(invalid_resource(
                    "logical sequence execution authority is permanently selected for sequence sessions",
                ));
            }
            SequenceExecutionAuthoritySource::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ));
            }
        };
        let backing_snapshot = self.backing_snapshot()?;
        let epoch = match self.next_activation_epoch.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |epoch| epoch.checked_add(1).filter(|next| *next <= (u64::MAX >> 2)),
        ) {
            Ok(epoch) => epoch,
            Err(_) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource("active sequence epoch space is exhausted"));
            }
        };
        let active_state = sequence_slot_active(epoch);
        if let Err(actual) =
            self.state
                .compare_exchange(0, active_state, Ordering::AcqRel, Ordering::Acquire)
        {
            if selecting_legacy {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
            }
            return Err(if sequence_slot_is_poisoned(actual) {
                invalid_resource("logical sequence was abandoned and is poisoned")
            } else {
                invalid_resource("logical sequence already owns an active stream")
            });
        }
        let slot = self.sequence_authority().sparse_id();
        let recovery_metadata = AbandonedSequenceMetadata {
            plan: self.request.plan.evidence(),
            sequence_authority: self.sequence_authority(),
            run_id: self.run_id().clone(),
            request_id: self.request_id().clone(),
            slot,
            activation_epoch: epoch,
            runtime_implementation_fingerprint: self
                .request
                .plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            state: Arc::clone(&self.state),
            sequence_dispatch_gate: Arc::clone(&self.sequence_dispatch_gate),
            drained: false,
        };
        let recovery_key = recovery_metadata.key();
        self.sequence_recovery.register(recovery_metadata);
        stream.abandoned_sequence = Some(recovery_key);
        stream.state = BoundExecutionStreamState::InUse;
        *authority_source = SequenceExecutionAuthoritySource::LegacyStream;
        Ok(ActiveSequencePermit {
            resources: self,
            backing_snapshot,
            epoch,
            state: Arc::clone(&self.state),
            stream,
            runtime_fingerprint: self
                .request
                .plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            stream_drained: false,
            completed: false,
        })
    }

    pub fn recover_abandoned_sequence(
        &self,
    ) -> Result<ActiveSequenceAbortReceipt, AbandonedSequenceRecoveryError<R::Error>> {
        self.sequence_recovery.recover(
            self.request.plan.runtime(),
            self.sequence_authority().sparse_id(),
        )
    }
}

/// Non-cloneable guard for an admitted active-sequence slot. Dispatch borrows
/// this permit; the sequence owner retains it until all asynchronous work is
/// synchronized or cancelled.
#[must_use = "an active sequence permit must live until asynchronous work is complete"]
pub struct ActiveSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    resources: &'resources AdmittedSequenceResources<R>,
    backing_snapshot: Arc<SequenceBackingSnapshot<R>>,
    epoch: u64,
    state: Arc<AtomicU64>,
    stream: &'exec mut BoundExecutionStream<R>,
    runtime_fingerprint: String,
    stream_drained: bool,
    completed: bool,
}

impl<'resources, 'exec, R> ActiveSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    pub fn resources(&self) -> &'resources AdmittedSequenceResources<R> {
        self.resources
    }

    pub fn run_id(&self) -> &RunId {
        self.resources.run_id()
    }

    pub fn request_id(&self) -> &RequestIdentity {
        self.resources.request_id()
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.resources.sequence_authority()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.resources.coordinator_id()
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.backing_snapshot.backing_slices()
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_fingerprint
    }

    pub(crate) fn with_runtime_and_stream<T>(
        &mut self,
        action: impl FnOnce(&R, &mut R::Stream) -> T,
    ) -> Result<T, VNextError> {
        if self.stream.state != BoundExecutionStreamState::InUse {
            return Err(invalid_resource(
                "operation dispatch requires one core-owned in-use stream",
            ));
        }
        let _dispatch_guard = enter_sequence_dispatch(&self.resources.sequence_dispatch_gate)?;
        Ok(action(
            self.resources.request.plan.runtime(),
            self.stream.stream_mut(),
        ))
    }

    /// Consumes dispatch authority before draining the exact bound stream.
    /// Successful synchronization returns a different typestate that cannot
    /// be passed back to `OperationDispatch`.
    pub fn synchronize(
        mut self,
    ) -> Result<
        SynchronizedSequencePermit<'resources, 'exec, R>,
        SequenceSynchronizationFailure<'resources, 'exec, R>,
    > {
        let preflight = self
            .resources
            .validate_runtime("sequence synchronization preflight")
            .and_then(|()| {
                if self
                    .resources
                    .request
                    .plan
                    .runtime()
                    .descriptor()
                    .runtime_implementation_fingerprint
                    == self.runtime_fingerprint
                {
                    Ok(())
                } else {
                    Err(invalid_resource(
                        "sequence synchronization runtime differs from its activation snapshot",
                    ))
                }
            });

        // Draining is attempted even when descriptor validation fails. The
        // stream/runtime pair is privately bound, while skipping the drain
        // could make later buffer quarantine unsafe.
        let runtime_error = match self
            .resources
            .request
            .plan
            .runtime()
            .synchronize(self.stream.stream_mut())
        {
            Ok(()) => None,
            Err(error) => Some(error),
        };
        let stream_ready = self
            .resources
            .request
            .plan
            .runtime()
            .stream_state(self.stream.stream())
            == StreamState::Ready;
        self.stream_drained = runtime_error.is_none() && stream_ready;
        if self.stream_drained {
            self.stream
                .sequence_recovery
                .set_drained((self.sequence_authority().sparse_id(), self.epoch), true);
        }
        let completion = self
            .resources
            .validate_runtime("sequence synchronization completion")
            .and_then(|()| {
                if stream_ready {
                    Ok(())
                } else {
                    Err(invalid_resource(
                        "sequence synchronization did not return the bound stream to ready",
                    ))
                }
            });
        let error = preflight
            .err()
            .map(SequenceSynchronizationError::Contract)
            .or_else(|| runtime_error.map(SequenceSynchronizationError::Runtime))
            .or_else(|| completion.err().map(SequenceSynchronizationError::Contract));
        if let Some(error) = error {
            return Err(SequenceSynchronizationFailure {
                permit: Some(self),
                error,
            });
        }
        self.stream.state = BoundExecutionStreamState::Ready;
        Ok(SynchronizedSequencePermit { permit: Some(self) })
    }
}

#[derive(Debug)]
pub enum SequenceSynchronizationError<E> {
    Contract(VNextError),
    Runtime(E),
}

/// Retry owner for a failed stream drain. It intentionally does not expose
/// the active dispatch permit, so no operation can be submitted between a
/// failed synchronization attempt and its retry.
#[must_use = "failed sequence synchronization must be retried or retained"]
pub struct SequenceSynchronizationFailure<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    permit: Option<ActiveSequencePermit<'resources, 'exec, R>>,
    error: SequenceSynchronizationError<R::Error>,
}

impl<R> fmt::Debug for SequenceSynchronizationFailure<'_, '_, R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SequenceSynchronizationFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<'resources, 'exec, R> SequenceSynchronizationFailure<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    pub fn error(&self) -> &SequenceSynchronizationError<R::Error> {
        &self.error
    }

    pub fn retry(
        mut self,
    ) -> Result<
        SynchronizedSequencePermit<'resources, 'exec, R>,
        SequenceSynchronizationFailure<'resources, 'exec, R>,
    > {
        self.permit
            .take()
            .expect("synchronization failure owns its active permit")
            .synchronize()
    }
}

/// Stream-drained typestate. It has no dispatch API and must choose exactly
/// one terminal slot disposition.
#[must_use = "a synchronized sequence must be completed or aborted"]
pub struct SynchronizedSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    permit: Option<ActiveSequencePermit<'resources, 'exec, R>>,
}

impl<R> SynchronizedSequencePermit<'_, '_, R>
where
    R: DeviceRuntime,
{
    pub fn complete(mut self) -> Result<ActiveSequenceCompletionReceipt, VNextError> {
        let mut permit = self
            .permit
            .take()
            .expect("synchronized sequence owns its active permit");
        let sequence_poisoned =
            sequence_dispatch_is_poisoned(&permit.resources.sequence_dispatch_gate);
        let terminal_state = if sequence_poisoned {
            sequence_slot_poisoned_drained(permit.epoch)
        } else {
            0
        };
        permit
            .state
            .compare_exchange(
                sequence_slot_active(permit.epoch),
                terminal_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_resource("active sequence epoch is no longer completable"))?;
        permit
            .stream
            .sequence_recovery
            .clear((permit.sequence_authority().sparse_id(), permit.epoch));
        permit.stream.abandoned_sequence = None;
        permit.stream.state = BoundExecutionStreamState::Ready;
        permit.completed = true;
        if sequence_poisoned {
            return Err(invalid_resource(
                "sequence cannot complete successfully after its dispatch authority was poisoned",
            ));
        }
        Ok(ActiveSequenceCompletionReceipt {
            plan: permit.resources.request.plan.evidence(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.epoch,
            runtime_implementation_fingerprint: permit.runtime_fingerprint.clone(),
        })
    }

    /// Produces abort evidence only after the exact bound stream was drained.
    /// Only this exact logical sequence remains poisoned after abort.
    pub fn abort(mut self) -> Result<ActiveSequenceAbortReceipt, VNextError> {
        let mut permit = self
            .permit
            .take()
            .expect("synchronized sequence owns its active permit");
        permit
            .state
            .compare_exchange(
                sequence_slot_active(permit.epoch),
                sequence_slot_poisoned_drained(permit.epoch),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_resource("active sequence epoch is no longer abortable"))?;
        permit
            .resources
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        permit
            .stream
            .sequence_recovery
            .clear((permit.sequence_authority().sparse_id(), permit.epoch));
        permit.stream.abandoned_sequence = None;
        permit.stream.state = BoundExecutionStreamState::Ready;
        permit.completed = true;
        Ok(ActiveSequenceAbortReceipt {
            plan: permit.resources.request.plan.evidence(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.epoch,
            runtime_implementation_fingerprint: permit.runtime_fingerprint.clone(),
            disposition: ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
        })
    }
}

/// Core-signed evidence that synchronization succeeded and the exact active
/// slot epoch was atomically cleared. It is trusted output and deliberately
/// cannot be deserialized or constructed by a caller.
#[derive(Debug, Serialize)]
#[must_use = "sequence completion evidence must be recorded by execution"]
pub struct ActiveSequenceCompletionReceipt {
    plan: TrustedPlanRuntimeEvidence,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
}

impl ActiveSequenceCompletionReceipt {
    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }
}

impl<R> Drop for ActiveSequencePermit<'_, '_, R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.completed {
            let poisoned_state = if self.stream_drained {
                sequence_slot_poisoned_drained(self.epoch)
            } else {
                sequence_slot_poisoned_undrained(self.epoch)
            };
            let result = self.state.compare_exchange(
                sequence_slot_active(self.epoch),
                poisoned_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            debug_assert!(result.is_ok(), "active sequence slot guard lost ownership");
            if result.is_ok() {
                self.resources
                    .sequence_dispatch_gate
                    .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
                self.stream.state = BoundExecutionStreamState::Poisoned;
                self.stream.sequence_recovery.set_drained(
                    (self.sequence_authority().sparse_id(), self.epoch),
                    self.stream_drained,
                );
            }
        }
    }
}
