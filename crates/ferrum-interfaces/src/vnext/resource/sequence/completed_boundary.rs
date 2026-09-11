use super::{
    invalid_resource, ActiveSequenceFrame, ActiveSequenceSessionState, Arc, DeviceRuntime,
    PreparedSequenceStateTransfer, SequenceBackingGeneration, SequenceSessionEpoch,
    SequenceSessionFingerprint, SequenceSessionPhase, SequenceSessionSlotState,
    SequenceStateTransferKind, VNextError,
};
use crate::vnext::resource::{
    ParticipantFlightPhase, PreparedStepSubmissionWave, SessionFrameHold,
    StepParticipantFrameAssignment, StepParticipantRetirementDisposition, SubmissionWavePurpose,
};
use crate::vnext::{
    BatchInvocationId, BatchStepId, ExecutionFrameId, PlanHash, SuccessfulWaveCompletionSeal,
    TokenSpanWork,
};
use std::sync::MutexGuard;

/// Evidence about one exact completed sequence state. This is not a capture
/// permit: consumers must separately reserve state access and bind the current
/// frontier and backing. A backing generation alone does not detect decode
/// writes performed without an extension.
pub(crate) struct CompletedSequenceBoundary {
    plan_hash: PlanHash,
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    frame_id: ExecutionFrameId,
    epoch: SequenceSessionEpoch,
    session_fingerprint: SequenceSessionFingerprint,
    backing_generation: SequenceBackingGeneration,
    start_token: usize,
    end_token: usize,
    full_input: Arc<[u32]>,
}

impl std::fmt::Debug for CompletedSequenceBoundary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CompletedSequenceBoundary")
            .field("plan_hash", &self.plan_hash)
            .field("batch_step_id", &self.batch_step_id)
            .field("batch_invocation_id", &self.batch_invocation_id)
            .field("frame_id", &self.frame_id)
            .field("epoch", &self.epoch)
            .field("session_fingerprint", &self.session_fingerprint)
            .field("backing_generation", &self.backing_generation)
            .field("start_token", &self.start_token)
            .field("end_token", &self.end_token)
            .field("full_input_tokens", &self.full_input.len())
            .finish()
    }
}

impl CompletedSequenceBoundary {
    pub(crate) fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub(crate) fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub(crate) fn batch_invocation_id(&self) -> BatchInvocationId {
        self.batch_invocation_id
    }

    pub(crate) fn frame_id(&self) -> ExecutionFrameId {
        self.frame_id
    }

    pub(crate) fn epoch(&self) -> SequenceSessionEpoch {
        self.epoch
    }

    pub(crate) fn session_fingerprint(&self) -> &SequenceSessionFingerprint {
        &self.session_fingerprint
    }

    pub(crate) fn backing_generation(&self) -> SequenceBackingGeneration {
        self.backing_generation
    }

    pub(crate) fn completed_tokens(&self) -> usize {
        self.end_token
    }

    pub(crate) fn token_prefix(&self) -> &[u32] {
        &self.full_input[..self.end_token]
    }

    pub(crate) fn full_input(&self) -> &Arc<[u32]> {
        &self.full_input
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) enum SequenceCompletedFrontier {
    #[default]
    Fresh,
    Proven(Arc<CompletedSequenceBoundary>),
    Unproven,
}

impl SequenceCompletedFrontier {
    fn validate_continuation(
        &self,
        start: usize,
        full_input: &[u32],
        plan_hash: &PlanHash,
    ) -> Result<(), VNextError> {
        let matches = match self {
            Self::Fresh => start == 0,
            Self::Proven(previous) => {
                previous.plan_hash == *plan_hash
                    && start == previous.end_token
                    && full_input.get(..start) == Some(previous.token_prefix())
            }
            Self::Unproven => false,
        };
        if !matches {
            return Err(invalid_resource(
                "completed token work does not continue the proven exact sequence prefix",
            ));
        }
        Ok(())
    }
}

/// The Step owns the sole consumable success candidate. Failed signing cannot
/// leave an earlier candidate available for a later, otherwise normal Commit.
#[derive(Default)]
pub(crate) enum StepCompletedBoundarySlot {
    #[default]
    Empty,
    Succeeded(StepCompletedBoundaryProof),
    Invalid,
}

impl StepCompletedBoundarySlot {
    pub(crate) fn proof(&self) -> Option<&StepCompletedBoundaryProof> {
        match self {
            Self::Succeeded(proof) => Some(proof),
            Self::Empty | Self::Invalid => None,
        }
    }

    pub(crate) fn consume(&mut self) {
        *self = Self::Invalid;
    }
}

pub(crate) struct StepCompletedBoundaryProof {
    batch_step_id: BatchStepId,
    participants: Vec<Option<Arc<CompletedSequenceBoundary>>>,
}

pub(crate) fn record_completed_wave<R: DeviceRuntime>(
    wave: &PreparedStepSubmissionWave<R>,
    seal: &SuccessfulWaveCompletionSeal,
) -> Result<(), VNextError> {
    let step = wave.step_resources();
    // This lock is outside all session locks. Finalization uniquely owns the
    // Step and uses get_mut, so there is no session -> proof-lock inversion.
    let mut slot = step
        .completed_boundary
        .lock()
        .map_err(|_| invalid_resource("step completed-boundary mutex is poisoned"))?;
    if !matches!(*slot, StepCompletedBoundarySlot::Empty) {
        *slot = StepCompletedBoundarySlot::Invalid;
        return Err(invalid_resource(
            "complete-model-step success may be recorded only once",
        ));
    }
    // Fail closed even if validation unwinds before returning a Result.
    *slot = StepCompletedBoundarySlot::Invalid;
    let proof = validate_completed_wave(wave, seal)?;
    *slot = StepCompletedBoundarySlot::Succeeded(proof);
    Ok(())
}

fn validate_completed_wave<R: DeviceRuntime>(
    wave: &PreparedStepSubmissionWave<R>,
    seal: &SuccessfulWaveCompletionSeal,
) -> Result<StepCompletedBoundaryProof, VNextError> {
    let step = wave.step_resources();
    let plan = &step.participants[0].session.resources().request.plan;
    if wave.purpose() != SubmissionWavePurpose::FullPlan
        || step.finalized
        || seal.batch_step_id() != step.batch_step_id()
        || seal.batch_invocation_id() != wave.batch_invocation_id()
        || seal.lane_id() != wave.execution_lane_id()
        || seal.wave_fingerprint() != wave.fingerprint()
        || wave.claimed_backing().plan_hash() != plan.plan_hash()
        || wave.node_count() != plan.nodes().len()
        || wave.claimed_backing().work_shape() != step.work_shape()
        || wave.nodes().iter().enumerate().any(|(index, node)| {
            node.plan_node_index() != index || node.work_shape() != step.work_shape()
        })
    {
        return Err(invalid_resource(
            "successful wave does not prove this exact full-plan step",
        ));
    }
    let frames = step
        .participants
        .iter()
        .map(|participant| {
            StepParticipantFrameAssignment::new(
                participant.session.sequence_authority(),
                participant.session.request_authority(),
                participant.frame.frame_id,
            )
        })
        .collect::<Vec<_>>();
    for node in wave.nodes() {
        if node.participant_frames() != frames
            || !node
                .participant_session_identities()
                .eq(step.participants.iter().map(|participant| {
                    (
                        participant.session.epoch(),
                        participant.session.fingerprint(),
                    )
                }))
            || !node
                .participants()
                .zip(&step.participants)
                .all(|(actual, expected)| Arc::ptr_eq(actual, expected.session.resources()))
        {
            return Err(invalid_resource(
                "successful wave differs from its exact participant sessions and frames",
            ));
        }
    }
    // Same canonical participant ordering as frame admission/finalization.
    let states = step
        .participants
        .iter()
        .map(|participant| {
            participant
                .session
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut participants = Vec::with_capacity(step.participants.len());
    for ((participant, state), token_span) in step
        .participants
        .iter()
        .zip(&states)
        .zip(step.work_shape().participant_work())
    {
        let SequenceSessionSlotState::Active(active) = &**state else {
            return Err(invalid_resource(
                "completed wave session is no longer active",
            ));
        };
        validate_active_frame(active, &participant.frame)?;
        if active.phase == SequenceSessionPhase::Poisoned
            || active.state_transfer.is_reserved()
            || active.submission_wave_flight != Some(ParticipantFlightPhase::InFlight)
            || !active.participant_flights.is_empty()
        {
            return Err(invalid_resource(
                "completed wave sequence state is unavailable",
            ));
        }
        // The session slot precedes backing in the common lock order. Frames
        // prevent extension, but still verify the exact captured generation.
        let backing = participant.session.resources().lock_backing_state()?;
        if !Arc::ptr_eq(&backing.current, &participant.backing_snapshot) {
            return Err(invalid_resource(
                "completed wave backing generation changed",
            ));
        }
        participants.push(boundary_from_span(
            active,
            token_span.token_span(),
            plan.plan_hash(),
            wave.batch_step_id(),
            wave.batch_invocation_id(),
            participant.frame.frame_id,
            backing.current.generation(),
        )?);
    }
    Ok(StepCompletedBoundaryProof {
        batch_step_id: step.batch_step_id(),
        participants,
    })
}

fn boundary_from_span(
    active: &ActiveSequenceSessionState,
    span: &TokenSpanWork,
    plan_hash: &PlanHash,
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    frame_id: ExecutionFrameId,
    backing_generation: SequenceBackingGeneration,
) -> Result<Option<Arc<CompletedSequenceBoundary>>, VNextError> {
    let Some(tokens) = span.checkpoint_tokens() else {
        return Ok(None);
    };
    let range = span.immediate_token_range();
    let start = usize::try_from(range.start)
        .map_err(|_| invalid_resource("completed token start exceeds usize"))?;
    let end = usize::try_from(range.end)
        .map_err(|_| invalid_resource("completed token end exceeds usize"))?;
    if start >= end || end > tokens.len() {
        return Err(invalid_resource("completed token span is out of bounds"));
    }
    active
        .completed_boundary
        .validate_continuation(start, tokens, plan_hash)?;
    Ok(Some(Arc::new(CompletedSequenceBoundary {
        plan_hash: plan_hash.clone(),
        batch_step_id,
        batch_invocation_id,
        frame_id,
        epoch: active.epoch,
        session_fingerprint: active.fingerprint.clone(),
        backing_generation,
        start_token: start,
        end_token: end,
        full_input: Arc::clone(tokens),
    })))
}

fn validate_active_frame(
    active: &ActiveSequenceSessionState,
    hold: &SessionFrameHold,
) -> Result<(), VNextError> {
    if active.epoch != hold.epoch
        || active.fingerprint != hold.fingerprint
        || active.active_frame
            != Some(ActiveSequenceFrame {
                frame_id: hold.frame_id,
                batch_step_id: hold.batch_step_id,
            })
    {
        return Err(invalid_resource("completed wave session frame is stale"));
    }
    Ok(())
}

pub(in crate::vnext::resource) fn completed_frontier_updates(
    holds: &[&mut SessionFrameHold],
    states: &[MutexGuard<'_, SequenceSessionSlotState>],
    dispositions: &[StepParticipantRetirementDisposition],
    completed: Option<&StepCompletedBoundaryProof>,
) -> Result<Vec<Option<SequenceCompletedFrontier>>, VNextError> {
    if completed.is_some_and(|proof| proof.participants.len() != holds.len()) {
        return Err(invalid_resource("completed step participant count changed"));
    }
    holds
        .iter()
        .zip(states)
        .zip(dispositions)
        .enumerate()
        .map(|(index, ((hold, state), disposition))| {
            use StepParticipantRetirementDisposition as Disposition;
            match disposition {
                Disposition::RolledBackUnsubmitted => Ok(None),
                Disposition::Aborted | Disposition::DiscardedCancelled => {
                    Ok(Some(SequenceCompletedFrontier::Unproven))
                }
                Disposition::Committed => {
                    let Some(proof) = completed else {
                        return Ok(Some(SequenceCompletedFrontier::Unproven));
                    };
                    if proof.batch_step_id != hold.batch_step_id {
                        return Err(invalid_resource(
                            "completed proof belongs to a different step",
                        ));
                    }
                    let Some(boundary) = &proof.participants[index] else {
                        return Ok(Some(SequenceCompletedFrontier::Unproven));
                    };
                    let SequenceSessionSlotState::Active(active) = &**state else {
                        return Err(invalid_resource(
                            "completed step session is no longer active",
                        ));
                    };
                    validate_active_frame(active, hold)?;
                    if boundary.batch_step_id != hold.batch_step_id
                        || boundary.frame_id != hold.frame_id
                        || boundary.epoch != hold.epoch
                        || boundary.session_fingerprint != hold.fingerprint
                    {
                        return Err(invalid_resource("completed proof session identity changed"));
                    }
                    active.completed_boundary.validate_continuation(
                        boundary.start_token,
                        &boundary.full_input,
                        &boundary.plan_hash,
                    )?;
                    Ok(Some(SequenceCompletedFrontier::Proven(Arc::clone(
                        boundary,
                    ))))
                }
            }
        })
        .collect()
}

impl<R: DeviceRuntime> PreparedSequenceStateTransfer<R> {
    /// Returns the current proof only while this reservation prevents another
    /// frame from mutating its state. Retaining the returned host metadata does
    /// not extend that permission after the reservation is dropped.
    pub(crate) fn completed_boundary(&self) -> Result<Arc<CompletedSequenceBoundary>, VNextError> {
        if self.kind() != SequenceStateTransferKind::CaptureRead {
            return Err(invalid_resource(
                "completed capture boundary requires a read reservation",
            ));
        }
        self.with_reserved_frontier(|frontier| match frontier {
            SequenceCompletedFrontier::Proven(boundary)
                if boundary.epoch == self.session().epoch()
                    && boundary.session_fingerprint == *self.session().fingerprint()
                    && boundary.backing_generation == self.backing().generation() =>
            {
                Ok(Arc::clone(boundary))
            }
            _ => Err(invalid_resource(
                "reserved sequence has no current completed boundary",
            )),
        })
    }

    pub(crate) fn ensure_fresh_restore_target(&self) -> Result<(), VNextError> {
        if self.kind() != SequenceStateTransferKind::RestoreWrite {
            return Err(invalid_resource(
                "fresh restore target requires a write reservation",
            ));
        }
        self.with_reserved_frontier(|frontier| {
            if matches!(frontier, SequenceCompletedFrontier::Fresh) {
                Ok(())
            } else {
                Err(invalid_resource(
                    "restore target has already committed sequence work",
                ))
            }
        })
    }

    fn with_reserved_frontier<T>(
        &self,
        inspect: impl FnOnce(&SequenceCompletedFrontier) -> Result<T, VNextError>,
    ) -> Result<T, VNextError> {
        if self.session().resources().is_poisoned() {
            return Err(invalid_resource(
                "poisoned sequence has no usable completed frontier",
            ));
        }
        let state = self
            .session()
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let SequenceSessionSlotState::Active(active) = &*state else {
            return Err(invalid_resource(
                "state transfer session is no longer active",
            ));
        };
        self.ensure_active_reservation(active)?;
        if active.phase != SequenceSessionPhase::Open
            || active.active_frame.is_some()
            || active.has_participant_flights()
        {
            return Err(invalid_resource("state transfer frontier is unavailable"));
        }
        let backing = self.session().resources().lock_backing_state()?;
        if !Arc::ptr_eq(&backing.current, self.backing()) {
            return Err(invalid_resource(
                "state transfer backing generation changed",
            ));
        }
        inspect(&active.completed_boundary)
    }
}
