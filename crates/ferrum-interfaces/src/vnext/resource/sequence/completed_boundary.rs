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
    BatchInvocationId, BatchStepId, CheckpointCaptureAttemptId, CheckpointInputDependency,
    CheckpointTokenSpanConstraint, ExecutionFrameId, PlanHash, StateTransferIdentity,
    SuccessfulWaveCompletionSeal, TokenSpanWork,
};
use serde::Serialize;
use std::sync::MutexGuard;

mod imported;

/// Host-only provenance. Imported state was copied by one exact native restore;
/// it does not claim that a model frame executed on the target sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) enum CompletedSequenceProvenance {
    FullPlan {
        batch_step_id: BatchStepId,
        batch_invocation_id: BatchInvocationId,
        frame_id: ExecutionFrameId,
    },
    ImportedCheckpoint {
        capture_attempt: CheckpointCaptureAttemptId,
        #[serde(serialize_with = "serialize_restore_identity")]
        restore: Arc<StateTransferIdentity>,
    },
}

/// Restrictions established by a native import persist through every later
/// completed model frame. They are independent of which frame last wrote state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct SequenceCheckpointContinuation {
    input_dependency: CheckpointInputDependency,
    suffix_constraints: Vec<CheckpointTokenSpanConstraint>,
}

fn serialize_restore_identity<S: serde::Serializer>(
    identity: &Arc<StateTransferIdentity>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    identity.as_ref().serialize(serializer)
}

/// Evidence about one exact completed sequence state. This is not a capture
/// permit: consumers must separately reserve state access and bind the current
/// frontier and backing. A backing generation alone does not detect decode
/// writes performed without an extension.
pub(crate) struct CompletedSequenceBoundary {
    plan_hash: PlanHash,
    provenance: CompletedSequenceProvenance,
    continuation: Option<Arc<SequenceCheckpointContinuation>>,
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
            .field("provenance", &self.provenance)
            .field("continuation", &self.continuation)
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

    pub(crate) fn provenance(&self) -> &CompletedSequenceProvenance {
        &self.provenance
    }

    pub(crate) fn continuation_contract(&self) -> Option<&SequenceCheckpointContinuation> {
        self.continuation.as_deref()
    }

    pub(crate) fn batch_step_id(&self) -> Option<BatchStepId> {
        match self.provenance {
            CompletedSequenceProvenance::FullPlan { batch_step_id, .. } => Some(batch_step_id),
            CompletedSequenceProvenance::ImportedCheckpoint { .. } => None,
        }
    }

    pub(crate) fn batch_invocation_id(&self) -> Option<BatchInvocationId> {
        match self.provenance {
            CompletedSequenceProvenance::FullPlan {
                batch_invocation_id,
                ..
            } => Some(batch_invocation_id),
            CompletedSequenceProvenance::ImportedCheckpoint { .. } => None,
        }
    }

    pub(crate) fn frame_id(&self) -> Option<ExecutionFrameId> {
        match self.provenance {
            CompletedSequenceProvenance::FullPlan { frame_id, .. } => Some(frame_id),
            CompletedSequenceProvenance::ImportedCheckpoint { .. } => None,
        }
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

    /// The successful model partition that made this boundary capturable.
    /// Imports retain this eligibility span without claiming a target frame.
    pub(crate) fn capture_span_start(&self) -> usize {
        self.start_token
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
    /// A later invocation cannot replace work already checked for an imported
    /// participant at Step admission. Reuse the Step's retained token evidence
    /// even when the equivalent invocation shape omitted process-local data.
    pub(in crate::vnext::resource) fn bind_imported_invocation_work(
        &self,
        admitted: &TokenSpanWork,
        requested: &TokenSpanWork,
        plan_hash: &PlanHash,
    ) -> Result<Option<TokenSpanWork>, VNextError> {
        if !matches!(self, Self::Proven(previous) if previous.continuation.is_some()) {
            return Ok(None);
        }
        if admitted != requested {
            return Err(invalid_resource(
                "imported invocation token work differs from its admitted Step span",
            ));
        }
        self.validate_imported_work(admitted, plan_hash)?;
        Ok(Some(admitted.clone()))
    }

    /// Imported state carries semantic restrictions on every later execution.
    /// Check these while frame admission holds the session lock, before a
    /// caller can acquire any device submission authority for the new work.
    pub(in crate::vnext::resource) fn validate_imported_work(
        &self,
        span: &TokenSpanWork,
        plan_hash: &PlanHash,
    ) -> Result<(), VNextError> {
        let Self::Proven(previous) = self else {
            return Ok(());
        };
        if previous.continuation.is_none() {
            return Ok(());
        }
        let tokens = span.checkpoint_tokens().ok_or_else(|| {
            invalid_resource("imported state requires exact token evidence before execution")
        })?;
        let range = span.immediate_token_range();
        let start = usize::try_from(range.start)
            .map_err(|_| invalid_resource("continuation token start exceeds usize"))?;
        let end = usize::try_from(range.end)
            .map_err(|_| invalid_resource("continuation token end exceeds usize"))?;
        self.validate_continuation(start, end, tokens, plan_hash)
    }

    fn validate_continuation(
        &self,
        start: usize,
        end: usize,
        full_input: &[u32],
        plan_hash: &PlanHash,
    ) -> Result<(), VNextError> {
        let matches = match self {
            Self::Fresh => start == 0,
            Self::Proven(previous) => {
                previous.plan_hash == *plan_hash
                    && start == previous.end_token
                    && full_input.get(..start) == Some(previous.token_prefix())
                    && match previous.continuation.as_deref() {
                        None => true,
                        Some(contract) => {
                            (contract.input_dependency
                                != CheckpointInputDependency::EntireTokenInput
                                || previous.full_input.as_ref() == full_input)
                                && end
                                    .checked_sub(start)
                                    .and_then(|length| u64::try_from(length).ok())
                                    .is_some_and(|length| {
                                        contract
                                            .suffix_constraints
                                            .iter()
                                            .all(|constraint| constraint.permits(length))
                                    })
                        }
                    }
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
        .validate_continuation(start, end, tokens, plan_hash)?;
    Ok(Some(Arc::new(CompletedSequenceBoundary {
        plan_hash: plan_hash.clone(),
        provenance: CompletedSequenceProvenance::FullPlan {
            batch_step_id,
            batch_invocation_id,
            frame_id,
        },
        continuation: match &active.completed_boundary {
            SequenceCompletedFrontier::Proven(previous) => previous.continuation.clone(),
            _ => None,
        },
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
                    let SequenceSessionSlotState::Active(active) = &**state else {
                        return Err(invalid_resource(
                            "completed step session is no longer active",
                        ));
                    };
                    let missing_proof = || {
                        if matches!(&active.completed_boundary,
                            SequenceCompletedFrontier::Proven(previous) if previous.continuation.is_some())
                        {
                            Err(invalid_resource(
                                "imported continuation cannot commit without complete FullPlan proof",
                            ))
                        } else {
                            Ok(Some(SequenceCompletedFrontier::Unproven))
                        }
                    };
                    let Some(proof) = completed else {
                        return missing_proof();
                    };
                    if proof.batch_step_id != hold.batch_step_id {
                        return Err(invalid_resource(
                            "completed proof belongs to a different step",
                        ));
                    }
                    let Some(boundary) = &proof.participants[index] else {
                        return missing_proof();
                    };
                    validate_active_frame(active, hold)?;
                    if boundary.batch_step_id() != Some(hold.batch_step_id)
                        || boundary.frame_id() != Some(hold.frame_id)
                        || boundary.epoch != hold.epoch
                        || boundary.session_fingerprint != hold.fingerprint
                    {
                        return Err(invalid_resource("completed proof session identity changed"));
                    }
                    active.completed_boundary.validate_continuation(
                        boundary.start_token,
                        boundary.end_token,
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
        self.with_reserved_state(|active| inspect(&active.completed_boundary))
    }

    fn with_reserved_state<T>(
        &self,
        inspect: impl FnOnce(&mut ActiveSequenceSessionState) -> Result<T, VNextError>,
    ) -> Result<T, VNextError> {
        if self.session().resources().is_poisoned() {
            return Err(invalid_resource(
                "poisoned sequence has no usable completed frontier",
            ));
        }
        let mut state = self
            .session()
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let SequenceSessionSlotState::Active(active) = &mut *state else {
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
        inspect(active)
    }
}
