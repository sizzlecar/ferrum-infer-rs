mod forwarded_input;
pub use forwarded_input::SubmissionWaveInputForward;

use super::{
    invalid_resource, ActiveSequenceFrame, AdmittedSequenceResources, Arc, BatchWorkShape,
    ClaimedBackingTransaction, ClaimedSubmissionWaveBacking, DeviceRuntime, ExecutionLane,
    ExecutionLaneId, ParticipantFlightPhase, PreparedStepSubmissionWave, SequenceBackingSnapshot,
    SequenceFrameCaptureCandidate, SequenceSessionEpoch, SequenceSessionFingerprint,
    SequenceSessionPhase, SequenceSessionSlot, SequenceSessionSlotState, StepResourceLease,
    SubmissionWavePurpose, VNextError, Weak,
};
use crate::vnext::{
    BatchParticipantTokenSpan, CompletionReadbackBatchRequest, ElementType,
    SubmittedOperationReceipt, TokenSpanWork,
};

struct PredecessorParticipant<R: DeviceRuntime> {
    slot: Weak<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    frame: ActiveSequenceFrame,
    backing: Arc<SequenceBackingSnapshot<R>>,
    resources: Arc<AdmittedSequenceResources<R>>,
}

impl<R: DeviceRuntime> StepResourceLease<R> {
    /// A child may finish on device before the host retires its parent. Keep
    /// its completion slot intact until ordered retirement. Cancellation is
    /// participant-local: normal retirement discards that row while committing
    /// healthy peers. An aborted predecessor poisons the session instead.
    pub(crate) fn predecessor_is_retired(&self) -> Result<bool, VNextError> {
        let Some(predecessor) = &self.predecessor else {
            return Ok(true);
        };
        if self.participants.len() != predecessor.participant_count() {
            return Err(invalid_resource(
                "dependent step lost predecessor membership",
            ));
        }
        let states =
            self.participants
                .iter()
                .map(|participant| {
                    participant.session.slot.state.lock().map_err(|_| {
                        invalid_resource("dependent step session state mutex is poisoned")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        let mut ready = true;
        for (index, (participant, state)) in self.participants.iter().zip(&states).enumerate() {
            let SequenceSessionSlotState::Active(active) = &**state else {
                return Err(invalid_resource(
                    "dependent step session is no longer active",
                ));
            };
            let child = ActiveSequenceFrame {
                frame_id: participant.frame.frame_id,
                batch_step_id: self.batch_step_id(),
            };
            if active.epoch != participant.session.epoch()
                || &active.fingerprint != participant.session.fingerprint()
                || active.phase == SequenceSessionPhase::Poisoned
                || active.frames.get(child).is_none()
            {
                return Err(invalid_resource(
                    "dependent step was poisoned or lost its exact frame",
                ));
            }
            if active.frames.head() == Some(child) {
                continue;
            }
            if active.frames.head() != Some(predecessor.participant_frame(index))
                || active.frames.successor() != Some(child)
            {
                return Err(invalid_resource(
                    "dependent step has a different logical predecessor",
                ));
            }
            ready = false;
        }
        Ok(ready)
    }
}

/// One exact submitted full-plan wave that may supply a successor frame.
/// This is submission evidence, never successful-completion or token evidence.
/// It retains physical slots and their capacity without retaining the parent
/// Step, so normal parent retirement can still consume its unique Step owner.
#[must_use = "a predecessor retains backing capacity until consumed or dropped"]
pub struct SubmittedWavePredecessor<R: DeviceRuntime> {
    // Claims drop before their resource parents and execution lane.
    wave_backing: Arc<ClaimedSubmissionWaveBacking>,
    step_backing: Arc<ClaimedBackingTransaction>,
    participants: Vec<PredecessorParticipant<R>>,
    lane: Arc<ExecutionLane<R>>,
    lane_epoch: u64,
    receipt: SubmittedOperationReceipt,
}

impl<R: DeviceRuntime> std::fmt::Debug for SubmittedWavePredecessor<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SubmittedWavePredecessor")
            .field("submission", &self.receipt)
            .field("lane_epoch", &self.lane_epoch)
            .field("participant_count", &self.participants.len())
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> SubmittedWavePredecessor<R> {
    // Only the wave's successful fence-installed transition may call this.
    pub(super) fn capture(
        wave: &PreparedStepSubmissionWave<R>,
        receipt: SubmittedOperationReceipt,
    ) -> Result<Self, VNextError> {
        let step = wave.step_resources();
        let lane = step.execution_lane();
        let identity = receipt.batch_identity();
        if wave.purpose() != SubmissionWavePurpose::FullPlan
            || identity.batch_step_id() != step.batch_step_id()
            || identity.batch_invocation_id() != wave.batch_invocation_id()
            || identity.lane_id() != lane.id()
            || identity.node_count() != wave.node_count()
            || identity.plan_hash() != wave.claimed_backing().plan_hash()
            || !lane.is_reusable()
            || !lane.current_descriptor_matches_snapshot()
            || step.work_shape().participant_work().iter().any(|work| {
                let span = work.token_span();
                span.checkpoint_tokens().is_some()
                    || span.immediate_token_range().end != span.full_input_tokens()
            })
        {
            return Err(invalid_resource(
                "predecessor differs from its submitted full-plan wave or requires host checkpoint evidence",
            ));
        }
        let states =
            step.participants
                .iter()
                .map(|participant| {
                    participant.session.slot.state.lock().map_err(|_| {
                        invalid_resource("predecessor session state mutex is poisoned")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        for (participant, state) in step.participants.iter().zip(&states) {
            let frame = ActiveSequenceFrame {
                frame_id: participant.frame.frame_id,
                batch_step_id: step.batch_step_id(),
            };
            let SequenceSessionSlotState::Active(active) = &**state else {
                return Err(invalid_resource("predecessor session is no longer active"));
            };
            if active.epoch != participant.session.epoch()
                || &active.fingerprint != participant.session.fingerprint()
                || active.phase != SequenceSessionPhase::Open
                || active.state_transfer.is_reserved()
                || active.frame_has_node_flights(frame)
                || !active.frames.get(frame).is_some_and(|record| {
                    record.submission_wave_flight == Some(ParticipantFlightPhase::InFlight)
                })
            {
                return Err(invalid_resource(
                    "predecessor lost its exact open submitted participant frame",
                ));
            }
            let backing = participant.session.resources().lock_backing_state()?;
            if !Arc::ptr_eq(&backing.current, &participant.backing_snapshot) {
                return Err(invalid_resource("predecessor backing generation changed"));
            }
        }
        let participants = step
            .participants
            .iter()
            .map(|participant| PredecessorParticipant {
                slot: Arc::downgrade(&participant.session.slot),
                epoch: participant.session.epoch(),
                fingerprint: participant.session.fingerprint().clone(),
                frame: ActiveSequenceFrame {
                    frame_id: participant.frame.frame_id,
                    batch_step_id: step.batch_step_id(),
                },
                backing: Arc::clone(&participant.backing_snapshot),
                resources: Arc::clone(participant.session.resources()),
            })
            .collect();
        Ok(Self {
            wave_backing: wave.shared_backing_claim(),
            step_backing: Arc::clone(&step.claimed_backing),
            participants,
            lane: Arc::clone(lane),
            lane_epoch: lane.reusable_execution_epoch(),
            receipt,
        })
    }

    pub fn submission(&self) -> &SubmittedOperationReceipt {
        &self.receipt
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.step_backing.work_shape()
    }

    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Binds the next input to an actual submitted output without supplying a
    /// placeholder token. The product runner separately confirms that this
    /// output implements its chosen sampling policy before granting lookahead.
    pub fn bind_next_token_work(
        &self,
        sources: CompletionReadbackBatchRequest,
    ) -> Result<BatchWorkShape, VNextError> {
        sources.validate_for(self.receipt.batch_identity())?;
        if sources.len() != self.participants.len() {
            return Err(invalid_resource(
                "device token work requires exactly one source for every predecessor participant",
            ));
        }
        let first = &sources.requests()[0];
        let mut work = Vec::with_capacity(self.participants.len());
        for (index, (parent, source)) in self
            .work_shape()
            .participant_work()
            .iter()
            .zip(sources.requests())
            .enumerate()
        {
            if source.participant_index() as usize != index
                || source.output_layout().element_type() != ElementType::U32
                || source.output_layout().element_count() != 1
                || source.node_id() != first.node_id()
                || source.resource_id() != first.resource_id()
                || source.logical_offset_bytes() != first.logical_offset_bytes()
                || source.expected_usage() != first.expected_usage()
            {
                return Err(invalid_resource(
                    "device token sources must bind the same scalar U32 output in canonical cohort order",
                ));
            }
            self.validate_token_source(source)?;
            work.push(BatchParticipantTokenSpan::new(
                parent.participant(),
                TokenSpanWork::from_submitted_token(
                    parent.token_span(),
                    &self.receipt,
                    source.clone(),
                )?,
            ));
        }
        BatchWorkShape::new(work)
    }

    pub(super) fn validate_successor_work(&self, work: &BatchWorkShape) -> Result<(), VNextError> {
        if work.participant_work().len() != self.participants.len() {
            return Err(invalid_resource(
                "successor work changed predecessor membership",
            ));
        }
        for (index, (child, parent)) in work
            .participant_work()
            .iter()
            .zip(self.work_shape().participant_work())
            .enumerate()
        {
            let span = child.token_span();
            let parent_span = parent.token_span();
            let source = span.submitted_token_source().ok_or_else(|| {
                invalid_resource("successor token work is not bound to a submitted device output")
            })?;
            if child.participant() != parent.participant()
                || source.submission_fingerprint() != self.receipt.fingerprint()
                || source.parent_work_fingerprint() != parent_span.fingerprint()
                || source.source().participant_index() as usize != index
                || span.immediate_tokens() != 1
                || span.immediate_token_range().start != parent_span.full_input_tokens()
                || Some(span.immediate_token_range().end)
                    != parent_span.full_input_tokens().checked_add(1)
                || span.full_input_tokens() != span.immediate_token_range().end
                || span.fit_input_tokens() != parent_span.fit_input_tokens()
                || span.full_input_tokens() > span.fit_input_tokens()
                || span.checkpoint_tokens().is_some()
            {
                return Err(invalid_resource(
                    "successor token work differs from its sealed predecessor source and position",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn participant_frame(&self, index: usize) -> ActiveSequenceFrame {
        self.participants[index].frame
    }

    pub(super) fn matches_backing(
        &self,
        index: usize,
        backing: &Arc<SequenceBackingSnapshot<R>>,
    ) -> bool {
        self.participants
            .get(index)
            .is_some_and(|participant| Arc::ptr_eq(&participant.backing, backing))
    }

    /// Run before taking the cohort's session locks. Admission subsequently
    /// checks every live head frame and backing while holding those locks.
    pub(super) fn validate_successor_candidates(
        &self,
        candidates: &[SequenceFrameCaptureCandidate<R>],
        lane_id: ExecutionLaneId,
    ) -> Result<(), VNextError> {
        if lane_id != self.lane.id()
            || self.lane.reusable_execution_epoch() != self.lane_epoch
            || !self.lane.is_reusable()
            || !self.lane.current_descriptor_matches_snapshot()
            || candidates.len() != self.participants.len()
            || !candidates
                .iter()
                .zip(&self.participants)
                .all(|(candidate, expected)| {
                    candidate.frame.epoch == expected.epoch
                        && candidate.frame.fingerprint == expected.fingerprint
                        && Arc::ptr_eq(&candidate.resources, &expected.resources)
                        && Weak::ptr_eq(&Arc::downgrade(&candidate.frame.slot), &expected.slot)
                })
        {
            return Err(invalid_resource(
                "successor differs from the exact predecessor cohort, lane epoch or runtime",
            ));
        }
        Ok(())
    }
}
