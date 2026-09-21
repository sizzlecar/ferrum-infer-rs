use super::*;
use crate::vnext::resource::{SequenceCompletedFrontier, SubmittedWavePredecessor};
use crate::vnext::ExecutionLaneId;

/// Admits only the immediate successor of one exact submitted full-plan cohort.
/// The completion-owned predecessor proves the installed fence and retains
/// its physical claims; an InFlight bookkeeping phase alone is insufficient.
pub(in crate::vnext::resource) fn acquire_successor_session_frames_with_backing<
    R: DeviceRuntime,
>(
    candidates: &[SequenceFrameCaptureCandidate<R>],
    batch_step_id: BatchStepId,
    work_shape: &BatchWorkShape,
    lane_id: ExecutionLaneId,
    predecessor: &SubmittedWavePredecessor<R>,
) -> Result<Vec<CapturedSessionFrame<R>>, VNextError> {
    predecessor.validate_successor_candidates(candidates, lane_id)?;
    predecessor.validate_successor_work(work_shape)?;
    if candidates.is_empty()
        || candidates.len() != work_shape.participant_work().len()
        || candidates.len() != predecessor.work_shape().participant_work().len()
        || candidates.iter().enumerate().any(|(index, candidate)| {
            candidates[..index]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.frame.slot, &candidate.frame.slot))
        })
    {
        return Err(invalid_resource(
            "successor frame requires the complete unique predecessor cohort",
        ));
    }
    for ((candidate, work), parent_work) in candidates
        .iter()
        .zip(work_shape.participant_work())
        .zip(predecessor.work_shape().participant_work())
    {
        let participant = BatchParticipantAuthority::new(
            candidate.resources.sequence_authority(),
            candidate.resources.request_authority(),
        );
        if work.participant() != participant
            || parent_work.participant() != participant
            || work.token_span().immediate_token_range().start
                != parent_work.token_span().immediate_token_range().end
        {
            return Err(invalid_resource(
                "successor work does not continue its exact predecessor participant span",
            ));
        }
        // Request-scoped state needs a separate chained hazard permit. Keep
        // its existing serial arbitration until that capability is installed.
        if !candidate
            .resources
            .request
            .plan
            .dynamic_pools()
            .request_state_hazards
            .is_empty()
        {
            return Err(invalid_resource(
                "successor frames do not support request-state hazards",
            ));
        }
    }
    let mut states = candidates
        .iter()
        .map(|candidate| {
            candidate
                .frame
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (index, (candidate, state)) in candidates.iter().zip(&states).enumerate() {
        let SequenceSessionSlotState::Active(active) = &**state else {
            return Err(invalid_resource(
                "successor participant session is no longer active",
            ));
        };
        let parent = predecessor.participant_frame(index);
        if active.epoch != candidate.frame.epoch
            || active.fingerprint != candidate.frame.fingerprint
            || active.phase != SequenceSessionPhase::Open
            || active.frames.head() != Some(parent)
            || active.frames.successor().is_some()
            || active
                .frames
                .get(parent)
                .and_then(|record| record.submission_wave_flight)
                != Some(ParticipantFlightPhase::InFlight)
            || !active.participant_flights.is_empty()
            || active.state_transfer.is_reserved()
            || active.next_frame.is_none()
            || active.next_frame != execution_frame_successor(parent.frame_id)
            || batch_step_id == parent.batch_step_id
        {
            return Err(invalid_resource(
                "successor frame lost its exact open submitted predecessor",
            ));
        }
        // Imported continuation constrains the exact proven prefix. A pending
        // parent is not that proof, so imports remain serial in this stage.
        if matches!(&active.completed_boundary,
            SequenceCompletedFrontier::Proven(boundary)
                if boundary.continuation_contract().is_some())
        {
            return Err(invalid_resource(
                "imported checkpoint continuation requires serial frame admission",
            ));
        }
    }
    // Preserve slot -> backing lock order and validate every participant before
    // changing any frame. A successor cannot grow or replace live state backing.
    let backing_states = candidates
        .iter()
        .map(|candidate| candidate.resources.lock_backing_state())
        .collect::<Result<Vec<_>, _>>()?;
    for (index, backing) in backing_states.iter().enumerate() {
        if !predecessor.matches_backing(index, &backing.current) {
            return Err(invalid_resource(
                "successor frame backing differs from its submitted predecessor",
            ));
        }
    }
    let mut captured = Vec::with_capacity(candidates.len());
    for ((candidate, state), backing) in candidates.iter().zip(&mut states).zip(&backing_states) {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all successor session frames were validated while locked")
        };
        let frame_id = active
            .next_frame
            .take()
            .expect("validated successor frame remains available");
        active.next_frame = execution_frame_successor(frame_id);
        active.frames.insert_successor(ActiveSequenceFrame {
            frame_id,
            batch_step_id,
        });
        captured.push(CapturedSessionFrame {
            hold: SessionFrameHold {
                slot: Arc::clone(&candidate.frame.slot),
                epoch: candidate.frame.epoch,
                fingerprint: candidate.frame.fingerprint.clone(),
                frame_id,
                batch_step_id,
                finalized: false,
            },
            backing_snapshot: Arc::clone(&backing.current),
        });
    }
    Ok(captured)
}
