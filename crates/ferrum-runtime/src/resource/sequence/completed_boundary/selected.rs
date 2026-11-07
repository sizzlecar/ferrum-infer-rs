use super::*;
use crate::{BatchOperationIdentity, SelectedPrefix, StepResourceLease};

pub(crate) fn record_candidate_wave<R: DeviceRuntime>(
    wave: &PreparedStepSubmissionWave<R>,
    seal: &SuccessfulWaveCompletionSeal,
    identity: &BatchOperationIdentity,
) -> Result<(), VNextError> {
    if wave.step_resources().candidate_layout().is_none()
        || identity.fingerprint().is_empty()
        || identity.claimed_backing_fingerprint() != seal.wave_fingerprint()
    {
        return Err(invalid_resource("candidate success identity mismatch"));
    }
    let mut slot = wave
        .step_resources()
        .completed_boundary
        .lock()
        .map_err(|_| invalid_resource("candidate proof mutex poisoned"))?;
    if !matches!(*slot, StepCompletedBoundarySlot::Empty) {
        *slot = StepCompletedBoundarySlot::Invalid;
        return Err(invalid_resource("candidate wave completion repeated"));
    }
    *slot = StepCompletedBoundarySlot::Invalid;
    let proof = validate_completed_wave(wave, seal)?;
    if proof.participants.iter().any(Option::is_none) {
        return Err(invalid_resource(
            "candidate wave lacks exact token evidence",
        ));
    }
    *slot = StepCompletedBoundarySlot::CandidateReady {
        proof,
        identity: identity.clone(),
    };
    Ok(())
}

pub(in crate::resource) fn take_selected_proof<R: DeviceRuntime>(
    step: &StepResourceLease<R>,
    selections: &[SelectedPrefix],
) -> Result<(StepCompletedBoundaryProof, BatchOperationIdentity), VNextError> {
    if selections.len() != step.participants.len() {
        return Err(invalid_resource(
            "selection does not cover every participant",
        ));
    }
    let mut slot = step
        .completed_boundary
        .lock()
        .map_err(|_| invalid_resource("candidate proof mutex poisoned"))?;
    let StepCompletedBoundarySlot::CandidateReady { proof, .. } = &*slot else {
        return Err(invalid_resource(
            "candidate wave has no unconsumed successful proof",
        ));
    };
    for (p, selection) in proof.participants.iter().zip(selections) {
        let p = p
            .as_ref()
            .ok_or_else(|| invalid_resource("candidate has no token prefix"))?;
        if !matches!(selection, SelectedPrefix::All) && p.end_token - p.start_token != 2 {
            return Err(invalid_resource(
                "partial selection requires exactly two verified token rows",
            ));
        }
    }
    // A completed native wave must have relinquished every flight before a
    // second submission can access the state. The Step continues owning frames.
    for p in &step.participants {
        let state = p
            .session
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("candidate session poisoned"))?;
        let SequenceSessionSlotState::Active(active) = &*state else {
            return Err(invalid_resource("candidate session absent"));
        };
        validate_active_frame(active, &p.frame)?;
        if active.has_participant_flights()
            || active.state_transfer.is_reserved()
            || active.phase == SequenceSessionPhase::Poisoned
        {
            return Err(invalid_resource(
                "candidate state still in flight or unavailable",
            ));
        }
        let backing = p.session.resources().lock_backing_state()?;
        if !Arc::ptr_eq(&backing.current, &p.backing_snapshot) {
            return Err(invalid_resource("candidate backing generation changed"));
        }
    }
    let StepCompletedBoundarySlot::CandidateReady {
        mut proof,
        identity,
    } = std::mem::replace(&mut *slot, StepCompletedBoundarySlot::CopyPending)
    else {
        unreachable!()
    };
    for (p, selection) in proof.participants.iter_mut().zip(selections) {
        let p = Arc::get_mut(p.as_mut().expect("checked candidate prefix"))
            .expect("candidate boundary proof uniquely owns its boundary");
        if matches!(selection, SelectedPrefix::One) {
            p.end_token = p.start_token + 1;
        }
        p.selected_continuation = true;
    }
    Ok((proof, identity))
}
