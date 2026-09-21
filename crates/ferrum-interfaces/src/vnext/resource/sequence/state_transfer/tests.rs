use super::*;
use crate::vnext::resource::{
    acquire_session_frames, ActiveSequenceFrame, BatchStepId, ExecutionFrameId,
    SequenceFrameCandidate,
};
use std::collections::BTreeMap;
use std::sync::Mutex;

fn candidate() -> SequenceFrameCandidate {
    let fingerprint = SequenceSessionFingerprint("state-transfer-test".to_owned());
    let epoch = SequenceSessionEpoch(NonZeroU64::MIN);
    SequenceFrameCandidate {
        slot: Arc::new(SequenceSessionSlot {
            state: Mutex::new(SequenceSessionSlotState::Active(
                ActiveSequenceSessionState {
                    epoch,
                    fingerprint: fingerprint.clone(),
                    phase: SequenceSessionPhase::Open,
                    next_frame: Some(ExecutionFrameId::try_from(1_u64).unwrap()),
                    active_frame: None,
                    participant_flights: BTreeMap::new(),
                    submission_wave_flight: None,
                    state_transfer: SequenceStateTransferSlot::default(),
                    completed_boundary: super::super::SequenceCompletedFrontier::default(),
                    retired_frames: 0,
                },
            )),
        }),
        epoch,
        fingerprint,
    }
}

fn reserve(candidate: &SequenceFrameCandidate) -> PreparedStateTransferHold {
    let mut state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &mut *state else {
        panic!()
    };
    assert!(ensure_transfer_candidate(active).unwrap());
    let reservation = active
        .state_transfer
        .reserve(
            SequenceStateTransferKind::CaptureRead,
            SequenceBackingGeneration::INITIAL,
        )
        .unwrap();
    PreparedStateTransferHold {
        slot: Arc::clone(&candidate.slot),
        epoch: candidate.epoch,
        fingerprint: candidate.fingerprint.clone(),
        reservation,
        released: std::sync::atomic::AtomicBool::new(false),
    }
}

#[test]
fn checkpoint_transfer_reservation_excludes_frames_and_rolls_back_without_advancing() {
    let candidate = candidate();
    let hold = reserve(&candidate);
    assert!(acquire_session_frames(
        std::slice::from_ref(&candidate),
        BatchStepId::try_from(1_u64).unwrap()
    )
    .is_err());
    {
        let state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            panic!()
        };
        assert!(!ensure_transfer_candidate(active).unwrap());
        assert_eq!(active.next_frame.unwrap().get(), 1);
        assert_eq!(active.retired_frames, 0);
    }
    let first_serial = hold.reservation.serial;
    drop(hold);
    let second = reserve(&candidate);
    assert!(second.reservation.serial > first_serial);
    drop(second);
    assert!(acquire_session_frames(&[candidate], BatchStepId::try_from(1_u64).unwrap()).is_ok());
}

#[test]
fn checkpoint_transfer_rejects_mixed_batch_without_advancing_other_participants() {
    let first = candidate();
    let reserved = candidate();
    let hold = reserve(&reserved);
    assert!(acquire_session_frames(
        &[first.clone(), reserved],
        BatchStepId::try_from(1_u64).unwrap(),
    )
    .is_err());
    let state = first.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        panic!()
    };
    assert!(active.active_frame.is_none());
    assert_eq!(active.next_frame.unwrap().get(), 1);
    assert_eq!(active.retired_frames, 0);
    drop(state);
    drop(hold);
}

#[test]
fn checkpoint_transfer_reservation_respects_frames_flights_and_cancel() {
    let candidate = candidate();
    let mut state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &mut *state else {
        panic!()
    };
    active.active_frame = Some(ActiveSequenceFrame {
        frame_id: ExecutionFrameId::try_from(1_u64).unwrap(),
        batch_step_id: BatchStepId::try_from(1_u64).unwrap(),
    });
    assert!(!ensure_transfer_candidate(active).unwrap());
    active.active_frame = None;
    active.submission_wave_flight = Some(super::super::ParticipantFlightPhase::InFlight);
    assert!(!ensure_transfer_candidate(active).unwrap());
    active.submission_wave_flight = None;
    active.phase = SequenceSessionPhase::CancelRequested;
    assert!(ensure_transfer_candidate(active).is_err());
}

#[test]
fn checkpoint_transfer_prepared_drop_preserves_cancellation() {
    let candidate = candidate();
    let hold = reserve(&candidate);
    {
        let mut state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            panic!()
        };
        active.phase = SequenceSessionPhase::CancelRequested;
    }
    drop(hold);
    let state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        panic!()
    };
    assert_eq!(active.phase, SequenceSessionPhase::CancelRequested);
    assert!(!active.state_transfer.is_reserved());
}

#[test]
fn checkpoint_transfer_stale_drop_does_not_release_another_generation() {
    let candidate = candidate();
    let hold = reserve(&candidate);
    let new_reservation;
    {
        let mut state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            panic!()
        };
        active.epoch = SequenceSessionEpoch(NonZeroU64::new(2).unwrap());
        active.state_transfer = SequenceStateTransferSlot::default();
        new_reservation = active
            .state_transfer
            .reserve(
                SequenceStateTransferKind::RestoreWrite,
                SequenceBackingGeneration::INITIAL,
            )
            .unwrap();
    }
    drop(hold);
    let state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        panic!()
    };
    assert_eq!(active.state_transfer.active, Some(new_reservation));
    assert_eq!(active.phase, SequenceSessionPhase::Open);
}

#[test]
fn checkpoint_transfer_identity_exhaustion_never_reuses_authority() {
    let mut slot = SequenceStateTransferSlot {
        next_serial: NonZeroU64::new(u64::MAX),
        active: None,
    };
    let last = slot
        .reserve(
            SequenceStateTransferKind::CaptureRead,
            SequenceBackingGeneration::INITIAL,
        )
        .unwrap();
    assert_eq!(last.serial.get(), u64::MAX);
    assert!(slot
        .reserve(
            SequenceStateTransferKind::CaptureRead,
            SequenceBackingGeneration::INITIAL
        )
        .is_err());
    slot.active = None;
    assert!(slot
        .reserve(
            SequenceStateTransferKind::RestoreWrite,
            SequenceBackingGeneration::INITIAL
        )
        .is_err());
    assert!(!slot.is_reserved());
}

#[test]
fn checkpoint_transfer_same_session_mismatch_is_fail_closed_without_releasing_other_hold() {
    let candidate = candidate();
    let hold = reserve(&candidate);
    let replacement;
    {
        let mut state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            panic!()
        };
        active.state_transfer.active = None;
        replacement = active
            .state_transfer
            .reserve(
                SequenceStateTransferKind::RestoreWrite,
                SequenceBackingGeneration::INITIAL,
            )
            .unwrap();
    }
    drop(hold);
    let state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        panic!()
    };
    assert_eq!(active.phase, SequenceSessionPhase::Poisoned);
    assert_eq!(active.state_transfer.active, Some(replacement));
}
