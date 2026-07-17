use super::*;

fn frame(value: u64) -> ExecutionFrameId {
    ExecutionFrameId::try_from(value).expect("test frame id is non-zero")
}

fn step(value: u64) -> BatchStepId {
    BatchStepId::try_from(value).expect("test batch step id is non-zero")
}

fn invocation(value: u64) -> BatchInvocationId {
    BatchInvocationId::try_from(value).expect("test batch invocation id is non-zero")
}

fn node(value: &str) -> NodeId {
    NodeId::try_from(format!("node.{value}")).expect("test node id is valid")
}

fn participant_node_key(participant: u32, frame_id: u64, node_name: &str) -> ParticipantNodeKey {
    ParticipantNodeKey::new(
        BatchParticipantAuthority::new(
            SequenceAuthorityId::test_only(participant, 1),
            RequestAuthorityId::test_only(participant, 1),
        ),
        frame(frame_id),
        node(node_name),
    )
}

fn active_candidate(next_frame: u64, fingerprint: &str) -> SequenceFrameCandidate {
    let epoch = SequenceSessionEpoch(NonZeroU64::new(1).expect("one is non-zero"));
    let fingerprint = SequenceSessionFingerprint(fingerprint.to_owned());
    SequenceFrameCandidate {
        slot: Arc::new(SequenceSessionSlot {
            state: Mutex::new(SequenceSessionSlotState::Active(
                ActiveSequenceSessionState {
                    epoch,
                    fingerprint: fingerprint.clone(),
                    phase: SequenceSessionPhase::Open,
                    next_frame: Some(frame(next_frame)),
                    active_frame: None,
                    participant_flights: BTreeMap::new(),
                    retired_frames: next_frame - 1,
                },
            )),
        }),
        epoch,
        fingerprint,
    }
}

fn retire(holds: &mut [SessionFrameHold]) -> Vec<StepParticipantRetirementDisposition> {
    let mut references = holds.iter_mut().collect::<Vec<_>>();
    finalize_session_frames(&mut references, StepFrameFinalization::Commit)
        .expect("test frame retires")
}

fn flight_candidate(
    candidate: &SequenceFrameCandidate,
    hold: &SessionFrameHold,
) -> ParticipantFlightCandidate {
    flight_candidate_for(candidate, hold, 1)
}

fn flight_candidate_for(
    candidate: &SequenceFrameCandidate,
    hold: &SessionFrameHold,
    participant: u32,
) -> ParticipantFlightCandidate {
    ParticipantFlightCandidate {
        slot: Arc::clone(&candidate.slot),
        epoch: candidate.epoch,
        fingerprint: candidate.fingerprint.clone(),
        frame: ActiveSequenceFrame {
            frame_id: hold.frame_id,
            batch_step_id: hold.batch_step_id,
        },
        participant: BatchParticipantAuthority::new(
            SequenceAuthorityId::test_only(participant, 1),
            RequestAuthorityId::test_only(participant, 1),
        ),
    }
}

#[test]
fn core_frames_start_at_one_and_advance_contiguously() {
    let candidate = active_candidate(1, "session-a");
    let mut first = acquire_session_frames(std::slice::from_ref(&candidate), step(1)).unwrap();
    assert_eq!(first[0].frame_id, frame(1));
    assert_eq!(
        retire(&mut first),
        vec![StepParticipantRetirementDisposition::Committed]
    );

    let mut second = acquire_session_frames(std::slice::from_ref(&candidate), step(2)).unwrap();
    assert_eq!(second[0].frame_id, frame(2));
    retire(&mut second);
}

#[test]
fn one_batch_preserves_participant_local_frames() {
    let candidates = vec![
        active_candidate(7, "session-7"),
        active_candidate(2, "session-2"),
        active_candidate(19, "session-19"),
    ];
    let mut holds = acquire_session_frames(&candidates, step(9)).unwrap();
    assert_eq!(
        holds.iter().map(|hold| hold.frame_id).collect::<Vec<_>>(),
        vec![frame(7), frame(2), frame(19)]
    );
    retire(&mut holds);
}

#[test]
fn cancelled_participant_rejects_the_entire_frame_acquire_atomically() {
    let first = active_candidate(1, "session-first");
    let cancelled = active_candidate(1, "session-cancelled");
    {
        let mut state = cancelled.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        active.phase = SequenceSessionPhase::CancelRequested;
    }

    assert!(acquire_session_frames(&[first.clone(), cancelled], step(1)).is_err());
    let state = first.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        unreachable!();
    };
    assert_eq!(active.next_frame, Some(frame(1)));
    assert_eq!(active.active_frame, None);
}

#[test]
fn cancel_after_acquire_discards_only_that_participant() {
    let first = active_candidate(1, "session-first");
    let second = active_candidate(1, "session-second");
    let mut holds = acquire_session_frames(&[first, second.clone()], step(1)).unwrap();
    {
        let mut state = second.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        active.phase = SequenceSessionPhase::CancelRequested;
    }
    assert_eq!(
        retire(&mut holds),
        vec![
            StepParticipantRetirementDisposition::Committed,
            StepParticipantRetirementDisposition::DiscardedCancelled,
        ]
    );
    let state = second.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        unreachable!();
    };
    assert_eq!(active.retired_frames, 1);
}

#[test]
fn frame_hold_drop_is_fail_closed_and_never_reuses_the_frame() {
    let candidate = active_candidate(1, "session-drop");
    let holds = acquire_session_frames(std::slice::from_ref(&candidate), step(1)).unwrap();
    drop(holds);

    let state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        unreachable!();
    };
    assert_eq!(active.phase, SequenceSessionPhase::Poisoned);
    assert_eq!(active.active_frame.unwrap().frame_id, frame(1));
    assert_eq!(active.next_frame, Some(frame(2)));
    drop(state);
    assert!(acquire_session_frames(std::slice::from_ref(&candidate), step(2)).is_err());
}

#[test]
fn explicit_frame_abort_clears_the_hold_but_keeps_the_session_fail_closed() {
    let candidate = active_candidate(1, "session-abort");
    let mut holds = acquire_session_frames(std::slice::from_ref(&candidate), step(1)).unwrap();
    let mut references = holds.iter_mut().collect::<Vec<_>>();
    assert_eq!(
        finalize_session_frames(&mut references, StepFrameFinalization::Abort).unwrap(),
        vec![StepParticipantRetirementDisposition::Aborted]
    );
    drop(references);
    drop(holds);

    let state = candidate.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        unreachable!();
    };
    assert_eq!(active.phase, SequenceSessionPhase::Poisoned);
    assert_eq!(active.active_frame, None);
    assert_eq!(active.next_frame, Some(frame(2)));
    drop(state);
    assert!(acquire_session_frames(std::slice::from_ref(&candidate), step(2)).is_err());
}

#[test]
fn unsubmitted_frame_rollback_keeps_the_session_open_for_retry() {
    let candidate = active_candidate(1, "session-capacity-retry");
    let mut first = acquire_session_frames(std::slice::from_ref(&candidate), step(1)).unwrap();
    let mut references = first.iter_mut().collect::<Vec<_>>();
    assert_eq!(
        finalize_session_frames(&mut references, StepFrameFinalization::RollbackUnsubmitted,)
            .unwrap(),
        vec![StepParticipantRetirementDisposition::RolledBackUnsubmitted]
    );
    drop(references);
    drop(first);

    {
        let state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(active.phase, SequenceSessionPhase::Open);
        assert_eq!(active.active_frame, None);
        assert_eq!(active.retired_frames, 0);
        assert_eq!(active.next_frame, Some(frame(2)));
    }

    let mut retry = acquire_session_frames(std::slice::from_ref(&candidate), step(2)).unwrap();
    assert_eq!(retry[0].frame_id, frame(2));
    assert_eq!(
        retire(&mut retry),
        vec![StepParticipantRetirementDisposition::Committed]
    );
}

#[test]
fn cancel_after_step_rejects_new_invocation_with_zero_partial_flights() {
    let first = active_candidate(1, "flight-first");
    let cancelled = active_candidate(1, "flight-cancelled");
    let mut frames = acquire_session_frames(&[first.clone(), cancelled.clone()], step(1)).unwrap();
    let candidates = vec![
        flight_candidate_for(&first, &frames[0], 1),
        flight_candidate_for(&cancelled, &frames[1], 2),
    ];
    {
        let mut state = cancelled.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        active.phase = SequenceSessionPhase::CancelRequested;
    }

    assert!(prepare_participant_flights(&candidates, &node("main")).is_err());
    for candidate in [&first, &cancelled] {
        let state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
    }
    assert_eq!(
        retire(&mut frames),
        vec![
            StepParticipantRetirementDisposition::Committed,
            StepParticipantRetirementDisposition::DiscardedCancelled,
        ]
    );
}

#[test]
fn cancel_before_dispatch_rejects_all_participants_without_partial_transition() {
    let first = active_candidate(1, "dispatch-first");
    let cancelled = active_candidate(1, "dispatch-cancelled");
    let mut frames = acquire_session_frames(&[first.clone(), cancelled.clone()], step(1)).unwrap();
    let candidates = vec![
        flight_candidate_for(&first, &frames[0], 1),
        flight_candidate_for(&cancelled, &frames[1], 2),
    ];
    let mut prepared = prepare_participant_flights(&candidates, &node("main")).unwrap();
    {
        let mut state = cancelled.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        active.phase = SequenceSessionPhase::CancelRequested;
    }

    assert!(begin_participant_flights_dispatch(&mut prepared).is_err());
    for candidate in [&first, &cancelled] {
        let state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(
            active
                .participant_flights
                .values()
                .copied()
                .collect::<Vec<_>>(),
            vec![ParticipantFlightPhase::Prepared]
        );
    }
    drop(prepared);
    for candidate in [&first, &cancelled] {
        let state = candidate.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
    }
    assert_eq!(
        retire(&mut frames),
        vec![
            StepParticipantRetirementDisposition::Committed,
            StepParticipantRetirementDisposition::DiscardedCancelled,
        ]
    );
}

#[test]
fn dispatch_transition_wins_before_cancel_and_in_flight_drop_is_exact() {
    let session = active_candidate(1, "dispatch-wins");
    let mut frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let mut prepared =
        prepare_participant_flights(std::slice::from_ref(&candidate), &node("main")).unwrap();
    begin_participant_flights_dispatch(&mut prepared).unwrap();
    {
        let mut state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        assert_eq!(
            active.participant_flights.values().next(),
            Some(&ParticipantFlightPhase::InFlight)
        );
        active.phase = SequenceSessionPhase::CancelRequested;
    }
    drop(prepared);
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
        assert_eq!(active.phase, SequenceSessionPhase::CancelRequested);
    }
    assert_eq!(
        retire(&mut frames),
        vec![StepParticipantRetirementDisposition::DiscardedCancelled]
    );
}

#[test]
fn participant_flight_drop_poison_closes_a_phase_mismatch() {
    let session = active_candidate(1, "dispatch-phase-mismatch");
    let frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let mut prepared =
        prepare_participant_flights(std::slice::from_ref(&candidate), &node("main")).unwrap();
    begin_participant_flights_dispatch(&mut prepared).unwrap();
    {
        let mut state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        *active.participant_flights.values_mut().next().unwrap() = ParticipantFlightPhase::Prepared;
    }
    drop(prepared);
    let state = session.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        unreachable!();
    };
    assert_eq!(active.phase, SequenceSessionPhase::Poisoned);
}

#[test]
fn overlapping_prepared_flights_are_exact_and_duplicates_are_rejected() {
    let session = active_candidate(1, "flight-overlap");
    let mut frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let first =
        prepare_participant_flights(std::slice::from_ref(&candidate), &node("first")).unwrap();
    let second =
        prepare_participant_flights(std::slice::from_ref(&candidate), &node("second")).unwrap();
    assert!(prepare_participant_flights(std::slice::from_ref(&candidate), &node("first")).is_err());
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(active.participant_flights.len(), 2);
    }
    drop(first);
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(active.participant_flights.len(), 1);
    }
    drop(second);
    retire(&mut frames);
}

#[test]
fn prepared_flight_drop_removes_the_unsubmitted_sequence_flight() {
    let session = active_candidate(1, "flight-drop");
    let mut frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let prepared =
        prepare_participant_flights(std::slice::from_ref(&candidate), &node("main")).unwrap();
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(active.participant_flights.len(), 1);
    }
    drop(prepared);
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
        assert_eq!(active.phase, SequenceSessionPhase::Open);
    }
    retire(&mut frames);
}

#[test]
fn stale_or_missing_prepared_flight_drop_is_fail_closed() {
    let stale = active_candidate(1, "flight-stale");
    let stale_frames = acquire_session_frames(std::slice::from_ref(&stale), step(1)).unwrap();
    let stale_candidate = flight_candidate(&stale, &stale_frames[0]);
    let stale_prepared =
        prepare_participant_flights(std::slice::from_ref(&stale_candidate), &node("stale"))
            .unwrap();
    {
        let mut state = stale.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        active.epoch = SequenceSessionEpoch(NonZeroU64::new(2).unwrap());
    }
    drop(stale_prepared);
    assert!(matches!(
        &*stale.slot.state.lock().unwrap(),
        SequenceSessionSlotState::FailClosed
    ));

    let missing = active_candidate(1, "flight-missing");
    let missing_frames = acquire_session_frames(std::slice::from_ref(&missing), step(2)).unwrap();
    let missing_candidate = flight_candidate(&missing, &missing_frames[0]);
    let missing_prepared =
        prepare_participant_flights(std::slice::from_ref(&missing_candidate), &node("missing"))
            .unwrap();
    {
        let mut state = missing.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!();
        };
        active.participant_flights.clear();
    }
    drop(missing_prepared);
    let state = missing.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        unreachable!();
    };
    assert_eq!(active.phase, SequenceSessionPhase::Poisoned);
}

#[test]
fn physical_invocation_ledger_rejects_overlap_atomically_and_retains_tombstones() {
    let registry = Arc::new(InvocationRegistry::default());
    let first_key = participant_node_key(1, 7, "main");
    let overlap_key = participant_node_key(2, 2, "main");
    let untouched_key = participant_node_key(3, 19, "main");
    let guard = registry
        .enter(
            vec![first_key.clone(), overlap_key.clone()],
            invocation(1),
            &"a".repeat(64),
        )
        .unwrap();

    assert!(registry
        .enter(
            vec![overlap_key.clone(), untouched_key.clone()],
            invocation(2),
            &"b".repeat(64),
        )
        .is_err());
    {
        let state = registry.state.lock().unwrap();
        assert_eq!(state.entries.len(), 2);
        assert!(!state.entries.contains_key(&untouched_key));
        assert!(state
            .entries
            .values()
            .all(|entry| entry.phase == PhysicalInvocationPhase::Prepared));
    }

    drop(guard);
    {
        let state = registry.state.lock().unwrap();
        assert!(state
            .entries
            .values()
            .all(|entry| entry.phase == PhysicalInvocationPhase::Retired));
    }
    assert!(registry
        .enter(vec![first_key], invocation(3), &"a".repeat(64))
        .is_err());
}

#[test]
fn physical_invocation_retry_requires_not_submitted_and_a_fresh_attempt() {
    let registry = Arc::new(InvocationRegistry::default());
    let key = participant_node_key(1, 7, "retry");
    let mut guard = registry
        .enter(vec![key.clone()], invocation(11), &"c".repeat(64))
        .unwrap();

    guard.mark_not_submitted().unwrap();
    assert!(guard.mark_in_flight().is_err());
    assert!(guard.prepare_retry(invocation(11)).is_err());
    guard.prepare_retry(invocation(12)).unwrap();
    guard.mark_in_flight().unwrap();
    {
        let state = registry.state.lock().unwrap();
        let entry = state.entries.get(&key).unwrap();
        assert_eq!(entry.batch_invocation_id, invocation(12));
        assert_eq!(entry.work_fingerprint, "c".repeat(64));
        assert_eq!(entry.phase, PhysicalInvocationPhase::InFlight);
    }
    drop(guard);
    assert_eq!(
        registry.state.lock().unwrap().entries[&key].phase,
        PhysicalInvocationPhase::Retired
    );

    let dropped_key = participant_node_key(2, 2, "retry");
    let mut dropped = registry
        .enter(vec![dropped_key.clone()], invocation(13), &"d".repeat(64))
        .unwrap();
    dropped.mark_not_submitted().unwrap();
    drop(dropped);
    assert_eq!(
        registry.state.lock().unwrap().entries[&dropped_key].phase,
        PhysicalInvocationPhase::Retired
    );
    assert!(registry
        .enter(vec![dropped_key], invocation(14), &"d".repeat(64))
        .is_err());
}

#[test]
fn definitely_not_submitted_participant_flights_reset_for_exact_retry() {
    let session = active_candidate(1, "dispatch-retry");
    let mut frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let mut prepared =
        prepare_participant_flights(std::slice::from_ref(&candidate), &node("retry")).unwrap();
    begin_participant_flights_dispatch(&mut prepared).unwrap();
    reset_participant_flights_after_definitely_not_submitted(&mut prepared).unwrap();
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(
            active.participant_flights.values().next(),
            Some(&ParticipantFlightPhase::Prepared)
        );
    }
    begin_participant_flights_dispatch(&mut prepared).unwrap();
    drop(prepared);
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
    }
    retire(&mut frames);
}

#[test]
fn multi_node_wave_transitions_one_session_once_and_retries_atomically() {
    let session = active_candidate(1, "multi-node-wave");
    let mut frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let mut prepared = prepare_participant_flight_wave(vec![
        ParticipantFlightWaveCandidate::new(candidate.clone(), node("second")),
        ParticipantFlightWaveCandidate::new(candidate, node("first")),
    ])
    .unwrap();
    assert_eq!(prepared.len(), 2);
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert_eq!(active.participant_flights.len(), 2);
        assert!(active
            .participant_flights
            .values()
            .all(|phase| *phase == ParticipantFlightPhase::Prepared));
    }

    begin_participant_flights_dispatch(&mut prepared).unwrap();
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active
            .participant_flights
            .values()
            .all(|phase| *phase == ParticipantFlightPhase::InFlight));
    }
    reset_participant_flights_after_definitely_not_submitted(&mut prepared).unwrap();
    drop(prepared);
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
    }
    retire(&mut frames);
}

#[test]
fn duplicate_node_in_wave_rejects_without_partial_participant_flight() {
    let session = active_candidate(1, "duplicate-wave-node");
    let mut frames = acquire_session_frames(std::slice::from_ref(&session), step(1)).unwrap();
    let candidate = flight_candidate(&session, &frames[0]);
    let duplicate_node = node("duplicate");
    assert!(prepare_participant_flight_wave(vec![
        ParticipantFlightWaveCandidate::new(candidate.clone(), duplicate_node.clone()),
        ParticipantFlightWaveCandidate::new(candidate, duplicate_node),
    ])
    .is_err());
    {
        let state = session.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            unreachable!();
        };
        assert!(active.participant_flights.is_empty());
    }
    retire(&mut frames);
}

#[test]
fn sequence_session_fingerprint_is_domain_separated() {
    #[derive(Serialize)]
    struct Payload {
        epoch: u64,
    }
    let payload = Payload { epoch: 1 };
    let envelope = SequenceSessionFingerprintEnvelope {
        domain: SEQUENCE_SESSION_FINGERPRINT_DOMAIN,
        payload: &payload,
    };
    let value = serde_json::to_value(&envelope).unwrap();
    assert_eq!(value["domain"], "sequence-session-v1");
    let bytes = serde_json::to_vec(&envelope).unwrap();
    assert_eq!(
        sequence_session_fingerprint(&payload).unwrap().as_str(),
        format!("{:x}", Sha256::digest(bytes))
    );
}

#[test]
fn terminal_receipt_reports_retired_not_completed_frames() {
    let receipt = SequenceSessionTerminalReceipt {
        epoch: SequenceSessionEpoch(NonZeroU64::new(1).unwrap()),
        fingerprint: SequenceSessionFingerprint("receipt".to_owned()),
        disposition: SequenceSessionTerminalDisposition::Aborted,
        retired_frames: 7,
    };
    assert_eq!(receipt.retired_frames(), 7);
}
