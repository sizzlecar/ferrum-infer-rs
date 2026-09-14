use super::*;
use crate::vnext::{
    BatchOperationIdentity, CompletionHandle, CompletionObservation, CompletionReaper,
    CompletionRecoveryOutcome, DeviceTimingMode, LaneSubmitOutcome, OperationCompletionDisposition,
    SuccessfulWaveCompletionSeal,
};
use std::ops::Range;

struct BoundaryHarness {
    harness: Harness,
    lane: Arc<ExecutionLane<TestRuntime>>,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
}

impl BoundaryHarness {
    fn new(participants: usize) -> Self {
        Self::with_demand(participants, TestDemand::Fixed)
    }

    fn with_demand(participants: usize, demand: TestDemand) -> Self {
        let catalog = pool_catalog_with_options(
            paged_profile(),
            AllocationLifetime::Sequence,
            'b',
            1,
            512,
            demand,
            "state",
            false,
            StateInitialization::Zero,
        );
        let state_resource = catalog.descriptors[0].base_resource_id().clone();
        let runtime = new_runtime(&catalog, 512);
        let harness = harness_with_nodes(
            runtime,
            catalog,
            512,
            false,
            Arc::from(vec![
                PlanNode::resource_test_node_with_state_effect(
                    NodeId::new("node/sequence-state").unwrap(),
                    StateId::new("state/sequence-state").unwrap(),
                    ProgramValueId::new("value/sequence-state").unwrap(),
                    AllocationLifetime::Sequence,
                    TensorAccess::ReadWrite,
                    vec![state_resource],
                ),
                PlanNode::resource_test_node(NodeId::new("node/output").unwrap()),
            ]),
        );
        harness
            .root
            .maintenance_controller
            .grow_pool(&harness.pool_ids[0], 512)
            .unwrap();
        let lane = harness.root.create_execution_lane().unwrap();
        let sessions = (0..participants)
            .map(|index| {
                admitted_sequence_with_ceiling(&harness.root, &format!("boundary-{index}"), 4)
                    .open_session()
                    .unwrap()
            })
            .collect();
        Self {
            harness,
            lane,
            sessions,
        }
    }

    fn step(&self, spans: Vec<TokenSpanWork>) -> Arc<StepResourceLease<TestRuntime>> {
        let batch = ExecutionBatchParticipants::new(self.sessions.clone()).unwrap();
        let request = StepResourceAdmissionRequest::new(
            batch.bind_work_shape(spans).unwrap(),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        match batch.try_begin_step(request, &self.lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => step,
            _ => panic!("resident sequence state must admit its frame"),
        }
    }

    fn reserve(
        &self,
        index: usize,
        kind: SequenceStateTransferKind,
    ) -> PreparedSequenceStateTransfer<TestRuntime> {
        let session = &self.sessions[index];
        match session
            .try_prepare_state_transfer(kind, session.resources().backing_generation().unwrap())
            .unwrap()
        {
            SequenceStateTransferPreparation::Prepared(prepared) => prepared,
            _ => panic!("idle sequence must reserve its current state"),
        }
    }

    fn close(self) {
        let Self {
            harness,
            lane,
            sessions,
        } = self;
        for session in &sessions {
            if session.try_complete().is_err() {
                session
                    .try_abort_if_quiescent()
                    .or_else(|_| session.try_abort())
                    .unwrap();
            }
        }
        drop(sessions);
        drop(lane);
        close_dynamic_test_root(harness.root);
    }
}

fn span(tokens: &[u32], range: Range<usize>) -> TokenSpanWork {
    TokenSpanWork::from_token_ids(tokens, range)
        .unwrap()
        .with_checkpoint_tokens(Arc::from(tokens))
        .unwrap()
}

fn prepared_wave(
    step: &Arc<StepResourceLease<TestRuntime>>,
) -> PreparedStepSubmissionWave<TestRuntime> {
    match step
        .try_prepare_full_plan_submission_wave(
            Arc::new(step.work_shape().clone()),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    {
        StepSubmissionWaveAdmissionDecision::Prepared(wave) => wave,
        _ => panic!("resident full-model wave must prepare"),
    }
}

// Unit tests exercise core lifecycle and authority with a fake runtime. The
// cfg(test) seal substitutes only the terminal-fence observer; completion
// integration tests separately cover production seal issuance.
fn finish_wave(wave: &mut PreparedStepSubmissionWave<TestRuntime>) {
    wave.begin_dispatch().unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(1);
    assert!(wave
        .encode_backing_initializations(wave.runtime(), &mut commands)
        .is_ok());
    wave.mark_submission_fence_installed().unwrap();
    wave.finish_backing_initializations(true).unwrap();
    wave.finish_request_state_hazards(RequestStateHazardTerminalDisposition::Succeeded)
        .unwrap();
}

fn seal(wave: &PreparedStepSubmissionWave<TestRuntime>) -> SuccessfulWaveCompletionSeal {
    SuccessfulWaveCompletionSeal::test_only(
        wave.batch_step_id(),
        wave.batch_invocation_id(),
        wave.execution_lane_id(),
        wave.fingerprint().to_owned(),
    )
}

fn finish_and_record(step: &Arc<StepResourceLease<TestRuntime>>) {
    let mut wave = prepared_wave(step);
    finish_wave(&mut wave);
    wave.record_full_plan_success(&seal(&wave)).unwrap();
}

fn submit_through_reaper(
    harness: &BoundaryHarness,
    wave: PreparedStepSubmissionWave<TestRuntime>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> CompletionHandle<TestRuntime> {
    submit_fixture_wave_through_reaper(
        &harness.harness.root,
        &harness.sessions,
        &harness.lane,
        wave,
        reaper,
    )
}

pub(super) fn submit_fixture_wave_through_reaper(
    root: &Arc<PlanRuntimeResources<TestRuntime>>,
    sessions: &[Arc<SequenceSession<TestRuntime>>],
    lane: &Arc<ExecutionLane<TestRuntime>>,
    mut wave: PreparedStepSubmissionWave<TestRuntime>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> CompletionHandle<TestRuntime> {
    let active = sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity =
        BatchOperationIdentity::test_only_for_wave(&wave, &root.dynamic_pools.nodes, &active)
            .unwrap();
    wave.begin_dispatch().unwrap();
    let mut reservation =
        CompletionReaper::reserve_wave(reaper, wave, Arc::clone(lane), identity).unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(1);
    assert!(reservation
        .encode_backing_initializations(lane.runtime(), &mut commands)
        .is_ok());
    reservation.mark_submission_started();
    let fence = match lane.reserve_enqueue().unwrap().submit(commands) {
        LaneSubmitOutcome::Submitted(fence) => fence,
        _ => panic!("CPU fixture must return a real tracked fence"),
    };
    reservation.arm(fence, DeviceTimingMode::Off).unwrap()
}

fn frontier(session: &SequenceSession<TestRuntime>) -> SequenceCompletedFrontier {
    let state = session.slot.state.lock().unwrap();
    let SequenceSessionSlotState::Active(active) = &*state else {
        panic!("test session must remain active")
    };
    active.completed_boundary.clone()
}

#[test]
fn checkpoint_token_metadata_preserves_work_wire_identity_and_rejects_different_tokens() {
    let plain = TokenSpanWork::from_token_ids_with_fit(&[7, 11, 13], 1..3, 8).unwrap();
    let retained = plain
        .clone()
        .with_checkpoint_tokens(Arc::from([7, 11, 13]))
        .unwrap();
    assert!(plain.checkpoint_tokens().is_none());
    assert_eq!(plain, retained);
    assert_eq!(
        serde_json::to_vec(&plain).unwrap(),
        serde_json::to_vec(&retained).unwrap()
    );
    assert_eq!(plain.fingerprint(), retained.fingerprint());
    assert_eq!(
        ResourceWorkShape::single(plain.clone()).unwrap(),
        ResourceWorkShape::single(retained).unwrap()
    );
    assert!(plain
        .clone()
        .with_checkpoint_tokens(Arc::from([7, 11, 14]))
        .is_err());
    assert!(plain.with_checkpoint_tokens(Arc::from([7, 11])).is_err());
}

#[test]
fn completed_boundary_requires_success_then_commit_and_tracks_same_generation_decode() {
    let harness = BoundaryHarness::new(1);
    let restore = harness.reserve(0, SequenceStateTransferKind::RestoreWrite);
    restore.ensure_fresh_restore_target().unwrap();
    assert!(restore.completed_boundary().is_err());
    drop(restore);

    let first = harness.step(vec![span(&[3, 5], 0..2)]);
    let first_id = first.batch_step_id();
    let mut wave = prepared_wave(&first);
    let terminal = seal(&wave);
    finish_wave(&mut wave);
    wave.record_full_plan_success(&terminal).unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Fresh
    ));
    drop(wave);
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Fresh
    ));
    first.try_retire_normal().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    let old = capture.completed_boundary().unwrap();
    assert_eq!(old.token_prefix(), &[3, 5]);
    assert_eq!(old.batch_step_id(), Some(first_id));
    let generation = old.backing_generation();
    drop(capture);

    let second = harness.step(vec![span(&[3, 5, 8], 2..3)]);
    finish_and_record(&second);
    second.try_retire_normal().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    let new = capture.completed_boundary().unwrap();
    assert_eq!(new.backing_generation(), generation);
    assert_eq!(new.token_prefix(), &[3, 5, 8]);
    assert_ne!(new.frame_id(), old.frame_id());
    assert_eq!(old.token_prefix(), &[3, 5]);
    assert!(!Arc::ptr_eq(&old, &new));
    drop(capture);
    // Host proofs deliberately retain no execution or device resource owner.
    harness.close();
    assert_eq!(new.token_prefix(), &[3, 5, 8]);
}

#[test]
fn successful_untracked_step_invalidates_the_previous_same_generation_frontier() {
    let harness = BoundaryHarness::new(1);
    let first = harness.step(vec![span(&[3], 0..1)]);
    finish_and_record(&first);
    first.try_retire_normal().unwrap();
    let generation = harness.sessions[0]
        .resources()
        .backing_generation()
        .unwrap();
    let second = harness.step(vec![TokenSpanWork::from_token_ids(&[3, 5], 1..2).unwrap()]);
    finish_and_record(&second);
    second.try_retire_normal().unwrap();
    assert_eq!(
        harness.sessions[0]
            .resources()
            .backing_generation()
            .unwrap(),
        generation
    );
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    assert!(capture.completed_boundary().is_err());
    drop(capture);
    let third = harness.step(vec![span(&[3, 5, 8], 2..3)]);
    let mut wave = prepared_wave(&third);
    finish_wave(&mut wave);
    assert!(wave.record_full_plan_success(&seal(&wave)).is_err());
    drop(wave);
    third.try_retire_normal().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    harness.close();
}

#[test]
fn unsubmitted_rollback_preserves_frontier_but_retire_without_execution_invalidates_it() {
    let harness = BoundaryHarness::new(1);
    let first = harness.step(vec![span(&[2], 0..1)]);
    finish_and_record(&first);
    first.try_retire_normal().unwrap();
    let retry = harness.step(vec![span(&[2, 4], 1..2)]);
    retry.try_rollback_unsubmitted().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    assert_eq!(capture.completed_boundary().unwrap().token_prefix(), &[2]);
    drop(capture);
    let never_executed = harness.step(vec![span(&[2, 4], 1..2)]);
    never_executed.try_retire_normal().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    assert!(capture.completed_boundary().is_err());
    drop(capture);
    harness.close();
}

#[test]
fn cancellation_discards_only_cancelled_member_of_successful_batch() {
    let harness = BoundaryHarness::new(2);
    let step = harness.step(vec![span(&[2], 0..1), span(&[7, 9], 0..2)]);
    finish_and_record(&step);
    harness.sessions[0].request_cancel().unwrap();
    let receipt = step.try_retire_normal().unwrap();
    assert_eq!(
        receipt.participants()[0].disposition(),
        StepParticipantRetirementDisposition::DiscardedCancelled
    );
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    let capture = harness.reserve(1, SequenceStateTransferKind::CaptureRead);
    assert_eq!(
        capture.completed_boundary().unwrap().token_prefix(),
        &[7, 9]
    );
    drop(capture);
    harness.close();
}

#[test]
fn abort_does_not_publish_a_previously_signed_success_candidate() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[2], 0..1)]);
    finish_and_record(&step);
    step.try_abort().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    harness.close();
}

#[test]
fn discontinuous_ranges_and_changed_prefixes_cannot_advance_frontier() {
    for bad in [
        span(&[2, 4, 6], 2..3),
        span(&[9, 4], 1..2),
        span(&[2, 4], 0..2),
    ] {
        let harness = BoundaryHarness::new(1);
        let first = harness.step(vec![span(&[2], 0..1)]);
        finish_and_record(&first);
        first.try_retire_normal().unwrap();
        let second = harness.step(vec![bad]);
        let mut wave = prepared_wave(&second);
        finish_wave(&mut wave);
        assert!(wave.record_full_plan_success(&seal(&wave)).is_err());
        drop(wave);
        second.try_retire_normal().unwrap();
        assert!(matches!(
            frontier(&harness.sessions[0]),
            SequenceCompletedFrontier::Unproven
        ));
        harness.close();
    }
}

#[test]
fn foreign_success_identity_and_duplicate_signing_leave_no_candidate() {
    for mismatch in 0..4 {
        let harness = BoundaryHarness::new(1);
        let step = harness.step(vec![span(&[2], 0..1)]);
        let mut wave = prepared_wave(&step);
        finish_wave(&mut wave);
        let foreign = SuccessfulWaveCompletionSeal::test_only(
            if mismatch == 0 {
                BatchStepId::try_from(wave.batch_step_id().get() + 1).unwrap()
            } else {
                wave.batch_step_id()
            },
            if mismatch == 1 {
                BatchInvocationId::try_from(wave.batch_invocation_id().get() + 1).unwrap()
            } else {
                wave.batch_invocation_id()
            },
            if mismatch == 2 {
                ExecutionLaneId::mint().unwrap()
            } else {
                wave.execution_lane_id()
            },
            if mismatch == 3 {
                "0".repeat(64)
            } else {
                wave.fingerprint().to_owned()
            },
        );
        assert!(wave.record_full_plan_success(&foreign).is_err());
        assert!(wave.record_full_plan_success(&seal(&wave)).is_err());
        drop(wave);
        step.try_retire_normal().unwrap();
        assert!(matches!(
            frontier(&harness.sessions[0]),
            SequenceCompletedFrontier::Unproven
        ));
        harness.close();
    }
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[2], 0..1)]);
    let mut wave = prepared_wave(&step);
    finish_wave(&mut wave);
    wave.record_full_plan_success(&seal(&wave)).unwrap();
    assert!(wave.record_full_plan_success(&seal(&wave)).is_err());
    drop(wave);
    step.try_retire_normal().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    harness.close();
}

#[test]
fn determinism_probe_even_with_terminal_seal_is_not_a_complete_model_step() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[2], 0..1)]);
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        NodeId::new("node/sequence-state").unwrap(),
        step.work_shape().clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut wave = match step
        .try_prepare_determinism_submission_wave(vec![request])
        .unwrap()
    {
        StepSubmissionWaveAdmissionDecision::Prepared(wave) => wave,
        _ => panic!("probe must prepare"),
    };
    finish_wave(&mut wave);
    assert!(wave.record_full_plan_success(&seal(&wave)).is_err());
    drop(wave);
    step.try_retire_normal().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    harness.close();
}

#[test]
fn backing_extension_requires_a_new_completed_step_before_capture() {
    let harness = BoundaryHarness::with_demand(1, TestDemand::Tokens);
    let first = harness.step(vec![span(&[2], 0..1)]);
    finish_and_record(&first);
    first.try_retire_normal().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    let previous_generation = capture.completed_boundary().unwrap().backing_generation();
    drop(capture);
    let session = &harness.sessions[0];
    assert!(matches!(
        session
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    work(2),
                    AdmissionPressureAction::WaitForRelease
                )
                .unwrap(),
            )
            .unwrap(),
        SequenceResourceExtensionDecision::Extended(_)
    ));
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    assert_ne!(capture.backing().generation(), previous_generation);
    assert!(capture.completed_boundary().is_err());
    drop(capture);
    let second = harness.step(vec![span(&[2, 4], 1..2)]);
    finish_and_record(&second);
    second.try_retire_normal().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    let current = capture.completed_boundary().unwrap();
    assert_eq!(current.backing_generation(), capture.backing().generation());
    assert_eq!(current.token_prefix(), &[2, 4]);
    drop(capture);
    harness.close();
}

#[test]
fn stale_later_participant_cannot_record_an_earlier_participant_proof() {
    let harness = BoundaryHarness::new(2);
    let step = harness.step(vec![span(&[2], 0..1), span(&[7], 0..1)]);
    let mut wave = prepared_wave(&step);
    finish_wave(&mut wave);
    let original = {
        let mut state = harness.sessions[1].slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!()
        };
        let original = active.epoch;
        active.epoch = SequenceSessionEpoch(NonZeroU64::new(original.get() + 1).unwrap());
        original
    };
    assert!(wave.record_full_plan_success(&seal(&wave)).is_err());
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    {
        let mut state = harness.sessions[1].slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!()
        };
        active.epoch = original;
    }
    drop(wave);
    step.try_retire_normal().unwrap();
    for session in &harness.sessions {
        assert!(matches!(
            frontier(session),
            SequenceCompletedFrontier::Unproven
        ));
    }
    harness.close();
}

#[test]
fn commit_validates_all_frames_before_publishing_any_completed_frontier() {
    let harness = BoundaryHarness::new(2);
    let step = harness.step(vec![span(&[2], 0..1), span(&[7], 0..1)]);
    finish_and_record(&step);
    let original = {
        let mut state = harness.sessions[1].slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!()
        };
        active.next_frame.take()
    };
    let step = step.try_retire_normal().unwrap_err().into_step();
    for session in &harness.sessions {
        assert!(matches!(
            frontier(session),
            SequenceCompletedFrontier::Fresh
        ));
    }
    {
        let mut state = harness.sessions[1].slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!()
        };
        active.next_frame = original;
    }
    step.try_retire_normal().unwrap();
    for index in 0..2 {
        let capture = harness.reserve(index, SequenceStateTransferKind::CaptureRead);
        capture.completed_boundary().unwrap();
    }
    harness.close();
}

#[test]
fn real_reaper_keeps_pending_unproven_then_signs_full_plan_before_frame_commit() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[3, 5], 0..2)]);
    let reaper = CompletionReaper::new();
    harness
        .harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Pending);
    let handle = submit_through_reaper(&harness, prepared_wave(&step), &reaper);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    assert_eq!(reaper.retained_count(), 1);
    assert_eq!(harness.lane.in_flight_count(), 1);
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Fresh
    ));
    harness
        .harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("CPU fixture success must be terminal");
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(harness.lane.in_flight_count(), 0);
    assert!(step.completed_boundary.lock().unwrap().proof().is_some());
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Fresh
    ));
    assert!(handle.poll().is_err()); // Already reaped, so cannot sign twice.
    step.try_retire_normal().unwrap();
    let capture = harness.reserve(0, SequenceStateTransferKind::CaptureRead);
    assert_eq!(
        capture.completed_boundary().unwrap().token_prefix(),
        &[3, 5]
    );
    drop(capture);
    drop(handle);
    drop(reaper);
    harness.close();
}

#[test]
fn real_reaper_failed_fence_does_not_sign_complete_model_state() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[3], 0..1)]);
    let reaper = CompletionReaper::new();
    harness
        .harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::FailedButQuiescent);
    let handle = submit_through_reaper(&harness, prepared_wave(&step), &reaper);
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("quiescent CPU fixture failure must be terminal");
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    assert_eq!(reaper.retained_count(), 0);
    step.try_abort().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    drop(handle);
    drop(reaper);
    harness.close();
}

#[test]
fn real_reaper_indeterminate_fence_and_successful_drain_do_not_sign_model_state() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[3], 0..1)]);
    let reaper = CompletionReaper::new();
    harness
        .harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Indeterminate);
    let handle = submit_through_reaper(&harness, prepared_wave(&step), &reaper);
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Indeterminate(_)
    ));
    assert_eq!(reaper.retained_count(), 1);
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    assert!(matches!(
        reaper
            .recover_slot_by_draining_lane(handle.slot_id())
            .unwrap(),
        CompletionRecoveryOutcome::Drained(_)
    ));
    assert_eq!(reaper.retained_count(), 0);
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    step.try_abort().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    drop(handle);
    drop(reaper);
    harness.close();
}

#[test]
fn real_reaper_successful_probe_does_not_sign_full_plan_state() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[3], 0..1)]);
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        NodeId::new("node/sequence-state").unwrap(),
        step.work_shape().clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let StepSubmissionWaveAdmissionDecision::Prepared(wave) = step
        .try_prepare_determinism_submission_wave(vec![request])
        .unwrap()
    else {
        panic!("CPU fixture probe must prepare")
    };
    let reaper = CompletionReaper::new();
    let handle = submit_through_reaper(&harness, wave, &reaper);
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("CPU fixture probe must complete")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    step.try_retire_normal().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    drop(handle);
    drop(reaper);
    harness.close();
}

#[test]
fn real_reaper_success_after_cancellation_never_publishes_a_frontier() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[3], 0..1)]);
    let reaper = CompletionReaper::new();
    harness
        .harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Pending);
    let handle = submit_through_reaper(&harness, prepared_wave(&step), &reaper);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Pending
    ));
    harness.sessions[0].request_cancel().unwrap();
    harness
        .harness
        .runtime
        .set_fence_behavior(TestFenceBehavior::Succeeded);
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("cancelled CPU fixture submission still reaches its fence")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let retired = step.try_retire_normal().unwrap();
    assert_eq!(
        retired.participants()[0].disposition(),
        StepParticipantRetirementDisposition::DiscardedCancelled
    );
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    drop(handle);
    drop(reaper);
    harness.close();
}

#[test]
fn real_reaper_successful_single_invocation_cannot_prove_a_complete_model_step() {
    let harness = BoundaryHarness::new(1);
    let step = harness.step(vec![span(&[3], 0..1)]);
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        NodeId::new("node/sequence-state").unwrap(),
        step.work_shape().clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let InvocationResourceAdmissionDecision::Admitted(mut invocation) =
        step.try_admit_invocation(request).unwrap()
    else {
        panic!("CPU fixture invocation must admit")
    };
    let active = harness
        .sessions
        .iter()
        .map(|session| TrustedActiveSequenceBinding::from_session(session).unwrap())
        .collect::<Vec<_>>();
    let identity = BatchOperationIdentity::test_only_for_invocation(
        &invocation,
        &harness.harness.root.dynamic_pools.nodes[0],
        &active,
    )
    .unwrap();
    invocation.begin_dispatch().unwrap();
    let reaper = CompletionReaper::new();
    let mut reservation =
        CompletionReaper::reserve(&reaper, invocation, Arc::clone(&harness.lane), identity)
            .unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(1);
    assert!(reservation
        .encode_backing_initializations(&harness.harness.runtime, &mut commands,)
        .is_ok());
    reservation.mark_submission_started();
    let fence = match harness.lane.reserve_enqueue().unwrap().submit(commands) {
        LaneSubmitOutcome::Submitted(fence) => fence,
        _ => panic!("CPU fixture invocation must return its fence"),
    };
    let handle = reservation.arm(fence, DeviceTimingMode::Off).unwrap();
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("CPU fixture invocation must complete")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    assert!(step.completed_boundary.lock().unwrap().proof().is_none());
    step.try_retire_normal().unwrap();
    assert!(matches!(
        frontier(&harness.sessions[0]),
        SequenceCompletedFrontier::Unproven
    ));
    drop(handle);
    drop(reaper);
    harness.close();
}
