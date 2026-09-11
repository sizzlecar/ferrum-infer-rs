use super::*;
use crate::vnext::{
    BatchOperationIdentity, CheckpointBoundaryConstraint, CheckpointCapacityPolicy,
    CheckpointInputDependency, CheckpointPartitionNumerics, CheckpointTokenSpanConstraint,
    CompletedSequenceProvenance, DeviceTimingMode, LaneSubmitOutcome, PendingRestoreCommit,
};
use std::ops::Range;

#[path = "checkpoint_access_tests.rs"]
mod checkpoint_access_tests;

#[test]
fn imported_step_rejects_changed_wave_and_individual_work_before_submission() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "import-work-swap", &[19, 23]);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap()
        .acknowledge()
        .unwrap();
    let span = TokenSpanWork::from_token_ids(&[19, 23], 1..2)
        .unwrap()
        .with_checkpoint_tokens(Arc::from([19, 23]))
        .unwrap();
    let step = begin_continuation_step(&target, &lane, span);
    let before = harness
        .root
        .dynamic_pools
        .logical_admission
        .snapshot()
        .unwrap();
    for (tokens, range) in [([29, 23], 1..2), ([19, 29], 1..2), ([19, 23], 0..1)] {
        let changed = step
            .bind_all_invocation_work_shape(vec![
                TokenSpanWork::from_token_ids(&tokens, range).unwrap()
            ])
            .unwrap();
        assert!(step
            .try_prepare_full_plan_submission_wave(
                Arc::new(changed.clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .is_err());
        let request = InvocationResourceAdmissionRequest::for_all_step_participants(
            harness.root.dynamic_pools.nodes[0].id().clone(),
            changed,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap();
        assert!(step.try_admit_invocation(request).is_err());
        step.invocation_registry
            .ensure_pristine_for_step_rollback()
            .unwrap();
        assert_eq!(
            harness
                .root
                .dynamic_pools
                .logical_admission
                .snapshot()
                .unwrap(),
            before
        );
    }
    // Equivalent work may omit process-local evidence; every provider receives
    // the already authenticated Step token Arc, preserving bytes and identity.
    let plain = step
        .bind_all_invocation_work_shape(vec![
            TokenSpanWork::from_token_ids(&[19, 23], 1..2).unwrap()
        ])
        .unwrap();
    let StepSubmissionWaveAdmissionDecision::Prepared(wave) = step
        .try_prepare_full_plan_submission_wave(
            Arc::new(plain),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    else {
        panic!("unchanged work must retain its admitted evidence")
    };
    let original = step.work_shape().participant_work()[0]
        .token_span()
        .checkpoint_tokens()
        .unwrap();
    for node in wave.nodes() {
        assert!(Arc::ptr_eq(
            original,
            node.work_shape().participant_work()[0]
                .token_span()
                .checkpoint_tokens()
                .unwrap()
        ));
    }
    let handle = submit_fixture_wave_through_reaper(
        &harness.root,
        &[Arc::clone(&target)],
        &lane,
        wave,
        &reaper,
    );
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("valid continuation must reach its real CPU fence")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    assert_eq!(
        reserve_capture(&target)
            .completed_boundary()
            .unwrap()
            .token_prefix(),
        &[19, 23]
    );
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(handle);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn imported_empty_step_cannot_erase_continuation_and_can_rollback_then_execute() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "import-empty-retirement", &[19, 23]);
    let imported = restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap()
        .acknowledge()
        .unwrap();
    let span = TokenSpanWork::from_token_ids(&[19, 23], 1..2)
        .unwrap()
        .with_checkpoint_tokens(Arc::from([19, 23]))
        .unwrap();
    let step = begin_continuation_step(&target, &lane, span);
    let id = step.batch_step_id();
    let step = step.try_retire_normal().unwrap_err().into_step();
    {
        let state = target.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            panic!("target must remain active")
        };
        assert_eq!(active.phase, SequenceSessionPhase::Open);
        assert_eq!(active.active_frame.unwrap().batch_step_id, id);
        assert!(
            matches!(&active.completed_boundary, SequenceCompletedFrontier::Proven(current)
            if Arc::ptr_eq(current, &imported))
        );
    }
    step.try_rollback_unsubmitted().unwrap();
    assert_continuation_rejected_before_frame(&target, &lane, &[29, 23], 1..2, true);
    assert_continuation_rejected_before_frame(&target, &lane, &[19, 23], 1..2, false);
    let (step, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &[19, 23], 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn imported_individual_completion_without_full_plan_cannot_clear_the_contract() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "import-individual", &[19, 23]);
    let imported = restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap()
        .acknowledge()
        .unwrap();
    let plain = TokenSpanWork::from_token_ids(&[19, 23], 1..2).unwrap();
    let step = begin_continuation_step(
        &target,
        &lane,
        plain
            .clone()
            .with_checkpoint_tokens(Arc::from([19, 23]))
            .unwrap(),
    );
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        harness.root.dynamic_pools.nodes[0].id().clone(),
        step.bind_all_invocation_work_shape(vec![plain]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let InvocationResourceAdmissionDecision::Admitted(mut invocation) =
        step.try_admit_invocation(request).unwrap()
    else {
        panic!("unchanged individual work must be admitted")
    };
    assert!(Arc::ptr_eq(
        step.work_shape().participant_work()[0]
            .token_span()
            .checkpoint_tokens()
            .unwrap(),
        invocation.work_shape().participant_work()[0]
            .token_span()
            .checkpoint_tokens()
            .unwrap(),
    ));
    let active = [TrustedActiveSequenceBinding::from_session(&target).unwrap()];
    let identity = BatchOperationIdentity::test_only_for_invocation(
        &invocation,
        &harness.root.dynamic_pools.nodes[0],
        &active,
    )
    .unwrap();
    invocation.begin_dispatch().unwrap();
    let mut reservation =
        CompletionReaper::reserve(&reaper, invocation, Arc::clone(&lane), identity).unwrap();
    let mut commands = DeviceCommandBatch::with_capacity(1);
    assert!(reservation
        .encode_backing_initializations(&harness.runtime, &mut commands)
        .is_ok());
    reservation.mark_submission_started();
    let LaneSubmitOutcome::Submitted(fence) = lane.reserve_enqueue().unwrap().submit(commands)
    else {
        panic!("CPU invocation must return its actual fence")
    };
    let handle = reservation.arm(fence, DeviceTimingMode::Off).unwrap();
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("CPU invocation must reach a terminal")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::Succeeded
    ));
    let step = step.try_retire_normal().unwrap_err().into_step();
    {
        let state = target.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            panic!("target must remain active")
        };
        assert_eq!(active.phase, SequenceSessionPhase::Open);
        assert!(active.active_frame.is_some());
        assert!(
            matches!(&active.completed_boundary, SequenceCompletedFrontier::Proven(current)
            if Arc::ptr_eq(current, &imported))
        );
    }
    // Submitted individual work cannot be represented as an unsubmitted retry.
    let step = step.try_rollback_unsubmitted().unwrap_err().into_step();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&target)]).unwrap();
    step.try_abort().unwrap();
    {
        let state = target.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            panic!("aborted target must remain owned")
        };
        assert_eq!(active.phase, SequenceSessionPhase::Poisoned);
        assert!(matches!(
            active.completed_boundary,
            SequenceCompletedFrontier::Unproven
        ));
    }
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![TokenSpanWork::from_token_ids(&[29, 23], 1..2).unwrap()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(batch.try_begin_step(request, &lane).is_err());
    drop(batch);
    target.try_abort().unwrap();
    drop(target);
    drop(checkpoint);
    drop(handle);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn imported_batch_last_invalid_member_cannot_partially_acquire_frames() {
    let harness = prefix_harness_with_backing(checkpoint_fixture::Spec::default(), &[19, 31], true);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let first = admitted_full_target(&harness, "batch-import-first", &[19, 31]);
    let last = admitted_full_target(&harness, "batch-import-last", &[19, 31]);
    for target in [&first, &last] {
        restore_ready(&harness, target, &checkpoint, &lane, &reaper)
            .install_frontier(target, Arc::from([19, 31]))
            .unwrap()
            .acknowledge()
            .unwrap();
    }
    // One ordinary completed source and two imported targets share the same
    // actual three-slot plan; the failing member is last in canonical order.
    let batch = ExecutionBatchParticipants::new(vec![
        Arc::clone(&harness.session),
        Arc::clone(&first),
        Arc::clone(&last),
    ])
    .unwrap();
    assert!(Arc::ptr_eq(batch.sessions().last().unwrap(), &last));
    let snapshot = || {
        batch
            .sessions()
            .iter()
            .map(|session| {
                let state = session.slot.state.lock().unwrap();
                let SequenceSessionSlotState::Active(active) = &*state else {
                    panic!("batch member must remain active")
                };
                assert_eq!(active.phase, SequenceSessionPhase::Open);
                assert!(!active.has_participant_flights());
                let SequenceCompletedFrontier::Proven(boundary) = &active.completed_boundary else {
                    panic!("batch member must retain its frontier")
                };
                (
                    active.next_frame,
                    active.active_frame,
                    active.retired_frames,
                    Arc::as_ptr(boundary),
                )
            })
            .collect::<Vec<_>>()
    };
    let before = snapshot();
    let coordinator = &harness.root.dynamic_pools.logical_admission;
    let used_before = coordinator
        .snapshot()
        .unwrap()
        .domains()
        .iter()
        .map(|domain| domain.used())
        .collect::<Vec<_>>();
    let retained_before = coordinator.checkpoint_retained_bytes().unwrap();
    let plain = TokenSpanWork::from_token_ids(&[19, 31], 1..2).unwrap();
    let valid = plain
        .clone()
        .with_checkpoint_tokens(Arc::from([19, 31]))
        .unwrap();
    let invalid = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![valid.clone(), valid.clone(), plain])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(batch.try_begin_step(invalid, &lane).is_err());
    assert_eq!(snapshot(), before);
    assert_eq!(
        coordinator
            .snapshot()
            .unwrap()
            .domains()
            .iter()
            .map(|domain| domain.used())
            .collect::<Vec<_>>(),
        used_before
    );
    assert_eq!(
        coordinator.checkpoint_retained_bytes().unwrap(),
        retained_before
    );
    let valid = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![valid; 3]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let StepResourceAdmissionDecision::Admitted(step) = batch.try_begin_step(valid, &lane).unwrap()
    else {
        panic!("the unchanged batch must still admit valid work")
    };
    step.try_rollback_unsubmitted().unwrap();
    assert_eq!(snapshot(), before);
    drop(batch);
    first.try_abort_if_quiescent().unwrap();
    last.try_abort_if_quiescent().unwrap();
    drop(first);
    drop(last);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

fn admitted_target(
    harness: &RestoreHarness,
    suffix: &str,
    request_tokens: &[u32],
    sequence_tokens: &[u32],
) -> Arc<SequenceSession<TestRuntime>> {
    admitted_target_with_backing(harness, suffix, request_tokens, sequence_tokens, false)
}

fn admitted_full_target(
    harness: &RestoreHarness,
    suffix: &str,
    tokens: &[u32],
) -> Arc<SequenceSession<TestRuntime>> {
    admitted_target_with_backing(harness, suffix, tokens, tokens, true)
}

fn admitted_target_with_backing(
    harness: &RestoreHarness,
    suffix: &str,
    request_tokens: &[u32],
    sequence_tokens: &[u32],
    full_backing: bool,
) -> Arc<SequenceSession<TestRuntime>> {
    let work = |tokens: &[u32]| {
        let end = if full_backing { tokens.len() } else { 1 };
        ResourceWorkShape::single(
            TokenSpanWork::from_token_ids_with_fit(tokens, 0..end, 16).unwrap(),
        )
        .unwrap()
    };
    let binding = harness.root.trusted_runtime_binding().unwrap();
    let RequestResourceAdmissionDecision::Admitted(request) = binding
        .try_admit_request(
            RequestResourceAdmissionRequest::new(
                work(request_tokens),
                AdmissionFitPolicy::FullInputMustFit,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            RunId::new(format!("run/{suffix}")).unwrap(),
            RequestIdentity::new(format!("request/{suffix}")).unwrap(),
        )
        .unwrap()
    else {
        panic!("target request must admit")
    };
    let SequenceResourceAdmissionDecision::Admitted(sequence) = request
        .try_admit_sequence(
            SequenceResourceAdmissionRequest::new(
                work(sequence_tokens),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
        )
        .unwrap()
    else {
        panic!("target child sequence must admit")
    };
    sequence.open_session().unwrap()
}

fn prefix_harness(spec: checkpoint_fixture::Spec) -> RestoreHarness {
    prefix_harness_with_tokens(spec, &[19, 31])
}

fn prefix_harness_with_tokens(spec: checkpoint_fixture::Spec, tokens: &[u32]) -> RestoreHarness {
    prefix_harness_with_backing(spec, tokens, false)
}

fn prefix_harness_with_backing(
    mut spec: checkpoint_fixture::Spec,
    tokens: &[u32],
    full_backing: bool,
) -> RestoreHarness {
    spec.checkpoint_capacity
        .get_or_insert_with(|| CheckpointCapacityPolicy::new(1024).unwrap());
    let mut harness = RestoreHarness::new(spec);
    let source =
        admitted_target_with_backing(&harness, "prefix-source", tokens, tokens, full_backing);
    harness.session.try_abort_if_quiescent().unwrap();
    harness.session = source;
    harness
}

fn prove_prefix_source(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) {
    prove_prefix_source_with_tokens(harness, lane, reaper, &[19, 31]);
}

fn prove_prefix_source_with_tokens(
    harness: &RestoreHarness,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    tokens: &[u32],
) {
    let (step, disposition) =
        execute_continuation(harness, &harness.session, lane, reaper, tokens, 0..1, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
}

fn restore_ready(
    harness: &RestoreHarness,
    target: &Arc<SequenceSession<TestRuntime>>,
    checkpoint: &Arc<CapturedCheckpoint<TestRuntime>>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> PendingRestoreCommit<TestRuntime> {
    let handle = submitted(
        reaper
            .submit_restore(
                reserve_restore(target),
                Arc::clone(checkpoint),
                harness.fixture.layout(),
                Arc::clone(lane),
            )
            .unwrap(),
    );
    assert_eq!(handle.poll().unwrap(), StateTransferObservation::Ready);
    let Some(StateTransferResult::RestoreReady(pending)) = handle.take().unwrap() else {
        panic!("real successful restore must produce its gated result")
    };
    pending
}

fn execute_continuation(
    harness: &RestoreHarness,
    target: &Arc<SequenceSession<TestRuntime>>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    tokens: &[u32],
    range: Range<usize>,
    retain_tokens: bool,
) -> (
    Arc<StepResourceLease<TestRuntime>>,
    OperationCompletionDisposition,
) {
    let full = TokenSpanWork::from_token_ids(tokens, 0..tokens.len()).unwrap();
    assert!(matches!(
        target
            .try_ensure_backing_covers(
                SequenceResourceExtensionRequest::new(
                    ResourceWorkShape::single(full).unwrap(),
                    AdmissionPressureAction::WaitForRelease,
                )
                .unwrap()
            )
            .unwrap(),
        SequenceResourceExtensionDecision::Extended(_)
            | SequenceResourceExtensionDecision::Current(_)
    ));
    let span = TokenSpanWork::from_token_ids(tokens, range).unwrap();
    let span = if retain_tokens {
        span.with_checkpoint_tokens(Arc::from(tokens)).unwrap()
    } else {
        span
    };
    let step = begin_continuation_step(target, lane, span);
    let StepSubmissionWaveAdmissionDecision::Prepared(wave) = step
        .try_prepare_full_plan_submission_wave(
            Arc::new(step.work_shape().clone()),
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )
        .unwrap()
    else {
        panic!("continuation wave must prepare")
    };
    let handle = submit_fixture_wave_through_reaper(
        &harness.root,
        &[Arc::clone(target)],
        lane,
        wave,
        reaper,
    );
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("continuation must report its actual terminal result")
    };
    (step, receipt.disposition().clone())
}

fn begin_continuation_step(
    target: &Arc<SequenceSession<TestRuntime>>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    span: TokenSpanWork,
) -> Arc<StepResourceLease<TestRuntime>> {
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(target)]).unwrap();
    let StepResourceAdmissionDecision::Admitted(step) = batch
        .try_begin_step(
            StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![span]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            lane,
        )
        .unwrap()
    else {
        panic!("continuation step must admit")
    };
    step
}

fn assert_cancelled_and_release(target: Arc<SequenceSession<TestRuntime>>) {
    assert!(target
        .try_prepare_state_transfer(
            SequenceStateTransferKind::CaptureRead,
            target.resources().backing_generation().unwrap(),
        )
        .is_err());
    assert!(!target.request_cancel().unwrap().state_transfer_pending());
    target.try_abort_if_quiescent().unwrap();
}

fn assert_continuation_rejected_before_frame(
    target: &Arc<SequenceSession<TestRuntime>>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    tokens: &[u32],
    range: Range<usize>,
    retain_tokens: bool,
) {
    // These targets already own full input capacity. No extension may obscure
    // whether rejection alone changed the installed native state frontier.
    let boundary = reserve_capture(target).completed_boundary().unwrap();
    let snapshot = || {
        let state = target.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &*state else {
            panic!("rejection must retain the exact active session")
        };
        assert_eq!(active.phase, SequenceSessionPhase::Open);
        assert!(active.active_frame.is_none());
        assert!(!active.has_participant_flights());
        (
            active.epoch,
            active.fingerprint.clone(),
            active.next_frame,
            active.retired_frames,
        )
    };
    let before = snapshot();
    let span = TokenSpanWork::from_token_ids(tokens, range).unwrap();
    let span = if retain_tokens {
        span.with_checkpoint_tokens(Arc::from(tokens)).unwrap()
    } else {
        span
    };
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(target)]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    assert!(batch.try_begin_step(request, lane).is_err());
    assert_eq!(snapshot(), before);
    let after = reserve_capture(target).completed_boundary().unwrap();
    assert!(Arc::ptr_eq(&boundary, &after));
}

#[test]
fn imported_frontier_uses_child_admission_and_requires_ack_before_real_full_plan_continuation() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    // A child's own input is authoritative even when its Request root was
    // admitted using different token evidence with the same capacity ceiling.
    let target = admitted_target(&harness, "import-child", &[19, 97], &[19, 23]);
    let publication = restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap();
    assert_eq!(publication.completed_tokens(), 1);
    assert_eq!(publication.token_prefix(), &[19]);
    let imported = Arc::clone(publication.boundary());
    assert!(matches!(
        imported.provenance(),
        CompletedSequenceProvenance::ImportedCheckpoint { .. }
    ));
    assert_eq!(imported.epoch(), target.epoch());
    assert_eq!(imported.session_fingerprint(), target.fingerprint());
    assert_eq!(imported.frame_id(), None);
    assert_eq!(imported.batch_step_id(), None);
    assert_eq!(imported.batch_invocation_id(), None);
    assert_transfer_gate(&target);
    let acknowledged = publication.acknowledge().unwrap();
    assert!(Arc::ptr_eq(&imported, &acknowledged));
    let guard = reserve_capture(&target);
    assert!(Arc::ptr_eq(&guard.completed_boundary().unwrap(), &imported));
    drop(guard);

    let (step, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &[19, 23], 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    let completed = reserve_capture(&target).completed_boundary().unwrap();
    assert!(matches!(
        completed.provenance(),
        CompletedSequenceProvenance::FullPlan { .. }
    ));
    assert!(completed.frame_id().is_some());
    assert_eq!(completed.token_prefix(), &[19, 23]);
    assert_eq!(imported.token_prefix(), &[19]);
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
    // Both kinds of frontier retain only host evidence, not a plan/session.
    assert_eq!(acknowledged.token_prefix(), &[19]);
    assert_eq!(completed.token_prefix(), &[19, 23]);
}

#[test]
fn restore_rejects_parent_input_or_same_prefix_with_a_different_admitted_suffix() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "import-wrong-suffix", &[19, 97], &[19, 23]);
    assert!(
        restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
            .install_frontier(&target, Arc::from([19, 97]))
            .is_err()
    );
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn restore_rejects_valid_admitted_input_that_does_not_match_the_captured_prefix() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "import-wrong-prefix", &[17, 23], &[17, 23]);
    assert!(
        restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
            .install_frontier(&target, Arc::from([17, 23]))
            .is_err()
    );
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn restore_rejects_another_target_with_identical_input_without_cancelling_that_target() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "import-first", &[19, 23], &[19, 23]);
    let other = admitted_target(&harness, "import-other", &[19, 23], &[19, 23]);
    assert!(
        restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
            .install_frontier(&other, Arc::from([19, 23]))
            .is_err()
    );
    assert_cancelled_and_release(target);
    reserve_restore(&other)
        .ensure_fresh_restore_target()
        .unwrap();
    other.try_abort_if_quiescent().unwrap();
    drop(other);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn dropping_installed_publication_cancels_before_the_gate_is_released() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "import-abandoned", &[19, 23], &[19, 23]);
    let publication = restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap();
    assert_transfer_gate(&target);
    drop(publication);
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn cancellation_rejects_both_native_install_and_later_publication_acknowledgement() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let first = admitted_target(
        &harness,
        "import-cancel-before-install",
        &[19, 23],
        &[19, 23],
    );
    let pending = restore_ready(&harness, &first, &checkpoint, &lane, &reaper);
    assert!(first.request_cancel().unwrap().state_transfer_pending());
    assert!(pending
        .install_frontier(&first, Arc::from([19, 23]))
        .is_err());
    assert_cancelled_and_release(first);
    let second = admitted_target(&harness, "import-cancel-before-ack", &[19, 23], &[19, 23]);
    let publication = restore_ready(&harness, &second, &checkpoint, &lane, &reaper)
        .install_frontier(&second, Arc::from([19, 23]))
        .unwrap();
    assert!(second.request_cancel().unwrap().state_transfer_pending());
    assert!(publication.acknowledge().is_err());
    assert_cancelled_and_release(second);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn publication_cannot_ack_a_replaced_core_frontier() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "import-corrupt-frontier", &[19, 23], &[19, 23]);
    let publication = restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap();
    {
        let mut state = target.slot.state.lock().unwrap();
        let SequenceSessionSlotState::Active(active) = &mut *state else {
            unreachable!()
        };
        active.completed_boundary = SequenceCompletedFrontier::Unproven;
    }
    assert!(publication.acknowledge().is_err());
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn imported_state_can_be_captured_again_without_inventing_a_model_frame() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "import-recapture", &[19, 23], &[19, 23]);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap()
        .acknowledge()
        .unwrap();
    let (owner, bytes) = allocate_checkpoint(&harness);
    let handle = submitted(
        reaper
            .submit_capture(
                reserve_capture(&target),
                owner.try_reserve_capture().unwrap(),
                bytes,
                Arc::clone(&lane),
            )
            .unwrap(),
    );
    assert_eq!(handle.poll().unwrap(), StateTransferObservation::Ready);
    let Some(StateTransferResult::Captured(recaptured)) = handle.take().unwrap() else {
        panic!("imported state with current evidence must be capturable")
    };
    assert_eq!(recaptured.boundary().frame_id(), None);
    assert_eq!(recaptured.boundary().token_prefix(), &[19]);
    assert_ne!(
        recaptured.backing().authority(),
        checkpoint.backing().authority()
    );
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(recaptured);
    drop(owner);
    drop(checkpoint);
    drop(handle);
    assert_checkpoint_budget_released(&harness);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn untracked_continuation_is_rejected_before_altering_an_imported_frontier() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "import-untracked", &[19, 23]);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap()
        .acknowledge()
        .unwrap();
    assert_continuation_rejected_before_frame(&target, &lane, &[19, 23], 1..2, false);
    let (step, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &[19, 23], 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    assert_eq!(
        reserve_capture(&target)
            .completed_boundary()
            .unwrap()
            .token_prefix(),
        &[19, 23]
    );
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn discontinuous_full_plan_cannot_replace_an_imported_prefix_with_a_new_proof() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "import-discontinuous", &[19, 23]);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 23]))
        .unwrap()
        .acknowledge()
        .unwrap();
    assert_continuation_rejected_before_frame(&target, &lane, &[29, 23], 1..2, true);
    let (step, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &[19, 23], 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn entire_input_dependency_rejects_a_different_suffix_but_accepts_the_same_full_input() {
    let spec = checkpoint_fixture::Spec {
        dependency: CheckpointInputDependency::EntireTokenInput,
        ..checkpoint_fixture::Spec::default()
    };
    let harness = prefix_harness(spec);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let different = admitted_target(&harness, "entire-input-different", &[19, 23], &[19, 23]);
    assert!(
        restore_ready(&harness, &different, &checkpoint, &lane, &reaper)
            .install_frontier(&different, Arc::from([19, 23]))
            .is_err()
    );
    assert_cancelled_and_release(different);
    let identical = admitted_target(&harness, "entire-input-identical", &[19, 31], &[19, 31]);
    restore_ready(&harness, &identical, &checkpoint, &lane, &reaper)
        .install_frontier(&identical, Arc::from([19, 31]))
        .unwrap()
        .acknowledge()
        .unwrap();
    identical.try_abort_if_quiescent().unwrap();
    drop(identical);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn same_partition_only_copy_cannot_publish_without_real_partition_evidence() {
    let spec = checkpoint_fixture::Spec {
        numerics: CheckpointPartitionNumerics::SamePartitionOnly,
        ..checkpoint_fixture::Spec::default()
    };
    let harness = prefix_harness(spec);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "partition-unproven", &[19, 31], &[19, 31]);
    assert!(
        restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
            .install_frontier(&target, Arc::from([19, 31]))
            .is_err()
    );
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn conditioning_declaration_without_executed_input_evidence_blocks_publication() {
    let spec = checkpoint_fixture::Spec {
        conditioning: true,
        ..checkpoint_fixture::Spec::default()
    };
    let harness = prefix_harness(spec);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "conditioning-unproven", &[19, 31], &[19, 31]);
    assert!(
        restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
            .install_frontier(&target, Arc::from([19, 31]))
            .is_err()
    );
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn restore_cannot_publish_when_the_provider_requires_a_nonempty_suffix() {
    let harness = prefix_harness(checkpoint_fixture::Spec::default());
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_target(&harness, "empty-suffix", &[19], &[19]);
    assert!(
        restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
            .install_frontier(&target, Arc::from([19]))
            .is_err()
    );
    assert_cancelled_and_release(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn entire_input_dependency_remains_binding_after_publication_acknowledgement() {
    let spec = checkpoint_fixture::Spec {
        dependency: CheckpointInputDependency::EntireTokenInput,
        ..checkpoint_fixture::Spec::default()
    };
    let harness = prefix_harness(spec);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source(&harness, &lane, &reaper);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "entire-input-late-change", &[19, 31]);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from([19, 31]))
        .unwrap()
        .acknowledge()
        .unwrap();
    assert_continuation_rejected_before_frame(&target, &lane, &[19, 23], 1..2, true);
    let (step, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &[19, 31], 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    step.try_retire_normal().unwrap();
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn entire_input_contract_survives_a_valid_full_plan_continuation() {
    let tokens = [19, 31, 41];
    let spec = checkpoint_fixture::Spec {
        dependency: CheckpointInputDependency::EntireTokenInput,
        ..checkpoint_fixture::Spec::default()
    };
    let harness = prefix_harness_with_tokens(spec, &tokens);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source_with_tokens(&harness, &lane, &reaper, &tokens);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "entire-input-second-change", &tokens);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from(tokens))
        .unwrap()
        .acknowledge()
        .unwrap();
    let (first, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &tokens, 1..2, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    first.try_retire_normal().unwrap();
    let boundary = reserve_capture(&target).completed_boundary().unwrap();
    assert!(matches!(
        boundary.provenance(),
        CompletedSequenceProvenance::FullPlan { .. }
    ));
    assert!(boundary.continuation_contract().is_some());
    assert_continuation_rejected_before_frame(&target, &lane, &[19, 31, 43], 2..3, true);
    let (second, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &tokens, 2..3, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    second.try_retire_normal().unwrap();
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}

#[test]
fn provider_suffix_span_constraints_survive_a_valid_full_plan_continuation() {
    let tokens = [19, 31, 41, 43, 47];
    let two = std::num::NonZeroU64::new(2).unwrap();
    let spec = checkpoint_fixture::Spec {
        boundaries: CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(two, two).unwrap(),
        )
        .unwrap(),
        ..checkpoint_fixture::Spec::default()
    };
    // Allocate all five positions initially: this fixture's contiguous ABI
    // does not support appending another physical extent after four tokens.
    let harness = prefix_harness_with_backing(spec, &tokens, true);
    let lane = harness.root.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    prove_prefix_source_with_tokens(&harness, &lane, &reaper, &tokens);
    let checkpoint = complete_capture(&harness, &lane, &reaper);
    let target = admitted_full_target(&harness, "suffix-span-second-change", &tokens);
    restore_ready(&harness, &target, &checkpoint, &lane, &reaper)
        .install_frontier(&target, Arc::from(tokens))
        .unwrap()
        .acknowledge()
        .unwrap();
    let (first, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &tokens, 1..3, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    first.try_retire_normal().unwrap();
    assert!(matches!(
        reserve_capture(&target)
            .completed_boundary()
            .unwrap()
            .provenance(),
        CompletedSequenceProvenance::FullPlan { .. }
    ));
    assert_continuation_rejected_before_frame(&target, &lane, &tokens, 3..4, true);
    let (second, disposition) =
        execute_continuation(&harness, &target, &lane, &reaper, &tokens, 3..5, true);
    assert!(matches!(
        disposition,
        OperationCompletionDisposition::Succeeded
    ));
    second.try_retire_normal().unwrap();
    target.try_abort_if_quiescent().unwrap();
    drop(target);
    drop(checkpoint);
    drop(reaper);
    drop(lane);
    harness.close();
}
