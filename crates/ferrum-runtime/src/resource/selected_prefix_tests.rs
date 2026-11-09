//! Real token admission and native copies, with independently calculated state
//! values. These qualify Core settlement, not a model's speculative algorithm.
#[path = "selected_prefix_tests/fixture.rs"]
mod fixture;
use crate::*;
use fixture::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[test]
fn selected_prefix_on_demand_backing_keeps_current_candidates_and_exact_history() {
    let f = Fixture::with_replay_probe(
        ReusableExecutionProgramPreparation::OnDemand,
        Some(ReplayProbe::Checked),
    )
    .unwrap();
    let left = f.session("left");
    let right = f.session("right");
    let wave = f
        .try_reusable_step(
            &[left.clone(), right.clone()],
            &[&[3, 5], &[11, 13]],
            &[0, 0],
        )
        .unwrap();
    f.compute(&wave, &[&[3, 5], &[11, 13]]);
    let before = f.runtime.probes();
    assert_eq!(before[0][0].values(), [0., 0.]);
    assert_eq!(before[0][1].values(), [0., 0., 0.]);
    f.select(wave, vec![SelectedPrefix::One, SelectedPrefix::Two]);
    assert_eq!(before[0][0].values(), [0., 6.]);
    assert_eq!(before[0][1].values(), [3., 6., 9.]);
    assert_eq!(before[1][0].values(), [22., 26.]);
    assert_eq!(before[1][1].values(), [24., 48., 83.]);
    assert!(f
        .try_reusable_step(&[left.clone()], &[&[99, 7]], &[1])
        .is_err());
    assert!(f
        .try_reusable_step(&[left.clone()], &[&[3, 5, 7]], &[2])
        .is_err());
    let survivor = f
        .try_reusable_step(&[left.clone()], &[&[3, 7]], &[1])
        .unwrap();
    f.compute(&survivor, &[&[3, 7]]);
    f.select(survivor, vec![SelectedPrefix::All]);
    let actual = f.runtime.probes();
    assert_eq!(actual[0][0].values(), [6., 14.]);
    assert_eq!(actual[0][1].values(), [10., 20., 33.]);
    assert_eq!(before[1][0].values(), [22., 26.]);
    assert!(f.runtime.replay_address_checks.load(Ordering::SeqCst) > 0);
    assert!(f.runtime.eager_single_rows.load(Ordering::SeqCst) > 0);
    f.close(vec![left, right]);
}

#[test]
fn selected_prefix_replay_rejects_startup_state_reexecution() {
    assert!(
        Fixture::with_program_policy(ReusableExecutionProgramPreparation::BeforeReadiness).is_err()
    );
    assert!(Fixture::with_replay_probe(
        ReusableExecutionProgramPreparation::BeforeReadiness,
        Some(ReplayProbe::Checked),
    )
    .is_err());
}

#[test]
fn selected_prefix_replay_rejects_missing_or_invalid_current_address_proof() {
    for probe in [
        ReplayProbe::UncheckedStatic,
        ReplayProbe::UncheckedDynamic,
        ReplayProbe::CapturedCandidate,
        ReplayProbe::OmittedCandidate,
        ReplayProbe::MissingBindingDeclaration,
        ReplayProbe::ErrorAfterValid,
        ReplayProbe::UnavailableAfterValid,
    ] {
        let f =
            Fixture::with_replay_probe(ReusableExecutionProgramPreparation::OnDemand, Some(probe))
                .unwrap();
        let session = f.session("invalid-address");
        let wave = f
            .try_reusable_step(&[session.clone()], &[&[3, 5]], &[0])
            .unwrap();
        let submitted = f.runtime.submitted.load(Ordering::SeqCst);
        let error = match f.try_compute_with_visual(&wave, &[&[3, 5]], None) {
            Err(error) => error,
            Ok(()) => panic!("invalid candidate replay probe was accepted: {probe:?}"),
        };
        assert!(
            error.contains("candidate reusable topology lacks current program-bound state proof"),
            "{probe:?}: {error}"
        );
        assert_eq!(f.runtime.submitted.load(Ordering::SeqCst), submitted);
        assert_eq!(f.runtime.copied.load(Ordering::SeqCst), 0);
        assert!(f.runtime.probes().is_empty());
        if matches!(
            probe,
            ReplayProbe::ErrorAfterValid | ReplayProbe::UnavailableAfterValid
        ) {
            assert!(f.runtime.replay_address_checks.load(Ordering::SeqCst) > 0);
        }
        let Wave { step, batch } = wave;
        step.try_abort().unwrap();
        drop(batch);
        f.close(vec![session]);
    }
}

#[test]
fn selected_prefix_without_binding_resource_has_no_reusable_program() {
    let f = Fixture::with_replay_probe(
        ReusableExecutionProgramPreparation::OnDemand,
        Some(ReplayProbe::MissingBindingResource),
    )
    .unwrap();
    let session = f.session("no-binding-slot");
    let wave = f
        .try_reusable_step(&[session.clone()], &[&[3, 5]], &[0])
        .unwrap();
    // The fixture checks program_id=None on the same Prepared wave consumed
    // by compute; ordinary execution and selected retirement remain valid.
    f.compute(&wave, &[&[3, 5]]);
    f.select(wave, vec![SelectedPrefix::One]);
    assert_eq!(f.runtime.probes()[0][1].values(), [3., 6., 9.]);
    f.close(vec![session]);
}

#[test]
fn selected_prefix_replay_cannot_reuse_an_earlier_waves_address_proof() {
    let f = Fixture::with_replay_probe(
        ReusableExecutionProgramPreparation::OnDemand,
        Some(ReplayProbe::UncheckedAfterPriorWave),
    )
    .unwrap();
    let session = f.session("fresh-proof");
    let first = f
        .try_reusable_step(&[session.clone()], &[&[3, 5]], &[0])
        .unwrap();
    f.compute(&first, &[&[3, 5]]);
    f.select(first, vec![SelectedPrefix::One]);
    assert!(f.runtime.replay_address_checks.load(Ordering::SeqCst) > 0);
    let second = f
        .try_reusable_step(&[session.clone()], &[&[3, 7, 9]], &[1])
        .unwrap();
    let submitted = f.runtime.submitted.load(Ordering::SeqCst);
    let copied = f.runtime.copied.load(Ordering::SeqCst);
    let error = f
        .try_compute_with_visual(&second, &[&[3, 7, 9]], None)
        .unwrap_err();
    assert!(error.contains("candidate reusable topology lacks current program-bound state proof"));
    assert_eq!(f.runtime.submitted.load(Ordering::SeqCst), submitted);
    assert_eq!(f.runtime.copied.load(Ordering::SeqCst), copied);
    let Wave { step, batch } = second;
    step.try_abort().unwrap();
    drop(batch);
    f.close(vec![session]);
}

#[test]
fn selected_prefix_eager_execution_preserves_workspace_capacity_policy() {
    let fixture = Fixture::with_workspace_policy().unwrap();
    let session = fixture.session("workspace");
    let wave = fixture
        .try_step(&[session.clone()], &[&[3, 5]], &[0])
        .unwrap();
    fixture.compute(&wave, &[&[3, 5]]);
    fixture.select(wave, vec![SelectedPrefix::One]);
    let continued = fixture
        .try_step(&[session.clone()], &[&[3, 7]], &[1])
        .unwrap();
    fixture.compute(&continued, &[&[3, 7]]);
    fixture.select(continued, vec![SelectedPrefix::All]);
    let actual = fixture.runtime.probes();
    assert_eq!(actual[0][0].values(), [6., 14.]);
    assert_eq!(actual[0][1].values(), [10., 20., 33.]);
    fixture.close(vec![session]);
}

#[test]
fn selected_prefix_one_copies_both_states_and_rejects_discarded_or_changed_prefix() {
    let f = Fixture::new();
    let session = f.session("one");
    let mut wave = f.try_step(&[session.clone()], &[&[3, 5]], &[0]).unwrap();
    assert!(
        wave.step
            .try_prepare_selected_state_commit(vec![SelectedPrefix::One])
            .is_err(),
        "unexecuted wave has no authority"
    );
    f.compute(&wave, &[&[3, 5]]);
    let before = f.runtime.probes();
    // Compute produced candidates, but has not mutated either recurrent state.
    assert_eq!(before[0][0].values(), [0., 0.]);
    assert_eq!(before[0][1].values(), [0., 0., 0.]);
    assert_eq!(before[0][2].values(), [15., 37.]);
    let failure = wave.step.try_retire_normal().unwrap_err();
    wave.step = failure.into_step();
    let receipt = f.select(wave, vec![SelectedPrefix::One]);
    assert_eq!(
        receipt.participants()[0].disposition(),
        StepParticipantRetirementDisposition::Committed
    );
    assert_eq!(before[0][0].values(), [0., 6.]);
    assert_eq!(before[0][1].values(), [3., 6., 9.]);
    assert_eq!(f.runtime.copied.load(Ordering::SeqCst), 2);
    // Backing contains a rejected KV suffix; it confers no valid frontier.
    assert!(f.try_step(&[session.clone()], &[&[3, 5, 7]], &[2]).is_err());
    assert!(f.try_step(&[session.clone()], &[&[99, 7]], &[1]).is_err());
    let next = f.try_step(&[session.clone()], &[&[3, 7]], &[1]).unwrap();
    f.compute(&next, &[&[3, 7]]);
    f.select(next, vec![SelectedPrefix::All]);
    let after = f.runtime.probes();
    // Serial oracle for only [3,7], independent of the discarded token 5.
    assert_eq!(after[0][0].values(), [6., 14.]);
    assert_eq!(after[0][1].values(), [10., 20., 33.]);
    assert_eq!(after[0][2].values(), [15., 47.]);
    f.close(vec![session]);
}

#[test]
fn selected_prefix_two_advances_exact_full_prefix_and_both_participants_are_isolated() {
    let f = Fixture::new();
    let left = f.session("left");
    let right = f.session("right");
    let sessions = [left.clone(), right.clone()];
    let wave = f
        .try_step(&sessions, &[&[3, 5], &[11, 13]], &[0, 0])
        .unwrap();
    f.compute(&wave, &[&[3, 5], &[11, 13]]);
    assert!(
        wave.step
            .try_prepare_selected_state_commit(vec![SelectedPrefix::Two])
            .is_err(),
        "all participants required"
    );
    f.select(wave, vec![SelectedPrefix::One, SelectedPrefix::Two]);
    let values = f.runtime.probes();
    assert_eq!(values[0][0].values(), [0., 6.]);
    assert_eq!(values[0][1].values(), [3., 6., 9.]);
    assert_eq!(values[1][0].values(), [22., 26.]);
    assert_eq!(values[1][1].values(), [24., 48., 83.]);
    assert_eq!(values[1][2].values(), [55., 109.]);
    let next = f
        .try_step(&sessions, &[&[3, 7], &[11, 13, 2]], &[1, 2])
        .unwrap();
    f.compute(&next, &[&[3, 7], &[11, 13, 2]]);
    f.select(next, vec![SelectedPrefix::All, SelectedPrefix::All]);
    let after = f.runtime.probes();
    assert_eq!(after[0][0].values(), [6., 14.]);
    assert_eq!(after[0][1].values(), [10., 20., 33.]);
    assert_eq!(after[1][0].values(), [26., 26.]);
    assert_eq!(after[1][1].values(), [26., 74., 113.]);
    assert_eq!(after[1][2].values(), [55., 109., 139.]);
    drop(sessions);
    f.close(vec![left, right]);
}

#[test]
fn selected_prefix_cancelled_member_does_not_publish_or_discard_survivor_state() {
    let f = Fixture::new();
    let left = f.session("cancelled");
    let right = f.session("survivor");
    let wave = f
        .try_step(
            &[left.clone(), right.clone()],
            &[&[3, 5], &[11, 13]],
            &[0, 0],
        )
        .unwrap();
    f.compute(&wave, &[&[3, 5], &[11, 13]]);
    left.request_cancel().unwrap();
    let receipt = f.select(wave, vec![SelectedPrefix::One, SelectedPrefix::Two]);
    assert_eq!(
        receipt.participants()[0].disposition(),
        StepParticipantRetirementDisposition::DiscardedCancelled
    );
    assert_eq!(
        receipt.participants()[1].disposition(),
        StepParticipantRetirementDisposition::Committed
    );
    assert!(f.try_step(&[left.clone()], &[&[3, 7]], &[1]).is_err());
    let next = f.try_step(&[right.clone()], &[&[11, 13, 2]], &[2]).unwrap();
    f.compute(&next, &[&[11, 13, 2]]);
    f.select(next, vec![SelectedPrefix::All]);
    assert_eq!(f.runtime.probes()[0][1].values(), [26., 74., 113.]);
    f.close(vec![left, right]);
}

#[test]
fn selected_prefix_copy_failure_keeps_partial_writes_from_becoming_a_frontier() {
    let f = Fixture::new();
    let session = f.session("failed");
    let wave = f.try_step(&[session.clone()], &[&[3, 5]], &[0]).unwrap();
    f.compute(&wave, &[&[3, 5]]);
    let prepared = wave
        .step
        .try_prepare_selected_state_commit(vec![SelectedPrefix::One])
        .unwrap();
    assert!(
        wave.step
            .try_prepare_selected_state_commit(vec![SelectedPrefix::Two])
            .is_err(),
        "selection cannot be changed or copied twice"
    );
    *f.runtime.next_copy.lock().unwrap() = CopyOutcome::FailAfterFirst;
    let SelectedStateSubmission::Submitted(handle) =
        f.reaper.submit_selected_state_commit(prepared).unwrap()
    else {
        panic!("submitted partial failure")
    };
    let CompletionObservation::Terminal(receipt) = handle.wait().unwrap() else {
        panic!("known quiescence")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    assert_eq!(
        f.runtime.copied.load(Ordering::SeqCst),
        1,
        "failure genuinely follows a write"
    );
    drop((receipt, handle));
    let step = wave.step.try_retire_selected().unwrap_err().into_step();
    step.try_abort().unwrap();
    drop(wave.batch);
    assert!(f.try_step(&[session.clone()], &[&[3, 7]], &[1]).is_err());
    f.close(vec![session]);
}

#[test]
fn selected_prefix_unknown_copy_retains_owners_until_drain_and_never_commits() {
    let f = Fixture::new();
    let session = f.session("unknown");
    let wave = f.try_step(&[session.clone()], &[&[3, 5]], &[0]).unwrap();
    f.compute(&wave, &[&[3, 5]]);
    let prepared = wave
        .step
        .try_prepare_selected_state_commit(vec![SelectedPrefix::One])
        .unwrap();
    let occupancy = f.resources.dynamic_pool_status().unwrap();
    *f.runtime.next_copy.lock().unwrap() = CopyOutcome::Indeterminate;
    let SelectedStateSubmission::Submitted(handle) =
        f.reaper.submit_selected_state_commit(prepared).unwrap()
    else {
        panic!("fence returned")
    };
    assert!(matches!(
        handle.wait().unwrap(),
        CompletionObservation::Indeterminate(_)
    ));
    assert_eq!(f.reaper.retained_count(), 1);
    let bytes = f.runtime.live_bytes.load(Ordering::SeqCst);
    assert!(bytes > 0);
    assert!(session.try_abort_if_quiescent().is_err());
    assert!(wave
        .step
        .try_prepare_selected_state_commit(vec![SelectedPrefix::One])
        .is_err());
    assert_eq!(f.runtime.live_bytes.load(Ordering::SeqCst), bytes);
    assert_eq!(
        f.resources.dynamic_pool_status().unwrap(),
        occupancy,
        "unknown copy cannot return logical capacity"
    );
    // A lane drain proves only quiescence, never that all chosen states copied.
    assert!(matches!(
        f.reaper
            .recover_slot_by_draining_lane(handle.slot_id())
            .unwrap(),
        CompletionRecoveryOutcome::Drained(_)
    ));
    assert_eq!(f.reaper.retained_count(), 0);
    drop(handle);
    let step = wave.step.try_retire_selected().unwrap_err().into_step();
    step.try_abort().unwrap();
    drop(wave.batch);
    assert!(f.try_step(&[session.clone()], &[&[3, 7]], &[1]).is_err());
    f.close(vec![session]);
}

fn visual_source(host: &HostMemoryAccount, changed: bool) -> ManagedHostBytes {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 16 * 2 * 4)
        .unwrap();
    for (slot, value) in bytes.as_mut().chunks_exact_mut(4).zip([
        if changed { 123.0_f32 } else { 1.0 },
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    bytes.freeze()
}
fn visual_binding(
    host: &HostMemoryAccount,
    source: &ManagedHostBytes,
) -> Arc<SequenceConditioningInputs> {
    Arc::new(
        SequenceConditioningInputs::new(vec![SequenceConditioningInput::new(
            host,
            ProgramValueId::new(VISUAL).unwrap(),
            ResolvedTensorSpec::new(
                vec![16, 2],
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            SequenceConditioningCoverage::Exact,
            vec![SequenceConditioningSegment::new(0, source.clone()).unwrap()],
        )
        .unwrap()])
        .unwrap(),
    )
}

#[test]
fn selected_prefix_conditioning_is_bound_once_and_continuation_uses_original_source_row() {
    // Every invalid case uses a real compiled session and ordinary dispatch.
    // None may reach backend submit, including backing-zero commands.
    for case in ["unbound", "missing_upload", "changed_bytes", "wrong_row"] {
        let f = Fixture::visual();
        let session = f.session(case);
        let source = visual_source(&f.host, false);
        if case != "unbound" {
            session
                .bind_conditioning_inputs(visual_binding(&f.host, &source))
                .unwrap();
        }
        let changed = (case == "changed_bytes").then(|| visual_source(&f.host, true));
        let supplied = if case == "missing_upload" {
            None
        } else {
            Some(changed.as_ref().unwrap_or(&source))
        };
        let wave = f.try_step(&[session.clone()], &[&[3, 5]], &[0]).unwrap();
        let before = f.runtime.submitted.load(Ordering::SeqCst);
        assert!(
            f.try_compute_with_visual_row(
                &wave,
                &[&[3, 5]],
                supplied,
                (case == "wrong_row").then_some(1)
            )
            .is_err(),
            "{case}"
        );
        assert_eq!(
            f.runtime.submitted.load(Ordering::SeqCst),
            before,
            "{case} reached backend"
        );
        assert!(wave
            .step
            .try_prepare_selected_state_commit(vec![SelectedPrefix::One])
            .is_err());
        wave.step.try_abort().unwrap();
        drop((wave.batch, source, changed));
        f.close(vec![session]);
    }

    let f = Fixture::visual();
    let session = f.session("visual_continuation");
    let source = visual_source(&f.host, false);
    let original_charge = f.host.backing_bytes(source.len()).unwrap();
    let bound = visual_binding(&f.host, &source);
    session.bind_conditioning_inputs(bound.clone()).unwrap();
    assert!(session.bind_conditioning_inputs(bound.clone()).is_err());
    let first = f.try_step(&[session.clone()], &[&[3, 5]], &[0]).unwrap();
    f.try_compute_with_visual(&first, &[&[3, 5]], Some(&source))
        .unwrap();
    f.select(first, vec![SelectedPrefix::One]);
    assert_eq!(f.runtime.probes()[0][0].values(), [0., 16.]);
    assert_eq!(f.runtime.probes()[0][1].values(), [8., 16., 24.]);
    assert!(
        session.bind_conditioning_inputs(bound.clone()).is_err(),
        "cannot replace after accepting a prefix"
    );
    let next = f.try_step(&[session.clone()], &[&[3, 7]], &[1]).unwrap();
    f.try_compute_with_visual(&next, &[&[3, 7]], Some(&source))
        .unwrap();
    f.select(next, vec![SelectedPrefix::All]);
    // t0=3+1+2*2=8; accepted continuation t1=7+4+2*8=27.
    // Reusing row0 or reading discarded target token5 fails these constants.
    let actual = f.runtime.probes();
    assert_eq!(actual[0][0].values(), [16., 54.]);
    assert_eq!(actual[0][1].values(), [35., 70., 113.]);
    assert_eq!(actual[0][2].values(), [40., 167.]);
    drop((source, bound));
    assert_eq!(
        f.host.usage().unwrap().allocated_bytes(),
        original_charge,
        "session retains exactly the original Host allocation"
    );
    let host = f.host.clone();
    f.close(vec![session]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn selected_prefix_sparse_conditioning_continuation_requires_zero_outside_segments() {
    for nonzero_gap in [false, true] {
        let f = Fixture::visual();
        let session = f.session(if nonzero_gap {
            "nonzero_gap"
        } else {
            "zero_gap"
        });
        let mut source = f
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 2 * 2 * 4)
            .unwrap();
        for (slot, value) in source
            .as_mut()
            .chunks_exact_mut(4)
            .zip([1.0_f32, 2.0, 4.0, 8.0])
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        let source = source.freeze();
        let original_charge = f.host.backing_bytes(source.len()).unwrap();
        let bound = Arc::new(
            SequenceConditioningInputs::new(vec![SequenceConditioningInput::new(
                &f.host,
                ProgramValueId::new(VISUAL).unwrap(),
                ResolvedTensorSpec::new(
                    vec![16, 2],
                    ElementType::F32,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                SequenceConditioningCoverage::ZeroOutsideSegments,
                vec![SequenceConditioningSegment::new(0, source.clone()).unwrap()],
            )
            .unwrap()])
            .unwrap(),
        );
        session.bind_conditioning_inputs(bound.clone()).unwrap();
        // The immutable owner contains only rows 0/1. Actual wave uploads must
        // supply zeros for the declared gap; they cannot extend that owner.
        let mut upload = f
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 16 * 2 * 4)
            .unwrap();
        upload.as_mut()[..source.len()].copy_from_slice(source.bytes());
        if nonzero_gap {
            upload.as_mut()[16..20].copy_from_slice(&1.0_f32.to_le_bytes());
        }
        let upload = upload.freeze();
        let first = f.try_step(&[session.clone()], &[&[3, 5]], &[0]).unwrap();
        f.try_compute_with_visual(&first, &[&[3, 5]], Some(&upload))
            .unwrap();
        f.select(first, vec![SelectedPrefix::One]);
        assert_eq!(f.runtime.probes()[0][0].values(), [0., 16.]);
        assert_eq!(f.runtime.probes()[0][1].values(), [8., 16., 24.]);

        let next = f.try_step(&[session.clone()], &[&[3, 7, 2]], &[1]).unwrap();
        let before = f.runtime.submitted.load(Ordering::SeqCst);
        let result = f.try_compute_with_visual(&next, &[&[3, 7, 2]], Some(&upload));
        if nonzero_gap {
            assert!(result.is_err());
            assert_eq!(f.runtime.submitted.load(Ordering::SeqCst), before);
            assert_eq!(f.runtime.probes()[0][0].values(), [0., 16.]);
            assert_eq!(f.runtime.probes()[0][1].values(), [8., 16., 24.]);
            assert!(next
                .step
                .try_prepare_selected_state_commit(vec![SelectedPrefix::All])
                .is_err());
            next.step.try_abort().unwrap();
            drop(next.batch);
        } else {
            result.unwrap();
            f.select(next, vec![SelectedPrefix::All]);
            // Accepted t0=8, then row1 t1=27 and zero-gap row2 t2=2.
            // Discarded token5 or reusing a populated visual row changes these.
            let actual = f.runtime.probes();
            assert_eq!(actual[0][0].values(), [54., 20.]);
            assert_eq!(actual[0][1].values(), [37., 90., 154.]);
            assert_eq!(actual[0][2].values(), [40., 167., 174.]);
        }
        drop((upload, source, bound));
        assert_eq!(
            f.host.usage().unwrap().allocated_bytes(),
            original_charge,
            "session retains only the original two-row conditioning owner"
        );
        let host = f.host.clone();
        f.close(vec![session]);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn selected_prefix_output_only_input_is_allowed_only_after_all_state_dependencies() {
    // Same real compiled family/Stored schema. Only the selector's graph edge
    // differs: a pure output node is legal, a path into KV/state is not.
    let f = Fixture::with_output_only(false).unwrap();
    f.close(vec![]);
    assert!(Fixture::with_output_only(true).is_err());
}

#[test]
fn selected_prefix_terminal_only_outputs_cannot_feed_state() {
    // Actual selected providers, Stored weights and full candidate effects.
    // A terminal node after the state graph is legal. Moving it before the
    // state node and consuming its output is acyclic but not prefix-safe.
    let f = Fixture::with_terminal(false).unwrap();
    f.close(vec![]);
    let Err(error) = Fixture::with_terminal(true) else {
        panic!("terminal output affecting recurrent/KV state must be rejected");
    };
    assert!(
        error
            .to_string()
            .contains("terminal candidate output affects persistent state"),
        "{error}"
    );
}

#[test]
fn selected_prefix_proof_cannot_be_borrowed_from_another_step_or_partial_state_contract() {
    assert!(
        Fixture::with_effects(false).is_err(),
        "undeclared KV writer must prevent candidate compilation"
    );
    let f = Fixture::new();
    let first = f.session("first");
    let second = f.session("second");
    let first_wave = f.try_step(&[first.clone()], &[&[3, 5]], &[0]).unwrap();
    f.compute(&first_wave, &[&[3, 5]]);
    let second_wave = f.try_step(&[second.clone()], &[&[11, 13]], &[0]).unwrap();
    assert!(
        second_wave
            .step
            .try_prepare_selected_state_commit(vec![SelectedPrefix::Two])
            .is_err(),
        "other completed step is not this frame's proof"
    );
    second_wave.step.try_abort().unwrap();
    drop(second_wave.batch);
    f.select(first_wave, vec![SelectedPrefix::Two]);
    let next = f.try_step(&[first.clone()], &[&[3, 5, 7]], &[2]).unwrap();
    f.compute(&next, &[&[3, 5, 7]]);
    f.select(next, vec![SelectedPrefix::All]);
    assert_eq!(f.runtime.probes()[0][0].values(), [10., 20.]);
    assert_eq!(f.runtime.probes()[0][1].values(), [15., 36., 56.]);
    f.close(vec![first, second]);
}
