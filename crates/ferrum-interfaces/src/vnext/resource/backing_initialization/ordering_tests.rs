use super::*;
use crate::vnext::resource::backing_initialization::order::{
    InitializationOrder, InitializationRanges,
};
use crate::vnext::{ResourcePlanningAvailability, ResourcePlanningLimits, ResourcePlanningUnknown};

fn harness() -> RestoreHarness {
    let mut spec = checkpoint_fixture::Spec::default();
    spec.device_id = Some(
        DeviceId::new(format!(
            "zero-order-{}",
            NEXT_TEST_DEVICE.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap(),
    );
    spec.states[0].tensor.dimensions = vec![64];
    spec.states[0].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 64,
        maximum_tokens: 64,
    };
    RestoreHarness::new(spec)
}

fn snapshot(
    h: &RestoreHarness,
    sessions: &[Arc<SequenceSession<TestRuntime>>],
) -> crate::vnext::ResourcePlanningView {
    match h.root.resource_planning_view(
        &sessions.iter().map(Arc::as_ref).collect::<Vec<_>>(),
        ResourcePlanningLimits::default(),
        &mut || true,
    ) {
        ResourcePlanningAvailability::Known(view) => view,
        other => panic!("real idle initialization snapshot: {other:?}"),
    }
}

fn actual_zero_bytes(
    h: &RestoreHarness,
    sessions: Vec<Arc<SequenceSession<TestRuntime>>>,
) -> Vec<u64> {
    let batch = ExecutionBatchParticipants::new(sessions).unwrap();
    let lane = h.root.create_execution_lane().unwrap();
    let step = match batch
        .try_begin_step(
            StepResourceAdmissionRequest::new(
                batch
                    .bind_work_shape(vec![token_span(1); batch.len() as usize])
                    .unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            &lane,
        )
        .unwrap()
    {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("resident real initialization step must admit"),
    };
    let prepared = PreparedBackingInitializations::prepare(&step, "zero-order-wave").unwrap();
    h.runtime.encoded_zero_bytes.lock().unwrap().clear();
    let mut commands = DeviceCommandBatch::with_capacity(0);
    let count = prepared
        .encode(&step, h.runtime.as_ref(), &mut commands)
        .unwrap();
    let result = h.runtime.encoded_zero_bytes.lock().unwrap().clone();
    assert_eq!(count, result.len());
    assert_eq!(commands.len(), result.len());
    // No fence was submitted. Dropping only this preparation restores Pending;
    // the next subset test must execute the same real initialization obligations.
    drop(commands);
    drop(prepared);
    step.try_retire_normal().unwrap();
    drop(batch);
    drop(lane);
    result
}

#[test]
fn pending_zero_order_matches_actual_multi_participant_and_subsets_without_sorting_bytes() {
    let h = harness();
    // Contiguous backing cannot grow after admission. Reserve the second
    // participant's complete initial capacity through the real admission path.
    let second = admitted_sequence_with_initial_tokens(&h.root, "zero-order-second", 8, 16)
        .open_session()
        .unwrap();
    let sessions = vec![Arc::clone(&h.session), Arc::clone(&second)];
    let view = snapshot(&h, &sessions);
    let expected = view
        .pending_zero_transfer_bytes(&[0, 1], &mut || true)
        .unwrap()
        .unwrap();
    assert!(
        expected.iter().any(|&bytes| bytes != expected[0]),
        "heterogeneous real extents"
    );
    assert_eq!(expected, actual_zero_bytes(&h, sessions.clone()));
    assert_eq!(
        view.pending_zero_transfer_bytes(&[1, 0], &mut || true)
            .unwrap()
            .unwrap(),
        expected,
        "whole-wave cell ordering cannot become participant concatenation"
    );
    for index in 0..2 {
        assert_eq!(
            view.pending_zero_transfer_bytes(&[index], &mut || true)
                .unwrap()
                .unwrap(),
            actual_zero_bytes(&h, vec![Arc::clone(&sessions[index])]),
            "the same snapshot preserves actual relative ordering for each legal subset"
        );
    }
    assert_eq!(
        view.pending_zero_transfer_bytes(&[0, 1], &mut || false),
        Err(ResourcePlanningUnknown::BudgetExhausted)
    );
    assert_eq!(
        view.pending_zero_transfer_bytes(&[0, 0], &mut || true),
        Err(ResourcePlanningUnknown::InvalidDemand)
    );
    assert_eq!(
        view.pending_zero_transfer_bytes(&[2], &mut || true),
        Err(ResourcePlanningUnknown::InvalidInput)
    );
    drop(view);
    drop(sessions);
    second.try_abort_if_quiescent().unwrap();
    drop(second);
    h.close();
}

#[test]
fn pending_zero_order_deduplicates_same_cell_slices_but_rejects_cross_owner_alias() {
    let h = harness();
    let second = h.new_session("zero-order-other-owner");
    let before = snapshot(&h, &[Arc::clone(&h.session)]);
    let expected = before
        .pending_zero_transfer_bytes(&[0], &mut || true)
        .unwrap()
        .unwrap();
    let owner = BatchParticipantAuthority::new(
        h.session.sequence_authority(),
        h.session.request_authority(),
    );
    let other =
        BatchParticipantAuthority::new(second.sequence_authority(), second.request_authority());
    let guard = h.reserve();
    let mut order = InitializationOrder::new();
    for slice in guard.backing().backing_slices() {
        order.insert(owner, slice).unwrap();
        order.insert(owner, slice).unwrap();
    }
    let mut lengths = Vec::new();
    for (_, _, slices) in order.finish() {
        let mut seen = InitializationRanges::new();
        for slice in slices {
            for segment in slice.evidence().segments() {
                if seen.insert(segment) {
                    lengths.push(segment.length_bytes());
                }
            }
        }
    }
    assert_eq!(
        lengths, expected,
        "repeated authorities are not repeated zero commands"
    );
    let mut conflicting = InitializationOrder::new();
    let slice = &guard.backing().backing_slices()[0];
    conflicting.insert(owner, slice).unwrap();
    assert!(
        conflicting.insert(other, slice).is_err(),
        "actual cross-owner shared-cell guard remains closed"
    );
    drop(conflicting);
    drop(guard);
    assert_eq!(
        expected,
        actual_zero_bytes(&h, vec![Arc::clone(&h.session)])
    );
    drop(before);
    second.try_abort_if_quiescent().unwrap();
    drop(second);
    h.close();
}
