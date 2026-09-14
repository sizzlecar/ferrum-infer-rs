use super::*;
use crate::vnext::{
    PreparedSequenceStateTransfer, ProviderCheckpointStateLayout, SequenceCheckpointBytePlan,
    SequenceStateTransferKind, SequenceStateTransferPreparation, StateCapacityDemand,
};

#[path = "../../../../tests/vnext_sequence_checkpoint_contract/fixture.rs"]
pub(super) mod checkpoint_fixture;
#[path = "fixture.rs"]
mod fixture;
#[path = "../../../../tests/vnext_core_contract/mod.rs"]
mod vnext_core_contract;
pub(super) use fixture::{reserve_restore, RestoreHarness};

fn prepare(
    harness: &RestoreHarness,
    guard: &PreparedSequenceStateTransfer<TestRuntime>,
    boundary: u64,
) -> (PreparedBackingInitializations, SequenceCheckpointBytePlan) {
    let bytes = harness.fixture.plan.checkpoint_byte_plan(boundary).unwrap();
    let prepared = PreparedBackingInitializations::prepare_restore(
        guard,
        harness.fixture.layout(),
        &bytes,
        "transfer/restore-test",
    )
    .unwrap();
    (prepared, bytes)
}

fn all_statuses(
    guard: &PreparedSequenceStateTransfer<TestRuntime>,
) -> Vec<BackingInitializationStatus> {
    guard
        .backing()
        .backing_slices()
        .iter()
        .filter_map(|slice| slice.initialization_status().unwrap())
        .collect()
}

#[test]
fn restore_initialization_covers_capacity_and_is_not_repeated_by_the_first_step() {
    let harness = RestoreHarness::new(checkpoint_fixture::Spec::default());
    let guard = harness.reserve();
    let (mut prepared, bytes) = prepare(&harness, &guard, 1);
    assert!(
        bytes.logical_bytes()
            < guard
                .backing()
                .backing_slices()
                .iter()
                .map(|s| s.capacity_size_bytes())
                .sum()
    );
    assert!(all_statuses(&guard)
        .iter()
        .all(|s| *s == BackingInitializationStatus::Prepared));
    let mut commands = DeviceCommandBatch::with_capacity(0);
    assert_eq!(
        prepared
            .encode_restore(&guard, harness.runtime.as_ref(), &mut commands)
            .unwrap(),
        2
    );
    assert_eq!(commands.len(), 2);
    // This test drives the initialization participant's terminal protocol;
    // native transfer tests separately prove copy ordering and the real fence.
    prepared.mark_in_flight().unwrap();
    prepared.finish(true).unwrap();
    assert!(all_statuses(&guard)
        .iter()
        .all(|s| *s == BackingInitializationStatus::Initialized));
    drop(prepared);
    drop(guard);

    let lane = harness.root.create_execution_lane().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&harness.session)]).unwrap();
    let step = match batch
        .try_begin_step(
            StepResourceAdmissionRequest::new(
                batch.bind_work_shape(vec![token_span(1)]).unwrap(),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap(),
            &lane,
        )
        .unwrap()
    {
        StepResourceAdmissionDecision::Admitted(step) => step,
        _ => panic!("first model step must admit"),
    };
    let later = PreparedBackingInitializations::prepare(&step, "wave/first-model-step").unwrap();
    let mut later_commands = DeviceCommandBatch::with_capacity(0);
    assert_eq!(
        later
            .encode(&step, harness.runtime.as_ref(), &mut later_commands)
            .unwrap(),
        0
    );
    drop(later);
    step.try_retire_normal().unwrap();
    drop(batch);
    drop(lane);
    harness.close();
}

#[test]
fn restore_initialization_drop_before_submission_rolls_back_and_retry_is_independent() {
    let harness = RestoreHarness::new(checkpoint_fixture::Spec::default());
    let guard = harness.reserve();
    let (prepared, _) = prepare(&harness, &guard, 1);
    assert!(PreparedBackingInitializations::prepare_restore(
        &guard,
        harness.fixture.layout(),
        &harness.fixture.plan.checkpoint_byte_plan(1).unwrap(),
        "same-transfer",
    )
    .is_err());
    drop(prepared);
    assert!(all_statuses(&guard)
        .iter()
        .all(|s| *s == BackingInitializationStatus::Pending));
    let (retry, _) = prepare(&harness, &guard, 1);
    drop(retry);
    drop(guard);
    harness.close();
}

#[test]
fn restore_initialization_failed_or_unknown_submission_poison_all_owned_cells() {
    for failure in 0..3 {
        let harness = RestoreHarness::new(checkpoint_fixture::Spec::default());
        let guard = harness.reserve();
        let (mut prepared, _) = prepare(&harness, &guard, 1);
        match failure {
            0 => {
                prepared.mark_in_flight().unwrap();
                prepared.finish(false).unwrap();
            }
            1 => prepared.mark_indeterminate(),
            _ => prepared.mark_in_flight().unwrap(),
        }
        drop(prepared);
        assert!(all_statuses(&guard)
            .iter()
            .all(|s| *s == BackingInitializationStatus::Poisoned));
        assert!(PreparedBackingInitializations::prepare_restore(
            &guard,
            harness.fixture.layout(),
            &harness.fixture.plan.checkpoint_byte_plan(1).unwrap(),
            "retry",
        )
        .is_err());
        drop(guard);
        harness.close();
    }
}

#[test]
fn restore_initialization_preserves_none_and_handles_multiple_extents_of_one_resource() {
    let mut spec = checkpoint_fixture::Spec::default();
    // Four token positions fit in one supported physical page. Extending to
    // eight therefore allocates a second extent through the real paged path.
    spec.profile = vnext_core_contract::paged_storage_profile(65536);
    spec.port_profile = spec.profile;
    spec.states[0].tensor.dimensions = vec![16384];
    spec.states[0].capacity_demand = StateCapacityDemand::TokenScaled {
        bytes_per_token: 16384,
        maximum_tokens: 64,
    };
    spec.states[1].initialization = StateInitialization::None;
    let harness = RestoreHarness::new(spec);
    harness.extend(8);
    let guard = harness.reserve();
    let (mut prepared, _) = prepare(&harness, &guard, 5);
    let zero_slices = guard
        .backing()
        .backing_slices()
        .iter()
        .filter(|slice| slice.evidence().initialization() == StateInitialization::Zero)
        .count();
    assert_eq!(zero_slices, 2);
    let mut commands = DeviceCommandBatch::with_capacity(0);
    assert_eq!(
        prepared
            .encode_restore(&guard, harness.runtime.as_ref(), &mut commands)
            .unwrap(),
        zero_slices
    );
    prepared.mark_in_flight().unwrap();
    prepared.finish(true).unwrap();
    assert!(guard
        .backing()
        .backing_slices()
        .iter()
        .filter(|s| s.evidence().initialization() == StateInitialization::None)
        .all(|s| s.initialization_status().unwrap().is_none()));
    drop(prepared);
    drop(guard);
    harness.close();
}

#[test]
fn restore_initialization_disjoint_model_states_share_one_complete_capacity_initialization() {
    let mut spec = checkpoint_fixture::Spec::default();
    spec.states[0] = spec.states[1].clone();
    spec.states[0].id = crate::vnext::StateId::new("state.0").unwrap();
    spec.states[0].value_id = ProgramValueId::new("value.state.0").unwrap();
    spec.states[0].capacity_demand = StateCapacityDemand::FixedPerScope;
    spec.layouts[0] = ProviderCheckpointStateLayout::ContiguousBoundaryValue;
    spec.locations[1] = (spec.locations[0].0.clone(), 16);
    let harness = RestoreHarness::new(spec);
    let guard = harness.reserve();
    let (prepared, bytes) = prepare(&harness, &guard, 1);
    assert_eq!(bytes.resources().len(), 1);
    assert_eq!(bytes.resources()[0].ranges().len(), 2);
    assert_eq!(bytes.logical_bytes(), 8);
    assert_eq!(guard.backing().backing_slices().len(), 1);
    let mut commands = DeviceCommandBatch::with_capacity(0);
    assert_eq!(
        prepared
            .encode_restore(&guard, harness.runtime.as_ref(), &mut commands)
            .unwrap(),
        1
    );
    drop(prepared);
    drop(guard);
    harness.close();
}

#[test]
fn restore_initialization_rejects_wrong_layout_boundary_and_non_fresh_target() {
    let harness = RestoreHarness::new(checkpoint_fixture::Spec::default());
    let guard = harness.reserve();
    let bytes = harness.fixture.plan.checkpoint_byte_plan(1).unwrap();
    let mut wrong_spec = checkpoint_fixture::Spec::default();
    wrong_spec.states[1].initialization = StateInitialization::None;
    let wrong = checkpoint_fixture::Fixture::build(wrong_spec).unwrap();
    assert!(PreparedBackingInitializations::prepare_restore(
        &guard,
        wrong.layout(),
        &bytes,
        "wrong-layout"
    )
    .is_err());
    let too_long = harness.fixture.plan.checkpoint_byte_plan(17).unwrap();
    assert!(PreparedBackingInitializations::prepare_restore(
        &guard,
        harness.fixture.layout(),
        &too_long,
        "beyond-backing"
    )
    .is_err());
    assert!(all_statuses(&guard)
        .iter()
        .all(|s| *s == BackingInitializationStatus::Pending));
    harness.session.request_cancel().unwrap();
    assert!(PreparedBackingInitializations::prepare_restore(
        &guard,
        harness.fixture.layout(),
        &bytes,
        "cancelled"
    )
    .is_err());
    drop(guard);
    harness.close();
}

#[test]
fn restore_initialization_rejects_equal_bytes_from_other_layout_or_plan() {
    let harness = RestoreHarness::new(checkpoint_fixture::Spec::default());
    let guard = harness.reserve();
    let bytes = harness.fixture.plan.checkpoint_byte_plan(1).unwrap();

    // The state resources, offsets and byte counts remain equal. Changing only
    // the provider's partition contract must still reject this layout pairing.
    let other_layout = checkpoint_fixture::Fixture::build(checkpoint_fixture::Spec {
        numerics: crate::vnext::CheckpointPartitionNumerics::SamePartitionOnly,
        ..checkpoint_fixture::Spec::default()
    })
    .unwrap();
    let other_layout_bytes = other_layout.plan.checkpoint_byte_plan(1).unwrap();
    assert_eq!(bytes.resources(), other_layout_bytes.resources());
    assert_ne!(
        bytes.layout_fingerprint(),
        other_layout_bytes.layout_fingerprint()
    );
    assert!(PreparedBackingInitializations::prepare_restore(
        &guard,
        other_layout.layout(),
        &bytes,
        "same-bytes-different-layout"
    )
    .is_err());

    // A mutually consistent layout/byte-plan pair from another real plan is
    // still not bound to this target, even though all copy ranges are equal.
    let other_plan = checkpoint_fixture::Fixture::build(checkpoint_fixture::Spec {
        family_id: crate::vnext::ModelFamilyId::new("family.other-checkpoint-plan").unwrap(),
        ..checkpoint_fixture::Spec::default()
    })
    .unwrap();
    let other_bytes = other_plan.plan.checkpoint_byte_plan(1).unwrap();
    assert_eq!(bytes.resources(), other_bytes.resources());
    assert_ne!(bytes.plan_hash(), other_bytes.plan_hash());
    assert!(PreparedBackingInitializations::prepare_restore(
        &guard,
        other_plan.layout(),
        &other_bytes,
        "same-bytes-different-plan"
    )
    .is_err());
    assert!(all_statuses(&guard)
        .iter()
        .all(|status| *status == BackingInitializationStatus::Pending));

    // Rejection must leave the original matching preparation available.
    let (prepared, _) = prepare(&harness, &guard, 1);
    drop(prepared);
    drop(guard);
    harness.close();
}

#[test]
fn restore_initialization_boundary_only_state_does_not_infer_a_token_capacity_limit() {
    let mut spec = checkpoint_fixture::Spec::default();
    spec.states.remove(0);
    spec.locations.remove(0);
    spec.layouts.remove(0);
    let harness = RestoreHarness::new(spec);
    let guard = harness.reserve();
    assert_eq!(guard.backing().committed_tokens(), 1);
    let (prepared, bytes) = prepare(&harness, &guard, 9);
    assert_eq!(bytes.logical_bytes(), 4);
    let mut commands = DeviceCommandBatch::with_capacity(0);
    assert_eq!(
        prepared
            .encode_restore(&guard, harness.runtime.as_ref(), &mut commands)
            .unwrap(),
        1
    );
    drop(prepared);
    drop(guard);
    harness.close();
}

#[test]
fn restore_initialization_encode_rejects_other_runtime_and_replaced_reservation() {
    let harness = RestoreHarness::new(checkpoint_fixture::Spec::default());
    let guard = harness.reserve();
    let (prepared, _) = prepare(&harness, &guard, 1);
    let mut commands = DeviceCommandBatch::with_capacity(0);
    let wrong_runtime = TestRuntime::new(
        harness.runtime.descriptor.id.clone(),
        harness.runtime.descriptor.total_memory_bytes,
        harness.runtime.descriptor.dynamic_storage_profiles.clone(),
    );
    assert!(prepared
        .encode_restore(&guard, &wrong_runtime, &mut commands)
        .is_err());
    assert!(commands.is_empty());
    drop(guard);
    let replacement = harness.reserve();
    assert!(prepared
        .encode_restore(&replacement, harness.runtime.as_ref(), &mut commands)
        .is_err());
    assert!(commands.is_empty());
    drop(prepared);
    assert!(all_statuses(&replacement)
        .iter()
        .all(|s| *s == BackingInitializationStatus::Pending));
    drop(replacement);
    harness.close();
}
