use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    NonNegativePlanningEstimatorV1, OwnerInputReadinessV1,
};

fn candidate_header() -> StructuredServiceHeaderV7 {
    let mut h = block_header();
    h.declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1;
    h.declaration.schedule = OwnerBlockScheduleV1::new_with_input_readiness(
        8,
        [8; 3],
        [8; 3],
        OwnerInputReadinessV1::new([4; 3], 32_000_000).unwrap(),
    )
    .unwrap();
    h.declaration.settings.max_phase_samples = *h
        .declaration
        .schedule
        .maximum_phase_members
        .iter()
        .max()
        .unwrap();
    StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}

#[test]
fn source7_joint_bank_replays_original_three_phase_seals_and_keeps_expiry() {
    let (bytes, _, checkpoint, closing) = collected_with_header(candidate_header());
    let limits = CostProfileLoadLimits::default();
    let replay = replay_structured_source_v7(&bytes, &limits).unwrap();
    let now = paired(closing.monotonic_ns + 1);
    let live = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replay = replay.activate_same_process_memory(now, &limits).unwrap();
    let a = &live.children[0];
    let b = &replay.children[0];
    assert_eq!(a.parameters_signature(), b.parameters_signature());
    let p = a
        .predict_query_local(&old::fingerprint(), &query(), now.monotonic_ns)
        .unwrap();
    let replay_p = b
        .predict_query_local(&old::fingerprint(), &query(), now.monotonic_ns)
        .unwrap();
    assert_eq!(p.planning_ns, replay_p.planning_ns);
    assert_eq!(p.valid_until_ns, replay_p.valid_until_ns);
    assert!(a
        .predict_query_local(&old::fingerprint(), &query(), p.valid_until_ns + 1)
        .is_err());
    assert!(a
        .predict_query_local(&old::fingerprint(), &query(), closing.monotonic_ns - 1)
        .is_err());
    // The fixed input contract remains explicit and older wire mode unchanged.
    assert!(String::from_utf8(bytes)
        .unwrap()
        .contains("identified_fit_joint_cells_v1"));
    assert!(
        !String::from_utf8(record_bytes_v7(&block_header()).unwrap())
            .unwrap()
            .contains("joint_cells")
    );
}
