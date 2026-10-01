//! Independent source7 replay and profile14 import retain the declared policy
//! and original sample ages. The typed canonical fixture performs no GPU work.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    OwnerPredictionValidityPolicyV1, StructuredUnknownV2,
};

fn header_for(policy: Option<OwnerPredictionValidityPolicyV1>) -> StructuredServiceHeaderV7 {
    let h = block_header();
    let mut declaration = h.declaration.clone();
    declaration.schedule.prediction_validity = policy;
    declaration.maximum_window_ns = 66_000;
    StructuredServiceHeaderV7::new(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        declaration,
        h.maximum_file_bytes,
    )
    .unwrap()
}

#[test]
fn source7_prediction_validity_replay_and_profile_keep_original_sample_expiry() {
    let (bytes, _, checkpoint, _) = collected_with_header(header_for(Some(
        OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1,
    )));
    let limits = CostProfileLoadLimits::default();
    let now = paired(65_500); // source deadline66001; an original wave costs1000ns.
    let memory = checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let replay = replay_structured_source_v7(&bytes, &limits)
        .unwrap()
        .activate_same_process_memory(now, &limits)
        .unwrap();
    let files = Files::new(&bytes);
    export_structured_profile_v14(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    let imported = load_structured_profile_v14(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            wall_unix_ns: Some(now.wall_unix_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: now.monotonic_ns,
        },
    )
    .unwrap();
    let member = &memory.children[0];
    let p = member.provenance();
    let oldest_original = p.clock.model_anchor_ns - p.oldest_imported_age_ns;
    let expiry = oldest_original + member.runtime_limits().1;
    let baseline = member
        .predict_query_local(&old::fingerprint(), &query(), now.monotonic_ns)
        .unwrap();
    assert!(baseline.planning_ns > 66_001 - now.monotonic_ns);
    assert_eq!(baseline.valid_until_ns, expiry);
    for child in [&replay.children[0], &imported.children[0]] {
        assert_eq!(child.parameters_signature(), member.parameters_signature());
        let future = child
            .predict_query_local(&old::fingerprint(), &query(), 70_000)
            .unwrap();
        assert_eq!(future.planning_ns, baseline.planning_ns);
        assert_eq!(future.valid_until_ns, expiry);
        assert!(child
            .predict_query_local(&old::fingerprint(), &query(), expiry)
            .is_ok());
        assert!(matches!(
            child.predict_query_local(&old::fingerprint(), &query(), expiry + 1),
            Err(StructuredUnknownV2::Stale)
        ));
    }
    let tampered = bytes.clone();
    let newline = tampered.iter().position(|b| *b == b'\n').unwrap();
    let mut header: serde_json::Value = serde_json::from_slice(&tampered[..newline]).unwrap();
    header["declaration"]["schedule"]
        .as_object_mut()
        .unwrap()
        .remove("prediction_validity");
    let mut changed = serde_json::to_vec(&header).unwrap();
    changed.extend_from_slice(&tampered[newline..]);
    assert!(replay_structured_source_v7(&changed, &limits).is_err());
    // No option supplied retains the original model clamp despite fresher data.
    let (_, _, old_checkpoint, _) = collected_with_header(header_for(None));
    let old_memory = old_checkpoint
        .activate_same_process_memory(now, &limits)
        .unwrap();
    assert_eq!(
        old_memory.children[0]
            .predict_query_local(&old::fingerprint(), &query(), 65_500)
            .unwrap()
            .valid_until_ns,
        66_001
    );
    assert!(matches!(
        old_memory.children[0].predict_query_local(&old::fingerprint(), &query(), 66_002),
        Err(StructuredUnknownV2::Stale)
    ));
}
