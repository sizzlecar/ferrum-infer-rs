use super::*;
mod expiry;
mod original_bytes;
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use ferrum_types::SloCostProfileClockBasis;

fn domain(boot: u8, namespace: u64) -> CostMonotonicDomainV1 {
    CostMonotonicDomainV1::new_linux_boottime(
        [boot; 16],
        1,
        std::num::NonZeroU64::new(namespace).unwrap(),
        0,
        0,
    )
    .unwrap()
}
fn bound_header(domain: CostMonotonicDomainV1) -> StructuredServiceHeaderV7 {
    let h = block_header();
    StructuredServiceHeaderV7::new_with_monotonic_domain(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        h.declaration,
        h.maximum_file_bytes,
        domain,
    )
    .unwrap()
}

#[test]
fn source7_same_boot_header_none_preserves_original_wire_and_protocol() {
    let h = block_header();
    let bytes = record_bytes_v7(&h).unwrap();
    assert!(serde_json::from_slice::<serde_json::Value>(&bytes)
        .unwrap()
        .get("monotonic_domain")
        .is_none());
    let mut legacy = Sha256::new();
    legacy.update(SERVICE_SOURCE_PROTOCOL_V7.as_bytes());
    legacy.update(MODEL_REVISION_V2.as_bytes());
    legacy.update(h.declaration_sha256);
    legacy.update(serde_json::to_vec(&h.fingerprint).unwrap());
    legacy.update(h.maximum_file_bytes.to_le_bytes());
    assert_eq!(h.protocol, <[u8; 32]>::from(legacy.finalize()));
    let parsed: StructuredServiceHeaderV7 = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(record_bytes_v7(&parsed).unwrap(), bytes);
    let bound = bound_header(domain(1, 10));
    assert_ne!(bound.protocol, h.protocol);
    assert_ne!(bound.protocol, bound_header(domain(2, 10)).protocol);
    assert_ne!(bound.protocol, bound_header(domain(1, 11)).protocol);
    let mut forged = bound;
    forged.monotonic_domain = Some(domain(2, 10));
    assert!(StructuredServiceCollectorV7::new(forged, CostProfileLoadLimits::default()).is_err());
}

#[test]
fn source7_same_boot_replays_original_children_without_renewing_age() {
    let original = domain(1, 10);
    let (bytes, _, checkpoint, closing) = collected_with_header(bound_header(original.clone()));
    let limits = CostProfileLoadLimits::default();
    let memory = checkpoint
        .activate_same_boot_memory(paired(70_000), &original, &limits)
        .unwrap();
    let files = Files::new(&bytes);
    export_structured_profile_v14_same_boot(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &original,
        paired(70_000),
        &limits,
    )
    .unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    assert!(metadata.get("source_clock_max_error_ns").is_none());
    let restarted = domain(1, 10); // A separate runtime's independently constructed OS identity.
    let imported = load_structured_profile_v14_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &restarted,
        StructuredServiceClockV7 {
            monotonic_ns: 75_000,
            wall_unix_ns: 2,
        },
    )
    .unwrap();
    assert_eq!(memory.children.len(), imported.children.len());
    for (before, after) in memory.children.iter().zip(&imported.children) {
        assert_eq!(before.parameters_signature(), after.parameters_signature());
        assert_eq!(before.monotonic_domain(), Some(&original));
        assert_eq!(after.monotonic_domain(), Some(&original));
        assert_eq!(
            before.provenance().clock.source_monotonic_anchor_ns,
            after.provenance().clock.source_monotonic_anchor_ns
        );
        assert_eq!(
            before.provenance().clock.model_anchor_ns,
            after.provenance().clock.model_anchor_ns
        );
        assert_eq!(before.provenance().protocol, after.provenance().protocol);
        assert_eq!(
            after.provenance().clock_basis,
            SloCostProfileClockBasis::SameBootMonotonic
        );
        assert_eq!(
            after.provenance().oldest_imported_age_ns,
            before.provenance().oldest_imported_age_ns + 5_000
        );
        assert_eq!(
            after.provenance().newest_imported_age_ns,
            before.provenance().newest_imported_age_ns + 5_000
        );
        assert_eq!(before.provenance().phases, after.provenance().phases);
    }
    assert_eq!(imported.source_sha256, memory.source_sha256);
    assert_eq!(
        imported.children[0]
            .predict_query_local(&old::fingerprint(), &query(), 75_000)
            .unwrap()
            .planning_ns,
        memory.children[0]
            .predict_query_local(&old::fingerprint(), &query(), 75_000)
            .unwrap()
            .planning_ns
    );
    for incompatible in [domain(2, 10), domain(1, 11)] {
        assert!(matches!(
            load_structured_profile_v14_same_boot(
                &files.profile,
                &old::fingerprint(),
                &limits,
                &incompatible,
                paired(75_000)
            ),
            Err(CostProfileError::Clock(_))
        ));
    }
    let mut short = limits.clone();
    short.max_profile_age_ns = std::num::NonZeroU64::new(1).unwrap();
    assert!(load_structured_profile_v14_same_boot(
        &files.profile,
        &old::fingerprint(),
        &short,
        &restarted,
        paired(closing.monotonic_ns + 2)
    )
    .is_err());
    let expired = closing.monotonic_ns + block_header().declaration.settings.max_sample_age_ns + 1;
    assert!(load_structured_profile_v14_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &restarted,
        paired(expired)
    )
    .is_err());
    // Wall-clock APIs still require their original declared accuracy.
    assert!(load_structured_profile_v14(
        &files.profile,
        &old::fingerprint(),
        &limits,
        ProfileLoadClock {
            monotonic_now_ns: 75_000,
            wall_unix_ns: Some(paired(75_000).wall_unix_ns),
            wall_max_error_ns: Some(0)
        }
    )
    .is_err());
}

#[test]
fn source7_same_boot_metadata_cannot_upgrade_an_unbound_original_journal() {
    let (bytes, _, _, _) = collected();
    let files = Files::new(&bytes);
    let limits = CostProfileLoadLimits::default();
    let current = domain(1, 10);
    assert!(export_structured_profile_v14_same_boot(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        &current,
        paired(70_000),
        &limits,
    )
    .is_err());
    export_structured_profile_v14(
        &files.source,
        Sha256::digest(&bytes).into(),
        bytes.len() as u64,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    let mut metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    metadata["monotonic_domain"] = serde_json::to_value(&current).unwrap();
    std::fs::write(&files.profile, serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert!(load_structured_profile_v14_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &current,
        paired(70_000),
    )
    .is_err());
}

#[test]
fn source7_same_boot_preserves_sample_age_policy_and_legacy_collection_expiry() {
    use crate::implementations::continuous::cost_model::structured_v2::{
        NonNegativePlanningEstimatorV1, OwnerPredictionValidityPolicyV1,
    };
    let original = domain(1, 10);
    let limits = CostProfileLoadLimits::default();
    for sample_age in [false, true] {
        let h = block_header();
        let mut declaration = h.declaration;
        declaration.maximum_window_ns = 66_000;
        declaration.schedule.prediction_validity =
            sample_age.then_some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1);
        declaration
            .nonnegative_envelope
            .as_mut()
            .unwrap()
            .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2;
        let header = StructuredServiceHeaderV7::new_with_monotonic_domain(
            h.capture_identity,
            h.generation,
            h.fingerprint,
            h.producer,
            h.opening,
            declaration,
            h.maximum_file_bytes,
            original.clone(),
        )
        .unwrap();
        let (bytes, _, checkpoint, _) = collected_with_header(header);
        let memory = checkpoint
            .activate_same_boot_memory(paired(65_500), &original, &limits)
            .unwrap();
        let child = &memory.children[0];
        let before = child
            .predict_query_local(&old::fingerprint(), &query(), 65_500)
            .unwrap();
        let files = Files::new(&bytes);
        export_structured_profile_v14_same_boot(
            &files.source,
            Sha256::digest(&bytes).into(),
            bytes.len() as u64,
            &files.profile,
            &original,
            paired(65_500),
            &limits,
        )
        .unwrap();
        let restarted = load_structured_profile_v14_same_boot(
            &files.profile,
            &old::fingerprint(),
            &limits,
            &original,
            paired(70_000),
        );
        if sample_age {
            let restarted = restarted.unwrap();
            let after = restarted.children[0]
                .predict_query_local(&old::fingerprint(), &query(), 70_000)
                .unwrap();
            assert_eq!(after.valid_until_ns, before.valid_until_ns);
            assert_eq!(after.planning_ns, before.planning_ns);
            assert!(after.valid_until_ns > 70_000);
            assert!(load_structured_profile_v14_same_boot(
                &files.profile,
                &old::fingerprint(),
                &limits,
                &original,
                paired(after.valid_until_ns + 1),
            )
            .is_err());
        } else {
            assert_eq!(before.valid_until_ns, 66_001);
            assert!(matches!(restarted, Err(CostProfileError::Clock(_))));
        }
    }
}

#[test]
fn source7_same_boot_phase_support_replays_original_contract_receipts_and_expiry() {
    use crate::implementations::continuous::cost_model::structured_v2::{
        NonNegativePlanningEstimatorV1, OwnerPhaseSupportPolicyV1, OwnerPredictionValidityPolicyV1,
    };

    let original = domain(1, 10);
    let h = block_header();
    let mut declaration = h.declaration;
    declaration.maximum_window_ns = 66_000;
    declaration.schedule.phase_support = Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1);
    declaration.schedule.prediction_validity =
        Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1);
    declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2;
    let header = StructuredServiceHeaderV7::new_with_monotonic_domain(
        h.capture_identity,
        h.generation,
        h.fingerprint,
        h.producer,
        h.opening,
        declaration,
        h.maximum_file_bytes,
        original.clone(),
    )
    .unwrap();
    let protocol = header.protocol;
    // The existing producer executes discovery and three separate complete
    // phase blocks. Export must replay these original records, not re-fit a
    // metadata-only reconstruction or silently revert the new policy to None.
    let (bytes, collector, checkpoint, _) = collected_with_header(header);
    let source_receipt = checkpoint.source_receipt();
    assert_eq!(collector.source_receipt(), source_receipt);
    let limits = CostProfileLoadLimits::default();
    let memory = checkpoint
        .activate_same_boot_memory(paired(65_500), &original, &limits)
        .unwrap();
    let before = &memory.children[0];
    let original_contract = before.model.owner_block_contract().unwrap();
    assert_eq!(
        original_contract.schedule.phase_support,
        Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1)
    );
    let before_prediction = before
        .predict_query_local(&old::fingerprint(), &query(), 65_500)
        .unwrap();
    let files = Files::new(&bytes);
    let exported = export_structured_profile_v14_same_boot(
        &files.source,
        source_receipt.1,
        source_receipt.0,
        &files.profile,
        &original,
        paired(65_500),
        &limits,
    )
    .unwrap();
    assert_eq!(exported.source_sha256, source_receipt.1);
    assert_eq!(exported.source_bytes, source_receipt.0);

    // A new process has the same boot/namespace identity, a later monotonic
    // reading, and an unrelated wall clock. Collection has already expired.
    let restarted_domain = domain(1, 10);
    let imported = load_structured_profile_v14_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &restarted_domain,
        StructuredServiceClockV7 {
            monotonic_ns: 70_000,
            wall_unix_ns: 2,
        },
    )
    .unwrap();
    assert_eq!(imported.children.len(), memory.children.len());
    assert_eq!(imported.source_sha256, source_receipt.1);
    assert_eq!(imported.source_bytes, source_receipt.0);
    assert_eq!(imported.capture_protocol, protocol);
    assert_eq!(imported.offered_attempts, memory.offered_attempts);
    assert_eq!(imported.total_shape_rows, memory.total_shape_rows);
    let after = &imported.children[0];
    assert_eq!(after.model.owner_block_contract(), Some(original_contract));
    assert_eq!(after.parameters_signature(), before.parameters_signature());
    assert_eq!(after.monotonic_domain(), Some(&original));
    assert_eq!(after.provenance().phases, before.provenance().phases);
    assert_eq!(
        after.provenance().clock.source_monotonic_anchor_ns,
        before.provenance().clock.source_monotonic_anchor_ns
    );
    assert_eq!(
        after.provenance().clock.model_anchor_ns,
        before.provenance().clock.model_anchor_ns
    );
    assert_eq!(
        after.provenance().clock_basis,
        SloCostProfileClockBasis::SameBootMonotonic
    );
    assert_eq!(
        after.provenance().oldest_imported_age_ns,
        before.provenance().oldest_imported_age_ns + 4_500
    );
    assert_eq!(
        after.provenance().newest_imported_age_ns,
        before.provenance().newest_imported_age_ns + 4_500
    );
    let after_prediction = after
        .predict_query_local(&old::fingerprint(), &query(), 70_000)
        .unwrap();
    assert_eq!(after_prediction.planning_ns, before_prediction.planning_ns);
    assert_eq!(
        after_prediction.valid_until_ns,
        before_prediction.valid_until_ns
    );
    assert!(after_prediction.valid_until_ns > 70_000);
    assert!(load_structured_profile_v14_same_boot(
        &files.profile,
        &old::fingerprint(),
        &limits,
        &restarted_domain,
        paired(after_prediction.valid_until_ns + 1),
    )
    .is_err());
    for incompatible in [domain(2, 10), domain(1, 11)] {
        assert!(matches!(
            load_structured_profile_v14_same_boot(
                &files.profile,
                &old::fingerprint(),
                &limits,
                &incompatible,
                paired(70_000),
            ),
            Err(CostProfileError::Clock(_))
        ));
    }
    // Loading/exporting never appends records or renews the original receipt.
    assert_eq!(std::fs::read(&files.source).unwrap(), bytes);
    assert_eq!(collector.source_receipt(), source_receipt);
}
