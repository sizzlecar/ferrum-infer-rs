use super::*;

const OLD_REVISION: &str = "structured_whole_wave_pending_envelope_v2";
fn clock() -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(1_000_000 + 72 * 2000 + 1399),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 7,
    }
}
#[test]
fn structured_v10_fit_floor_roundtrip_preserves_frozen_components_and_original_ttl() {
    let (bytes, input) = source_with_fit_tail();
    let files = Files::new(&bytes);
    let receipt = export_structured_profile_v10(
        &files.source,
        Sha256::digest(&bytes).into(),
        &files.profile,
        0,
        &limits(),
    )
    .unwrap();
    let imported =
        load_structured_profile_v10(&files.profile, &fingerprint(), &limits(), clock()).unwrap();
    let query = StructuredQueryV2::exact(input);
    let p = imported
        .predict_query_local(&fingerprint(), &query, 17)
        .unwrap();
    assert_eq!(receipt.model_revision, MODEL_REVISION_V2);
    assert_ne!(receipt.model_revision, OLD_REVISION);
    assert_eq!(imported.provenance().schema_version, 10);
    assert_eq!(receipt.parameters_sha256, imported.parameters_signature());
    assert_eq!(receipt.uncertainty.fit_error_floor_ns, p.fit_error_floor_ns);
    assert_eq!(receipt.uncertainty.residual_ns, p.residual_ns);
    assert_eq!(
        receipt.uncertainty.effective_residual_ns,
        p.effective_residual_ns
    );
    assert!((524..=525).contains(&p.fit_error_floor_ns));
    assert_eq!(p.residual_ns, 0);
    assert_eq!(p.effective_residual_ns, p.fit_error_floor_ns);
    assert_eq!(p.planning_ns, 1610);
    assert_eq!(p.fit_samples, 8);
    assert_eq!(p.residual_samples, 8);
    // Original earliest selected member is call2, finalized at 5100. The tail
    // at a later call changes neither this clock nor the original one-second TTL.
    assert_eq!(p.valid_until_ns, 1_000_005_100);
    assert!(matches!(
        imported.predict_query_local(&fingerprint(), &query, 2_000_000_000),
        Err(StructuredUnknownV2::Stale)
    ));
    let report = serde_json::to_value(&receipt).unwrap();
    assert_eq!(
        report["uncertainty"]["fit_error_floor_ns"],
        p.fit_error_floor_ns
    );
    assert_eq!(report["uncertainty"]["residual_ns"], 0);
    assert_eq!(
        report["uncertainty"]["effective_residual_ns"],
        p.effective_residual_ns
    );
    assert_eq!(report["uncertainty"]["static_margin_ns"], 10);
}

#[test]
fn structured_v10_fit_floor_rejects_legacy_source_and_profile_revision_explicitly() {
    let (bytes, _) = source_with_fit_tail();
    let old_source = mutate(&bytes, |record| {
        if record["schema_version"] == 3 {
            record["model_revision"] = OLD_REVISION.into();
        }
    });
    let old = Files::new(&old_source);
    assert!(matches!(
        export_structured_profile_v10(
            &old.source,
            Sha256::digest(&old_source).into(),
            &old.profile,
            0,
            &limits(),
        ),
        Err(CostProfileError::Metadata(
            "unsupported structured source revision"
        ))
    ));
    assert!(!old.profile.exists());

    let files = Files::new(&bytes);
    export_structured_profile_v10(
        &files.source,
        Sha256::digest(&bytes).into(),
        &files.profile,
        0,
        &limits(),
    )
    .unwrap();
    let mut profile: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    profile["model_revision"] = OLD_REVISION.into();
    std::fs::write(&files.profile, serde_json::to_vec(&profile).unwrap()).unwrap();
    assert!(matches!(
        load_structured_profile_v10(&files.profile, &fingerprint(), &limits(), clock()),
        Err(CostProfileError::Metadata(
            "unsupported structured revision"
        ))
    ));
}

#[test]
fn structured_v10_fit_floor_cannot_replace_parameters_while_retaining_original_source() {
    let (bytes, _) = source_with_fit_tail();
    let files = Files::new(&bytes);
    export_structured_profile_v10(
        &files.source,
        Sha256::digest(&bytes).into(),
        &files.profile,
        0,
        &limits(),
    )
    .unwrap();
    let mut profile: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&files.profile).unwrap()).unwrap();
    let first = profile["parameters_sha256"][0].as_u64().unwrap();
    profile["parameters_sha256"][0] = (first ^ 1).into();
    std::fs::write(&files.profile, serde_json::to_vec(&profile).unwrap()).unwrap();
    assert!(matches!(
        load_structured_profile_v10(&files.profile, &fingerprint(), &limits(), clock()),
        Err(CostProfileError::Metadata(
            "structured envelope differs from original replay"
        ))
    ));
}
