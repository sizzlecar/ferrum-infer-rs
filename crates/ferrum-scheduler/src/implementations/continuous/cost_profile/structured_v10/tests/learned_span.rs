use super::*;
fn clock() -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(1_000_000 + 72 * 2000 + 1399),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 7,
    }
}
#[test]
fn structured_v10_learned_span_roundtrip_and_legacy_disabled_wire() {
    let (bytes, input) = source_with_learned_span(None);
    let files = Files::new(&bytes);
    let receipt = export_structured_profile_v10(
        &files.source,
        Sha256::digest(&bytes).into(),
        &files.profile,
        0,
        &limits(),
    )
    .unwrap();
    let model =
        load_structured_profile_v10(&files.profile, &fingerprint(), &limits(), clock()).unwrap();
    let query = StructuredQueryV2::exact(input);
    let p = model
        .predict_query_local(&fingerprint(), &query, 7)
        .unwrap();
    assert_eq!(p.learned_span_margin_ns, 400);
    assert_eq!(receipt.uncertainty.learned_span_margin_ns, 400);
    assert_eq!(p.planning_ns, 1660); // fit1000 + residual250 + margin10 + span400.
    assert_eq!(p.valid_until_ns, 1_000_005_100);
    assert!(matches!(
        model.predict_query_local(&fingerprint(), &query, 2_000_000_000),
        Err(StructuredUnknownV2::Stale)
    ));
    let (old, old_input) = source();
    let header: serde_json::Value =
        serde_json::from_slice(old.split(|b| *b == b'\n').next().unwrap()).unwrap();
    assert!(header["record"]["settings"].get("learned_drift").is_none());
    let old_files = Files::new(&old);
    let old_receipt = export_structured_profile_v10(
        &old_files.source,
        Sha256::digest(&old).into(),
        &old_files.profile,
        0,
        &limits(),
    )
    .unwrap();
    let old_model =
        load_structured_profile_v10(&old_files.profile, &fingerprint(), &limits(), clock())
            .unwrap();
    assert_eq!(
        old_model
            .predict_query_local(&fingerprint(), &StructuredQueryV2::exact(old_input), 7)
            .unwrap()
            .learned_span_margin_ns,
        0
    );
    assert!(serde_json::to_value(old_receipt).unwrap()["uncertainty"]
        .get("learned_span_margin_ns")
        .is_none());
}
#[test]
fn structured_v10_learned_span_policy_cannot_be_removed_or_changed_on_original_receipt() {
    let (bytes, _) = source_with_learned_span(None);
    for replacement in [
        serde_json::json!({"kind":"disabled"}),
        serde_json::json!({"kind":"observed_residual_span_v1","maximum_span_margin_ns":401}),
    ] {
        let changed = mutate(&bytes, |record| {
            if record["schema_version"] == 3 {
                record["settings"]["learned_drift"] = replacement.clone();
            }
        });
        let files = Files::new(&changed);
        assert!(export_structured_profile_v10(
            &files.source,
            Sha256::digest(&changed).into(),
            &files.profile,
            0,
            &limits()
        )
        .is_err());
        assert!(!files.profile.exists());
    }
}
