//! Real private receipt/FIFO and product loader; no device performance claim.
use super::*;
use crate::continuous_engine::inner::cost_observation::{profile, trainer};
use ferrum_scheduler::implementations::continuous::cost_model::statistical::{
    model::{WholeWaveModelRevision, WORK_SUPPORT_MODEL_REVISION},
    SelectedStatisticalFamily,
};

#[test]
fn work_support_live_phases_source3_profile8_and_serving_import_are_bound() {
    let mut fixture = Fixture::new();
    fixture.config.predictor = ferrum_types::SloCostPredictor::SelectedWorkSupportV1;
    let mut capture = fixture.begin();
    let mut through = 0;
    for _ in 0..fixture.config.model.min_samples.get() {
        let (ordinal, _, actual) = fixture.completed();
        capture.record(&actual, true).unwrap();
        through = ordinal;
    }
    let frozen = capture.freeze_fit(through).unwrap();
    assert!(!fixture.options.path.exists(), "fit alone cannot publish");
    for _ in 0..fixture.config.model.min_samples.get() {
        let (ordinal, _, actual) = fixture.completed();
        capture.record(&actual, true).unwrap();
        through = ordinal;
    }
    let (cut, support) = capture.finish(through).unwrap();
    let source_bytes = fs::read(&cut.source).unwrap();
    let profile_bytes = fs::read(&cut.profile).unwrap();
    let records: Vec<serde_json::Value> = source_bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(records[0]["schema_version"], 3);
    assert_eq!(records[0]["model_revision"], WORK_SUPPORT_MODEL_REVISION);
    let freeze_at = records
        .iter()
        .position(|r| r["kind"] == "fit_frozen")
        .unwrap();
    assert!(records[1..freeze_at].iter().all(|r| r["phase"] == "fit"));
    assert!(records[freeze_at + 1..records.len() - 1]
        .iter()
        .all(|r| r["phase"] == "residual"));
    assert!(records
        .iter()
        .filter(|r| r["kind"] == "observation")
        .all(|r| r["selected"]["schema_version"] == 1
            && r["independent_attention"]["schema_version"] == 2));
    assert_eq!(support["family_support"][0]["fit_samples"], 8);
    assert_eq!(support["family_support"][0]["residual_samples"], 8);
    let runtime = fixture.runtime();
    let loaded = runtime
        .load_calibration_profile(&fixture.config, &cut)
        .unwrap();
    assert_eq!(loaded.receipt.schema_version, 8);
    let phases = loaded.receipt.selected_whole_wave.as_ref().unwrap();
    assert_eq!(phases.model_revision, WORK_SUPPORT_MODEL_REVISION);
    assert_eq!(phases.fit_parameters_sha256, frozen.fit_parameters_sha256);
    assert_eq!(
        phases.capture_identity_sha256,
        frozen.capture_identity_sha256
    );
    assert_eq!(
        loaded.receipt.source_observation_artifact_sha256,
        cut.source_digest
    );
    assert_eq!(
        loaded.receipt.source_measurement_protocol,
        "selected-work-support-fit-residual-v1"
    );
    let imported = loaded.snapshot.selected_import().unwrap();
    assert_eq!(
        imported.selected_family(),
        SelectedStatisticalFamily::IndependentAttentionV2
    );
    assert_eq!(
        imported.model_revision(),
        WholeWaveModelRevision::IndependentAttentionWorkSupportV1
    );

    let (ordinal, entry, _) = fixture.completed();
    assert!(ordinal > cut.accepted_ordinal);
    let heldout =
        trainer::whole_wave_observation(&entry, ordinal, frozen.capture_identity_sha256).unwrap();
    let result =
        loaded
            .snapshot
            .predict_selected_wave_identified(&heldout.exact, &heldout.selected, 21);
    let query = result.query_identity.unwrap();
    assert_eq!(query.model_revision, WORK_SUPPORT_MODEL_REVISION);
    assert_eq!(query.family_schema_version, 2);
    assert_eq!(
        query.family_signature,
        *heldout
            .selected
            .independent_attention_v2()
            .unwrap()
            .family_signature()
    );
    assert!(result.prediction.is_ok());
    let missing = ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1::from_wire_v1(
        heldout.selected.to_wire_v1(),
        &heldout.exact,
    )
    .unwrap();
    assert!(matches!(
        loaded
            .snapshot
            .predict_selected_wave(&heldout.exact, &missing, 21),
        Err(ModelUnknown::Evidence(
            StatisticalEvidenceUnknown::MissingProducer
        ))
    ));

    // Same shared startup path used by run and serve. Ordinary real terminal
    // receipts may update audit, but must not republish the fit or reset age.
    let load_clock =
        profile::read_load_clock(fixture.clock.as_ref(), &fixture.config.profile_import).unwrap();
    let serving = EngineCostRuntime::build_with_profile(
        identity(),
        fixture.clock.clone(),
        &fixture.config,
        false,
        Some(&cut.profile),
        Some(load_clock),
    )
    .unwrap();
    let before = serving.snapshot().unwrap();
    let (_, terminal, _) = fixture.completed();
    serving.sink.offer_evidence_numbered(terminal).unwrap();
    serving.consume_samples();
    assert!(Arc::ptr_eq(&before, &serving.snapshot().unwrap()));
    assert_eq!(serving.audit_snapshot().training.publish_attempts, 0);
    assert_eq!(serving.profile_receipt().unwrap().schema_version, 8);
    fixture.config.predictor = ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2;
    assert!(runtime
        .load_calibration_profile(&fixture.config, &cut)
        .is_err());
    let later = 21 + fixture.config.model.max_sample_age_ns.get();
    fixture.clock.set(later);
    assert_eq!(
        before.predict_selected_wave(&heldout.exact, &heldout.selected, later),
        Err(ModelUnknown::Stale)
    );
    assert_eq!(fs::read(&cut.source).unwrap(), source_bytes);
    assert_eq!(fs::read(&cut.profile).unwrap(), profile_bytes);
}

#[test]
fn work_support_product_loader_does_not_relabel_old_profile7() {
    let mut fixture = Fixture::new();
    fixture.config.predictor = ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2;
    let mut capture = fixture.begin();
    for _ in 0..8 {
        let (_, _, actual) = fixture.completed();
        capture.record(&actual, true).unwrap();
    }
    capture.freeze_fit(8).unwrap();
    for _ in 0..8 {
        let (_, _, actual) = fixture.completed();
        capture.record(&actual, true).unwrap();
    }
    let (cut, _) = capture.finish(16).unwrap();
    let source = fs::read(&cut.source).unwrap();
    let profile = fs::read(&cut.profile).unwrap();
    let runtime = fixture.runtime();
    assert_eq!(
        runtime
            .load_calibration_profile(&fixture.config, &cut)
            .unwrap()
            .receipt
            .schema_version,
        7
    );
    fixture.config.predictor = ferrum_types::SloCostPredictor::SelectedWorkSupportV1;
    assert!(runtime
        .load_calibration_profile(&fixture.config, &cut)
        .is_err());
    assert_eq!(fs::read(&cut.source).unwrap(), source);
    assert_eq!(fs::read(&cut.profile).unwrap(), profile);
}
