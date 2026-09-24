use super::*;
use crate::continuous_engine::inner::cost_observation::tests::statistical_model::capture::Fixture;
use ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown;

#[test]
fn selected_heldout_keeps_real_receipt_session_cut_and_model_identity() {
    for predictor in [
        ferrum_types::SloCostPredictor::SelectedWholeWaveV1,
        ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2,
    ] {
        selected_heldout_query_identity_for(predictor);
    }
}

fn selected_heldout_query_identity_for(predictor: ferrum_types::SloCostPredictor) {
    let mut fixture = Fixture::new();
    fixture.config.predictor = predictor;
    let mut capture = fixture.begin();
    let mut fit_capture = None;
    for _ in 0..8 {
        let (_, _, actual) = fixture.completed();
        capture.record(&actual, true).unwrap();
        fit_capture = Some(actual);
    }
    let frozen = capture.freeze_fit(8).unwrap();
    for _ in 0..8 {
        let (_, _, actual) = fixture.completed();
        capture.record(&actual, true).unwrap();
    }
    let (cut, support) = capture.finish(16).unwrap();
    let runtime = fixture.runtime();
    let imported = runtime
        .load_calibration_profile(&fixture.config, &cut)
        .unwrap();
    let session = Arc::new(());
    let model = ImportedCalibrationModel {
        selected_session: Some(Arc::clone(&session)),
        artifact: cut.into(),
        imported,
        clock: fixture.clock.clone(),
        audit: support,
    };
    let (_, _, heldout) = fixture.completed();
    let mut evidence = SelectedCalibrationEvidence {
        session: Arc::clone(&session),
        observation: SelectedCalibrationCapture::observation(
            &heldout,
            true,
            frozen.capture_identity_sha256,
        ),
    };
    let result = model.evaluate_selected_evidence(&evidence).unwrap();
    let identity = result.query_identity.unwrap();
    let observation = evidence.observation.as_ref().unwrap();
    let expected = match predictor {
        ferrum_types::SloCostPredictor::SelectedWholeWaveV1 => {
            observation.selected.family_signature()
        }
        ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2 => observation
            .selected
            .independent_attention_v2()
            .unwrap()
            .family_signature(),
        _ => unreachable!(),
    };
    assert_eq!(identity.family_signature, *expected);
    assert_eq!(
        identity.model_revision,
        model
            .import_receipt()
            .selected_whole_wave
            .as_ref()
            .unwrap()
            .model_revision
    );
    assert_eq!(
        identity.family_schema_version,
        match predictor {
            ferrum_types::SloCostPredictor::SelectedWholeWaveV1 => 1,
            ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2 => 2,
            _ => unreachable!(),
        }
    );
    assert_eq!(
        result.prediction,
        model.imported.snapshot.predict_selected_wave(
            &observation.exact,
            &observation.selected,
            21
        )
    );
    assert_eq!(result.actual_ns, 19);
    let prediction = result.prediction.unwrap();
    assert_eq!(
        (prediction.fit_samples, prediction.residual_samples),
        (8, 8)
    );
    assert_eq!(result.underestimate_ns, Some(0));
    // Public record similarity cannot cross a different actual session.
    evidence.session = Arc::new(());
    assert_eq!(
        model.evaluate_selected_evidence(&evidence),
        Err(ModelUnknown::WrongSource)
    );
    evidence.session = session;
    evidence.observation = SelectedCalibrationCapture::observation(
        fit_capture.as_ref().unwrap(),
        true,
        frozen.capture_identity_sha256,
    );
    assert_eq!(
        model.evaluate_selected_evidence(&evidence),
        Err(ModelUnknown::PhaseLeakage)
    );
    evidence.observation =
        SelectedCalibrationCapture::observation(&heldout, false, frozen.capture_identity_sha256);
    assert_eq!(
        model.evaluate_selected_evidence(&evidence),
        Err(ModelUnknown::InvalidSample)
    );
    evidence.observation = SelectedCalibrationCapture::observation(&heldout, true, [99; 32]);
    assert_eq!(
        model.evaluate_selected_evidence(&evidence),
        Err(ModelUnknown::WrongSource)
    );
    assert_eq!(
        model.import_receipt().recorded_samples,
        16,
        "heldout does not retrain the imported model"
    );
}
