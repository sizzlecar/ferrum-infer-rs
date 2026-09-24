use super::*;

#[test]
fn work_support_manifest_requires_its_product_version_and_full_residual_inputs() {
    let Command::CalibrateSlo(mut cmd) = Cli::try_parse_from([
        "ferrum",
        "calibrate-slo",
        "model",
        "--manifest",
        "inputs.json",
        "--slo-config",
        "slo.toml",
        "--out",
        "report.json",
        "--observations",
        "raw.jsonl",
        "--startup-usage",
        "serve",
    ])
    .unwrap()
    .command;
    let directory = tempfile::tempdir().unwrap();
    cmd.out = directory.path().join("report.json");
    cmd.observations = directory.path().join("raw.jsonl");
    let mut value = selected_manifest();
    let budget = value.prompts[0].sampling.max_tokens;
    let manifest::ValidationSource::SelectedWholeWaveV1 {
        mut export,
        residual,
    } = value.validation_model
    else {
        unreachable!()
    };
    export.path = directory.path().join("profile8.json");
    export.observations_path = directory.path().join("source3.jsonl");
    value.validation_model = manifest::ValidationSource::SelectedWorkSupportV1 { export, residual };
    value.validate().unwrap();
    let path = directory.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let loaded = manifest::load(&path).unwrap();
    assert_eq!(
        loaded.validation_model.selected_kind(),
        Some("selected_work_support_v1")
    );
    assert_eq!(loaded.prompts[0].sampling.max_tokens, budget);
    assert_eq!(
        loaded.validation_model.residual()[0].prompts,
        value.training[0].prompts
    );
    let mut policy = ferrum_types::SloConfig::default();
    policy.cost_observation = SloCostObservationConfig::selected_work_support_v1();
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(0);
    policy.validate().unwrap();
    startup::validate_export_configuration(&cmd, &loaded, &policy).unwrap();
    for predictor in [
        ferrum_types::SloCostPredictor::SelectedWholeWaveV1,
        ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2,
        ferrum_types::SloCostPredictor::LegacyFeatureModel,
    ] {
        let mut wrong = policy.clone();
        wrong.cost_observation.predictor = predictor;
        assert!(startup::validate_export_configuration(&cmd, &loaded, &wrong).is_err());
    }
    let mut no_clock = policy;
    no_clock
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = None;
    assert!(startup::validate_export_configuration(&cmd, &loaded, &no_clock).is_err());
    let manifest::ValidationSource::SelectedWorkSupportV1 { residual, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    residual.clear();
    assert!(value.validate().is_err());
}
