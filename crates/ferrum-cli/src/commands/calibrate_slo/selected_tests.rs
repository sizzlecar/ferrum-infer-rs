use super::*;
use clap::{Parser, Subcommand};
use ferrum_types::{SloCostObservationConfig, SloCostProfileExportConfig};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    CalibrateSlo(CalibrateSloCommand),
}

fn selected_manifest() -> manifest::Manifest {
    let mut value = tests::manifest();
    value.validation_model = manifest::ValidationSource::SelectedWholeWaveV1 {
        export: SloCostProfileExportConfig {
            path: "selected.json".into(),
            observations_path: "selected-source.jsonl".into(),
            declared_clock_max_error_ns: Some(0),
            ..Default::default()
        },
        residual: value.training.clone(),
    };
    value
}

#[test]
fn independent_attention_v2_manifest_requires_matching_product_version_and_residual() {
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
    let mut manifest = selected_manifest();
    let original_budget = manifest.prompts[0].sampling.max_tokens;
    let manifest::ValidationSource::SelectedWholeWaveV1 {
        mut export,
        residual,
    } = manifest.validation_model
    else {
        unreachable!()
    };
    export.path = directory.path().join("profile7.json");
    export.observations_path = directory.path().join("source2.jsonl");
    manifest.validation_model =
        manifest::ValidationSource::SelectedIndependentAttentionV2 { export, residual };
    manifest.validate().unwrap();
    let bytes = serde_json::to_vec(&manifest).unwrap();
    let decoded: manifest::Manifest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        decoded.validation_model.selected_kind(),
        Some("selected_independent_attention_v2")
    );
    assert_eq!(decoded.prompts[0].sampling.max_tokens, original_budget);
    let mut policy = ferrum_types::SloConfig::default();
    policy.cost_observation = SloCostObservationConfig::selected_whole_wave_v1();
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(0);
    assert!(startup::validate_export_configuration(&cmd, &decoded, &policy).is_err());
    policy.cost_observation.predictor =
        ferrum_types::SloCostPredictor::SelectedIndependentAttentionV2;
    policy.validate().unwrap();
    startup::validate_export_configuration(&cmd, &decoded, &policy).unwrap();
    let mut low = policy.clone();
    low.cost_observation.model.min_samples = std::num::NonZeroUsize::new(7).unwrap();
    assert!(low.validate().is_err());
    let manifest::ValidationSource::SelectedIndependentAttentionV2 { residual, .. } =
        &mut manifest.validation_model
    else {
        unreachable!()
    };
    residual.clear();
    assert!(manifest.validate().is_err());
}

#[test]
fn selected_calibration_manifest_preserves_output_and_resolves_both_artifacts() {
    let value = selected_manifest();
    let maximum = value.prompts[0].sampling.max_tokens;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let loaded = manifest::load(&path).unwrap();
    assert_eq!(loaded.prompts[0].sampling.max_tokens, maximum);
    let (profile, source) = loaded.validation_model.destinations().unwrap();
    let base = directory.path().canonicalize().unwrap();
    assert_eq!(profile, base.join("selected.json"));
    assert_eq!(source, base.join("selected-source.jsonl"));
    assert_eq!(
        loaded.validation_model.residual()[0].prompts,
        value.training[0].prompts
    );
    let roundtrip: manifest::Manifest =
        serde_json::from_value(serde_json::to_value(loaded).unwrap()).unwrap();
    roundtrip.validate().unwrap();
}

#[test]
fn selected_calibration_requires_independent_residual_cohorts_with_valid_inputs() {
    let mut value = selected_manifest();
    value.validate().unwrap();
    let manifest::ValidationSource::SelectedWholeWaveV1 { residual, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    residual[0].prompts.push(value.prompts.len());
    assert!(value.validate().is_err());
    let manifest::ValidationSource::SelectedWholeWaveV1 { residual, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    residual.clear();
    assert!(value.validate().is_err());
}

#[test]
fn selected_residual_owners_count_toward_the_shared_capture_boundary() {
    let mut value = selected_manifest();
    value.protocol.maximum_requests = std::num::NonZeroUsize::new(256).unwrap();
    let manifest::ValidationSource::SelectedWholeWaveV1 { residual, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    let mut case = residual[0].clone();
    case.prompts = vec![0; 256];
    case.repetitions = std::num::NonZeroUsize::new(64).unwrap();
    *residual = vec![case; 4];
    // Residual alone fills the public limit; fit and heldout are additional
    // fresh owners and must not disappear from the resource accounting.
    assert!(value
        .validate()
        .unwrap_err()
        .to_string()
        .contains("fresh owners"));
}

#[test]
fn selected_calibration_startup_rejects_legacy_predictor_clock_and_path_aliases() {
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
    cmd.observations = directory.path().join("raw.jsonl");
    cmd.out = directory.path().join("report.json");
    let mut value = selected_manifest();
    let manifest::ValidationSource::SelectedWholeWaveV1 { export, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    export.path = directory.path().join("selected.json");
    export.observations_path = directory.path().join("source.jsonl");
    let mut policy = ferrum_types::SloConfig::default();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation = SloCostObservationConfig::selected_whole_wave_v1();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(0);
    startup::validate_export_configuration(&cmd, &value, &policy).unwrap();
    let manifest::ValidationSource::SelectedWholeWaveV1 { export, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    export.path = cmd.observations.clone();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    value.validation_model = manifest::ValidationSource::LiveFrozen;
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
}
