use super::*;
use clap::{Parser, Subcommand};
use ferrum_engine::continuous_engine::StructuredCapturePhase;
use std::num::{NonZeroU64, NonZeroUsize};

fn manifest() -> manifest::Manifest {
    let mut value = super::super::tests::manifest();
    value.protocol.maximum_wave_attempts = NonZeroU64::new(512).unwrap();
    value.validation_model = manifest::ValidationSource::StructuredWholeWaveV1 {
        capture: CaptureConfig {
            profile: "structured-profile9.json".into(),
            source: "structured-source.jsonl".into(),
            rows: NonZeroUsize::new(2).unwrap(),
            domain_signature: [37; 32],
            settings: config::Settings::default(),
            fit_members: NonZeroUsize::new(16).unwrap(),
            residual_members: NonZeroUsize::new(12).unwrap(),
            qualification_members: NonZeroUsize::new(12).unwrap(),
            maximum_offered_waves: NonZeroUsize::new(512).unwrap(),
            maximum_file_bytes: NonZeroU64::new(8 * 1024 * 1024).unwrap(),
            declared_source_clock_error_ns: 100,
        },
        residual: value.training.clone(),
    };
    value
}
fn capture_mut(value: &mut manifest::Manifest) -> &mut CaptureConfig {
    match &mut value.validation_model {
        manifest::ValidationSource::StructuredWholeWaveV1 { capture, .. } => capture,
        _ => unreachable!(),
    }
}

#[test]
fn structured_manifest_freezes_expanded_settings_and_original_request_policy() {
    let value = manifest();
    value.validate().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest.json");
    let mut wire = serde_json::to_value(&value).unwrap();
    // Omitted individual settings use the core defaults, then become explicit
    // before protocol hashing. There is no hidden legacy-model knob.
    wire["validation_model"]["capture"]["settings"] = serde_json::json!({});
    std::fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    let loaded = manifest::load(&path).unwrap();
    let capture = loaded.validation_model.structured().unwrap();
    let options = capture.options(&loaded).unwrap();
    assert_eq!(options.scope.domain_signature, [37; 32]);
    assert_eq!(options.settings.min_phase_samples, 8);
    assert_eq!(options.fit_members.get(), 16);
    assert_eq!(options.qualification_members.get(), 12);
    assert_eq!(
        serde_json::to_value(&loaded.prompts[0].sampling).unwrap(),
        serde_json::to_value(&value.prompts[0].sampling).unwrap(),
    );
    assert_eq!(loaded.training[0].prompts, value.training[0].prompts);
    assert_eq!(loaded.validation[0].prompts, value.validation[0].prompts);
    let base = directory.path().canonicalize().unwrap();
    let (profile, source) = loaded.validation_model.destinations().unwrap();
    assert_eq!(profile, base.join("structured-profile9.json"));
    assert_eq!(source, base.join("structured-source.jsonl"));
    assert_eq!(loaded.cohorts().count(), 3);
    let mut changed = loaded.clone();
    capture_mut(&mut changed).qualification_members = NonZeroUsize::new(13).unwrap();
    assert_ne!(
        options.protocol_sha256,
        changed
            .validation_model
            .structured()
            .unwrap()
            .options(&changed)
            .unwrap()
            .protocol_sha256,
    );
    wire["validation_model"]["capture"]["settings"]["relax_missing_positions"] =
        serde_json::json!(true);
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
}

#[test]
fn structured_manifest_rejects_unfrozen_or_incomplete_populations_before_engine_creation() {
    let mut value = manifest();
    capture_mut(&mut value).domain_signature = [0; 32];
    assert!(value.validate().is_err());
    let mut value = manifest();
    capture_mut(&mut value).fit_members = NonZeroUsize::new(7).unwrap();
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.protocol.maximum_requests = NonZeroUsize::new(8).unwrap();
    capture_mut(&mut value).rows = NonZeroUsize::new(8).unwrap();
    capture_mut(&mut value).qualification_members = NonZeroUsize::new(8).unwrap();
    assert!(value.validate().is_err());
    let mut value = manifest();
    capture_mut(&mut value).maximum_offered_waves = NonZeroUsize::new(39).unwrap();
    assert!(value.validate().is_err());
    let mut value = manifest();
    if let manifest::ValidationSource::StructuredWholeWaveV1 { residual, .. } =
        &mut value.validation_model
    {
        residual.clear();
    }
    assert!(value.validate().is_err());
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    CalibrateSlo(CalibrateSloCommand),
}

#[test]
fn structured_protocol_requires_matching_fresh_product_mode_and_export_capacity() {
    let Command::CalibrateSlo(cmd) = Cli::try_parse_from([
        "ferrum",
        "calibrate-slo",
        "model",
        "--manifest",
        "manifest.json",
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
    let value = manifest();
    let mut policy = ferrum_types::SloConfig::default();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v1();
    startup::validate_export_configuration(&cmd, &value, &policy).unwrap();
    assert!(startup::validate_export_configuration(
        &cmd,
        &super::super::tests::manifest(),
        &policy
    )
    .is_err());
    policy.cost_profile = Some("preexisting-profile.json".into());
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_profile = None;
    policy.cost_observation.structured_capture = ferrum_types::SloStructuredCostCapture::Disabled;
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation.structured_capture =
        ferrum_types::SloStructuredCostCapture::HostSettledV1;
    policy.cost_observation.profile_import.max_samples = NonZeroUsize::new(39).unwrap();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation.profile_import.max_samples = NonZeroUsize::new(40).unwrap();
    policy.cost_observation.profile_import.max_file_bytes =
        NonZeroUsize::new(8 * 1024 * 1024).unwrap();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
}

#[test]
fn structured_failed_or_incomplete_source_never_authorizes_export() {
    // This is only a report/export guard fixture. It cannot mint the private
    // live model; successful schema-9 publication additionally replays source.
    let mut source = report::SourceReceipt {
        source_path: "source.jsonl".into(),
        source_sha256: [1; 32],
        source_bytes: 1,
        phase: StructuredCapturePhase::Qualified,
        offered_waves: 40,
        scope_members: 40,
        scope_failures: 0,
        failure: None,
        numerical_model_present: true,
    };
    report::require_exportable(&source, true).unwrap();
    assert!(report::require_exportable(&source, false).is_err());
    source.numerical_model_present = false;
    assert!(report::require_exportable(&source, true).is_err());
    source.numerical_model_present = true;
    source.scope_failures = 1;
    assert!(report::require_exportable(&source, true).is_err());
    source.scope_failures = 0;
    source.failure = Some("reserved member lacks original settlement".into());
    assert!(report::require_exportable(&source, true).is_err());
    source.failure = None;
    for phase in [
        StructuredCapturePhase::Fit,
        StructuredCapturePhase::Residual,
        StructuredCapturePhase::Qualification,
        StructuredCapturePhase::Failed,
    ] {
        source.phase = phase;
        assert!(report::require_exportable(&source, true).is_err());
    }
}

#[test]
fn structured_qualification_is_not_reported_as_a_fourth_heldout() {
    let mut summary = super::super::report::Summary::default();
    let old = serde_json::to_value(&summary).unwrap();
    assert!(old.get("structured_calibration").is_none());
    assert!(old["phases"].get("qualification").is_none());
    summary.structured_calibration = Some(StructuredReport::new());
    summary.request(super::super::report::Phase::Qualification);
    let wire = serde_json::to_value(&summary).unwrap();
    assert_eq!(wire["phases"]["qualification"]["requests"], 1);
    assert_eq!(summary.validation_requests, 0);
    assert_eq!(summary.validation_known, 0);
    assert_eq!(wire["phases"]["heldout"]["requests"], 0);
    assert_eq!(
        wire["structured_calibration"]["exported_profile"],
        serde_json::Value::Null
    );
}
