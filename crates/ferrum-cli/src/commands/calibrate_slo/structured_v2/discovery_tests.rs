use super::*;
use std::num::{NonZeroU32, NonZeroUsize};

fn discovery_manifest() -> manifest::Manifest {
    let mut value = super::super::tests::manifest();
    value.validation_model = manifest::ValidationSource::StructuredDiscoveryV2 {
        warmup: value.training.clone(),
    };
    value.validation.clear();
    value
}
fn warmup(value: &mut manifest::Manifest) -> &mut Vec<manifest::Cohort> {
    match &mut value.validation_model {
        manifest::ValidationSource::StructuredDiscoveryV2 { warmup } => warmup,
        _ => unreachable!(),
    }
}
#[test]
fn discovery_manifest_has_no_model_destination_and_preserves_complete_budgets() {
    let value = discovery_manifest();
    value.validate().unwrap();
    let bytes = serde_json::to_vec(&value).unwrap();
    let parsed: manifest::Manifest = serde_json::from_slice(&bytes).unwrap();
    parsed.validate().unwrap();
    assert!(parsed.validation_model.destinations().is_none());
    assert!(!parsed.validation_model.is_structured());
    assert_eq!(parsed.prompts[0].sampling.max_tokens, 73);
    assert_eq!(parsed.validation_model.warmup_v2()[0].prompts, vec![0, 0]);
    assert_eq!(parsed.cohorts().count(), 2);
    let mut invalid = parsed.clone();
    invalid.validation = invalid.training.clone();
    assert!(invalid.validate().is_err());
    invalid.validation.clear();
    invalid.training.clear();
    assert!(invalid.validate().is_err());
    // Discovery cannot accidentally invoke the reference request-budget policy.
    invalid = parsed.clone();
    invalid.reference = Some(reference::ReferenceConfig {
        graph_routes: Default::default(),
        request_policy: Default::default(),
        revision: std::num::NonZeroU64::MIN,
        artifact_path: "reference.json".into(),
        frozen_plan_path: "plan.json".into(),
        warmup: Vec::new(),
        curve_prompt_indices: vec![0],
        granule_tokens: NonZeroU32::MIN,
        piecewise: None,
        repetitions: NonZeroUsize::MIN,
        decode_unit: reference::DecodeUnit {
            prompt_index: 0,
            generated_before: NonZeroU32::MIN,
        },
        limits: Default::default(),
    });
    assert!(invalid
        .validate()
        .unwrap_err()
        .to_string()
        .contains("without reference"));
}
#[test]
fn discovery_warmup_uses_shared_arrival_input_waveplan_and_total_owner_limits() {
    let mut value = discovery_manifest();
    warmup(&mut value)[0].prompts[0] = 1;
    assert!(value.validate().is_err());
    warmup(&mut value)[0].prompts[0] = 0;
    warmup(&mut value)[0].rolling_window = Some(manifest::RollingWindow {
        maximum_in_flight: NonZeroUsize::new(3).unwrap(),
    });
    assert!(value.validate().is_err());
    warmup(&mut value)[0].rolling_window = None;
    warmup(&mut value)[0].wave_plan = Some(manifest::WavePlan {
        prefill_chunks: Some(vec![]),
        decode_routes: None,
    });
    assert!(value.validate().is_err());
    warmup(&mut value)[0].wave_plan = None;
    value.validate().unwrap();
    let mut case = warmup(&mut value)[0].clone();
    case.prompts = vec![0; 256];
    case.repetitions = NonZeroUsize::new(64).unwrap();
    value.protocol.maximum_requests = NonZeroUsize::new(256).unwrap();
    *warmup(&mut value) = vec![case; 4];
    assert!(value
        .validate()
        .unwrap_err()
        .to_string()
        .contains("fresh owners"));
}
#[test]
fn discovery_startup_requires_matching_fresh_capture() {
    use clap::{Parser, Subcommand};
    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }
    #[derive(Subcommand)]
    enum Command {
        CalibrateSlo(CalibrateSloCommand),
    }
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
    let value = discovery_manifest();
    let mut policy = ferrum_types::SloConfig::default();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    startup::validate_export_configuration(&cmd, &value, &policy).unwrap();
    policy.cost_profile = Some("child.json".into());
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_profile = None;
    policy.cost_observation.profile_export =
        Some(ferrum_types::SloCostProfileExportConfig::default());
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation.profile_export = None;
    policy.cost_observation.structured_capture = ferrum_types::SloStructuredCostCapture::Disabled;
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
}
