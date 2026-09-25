use super::*;
use clap::{Parser, Subcommand};
use ferrum_engine::continuous_engine::{
    RequiredFutureAuditActionV2, RequiredFutureAuditLimitsV2, RequiredFutureAuditPathV2,
    RequiredFutureAuditRowV2,
};
use std::num::NonZeroUsize;

fn declared() -> manifest::Manifest {
    let mut manifest = super::super::tests::manifest();
    manifest.validation.clear();
    let audit = AuditConfigV2 {
        triggers: vec![AuditTriggerV2 {
            case_index: 0,
            repetition: 0,
            before_wave_attempt: NonZeroU64::new(2).unwrap(),
            plan: RequiredFutureAuditPlanV2 {
                paths: vec![RequiredFutureAuditPathV2 {
                    waves: vec![vec![RequiredFutureAuditRowV2 {
                        frontier_index: 0,
                        action: RequiredFutureAuditActionV2::Decode,
                    }]],
                }],
                limits: RequiredFutureAuditLimitsV2 {
                    budget_ms: NonZeroU64::new(100).unwrap(),
                    maximum_queries: NonZeroUsize::new(16).unwrap(),
                    maximum_coordinates: NonZeroUsize::new(1024).unwrap(),
                },
            },
        }],
    };
    manifest.validation_model = manifest::ValidationSource::RequiredFutureAuditV2 {
        warmup: vec![],
        audit,
    };
    manifest
}

#[test]
fn required_future_cli_trigger_is_an_exact_declared_attempt_not_a_known_retry() {
    let manifest = declared();
    manifest.validate().unwrap();
    let config = manifest.validation_model.required_audit().unwrap();
    assert!(config.trigger(0, 0, 1).is_none());
    assert_eq!(config.trigger(0, 0, 2).unwrap().0, 0);
    assert!(config.trigger(0, 0, 3).is_none());
    assert!(config.trigger(1, 0, 2).is_none());
    let mut summary = AuditSummaryV2::new(1);
    summary.record(0, None).unwrap(); // actual API error consumes the trigger
    assert!(summary.record(0, None).is_err());
    summary.finish();
    assert!(summary.collection_completed);
    assert!(!summary.all_declared_requirements_recorded);
    assert_eq!(summary.unavailable_triggers, 1);
}

#[test]
fn required_future_cli_rejects_duplicate_outside_cohort_and_out_of_bound_plans() {
    let manifest = declared();
    let config = manifest.validation_model.required_audit().unwrap().clone();
    let mut invalid = config.clone();
    invalid.triggers.push(invalid.triggers[0].clone());
    assert!(invalid.validate(&manifest).is_err());
    let mut invalid = config.clone();
    invalid.triggers[0].case_index = 1;
    assert!(invalid.validate(&manifest).is_err());
    let mut invalid = config.clone();
    invalid.triggers[0].repetition = 1;
    assert!(invalid.validate(&manifest).is_err());
    let mut invalid = config.clone();
    invalid.triggers[0].plan.paths[0].waves[0][0].frontier_index = 2;
    assert!(invalid.validate(&manifest).is_err());
    let mut invalid = config;
    invalid.triggers[0].before_wave_attempt = NonZeroU64::new(101).unwrap();
    assert!(invalid.validate(&manifest).is_err());
    let mut wire = serde_json::to_value(&manifest).unwrap();
    wire["validation_model"]["audit"]["advance_clock_ns"] = 1.into();
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
    let mut summary = AuditSummaryV2::new(2);
    summary.record(0, None).unwrap();
    summary.finish();
    assert_eq!(summary.remaining_trigger_indices, vec![1]);
    assert!(!summary.all_declared_requirements_recorded);
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
fn required_future_cli_requires_real_loader_inputs_and_cannot_overwrite_them() {
    let Command::CalibrateSlo(cmd) = Cli::try_parse_from([
        "ferrum",
        "calibrate-slo",
        "model",
        "--manifest",
        "input.json",
        "--slo-config",
        "policy.toml",
        "--out",
        "report.json",
        "--observations",
        "raw.jsonl",
        "--startup-usage",
        "serve",
    ])
    .unwrap()
    .command;
    let manifest = declared();
    let mut policy = ferrum_types::SloConfig::default();
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    assert!(startup::validate_export_configuration(&cmd, &manifest, &policy).is_err());
    policy.cost_profile = Some("actual-seed-profile10.json".into());
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(0);
    assert!(startup::validate_export_configuration(&cmd, &manifest, &policy).is_err());
    policy.prefill_reference = Some(ferrum_types::SloPrefillReferenceConfig {
        artifact_path: "actual-reference.json".into(),
        expected_protocol_sha256: [1; 32],
        limits: Default::default(),
    });
    // Configuration only: these paths are not opened here. Real startup still
    // runs the original fingerprint/source/clock/artifact loaders.
    startup::validate_export_configuration(&cmd, &manifest, &policy).unwrap();
    policy.cost_profile = Some(cmd.observations.clone());
    assert!(startup::validate_export_configuration(&cmd, &manifest, &policy).is_err());
    policy.cost_profile = Some("actual-seed-profile10.json".into());
    policy
        .prefill_reference
        .as_mut()
        .unwrap()
        .expected_protocol_sha256 = [0; 32];
    assert!(startup::validate_export_configuration(&cmd, &manifest, &policy).is_err());
    policy
        .prefill_reference
        .as_mut()
        .unwrap()
        .expected_protocol_sha256 = [1; 32];
    policy.cost_observation = ferrum_types::SloCostObservationConfig::default();
    assert!(startup::validate_export_configuration(&cmd, &manifest, &policy).is_err());
}
