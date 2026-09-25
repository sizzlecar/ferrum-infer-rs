use super::*;
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

#[test]
fn calibration_token_policy_residency_is_explicit_and_legacy_manifest_preserves_it() {
    let original = manifest();
    let mut wire = serde_json::to_value(&original).unwrap();
    wire["training"][0]
        .as_object_mut()
        .unwrap()
        .remove("token_policy_residency");
    let legacy: manifest::Manifest = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(
        legacy.training[0].token_policy_residency,
        manifest::TokenPolicyResidencyPolicy::Preserve
    );
    wire["training"][0]["token_policy_residency"] = serde_json::json!("invalidate_before_cohort");
    let explicit: manifest::Manifest = serde_json::from_value(wire.clone()).unwrap();
    explicit.validate().unwrap();
    assert_eq!(
        explicit.training[0].token_policy_residency,
        manifest::TokenPolicyResidencyPolicy::InvalidateBeforeCohort
    );
    assert_eq!(
        explicit.validation[0].token_policy_residency,
        manifest::TokenPolicyResidencyPolicy::Preserve
    );
    assert_eq!(
        explicit.prompts[0].sampling.max_tokens,
        original.prompts[0].sampling.max_tokens
    );
    assert_eq!(explicit.training[0].prompts, original.training[0].prompts);
    wire["training"][0]["token_policy_residency"] = serde_json::json!("pretend_full_cold_start");
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
}

#[test]
fn calibration_decode_route_is_explicit_and_defaults_to_actual_execution() {
    use ferrum_engine::continuous_engine::CalibrationDecodeRoute;
    let original = manifest();
    let mut wire = serde_json::to_value(&original).unwrap();
    wire["training"][0]
        .as_object_mut()
        .unwrap()
        .remove("decode_route");
    let legacy: manifest::Manifest = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(
        legacy.training[0].decode_route,
        CalibrationDecodeRoute::Actual
    );
    wire["training"][0]["decode_route"] = serde_json::json!("full_logits");
    let full: manifest::Manifest = serde_json::from_value(wire.clone()).unwrap();
    full.validate().unwrap();
    assert_eq!(
        full.training[0].decode_route,
        CalibrationDecodeRoute::FullLogits
    );
    assert_eq!(
        full.prompts[0].sampling.max_tokens,
        original.prompts[0].sampling.max_tokens
    );
    wire["training"][0]["decode_route"] = serde_json::json!("pretend_pending_utf8");
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
}

pub(super) fn manifest() -> manifest::Manifest {
    let prompt = "A pinned, already rendered model input.";
    let case = manifest::Cohort {
        prompts: vec![0, 0],
        repetitions: NonZeroUsize::new(1).unwrap(),
        prefill_chunk_tokens: NonZeroU32::new(4).unwrap(),
        execution: manifest::Execution::Mixed,
        decode_route: ferrum_engine::continuous_engine::CalibrationDecodeRoute::Actual,
        token_policy_residency: manifest::TokenPolicyResidencyPolicy::Preserve,
        rolling_window: None,
        wave_plan: None,
    };
    manifest::Manifest {
        schema_version: 1,
        validation_model: manifest::ValidationSource::LiveFrozen,
        reference: None,
        sharegpt: None,
        input_preprocessing_sha256: [1; 32],
        protocol: manifest::Protocol {
            total_timeout_ms: NonZeroU64::new(10_000).unwrap(),
            shutdown_timeout_ms: NonZeroU64::new(1000).unwrap(),
            maximum_wave_attempts: NonZeroU64::new(100).unwrap(),
            maximum_raw_bytes: NonZeroU64::new(1024 * 1024).unwrap(),
            maximum_requests: NonZeroUsize::new(2).unwrap(),
            output: manifest::Codec::CliText,
        },
        prompts: vec![manifest::Prompt {
            source_id: "sharegpt:selected-request".into(),
            rendered_prompt: prompt.into(),
            rendered_prompt_sha256: Sha256::digest(prompt.as_bytes()).into(),
            sampling: ferrum_types::SamplingParams {
                max_tokens: 73,
                ..Default::default()
            },
        }],
        training: vec![case.clone()],
        validation: vec![case],
    }
}

#[test]
fn pinned_manifest_preserves_full_output_limit_and_explicit_cohorts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&manifest()).unwrap()).unwrap();
    let loaded = manifest::load(&path).unwrap();
    assert_eq!(loaded.prompts[0].sampling.max_tokens, 73);
    assert_eq!(loaded.training[0].prompts, vec![0, 0]);
    assert_eq!(loaded.validation[0].prompts, vec![0, 0]);
    let mut changed = loaded.clone();
    changed.prompts[0].rendered_prompt.push('!');
    assert!(changed.validate().is_err());
    let mut changed = loaded;
    changed.training[0].prompts.push(0);
    assert!(changed.validate().is_err());
}

#[test]
fn manifest_rejects_absent_validation_invalid_sampling_and_unbounded_protocol() {
    let mut value = manifest();
    value.validation.clear();
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.prompts[0].sampling.temperature = f32::NAN;
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.protocol.maximum_raw_bytes = NonZeroU64::new(u64::MAX).unwrap();
    assert!(value.validate().is_err());
    let mut value = serde_json::to_value(manifest()).unwrap();
    value["protocol"]["typo"] = true.into();
    assert!(serde_json::from_value::<manifest::Manifest>(value).is_err());
}

#[test]
fn bounded_raw_writer_rejects_before_writing_and_never_overwrites() {
    let directory = tempfile::tempdir().unwrap();
    let raw = directory.path().join("raw.jsonl");
    let report_path = directory.path().join("report.json");
    let mut value = manifest();
    value.protocol.maximum_raw_bytes = NonZeroU64::new(8).unwrap();
    let mut output = report::Artifacts::create(&raw, &report_path, &value).unwrap();
    output.record(&serde_json::json!({})).unwrap();
    assert!(output
        .record(&serde_json::json!({"too_large": true}))
        .is_err());
    output
        .finish(None, None, Some(&FerrumError::backend("test failure")))
        .unwrap();
    assert_eq!(std::fs::read_to_string(&raw).unwrap(), "{}\n");
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["status"], "failed");
    assert_eq!(report["raw_bytes"], 3);
    assert!(report::Artifacts::create(&raw, &report_path, &value).is_err());
    assert_eq!(std::fs::read_to_string(&raw).unwrap(), "{}\n");
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
fn calibration_requires_explicit_product_defaults_and_output_paths() {
    let base = [
        "ferrum",
        "calibrate-slo",
        "qwen3.5:9b",
        "--manifest",
        "inputs.json",
        "--slo-config",
        "observe.toml",
        "--out",
        "report.json",
        "--observations",
        "raw.jsonl",
    ];
    assert!(Cli::try_parse_from(base).is_err());
    let cli = Cli::try_parse_from(base.into_iter().chain(["--startup-usage", "serve"])).unwrap();
    let Command::CalibrateSlo(command) = cli.command;
    assert!(matches!(
        command.startup_usage,
        CalibrationStartupUsage::Serve
    ));
}

#[test]
fn documented_manifest_parses_and_its_pinned_bytes_match() {
    let documentation = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/slo-configuration.md"
    ));
    let section = documentation
        .split_once("<!-- calibrate-slo-manifest-v1 -->")
        .unwrap()
        .1
        .split_once("<!-- /calibrate-slo-manifest-v1 -->")
        .unwrap()
        .0;
    let json = section
        .split_once("```json\n")
        .unwrap()
        .1
        .split_once("```")
        .unwrap()
        .0;
    let manifest: manifest::Manifest = serde_json::from_str(json).unwrap();
    manifest.validate().unwrap();
    let expected: [u8; 32] = Sha256::digest(b"documentation:raw-utf8-no-template:Hello:v1").into();
    assert_eq!(manifest.input_preprocessing_sha256, expected);
    assert_eq!(manifest.prompts[0].rendered_prompt, "Hello");
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 16);
}

#[test]
fn validation_model_is_explicit_and_old_manifests_remain_live_diagnostics() {
    let mut old = serde_json::to_value(manifest()).unwrap();
    old.as_object_mut().unwrap().remove("validation_model");
    let old: manifest::Manifest = serde_json::from_value(old).unwrap();
    assert!(matches!(
        old.validation_model,
        manifest::ValidationSource::LiveFrozen
    ));
    let mut exported = serde_json::to_value(manifest()).unwrap();
    exported["validation_model"] =
        serde_json::json!({"kind":"exported_profile","profile":"cut.json","source":"cut.jsonl"});
    let parsed: manifest::Manifest = serde_json::from_value(exported.clone()).unwrap();
    parsed.validate().unwrap();
    assert!(matches!(
        parsed.validation_model,
        manifest::ValidationSource::ExportedProfile { .. }
    ));
    exported["validation_model"]["live_fallback"] = true.into();
    assert!(serde_json::from_value::<manifest::Manifest>(exported).is_err());
}

#[test]
fn exported_paths_belong_to_manifest_directory_and_aliases_fail_before_creation() {
    let directory = tempfile::tempdir().unwrap();
    let manifest_path = directory.path().join("manifest.json");
    let mut value = manifest();
    value.validation_model = manifest::ValidationSource::ExportedProfile {
        profile: "cut.json".into(),
        source: "cut.jsonl".into(),
    };
    std::fs::write(&manifest_path, serde_json::to_vec(&value).unwrap()).unwrap();
    let value = manifest::load(&manifest_path).unwrap();
    let manifest::ValidationSource::ExportedProfile { profile, source } = value.validation_model
    else {
        panic!("wrong model source")
    };
    assert_eq!(
        profile,
        directory.path().canonicalize().unwrap().join("cut.json")
    );
    assert_eq!(
        source,
        directory.path().canonicalize().unwrap().join("cut.jsonl")
    );
    paths::validate([&profile, &source]).unwrap();
    assert!(paths::validate([&profile, &directory.path().join("./cut.json")]).is_err());
    assert!(!profile.exists());
    std::fs::write(&profile, b"original").unwrap();
    assert!(paths::validate([&profile, &source]).is_err());
    assert_eq!(std::fs::read(&profile).unwrap(), b"original");
}

#[test]
fn exported_validation_requires_real_retention_clock_and_distinct_live_files() {
    let directory = tempfile::tempdir().unwrap();
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
    cmd.out = directory.path().join("report.json");
    cmd.observations = directory.path().join("raw.jsonl");
    let mut value = manifest();
    value.validation_model = manifest::ValidationSource::ExportedProfile {
        profile: directory.path().join("cut.json"),
        source: directory.path().join("cut.jsonl"),
    };
    let mut policy = ferrum_types::SloConfig::default();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation.profile_export = Some(ferrum_types::SloCostProfileExportConfig {
        path: directory.path().join("live.json"),
        observations_path: directory.path().join("live.jsonl"),
        declared_clock_max_error_ns: Some(1000),
        ..Default::default()
    });
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(1000);
    startup::validate_export_configuration(&cmd, &value, &policy).unwrap();
    policy
        .cost_observation
        .profile_export
        .as_mut()
        .unwrap()
        .path = cmd.observations.clone();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
}

#[test]
fn failed_validation_report_keeps_cut_identity_and_partial_denominators() {
    let directory = tempfile::tempdir().unwrap();
    let raw = directory.path().join("raw.jsonl");
    let report_path = directory.path().join("report.json");
    let mut artifacts = report::Artifacts::create(&raw, &report_path, &manifest()).unwrap();
    let summary = report::Summary {
        validation_known: 2,
        validation_unknown: 3,
        validation_model: Some(serde_json::json!({"kind":"exported_profile","accepted_ordinal":9})),
        ..Default::default()
    };
    artifacts
        .finish(
            None,
            Some(summary),
            Some(&FerrumError::backend("heldout output failed")),
        )
        .unwrap();
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(value["status"], "failed");
    assert_eq!(value["summary"]["validation_model"]["accepted_ordinal"], 9);
    assert_eq!(value["summary"]["validation_unknown"], 3);
}

#[test]
fn collection_and_shutdown_failures_remain_independently_visible() {
    let mut summary = report::Summary::default();
    let error = finish_outcome(
        Err(FerrumError::backend("collection failed")),
        Err(FerrumError::backend("shutdown failed")),
        &mut summary,
    )
    .unwrap();
    assert!(error
        .to_string()
        .contains("resource reconciliation is unconfirmed"));
    assert!(summary
        .collection_error
        .as_deref()
        .unwrap()
        .contains("collection failed"));
    let shutdown = summary.shutdown.as_ref().unwrap();
    assert!(!shutdown.completed);
    assert!(shutdown
        .error
        .as_deref()
        .unwrap()
        .contains("shutdown failed"));
    assert!(finish_outcome(Ok(()), Ok(()), &mut summary).is_none());
    assert!(summary.collection_error.is_none());
    assert!(summary.shutdown.as_ref().unwrap().completed);
}

#[path = "rolling_window_tests.rs"]
mod rolling_window_tests;

#[test]
fn wave_plan_wire_binds_cycles_preserves_legacy_and_rejects_unbounded_choices() {
    let legacy = manifest();
    let old = serde_json::to_value(&legacy).unwrap();
    assert!(old["training"][0].get("wave_plan").is_none());
    let roundtrip: manifest::Manifest = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(serde_json::to_value(roundtrip).unwrap(), old);
    let plan = serde_json::json!({"prefill_chunks":[16,32,64,128], "decode_routes":["actual","full_logits"]});
    let mut wire = old.clone();
    wire["training"][0]["wave_plan"] = plan.clone();
    let scheduled: manifest::Manifest = serde_json::from_value(wire.clone()).unwrap();
    scheduled.validate().unwrap();
    assert_eq!(scheduled.training[0].prompts, legacy.training[0].prompts);
    assert_eq!(
        serde_json::to_value(&scheduled.prompts).unwrap(),
        serde_json::to_value(&legacy.prompts).unwrap()
    );
    assert_ne!(
        Sha256::digest(serde_json::to_vec(&scheduled).unwrap()),
        Sha256::digest(serde_json::to_vec(&legacy).unwrap())
    );
    let mut reordered = wire.clone();
    reordered["training"][0]["wave_plan"]["decode_routes"] =
        serde_json::json!(["full_logits", "actual"]);
    let reordered: manifest::Manifest = serde_json::from_value(reordered).unwrap();
    assert_ne!(
        Sha256::digest(serde_json::to_vec(&scheduled).unwrap()),
        Sha256::digest(serde_json::to_vec(&reordered).unwrap())
    );
    for invalid in [
        serde_json::json!({}),
        serde_json::json!({"prefill_chunks":[]}),
        serde_json::json!({"decode_routes":[]}),
        serde_json::json!({"prefill_chunks":[1048577]}),
        serde_json::json!({"prefill_chunks":vec![16;17]}),
    ] {
        let mut value = wire.clone();
        value["training"][0]["wave_plan"] = invalid;
        assert!(serde_json::from_value::<manifest::Manifest>(value)
            .unwrap()
            .validate()
            .is_err());
    }
    for invalid in [
        serde_json::json!({"prefill_chunks":[0]}),
        serde_json::json!({"decode_routes":["guess_known"]}),
        serde_json::json!({"prefill_chunks":[16],"adapt_to_timing":true}),
    ] {
        let mut value = wire.clone();
        value["training"][0]["wave_plan"] = invalid;
        assert!(serde_json::from_value::<manifest::Manifest>(value).is_err());
    }
}
