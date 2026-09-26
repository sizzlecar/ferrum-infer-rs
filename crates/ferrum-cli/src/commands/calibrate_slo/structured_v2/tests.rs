use super::*;
use clap::{Parser, Subcommand};
use ferrum_engine::continuous_engine::StructuredCapturePhase;
use ferrum_interfaces::execution_cost::{CoreReadbackRoute, HostPendingConstraintV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{windows::*, *};
use std::num::{NonZeroU64, NonZeroUsize};

#[test]
fn prepared_projection_single_v2_budget_binds_manifest_without_changing_cohorts() {
    use ferrum_engine::continuous_engine::{
        CalibrationLimits, StructuredPreparedProjectionBudgetV2,
    };
    let mut value = manifest();
    let original = serde_json::to_value(&value).unwrap();
    let plan = config::cohort_plan(&value, |_, _| Ok(73)).unwrap();
    value.protocol.structured_prepared_projection_budget_us =
        Some(StructuredPreparedProjectionBudgetV2::new(1_000_000).unwrap());
    value.validate().unwrap();
    let explicit = serde_json::to_value(&value).unwrap();
    assert_ne!(
        plan.signature(&original).unwrap(),
        plan.signature(&explicit).unwrap()
    );
    assert_eq!(original["training"], explicit["training"]);
    assert_eq!(original["validation"], explicit["validation"]);
    assert_eq!(original["prompts"], explicit["prompts"]);
    let loaded: manifest::Manifest = serde_json::from_value(explicit).unwrap();
    let limits = CalibrationLimits::new(loaded.protocol.maximum_requests)
        .unwrap()
        .with_structured_prepared_projection_budget(
            loaded.protocol.structured_prepared_projection_budget_us,
        );
    assert_eq!(
        limits
            .structured_prepared_projection_budget()
            .unwrap()
            .microseconds(),
        1_000_000
    );
}
fn manifest() -> manifest::Manifest {
    let mut value = super::super::tests::manifest();
    value.protocol.maximum_wave_attempts = NonZeroU64::new(512).unwrap();
    let owner = StructuredOwnerKeyV2 {
        rows: 2,
        role: StructuredWaveRoleV2::OrdinaryDecode,
        product: StructuredProductV2::GreedyToken,
        readback: CoreReadbackRoute::HostSynchronized,
        provider_template: StructuredTemplateV2::Ordered([3; 32]),
        algorithm_domain: [4; 32],
        installed_policy: [5; 32],
    };
    let scope = StructuredScopeV2 {
        owner: owner.clone(),
        coverage: StructuredCoverageV2 {
            pending_eligible_positions: vec![],
            authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
            pending_counts: vec![0],
            length_counts: vec![0, 1, 2],
            pending_positions: vec![],
            length_positions: vec![0, 1],
            joint_counts: vec![(0, 0), (0, 1), (0, 2)],
        },
    };
    let row = RowWindowV2 {
        generated_before: ClosedRangeV2 {
            minimum: 1,
            maximum: 3,
        },
        remaining_output: ClosedRangeV2::ALL,
        context_before: ClosedRangeV2::ALL,
        work: WorkWindowV2::Decode {
            kv_tokens: ClosedRangeV2::ALL,
        },
    };
    value.validation_model = manifest::ValidationSource::StructuredWholeWaveV2 {
        capture: CaptureConfigV2 {
            warmup: Vec::new(),
            profile: "profile10.json".into(),
            source: "source3.jsonl".into(),
            scope,
            membership_rule: MembershipRuleV2 {
                owner,
                windows: vec![FrontierWindowV2 {
                    rows: vec![row.clone(), row],
                }],
            },
            settings: structured::Settings::default(),
            phase_members: [16, 12, 12],
            maximum_offered_waves: NonZeroUsize::new(512).unwrap(),
            maximum_file_bytes: NonZeroU64::new(8 * 1024 * 1024).unwrap(),
            declared_source_clock_error_ns: 100,
        },
        residual: value.training.clone(),
    };
    value
}
#[test]
fn structured_v2_manifest_plan_keeps_repetitions_duplicates_and_full_budgets() {
    let mut value = manifest();
    value.training[0].repetitions = NonZeroUsize::new(2).unwrap();
    value.validate().unwrap();
    let mut calls = 0;
    let plan = config::cohort_plan(&value, |prompt, phase| {
        assert_eq!(prompt, 0);
        calls += 1;
        Ok(match phase {
            super::super::report::Phase::Training => 73,
            super::super::report::Phase::Residual => 74,
            super::super::report::Phase::Qualification => 75,
            _ => panic!("wrong phase"),
        })
    })
    .unwrap();
    assert_eq!(calls, 8);
    assert_eq!(plan.phases.map(|p| p.len()), [2, 1, 1]);
    // Regenerate after consuming the previous plan's phase array above.
    let plan = config::cohort_plan(&value, |_, _| Ok(73)).unwrap();
    assert_eq!(plan.phases[0][1].repetition, 1);
    assert_eq!(plan.phases[0][0].requests.len(), 2);
    assert_eq!(plan.phases[0][0].requests[0].maximum_output, 73);
    assert_eq!(plan.phases[0][0].requests[1].manifest_prompt, 0);
    let wire = serde_json::to_value(&value).unwrap();
    let loaded: manifest::Manifest = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(loaded.prompts[0].sampling.max_tokens, 73);
    let mut changed = plan.clone();
    changed.phases[0][1].requests[0].maximum_output = 72;
    assert_ne!(
        plan.signature(&wire).unwrap(),
        changed.signature(&wire).unwrap()
    );
    assert!(config::cohort_plan(&value, |_, _| Ok(0)).is_err());
}
#[test]
fn structured_v2_manifest_rejects_owner_mismatch_and_undeclared_result_filters() {
    let value = manifest();
    value.validate().unwrap();
    let mut wire = serde_json::to_value(&value).unwrap();
    wire["validation_model"]["capture"]["membership_rule"]["windows"][0]["observed_pending"] =
        true.into();
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
    let mut changed = value.clone();
    if let manifest::ValidationSource::StructuredWholeWaveV2 { capture, .. } =
        &mut changed.validation_model
    {
        capture.membership_rule.owner.algorithm_domain = [9; 32];
    }
    assert!(changed.validate().is_err());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let loaded = manifest::load(&path).unwrap();
    let (profile, source) = loaded.validation_model.destinations().unwrap();
    assert_eq!(
        profile,
        directory
            .path()
            .canonicalize()
            .unwrap()
            .join("profile10.json")
    );
    assert_eq!(
        source,
        directory
            .path()
            .canonicalize()
            .unwrap()
            .join("source3.jsonl")
    );
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
fn structured_v2_startup_requires_exact_version_and_fresh_capture() {
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
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v1();
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    startup::validate_export_configuration(&cmd, &value, &policy).unwrap();
    assert!(startup::validate_export_configuration(
        &cmd,
        &super::super::tests::manifest(),
        &policy
    )
    .is_err());
    policy.cost_profile = Some("already-loaded.json".into());
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
    policy.cost_profile = None;
    policy.cost_observation.structured_capture = ferrum_types::SloStructuredCostCapture::Disabled;
    assert!(startup::validate_export_configuration(&cmd, &value, &policy).is_err());
}
#[test]
fn structured_v2_failed_source_cannot_authorize_export() {
    let mut source = report::SourceReceiptV2 {
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
    source.scope_failures = 1;
    assert!(report::require_exportable(&source, true).is_err());
    source.scope_failures = 0;
    source.numerical_model_present = false;
    assert!(report::require_exportable(&source, true).is_err());
    source.numerical_model_present = true;
    source.phase = StructuredCapturePhase::Fit;
    assert!(report::require_exportable(&source, true).is_err());
    // A report fixture has no live/private receipt; success here still requires
    // the independent source3 replay inside the actual profile10 exporter.
}

#[test]
fn structured_v2_warmup_is_hashed_but_never_enters_three_phase_slots() {
    let mut value = manifest();
    let old_wire = serde_json::to_value(&value).unwrap();
    assert!(old_wire["validation_model"]["capture"]
        .get("warmup")
        .is_none());
    let loaded: manifest::Manifest = serde_json::from_value(old_wire).unwrap();
    assert!(loaded.validation_model.warmup_v2().is_empty());
    let before = config::cohort_plan(&value, |_, _| Ok(73)).unwrap();
    let before_hash = before
        .signature(&serde_json::to_value(&value).unwrap())
        .unwrap();
    let mut warmup = value.training[0].clone();
    warmup.repetitions = NonZeroUsize::new(3).unwrap();
    let manifest::ValidationSource::StructuredWholeWaveV2 { capture, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    capture.warmup.push(warmup);
    value.validate().unwrap();
    let after = config::cohort_plan(&value, |_, phase| {
        assert!(matches!(
            phase,
            super::super::report::Phase::Training
                | super::super::report::Phase::Residual
                | super::super::report::Phase::Qualification
        ));
        Ok(73)
    })
    .unwrap();
    assert_eq!(
        serde_json::to_value(&before).unwrap(),
        serde_json::to_value(&after).unwrap()
    );
    assert_ne!(
        before_hash,
        after
            .signature(&serde_json::to_value(&value).unwrap())
            .unwrap()
    );
    let manifest::ValidationSource::StructuredWholeWaveV2 { capture, .. } =
        &mut value.validation_model
    else {
        unreachable!()
    };
    capture.warmup[0].prompts[0] = 1;
    assert!(value.validate().is_err());
}
