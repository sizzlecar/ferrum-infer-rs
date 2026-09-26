use super::*;
use ferrum_interfaces::execution_cost::{CoreReadbackRoute, HostPendingConstraintV2};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{windows::*, *};
use std::num::{NonZeroU64, NonZeroUsize};

#[test]
fn prepared_projection_group_v2_uses_one_bound_for_shared_population() {
    use ferrum_engine::continuous_engine::StructuredPreparedProjectionBudgetV2;
    let mut value = manifest();
    let original = serde_json::to_value(&value).unwrap();
    value.protocol.structured_prepared_projection_budget_us =
        Some(StructuredPreparedProjectionBudgetV2::new(1_000_000).unwrap());
    value.validate().unwrap();
    let explicit = serde_json::to_value(&value).unwrap();
    let plan = super::super::config::cohort_plan(&value, |_, _| Ok(73)).unwrap();
    assert_ne!(
        plan.signature(&original).unwrap(),
        plan.signature(&explicit).unwrap()
    );
    assert_eq!(original["validation_model"], explicit["validation_model"]);
    assert_eq!(original["training"], explicit["training"]);
    assert_eq!(original["validation"], explicit["validation"]);
    let loaded: manifest::Manifest = serde_json::from_value(explicit).unwrap();
    assert_eq!(
        loaded
            .protocol
            .structured_prepared_projection_budget_us
            .unwrap()
            .microseconds(),
        1_000_000
    );
}

fn manifest() -> manifest::Manifest {
    let mut m = super::super::super::tests::manifest();
    m.protocol.maximum_wave_attempts = NonZeroU64::new(512).unwrap();
    let children = (0..2)
        .map(|index| {
            let owner = StructuredOwnerKeyV2 {
                rows: 2,
                role: StructuredWaveRoleV2::OrdinaryDecode,
                product: StructuredProductV2::GreedyToken,
                readback: CoreReadbackRoute::HostSynchronized,
                provider_template: StructuredTemplateV2::Ordered([index + 1; 32]),
                algorithm_domain: [3; 32],
                installed_policy: [4; 32],
            };
            CaptureConfigV2 {
                warmup: vec![],
                profile: format!("child{index}.json").into(),
                source: format!("child{index}.jsonl").into(),
                scope: StructuredScopeV2 {
                    owner: owner.clone(),
                    coverage: StructuredCoverageV2 {
                        pending_eligible_positions: vec![],
                        authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
                        pending_counts: vec![0],
                        length_counts: vec![0],
                        pending_positions: vec![],
                        length_positions: vec![],
                        joint_counts: vec![(0, 0)],
                    },
                },
                membership_rule: MembershipRuleV2 {
                    owner,
                    windows: vec![FrontierWindowV2 {
                        rows: vec![
                            RowWindowV2 {
                                generated_before: ClosedRangeV2 {
                                    minimum: 1,
                                    maximum: 3
                                },
                                remaining_output: ClosedRangeV2::ALL,
                                context_before: ClosedRangeV2::ALL,
                                work: WorkWindowV2::Decode {
                                    kv_tokens: ClosedRangeV2::ALL
                                }
                            };
                            2
                        ],
                    }],
                },
                settings: structured::Settings::default(),
                phase_members: [8; 3],
                maximum_offered_waves: NonZeroUsize::new(512).unwrap(),
                maximum_file_bytes: NonZeroU64::new(8 * 1024 * 1024).unwrap(),
                declared_source_clock_error_ns: 100,
            }
        })
        .collect();
    m.validation_model = manifest::ValidationSource::StructuredWholeWaveGroupV2 {
        capture: GroupCaptureConfigV2 {
            shared_source: None,
            warmup: vec![m.training[0].clone()],
            catalog: "catalog.json".into(),
            children,
            limits: config::GroupLimitsV2::default(),
        },
        residual: m.training.clone(),
    };
    m
}
fn capture(m: &mut manifest::Manifest) -> &mut GroupCaptureConfigV2 {
    let manifest::ValidationSource::StructuredWholeWaveGroupV2 { capture, .. } =
        &mut m.validation_model
    else {
        panic!("group")
    };
    capture
}
#[test]
fn structured_group_cli_shared_plan_and_all_destinations_survive_roundtrip() {
    let m = manifest();
    m.validate().unwrap();
    let plan = super::super::config::cohort_plan(&m, |_, _| Ok(73)).unwrap();
    assert_eq!(plan.phases.each_ref().map(|p| p.len()), [1; 3]);
    assert_eq!(plan.phases[0][0].requests.len(), 2);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&m).unwrap()).unwrap();
    let loaded = manifest::load(&path).unwrap();
    let c = loaded.validation_model.structured_group_v2().unwrap();
    assert_eq!(
        c.catalog,
        directory
            .path()
            .canonicalize()
            .unwrap()
            .join("catalog.json")
    );
    for (i, child) in c.children.iter().enumerate() {
        assert_eq!(
            child.source,
            directory
                .path()
                .canonicalize()
                .unwrap()
                .join(format!("child{i}.jsonl"))
        );
    }
    assert_eq!(loaded.validation_model.warmup_v2().len(), 1);
    assert!(loaded.validation_model.structured_v2().is_none());
    assert!(loaded.validation_model.is_structured());
    let mut wire = serde_json::to_value(&m).unwrap();
    wire["validation_model"]["capture"]["children"][0]["filter_known_only"] = true.into();
    assert!(serde_json::from_value::<manifest::Manifest>(wire).is_err());
}
#[test]
fn structured_group_cli_rejects_population_alias_and_aggregate_bounds() {
    let mut m = manifest();
    let original = m.clone();
    let duplicate = capture(&mut m).children[0].scope.owner.clone();
    capture(&mut m).children[1].scope.owner = duplicate;
    // Keep the rule paired so rejection is specifically duplicate ownership.
    let c = capture(&mut m);
    c.children[1].membership_rule.owner = c.children[1].scope.owner.clone();
    assert!(m.validate().is_err());
    let mut m = original.clone();
    let warmup = m.training[0].clone();
    capture(&mut m).children[0].warmup.push(warmup);
    assert!(m.validate().is_err());
    let mut m = original.clone();
    capture(&mut m).limits.maximum_total_file_bytes = NonZeroU64::new(15 * 1024 * 1024).unwrap();
    assert!(m.validate().is_err());
    let mut m = original.clone();
    capture(&mut m).limits.maximum_retained_coordinates = NonZeroUsize::new(1).unwrap();
    assert!(m.validate().is_err());
    let mut m = original;
    capture(&mut m).catalog = "child0.jsonl".into();
    assert!(m.validate().is_err());
}
#[test]
fn structured_group_cli_requires_fresh_original_clock_and_total_import_capacity() {
    let m = manifest();
    let c = m.validation_model.structured_group_v2().unwrap();
    let mut policy = ferrum_types::SloConfig::default();
    policy.cost_observation = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(100);
    policy.cost_observation.profile_import.max_file_bytes =
        NonZeroUsize::new(32 * 1024 * 1024).unwrap();
    // Bound the real declared child files, leaving explicit catalog/profile space.
    let mut configured = c.clone();
    configured.limits.maximum_total_file_bytes = NonZeroU64::new(16 * 1024 * 1024).unwrap();
    configured.validate_policy(&m, &policy).unwrap();
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = None;
    assert!(configured.validate_policy(&m, &policy).is_err());
    policy
        .cost_observation
        .profile_import
        .declared_local_clock_max_error_ns = Some(100);
    policy.cost_profile = Some("old-catalog.json".into());
    assert!(configured.validate_policy(&m, &policy).is_err());
    policy.cost_profile = None;
    policy.cost_observation.profile_import.max_samples = NonZeroUsize::new(47).unwrap();
    assert!(configured.validate_policy(&m, &policy).is_err());
}
#[test]
fn structured_group_cli_one_failed_child_prevents_whole_group_export() {
    let m = manifest();
    let mut group = GroupReportV2::new(m.validation_model.structured_group_v2().unwrap());
    for (i, child) in group.children.iter_mut().enumerate() {
        child.source = Some(report::SourceReceiptV2 {
            source_path: format!("{i}.jsonl").into(),
            source_sha256: [1; 32],
            source_bytes: 1,
            phase: StructuredCapturePhase::Qualified,
            offered_waves: 48,
            scope_members: 24,
            scope_failures: 0,
            failure: None,
            numerical_model_present: true,
        });
    }
    // Pure report gating creates no source, receipt, model or export authority.
    export::require_group_exportable(&group).unwrap();
    group.children[1].source.as_mut().unwrap().phase = StructuredCapturePhase::Failed;
    assert!(export::require_group_exportable(&group).is_err());
    group.children[1].source.as_mut().unwrap().phase = StructuredCapturePhase::Qualified;
    group.group_failure = Some("closing clock invalid".into());
    assert!(export::require_group_exportable(&group).is_err());
    assert!(group.verified_catalog.is_none());
}

#[test]
fn structured_group_cli_source4_explicit_shared_paths_and_physical_budget_roundtrip() {
    let mut m = manifest();
    let c = capture(&mut m);
    c.shared_source = Some("shared.jsonl".into());
    for child in &mut c.children {
        child.source = "shared.jsonl".into();
        child.profile = c.catalog.clone();
    }
    c.limits.maximum_total_file_bytes = NonZeroU64::new(8 * 1024 * 1024).unwrap();
    m.validate().unwrap();
    let wire = serde_json::to_value(&m).unwrap();
    let loaded: manifest::Manifest = serde_json::from_value(wire).unwrap();
    loaded.validate().unwrap();
    let mut changed = m.clone();
    capture(&mut changed).children[1].source = "other.jsonl".into();
    assert!(changed.validate().is_err());
    let mut changed = m.clone();
    capture(&mut changed).children[1].profile = "other.json".into();
    assert!(changed.validate().is_err());
    let mut changed = m.clone();
    capture(&mut changed).shared_source = None;
    assert!(
        changed.validate().is_err(),
        "aliasing remains illegal without typed source4 mode"
    );
    let mut changed = m;
    capture(&mut changed).limits.maximum_total_file_bytes = NonZeroU64::MIN;
    assert!(changed.validate().is_err());
}

#[test]
fn shared_source4_cli_startup_counts_declared_physical_outputs_once() {
    use clap::Parser;
    #[derive(Parser)]
    struct Args {
        #[command(flatten)]
        command: super::super::super::CalibrateSloCommand,
    }
    let mut m = manifest();
    let c = capture(&mut m);
    c.shared_source = Some("shared.jsonl".into());
    for child in &mut c.children {
        child.source = "shared.jsonl".into();
        child.profile = c.catalog.clone();
    }
    let directory = tempfile::tempdir().unwrap();
    c.resolve_paths(directory.path());
    m.validate().unwrap();
    let mut cmd = Args::try_parse_from([
        "fixture",
        "registered-model",
        "--manifest",
        "manifest.json",
        "--slo-config",
        "slo.json",
        "--startup-usage",
        "serve",
        "--observations",
        "raw.jsonl",
        "--out",
        "report.json",
    ])
    .unwrap()
    .command;
    cmd.observations = directory.path().join("raw.jsonl");
    cmd.out = directory.path().join("report.json");
    let paths = super::super::super::paths::outputs(&cmd, &m);
    super::super::super::paths::validate(paths).unwrap();
    // Aliasing an independent output is still rejected; sharing is declared by
    // source4 identity, not a generic deduplication of arbitrary paths.
    cmd.out = m
        .validation_model
        .structured_group_v2()
        .unwrap()
        .catalog
        .clone();
    assert!(
        super::super::super::paths::validate(super::super::super::paths::outputs(&cmd, &m))
            .is_err()
    );
}
