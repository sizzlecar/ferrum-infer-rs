use super::*;
use ferrum_bench_core::slo_comparison::{
    artifact::EligibilityArtifactRefs, FrozenEligibilityPlan, InferenceAssumptions,
};
use serde::Serialize;

fn save<T: Serialize>(root: &Path, name: &str, value: &T) -> ArtifactFileRef {
    let bytes = serde_json::to_vec(value).unwrap();
    fs::write(root.join(name), &bytes).unwrap();
    ArtifactFileRef {
        path: name.into(),
        sha256: hash(&bytes),
        bytes: bytes.len() as u64,
    }
}

fn config(root: &Path, cmd: &mut SloCompareCommand, limits: serde_json::Value) {
    let path = root.join("limits.json");
    fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1, "limits": limits
        }))
        .unwrap(),
    )
    .unwrap();
    cmd.limits_config = Some(path);
}

fn no_outputs(cmd: &SloCompareCommand) -> bool {
    [&cmd.out, &cmd.markdown_en, &cmd.markdown_zh]
        .iter()
        .all(|path| !path.exists())
}

#[test]
fn typed_limits_option_preserves_defaults_and_accepts_explicit_partial_overrides() {
    let parsed = TestCli::try_parse_from([
        "ferrum",
        "slo-compare",
        "manifest.json",
        "--limits-config",
        "limits.json",
        "--out",
        "out.json",
        "--markdown-en",
        "en.md",
        "--markdown-zh",
        "zh.md",
    ])
    .unwrap();
    let TestCommand::SloCompare(cmd) = parsed.command;
    assert_eq!(cmd.limits_config, Some(PathBuf::from("limits.json")));
    let loaded = limits::load(None).unwrap();
    assert_eq!(
        serde_json::to_value(loaded.evidence).unwrap(),
        serde_json::to_value(ArtifactLoadLimits::default()).unwrap()
    );
    let (dir, mut cmd) = fixture();
    config(
        dir.path(),
        &mut cmd,
        serde_json::json!({
            "max_files": 512, "max_total_bytes": 536870912, "max_total_visible_gaps": 8000000
        }),
    );
    let loaded = limits::load(cmd.limits_config.as_deref()).unwrap();
    assert_eq!(loaded.evidence.max_files, 512);
    assert_eq!(loaded.evidence.max_total_bytes, 536870912);
    assert_eq!(loaded.evidence.max_total_visible_gaps, 8000000);
    assert_eq!(
        loaded.evidence.max_file_bytes,
        ArtifactLoadLimits::default().max_file_bytes
    );
    assert_eq!(execute(cmd).unwrap(), SloCompareExit::InsufficientEvidence);
}

#[test]
fn raising_cell_capacity_retains_the_full_frozen_scope_without_changing_evidence_status() {
    let (dir, mut cmd) = fixture();
    let count = ArtifactLoadLimits::default().max_cells + 1;
    let mut contract = frozen_contract();
    contract.cells = (1..=count)
        .map(|concurrency| FrozenCell {
            concurrency: concurrency as u32,
            scope: CellScope::Primary,
        })
        .collect();
    let mut manifest: ComparisonArtifactManifest =
        serde_json::from_slice(&fs::read(&cmd.manifest).unwrap()).unwrap();
    manifest.contract = save(dir.path(), "contract.json", &contract);
    save(dir.path(), "manifest.json", &manifest);
    assert!(execute(cmd.clone()).is_err());
    assert!(no_outputs(&cmd));
    config(
        dir.path(),
        &mut cmd,
        serde_json::json!({"max_cells": count}),
    );
    assert_eq!(
        execute(cmd.clone()).unwrap(),
        SloCompareExit::InsufficientEvidence
    );
    let output: ArtifactComparisonReport =
        serde_json::from_slice(&fs::read(&cmd.out).unwrap()).unwrap();
    assert_eq!(output.comparison.cells.len(), count);
    assert_eq!(
        output.comparison.frozen_contract_sha256,
        hash(&serde_json::to_vec(&contract).unwrap())
    );
    assert!(output
        .comparison
        .cells
        .iter()
        .all(|cell| cell.status == ComparisonStatus::Unknown));
}

#[test]
fn configured_budget_is_cumulative_across_main_and_pilot_files() {
    let (dir, mut cmd) = fixture();
    let main = compare_manifest(&cmd.manifest, &ArtifactLoadLimits::default()).unwrap();
    // max_files counts distinct referenced artifacts. The outer manifest is
    // listed for provenance and charged to total bytes, but is not a reference.
    let main_references = main.verified_files.len() - 1;
    let mut manifest: ComparisonArtifactManifest =
        serde_json::from_slice(&fs::read(&cmd.manifest).unwrap()).unwrap();
    let pilot = save(dir.path(), "pilot-manifest.json", &manifest);
    let plan = FrozenEligibilityPlan {
        schema_version: 1,
        frozen_unix_ns: 1,
        assumptions: InferenceAssumptions::IndependentExchangeablePairedBlocksFixedWorkload,
        planning_pair_counts: vec![2],
        planning_resamples: 4000,
        seed: 5,
        maximum_request_rank_step: 0.01,
        maximum_visible_gap_rank_step: 0.01,
        maximum_order_log_ratio_shift: 0.1,
        maximum_time_trend_log_ratio_shift: 0.1,
        maximum_absolute_lag_one_correlation: 0.9,
        maximum_within_pair_idle_ns: 1_000_000,
        independent_block_protocol: "fixture acquisition declaration".into(),
    };
    manifest.eligibility = Some(EligibilityArtifactRefs {
        plan: save(dir.path(), "plan.json", &plan),
        pilot_manifest: pilot,
    });
    save(dir.path(), "manifest.json", &manifest);
    let baseline = compare_manifest(&cmd.manifest, &ArtifactLoadLimits::default()).unwrap();
    let required_references = baseline.verified_files.len() - 1;
    let insufficient_references = required_references - 1;
    assert!(
        insufficient_references >= main_references,
        "failure budget must fit main evidence so the pilot crosses the shared limit"
    );
    config(
        dir.path(),
        &mut cmd,
        serde_json::json!({"max_files": insufficient_references}),
    );
    let error = execute(cmd.clone()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("referenced artifact count limit"),
        "expected pilot loading to exceed the shared file budget: {error}"
    );
    assert!(no_outputs(&cmd));
    config(
        dir.path(),
        &mut cmd,
        serde_json::json!({"max_files": required_references}),
    );
    assert_eq!(
        execute(cmd.clone()).unwrap(),
        SloCompareExit::InsufficientEvidence
    );
    let output: ArtifactComparisonReport =
        serde_json::from_slice(&fs::read(&cmd.out).unwrap()).unwrap();
    assert_eq!(output.verified_files.len(), required_references + 1);
    assert_eq!(
        serde_json::to_value(&output.verified_files).unwrap(),
        serde_json::to_value(&baseline.verified_files).unwrap(),
        "the exact budget must retain all original main and pilot provenance"
    );
    assert!(output.comparison.inference_eligibility_failure.is_some());
}

#[test]
fn malformed_unknown_negative_and_over_ceiling_limits_fail_before_output_creation() {
    for limits in [
        serde_json::json!({"max_files": 0}),
        serde_json::json!({"max_files": -1}),
        serde_json::json!({"max_files": 1.5}),
        serde_json::json!({"max_files": 4097}),
        serde_json::json!({"max_total_bytes": 1073741825_u64}),
        serde_json::json!({"max_total_visible_gaps": 16000001}),
        serde_json::json!({"max_fiels": 512}),
    ] {
        let (dir, mut cmd) = fixture();
        config(dir.path(), &mut cmd, limits);
        assert!(execute(cmd.clone())
            .unwrap_err()
            .to_string()
            .contains("limits-config"));
        assert!(no_outputs(&cmd));
    }
    for bytes in [
        b"not JSON".as_slice(),
        br#"{"schema_version":2,"limits":{}}"#,
        br#"{"schema_version":1,"limits":{},"trust_me":true}"#,
        br#"{"schema_version":1,"limits":{"max_files":3,"max_files":4}}"#,
    ] {
        let (dir, mut cmd) = fixture();
        let path = dir.path().join("limits.json");
        fs::write(&path, bytes).unwrap();
        cmd.limits_config = Some(path);
        assert!(execute(cmd.clone()).is_err());
        assert!(no_outputs(&cmd));
    }
}

#[test]
fn configuration_input_is_bounded_and_must_be_a_regular_file() {
    let (dir, mut cmd) = fixture();
    let path = dir.path().join("limits.json");
    fs::write(&path, vec![b' '; 16 * 1024 + 1]).unwrap();
    cmd.limits_config = Some(path);
    assert!(execute(cmd.clone())
        .unwrap_err()
        .to_string()
        .contains("16 KiB"));
    assert!(no_outputs(&cmd));
    cmd.limits_config = Some(dir.path().to_owned());
    assert!(execute(cmd.clone())
        .unwrap_err()
        .to_string()
        .contains("regular"));
    assert!(no_outputs(&cmd));
}

#[test]
fn limits_configuration_is_protected_from_output_aliases() {
    let (dir, mut cmd) = fixture();
    config(dir.path(), &mut cmd, serde_json::json!({}));
    let path = cmd.limits_config.clone().unwrap();
    let original = fs::read(&path).unwrap();
    cmd.out = path.clone();
    assert!(execute(cmd.clone())
        .unwrap_err()
        .to_string()
        .contains("aliases an input"));
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(!cmd.markdown_en.exists() && !cmd.markdown_zh.exists());
    #[cfg(unix)]
    {
        let alias = dir.path().join("config-alias.json");
        fs::hard_link(&path, &alias).unwrap();
        cmd.out = alias;
        assert!(execute(cmd.clone())
            .unwrap_err()
            .to_string()
            .contains("aliases an input"));
        assert_eq!(fs::read(&path).unwrap(), original);
    }
}
