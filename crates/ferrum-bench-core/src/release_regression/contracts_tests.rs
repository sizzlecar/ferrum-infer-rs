use super::*;
use serde_json::json;

fn artifact() -> HarnessArtifact {
    HarnessArtifact {
        package: "ferrum-bench-core".into(),
        package_id: "self-test-harness".into(),
        target: "ferrum_bench_core".into(),
        kind: "lib".into(),
        executable: std::env::current_exe().unwrap(),
        manifest_path: Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
    }
}
fn group(name: &str) -> ContractGroup {
    ContractGroup {
        id: "self-test.contract".into(),
        behavior: Behavior::WorkspaceChecks,
        entrypoints: vec![Entrypoint::Run],
        tests: vec![lib("ferrum-bench-core", name)],
    }
}
fn probe_name() -> String {
    format!(
        "{}::harness_assertion_probe",
        module_path!().split_once("::").unwrap().1
    )
}
fn options(directory: &Path) -> ContractRunOptions {
    ContractRunOptions {
        log_dir: directory.join("logs"),
        timeout: Duration::from_secs(5),
    }
}

// Invoked through this very compiled test harness. It is intentionally ignored
// in ordinary workspace runs; --include-ignored must make its assertion execute.
// These subprocess-only controls are test plumbing, not a product environment API.
#[test]
#[ignore = "executed explicitly by the contract harness self-tests"]
fn harness_assertion_probe() {
    let mode = std::env::var("FERRUM_CONTRACT_SELFTEST_MODE").unwrap_or_default();
    if mode == "timeout" {
        std::thread::sleep(Duration::from_secs(10));
    }
    let expected = if mode == "fail" { 41 } else { 42 };
    assert_eq!(
        17 + 25,
        expected,
        "actual assertion selected by the contract executor"
    );
}

#[test]
fn cargo_artifacts_use_manifest_package_and_require_successful_build_completion() {
    let directory = tempfile::tempdir().unwrap();
    let manifest = directory.path().join("Cargo.toml");
    fs::write(
        &manifest,
        "[package]\nname = 'fixture-package'\nversion = '1.0.0'\n",
    )
    .unwrap();
    let artifact = json!({
        "reason": "compiler-artifact", "package_id": "path+file:///fixture#1.0.0",
        "manifest_path": manifest, "target": {"name": "different_target", "kind": ["test"]},
        "profile": {"test": true}, "executable": std::env::current_exe().unwrap(), "fresh": true,
    });
    let complete =
        format!("{artifact}\n{artifact}\n{{\"reason\":\"build-finished\",\"success\":true}}\n");
    let parsed = parse_compiler_artifacts(&complete).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].package, "fixture-package");
    assert_eq!(parsed[0].target, "different_target");
    for invalid in [
        artifact.to_string(),
        complete.replace("\"success\":true", "\"success\":false"),
        format!("{complete}{artifact}\n"),
        format!("non-JSON build message\n{complete}"),
    ] {
        assert!(parse_compiler_artifacts(&invalid).is_err());
    }
    let mut ordinary = artifact;
    ordinary["profile"]["test"] = json!(false);
    assert!(parse_compiler_artifacts(&format!(
        "{ordinary}\n{{\"reason\":\"build-finished\",\"success\":true}}\n"
    ))
    .unwrap()
    .is_empty());
}

#[test]
fn exact_registered_ignored_assertion_executes_and_an_actual_failure_blocks() {
    let groups = vec![group(&probe_name())];
    for (mode, expected) in [
        ("pass", ContractStatus::Passed),
        ("fail", ContractStatus::Failed),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let report = run_checks(
            &[artifact()],
            &groups,
            &options(directory.path()),
            &|command| {
                command.env("FERRUM_CONTRACT_SELFTEST_MODE", mode);
            },
        )
        .unwrap();
        assert_eq!(report.status, expected);
        assert!(report.groups[0].tests[0].registered);
        let exit = report.groups[0].tests[0]
            .execution
            .as_ref()
            .unwrap()
            .exit_code;
        if mode == "pass" {
            assert_eq!(exit, Some(0));
            verify_contract_report(&groups, &report).unwrap();
        } else {
            assert_ne!(exit, Some(0));
            assert!(verify_contract_report(&groups, &report).is_err());
            let mut false_summary = report;
            false_summary.status = ContractStatus::Passed;
            false_summary.groups[0].status = ContractStatus::Passed;
            assert!(verify_contract_report(&groups, &false_summary).is_err());
        }
    }
}

#[test]
fn removed_test_missing_harness_and_ambiguous_artifacts_never_become_empty_passes() {
    let name = probe_name();
    for (artifacts, required) in [
        (vec![artifact()], group(&format!("{name}_removed"))),
        (vec![], group(&name)),
        (
            vec![
                artifact(),
                HarnessArtifact {
                    executable: PathBuf::from("/different/harness"),
                    ..artifact()
                },
            ],
            group(&name),
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let report =
            run_contract_checks(&artifacts, &[required.clone()], &options(directory.path()))
                .unwrap();
        assert_eq!(report.status, ContractStatus::Failed);
        assert!(!report.groups[0].tests[0].registered);
        assert!(report.groups[0].tests[0].execution.is_none());
        assert!(verify_contract_report(&[required], &report).is_err());
    }
}

#[test]
fn a_registered_test_timeout_is_killed_and_cannot_be_reported_as_passed() {
    let directory = tempfile::tempdir().unwrap();
    let groups = vec![group(&probe_name())];
    let mut options = options(directory.path());
    options.timeout = Duration::from_secs(1);
    let report = run_checks(&[artifact()], &groups, &options, &|command| {
        command.env("FERRUM_CONTRACT_SELFTEST_MODE", "timeout");
    })
    .unwrap();
    let execution = report.groups[0].tests[0].execution.as_ref().unwrap();
    assert_eq!(execution.status, CommandStatus::TimedOut);
    assert_eq!(report.status, ContractStatus::Failed);
    assert!(verify_contract_report(&groups, &report).is_err());
}

#[test]
fn report_verification_requires_the_selected_groups_and_every_bound_assertion() {
    let directory = tempfile::tempdir().unwrap();
    let groups = vec![group(&probe_name())];
    let report = run_contract_checks(&[artifact()], &groups, &options(directory.path())).unwrap();
    verify_contract_report(&groups, &report).unwrap();
    let mut missing = report.clone();
    missing.groups.clear();
    assert!(verify_contract_report(&groups, &missing).is_err());
    let mut missing = report.clone();
    missing.groups[0].tests.clear();
    assert!(verify_contract_report(&groups, &missing).is_err());
    let mut contradictory = report.clone();
    contradictory.groups[0].tests[0].registered = false;
    assert!(verify_contract_report(&groups, &contradictory).is_err());
    let mut substituted = report;
    substituted.groups[0].tests[0]
        .artifact
        .as_mut()
        .unwrap()
        .package = "other-package".into();
    assert!(verify_contract_report(&groups, &substituted).is_err());
}

#[test]
fn descriptors_preserve_actual_entrypoints_and_leave_unimplemented_groups_unbound() {
    let groups = contract_groups();
    let descriptors = contract_check_descriptors();
    for group in &groups {
        let descriptor = descriptors
            .iter()
            .find(|descriptor| descriptor.id == group.id)
            .unwrap();
        assert_eq!(descriptor.behavior, group.behavior);
        assert_eq!(descriptor.entrypoints, group.entrypoints);
        assert_eq!(descriptor.layer, EvidenceLayer::Contract);
        assert!(!group.tests.is_empty());
    }
    // Structured/tool engine probes and captured continuation requests are HTTP-specific.
    for behavior in [
        Behavior::StructuredSampling,
        Behavior::ToolHandoff,
        Behavior::ToolSelection,
        Behavior::ToolContinuation,
    ] {
        let descriptor = descriptors
            .iter()
            .find(|descriptor| descriptor.behavior == behavior)
            .unwrap();
        assert!(!descriptor.entrypoints.contains(&Entrypoint::Run));
    }
    assert!(!descriptors.iter().any(|descriptor| matches!(
        descriptor.behavior,
        Behavior::KernelNumerics | Behavior::KernelBoundaries | Behavior::ArchitectureState
    )));
}

#[test]
fn weight_materialization_requires_executed_assertions_not_a_passed_summary() {
    let required = contract_groups()
        .into_iter()
        .find(|group| group.behavior == Behavior::WeightMaterialization)
        .expect("weight materialization has real registered source contracts");
    let directory = tempfile::tempdir().unwrap();
    let mut report = run_contract_checks(
        &[],
        std::slice::from_ref(&required),
        &options(directory.path()),
    )
    .unwrap();
    assert_eq!(report.status, ContractStatus::Failed);
    report.status = ContractStatus::Passed;
    report.groups[0].status = ContractStatus::Passed;
    assert!(verify_contract_report(std::slice::from_ref(&required), &report).is_err());
    report.groups[0].tests.clear();
    assert!(verify_contract_report(std::slice::from_ref(&required), &report).is_err());
    report.groups.clear();
    assert!(verify_contract_report(&[required], &report).is_err());
}
