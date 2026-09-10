use super::*;
use ferrum_bench_core::release_regression::model_schedule::model_check_descriptors;
use ferrum_bench_core::release_regression::model_tasks::ModelCheck;
use ferrum_bench_core::release_regression::{
    analyze_paths, plan, ExecutionTarget, ModelProfile, PlanInput, Stage,
};
use ferrum_types::ModelOutputProtocol;
fn fixture() -> Plan {
    fixture_with_quick_start(true)
}

fn fixture_with_quick_start(all_quick_start: bool) -> Plan {
    let profiles: Vec<_> = [Backend::Metal, Backend::Cuda]
        .into_iter()
        .map(|backend| ModelProfile {
            gguf: None,
            reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened,
            id: format!("quick-{backend:?}"),
            model: format!("public-{backend:?}:small"),
            available: true,
            estimate: None,
            target: ExecutionTarget {
                architecture: "dense-attention".into(),
                protocol: ModelOutputProtocol::Text,
                precision: "f16".into(),
                backend,
                execution_path: "production-plan-runtime".into(),
            },
        })
        .collect();
    plan(&PlanInput {
        release_performance: Default::default(),
        stage: Stage::Release,
        impact: analyze_paths(Vec::<String>::new()),
        quick_start_profile_ids: profiles
            .iter()
            .take(if all_quick_start { profiles.len() } else { 1 })
            .map(|p| p.id.clone())
            .collect(),
        required_targets: profiles.iter().map(|p| p.target.clone()).collect(),
        profiles,
        checks: model_check_descriptors(),
    })
    .unwrap()
}
fn assets() -> Vec<StagedBinary> {
    [Backend::Metal, Backend::Cuda]
        .into_iter()
        .enumerate()
        .map(|(i, backend)| StagedBinary {
            backend,
            sha256: format!("{i:064x}"),
        })
        .collect()
}
#[test]
fn prepares_distinct_quick_start_defaults_and_keeps_other_release_gaps() {
    let plan = fixture();
    let tasks = prepare(&plan, &assets(), "1.2.3", 512).unwrap();
    for selected in &plan.selected {
        let task = tasks
            .expectations
            .iter()
            .find(|task| task.profile.id == selected.profile.id)
            .unwrap();
        assert_eq!(task.profile, selected.profile);
        assert_eq!(task.checks, vec![ModelCheck::Basic]);
        assert!(task.use_default_backend && task.disable_thinking);
        assert_eq!(task.runtime_capacity, None);
        assert_eq!(
            task.binary_sha256,
            assets()
                .iter()
                .find(|a| a.backend == task.profile.target.backend)
                .unwrap()
                .sha256
        );
    }
    assert!(tasks.unsupported_obligations.is_empty());
    assert!(!plan.gaps.is_empty());
    assert_eq!(tasks.remaining_plan_gaps, plan.gaps);
    assert!(prepare(&plan, &assets()[..1], "1.2.3", 512).is_err());
    assert!(prepare(&plan, &assets(), "1.2.3-rc.1", 512).is_err());
    assert!(prepare(&plan, &assets(), "1.2.3", 0).is_err());
}
#[test]
fn conflicting_or_missing_staged_bytes_are_rejected() {
    let abi = json!({"schema_version":1,"backend":"cuda","asset_name":"candidate.tar.gz","asset_sha256":"archive","binary_sha256":"a".repeat(64),"release_candidate_sha":"candidate-a"});
    let mut version = abi.clone();
    version["version"] = json!("1.2.3");
    assert!(staged_binary(&abi, &version, "1.2.3", "candidate-a").is_ok());
    for (field, value) in [
        ("binary_sha256", json!("b".repeat(64))),
        ("asset_sha256", json!("different")),
        ("asset_name", json!(null)),
        ("version", json!("1.2.4")),
    ] {
        let mut changed = version.clone();
        changed[field] = value;
        assert!(
            staged_binary(&abi, &changed, "1.2.3", "candidate-a").is_err(),
            "{field}"
        );
    }
    assert!(staged_binary(&abi, &version, "1.2.3", "candidate-b").is_err());
    let mut invalid = abi.clone();
    invalid["backend"] = json!("auto");
    assert!(staged_binary(&invalid, &version, "1.2.3", "candidate-a").is_err());
    let mut duplicates = assets();
    duplicates.push(StagedBinary {
        backend: Backend::Cuda,
        sha256: "b".repeat(64),
    });
    assert!(prepare(&fixture(), &duplicates, "1.2.3", 512).is_err());
}
#[test]
fn cli_verifier_saves_failure_for_missing_required_model_results() {
    let dir = tempfile::tempdir().unwrap();
    let tasks = prepare(&fixture(), &assets(), "1.2.3", 512).unwrap();
    let task_path = dir.path().join("tasks.json");
    let output = dir.path().join("verification.json");
    write_new(&task_path, &tasks).unwrap();
    assert!(run(Action::Verify {
        tasks: task_path,
        report: Vec::new(),
        output: output.clone()
    })
    .is_err());
    let report = read_json(&output).unwrap();
    assert_eq!(report["model_tasks_passed"], false);
    assert_eq!(report["release_approved"], false);
    assert!(!report["issues"].as_array().unwrap().is_empty());
}

#[test]
fn functional_tasks_declare_context_room_without_changing_output_budget() {
    let tasks = prepare(&fixture_with_quick_start(false), &assets(), "1.2.3", 512).unwrap();
    assert!(tasks
        .expectations
        .iter()
        .any(|task| task.use_default_backend));
    assert!(tasks
        .expectations
        .iter()
        .any(|task| !task.use_default_backend));
    for task in tasks.expectations {
        let expected_capacity = if task.profile.target.backend == Backend::Metal {
            None
        } else {
            Some(DEFAULT_FUNCTIONAL_CAPACITY)
        };
        assert_eq!(task.runtime_capacity, expected_capacity);
        assert_eq!(task.max_tokens, 512);
        assert_eq!(task.use_default_backend, expected_capacity.is_none());
    }
    assert!(prepare(&fixture_with_quick_start(false), &assets(), "1.2.3", 2048).is_err());
}
