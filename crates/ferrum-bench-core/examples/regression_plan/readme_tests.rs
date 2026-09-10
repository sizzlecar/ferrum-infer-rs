use super::*;
use ferrum_bench_core::release_regression::{
    analyze_paths, plan, Backend, ExecutionTarget, Gap, ModelProfile, PlanInput, Stage,
};
use ferrum_types::{ModelOutputProtocol, ModelReasoningProtocol};
use serde_json::json;

fn review(path: &str, before: &str, after: &str) -> ReadmeReview {
    ReadmeReview {
        path: path.into(),
        before_sha256: digest(before.as_bytes()),
        after_sha256: digest(after.as_bytes()),
        areas: vec![ChangeArea::Build, ChangeArea::Template],
        rationale: "Reviewed installation instructions and existing reasoning controls.".into(),
    }
}

fn input(impact: Impact) -> PlanInput {
    let target = ExecutionTarget {
        architecture: "dense".into(),
        protocol: ModelOutputProtocol::Text,
        precision: "bf16".into(),
        backend: Backend::Cuda,
        execution_path: "production-plan-runtime".into(),
    };
    PlanInput {
        release_profile_ids: Vec::new(),
        release_performance: Default::default(),
        stage: Stage::Release,
        impact,
        profiles: vec![ModelProfile {
            gguf: None,
            reasoning_protocol: ModelReasoningProtocol::PromptOpened,
            id: "quick-start".into(),
            model: "fixture/dense".into(),
            target: target.clone(),
            available: true,
            estimate: None,
        }],
        quick_start_profile_ids: vec!["quick-start".into()],
        required_targets: vec![target],
        checks: Vec::new(),
    }
}

#[test]
fn reviewed_readmes_preserve_source_unknown_paths_and_model_obligations() {
    let before = "Install the package.";
    let after = "Install the package and use the reasoning control.";
    let reviews = [
        review("README.md", before, after),
        review("README_zh.md", before, after),
    ];
    let mut impact = analyze_paths([
        "README.md",
        "README_zh.md",
        "crates/ferrum-engine/src/continuous_engine/sequence.rs",
        "unknown-runtime/forward.rs",
    ]);
    let sources: Vec<_> = impact
        .paths
        .iter()
        .filter(|entry| !readme(&entry.path))
        .cloned()
        .collect();
    let original_plan = plan(&input(impact.clone())).unwrap();
    assert!(original_plan.gaps.contains(&Gap::ProductContractReview));
    let applied = apply_reviews(&mut impact, &reviews, |_| {
        Ok((before.as_bytes().to_vec(), after.as_bytes().to_vec()))
    })
    .unwrap();
    assert_eq!(applied.applied, reviews);
    assert!(applied.unmatched.is_empty());
    assert!(!impact.product_contract_changed);
    assert_eq!(impact.unknown_paths, ["unknown-runtime/forward.rs"]);
    assert_eq!(impact.areas, ChangeArea::ALL);
    assert_eq!(
        impact
            .paths
            .iter()
            .filter(|entry| !readme(&entry.path))
            .cloned()
            .collect::<Vec<_>>(),
        sources
    );
    let reviewed_plan = plan(&input(impact)).unwrap();
    assert!(!reviewed_plan.gaps.contains(&Gap::ProductContractReview));
    assert!(reviewed_plan
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::UnmappedChange { .. })));
    assert_eq!(reviewed_plan.obligations, original_plan.obligations);
    assert_eq!(reviewed_plan.selected, original_plan.selected);
}

#[test]
fn changed_prose_model_or_performance_cannot_reuse_a_review() {
    let before = "Install the package.";
    let after = "Use the installed CLI.";
    let reviews = [review("README.md", before, after)];
    for (actual_before, actual_after) in [
        ("Different baseline.", after),
        (before, "Use the installed CLI. Supports a new model."),
        (before, "Use the installed CLI. Twice the throughput."),
        (before, "Use the installed CLI. Unlimited context."),
    ] {
        let mut impact = analyze_paths(["README.md"]);
        let original = impact.clone();
        let outcome = apply_reviews(&mut impact, &reviews, |_| {
            Ok((
                actual_before.as_bytes().to_vec(),
                actual_after.as_bytes().to_vec(),
            ))
        })
        .unwrap();
        assert!(outcome.applied.is_empty());
        assert_eq!(outcome.unmatched.len(), 1);
        assert_eq!(outcome.unmatched[0].path, "README.md");
        assert_eq!(
            outcome.unmatched[0].actual_after_sha256,
            digest(actual_after.as_bytes())
        );
        assert!(plan(&input(impact.clone()))
            .unwrap()
            .gaps
            .contains(&Gap::ProductContractReview));
        assert_eq!(impact, original);
    }
}

#[test]
fn one_translation_does_not_approve_another_and_failed_reviews_are_atomic() {
    let before = "old";
    let after = "new";
    let english = review("README.md", before, after);
    let chinese = review("README_zh.md", before, after);
    let mut impact = analyze_paths(["README.md", "README_zh.md"]);
    apply_reviews(&mut impact, &[english.clone()], |_| {
        Ok((b"old".to_vec(), b"new".to_vec()))
    })
    .unwrap();
    assert!(impact.product_contract_changed);
    let mut impact = analyze_paths(["README.md", "README_zh.md"]);
    let original = impact.clone();
    assert!(apply_reviews(&mut impact, &[english, chinese], |path| {
        if path == "README.md" {
            Ok((b"old".to_vec(), b"new".to_vec()))
        } else {
            Err("missing immutable Git blob".into())
        }
    })
    .is_err());
    assert_eq!(impact, original);
}

#[test]
fn old_reviews_for_unchanged_readmes_are_inert() {
    let mut impact = analyze_paths(["crates/ferrum-engine/src/continuous_engine/sequence.rs"]);
    let original = impact.clone();
    let outcome = apply_reviews(&mut impact, &[review("README.md", "old", "new")], |_| {
        panic!("unchanged README must not be read against a new baseline")
    })
    .unwrap();
    assert!(outcome.applied.is_empty());
    assert!(outcome.unmatched.is_empty());
    assert_eq!(impact, original);
}

#[test]
fn malformed_or_broad_review_configuration_is_rejected() {
    let good = serde_json::to_value(review("README.md", "old", "new")).unwrap();
    let mut invalid = Vec::new();
    for (field, value) in [
        ("path", json!("crates/ferrum-engine/src/lib.rs")),
        ("path", json!("docs/new-model.md")),
        ("before_sha256", json!("not a digest")),
        ("rationale", json!(" \n ")),
        ("areas", json!([])),
        ("areas", json!(["validation"])),
        ("areas", json!(["architecture"])),
        ("areas", json!(["build", "build"])),
        ("passed", json!(true)),
    ] {
        let mut changed = good.clone();
        changed[field] = value;
        invalid.push(json!([changed]));
    }
    invalid.push(json!([good.clone(), good.clone()]));
    invalid.push(Value::Null);
    for reviews in invalid {
        assert!(take_reviews(&mut json!({"readme_reviews": reviews})).is_err());
    }
    let mut catalog = json!({"profiles": [], "readme_reviews": [good]});
    assert_eq!(take_reviews(&mut catalog).unwrap().len(), 1);
    assert_eq!(catalog, json!({"profiles": []}));
    assert!(take_reviews(&mut catalog).unwrap().is_empty());
}
