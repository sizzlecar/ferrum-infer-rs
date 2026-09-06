use super::*;
use ferrum_types::ModelOutputProtocol;

fn target(
    architecture: &str,
    protocol: ModelOutputProtocol,
    precision: &str,
    backend: Backend,
) -> ExecutionTarget {
    ExecutionTarget {
        architecture: architecture.into(),
        protocol,
        precision: precision.into(),
        backend,
        execution_path: "production-plan-runtime".into(),
    }
}

fn profile(id: &str, target: ExecutionTarget) -> ModelProfile {
    ModelProfile {
        id: id.into(),
        model: format!("fixture/{id}"),
        target,
        available: true,
        estimate: None,
    }
}

fn input(stage: Stage, targets: Vec<ExecutionTarget>, profiles: Vec<ModelProfile>) -> PlanInput {
    PlanInput {
        stage,
        impact: Impact {
            areas: Vec::new(),
            paths: Vec::new(),
            unknown_paths: Vec::new(),
            product_contract_changed: false,
        },
        profiles,
        quick_start_profile_ids: Vec::new(),
        required_targets: targets,
        checks: Vec::new(),
    }
}

fn text_target(precision: &str, backend: Backend) -> ExecutionTarget {
    target("dense", ModelOutputProtocol::Text, precision, backend)
}

fn estimate(duration: u64, rate: Option<u64>, currency: Option<&str>) -> CostEstimate {
    CostEstimate {
        download_ms: 0,
        load_ms: 0,
        check_ms: duration,
        setup_ms: 0,
        cleanup_ms: 0,
        billable_ms: duration,
        hourly_rate_microunits: rate,
        fixed_cost_microunits: Some(0),
        currency: currency.map(str::to_owned),
    }
}

fn selected_ids(plan: &Plan) -> Vec<&str> {
    plan.selected
        .iter()
        .map(|selected| selected.profile.id.as_str())
        .collect()
}

#[test]
fn release_keeps_every_quick_start_and_samples_architecture_with_protocol() {
    let text = text_target("bf16", Backend::Metal);
    let alternate = text_target("int4", Backend::Metal);
    let harmony = target(
        "dense",
        ModelOutputProtocol::HarmonyGptOss,
        "mxfp4",
        Backend::Cuda,
    );
    let mut request = input(
        Stage::Release,
        vec![text.clone(), alternate.clone(), harmony.clone()],
        vec![
            profile("readme", text.clone()),
            profile("cheaper", alternate),
            profile("harmony", harmony.clone()),
        ],
    );
    request.profiles[1].estimate = Some(estimate(1, Some(1), Some("USD")));
    request.quick_start_profile_ids = vec!["readme".into()];
    let result = plan(&request).unwrap();
    assert_eq!(selected_ids(&result), ["harmony", "readme"]);
    assert!(result.obligations.iter().any(|obligation| obligation.behavior == Behavior::QuickStart
        && matches!(&obligation.scope, ObligationScope::Profile { profile_id, .. } if profile_id == "readme")));
    for expected in [text, harmony] {
        assert!(result
            .obligations
            .iter()
            .any(|obligation| obligation.behavior == Behavior::ModelForward
                && obligation.scope
                    == ObligationScope::Architecture {
                        architecture: expected.architecture.clone(),
                        protocol: expected.protocol
                    }));
        assert!(result.obligations.iter().any(|obligation| obligation.layer
            == EvidenceLayer::ModelRuntime
            && obligation.scope
                == ObligationScope::Backend {
                    backend: expected.backend
                }));
    }
    assert!(!result
        .obligations
        .iter()
        .any(|obligation| matches!(obligation.scope, ObligationScope::Target { .. })));
    assert!(result
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::UnassignedCheck { .. })));
}

#[test]
fn protocol_changes_use_protocol_backend_representatives_not_all_precisions() {
    let metal = text_target("bf16", Backend::Metal);
    let unselected_precision = text_target("int4", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let mut request = input(
        Stage::PullRequest,
        vec![metal.clone(), unselected_precision, cuda.clone()],
        vec![profile("metal", metal), profile("cuda", cuda)],
    );
    request.impact.areas = vec![ChangeArea::Termination];
    let result = plan(&request).unwrap();
    assert_eq!(selected_ids(&result), ["cuda", "metal"]);
    assert!(!result
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::MissingRepresentative { .. })));
    assert!(!result
        .obligations
        .iter()
        .any(|obligation| matches!(obligation.scope, ObligationScope::Target { .. })));
    for backend in [Backend::Metal, Backend::Cuda] {
        assert!(result
            .obligations
            .iter()
            .any(|obligation| obligation.behavior == Behavior::UserStop
                && obligation.layer == EvidenceLayer::ModelRuntime
                && obligation.scope
                    == ObligationScope::Protocol {
                        protocol: ModelOutputProtocol::Text,
                        backend
                    }));
    }
}

#[test]
fn compute_change_retains_missing_exact_precision_and_numeric_checker() {
    let available = text_target("bf16", Backend::Cuda);
    let missing = text_target("int4", Backend::Cuda);
    let mut request = input(
        Stage::PullRequest,
        vec![available.clone(), missing.clone()],
        vec![profile("available", available)],
    );
    request.impact.areas = vec![ChangeArea::Kernel];
    request.checks = vec![CheckDescriptor {
        id: "compile".into(),
        behavior: Behavior::WorkspaceChecks,
        layer: EvidenceLayer::Compilation,
        entrypoints: ENTRYPOINTS.to_vec(),
        target: None,
    }];
    let result = plan(&request).unwrap();
    let missing_forward = result
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::ModelForward
                && obligation.scope
                    == ObligationScope::Target {
                        target: missing.clone(),
                    }
        })
        .unwrap();
    assert!(result.gaps.contains(&Gap::MissingRepresentative {
        obligation: missing_forward
    }));
    for (index, obligation) in result
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| obligation.behavior == Behavior::KernelNumerics)
    {
        assert_eq!(obligation.layer, EvidenceLayer::BackendNumerics);
        assert!(obligation.checkers.is_empty());
        assert!(result
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
}

#[test]
fn global_contracts_do_not_require_a_model_and_compile_cannot_cover_them() {
    let mut request = input(Stage::PullRequest, Vec::new(), Vec::new());
    request.impact.areas = vec![ChangeArea::Template];
    request.checks = vec![
        CheckDescriptor {
            id: "wrong-layer".into(),
            behavior: Behavior::TemplateHistory,
            layer: EvidenceLayer::Compilation,
            entrypoints: ENTRYPOINTS.to_vec(),
            target: None,
        },
        CheckDescriptor {
            id: "run-only".into(),
            behavior: Behavior::TemplateHistory,
            layer: EvidenceLayer::Contract,
            entrypoints: vec![Entrypoint::Run],
            target: None,
        },
        CheckDescriptor {
            id: "shared-contract".into(),
            behavior: Behavior::TemplateHistory,
            layer: EvidenceLayer::Contract,
            entrypoints: ENTRYPOINTS.to_vec(),
            target: None,
        },
    ];
    let result = plan(&request).unwrap();
    assert!(result.selected.is_empty());
    assert!(result.gaps.contains(&Gap::EmptyInventory));
    assert!(!result
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::MissingRepresentative { .. })));
    let obligation = result
        .obligations
        .iter()
        .find(|obligation| obligation.behavior == Behavior::TemplateHistory)
        .unwrap();
    assert_eq!(obligation.scope, ObligationScope::Global);
    assert_eq!(obligation.checkers, ["run-only", "shared-contract"]);
}

#[test]
fn target_specific_checker_cannot_cover_a_different_selected_representative() {
    let selected = text_target("bf16", Backend::Metal);
    let other = text_target("int4", Backend::Metal);
    let mut request = input(
        Stage::PullRequest,
        vec![selected.clone(), other.clone()],
        vec![profile("selected", selected)],
    );
    request.impact.areas = vec![ChangeArea::Termination];
    request.checks = vec![CheckDescriptor {
        id: "other-target".into(),
        behavior: Behavior::UserStop,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: ENTRYPOINTS.to_vec(),
        target: Some(other),
    }];
    let result = plan(&request).unwrap();
    let (index, obligation) = result
        .obligations
        .iter()
        .enumerate()
        .find(|(_, obligation)| {
            obligation.behavior == Behavior::UserStop
                && obligation.layer == EvidenceLayer::ModelRuntime
        })
        .unwrap();
    assert!(obligation.checkers.is_empty());
    assert!(result
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
}

#[test]
fn unknown_change_expands_even_if_caller_omits_precomputed_areas() {
    let target = text_target("bf16", Backend::Cpu);
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("cpu", target)],
    );
    request.impact.unknown_paths = vec!["new-executor/forward.rs".into()];
    request.impact.product_contract_changed = true;
    let result = plan(&request).unwrap();
    assert!(result.gaps.contains(&Gap::UnmappedChange {
        path: "new-executor/forward.rs".into()
    }));
    assert!(result.gaps.contains(&Gap::ProductContractReview));
    for behavior in [
        Behavior::SourceRevision,
        Behavior::UserStop,
        Behavior::StructuredSampling,
        Behavior::ToolContinuation,
        Behavior::KvResume,
        Behavior::KernelNumerics,
    ] {
        assert!(
            result
                .obligations
                .iter()
                .any(|obligation| obligation.behavior == behavior),
            "{behavior:?}"
        );
    }
}

#[test]
fn empty_and_invalid_release_inputs_cannot_vacuously_cover_promises() {
    assert!(plan(&input(Stage::Release, Vec::new(), Vec::new())).is_err());
    let mut request = input(Stage::Release, Vec::new(), Vec::new());
    request.quick_start_profile_ids = vec!["missing-readme".into()];
    let result = plan(&request).unwrap();
    assert!(result.gaps.contains(&Gap::EmptyInventory));
    assert!(result.gaps.contains(&Gap::MissingQuickStart {
        profile_id: "missing-readme".into()
    }));
    let invalid = text_target(" ", Backend::Metal);
    request.required_targets.push(invalid.clone());
    request.profiles.push(profile("missing-readme", invalid));
    let result = plan(&request).unwrap();
    assert!(result
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::InvalidTarget { .. })));
    assert!(result.gaps.contains(&Gap::EmptyInventory));
    assert!(result.selected.is_empty());
}

#[test]
fn duplicate_identifiers_and_empty_checker_entrypoints_are_errors() {
    let target = text_target("bf16", Backend::Cpu);
    let one = profile("same", target.clone());
    let mut request = input(Stage::PullRequest, vec![target], vec![one.clone(), one]);
    assert!(plan(&request).is_err());
    request.profiles.pop();
    request.checks.push(CheckDescriptor {
        id: "invalid".into(),
        behavior: Behavior::UserStop,
        layer: EvidenceLayer::Contract,
        entrypoints: Vec::new(),
        target: None,
    });
    assert!(plan(&request).is_err());
}

#[test]
fn empty_pr_diff_is_cheap_but_nightly_still_requires_runtime_coverage() {
    let target = text_target("bf16", Backend::Metal);
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("metal", target)],
    );
    let result = plan(&request).unwrap();
    assert!(result.obligations.is_empty());
    assert!(result.selected.is_empty());
    request.stage = Stage::Nightly;
    let result = plan(&request).unwrap();
    assert!(result
        .obligations
        .iter()
        .any(|obligation| obligation.layer == EvidenceLayer::ModelRuntime));
    assert_eq!(selected_ids(&result), ["metal"]);
}

#[test]
fn selection_prefers_known_time_then_same_currency_price_and_does_not_convert_currencies() {
    let target = text_target("bf16", Backend::Metal);
    let mut request = input(
        Stage::Nightly,
        vec![target.clone()],
        vec![
            profile("a-unknown", target.clone()),
            profile("z-known", target.clone()),
        ],
    );
    request.profiles[1].estimate = Some(estimate(3_600_000, Some(7), Some("USD")));
    assert_eq!(selected_ids(&plan(&request).unwrap()), ["z-known"]);
    request.profiles[0].estimate = Some(estimate(3_600_000, Some(9), Some("USD")));
    assert_eq!(selected_ids(&plan(&request).unwrap()), ["z-known"]);
    request.profiles[0].estimate = Some(estimate(3_600_000, Some(1_000_000), Some("EUR")));
    assert_eq!(selected_ids(&plan(&request).unwrap()), ["a-unknown"]);
    request.profiles[0].estimate = Some(estimate(3_599_999, None, None));
    assert_eq!(selected_ids(&plan(&request).unwrap()), ["a-unknown"]);
}

#[test]
fn reused_profiles_are_counted_once_and_unknown_prices_remain_unknown() {
    let metal = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let mut request = input(
        Stage::Release,
        vec![metal.clone(), cuda.clone()],
        vec![profile("metal", metal), profile("cuda", cuda)],
    );
    request.quick_start_profile_ids = vec!["metal".into(), "cuda".into()];
    request.profiles[0].estimate = Some(estimate(100, None, None));
    let result = plan(&request).unwrap();
    assert_eq!(result.cost.known_total_ms, 100);
    assert_eq!(result.cost.unknown_duration_profiles, ["cuda"]);
    assert_eq!(result.cost.unknown_price_profiles, ["cuda", "metal"]);
    assert!(result.cost.estimated_costs.is_empty());
    assert!(result
        .selected
        .iter()
        .all(|selected| selected.obligations.len() > 1));
}

#[test]
fn invalid_estimate_is_visible_and_total_duration_overflow_fails() {
    let target = text_target("bf16", Backend::Metal);
    let mut request = input(
        Stage::Release,
        vec![target.clone()],
        vec![profile("a", target.clone()), profile("b", target)],
    );
    request.quick_start_profile_ids = vec!["a".into(), "b".into()];
    request.profiles[0].estimate = Some(estimate(u64::MAX, None, None));
    request.profiles[1].estimate = Some(estimate(1, None, None));
    assert!(plan(&request)
        .unwrap_err()
        .contains("duration sum overflow"));
    request.profiles[0].estimate = Some(estimate(1, Some(1), Some(" ")));
    let result = plan(&request).unwrap();
    assert!(result
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::InvalidEstimate { profile_id, .. } if profile_id == "a")));
    assert!(result.cost.unknown_price_profiles.contains(&"a".into()));
    assert!(result.cost.estimated_costs.is_empty());
}

#[test]
fn currency_totals_remain_separate_and_same_currency_sum_overflow_fails() {
    let target = text_target("bf16", Backend::Metal);
    let mut request = input(
        Stage::Release,
        vec![target.clone()],
        vec![profile("a", target.clone()), profile("b", target)],
    );
    request.quick_start_profile_ids = vec!["a".into(), "b".into()];
    request.profiles[0].estimate = Some(estimate(3_600_000, Some(u64::MAX), Some("USD")));
    request.profiles[1].estimate = Some(estimate(3_600_000, Some(u64::MAX), Some("EUR")));
    let result = plan(&request).unwrap();
    assert_eq!(
        result.cost.estimated_costs,
        [
            CurrencyEstimate {
                currency: "EUR".into(),
                microunits: u64::MAX
            },
            CurrencyEstimate {
                currency: "USD".into(),
                microunits: u64::MAX
            }
        ]
    );
    request.profiles[1].estimate.as_mut().unwrap().currency = Some("USD".into());
    assert!(plan(&request).unwrap_err().contains("price sum overflow"));
}

#[test]
fn independent_entrypoint_checkers_can_jointly_assign_one_contract() {
    let mut request = input(Stage::PullRequest, Vec::new(), Vec::new());
    request.impact.areas = vec![ChangeArea::Termination];
    request.checks = vec![
        CheckDescriptor {
            id: "run-stop".into(),
            behavior: Behavior::UserStop,
            layer: EvidenceLayer::Contract,
            entrypoints: vec![Entrypoint::Run],
            target: None,
        },
        CheckDescriptor {
            id: "http-stop".into(),
            behavior: Behavior::UserStop,
            layer: EvidenceLayer::Contract,
            entrypoints: vec![Entrypoint::ServeSync, Entrypoint::ServeStream],
            target: None,
        },
    ];
    let complete = plan(&request).unwrap();
    let index = complete
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::UserStop && obligation.layer == EvidenceLayer::Contract
        })
        .unwrap();
    assert!(!complete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    assert_eq!(
        complete.obligations[index].checkers,
        ["http-stop", "run-stop"]
    );
    request.checks[1].entrypoints = vec![Entrypoint::ServeSync];
    let incomplete = plan(&request).unwrap();
    assert!(incomplete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
}

#[test]
fn unresolved_target_dimensions_never_supply_inventory_or_representative_coverage() {
    for placeholder in [
        "unknown",
        "UNKNOWN",
        "model-native",
        "model_native",
        "unknown-bf16",
        "architecture-TBD",
        " ",
    ] {
        for dimension in 0..3 {
            let mut target = text_target("bf16", Backend::Metal);
            match dimension {
                0 => target.architecture = placeholder.into(),
                1 => target.precision = placeholder.into(),
                _ => target.execution_path = placeholder.into(),
            }
            let request = input(
                Stage::Nightly,
                vec![target.clone()],
                vec![profile("unresolved", target)],
            );
            let result = plan(&request).unwrap();
            assert!(
                result.gaps.contains(&Gap::EmptyInventory),
                "{dimension}: {placeholder}"
            );
            assert!(result
                .gaps
                .iter()
                .any(|gap| matches!(gap, Gap::InvalidTarget { .. })));
            assert!(result.selected.is_empty());
        }
    }
    let target = target(
        "new-concrete-recurrent-layout",
        ModelOutputProtocol::Text,
        "block-fp8-e4m3",
        Backend::Cuda,
    );
    let result = plan(&input(
        Stage::Nightly,
        vec![target.clone()],
        vec![profile("new-layout", target)],
    ))
    .unwrap();
    assert!(!result
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::InvalidTarget { .. })));
    assert_eq!(selected_ids(&result), ["new-layout"]);
}

#[test]
fn structured_and_tool_obligations_use_only_exposed_http_entrypoints() {
    let target = text_target("bf16", Backend::Cpu);
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("cpu", target)],
    );
    request.impact.areas = vec![ChangeArea::Structured, ChangeArea::Tools];
    let result = plan(&request).unwrap();
    for obligation in result.obligations.iter().filter(|obligation| {
        matches!(
            obligation.behavior,
            Behavior::StructuredSampling
                | Behavior::StructuredValidity
                | Behavior::ToolSelection
                | Behavior::ToolHandoff
                | Behavior::ToolContinuation
        )
    }) {
        assert_eq!(
            obligation.entrypoints,
            [Entrypoint::ServeSync, Entrypoint::ServeStream]
        );
    }
    request.checks.push(CheckDescriptor {
        id: "http-structure".into(),
        behavior: Behavior::StructuredValidity,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: vec![Entrypoint::ServeSync, Entrypoint::ServeStream],
        target: None,
    });
    let complete = plan(&request).unwrap();
    let index = complete
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::StructuredValidity
                && obligation.layer == EvidenceLayer::ModelRuntime
        })
        .unwrap();
    assert!(!complete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    request.checks[0].entrypoints = vec![Entrypoint::Run];
    let wrong_entrypoint = plan(&request).unwrap();
    assert!(wrong_entrypoint
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
}

#[test]
fn compile_and_numerical_bindings_require_a_checker_but_no_product_entrypoint() {
    let target = text_target("bf16", Backend::Cuda);
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("cuda", target.clone())],
    );
    request.impact.areas = vec![ChangeArea::Kernel];
    request.checks = vec![
        CheckDescriptor {
            id: "compile".into(),
            behavior: Behavior::WorkspaceChecks,
            layer: EvidenceLayer::Compilation,
            entrypoints: Vec::new(),
            target: None,
        },
        CheckDescriptor {
            id: "numerics".into(),
            behavior: Behavior::KernelNumerics,
            layer: EvidenceLayer::BackendNumerics,
            entrypoints: Vec::new(),
            target: Some(target),
        },
    ];
    let complete = plan(&request).unwrap();
    for behavior in [Behavior::WorkspaceChecks, Behavior::KernelNumerics] {
        let (index, obligation) = complete
            .obligations
            .iter()
            .enumerate()
            .find(|(_, obligation)| obligation.behavior == behavior)
            .unwrap();
        assert!(obligation.entrypoints.is_empty());
        assert!(!complete
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
    request.checks.clear();
    let missing = plan(&request).unwrap();
    for (index, obligation) in missing
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| {
            matches!(
                obligation.behavior,
                Behavior::WorkspaceChecks | Behavior::KernelNumerics
            )
        })
    {
        assert!(obligation.entrypoints.is_empty());
        assert!(missing
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
}

#[test]
fn baseline_architecture_execution_does_not_silently_require_performance_work() {
    let target = text_target("bf16", Backend::Metal);
    let mut request = input(
        Stage::Release,
        vec![target.clone()],
        vec![profile("readme", target)],
    );
    request.quick_start_profile_ids = vec!["readme".into()];
    for stage in [Stage::Release, Stage::Nightly] {
        request.stage = stage;
        let result = plan(&request).unwrap();
        assert!(result
            .obligations
            .iter()
            .any(|obligation| obligation.behavior == Behavior::ModelForward
                && obligation.layer == EvidenceLayer::ModelRuntime));
        assert!(!result
            .obligations
            .iter()
            .any(|obligation| obligation.layer == EvidenceLayer::Performance));
    }
    request.impact.areas = vec![ChangeArea::Kernel];
    let affected = plan(&request).unwrap();
    assert!(affected
        .obligations
        .iter()
        .any(|obligation| obligation.behavior == Behavior::Performance
            && obligation.layer == EvidenceLayer::Performance
            && matches!(obligation.scope, ObligationScope::Target { .. })));
}

#[test]
fn checker_entrypoints_cannot_be_combined_across_different_selected_targets() {
    let first = text_target("bf16", Backend::Metal);
    let second = text_target("int4", Backend::Metal);
    let mut request = input(
        Stage::Release,
        vec![first.clone(), second.clone()],
        vec![
            profile("first", first.clone()),
            profile("second", second.clone()),
        ],
    );
    request.quick_start_profile_ids = vec!["first".into(), "second".into()];
    request.checks = vec![
        CheckDescriptor {
            id: "first-run".into(),
            behavior: Behavior::ModelForward,
            layer: EvidenceLayer::ModelRuntime,
            entrypoints: vec![Entrypoint::Run],
            target: Some(first.clone()),
        },
        CheckDescriptor {
            id: "second-http".into(),
            behavior: Behavior::ModelForward,
            layer: EvidenceLayer::ModelRuntime,
            entrypoints: vec![Entrypoint::ServeSync, Entrypoint::ServeStream],
            target: Some(second.clone()),
        },
    ];
    let incomplete = plan(&request).unwrap();
    let index = incomplete
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::ModelForward
                && matches!(obligation.scope, ObligationScope::Architecture { .. })
        })
        .unwrap();
    assert!(incomplete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    request.checks.push(CheckDescriptor {
        id: "first-http".into(),
        behavior: Behavior::ModelForward,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: vec![Entrypoint::ServeSync, Entrypoint::ServeStream],
        target: Some(first),
    });
    let second_still_missing_run = plan(&request).unwrap();
    assert!(second_still_missing_run
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    request.checks.push(CheckDescriptor {
        id: "second-run".into(),
        behavior: Behavior::ModelForward,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: vec![Entrypoint::Run],
        target: Some(second),
    });
    let complete = plan(&request).unwrap();
    assert!(!complete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
}
