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
        gguf: None,
        reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened,
        id: id.into(),
        model: format!("fixture/{id}"),
        target,
        available: true,
        estimate: None,
    }
}

fn input(stage: Stage, targets: Vec<ExecutionTarget>, profiles: Vec<ModelProfile>) -> PlanInput {
    PlanInput {
        release_profile_ids: Vec::new(),
        release_performance: Default::default(),
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

fn obligation_owners(plan: &Plan, obligation: usize) -> Vec<&str> {
    plan.selected
        .iter()
        .filter(|selected| selected.obligations.contains(&obligation))
        .map(|selected| selected.profile.id.as_str())
        .collect()
}

#[test]
fn weight_materialization_binds_cpu_contracts_and_loads_every_declared_target() {
    let targets = vec![
        text_target("gguf-q4_k_m", Backend::Metal),
        text_target("gptq-int4", Backend::Cuda),
        text_target("bf16", Backend::Cuda),
    ];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| profile(&format!("weight-{index}"), target.clone()))
            .collect(),
    );
    request.impact = super::super::analyze_paths([
        "crates/ferrum-quantization/src/gptq_marlin_source.rs",
        "crates/ferrum-models/src/vnext/qwen3_moe/weights.rs",
    ]);
    request.checks = super::super::contracts::contract_check_descriptors();
    request
        .checks
        .extend(super::super::model_schedule::model_check_descriptors());
    let result = plan(&request).unwrap();
    let materialization = result
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::WeightMaterialization
                && obligation.layer == EvidenceLayer::Contract
                && obligation.scope == ObligationScope::Global
        })
        .unwrap();
    assert_eq!(
        result.obligations[materialization].checkers,
        ["cpu-contract.weight-materialization"]
    );
    assert!(!result.gaps.contains(&Gap::UnassignedCheck {
        obligation: materialization
    }));
    for target in &targets {
        for behavior in [Behavior::ModelLoad, Behavior::ModelForward] {
            let index = result
                .obligations
                .iter()
                .position(|obligation| {
                    obligation.behavior == behavior
                        && obligation.layer == EvidenceLayer::ModelRuntime
                        && obligation.scope
                            == ObligationScope::Target {
                                target: target.clone(),
                            }
                })
                .unwrap();
            assert_eq!(obligation_owners(&result, index).len(), 1);
            assert!(!result.obligations[index].checkers.is_empty());
            assert!(!result
                .gaps
                .contains(&Gap::UnassignedCheck { obligation: index }));
        }
    }
    assert!(!result.obligations.iter().any(|obligation| matches!(
        obligation.behavior,
        Behavior::KernelNumerics
            | Behavior::KernelBoundaries
            | Behavior::ArchitectureState
            | Behavior::Performance
    )));
    // A registration is still required: model forward cannot substitute for
    // source numerics merely because the same profile was selected.
    request
        .checks
        .retain(|check| check.behavior != Behavior::WeightMaterialization);
    let unbound = plan(&request).unwrap();
    let index = unbound
        .obligations
        .iter()
        .position(|obligation| obligation.behavior == Behavior::WeightMaterialization)
        .unwrap();
    assert!(unbound
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
}

#[test]
fn weight_source_classification_preserves_quick_start_and_other_performance_reach() {
    let mut legacy = text_target("gguf-q4_k_m", Backend::Metal);
    legacy.execution_path = "legacy-model-executor".into();
    let cuda = text_target("gptq-int4", Backend::Cuda);
    let mut request = input(
        Stage::Release,
        vec![legacy.clone(), cuda.clone()],
        vec![profile("quick-metal", legacy), profile("quick-cuda", cuda)],
    );
    request.quick_start_profile_ids = vec!["quick-metal".into(), "quick-cuda".into()];
    request.impact =
        super::super::analyze_paths(["crates/ferrum-kernels/src/backend/metal/mod.rs"]);
    mark_submission_proof(&mut request.impact);
    let baseline = plan(&request).unwrap();
    let weight_impact =
        super::super::analyze_paths(["crates/ferrum-models/src/vnext/qwen3_moe/weights.rs"]);
    request.impact.areas.extend(weight_impact.areas);
    request.impact.paths.extend(weight_impact.paths);
    let changed = plan(&request).unwrap();
    for prior in &baseline.obligations {
        assert!(
            changed.obligations.iter().any(|current| {
                current.behavior == prior.behavior
                    && current.layer == prior.layer
                    && current.scope == prior.scope
                    && current.entrypoints == prior.entrypoints
            }),
            "lost baseline obligation {prior:?}"
        );
    }
    for id in &request.quick_start_profile_ids {
        assert!(selected_ids(&changed).contains(&id.as_str()));
    }
    assert!(changed
        .obligations
        .iter()
        .any(|obligation| obligation.behavior == Behavior::Performance));
    assert!(changed
        .obligations
        .iter()
        .any(|obligation| obligation.behavior == Behavior::WeightMaterialization));

    // An actual device implementation contributor still requires device
    // numerical and performance evidence, independently of source decoding.
    let other = super::super::analyze_paths(["crates/ferrum-kernels/src/backend/cuda/quant.rs"]);
    request.impact.areas.extend(other.areas);
    request.impact.paths.extend(other.paths);
    let kernel = plan(&request).unwrap();
    assert!(kernel.obligations.iter().any(|obligation| {
        obligation.behavior == Behavior::KernelNumerics
            && obligation.layer == EvidenceLayer::BackendNumerics
            && matches!(&obligation.scope, ObligationScope::Target { target } if target.backend == Backend::Cuda)
    }));
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
                        protocol: expected.protocol,
                        execution_path: expected.execution_path.clone()
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
                        backend,
                        execution_path: "production-plan-runtime".into()
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
    let first_is_fully_bound = plan(&request).unwrap();
    assert!(!first_is_fully_bound
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    assert_eq!(obligation_owners(&first_is_fully_bound, index), ["first"]);
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

#[test]
fn cheaper_models_cannot_substitute_a_different_execution_path() {
    let production = text_target("bf16", Backend::Metal);
    let mut legacy = production.clone();
    legacy.execution_path = "legacy-model-executor".into();
    let mut small = profile("small-legacy", legacy.clone());
    small.estimate = Some(estimate(1, Some(1), Some("USD")));
    let mut large = profile("large-production", production.clone());
    large.estimate = Some(estimate(10_000, Some(1), Some("USD")));
    for (stage, area) in [
        (Stage::Nightly, None),
        (Stage::PullRequest, Some(ChangeArea::Termination)),
    ] {
        let mut request = input(
            stage,
            vec![legacy.clone(), production.clone()],
            vec![small.clone(), large.clone()],
        );
        request.impact.areas = area.into_iter().collect();
        let complete = plan(&request).unwrap();
        assert_eq!(
            selected_ids(&complete),
            ["large-production", "small-legacy"]
        );
        let production_obligation = complete
            .obligations
            .iter()
            .position(|obligation| {
                matches!(&obligation.scope,
                ObligationScope::Architecture { execution_path, .. }
                | ObligationScope::Protocol { execution_path, .. }
                if execution_path == &production.execution_path)
            })
            .unwrap();
        assert_eq!(
            obligation_owners(&complete, production_obligation),
            ["large-production"]
        );
        // An implementation available only on the cheap executor cannot bind
        // the production-path obligation, even though arch/protocol match.
        let obligation = &complete.obligations[production_obligation];
        request.checks.push(CheckDescriptor {
            id: "legacy-only".into(),
            behavior: obligation.behavior,
            layer: obligation.layer,
            entrypoints: obligation.entrypoints.clone(),
            target: Some(legacy.clone()),
        });
        let wrong_check = plan(&request).unwrap();
        assert!(wrong_check.gaps.contains(&Gap::UnassignedCheck {
            obligation: production_obligation
        }));
        request
            .profiles
            .retain(|profile| profile.id != "large-production");
        let missing = plan(&request).unwrap();
        assert!(missing.gaps.contains(&Gap::MissingRepresentative {
            obligation: production_obligation
        }));
        assert!(obligation_owners(&missing, production_obligation).is_empty());
    }
}

#[test]
fn architecture_sampling_retains_paths_without_forming_a_backend_cartesian_matrix() {
    let production_metal = text_target("bf16", Backend::Metal);
    let mut production_cuda = production_metal.clone();
    production_cuda.backend = Backend::Cuda;
    let mut legacy_metal = production_metal.clone();
    legacy_metal.execution_path = "legacy-model-executor".into();
    let result = plan(&input(
        Stage::Nightly,
        vec![
            production_metal,
            production_cuda.clone(),
            legacy_metal.clone(),
        ],
        vec![
            profile("production-cuda", production_cuda),
            profile("legacy-metal", legacy_metal),
        ],
    ))
    .unwrap();
    for (index, obligation) in result
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| obligation.layer == EvidenceLayer::ModelRuntime)
    {
        assert!(!result
            .gaps
            .contains(&Gap::MissingRepresentative { obligation: index }));
        let owner = obligation_owners(&result, index);
        assert_eq!(
            owner.len(),
            1,
            "each runtime obligation has one assigned representative"
        );
        if let ObligationScope::Architecture { execution_path, .. } = &obligation.scope {
            let assigned = result
                .selected
                .iter()
                .find(|selected| selected.profile.id == owner[0])
                .unwrap();
            assert_eq!(&assigned.profile.target.execution_path, execution_path);
        }
    }
    assert!(!result
        .obligations
        .iter()
        .any(|obligation| matches!(obligation.scope, ObligationScope::Target { .. })));
}

#[test]
fn overlapping_quick_starts_keep_their_own_checks_and_share_other_model_work_once() {
    let target = text_target("bf16", Backend::Metal);
    let mut first = profile("first-readme", target.clone());
    first.estimate = Some(estimate(100, Some(1), Some("USD")));
    let mut second = profile("second-readme", target.clone());
    second.estimate = Some(estimate(200, Some(1), Some("USD")));
    let mut alternative = profile("cheapest-extra", target.clone());
    alternative.estimate = Some(estimate(1, Some(1), Some("USD")));
    let mut request = input(
        Stage::Release,
        vec![target],
        vec![first, second, alternative],
    );
    request.quick_start_profile_ids = vec!["first-readme".into(), "second-readme".into()];
    request.impact.areas = vec![ChangeArea::Termination];
    let result = plan(&request).unwrap();
    assert_eq!(selected_ids(&result), ["first-readme", "second-readme"]);
    for (index, obligation) in result
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| requires_model(obligation))
    {
        match &obligation.scope {
            ObligationScope::Profile { profile_id, .. } => {
                assert_eq!(obligation.behavior, Behavior::QuickStart);
                assert_eq!(obligation_owners(&result, index), [profile_id.as_str()]);
            }
            _ => assert_eq!(obligation_owners(&result, index), ["first-readme"]),
        }
    }
    for id in &request.quick_start_profile_ids {
        assert!(result.obligations.iter().any(|obligation| {
            obligation.behavior == Behavior::QuickStart
                && matches!(&obligation.scope, ObligationScope::Profile { profile_id, .. } if profile_id == id)
        }));
    }
}

#[test]
fn shared_work_prefers_a_fully_bound_selected_representative_without_combining_targets() {
    let cheap_target = text_target("bf16", Backend::Metal);
    let bound_target = text_target("int4", Backend::Metal);
    let mut cheap = profile("cheap-readme", cheap_target.clone());
    cheap.estimate = Some(estimate(1, Some(1), Some("USD")));
    let mut bound = profile("bound-readme", bound_target.clone());
    bound.estimate = Some(estimate(100, Some(1), Some("USD")));
    let mut request = input(
        Stage::Release,
        vec![cheap_target.clone(), bound_target.clone()],
        vec![cheap, bound],
    );
    request.quick_start_profile_ids = vec!["cheap-readme".into(), "bound-readme".into()];
    request.impact.areas = vec![ChangeArea::Termination];
    request.checks.push(CheckDescriptor {
        id: "complete-bound-target".into(),
        behavior: Behavior::UserStop,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: ENTRYPOINTS.to_vec(),
        target: Some(bound_target),
    });
    let complete = plan(&request).unwrap();
    let index = complete
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::UserStop
                && obligation.layer == EvidenceLayer::ModelRuntime
        })
        .unwrap();
    assert_eq!(obligation_owners(&complete, index), ["bound-readme"]);
    assert!(!complete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    assert_eq!(
        complete.obligations[index].checkers,
        ["complete-bound-target"]
    );
    request.checks[0].entrypoints = vec![Entrypoint::ServeSync, Entrypoint::ServeStream];
    request.checks.push(CheckDescriptor {
        id: "cheap-run-only".into(),
        behavior: Behavior::UserStop,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: vec![Entrypoint::Run],
        target: Some(cheap_target),
    });
    let incomplete = plan(&request).unwrap();
    assert!(incomplete
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    assert_eq!(obligation_owners(&incomplete, index), ["cheap-readme"]);
    assert_eq!(incomplete.obligations[index].checkers, ["cheap-run-only"]);
}

#[test]
fn observability_contract_requires_real_registered_assertions_without_model_sampling() {
    let targets = vec![
        text_target("bf16", Backend::Metal),
        text_target("bf16", Backend::Cuda),
    ];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .into_iter()
            .map(|target| profile(&format!("{:?}", target.backend), target))
            .collect(),
    );
    request.impact.areas = vec![ChangeArea::ObservabilityContract];
    request.checks = super::super::contracts::contract_check_descriptors();
    let result = plan(&request).unwrap();
    let (index, obligation) = result
        .obligations
        .iter()
        .enumerate()
        .find(|(_, obligation)| obligation.behavior == Behavior::Observability)
        .unwrap();
    assert_eq!(obligation.layer, EvidenceLayer::Contract);
    assert_eq!(obligation.scope, ObligationScope::Global);
    assert_eq!(obligation.entrypoints, ENTRYPOINTS);
    assert_eq!(obligation.checkers, ["cpu-contract.observability"]);
    assert!(!result
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    assert!(result.selected.is_empty());
    assert!(!result.obligations.iter().any(requires_model));

    request
        .checks
        .retain(|check| check.behavior != Behavior::Observability);
    let missing = plan(&request).unwrap();
    let index = missing
        .obligations
        .iter()
        .position(|obligation| obligation.behavior == Behavior::Observability)
        .unwrap();
    assert!(missing
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
}

#[test]
fn contract_only_observability_cannot_erase_runtime_or_accept_basic_as_profile_evidence() {
    let targets = vec![
        text_target("bf16", Backend::Metal),
        text_target("bf16", Backend::Cuda),
    ];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .map(|target| profile(&format!("{:?}", target.backend), target.clone()))
            .collect(),
    );
    request.impact.areas = vec![ChangeArea::ObservabilityContract, ChangeArea::Observability];
    request.checks = super::super::contracts::contract_check_descriptors();
    // Basic and protocol checks do not inspect profile sinks. Only the actual
    // observation runner can provide the separate runtime evidence.
    request.checks.extend(
        super::super::model_schedule::model_check_descriptors()
            .into_iter()
            .filter(|check| check.behavior != Behavior::Observability),
    );
    let result = plan(&request).unwrap();
    for target in targets {
        let (index, obligation) = result
            .obligations
            .iter()
            .enumerate()
            .find(|(_, obligation)| {
                obligation.behavior == Behavior::Observability
                    && obligation.layer == EvidenceLayer::ModelRuntime
                    && obligation.scope
                        == ObligationScope::Backend {
                            backend: target.backend,
                        }
            })
            .unwrap();
        assert!(obligation.checkers.is_empty());
        assert!(result
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
    request.checks.extend(
        super::super::model_schedule::model_check_descriptors()
            .into_iter()
            .filter(|check| check.behavior == Behavior::Observability),
    );
    let executable = plan(&request).unwrap();
    for (index, obligation) in executable.obligations.iter().enumerate().filter(|(_, o)| {
        o.behavior == Behavior::Observability && o.layer == EvidenceLayer::ModelRuntime
    }) {
        assert_eq!(
            obligation.checkers,
            ["model-regression.observability.request-lifecycle"]
        );
        assert!(!executable
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
    request.impact.areas.push(ChangeArea::Kernel);
    let mixed = plan(&request).unwrap();
    assert!(mixed.obligations.iter().any(|obligation| {
        obligation.behavior == Behavior::KernelNumerics
            && obligation.layer == EvidenceLayer::BackendNumerics
    }));
}

#[test]
fn observability_requires_metadata_and_sink_checks_without_full_compute_expansion() {
    let metal = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let unrelated = target(
        "different-architecture",
        ModelOutputProtocol::Text,
        "int4",
        Backend::Metal,
    );
    let mut quick = profile("small-metal", metal.clone());
    quick.estimate = Some(estimate(1, Some(1), Some("USD")));
    let mut expensive = profile("large-metal", unrelated.clone());
    expensive.estimate = Some(estimate(1000, Some(1), Some("USD")));
    let mut request = input(
        Stage::PullRequest,
        vec![metal, cuda.clone(), unrelated],
        vec![quick, expensive, profile("cuda", cuda)],
    );
    request.impact.areas = vec![ChangeArea::Observability, ChangeArea::Validation];
    let missing = plan(&request).unwrap();
    assert_eq!(selected_ids(&missing), ["cuda", "small-metal"]);
    for (index, obligation) in missing.obligations.iter().enumerate() {
        assert!(!matches!(
            obligation.layer,
            EvidenceLayer::BackendNumerics | EvidenceLayer::Performance
        ));
        assert!(!matches!(
            obligation.scope,
            ObligationScope::Target { .. } | ObligationScope::Architecture { .. }
        ));
        if obligation.behavior == Behavior::Observability {
            assert!(missing
                .gaps
                .contains(&Gap::UnassignedCheck { obligation: index }));
            assert_eq!(obligation.entrypoints, ENTRYPOINTS);
            match obligation.layer {
                EvidenceLayer::Contract => assert_eq!(obligation.scope, ObligationScope::Global),
                EvidenceLayer::ModelRuntime => {
                    assert!(matches!(obligation.scope, ObligationScope::Backend { .. }))
                }
                other => panic!("observability is not {other:?}"),
            }
        }
    }
    for layer in [EvidenceLayer::Contract, EvidenceLayer::ModelRuntime] {
        request.checks.push(CheckDescriptor {
            id: format!("observability-{layer:?}"),
            behavior: Behavior::Observability,
            layer,
            entrypoints: ENTRYPOINTS.to_vec(),
            target: None,
        });
    }
    let bound = plan(&request).unwrap();
    for (index, obligation) in bound
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| obligation.behavior == Behavior::Observability)
    {
        assert!(!bound
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
        if requires_model(obligation) {
            assert_eq!(obligation_owners(&bound, index).len(), 1);
        }
    }
    // Narrow impact does not remove the release's independent promises.
    request.stage = Stage::Release;
    request.quick_start_profile_ids = vec!["small-metal".into(), "cuda".into()];
    let release = plan(&request).unwrap();
    assert!(release
        .selected
        .iter()
        .any(|selected| selected.profile.id == "large-metal"));
    for id in &request.quick_start_profile_ids {
        let index = release.obligations.iter().position(|obligation| {
            matches!(&obligation.scope, ObligationScope::Profile { profile_id, .. } if profile_id == id)
                && obligation.behavior == Behavior::QuickStart
        }).unwrap();
        assert_eq!(obligation_owners(&release, index), [id.as_str()]);
    }
    assert!(release
        .obligations
        .iter()
        .any(|obligation| matches!(obligation.scope, ObligationScope::Architecture { .. })));
    assert!(!release
        .obligations
        .iter()
        .any(|obligation| obligation.layer == EvidenceLayer::Performance));
}

fn observability_contract_bindings() -> Vec<CheckDescriptor> {
    vec![
        CheckDescriptor {
            id: "workspace".into(),
            behavior: Behavior::WorkspaceChecks,
            layer: EvidenceLayer::Compilation,
            entrypoints: Vec::new(),
            target: None,
        },
        CheckDescriptor {
            id: "metadata-contract".into(),
            behavior: Behavior::Observability,
            layer: EvidenceLayer::Contract,
            entrypoints: ENTRYPOINTS.to_vec(),
            target: None,
        },
    ]
}

fn runtime_binding(id: &str, behavior: Behavior, target: ExecutionTarget) -> CheckDescriptor {
    CheckDescriptor {
        id: id.into(),
        behavior,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: ENTRYPOINTS.to_vec(),
        target: Some(target),
    }
}

#[test]
fn scope_only_greedy_choice_cannot_hide_an_available_complete_checker_binding() {
    let cheap_target = text_target("bf16", Backend::Metal);
    let checked_target = text_target("int4", Backend::Metal);
    let mut cheap = profile("cheap-unbound", cheap_target.clone());
    cheap.estimate = Some(estimate(1, Some(1), Some("USD")));
    let mut checked = profile("checked", checked_target.clone());
    checked.estimate = Some(estimate(100, Some(1), Some("USD")));
    let mut request = input(
        Stage::PullRequest,
        vec![cheap_target, checked_target.clone()],
        vec![cheap, checked],
    );
    request.impact.areas = vec![ChangeArea::Observability];
    request.checks = observability_contract_bindings();
    request.checks.push(runtime_binding(
        "metadata-runtime",
        Behavior::Observability,
        checked_target,
    ));
    let complete = plan(&request).unwrap();
    assert!(complete.gaps.is_empty(), "{:#?}", complete.gaps);
    let index = complete
        .obligations
        .iter()
        .position(|obligation| {
            obligation.behavior == Behavior::Observability
                && obligation.layer == EvidenceLayer::ModelRuntime
        })
        .unwrap();
    assert_eq!(obligation_owners(&complete, index), ["checked"]);
    assert_eq!(selected_ids(&complete), ["checked"]);
    assert_eq!(
        complete.cost.known_total_ms, 100,
        "do not charge the unassigned initial choice"
    );
    request
        .checks
        .retain(|check| check.layer != EvidenceLayer::ModelRuntime);
    let missing = plan(&request).unwrap();
    assert!(missing
        .gaps
        .contains(&Gap::UnassignedCheck { obligation: index }));
    assert_eq!(obligation_owners(&missing, index), ["cheap-unbound"]);
}

#[test]
fn mandatory_quick_start_does_not_prevent_adding_a_bound_shared_representative() {
    let readme_target = text_target("bf16", Backend::Metal);
    let checked_target = text_target("int4", Backend::Metal);
    let mut readme = profile("readme", readme_target.clone());
    readme.estimate = Some(estimate(1, Some(1), Some("USD")));
    let mut checked = profile("checked", checked_target.clone());
    checked.estimate = Some(estimate(100, Some(1), Some("USD")));
    let mut request = input(
        Stage::Release,
        vec![readme_target.clone(), checked_target.clone()],
        vec![readme, checked],
    );
    request.quick_start_profile_ids = vec!["readme".into()];
    request.impact.areas = vec![ChangeArea::Observability];
    request.checks = observability_contract_bindings();
    request.checks.extend([
        runtime_binding("readme-command", Behavior::QuickStart, readme_target),
        runtime_binding(
            "actual-forward",
            Behavior::ModelForward,
            checked_target.clone(),
        ),
        runtime_binding(
            "actual-metadata-sink",
            Behavior::Observability,
            checked_target,
        ),
        CheckDescriptor {
            id: "installation".into(),
            behavior: Behavior::Installation,
            layer: EvidenceLayer::Installation,
            entrypoints: Vec::new(),
            target: None,
        },
    ]);
    let complete = plan(&request).unwrap();
    assert!(complete.gaps.is_empty(), "{:#?}", complete.gaps);
    assert_eq!(selected_ids(&complete), ["checked", "readme"]);
    for (index, obligation) in complete
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| requires_model(obligation))
    {
        let expected = if obligation.behavior == Behavior::QuickStart {
            "readme"
        } else {
            "checked"
        };
        assert_eq!(obligation_owners(&complete, index), [expected]);
    }
    // The alternate representative cannot erase a missing mandatory QuickStart
    // check even when its other runtime bindings are complete.
    request
        .checks
        .retain(|check| check.behavior != Behavior::QuickStart);
    let missing = plan(&request).unwrap();
    let quick_start = missing
        .obligations
        .iter()
        .position(|obligation| obligation.behavior == Behavior::QuickStart)
        .unwrap();
    assert!(missing.gaps.contains(&Gap::UnassignedCheck {
        obligation: quick_start
    }));
    assert_eq!(obligation_owners(&missing, quick_start), ["readme"]);
}

#[test]
fn deterministic_resource_mechanisms_do_not_multiply_large_model_checks() {
    let cpu = text_target("bf16", Backend::Cpu);
    let cuda = text_target("int4", Backend::Cuda);
    let mut request = input(
        Stage::PullRequest,
        vec![cpu.clone(), cuda.clone()],
        vec![profile("cpu", cpu.clone()), profile("cuda", cuda.clone())],
    );
    request.impact.areas = vec![ChangeArea::Scheduler, ChangeArea::Kv];
    request.checks = super::super::contracts::contract_check_descriptors();
    request
        .checks
        .extend(super::super::model_schedule::model_check_descriptors());
    let result = plan(&request).unwrap();
    for behavior in [
        Behavior::SchedulingProgress,
        Behavior::Cancellation,
        Behavior::CapacityAdmission,
        Behavior::KvIsolation,
        Behavior::KvRelease,
        Behavior::KvResume,
    ] {
        let (index, obligation) = result
            .obligations
            .iter()
            .enumerate()
            .find(|(_, obligation)| obligation.behavior == behavior)
            .unwrap();
        assert_eq!(obligation.layer, EvidenceLayer::Contract);
        assert_eq!(obligation.scope, ObligationScope::Global);
        assert_eq!(
            result
                .obligations
                .iter()
                .filter(|obligation| obligation.behavior == behavior)
                .count(),
            1
        );
        assert!(!obligation.checkers.is_empty());
        assert!(!result
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
    for target in [cpu, cuda] {
        for behavior in [Behavior::ModelLoad, Behavior::ModelForward] {
            let obligation = result
                .obligations
                .iter()
                .find(|obligation| {
                    obligation.behavior == behavior
                        && obligation.scope
                            == ObligationScope::Target {
                                target: target.clone(),
                            }
                })
                .unwrap();
            assert_eq!(obligation.layer, EvidenceLayer::ModelRuntime);
            assert_eq!(obligation.entrypoints, ENTRYPOINTS);
            assert!(!obligation.checkers.is_empty());
        }
    }
    assert!(result
        .obligations
        .iter()
        .filter(|obligation| obligation.layer == EvidenceLayer::ModelRuntime)
        .all(|obligation| matches!(
            obligation.behavior,
            Behavior::ModelLoad | Behavior::ModelForward
        )));
    let schedule = super::super::model_schedule::model_task_schedule(&result);
    assert!(schedule.unsupported_obligations.is_empty());
    assert!(schedule
        .runs
        .iter()
        .all(|run| run.checks == [super::super::model_tasks::ModelCheck::Basic]));
}

#[test]
fn structured_masking_keeps_its_real_contract_and_actual_http_validity_sample() {
    let target = text_target("bf16", Backend::Cpu);
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("cpu", target)],
    );
    request.impact.areas = vec![ChangeArea::Structured];
    request.checks = super::super::contracts::contract_check_descriptors();
    request
        .checks
        .extend(super::super::model_schedule::model_check_descriptors());
    let result = plan(&request).unwrap();
    let sampling: Vec<_> = result
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| obligation.behavior == Behavior::StructuredSampling)
        .collect();
    assert_eq!(sampling.len(), 1);
    assert_eq!(sampling[0].1.layer, EvidenceLayer::Contract);
    assert!(sampling[0]
        .1
        .checkers
        .contains(&"cpu-contract.structured-sampling".into()));
    assert!(!result.gaps.contains(&Gap::UnassignedCheck {
        obligation: sampling[0].0
    }));
    assert!(result
        .obligations
        .iter()
        .any(
            |obligation| obligation.behavior == Behavior::StructuredValidity
                && obligation.layer == EvidenceLayer::ModelRuntime
                && obligation.entrypoints == [Entrypoint::ServeSync, Entrypoint::ServeStream]
        ));
}

#[test]
fn protocol_bindings_cannot_substitute_for_unimplemented_device_state() {
    let target = text_target("bf16", Backend::Cuda);
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("cuda", target)],
    );
    request.impact.areas = vec![
        ChangeArea::Architecture,
        ChangeArea::Template,
        ChangeArea::Termination,
    ];
    request.checks = super::super::contracts::contract_check_descriptors();
    request
        .checks
        .extend(super::super::model_schedule::model_check_descriptors());
    let result = plan(&request).unwrap();
    let schedule = super::super::model_schedule::model_task_schedule(&result);
    for behavior in [
        Behavior::ArchitectureState,
        Behavior::ReasoningBoundaries,
        Behavior::LengthLimit,
    ] {
        let (index, obligation) = result
            .obligations
            .iter()
            .enumerate()
            .find(|(_, obligation)| {
                obligation.behavior == behavior && obligation.layer == EvidenceLayer::ModelRuntime
            })
            .unwrap();
        if behavior == Behavior::ArchitectureState {
            assert!(obligation.checkers.is_empty());
            assert!(result
                .gaps
                .contains(&Gap::UnassignedCheck { obligation: index }));
            assert!(schedule.unsupported_obligations.contains(&index));
        } else {
            // Dedicated protocol checks are implementations, not evidence that
            // they ran or that another device-state obligation was satisfied.
            assert!(!obligation.checkers.is_empty());
            assert!(!result
                .gaps
                .contains(&Gap::UnassignedCheck { obligation: index }));
            assert!(!schedule.unsupported_obligations.contains(&index));
        }
    }
}

#[test]
fn isolated_backend_change_keeps_its_forward_numerics_and_performance_unbound() {
    let targets = vec![
        text_target("f32", Backend::Cpu),
        text_target("bf16", Backend::Metal),
        text_target("bf16", Backend::Cuda),
    ];
    for (path, expected) in [
        ("crates/ferrum-kernels/src/backend/cpu.rs", Backend::Cpu),
        (
            "crates/ferrum-kernels/src/backend/cpu/vnext_ops.rs",
            Backend::Cpu,
        ),
        (
            "crates/ferrum-kernels/src/backend/cpu/vnext_runtime.rs",
            Backend::Cpu,
        ),
        (
            "crates/ferrum-kernels/src/backend/metal/mod.rs",
            Backend::Metal,
        ),
        (
            "crates/ferrum-kernels/src/backend/cuda/fused_silu_mul.rs",
            Backend::Cuda,
        ),
    ] {
        let mut request = input(
            Stage::PullRequest,
            targets.clone(),
            targets
                .iter()
                .enumerate()
                .map(|(index, target)| profile(&format!("backend-{index}"), target.clone()))
                .collect(),
        );
        request.impact = super::super::analyze_paths([path]);
        let result = plan(&request).unwrap();
        assert!(result
            .obligations
            .iter()
            .all(|obligation| obligation.behavior != Behavior::ArchitectureState));
        for (behavior, layer) in [
            (Behavior::ModelForward, EvidenceLayer::ModelRuntime),
            (Behavior::KernelNumerics, EvidenceLayer::BackendNumerics),
            (Behavior::KernelBoundaries, EvidenceLayer::BackendNumerics),
            (Behavior::Performance, EvidenceLayer::Performance),
        ] {
            let obligations: Vec<_> = result
                .obligations
                .iter()
                .enumerate()
                .filter(|(_, obligation)| {
                    obligation.behavior == behavior && obligation.layer == layer
                })
                .collect();
            assert!(!obligations.is_empty(), "missing {behavior:?}");
            for (index, obligation) in obligations {
                let ObligationScope::Target { target } = &obligation.scope else {
                    panic!("exact backend target required")
                };
                assert_eq!(target.backend, expected);
                assert!(obligation.checkers.is_empty());
                assert!(result
                    .gaps
                    .contains(&Gap::UnassignedCheck { obligation: index }));
            }
        }
        assert!(result
            .obligations
            .iter()
            .any(
                |obligation| obligation.behavior == Behavior::KernelBoundaries
                    && obligation.layer == EvidenceLayer::Contract
            ));
        assert!(result
            .selected
            .iter()
            .all(|selected| selected.profile.target.backend == expected));
        request
            .required_targets
            .retain(|target| target.backend != expected);
        assert!(plan(&request)
            .unwrap_err()
            .contains("no target in the declared regression inventory"));
    }
}

#[test]
fn backend_reach_is_per_area_and_preserves_shared_download_protocol_and_quick_starts() {
    let metal = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let other = target(
        "hybrid",
        ModelOutputProtocol::GemmaThought,
        "int4",
        Backend::Cuda,
    );
    let mut request = input(
        Stage::Release,
        vec![metal.clone(), cuda.clone(), other.clone()],
        vec![
            profile("metal-quick-start", metal.clone()),
            profile("cuda-quick-start", cuda.clone()),
            profile("other-architecture", other.clone()),
        ],
    );
    request.quick_start_profile_ids = vec!["metal-quick-start".into(), "cuda-quick-start".into()];
    request.impact = super::super::analyze_paths([
        "crates/ferrum-kernels/src/backend/metal/mod.rs",
        "crates/ferrum-types/src/reasoning.rs",
        "crates/ferrum-models/src/hf_download.rs",
    ]);
    let result = plan(&request).unwrap();
    for id in &request.quick_start_profile_ids {
        assert!(result.obligations.iter().any(|obligation| obligation.behavior == Behavior::QuickStart
            && matches!(&obligation.scope, ObligationScope::Profile { profile_id, .. } if profile_id == id)));
        assert!(selected_ids(&result).contains(&id.as_str()));
    }
    for expected in [&metal, &cuda, &other] {
        assert!(result
            .obligations
            .iter()
            .any(
                |obligation| obligation.behavior == Behavior::ReasoningBoundaries
                    && obligation.layer == EvidenceLayer::ModelRuntime
                    && obligation.scope
                        == ObligationScope::Reasoning {
                            protocol: expected.protocol,
                            backend: expected.backend,
                            execution_path: expected.execution_path.clone(),
                            reasoning_protocol: ferrum_types::ModelReasoningProtocol::PromptOpened
                        }
            ));
        assert!(result
            .obligations
            .iter()
            .any(|obligation| obligation.behavior == Behavior::ModelLoad
                && obligation.layer == EvidenceLayer::ModelRuntime
                && obligation.scope
                    == ObligationScope::Architecture {
                        architecture: expected.architecture.clone(),
                        protocol: expected.protocol,
                        execution_path: expected.execution_path.clone()
                    }));
    }
    for obligation in result.obligations.iter().filter(|obligation| {
        obligation.layer == EvidenceLayer::BackendNumerics
            || obligation.layer == EvidenceLayer::Performance
    }) {
        assert!(
            matches!(&obligation.scope, ObligationScope::Target { target } if target.backend == Backend::Metal)
        );
    }
    assert!(!result
        .obligations
        .iter()
        .any(|obligation| obligation.behavior == Behavior::ArchitectureState));
}

#[test]
fn backend_path_union_and_unknown_or_shared_kernel_reach_remain_conservative() {
    let targets = vec![
        text_target("f32", Backend::Cpu),
        text_target("bf16", Backend::Metal),
        text_target("bf16", Backend::Cuda),
    ];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| profile(&format!("backend-{index}"), target.clone()))
            .collect(),
    );
    let metal = "crates/ferrum-kernels/src/backend/metal/mod.rs";
    request.impact =
        super::super::analyze_paths([metal, "crates/ferrum-kernels/src/backend/cuda/mod.rs"]);
    // Path contributors cannot disappear because the aggregate was incomplete.
    request.impact.areas.clear();
    let result = plan(&request).unwrap();
    let numerical_targets = |result: &Plan| -> Vec<Backend> {
        result
            .obligations
            .iter()
            .filter(|obligation| obligation.behavior == Behavior::KernelNumerics)
            .filter_map(|obligation| match &obligation.scope {
                ObligationScope::Target { target } => Some(target.backend),
                _ => None,
            })
            .collect()
    };
    let mut observed = numerical_targets(&result);
    observed.sort();
    assert_eq!(observed, [Backend::Metal, Backend::Cuda]);
    let cpu = "crates/ferrum-kernels/src/backend/cpu/vnext_runtime.rs";
    request.impact =
        super::super::analyze_paths([cpu, "crates/ferrum-kernels/src/backend/cuda/mod.rs"]);
    let result = plan(&request).unwrap();
    let mut observed = numerical_targets(&result);
    observed.sort();
    assert_eq!(observed, [Backend::Cpu, Backend::Cuda]);
    for shared in [
        "crates/ferrum-kernels/src/backend/traits.rs",
        "crates/ferrum-kernels/src/backend/unreviewed/ops.rs",
        "crates/ferrum-kernels/src/metal/ops.rs",
        "unknown-runtime/forward.rs",
    ] {
        request.impact = super::super::analyze_paths([cpu, metal, shared]);
        let result = plan(&request).unwrap();
        let mut observed = numerical_targets(&result);
        observed.sort();
        assert_eq!(
            observed,
            [Backend::Cpu, Backend::Metal, Backend::Cuda],
            "{shared}"
        );
        assert!(
            result
                .obligations
                .iter()
                .any(|obligation| obligation.behavior == Behavior::ArchitectureState),
            "{shared}"
        );
    }
}

#[test]
fn template_reasoning_modes_have_distinct_runtime_representatives() {
    use ferrum_types::ModelReasoningProtocol as Reasoning;
    let target = text_target("bf16", Backend::Cpu);
    let mut profiles = Vec::new();
    for (id, capability) in [
        ("plain", Reasoning::None),
        ("opened", Reasoning::PromptOpened),
        ("generated", Reasoning::ModelGenerated),
    ] {
        let mut candidate = profile(id, target.clone());
        candidate.reasoning_protocol = capability;
        profiles.push(candidate);
    }
    let mut request = input(Stage::PullRequest, vec![target], profiles);
    request.impact.areas = vec![ChangeArea::Template];
    request.checks = super::super::contracts::contract_check_descriptors();
    request
        .checks
        .extend(super::super::model_schedule::model_check_descriptors());
    let result = plan(&request).unwrap();
    for candidate in &request.profiles {
        let (index, obligation) = result
            .obligations
            .iter()
            .enumerate()
            .find(|(_, obligation)| {
                matches!(&obligation.scope, ObligationScope::Reasoning { reasoning_protocol, .. }
                if *reasoning_protocol == candidate.reasoning_protocol)
            })
            .unwrap();
        assert_eq!(
            obligation.behavior,
            if candidate.reasoning_protocol == Reasoning::None {
                Behavior::ReasoningAbsence
            } else {
                Behavior::ReasoningBoundaries
            }
        );
        let owner = result
            .selected
            .iter()
            .find(|selected| selected.obligations.contains(&index))
            .unwrap();
        assert_eq!(
            owner.profile.reasoning_protocol,
            candidate.reasoning_protocol
        );
    }
    let schedule = super::super::model_schedule::model_task_schedule(&result);
    assert!(schedule.unsupported_obligations.is_empty());
    let plain = schedule
        .runs
        .iter()
        .find(|run| run.profile.reasoning_protocol == Reasoning::None)
        .unwrap();
    assert_eq!(plain.checks, [super::super::model_tasks::ModelCheck::Basic]);
    assert!(schedule
        .runs
        .iter()
        .filter(|run| run.profile.reasoning_protocol.supports_reasoning())
        .all(|run| run
            .checks
            .contains(&super::super::model_tasks::ModelCheck::Reasoning)));
}

#[test]
fn unknown_reasoning_capability_is_a_pre_execution_gap_only_when_affected() {
    let target = text_target("bf16", Backend::Cpu);
    let mut candidate = profile("unresolved", target.clone());
    candidate.reasoning_protocol = ferrum_types::ModelReasoningProtocol::Unknown;
    let mut request = input(Stage::PullRequest, vec![target], vec![candidate]);
    request.impact.areas = vec![ChangeArea::Template];
    let result = plan(&request).unwrap();
    assert!(result.gaps.contains(&Gap::UnknownReasoningCapability {
        profile_id: "unresolved".into()
    }));
    request.impact.areas = vec![ChangeArea::Download];
    assert!(!plan(&request)
        .unwrap()
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::UnknownReasoningCapability { .. })));
}

#[test]
fn execution_route_restriction_alone_retains_kernel_obligations() {
    let mut legacy = text_target("f32", Backend::Metal);
    legacy.execution_path = "legacy-model-executor".into();
    let production = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let targets = vec![legacy.clone(), production.clone(), cuda.clone()];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| profile(&format!("route-{index}"), target.clone()))
            .collect(),
    );
    request.impact =
        super::super::analyze_paths(["crates/ferrum-kernels/src/backend/metal/mod.rs"]);
    request.impact.paths[0].execution_paths = Some(vec![legacy.execution_path.clone()]);
    let result = plan(&request).unwrap();
    for (behavior, layer) in [
        (Behavior::ModelForward, EvidenceLayer::ModelRuntime),
        (Behavior::KernelNumerics, EvidenceLayer::BackendNumerics),
        (Behavior::KernelBoundaries, EvidenceLayer::BackendNumerics),
        (Behavior::Performance, EvidenceLayer::Performance),
    ] {
        let obligations: Vec<_> = result
            .obligations
            .iter()
            .enumerate()
            .filter(|(_, obligation)| obligation.behavior == behavior && obligation.layer == layer)
            .collect();
        assert!(!obligations.is_empty(), "{behavior:?}");
        for (index, obligation) in obligations {
            assert_eq!(
                obligation.scope,
                ObligationScope::Target {
                    target: legacy.clone()
                }
            );
            assert!(result
                .gaps
                .contains(&Gap::UnassignedCheck { obligation: index }));
        }
    }
    // Scope restrictions cannot silently succeed when the inventory has only
    // the independent production route, or names a nonexistent/empty route.
    request.required_targets.retain(|target| target != &legacy);
    assert!(plan(&request)
        .unwrap_err()
        .contains("no target in the declared regression inventory"));
    request.required_targets = targets;
    for routes in [
        vec![],
        vec![String::new()],
        vec!["unknown-route".into()],
        vec![legacy.execution_path.clone(), legacy.execution_path.clone()],
    ] {
        request.impact.paths[0].execution_paths = Some(routes);
        assert!(plan(&request).is_err());
    }
}

#[test]
fn execution_reach_unions_each_path_without_narrowing_shared_areas_or_quick_starts() {
    let mut legacy = text_target("f32", Backend::Metal);
    legacy.execution_path = "legacy-model-executor".into();
    let production = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let targets = vec![legacy.clone(), production.clone(), cuda.clone()];
    let path = "crates/ferrum-kernels/src/backend/metal/mod.rs";
    let mut request = input(
        Stage::Release,
        targets.clone(),
        vec![
            profile("legacy", legacy.clone()),
            profile("production", production.clone()),
            profile("cuda", cuda.clone()),
        ],
    );
    request.quick_start_profile_ids = vec!["production".into(), "cuda".into()];
    request.impact = super::super::analyze_paths([
        path,
        "crates/ferrum-types/src/reasoning.rs",
        "crates/ferrum-cli/src/source_resolver.rs",
    ]);
    request
        .impact
        .paths
        .iter_mut()
        .find(|entry| entry.path == path)
        .unwrap()
        .execution_paths = Some(vec![legacy.execution_path.clone()]);
    assert_eq!(
        area_targets(&request.impact, ChangeArea::Kernel, &targets).unwrap(),
        [legacy.clone()]
    );
    for area in [
        ChangeArea::Template,
        ChangeArea::Download,
        ChangeArea::Scheduler,
    ] {
        assert_eq!(
            area_targets(&request.impact, area, &targets).unwrap(),
            targets
        );
    }
    let result = plan(&request).unwrap();
    for id in &request.quick_start_profile_ids {
        assert!(result.obligations.iter().any(|obligation| obligation.behavior == Behavior::QuickStart
            && matches!(&obligation.scope, ObligationScope::Profile { profile_id, .. } if profile_id == id)));
    }
    for additional in [
        "crates/ferrum-kernels/src/backend/metal/vnext_runtime.rs",
        "crates/ferrum-kernels/src/backend/metal/quant.rs",
    ] {
        request.impact = super::super::analyze_paths([path, additional]);
        request
            .impact
            .paths
            .iter_mut()
            .find(|entry| entry.path == path)
            .unwrap()
            .execution_paths = Some(vec![legacy.execution_path.clone()]);
        assert_eq!(
            area_targets(&request.impact, ChangeArea::Kernel, &targets).unwrap(),
            [legacy.clone(), production.clone()]
        );
    }
    // Backend reach and execution reach are intersected for each contributor
    // before union, so CUDA cannot broaden the unrelated Metal production route.
    request.impact =
        super::super::analyze_paths([path, "crates/ferrum-kernels/src/backend/cuda/mod.rs"]);
    request
        .impact
        .paths
        .iter_mut()
        .find(|entry| entry.path == path)
        .unwrap()
        .execution_paths = Some(vec![legacy.execution_path.clone()]);
    assert_eq!(
        area_targets(&request.impact, ChangeArea::Kernel, &targets).unwrap(),
        [legacy.clone(), cuda]
    );
    for unknown in [
        "unknown/forward.rs",
        "crates/ferrum-kernels/src/backend/unreviewed/ops.rs",
    ] {
        request.impact = super::super::analyze_paths([path, unknown]);
        request
            .impact
            .paths
            .iter_mut()
            .find(|entry| entry.path == path)
            .unwrap()
            .execution_paths = Some(vec![legacy.execution_path.clone()]);
        assert_eq!(
            area_targets(&request.impact, ChangeArea::Kernel, &targets).unwrap(),
            targets
        );
    }
}

fn submission_scope_set(scopes: &[ObligationScope]) -> BTreeSet<String> {
    scopes
        .iter()
        .map(|scope| serde_json::to_string(scope).unwrap())
        .collect()
}

fn mark_submission_proof(impact: &mut Impact) {
    let entry = impact
        .paths
        .iter_mut()
        .find(|entry| entry.path == "crates/ferrum-kernels/src/backend/metal/mod.rs")
        .unwrap();
    entry.areas = vec![ChangeArea::BackendSubmission];
    entry.execution_paths = Some(vec!["legacy-model-executor".into()]);
    impact.areas = ChangeArea::ALL
        .into_iter()
        .filter(|area| impact.paths.iter().any(|path| path.areas.contains(area)))
        .collect();
}

#[test]
fn submission_completion_is_shared_by_route_while_model_and_performance_targets_remain_exact() {
    let mut floating = text_target("f32", Backend::Metal);
    floating.execution_path = "legacy-model-executor".into();
    let mut quantized = floating.clone();
    quantized.architecture = "attention-moe".into();
    quantized.precision = "gguf-q4_k_m".into();
    let production = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let targets = vec![floating.clone(), quantized.clone(), production, cuda];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| profile(&format!("submission-{index}"), target.clone()))
            .collect(),
    );
    request.impact =
        super::super::analyze_paths(["crates/ferrum-kernels/src/backend/metal/mod.rs"]);
    mark_submission_proof(&mut request.impact);
    let result = plan(&request).unwrap();
    let submission: Vec<_> = result
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| obligation.behavior == Behavior::SubmissionCompletion)
        .collect();
    let scopes: Vec<_> = submission
        .iter()
        .map(|(_, obligation)| obligation.scope.clone())
        .collect();
    assert_eq!(
        scopes,
        [ObligationScope::ExecutionPath {
            backend: Backend::Metal,
            execution_path: floating.execution_path.clone()
        }]
    );
    for (index, obligation) in submission {
        assert_eq!(obligation.layer, EvidenceLayer::BackendNumerics);
        assert!(obligation.entrypoints.is_empty());
        assert!(obligation.checkers.is_empty());
        assert!(result
            .gaps
            .contains(&Gap::UnassignedCheck { obligation: index }));
    }
    for (behavior, layer) in [
        (Behavior::ModelLoad, EvidenceLayer::ModelRuntime),
        (Behavior::ModelForward, EvidenceLayer::ModelRuntime),
        (Behavior::Performance, EvidenceLayer::Performance),
    ] {
        let scopes: Vec<_> = result
            .obligations
            .iter()
            .filter(|obligation| obligation.behavior == behavior && obligation.layer == layer)
            .map(|obligation| obligation.scope.clone())
            .collect();
        assert_eq!(
            submission_scope_set(&scopes),
            submission_scope_set(&[
                ObligationScope::Target {
                    target: floating.clone()
                },
                ObligationScope::Target {
                    target: quantized.clone()
                }
            ]),
            "{behavior:?}"
        );
    }
    assert!(!result.obligations.iter().any(|obligation| matches!(
        obligation.behavior,
        Behavior::KernelNumerics | Behavior::KernelBoundaries | Behavior::ArchitectureState
    )));
}

#[test]
fn submission_proof_cannot_replace_any_unproven_kernel_contributor() {
    let mut floating = text_target("f32", Backend::Metal);
    floating.execution_path = "legacy-model-executor".into();
    let mut quantized = floating.clone();
    quantized.precision = "gguf-q4_k_m".into();
    let production = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let targets = vec![floating, quantized, production, cuda];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| profile(&format!("mixed-{index}"), target.clone()))
            .collect(),
    );
    for (additional, affected_backend) in [
        (
            "crates/ferrum-kernels/src/backend/metal/quant.rs",
            Some(Backend::Metal),
        ),
        (
            "crates/ferrum-kernels/src/backend/cuda/mod.rs",
            Some(Backend::Cuda),
        ),
        ("crates/ferrum-kernels/src/backend/traits.rs", None),
        ("crates/ferrum-cli/src/commands/run.rs", None),
        ("unreviewed-runtime/forward.rs", None),
    ] {
        request.impact = super::super::analyze_paths([
            "crates/ferrum-kernels/src/backend/metal/mod.rs",
            additional,
        ]);
        mark_submission_proof(&mut request.impact);
        let result = plan(&request).unwrap();
        for behavior in [Behavior::KernelNumerics, Behavior::KernelBoundaries] {
            let observed: Vec<_> = result
                .obligations
                .iter()
                .filter(|obligation| {
                    obligation.behavior == behavior
                        && obligation.layer == EvidenceLayer::BackendNumerics
                })
                .map(|obligation| obligation.scope.clone())
                .collect();
            let expected: Vec<_> = targets
                .iter()
                .filter(|target| affected_backend.is_none_or(|backend| target.backend == backend))
                .cloned()
                .map(|target| ObligationScope::Target { target })
                .collect();
            assert_eq!(
                submission_scope_set(&observed),
                submission_scope_set(&expected),
                "{additional}: {behavior:?}"
            );
        }
        assert!(result
            .obligations
            .iter()
            .any(|obligation| obligation.behavior == Behavior::SubmissionCompletion));
        assert!(result
            .obligations
            .iter()
            .any(
                |obligation| obligation.behavior == Behavior::KernelBoundaries
                    && obligation.layer == EvidenceLayer::Contract
            ));
    }
}

#[test]
fn submission_scope_distinguishes_backends_and_independent_routes() {
    let mut legacy = text_target("f32", Backend::Metal);
    legacy.execution_path = "legacy-model-executor".into();
    let mut quantized = legacy.clone();
    quantized.precision = "gguf-q4_k_m".into();
    quantized.architecture = "different-architecture".into();
    let production = text_target("bf16", Backend::Metal);
    let cuda = text_target("bf16", Backend::Cuda);
    let targets = vec![
        legacy.clone(),
        quantized.clone(),
        production.clone(),
        cuda.clone(),
    ];
    let mut request = input(
        Stage::PullRequest,
        targets.clone(),
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| profile(&format!("scope-{index}"), target.clone()))
            .collect(),
    );
    request.impact.areas = vec![ChangeArea::BackendSubmission];
    let result = plan(&request).unwrap();
    let scopes: Vec<_> = result
        .obligations
        .iter()
        .filter(|obligation| obligation.behavior == Behavior::SubmissionCompletion)
        .map(|obligation| obligation.scope.clone())
        .collect();
    assert_eq!(
        scopes,
        [
            ObligationScope::ExecutionPath {
                backend: Backend::Metal,
                execution_path: legacy.execution_path.clone()
            },
            ObligationScope::ExecutionPath {
                backend: Backend::Metal,
                execution_path: production.execution_path.clone()
            },
            ObligationScope::ExecutionPath {
                backend: Backend::Cuda,
                execution_path: cuda.execution_path.clone()
            },
        ]
    );
    assert!(scope_matches(&scopes[0], &quantized, None));
    assert!(!scope_matches(&scopes[0], &production, None));
    let mut other_backend = legacy.clone();
    other_backend.backend = Backend::Cuda;
    assert!(!scope_matches(&scopes[0], &other_backend, None));
    let wire = serde_json::to_value(&scopes[0]).unwrap();
    assert_eq!(wire["kind"], "execution_path");
    assert_eq!(
        serde_json::from_value::<ObligationScope>(wire).unwrap(),
        scopes[0]
    );
}

#[test]
fn submission_latency_sample_never_narrows_an_independent_kernel_requirement() {
    let target = text_target("gguf-q4_k_m", Backend::Metal);
    let scope = ObligationScope::Target { target };
    for narrow_first in [true, false] {
        let mut result = plan(&input(Stage::PullRequest, Vec::new(), Vec::new())).unwrap();
        result.obligations.clear();
        let narrow = vec![Entrypoint::ServeStream];
        let broad = ENTRYPOINTS.to_vec();
        let ordered = if narrow_first {
            [narrow, broad]
        } else {
            [broad, narrow]
        };
        for entrypoints in ordered {
            add_with_entrypoints(
                &mut result,
                Behavior::Performance,
                EvidenceLayer::Performance,
                scope.clone(),
                "changed execution path",
                entrypoints,
            );
        }
        assert_eq!(result.obligations.len(), 1);
        assert_eq!(result.obligations[0].entrypoints, ENTRYPOINTS);
    }
    let mut target = text_target("gguf-q4_k_m", Backend::Metal);
    target.execution_path = "legacy-model-executor".into();
    let mut request = input(
        Stage::PullRequest,
        vec![target.clone()],
        vec![profile("sample", target)],
    );
    request.impact =
        super::super::analyze_paths(["crates/ferrum-kernels/src/backend/metal/mod.rs"]);
    mark_submission_proof(&mut request.impact);
    let result = plan(&request).unwrap();
    for obligation in &result.obligations {
        if obligation.behavior == Behavior::Performance {
            assert_eq!(obligation.entrypoints, [Entrypoint::ServeStream]);
        } else if matches!(
            obligation.behavior,
            Behavior::ModelLoad | Behavior::ModelForward
        ) {
            assert_eq!(obligation.entrypoints, ENTRYPOINTS);
        }
    }
}

#[test]
fn release_performance_deferral_preserves_every_correctness_obligation() {
    let target = text_target("bf16", Backend::Cuda);
    let mut request = input(
        Stage::Release,
        vec![target.clone()],
        vec![profile("quick", target)],
    );
    request.quick_start_profile_ids.push("quick".into());
    request.impact.areas.push(ChangeArea::BackendSubmission);
    let required = plan(&request).unwrap();
    let performance: Vec<_> = required
        .obligations
        .iter()
        .filter(|o| o.layer == EvidenceLayer::Performance)
        .cloned()
        .collect();
    let correctness: Vec<_> = required
        .obligations
        .iter()
        .filter(|o| o.layer != EvidenceLayer::Performance)
        .map(|o| (o.behavior, o.layer, o.scope.clone(), o.entrypoints.clone()))
        .collect();
    assert!(!performance.is_empty());
    assert!(correctness
        .iter()
        .any(|o| o.1 == EvidenceLayer::BackendNumerics));
    request.release_performance = ReleasePerformancePolicy::Deferred {
        reason: "Complete measurements after the usability release".into(),
    };
    let deferred = plan(&request).unwrap();
    deferred.validate_performance_deferral().unwrap();
    let actual: Vec<_> = deferred
        .obligations
        .iter()
        .map(|o| (o.behavior, o.layer, o.scope.clone(), o.entrypoints.clone()))
        .collect();
    assert_eq!(actual, correctness);
    let deferred_obligations = &deferred.deferred_performance.as_ref().unwrap().obligations;
    assert_eq!(deferred_obligations.len(), performance.len());
    for (actual, expected) in deferred_obligations.iter().zip(&performance) {
        assert_eq!(
            (
                actual.behavior,
                actual.layer,
                &actual.scope,
                &actual.entrypoints
            ),
            (
                expected.behavior,
                expected.layer,
                &expected.scope,
                &expected.entrypoints
            )
        );
    }
    assert!(
        super::super::performance::performance_task_schedule(&deferred)
            .runs
            .is_empty()
    );
    assert!(
        deferred
            .gaps
            .iter()
            .any(|gap| matches!(gap, Gap::UnassignedCheck { .. })),
        "missing correctness checkers must remain gaps"
    );
    let mut invalid = deferred.clone();
    invalid.deferred_performance.as_mut().unwrap().obligations[0].layer =
        EvidenceLayer::BackendNumerics;
    assert!(invalid.validate_performance_deferral().is_err());
    request.stage = Stage::PullRequest;
    let pr = plan(&request).unwrap();
    assert!(pr.deferred_performance.is_none());
    assert!(pr
        .obligations
        .iter()
        .any(|o| o.layer == EvidenceLayer::Performance));
    request.release_performance = ReleasePerformancePolicy::Deferred { reason: " ".into() };
    assert!(plan(&request).is_err());
}

#[test]
fn explicit_release_models_cannot_be_replaced_by_a_cheaper_same_family_sample() {
    let t = text_target("gguf-q4_k_m", Backend::Cuda);
    let mut request = input(
        Stage::Release,
        vec![t.clone()],
        vec![profile("quick", t.clone()), profile("large", t)],
    );
    request.quick_start_profile_ids = vec!["quick".into()];
    request.release_profile_ids = vec!["large".into()];
    let result = plan(&request).unwrap();
    assert_eq!(selected_ids(&result), ["large", "quick"]);
    assert!(result.obligations.iter().any(|o| o.behavior == Behavior::ModelForward && matches!(&o.scope, ObligationScope::Profile { profile_id, .. } if profile_id == "large")));
    request.profiles[1].available = false;
    assert!(plan(&request)
        .unwrap()
        .gaps
        .contains(&Gap::MissingReleaseProfile {
            profile_id: "large".into()
        }));
    request.stage = Stage::PullRequest;
    assert!(!plan(&request)
        .unwrap()
        .gaps
        .iter()
        .any(|g| matches!(g, Gap::MissingReleaseProfile { .. })));
    request.stage = Stage::Release;
    request.release_profile_ids.push("large".into());
    assert!(plan(&request).is_err());
}
