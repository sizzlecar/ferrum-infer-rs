//! Deterministic coverage planning; assignments and estimates are never execution evidence.
use super::types::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const ENTRYPOINTS: &[Entrypoint] = &[
    Entrypoint::Run,
    Entrypoint::ServeSync,
    Entrypoint::ServeStream,
];

fn nonblank(value: &str) -> bool {
    !value.trim().is_empty() && value == value.trim()
}

fn concrete_dimension(value: &str) -> bool {
    if !nonblank(value) {
        return false;
    }
    let normalized = value.to_ascii_lowercase().replace('_', "-");
    // Reject unresolved declarations, not unfamiliar but concrete capability names.
    // This is not a model allowlist or a claim that a declared target was executed.
    !matches!(
        normalized.as_str(),
        "auto" | "default" | "native" | "any" | "none" | "n/a" | "*" | "mixed" | "quantized"
    ) && !normalized.starts_with("model-native")
        && !normalized.starts_with("model-default")
        && !normalized.starts_with("model-dependent")
        && !normalized
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|part| {
                matches!(
                    part,
                    "unknown"
                        | "unresolved"
                        | "unspecified"
                        | "placeholder"
                        | "tbd"
                        | "todo"
                        | "pending"
                )
            })
}

fn valid_target(target: &ExecutionTarget) -> bool {
    concrete_dimension(&target.architecture)
        && concrete_dimension(&target.precision)
        && concrete_dimension(&target.execution_path)
}

fn entrypoint_independent(layer: EvidenceLayer) -> bool {
    matches!(
        layer,
        EvidenceLayer::Compilation | EvidenceLayer::BackendNumerics | EvidenceLayer::Installation
    )
}

fn behavior_entrypoints(behavior: Behavior, layer: EvidenceLayer) -> Vec<Entrypoint> {
    if entrypoint_independent(layer) {
        return Vec::new();
    }
    // `run` exposes neither response_format nor tool definitions/selection and
    // constructs requests with api_request: None. These are HTTP product flows.
    match behavior {
        Behavior::StructuredSampling
        | Behavior::StructuredValidity
        | Behavior::ToolSelection
        | Behavior::ToolHandoff
        | Behavior::ToolContinuation => vec![Entrypoint::ServeSync, Entrypoint::ServeStream],
        _ => ENTRYPOINTS.to_vec(),
    }
}

fn scope_matches(scope: &ObligationScope, target: &ExecutionTarget, id: Option<&str>) -> bool {
    match scope {
        ObligationScope::Global => false,
        ObligationScope::Backend { backend } => *backend == target.backend,
        ObligationScope::ExecutionPath {
            backend,
            execution_path,
        } => *backend == target.backend && execution_path == &target.execution_path,
        ObligationScope::Architecture {
            architecture,
            protocol,
            execution_path,
        } => {
            architecture == &target.architecture
                && protocol == &target.protocol
                && execution_path == &target.execution_path
        }
        ObligationScope::Protocol {
            protocol,
            backend,
            execution_path,
        } => {
            protocol == &target.protocol
                && *backend == target.backend
                && execution_path == &target.execution_path
        }
        ObligationScope::Reasoning {
            protocol,
            backend,
            execution_path,
            ..
        } => {
            protocol == &target.protocol
                && *backend == target.backend
                && execution_path == &target.execution_path
        }
        ObligationScope::Target { target: expected } => expected == target,
        ObligationScope::Profile {
            profile_id,
            target: expected,
        } => Some(profile_id.as_str()) == id && expected == target,
    }
}

fn requires_model(obligation: &Obligation) -> bool {
    matches!(
        obligation.layer,
        EvidenceLayer::ModelRuntime | EvidenceLayer::Performance
    )
}

fn add(
    plan: &mut Plan,
    behavior: Behavior,
    layer: EvidenceLayer,
    scope: ObligationScope,
    reason: &str,
) {
    add_with_entrypoints(
        plan,
        behavior,
        layer,
        scope,
        reason,
        behavior_entrypoints(behavior, layer),
    );
}

fn add_with_entrypoints(
    plan: &mut Plan,
    behavior: Behavior,
    layer: EvidenceLayer,
    scope: ObligationScope,
    reason: &str,
    entrypoints: Vec<Entrypoint>,
) {
    if let Some(existing) = plan
        .obligations
        .iter_mut()
        .find(|item| item.behavior == behavior && item.layer == layer && item.scope == scope)
    {
        // A narrow contributor must not erase another path's broader reach.
        existing.entrypoints.extend(entrypoints);
        existing.entrypoints.sort();
        existing.entrypoints.dedup();
        return;
    }
    plan.obligations.push(Obligation {
        behavior,
        layer,
        entrypoints,
        scope,
        reason: reason.into(),
        checkers: Vec::new(),
    });
}

fn architectures(targets: &[ExecutionTarget]) -> Vec<ObligationScope> {
    let mut result = Vec::new();
    for target in targets {
        let scope = ObligationScope::Architecture {
            architecture: target.architecture.clone(),
            protocol: target.protocol,
            execution_path: target.execution_path.clone(),
        };
        if !result.contains(&scope) {
            result.push(scope);
        }
    }
    result
}

fn protocols(targets: &[ExecutionTarget]) -> Vec<ObligationScope> {
    let mut result = Vec::new();
    for target in targets {
        let scope = ObligationScope::Protocol {
            protocol: target.protocol,
            backend: target.backend,
            execution_path: target.execution_path.clone(),
        };
        if !result.contains(&scope) {
            result.push(scope);
        }
    }
    result
}

fn backends(targets: &[ExecutionTarget]) -> Vec<ObligationScope> {
    targets
        .iter()
        .map(|target| target.backend)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|backend| ObligationScope::Backend { backend })
        .collect()
}

fn add_for_scopes(
    plan: &mut Plan,
    behaviors: &[Behavior],
    layer: EvidenceLayer,
    scopes: &[ObligationScope],
    reason: &str,
) {
    for scope in scopes {
        for behavior in behaviors {
            add(plan, *behavior, layer, scope.clone(), reason);
        }
    }
}

/// Union the reach of paths contributing this area, not the reach of the whole
/// diff. Missing path attribution and any shared/unrecognized contributor retain
/// every target. Explicit unknown changes also defeat narrowing, even if their
/// precomputed area list was incomplete.
fn area_targets(
    impact: &Impact,
    area: ChangeArea,
    targets: &[ExecutionTarget],
) -> Result<Vec<ExecutionTarget>, String> {
    if area == ChangeArea::Validation || !impact.unknown_paths.is_empty() {
        return Ok(targets.to_vec());
    }
    let contributors: Vec<_> = impact
        .paths
        .iter()
        .filter(|path| path.areas.contains(&area))
        .collect();
    if contributors.is_empty() {
        return Ok(targets.to_vec());
    }
    let mut selected = BTreeSet::new();
    for path in contributors {
        let backend = super::impact::path_backend(&path.path);
        if let Some(routes) = &path.execution_paths {
            if routes.is_empty()
                || routes.iter().any(|route| route.trim().is_empty())
                || routes.iter().collect::<BTreeSet<_>>().len() != routes.len()
            {
                return Err(format!("invalid execution-path reach for {}", path.path));
            }
            for route in routes {
                if !targets.iter().any(|target| {
                    backend.is_none_or(|backend| target.backend == backend)
                        && target.execution_path == *route
                }) {
                    return Err(format!("changed path {} execution route {route} has no target in the declared regression inventory", path.path));
                }
            }
        }
        let mut matched = false;
        for (index, target) in targets.iter().enumerate() {
            if backend.is_none_or(|backend| target.backend == backend)
                && path
                    .execution_paths
                    .as_ref()
                    .is_none_or(|routes| routes.contains(&target.execution_path))
            {
                selected.insert(index);
                matched = true;
            }
        }
        if !matched {
            return Err(format!(
                "changed path {} backend has no target in the declared regression inventory",
                path.path
            ));
        }
    }
    Ok(targets
        .iter()
        .enumerate()
        .filter(|(index, _)| selected.contains(index))
        .map(|(_, target)| target.clone())
        .collect())
}

fn area_obligations(plan: &mut Plan, area: ChangeArea, targets: &[ExecutionTarget]) {
    use Behavior::*;
    use ChangeArea::*;
    use EvidenceLayer::{BackendNumerics, Contract, ModelRuntime};
    let (contracts, runtime): (&[Behavior], &[Behavior]) = match area {
        Download => (
            &[
                SourceClosure,
                SourceRevision,
                DownloadRecovery,
                CacheCompleteness,
            ],
            &[ModelLoad],
        ),
        Template => (
            &[TemplateHistory, ProtocolFraming, ReasoningBoundaries],
            &[TemplateHistory, ProtocolFraming],
        ),
        Termination => (
            &[UserStop, NaturalEnd, LengthLimit],
            &[UserStop, NaturalEnd, LengthLimit],
        ),
        Structured => (
            // Competing-logit production-engine contracts prove token masking.
            // A model that follows the JSON prompt cannot establish that mechanism;
            // its actual sync/SSE result still must satisfy the requested schema.
            &[StructuredSampling, StructuredValidity],
            &[StructuredValidity],
        ),
        Tools => (
            &[ToolSelection, ToolHandoff, ToolContinuation],
            &[ToolSelection, ToolHandoff, ToolContinuation],
        ),
        Scheduler => (
            // Deterministic queue/production-engine assertions own waiting,
            // cancellation and admission. A normal model answer does not prove
            // these internal mechanisms, even when repeated on every entrypoint.
            &[SchedulingProgress, Cancellation, CapacityAdmission],
            &[ModelLoad, ModelForward],
        ),
        Kv => (
            // CPU contracts check isolation, release and recomputation after
            // supported preemption. Selected models verify integration; these
            // checks cannot certify GPU KV arithmetic or swapping.
            &[KvIsolation, KvRelease, KvResume],
            &[ModelLoad, ModelForward],
        ),
        BackendSubmission => (&[], &[ModelLoad, ModelForward]),
        ChangeArea::WeightMaterialization => (
            &[Behavior::WeightMaterialization],
            &[ModelLoad, ModelForward],
        ),
        Kernel => (&[KernelBoundaries], &[ModelForward]),
        Architecture => (
            &[ModelLoad, ArchitectureState],
            &[ModelLoad, ModelForward, ArchitectureState],
        ),
        Build => (&[], &[ModelLoad]),
        ChangeArea::Observability => (&[Behavior::Observability], &[Behavior::Observability]),
        Validation => (&[], &[]),
    };
    let reason = if area == ChangeArea::Observability {
        "shared request metadata and instrumentation: contract checks cover propagation, sink completion and errors; runtime uses one representative per backend, not all operators/layouts".into()
    } else {
        format!("affected component: {area:?}")
    };
    add_for_scopes(
        plan,
        contracts,
        Contract,
        &[ObligationScope::Global],
        &reason,
    );
    let scopes = match area {
        Kernel
        | BackendSubmission
        | ChangeArea::WeightMaterialization
        | Architecture
        | Scheduler
        | Kv => targets
            .iter()
            .cloned()
            .map(|target| ObligationScope::Target { target })
            .collect(),
        Template | Termination | Structured | Tools => protocols(targets),
        Download => architectures(targets),
        Build | ChangeArea::Observability => backends(targets),
        Validation => Vec::new(),
    };
    add_for_scopes(plan, runtime, ModelRuntime, &scopes, &reason);
    if area == BackendSubmission {
        let routes: Vec<_> = targets
            .iter()
            .map(|target| (target.backend, target.execution_path.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|(backend, execution_path)| ObligationScope::ExecutionPath {
                backend,
                execution_path,
            })
            .collect();
        add_for_scopes(
            plan,
            &[SubmissionCompletion],
            BackendNumerics,
            &routes,
            &reason,
        );
    }
    if matches!(area, Kernel | Architecture) {
        let exact: Vec<_> = targets
            .iter()
            .cloned()
            .map(|target| ObligationScope::Target { target })
            .collect();
        add_for_scopes(
            plan,
            &[KernelNumerics, KernelBoundaries],
            BackendNumerics,
            &exact,
            &reason,
        );
    }
    if matches!(area, Kernel | BackendSubmission | Architecture) {
        let exact: Vec<_> = targets
            .iter()
            .cloned()
            .map(|target| ObligationScope::Target { target })
            .collect();
        for scope in exact {
            // The proven change only alters shared backend submission. One
            // measured streaming path exercises its latency; model correctness
            // still requires run and both HTTP modes above. Unproven kernel or
            // architecture edits retain their independent entrypoint reach.
            let entrypoints = if area == BackendSubmission {
                vec![Entrypoint::ServeStream]
            } else {
                behavior_entrypoints(Performance, EvidenceLayer::Performance)
            };
            add_with_entrypoints(
                plan,
                Performance,
                EvidenceLayer::Performance,
                scope,
                &reason,
                entrypoints,
            );
        }
    }
    if matches!(area, Download | Build) {
        add_for_scopes(
            plan,
            &[Installation],
            EvidenceLayer::Installation,
            &backends(targets),
            &reason,
        );
    }
}

fn estimate_cmp(left: &ModelProfile, right: &ModelProfile) -> Ordering {
    let duration =
        |profile: &ModelProfile| profile.estimate.as_ref().and_then(CostEstimate::total_ms);
    let known_first = |left: Option<u64>, right: Option<u64>| match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    let duration_order = known_first(duration(left), duration(right));
    if duration_order != Ordering::Equal {
        return duration_order;
    }
    let price = |profile: &ModelProfile| {
        profile.estimate.as_ref().and_then(|estimate| {
            let currency = estimate
                .currency
                .as_deref()
                .filter(|currency| nonblank(currency))?;
            Some((currency.to_owned(), estimate.cost_microunits()?))
        })
    };
    let price_order = match (price(left), price(right)) {
        (Some((left_currency, left)), Some((right_currency, right)))
            if left_currency == right_currency =>
        {
            left.cmp(&right)
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        _ => Ordering::Equal,
    };
    price_order.then_with(|| left.id.cmp(&right.id))
}

fn complete_model_binding(
    checks: &[CheckDescriptor],
    obligation: &Obligation,
    target: &ExecutionTarget,
) -> bool {
    let candidates: Vec<_> = checks
        .iter()
        .filter(|check| {
            check.behavior == obligation.behavior
                && check.layer == obligation.layer
                && check
                    .target
                    .as_ref()
                    .is_none_or(|declared| declared == target)
        })
        .collect();
    !candidates.is_empty()
        && obligation.entrypoints.iter().all(|entrypoint| {
            candidates
                .iter()
                .any(|check| check.entrypoints.contains(entrypoint))
        })
}

fn record_cost(plan: &mut Plan) -> Result<(), String> {
    let mut currencies = BTreeMap::<String, u64>::new();
    for selected in &plan.selected {
        let profile = &selected.profile;
        if let Some(estimate) = &profile.estimate {
            plan.cost.known_billable_ms = plan
                .cost
                .known_billable_ms
                .checked_add(estimate.billable_ms)
                .ok_or("selected profile billable duration sum overflow")?;
        }
        match profile.estimate.as_ref().and_then(CostEstimate::total_ms) {
            Some(duration) => {
                plan.cost.known_total_ms = plan
                    .cost
                    .known_total_ms
                    .checked_add(duration)
                    .ok_or("selected profile duration sum overflow")?
            }
            None => plan.cost.unknown_duration_profiles.push(profile.id.clone()),
        }
        let price = profile.estimate.as_ref().and_then(|estimate| {
            let currency = estimate
                .currency
                .as_deref()
                .filter(|currency| nonblank(currency))?;
            Some((currency, estimate.cost_microunits()?))
        });
        if let Some((currency, cost)) = price {
            let sum = currencies.entry(currency.into()).or_default();
            *sum = sum
                .checked_add(cost)
                .ok_or("selected profile price sum overflow")?;
        } else {
            plan.cost.unknown_price_profiles.push(profile.id.clone());
        }
    }
    plan.cost.estimated_costs = currencies
        .into_iter()
        .map(|(currency, microunits)| CurrencyEstimate {
            currency,
            microunits,
        })
        .collect();
    Ok(())
}

/// Plan coverage from declared capabilities and complete change impact. Gaps are
/// retained for review; a successful return means a plan was made, not a release passed.
pub fn plan(input: &PlanInput) -> Result<Plan, String> {
    let mut result = Plan {
        stage: input.stage,
        impact: input.impact.clone(),
        obligations: Vec::new(),
        selected: Vec::new(),
        omitted: Vec::new(),
        gaps: Vec::new(),
        cost: PlanCost::default(),
    };
    let mut ids = BTreeSet::new();
    for profile in &input.profiles {
        if !nonblank(&profile.id) || !nonblank(&profile.model) {
            return Err("profile id and model must be nonblank, unpadded strings".into());
        }
        if !ids.insert(&profile.id) {
            return Err(format!("duplicate profile id: {}", profile.id));
        }
        if let Some(estimate) = &profile.estimate {
            let reason = if estimate.total_ms().is_none() {
                Some("duration arithmetic overflow")
            } else if estimate
                .currency
                .as_deref()
                .is_some_and(|currency| !nonblank(currency))
            {
                Some("currency must be nonblank and unpadded")
            } else if estimate.currency.is_some()
                && estimate.hourly_rate_microunits.is_some()
                && estimate.fixed_cost_microunits.is_some()
                && estimate.cost_microunits().is_none()
            {
                Some("price arithmetic overflow")
            } else {
                None
            };
            if let Some(reason) = reason {
                result.gaps.push(Gap::InvalidEstimate {
                    profile_id: profile.id.clone(),
                    reason: reason.into(),
                });
            }
        }
    }
    let mut check_ids = BTreeSet::new();
    for check in &input.checks {
        if !nonblank(&check.id) || !check_ids.insert(&check.id) {
            return Err("checker ids must be nonblank and unique".into());
        }
        if (check.entrypoints.is_empty() && !entrypoint_independent(check.layer))
            || check.entrypoints.iter().collect::<BTreeSet<_>>().len() != check.entrypoints.len()
        {
            return Err(format!(
                "checker {} must declare unique entrypoints; product-flow checks cannot omit them",
                check.id
            ));
        }
        if check
            .target
            .as_ref()
            .is_some_and(|target| !valid_target(target))
        {
            return Err(format!(
                "checker {} has invalid target dimensions",
                check.id
            ));
        }
    }
    let mut targets = Vec::new();
    for target in &input.required_targets {
        if valid_target(target) {
            if !targets.contains(target) {
                targets.push(target.clone());
            }
        } else {
            result.gaps.push(Gap::InvalidTarget {
                target: target.clone(),
                reason: "inventory dimensions must be nonblank and unpadded".into(),
            });
        }
    }
    // Canonical serialization gives deterministic traversal without imposing ordering
    // requirements on a product protocol enum or silently collapsing its dimensions.
    targets.sort_by_cached_key(|target| serde_json::to_string(target).expect("target serializes"));
    for profile in &input.profiles {
        if !valid_target(&profile.target) {
            result.gaps.push(Gap::InvalidTarget {
                target: profile.target.clone(),
                reason: format!("invalid profile dimensions: {}", profile.id),
            });
        }
    }
    let mut areas: BTreeSet<_> = input.impact.areas.iter().copied().collect();
    areas.extend(
        input
            .impact
            .paths
            .iter()
            .flat_map(|path| path.areas.iter().copied()),
    );
    for path in &input.impact.unknown_paths {
        result.gaps.push(Gap::UnmappedChange { path: path.clone() });
        areas.extend(ChangeArea::ALL);
    }
    if input.impact.product_contract_changed {
        result.gaps.push(Gap::ProductContractReview);
    }
    let active = input.stage != Stage::PullRequest
        || !areas.is_empty()
        || input.impact.product_contract_changed;
    if active && targets.is_empty() {
        result.gaps.push(Gap::EmptyInventory);
    }
    if input.stage == Stage::Release && input.quick_start_profile_ids.is_empty() {
        return Err("release catalog must declare README quick-start profiles".into());
    }
    let mut quick_start_ids = BTreeSet::new();
    for id in &input.quick_start_profile_ids {
        if !nonblank(id) || !quick_start_ids.insert(id) {
            return Err("quick-start ids must be nonblank and unique".into());
        }
    }
    if input.stage == Stage::Release {
        for id in &input.quick_start_profile_ids {
            match input.profiles.iter().find(|profile| {
                &profile.id == id && profile.available && valid_target(&profile.target)
            }) {
                Some(profile) => add(
                    &mut result,
                    Behavior::QuickStart,
                    EvidenceLayer::ModelRuntime,
                    ObligationScope::Profile {
                        profile_id: id.clone(),
                        target: profile.target.clone(),
                    },
                    "independent README quick-start promise",
                ),
                None => result.gaps.push(Gap::MissingQuickStart {
                    profile_id: id.clone(),
                }),
            }
        }
    }
    if input.stage != Stage::PullRequest {
        add_for_scopes(
            &mut result,
            &[Behavior::ModelForward],
            EvidenceLayer::ModelRuntime,
            &architectures(&targets),
            "architecture and output-protocol representative",
        );
        add_for_scopes(
            &mut result,
            &[Behavior::ModelForward],
            EvidenceLayer::ModelRuntime,
            &backends(&targets),
            "actual execution on each advertised backend",
        );
        add_for_scopes(
            &mut result,
            &[Behavior::Installation],
            EvidenceLayer::Installation,
            &backends(&targets),
            "final distribution installation and startup",
        );
    }
    if active {
        add(
            &mut result,
            Behavior::WorkspaceChecks,
            EvidenceLayer::Compilation,
            ObligationScope::Global,
            "code and checker compilation is independent of runtime evidence",
        );
    }
    for area in areas {
        let affected = area_targets(&input.impact, area, &targets)?;
        area_obligations(&mut result, area, &affected);
        if area == ChangeArea::Template {
            // Text framing alone cannot distinguish a non-thinking template from
            // one with prompt-opened or model-generated reasoning. Keep positive
            // and absence checks distinct even when they share the output protocol.
            for scope in protocols(&affected) {
                let matching: Vec<_> = input
                    .profiles
                    .iter()
                    .filter(|profile| scope_matches(&scope, &profile.target, Some(&profile.id)))
                    .collect();
                if matching.is_empty() {
                    add(
                        &mut result,
                        Behavior::ReasoningBoundaries,
                        EvidenceLayer::ModelRuntime,
                        scope.clone(),
                        "reasoning capability has no declared representative",
                    );
                }
                for profile in matching {
                    use ferrum_types::ModelReasoningProtocol as Reasoning;
                    let behavior = match profile.reasoning_protocol {
                        Reasoning::Unknown => {
                            let gap = Gap::UnknownReasoningCapability {
                                profile_id: profile.id.clone(),
                            };
                            if !result.gaps.contains(&gap) {
                                result.gaps.push(gap);
                            }
                            continue;
                        }
                        Reasoning::None => Behavior::ReasoningAbsence,
                        Reasoning::PromptOpened | Reasoning::ModelGenerated => {
                            Behavior::ReasoningBoundaries
                        }
                    };
                    let reasoning_scope = ObligationScope::Reasoning {
                        protocol: profile.target.protocol,
                        backend: profile.target.backend,
                        execution_path: profile.target.execution_path.clone(),
                        reasoning_protocol: profile.reasoning_protocol,
                    };
                    add(&mut result, behavior, EvidenceLayer::ModelRuntime, reasoning_scope,
                        "declared reasoning capability; runtime must verify the loaded capability and its matching positive or absence observation");
                }
            }
        }
    }

    let mut profiles: Vec<_> = input
        .profiles
        .iter()
        .filter(|profile| profile.available && valid_target(&profile.target))
        .collect();
    profiles.sort_by(|left, right| left.id.cmp(&right.id));
    let covers = |profile: &ModelProfile, obligation: &Obligation| {
        requires_model(obligation)
            && scope_matches(&obligation.scope, &profile.target, Some(&profile.id))
            && match &obligation.scope {
                ObligationScope::Reasoning {
                    reasoning_protocol, ..
                } => *reasoning_protocol == profile.reasoning_protocol,
                _ => true,
            }
            && match obligation.behavior {
                Behavior::ReasoningBoundaries => profile.reasoning_protocol.supports_reasoning(),
                Behavior::ReasoningAbsence => {
                    profile.reasoning_protocol == ferrum_types::ModelReasoningProtocol::None
                }
                _ => true,
            }
    };
    // Reserve every required QuickStart first, then reuse it for all matching scopes.
    let mut selected = BTreeSet::<usize>::new();
    for (index, profile) in profiles.iter().enumerate() {
        if result.obligations.iter().any(|obligation| {
            obligation.behavior == Behavior::QuickStart && covers(profile, obligation)
        }) {
            selected.insert(index);
        }
    }
    let mut remaining: Vec<_> = result
        .obligations
        .iter()
        .enumerate()
        .filter(|(_, obligation)| requires_model(obligation))
        .filter(|(_, obligation)| {
            !selected
                .iter()
                .any(|index| covers(profiles[*index], obligation))
        })
        .map(|(index, _)| index)
        .collect();
    while !remaining.is_empty() {
        let mut best: Option<(usize, usize)> = None;
        for (index, profile) in profiles
            .iter()
            .enumerate()
            .filter(|(index, _)| !selected.contains(index))
        {
            let coverage = remaining
                .iter()
                .filter(|obligation| covers(profile, &result.obligations[**obligation]))
                .count();
            if coverage == 0 {
                continue;
            }
            if best.is_none_or(|(previous, score)| {
                coverage > score
                    || (coverage == score
                        && estimate_cmp(profile, profiles[previous]) == Ordering::Less)
            }) {
                best = Some((index, coverage));
            }
        }
        let Some((index, _)) = best else {
            break;
        };
        selected.insert(index);
        remaining.retain(|obligation| !covers(profiles[index], &result.obligations[*obligation]));
    }
    for obligation in remaining {
        result.gaps.push(Gap::MissingRepresentative { obligation });
    }
    // Scope coverage alone can reserve a cheap but unbound model, including a
    // mandatory QuickStart. Complete any feasible checker binding before final
    // assignment; a compatible unselected model must not be hidden by that
    // reservation. Profile scope still prevents replacing a QuickStart itself.
    for obligation in result
        .obligations
        .iter()
        .filter(|obligation| requires_model(obligation))
    {
        if selected.iter().any(|index| {
            covers(profiles[*index], obligation)
                && complete_model_binding(&input.checks, obligation, &profiles[*index].target)
        }) {
            continue;
        }
        let representative = profiles
            .iter()
            .enumerate()
            .filter(|(_, profile)| {
                covers(profile, obligation)
                    && complete_model_binding(&input.checks, obligation, &profile.target)
            })
            .min_by(|(_, left), (_, right)| estimate_cmp(left, right))
            .map(|(index, _)| index);
        if let Some(index) = representative {
            selected.insert(index);
        }
    }
    // Selection reserves models; assignment chooses one concrete execution for
    // each obligation. Merely overlapping a scope must not duplicate checks on
    // every QuickStart or architecture model that is already selected.
    let mut assignments = BTreeMap::<usize, Vec<usize>>::new();
    for (obligation_index, obligation) in result.obligations.iter().enumerate() {
        let representative = selected
            .iter()
            .copied()
            .filter(|index| covers(profiles[*index], obligation))
            .min_by(|left, right| {
                let left_bound =
                    complete_model_binding(&input.checks, obligation, &profiles[*left].target);
                let right_bound =
                    complete_model_binding(&input.checks, obligation, &profiles[*right].target);
                right_bound
                    .cmp(&left_bound)
                    .then_with(|| estimate_cmp(profiles[*left], profiles[*right]))
            });
        if let Some(index) = representative {
            assignments.entry(index).or_default().push(obligation_index);
        }
    }
    for (index, obligations) in assignments {
        let profile = profiles[index];
        let reasons = obligations
            .iter()
            .map(|index| result.obligations[*index].reason.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        result.selected.push(SelectedProfile {
            profile: profile.clone(),
            obligations,
            reasons,
        });
    }
    for (index, obligation) in result.obligations.iter_mut().enumerate() {
        obligation.checkers = input
            .checks
            .iter()
            .filter(|check| {
                if check.behavior != obligation.behavior
                    || check.layer != obligation.layer
                    || (!obligation.entrypoints.is_empty()
                        && !obligation
                            .entrypoints
                            .iter()
                            .any(|entry| check.entrypoints.contains(entry)))
                {
                    return false;
                }
                match &check.target {
                    None => true,
                    Some(target) if requires_model(obligation) => {
                        result.selected.iter().any(|selected| {
                            selected.obligations.contains(&index)
                                && &selected.profile.target == target
                        })
                    }
                    Some(target) => scope_matches(&obligation.scope, target, None),
                }
            })
            .map(|check| check.id.clone())
            .collect();
        obligation.checkers.sort();
        let assigned_for_target = |target: Option<&ExecutionTarget>| {
            let candidates: Vec<_> = input
                .checks
                .iter()
                .filter(|check| {
                    obligation.checkers.contains(&check.id)
                        && target.is_none_or(|target| {
                            check
                                .target
                                .as_ref()
                                .is_none_or(|declared| declared == target)
                        })
                })
                .collect();
            !candidates.is_empty()
                && obligation.entrypoints.iter().all(|entrypoint| {
                    candidates
                        .iter()
                        .any(|check| check.entrypoints.contains(entrypoint))
                })
        };
        let assigned = if requires_model(obligation) {
            let selected_targets: Vec<_> = result
                .selected
                .iter()
                .filter(|selected| selected.obligations.contains(&index))
                .collect();
            // Do not combine Run on one target with HTTP on another. Only the
            // assigned representative supplies this obligation's binding.
            !selected_targets.is_empty()
                && selected_targets
                    .iter()
                    .all(|selected| assigned_for_target(Some(&selected.profile.target)))
        } else {
            assigned_for_target(None)
        };
        if !assigned {
            result.gaps.push(Gap::UnassignedCheck { obligation: index });
        }
    }
    for profile in &input.profiles {
        if !result
            .selected
            .iter()
            .any(|selected| selected.profile.id == profile.id)
        {
            let reason = if !profile.available {
                "profile unavailable"
            } else if !valid_target(&profile.target) {
                "invalid execution target; see gap"
            } else {
                "not needed after required coverage and selected-profile reuse; no execution is certified"
            };
            result.omitted.push(OmittedProfile {
                profile_id: profile.id.clone(),
                reason: reason.into(),
            });
        }
    }
    result
        .omitted
        .sort_by(|left, right| left.profile_id.cmp(&right.profile_id));
    record_cost(&mut result)?;
    Ok(result)
}

#[cfg(test)]
#[path = "selection_tests.rs"]
mod tests;
