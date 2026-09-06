use super::*;
use ferrum_types::ModelOutputProtocol;

fn profile(id: &str, backend: Backend) -> ModelProfile {
    ModelProfile {
        id: id.into(),
        model: "fixture/shared-model".into(),
        target: ExecutionTarget {
            architecture: "dense".into(),
            protocol: ModelOutputProtocol::Text,
            precision: "bf16".into(),
            backend,
            execution_path: "production-plan-runtime".into(),
        },
        available: true,
        estimate: None,
    }
}

fn obligation(behavior: Behavior, owner: &ModelProfile) -> Obligation {
    Obligation {
        behavior,
        layer: EvidenceLayer::ModelRuntime,
        entrypoints: capability(behavior)
            .map(|(_, _, entrypoints)| entrypoints)
            .unwrap_or(ALL_ENTRYPOINTS)
            .to_vec(),
        scope: ObligationScope::Profile {
            profile_id: owner.id.clone(),
            target: owner.target.clone(),
        },
        reason: "fixture obligation".into(),
        checkers: Vec::new(),
    }
}

fn selected(profile: ModelProfile, obligations: Vec<usize>) -> SelectedProfile {
    SelectedProfile {
        profile,
        obligations,
        reasons: Vec::new(),
    }
}

fn make_plan(obligations: Vec<Obligation>, selected: Vec<SelectedProfile>) -> Plan {
    Plan {
        stage: Stage::Release,
        impact: Impact {
            areas: Vec::new(),
            paths: Vec::new(),
            unknown_paths: Vec::new(),
            product_contract_changed: false,
        },
        obligations,
        selected,
        omitted: Vec::new(),
        gaps: vec![Gap::ProductContractReview],
        cost: PlanCost::default(),
    }
}

#[test]
fn model_schedule_coalesces_behavior_checks_without_claiming_release_approval() {
    let owner = profile("shared", Backend::Cuda);
    let obligations = [
        Behavior::ToolContinuation,
        Behavior::UserStop,
        Behavior::ModelForward,
        Behavior::ModelLoad,
        Behavior::TemplateHistory,
        Behavior::NaturalEnd,
        Behavior::StructuredValidity,
    ]
    .into_iter()
    .map(|behavior| obligation(behavior, &owner))
    .collect();
    let plan = make_plan(obligations, vec![selected(owner, (0..7).collect())]);
    let original = plan.clone();
    let schedule = model_task_schedule(&plan);
    assert_eq!(schedule.runs.len(), 1);
    assert_eq!(
        schedule.runs[0].checks,
        [
            ModelCheck::Basic,
            ModelCheck::Stop,
            ModelCheck::Structured,
            ModelCheck::Tools,
        ]
    );
    assert_eq!(schedule.runs[0].obligations, (0..7).collect::<Vec<_>>());
    assert!(!schedule.runs[0].quick_start);
    assert!(schedule.unsupported_obligations.is_empty());
    assert_eq!(plan, original);
}

#[test]
fn model_schedule_preserves_independent_quick_starts_for_the_same_model() {
    let metal = profile("readme-metal", Backend::Metal);
    let cuda = profile("readme-cuda", Backend::Cuda);
    let plan = make_plan(
        vec![
            obligation(Behavior::QuickStart, &metal),
            obligation(Behavior::QuickStart, &cuda),
            obligation(Behavior::ModelLoad, &metal),
        ],
        vec![selected(metal, vec![0, 2]), selected(cuda, vec![1])],
    );
    let schedule = model_task_schedule(&plan);
    assert_eq!(schedule.runs.len(), 2);
    for run in &schedule.runs {
        assert!(run.quick_start);
        assert_eq!(run.checks, [ModelCheck::Basic]);
        let quick_start = &plan.obligations[run.obligations[0]];
        assert!(
            matches!(&quick_start.scope, ObligationScope::Profile { profile_id, .. }
            if profile_id == &run.profile.id)
        );
    }
    assert!(schedule.unsupported_obligations.is_empty());
}

#[test]
fn model_schedule_rejects_missing_entrypoints_and_unsupported_evidence() {
    let owner = profile("limited", Backend::Cpu);
    let mut obligations: Vec<_> = [
        Behavior::StructuredValidity,
        Behavior::ToolContinuation,
        Behavior::ModelLoad,
        Behavior::StructuredSampling,
        Behavior::Performance,
        Behavior::ModelForward,
        Behavior::LengthLimit,
        Behavior::ToolSelection,
        Behavior::ProtocolFraming,
    ]
    .into_iter()
    .map(|behavior| obligation(behavior, &owner))
    .collect();
    obligations[0].entrypoints.push(Entrypoint::Run);
    obligations[1].entrypoints = vec![Entrypoint::Run];
    obligations[2].entrypoints.clear();
    obligations[4].layer = EvidenceLayer::Performance;
    obligations[5].layer = EvidenceLayer::Performance;
    let plan = make_plan(obligations, vec![selected(owner, (0..9).collect())]);
    let schedule = model_task_schedule(&plan);
    assert!(schedule.runs.is_empty());
    assert_eq!(schedule.unsupported_obligations, (0..9).collect::<Vec<_>>());
}

#[test]
fn model_schedule_cannot_reassign_a_quick_start_or_hide_unassigned_models() {
    let owner = profile("selected", Backend::Metal);
    let other = profile("other-readme", Backend::Metal);
    let mut broad_quick_start = obligation(Behavior::QuickStart, &owner);
    broad_quick_start.scope = ObligationScope::Backend {
        backend: Backend::Metal,
    };
    let mut changed_target = obligation(Behavior::QuickStart, &owner);
    if let ObligationScope::Profile { target, .. } = &mut changed_target.scope {
        target.precision = "int4".into();
    }
    let plan = make_plan(
        vec![
            obligation(Behavior::QuickStart, &other),
            broad_quick_start,
            changed_target,
            obligation(Behavior::ModelLoad, &owner),
        ],
        vec![selected(owner, vec![0, 1, 2])],
    );
    let schedule = model_task_schedule(&plan);
    assert!(schedule.runs.is_empty());
    assert_eq!(schedule.unsupported_obligations, [0, 1, 2, 3]);
}

#[test]
fn model_schedule_descriptors_bind_only_implemented_product_flows() {
    let descriptors = model_check_descriptors();
    for behavior in [
        Behavior::ModelLoad,
        Behavior::ModelForward,
        Behavior::NaturalEnd,
        Behavior::QuickStart,
        Behavior::TemplateHistory,
        Behavior::UserStop,
    ] {
        let descriptor = descriptors
            .iter()
            .find(|descriptor| descriptor.behavior == behavior)
            .unwrap();
        assert_eq!(descriptor.entrypoints, ALL_ENTRYPOINTS);
        assert_eq!(descriptor.layer, EvidenceLayer::ModelRuntime);
        assert!(descriptor.target.is_none());
    }
    for behavior in [Behavior::StructuredValidity, Behavior::ToolContinuation] {
        let descriptor = descriptors
            .iter()
            .find(|descriptor| descriptor.behavior == behavior)
            .unwrap();
        assert_eq!(descriptor.entrypoints, HTTP_ENTRYPOINTS);
    }
    assert!(!descriptors.iter().any(|descriptor| matches!(
        descriptor.behavior,
        Behavior::StructuredSampling | Behavior::Performance | Behavior::Installation
    )));
    let owner = profile("selected", Backend::Cpu);
    let input = PlanInput {
        stage: Stage::Release,
        impact: Impact {
            areas: Vec::new(),
            paths: Vec::new(),
            unknown_paths: Vec::new(),
            product_contract_changed: false,
        },
        required_targets: vec![owner.target.clone()],
        quick_start_profile_ids: vec![owner.id.clone()],
        profiles: vec![owner],
        checks: descriptors,
    };
    let plan = super::super::selection::plan(&input).unwrap();
    let schedule = model_task_schedule(&plan);
    assert_eq!(schedule.runs.len(), 1);
    assert!(schedule.runs[0].quick_start);
    assert!(schedule.unsupported_obligations.is_empty());
    for (index, obligation) in plan.obligations.iter().enumerate() {
        if obligation.layer == EvidenceLayer::ModelRuntime {
            assert!(!obligation.checkers.is_empty());
            assert!(!plan
                .gaps
                .contains(&Gap::UnassignedCheck { obligation: index }));
        }
    }
    assert!(plan
        .gaps
        .iter()
        .any(|gap| matches!(gap, Gap::UnassignedCheck { .. })));
}
