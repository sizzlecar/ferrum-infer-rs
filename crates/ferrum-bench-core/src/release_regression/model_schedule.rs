//! Bind the model runner's implemented cases to selected model obligations.
//! This schedule neither changes plan gaps nor approves other release evidence.
use super::model_tasks::ModelCheck;
use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRunRequirements {
    pub profile: ModelProfile,
    pub checks: Vec<ModelCheck>,
    pub quick_start: bool,
    pub obligations: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelTaskSchedule {
    pub runs: Vec<ModelRunRequirements>,
    /// Model-runtime/performance obligations without an executable assignment.
    /// Non-model evidence remains the responsibility of the original plan.
    pub unsupported_obligations: Vec<usize>,
}

const ALL_ENTRYPOINTS: &[Entrypoint] = &[
    Entrypoint::Run,
    Entrypoint::ServeSync,
    Entrypoint::ServeStream,
];
const HTTP_ENTRYPOINTS: &[Entrypoint] = &[Entrypoint::ServeSync, Entrypoint::ServeStream];

fn capability(behavior: Behavior) -> Option<(ModelCheck, &'static str, &'static [Entrypoint])> {
    use Behavior::*;
    let (check, id) = match behavior {
        ModelLoad => (ModelCheck::Basic, "model-regression.basic.model-load"),
        ModelForward => (ModelCheck::Basic, "model-regression.basic.model-forward"),
        // A naturally completed positive example, not a distinction between all
        // internal EOS kinds or a substitute for stop/length contract tests.
        NaturalEnd => (ModelCheck::Basic, "model-regression.basic.natural-end"),
        QuickStart => (ModelCheck::Basic, "model-regression.basic.quick-start"),
        TemplateHistory => (ModelCheck::Basic, "model-regression.basic.template-history"),
        ProtocolFraming => (ModelCheck::Basic, "model-regression.basic.protocol-framing"),
        UserStop => (ModelCheck::Stop, "model-regression.stop.user-stop"),
        ReasoningBoundaries => (
            ModelCheck::Reasoning,
            "model-regression.reasoning.boundaries",
        ),
        ReasoningAbsence => (
            ModelCheck::Basic,
            "model-regression.basic.reasoning-absence",
        ),
        LengthLimit => (ModelCheck::Length, "model-regression.length.limit"),
        Observability => (
            ModelCheck::Observability,
            "model-regression.observability.request-lifecycle",
        ),
        StructuredValidity => (
            ModelCheck::Structured,
            "model-regression.structured.validity",
        ),
        // Both initial HTTP modes execute a named call with a distractor tool;
        // each returned identity is used for its actual continuation.
        ToolSelection => (ModelCheck::Tools, "model-regression.tools.selection"),
        ToolHandoff => (ModelCheck::Tools, "model-regression.tools.handoff"),
        ToolContinuation => (ModelCheck::Tools, "model-regression.tools.continuation"),
        _ => return None,
    };
    let entrypoints = match behavior {
        StructuredValidity | ToolSelection | ToolHandoff | ToolContinuation => HTTP_ENTRYPOINTS,
        _ => ALL_ENTRYPOINTS,
    };
    Some((check, id, entrypoints))
}

/// Implemented model-runtime checks only. These declarations are not completed
/// executions, sampler-mask proofs, installation checks, or performance evidence.
pub fn model_check_descriptors() -> Vec<CheckDescriptor> {
    [
        Behavior::ModelLoad,
        Behavior::ModelForward,
        Behavior::NaturalEnd,
        Behavior::QuickStart,
        Behavior::TemplateHistory,
        Behavior::ProtocolFraming,
        Behavior::UserStop,
        Behavior::ReasoningBoundaries,
        Behavior::ReasoningAbsence,
        Behavior::LengthLimit,
        Behavior::Observability,
        Behavior::StructuredValidity,
        Behavior::ToolSelection,
        Behavior::ToolHandoff,
        Behavior::ToolContinuation,
    ]
    .into_iter()
    .map(|behavior| {
        let (_, id, entrypoints) = capability(behavior).expect("listed runner capability");
        CheckDescriptor {
            id: id.into(),
            behavior,
            layer: EvidenceLayer::ModelRuntime,
            entrypoints: entrypoints.to_vec(),
            target: None,
        }
    })
    .collect()
}

pub(super) fn scope_matches(scope: &ObligationScope, profile: &ModelProfile) -> bool {
    let target = &profile.target;
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
            reasoning_protocol,
        } => {
            protocol == &target.protocol
                && *backend == target.backend
                && execution_path == &target.execution_path
                && *reasoning_protocol == profile.reasoning_protocol
        }
        ObligationScope::Target { target: expected } => expected == target,
        ObligationScope::Profile {
            profile_id,
            target: expected,
        } => profile_id == &profile.id && expected == target,
    }
}

/// Coalesce assigned obligations without selecting replacement models. Missing
/// representatives and unsupported behaviors stay visible; Plan.gaps is untouched.
pub fn model_task_schedule(plan: &Plan) -> ModelTaskSchedule {
    let performance = super::performance::performance_task_schedule(plan);
    let performance_owned: Vec<_> = performance
        .runs
        .iter()
        .flat_map(|run| &run.obligations)
        .copied()
        .collect();
    let mut runs = BTreeMap::<String, ModelRunRequirements>::new();
    let mut unsupported_obligations = Vec::new();
    for (index, obligation) in plan.obligations.iter().enumerate() {
        if performance_owned.contains(&index) {
            continue;
        }
        if !matches!(
            obligation.layer,
            EvidenceLayer::ModelRuntime | EvidenceLayer::Performance
        ) {
            continue;
        }
        let mut owners = plan
            .selected
            .iter()
            .filter(|selected| selected.obligations.contains(&index));
        let owner = owners.next();
        let binding = capability(obligation.behavior);
        let usable = owner.zip(binding).filter(|(owner, (_, _, entrypoints))| {
            obligation.layer == EvidenceLayer::ModelRuntime
                && owners.next().is_none()
                && owner.profile.available
                && match obligation.behavior {
                    Behavior::ReasoningBoundaries => {
                        owner.profile.reasoning_protocol.supports_reasoning()
                    }
                    Behavior::ReasoningAbsence => {
                        owner.profile.reasoning_protocol
                            == ferrum_types::ModelReasoningProtocol::None
                    }
                    _ => true,
                }
                && scope_matches(&obligation.scope, &owner.profile)
                && !obligation.entrypoints.is_empty()
                && obligation
                    .entrypoints
                    .iter()
                    .all(|entrypoint| entrypoints.contains(entrypoint))
                && (obligation.behavior != Behavior::QuickStart
                    || matches!(&obligation.scope, ObligationScope::Profile { .. }))
                && runs
                    .get(&owner.profile.id)
                    .is_none_or(|run| run.profile == owner.profile)
        });
        let Some((owner, (check, _, _))) = usable else {
            unsupported_obligations.push(index);
            continue;
        };
        let run = runs
            .entry(owner.profile.id.clone())
            .or_insert_with(|| ModelRunRequirements {
                profile: owner.profile.clone(),
                checks: Vec::new(),
                quick_start: false,
                obligations: Vec::new(),
            });
        if !run.checks.contains(&check) {
            run.checks.push(check);
        }
        run.quick_start |= obligation.behavior == Behavior::QuickStart;
        run.obligations.push(index);
    }
    let runs = runs
        .into_values()
        .map(|mut run| {
            run.checks.sort_by_key(|check| match check {
                ModelCheck::Basic => 0,
                ModelCheck::Stop => 1,
                ModelCheck::Structured => 2,
                ModelCheck::Tools => 3,
                ModelCheck::Reasoning => 4,
                ModelCheck::Length => 5,
                ModelCheck::AutoToolsJson => 6,
                ModelCheck::Observability => 7,
            });
            run
        })
        .collect();
    ModelTaskSchedule {
        runs,
        unsupported_obligations,
    }
}

#[cfg(test)]
#[path = "model_schedule_tests.rs"]
mod tests;
