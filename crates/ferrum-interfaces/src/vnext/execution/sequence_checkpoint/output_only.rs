use std::collections::BTreeMap;

use super::SequenceCheckpointUnsupportedReason;
use crate::vnext::{
    ExecutionPlan, PlanNode, ProgramCheckpointInputs, ProgramValueId, ResolvedValueRole,
    TensorAccess, VNextError,
};

/// Run only on nodes from the semantic plan build: operation signatures,
/// binding coverage, state effects, and global storage aliases have already
/// been validated. Unknown operations cannot reach this point. Provider
/// checkpoint declarations and hidden persistent workspace are checked by the
/// enclosing derivation, independently of this value-dependency proof.
pub(super) fn state_dependencies(
    inputs: &ProgramCheckpointInputs,
    nodes: &[PlanNode],
) -> Result<Vec<SequenceCheckpointUnsupportedReason>, VNextError> {
    if inputs.output_only_inputs().is_empty() {
        return Ok(Vec::new());
    }
    let aliases = ExecutionPlan::alias_classes(nodes)?;
    // Keep the original input as diagnostic evidence through intermediate
    // values. Every input may affect every output; no operation-name whitelist
    // or numerical assumption is used to discard a dependency.
    let mut affected: BTreeMap<ProgramValueId, ProgramValueId> = inputs
        .output_only_inputs()
        .iter()
        .map(|id| (id.clone(), id.clone()))
        .collect();
    loop {
        let before = affected.len();
        let affected_classes: BTreeMap<_, _> = affected
            .iter()
            .filter_map(|(value, origin)| aliases.get(value).map(|class| (class, origin.clone())))
            .collect();
        for (value, class) in &aliases {
            if let Some(origin) = affected_classes.get(class) {
                affected
                    .entry(value.clone())
                    .or_insert_with(|| origin.clone());
            }
        }
        for node in nodes {
            let origin = node
                .values()
                .iter()
                .filter(|binding| binding.role() == ResolvedValueRole::Input)
                .map(|binding| binding.value_id())
                .chain(node.work().token_source().map(|source| source.value_id()))
                .find_map(|value| affected.get(value))
                .cloned();
            let Some(origin) = origin else { continue };
            if let Some(effect) = node.state_effects().first() {
                // Even a read-only state binding is conservatively rejected:
                // this API does not claim an operation's finer input/state
                // dependency matrix. All supported states must remain outside
                // the affected operation closure.
                return Ok(vec![
                    SequenceCheckpointUnsupportedReason::OutputOnlyInputAffectsState {
                        value_id: origin,
                        node_id: node.id().clone(),
                        state_id: effect.state_id().clone(),
                    },
                ]);
            }
            for binding in node.values().iter().filter(|binding| {
                binding.role() == ResolvedValueRole::Output
                    || matches!(
                        binding.access(),
                        TensorAccess::Write | TensorAccess::ReadWrite
                    )
            }) {
                affected
                    .entry(binding.value_id().clone())
                    .or_insert_with(|| origin.clone());
            }
        }
        if affected.len() == before {
            return Ok(Vec::new());
        }
    }
}

#[cfg(test)]
mod tests;
