//! Explicit logical-weight selection; physical conversion belongs to the materializer.
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn select(
    program: ModelProgram,
    groups: &[Vec<WeightId>],
) -> Result<ModelProgram, VNextError> {
    let invalid = |reason: &str| VNextError::InvalidModelConfig {
        family_id: program.family_id().to_string(),
        field: "precision".into(),
        reason: reason.into(),
    };
    let mut selected = BTreeSet::new();
    for group in groups {
        if group.is_empty() || group.iter().any(|id| !selected.insert(id.clone())) {
            return Err(invalid("FP8 groups must be nonempty and disjoint"));
        }
    }
    let references: BTreeMap<_, _> = program
        .weights()
        .iter()
        .map(|w| (&w.value_id, &w.weight_id))
        .collect();
    let mut consumed = BTreeSet::new();
    let mut blocks = program.blocks().to_vec();
    for node in blocks.iter_mut().flat_map(|b| &mut b.nodes) {
        for (index, input) in node.inputs.iter().enumerate() {
            let Some(weight) = references.get(input).filter(|w| selected.contains(**w)) else {
                continue;
            };
            if index != 1
                || node.operation_id.as_str() != DENSE_LINEAR_BF16_OPERATION_ID
                || node.required_version != ContractVersion::new(1, 0)
            {
                return Err(invalid(
                    "selected weight must have only BF16 dense weight-port consumers",
                ));
            }
            node.operation_id = OperationId::new(DENSE_LINEAR_FP8_BF16_OPERATION_ID)?;
            consumed.insert((*weight).clone());
        }
    }
    if consumed != selected {
        return Err(invalid("FP8 selection contains an unconsumed weight"));
    }
    with_blocks(&program, blocks)
}

/// Fuse independent selected projections of one immutable activation. This is
/// a semantic graph rewrite, so the resulting program must be planned again;
/// no old activation or invocation-scratch allocation is reused implicitly.
pub(crate) fn share_input_quantization(program: ModelProgram) -> Result<ModelProgram, VNextError> {
    let weights: BTreeMap<_, _> = program
        .weights()
        .iter()
        .map(|weight| (&weight.value_id, weight))
        .collect();
    let states: BTreeSet<_> = program
        .states()
        .iter()
        .map(|state| &state.value_id)
        .collect();
    let mut available: BTreeSet<_> = program
        .inputs()
        .iter()
        .chain(weights.keys().copied())
        .chain(states.iter().copied())
        .cloned()
        .collect();
    let mut blocks = program.blocks().to_vec();
    for block in &mut blocks {
        // Inputs and attributes define the shared quantization domain. Static
        // weights may differ; shapes and arithmetic contracts must not differ.
        // Nonadjacent Q/K/V nodes can be combined at their first consumer since
        // their only non-weight input is already defined at that position.
        let mut candidates: BTreeMap<ProgramValueId, Vec<Vec<usize>>> = BTreeMap::new();
        for (index, node) in block.nodes.iter().enumerate() {
            let eligible = node.operation_id.as_str() == DENSE_LINEAR_FP8_BF16_OPERATION_ID
                && node.required_version == ContractVersion::new(1, 0)
                && node.work == ProgramNodeWorkSpec::Fixed
                && node.inputs.len() == 2
                && node.outputs.len() == 1
                && node.attributes.len() == 2
                && !states.contains(&node.inputs[0])
                && !states.contains(&node.outputs[0])
                && available.contains(&node.inputs[0])
                && weights.get(&node.inputs[1]).is_some_and(|weight| {
                    let shape = |name: &str| {
                        node.attributes.iter().find_map(|(key, value)| {
                            if key.as_str() == name {
                                if let SemanticValue::Unsigned(n) = value {
                                    return (*n > 0).then_some(*n);
                                }
                            }
                            None
                        })
                    };
                    matches!((shape("in_features"), shape("out_features")), (Some(inputs), Some(outputs))
                        if weight.tensor.dimensions == [outputs, inputs]
                            && weight.tensor.element_type == ElementType::Bf16
                            && weight.tensor.layout == ResolvedTensorLayout::Contiguous)
                });
            if eligible {
                let groups = candidates.entry(node.inputs[0].clone()).or_default();
                if let Some(group) = groups.iter_mut().find(|group| {
                    group.len() < 3 && block.nodes[group[0]].attributes == node.attributes
                }) {
                    group.push(index);
                } else {
                    groups.push(vec![index]);
                }
            }
            available.extend(node.outputs.iter().cloned());
        }
        let mut replacements = BTreeMap::new();
        let mut removed = BTreeSet::new();
        for group in candidates
            .into_values()
            .flatten()
            .filter(|group| group.len() > 1)
        {
            let mut fused = block.nodes[group[0]].clone();
            fused.operation_id = OperationId::new(match group.len() {
                2 => DENSE_LINEAR_FP8_PAIR_BF16_OPERATION_ID,
                3 => DENSE_LINEAR_FP8_TRIPLE_BF16_OPERATION_ID,
                _ => unreachable!("groups contain at most three projections"),
            })?;
            fused.inputs.truncate(1);
            fused.inputs.extend(
                group
                    .iter()
                    .map(|&index| block.nodes[index].inputs[1].clone()),
            );
            fused.outputs = group
                .iter()
                .map(|&index| block.nodes[index].outputs[0].clone())
                .collect();
            replacements.insert(group[0], fused);
            removed.extend(group.into_iter().skip(1));
        }
        block.nodes = block
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                (!removed.contains(&index))
                    .then(|| replacements.remove(&index).unwrap_or_else(|| node.clone()))
            })
            .collect();
    }
    with_blocks(&program, blocks)
}

fn with_blocks(
    program: &ModelProgram,
    blocks: Vec<ProgramBlock>,
) -> Result<ModelProgram, VNextError> {
    let mut result = ModelProgram::new(
        program.family_id().clone(),
        program.inputs().to_vec(),
        blocks,
        program.states().to_vec(),
        program.weights().to_vec(),
        program.outputs().to_vec(),
    )?
    .with_input_tensor_specs(program.input_tensor_specs().clone())?;
    if let Some(inputs) = program.checkpoint_inputs() {
        result = result.with_checkpoint_inputs(inputs.clone())?;
    }
    Ok(result)
}

pub(crate) fn operation() -> Result<NumericalOperationContract, VNextError> {
    numerical_operation(DENSE_LINEAR_FP8_BF16_OPERATION_ID)
}

pub(crate) fn shared_quantization_operations() -> Result<[NumericalOperationContract; 2], VNextError>
{
    Ok([
        numerical_operation(DENSE_LINEAR_FP8_PAIR_BF16_OPERATION_ID)?,
        numerical_operation(DENSE_LINEAR_FP8_TRIPLE_BF16_OPERATION_ID)?,
    ])
}

fn numerical_operation(name: &str) -> Result<NumericalOperationContract, VNextError> {
    Ok(NumericalOperationContract {
        operation_id: OperationId::new(name)?,
        version: ContractVersion::new(1, 0),
        // E4M3 inputs and vendor accumulation have a separate OperationDefined contract.
        multiplication_type: None,
        accumulation_type: None,
        staged_arithmetic: None,
        composite_arithmetic: None,
    })
}

#[cfg(test)]
mod tests;
