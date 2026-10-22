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
    Ok(NumericalOperationContract {
        operation_id: OperationId::new(DENSE_LINEAR_FP8_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        // E4M3 inputs and vendor accumulation have a separate OperationDefined contract.
        multiplication_type: None,
        accumulation_type: None,
        staged_arithmetic: None,
        composite_arithmetic: None,
    })
}
