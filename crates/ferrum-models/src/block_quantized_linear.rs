//! Explicit logical-weight selection for fixed-tensor weight-only Linear.
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn select(
    program: ModelProgram,
    selected: &BTreeSet<WeightId>,
) -> Result<ModelProgram, VNextError> {
    let invalid = |reason: &str| VNextError::InvalidModelConfig {
        family_id: program.family_id().to_string(),
        field: "weight_encoding".into(),
        reason: reason.into(),
    };
    let references: BTreeMap<_, _> = program
        .weights()
        .iter()
        .map(|w| (&w.value_id, &w.weight_id))
        .collect();
    let mut consumed = BTreeSet::new();
    let mut blocks = program.blocks().to_vec();
    for node in blocks.iter_mut().flat_map(|block| &mut block.nodes) {
        for (ordinal, input) in node.inputs.iter().enumerate() {
            let Some(weight) = references
                .get(input)
                .filter(|weight| selected.contains(**weight))
            else {
                continue;
            };
            if ordinal != 1
                || node.operation_id.as_str() != DENSE_LINEAR_BF16_OPERATION_ID
                || node.required_version != ContractVersion::new(1, 0)
            {
                return Err(invalid(
                    "native blocks require only BF16 Linear weight-port consumers",
                ));
            }
            node.operation_id = OperationId::new(DENSE_LINEAR_BLOCK_BF16_OPERATION_ID)?;
            consumed.insert((*weight).clone());
        }
    }
    // Prefix decode deliberately omits text entrance references. Those source
    // weights remain installed, but there must be no unconsumed selected value
    // among the actual program references.
    let used: BTreeSet<_> = program
        .weights()
        .iter()
        .filter(|w| selected.contains(&w.weight_id))
        .map(|w| w.weight_id.clone())
        .collect();
    if consumed != used {
        return Err(invalid(
            "native block selection contains an unconsumed program weight",
        ));
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
        operation_id: OperationId::new(DENSE_LINEAR_BLOCK_BF16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        multiplication_type: None,
        accumulation_type: None,
        staged_arithmetic: None,
        composite_arithmetic: None,
    })
}
