//! Same main arithmetic with explicit GDN candidates and a bounded greedy head.
use super::*;
use ferrum_runtime::{
    GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID, TAIL_TWO_GREEDY_F16_OPERATION_ID,
};

pub(super) fn build(
    c: &Qwen35MtpFamilyConfig,
    main: &ModelProgram,
    visual: &NumericalExecutionProfile,
    references: Vec<WeightReference>,
) -> Result<(ModelProgram, NumericalExecutionProfile), VNextError> {
    let text = Qwen35FamilyProvider::text_config(&c.main)?;
    let original = Qwen35FamilyProvider::new()?.numerical_profiles(&c.main)?;
    let base = original.resolve(
        &NumericalProfileId::new(F16_NUMERICAL_PROFILE_ID)
            .map_err(|e| invalid_config("mtp.profile", e))?,
    )?;
    let mut profile = visual.clone();
    profile.family_id = ModelFamilyId::new(FAMILY_ID)?;
    profile.id = NumericalProfileId::new(Qwen35MtpRole::MainVerify.numerical_profile())
        .map_err(|e| invalid_config("mtp.profile", e))?;
    // Preserve the original ExactTokenPrefix/BoundaryValue and PrefixPositions
    // state semantics; real candidate providers must cover every state effect.
    profile.states = base.states.clone();
    profile.kv_storage = base.kv_storage.clone();
    let mut blocks = main.blocks().to_vec();
    for block in &mut blocks {
        block.nodes.retain(|n| n.id.as_str() != "node.greedy_token");
        for n in &mut block.nodes {
            if n.operation_id.as_str() == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID {
                n.operation_id = operation_id(GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID)?;
                n.required_version = ContractVersion::new(1, 0);
                for (ordinal, suffix) in [(8, "conv"), (9, "delta")] {
                    let state = profile
                        .states
                        .iter()
                        .find(|s| s.value_id == n.inputs[ordinal])
                        .ok_or_else(|| {
                            invalid_config("mtp.verify", "original recurrent state missing")
                        })?;
                    let output = value_id(format!("{}.candidate.{suffix}", n.outputs[0]))?;
                    profile
                        .boundaries
                        .insert(output.clone(), state.tensor.element_type);
                    n.outputs.push(output);
                }
            }
            if n.id.as_str() == "node.logits" {
                n.id = node_id("node.verify_head")?;
                n.operation_id = operation_id(TAIL_TWO_GREEDY_F16_OPERATION_ID)?;
                n.required_version = ContractVersion::new(1, 0);
                n.outputs = vec![value_id(MAIN_GREEDY_TOKENS)?, value_id(MAIN_HIDDEN_TAIL)?];
                n.attributes = BTreeMap::from([
                    attribute("hidden_size", text.hidden_size as u64)?,
                    attribute("vocab_size", c.main.vocab_size)?,
                ]);
            }
        }
    }
    profile.boundaries.remove(&value_id("value.output.logits")?);
    profile
        .boundaries
        .remove(&value_id("value.output.greedy_token")?);
    profile
        .boundaries
        .insert(value_id(MAIN_GREEDY_TOKENS)?, ElementType::U32);
    profile
        .boundaries
        .insert(value_id(MAIN_HIDDEN_TAIL)?, ElementType::F16);
    profile.operations.retain(|o| {
        !matches!(
            o.operation_id.as_str(),
            LAST_TOKEN_DENSE_LINEAR_OPERATION_ID | LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID
        )
    });
    for op in &mut profile.operations {
        if op.operation_id.as_str() == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID {
            op.operation_id = operation_id(GATED_DELTA_CANDIDATE_SCAN_F16_OPERATION_ID)?;
            op.version = ContractVersion::new(1, 0);
        }
    }
    profile.operations.push(NumericalOperationContract {
        operation_id: operation_id(TAIL_TWO_GREEDY_F16_OPERATION_ID)?,
        version: ContractVersion::new(1, 0),
        multiplication_type: Some(ElementType::F32),
        accumulation_type: Some(ElementType::F32),
        staged_arithmetic: None,
        composite_arithmetic: None,
    });
    profile
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    let original_inputs = main
        .checkpoint_inputs()
        .ok_or_else(|| invalid_config("mtp.verify", "main token and visual input roles absent"))?;
    let inputs = std::iter::once(original_inputs.token_input().clone())
        .chain(original_inputs.conditioning_inputs().iter().cloned())
        .collect();
    let binding = ProgramCheckpointInputs::new(
        original_inputs.token_input().clone(),
        original_inputs.conditioning_inputs().clone(),
    )?;
    let program = ModelProgram::new(
        profile.family_id.clone(),
        inputs,
        blocks,
        profile.states.clone(),
        references,
        vec![value_id(MAIN_GREEDY_TOKENS)?, value_id(MAIN_HIDDEN_TAIL)?],
    )?
    .with_checkpoint_inputs(binding)?;
    Ok((program, profile))
}
