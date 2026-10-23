use super::*;

/// Reuse one validated block template. Every source input, weight and produced
/// value has an explicit destination; no unrecognized value survives remapping.
pub(super) fn append(
    nodes: &mut Vec<ProgramNode>,
    source: &ModelProgram,
    weights: &[WeightReference],
    layer: u64,
    input: ProgramValueId,
    config: &QwenImage21TransformerConfig,
) -> Result<ProgramValueId, VNextError> {
    type F = QwenImage21TransformerFamily;
    if !source.states().is_empty() {
        return Err(invalid("program", "block template must be stateless"));
    }
    let mut map = BTreeMap::from([
        (ProgramValueId::new(Block::INPUT_ID)?, input),
        (
            ProgramValueId::new(Block::FREQUENCIES_ID)?,
            ProgramValueId::new(F::FREQUENCIES_ID)?,
        ),
    ]);
    for (source, target) in [
        Block::SCALE1_ID,
        Block::GATE1_ID,
        Block::SCALE2_ID,
        Block::GATE2_ID,
    ]
    .into_iter()
    .zip(program::CONDITIONING)
    {
        map.insert(ProgramValueId::new(source)?, ProgramValueId::new(target)?);
    }
    if map.keys().cloned().collect::<BTreeSet<_>>() != source.inputs().iter().cloned().collect() {
        return Err(invalid(
            "program",
            "block entry remapping does not cover all inputs",
        ));
    }
    for weight in source.weights() {
        let suffix = weight
            .weight_id
            .as_str()
            .strip_prefix("weight.qwen_image21.transformer_blocks.0.")
            .ok_or_else(|| invalid("weights", "block template contains a non-block-zero weight"))?;
        let name = format!("transformer_blocks.{layer}.{suffix}");
        let destination = WeightId::new(format!("weight.qwen_image21.{name}"))?;
        let checked = weights
            .iter()
            .find(|candidate| candidate.weight_id == destination)
            .ok_or_else(|| {
                invalid(
                    "weights",
                    "mapped layer weight is absent from complete schema",
                )
            })?;
        if checked.tensor != weight.tensor {
            return Err(invalid(
                "weights",
                "mapped layer weight changes tensor structure",
            ));
        }
        if map
            .insert(weight.value_id.clone(), checked.value_id.clone())
            .is_some()
        {
            return Err(invalid(
                "program",
                "block weight value conflicts with an entry",
            ));
        }
    }
    let mut shared_gate_nodes = BTreeSet::new();
    for (index, node) in source
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .enumerate()
    {
        if let Some(destination) = shared_gate_destination(node)? {
            if map
                .insert(node.outputs[0].clone(), ProgramValueId::new(destination)?)
                .is_some()
            {
                return Err(invalid("program", "shared gate has a duplicate producer"));
            }
            shared_gate_nodes.insert(index);
            continue;
        }
        for (output_index, output) in node.outputs.iter().enumerate() {
            let destination = ProgramValueId::new(format!(
                "value.qwen_image21.transformer.layer{layer}.node{index}.output{output_index}"
            ))?;
            if map.insert(output.clone(), destination).is_some() {
                return Err(invalid(
                    "program",
                    "block producer has a duplicate identity",
                ));
            }
        }
    }
    if shared_gate_nodes.len() != program::TANH_GATES.len() {
        return Err(invalid(
            "program",
            "block template lacks the two shared gate producers",
        ));
    }
    for (index, source_node) in source
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .enumerate()
    {
        if shared_gate_nodes.contains(&index) {
            continue;
        }
        if source_node.work != ProgramNodeWorkSpec::Fixed {
            return Err(invalid("program", "block template introduced token work"));
        }
        let mut node = source_node.clone();
        node.id = NodeId::new(format!(
            "node.qwen_image21.transformer.layer{layer}.node{index}"
        ))?;
        node.inputs = source_node
            .inputs
            .iter()
            .map(|id| {
                map.get(id)
                    .cloned()
                    .ok_or_else(|| invalid("program", "unmapped block input"))
            })
            .collect::<Result<_, _>>()?;
        node.outputs = source_node
            .outputs
            .iter()
            .map(|id| {
                map.get(id)
                    .cloned()
                    .ok_or_else(|| invalid("program", "unmapped block output"))
            })
            .collect::<Result<_, _>>()?;
        use super::super::QwenImage21Block0AttentionFamily as Attention;
        if config.is_decode()
            && source_node
                .outputs
                .iter()
                .any(|v| v.as_str() == Attention::ATTENTION_ID)
        {
            if node.inputs.len() != 3 || node.outputs.len() != 1 {
                return Err(invalid(
                    "program",
                    "attention template changed its port contract",
                ));
            }
            for (input_index, part) in [
                (1, QwenImage21PrefixPart::Key),
                (2, QwenImage21PrefixPart::Value),
            ] {
                let joined = prefix::joined_value(layer, part)?;
                nodes.push(ProgramNode {
                    id: NodeId::new(format!(
                        "node.qwen_image21.transformer.layer{layer}.join_prefix_{input_index}"
                    ))?,
                    operation_id: OperationId::new(BATCHED_ROW_CONCAT_BF16_OPERATION_ID)?,
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![
                        prefix::cache_value(layer, part)?,
                        node.inputs[input_index].clone(),
                    ],
                    outputs: vec![joined.clone()],
                    attributes: BTreeMap::from([(
                        AttributeId::new("batch")?,
                        SemanticValue::Unsigned(config.layout().batch()),
                    )]),
                });
                node.inputs[input_index] = joined;
            }
            node.operation_id = OperationId::new(
                if source_node.operation_id.as_str() == STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID
                {
                    CROSS_ATTENTION_FUSED_BF16_OPERATION_ID
                } else {
                    CROSS_ATTENTION_BF16_OPERATION_ID
                },
            )?;
            node.attributes = config.cross_attention_attributes()?;
        }
        let capture = if config.execution_phase() == QwenImage21TransformerPhase::PrefixPrefill {
            source_node.outputs.iter().find_map(|v| match v.as_str() {
                Attention::KEY_ROTATED_ID => Some(QwenImage21PrefixPart::Key),
                Attention::VALUE_ID => Some(QwenImage21PrefixPart::Value),
                _ => None,
            })
        } else {
            None
        };
        let produced = node.outputs[0].clone();
        nodes.push(node);
        if let Some(part) = capture {
            let output = prefix::cache_value(layer, part)?;
            nodes.push(ProgramNode {
                id: NodeId::new(format!(
                    "node.qwen_image21.transformer.layer{layer}.capture_{}",
                    if part == QwenImage21PrefixPart::Key {
                        "key"
                    } else {
                        "value"
                    }
                ))?,
                operation_id: OperationId::new(ROW_GATHER_BF16_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![produced],
                outputs: vec![output],
                attributes: program::gather_attributes(
                    config.layout().batch(),
                    0,
                    config.transformer().hidden_size(),
                    &[(
                        config.layout().target_start(),
                        0,
                        config.layout().sequence_length(),
                        1,
                    )],
                )?,
            });
        }
    }
    map.get(&ProgramValueId::new(Block::OUTPUT_ID)?)
        .cloned()
        .ok_or_else(|| invalid("program", "block template lacks final residual"))
}

/// Sharing is authorized only for the exact two existing unary operations.
/// A future block with different gate arithmetic must not silently reuse them.
pub(super) fn shared_gate_destination(
    node: &ProgramNode,
) -> Result<Option<&'static str>, VNextError> {
    for (index, (output, input)) in [
        (Block::TANH_GATE1_ID, Block::GATE1_ID),
        (Block::TANH_GATE2_ID, Block::GATE2_ID),
    ]
    .into_iter()
    .enumerate()
    {
        if !node.outputs.iter().any(|value| value.as_str() == output) {
            continue;
        }
        if node.operation_id.as_str() != TANH_BF16_OPERATION_ID
            || node.required_version != ContractVersion::new(1, 0)
            || node.work != ProgramNodeWorkSpec::Fixed
            || node.inputs.len() != 1
            || node.inputs[0].as_str() != input
            || node.outputs.len() != 1
            || !node.attributes.is_empty()
        {
            return Err(invalid(
                "program",
                "shared gate producer differs from unary BF16 tanh",
            ));
        }
        return Ok(Some(program::TANH_GATES[index]));
    }
    Ok(None)
}
