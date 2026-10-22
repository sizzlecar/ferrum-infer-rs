use super::*;

/// Reuse one validated block template. Every source input, weight and produced
/// value has an explicit destination; no unrecognized value survives remapping.
pub(super) fn append(
    nodes: &mut Vec<ProgramNode>,
    source: &ModelProgram,
    weights: &[WeightReference],
    layer: u64,
    input: ProgramValueId,
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
    for (index, node) in source
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .enumerate()
    {
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
    for (index, source_node) in source
        .blocks()
        .iter()
        .flat_map(|block| &block.nodes)
        .enumerate()
    {
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
        nodes.push(node);
    }
    map.get(&ProgramValueId::new(Block::OUTPUT_ID)?)
        .cloned()
        .ok_or_else(|| invalid("program", "block template lacks final residual"))
}
