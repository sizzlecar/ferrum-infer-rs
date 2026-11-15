use super::*;
pub(super) fn node(
    prefix: &str,
    name: &str,
    op: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: id(&format!("node.fast_h3.{prefix}.{name}"))?,
        operation_id: id(op)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
pub(super) fn build(
    c: &FastH3InputProjectionConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    c.check_inventory()?;
    type F = FastH3InputProjectionFamily;
    let mut specs = BTreeMap::new();
    let mut inputs = Vec::new();
    let mut nodes = Vec::new();
    let mut outputs = Vec::new();
    let ports = [
        (F::VIDEO_INPUT_ID, F::VIDEO_OUTPUT_ID),
        (F::AUDIO_INPUT_ID, F::AUDIO_OUTPUT_ID),
        (F::TEXT_INPUT_ID, F::PROJECTED_TEXT_ID),
    ];
    for (local, &i) in c.phase.streams().iter().enumerate() {
        let (input, output) = ports[i];
        let input: ProgramValueId = id(input)?;
        let output: ProgramValueId = id(output)?;
        let dtype = if i < 2 {
            ElementType::F32
        } else {
            ElementType::Bf16
        };
        specs.insert(
            input.clone(),
            tensor(vec![c.rows[local], c.widths[local]], dtype),
        );
        inputs.push(input.clone());
        nodes.push(node(
            "input_projection",
            &format!("project_{i}"),
            if i < 2 {
                AFFINE_LINEAR_F32_TO_BF16_OPERATION_ID
            } else {
                AFFINE_LINEAR_BF16_OPERATION_ID
            },
            vec![
                input,
                id(&format!("value.fast_h3.input_projection.weight{}", 2 * i))?,
                id(&format!(
                    "value.fast_h3.input_projection.weight{}",
                    2 * i + 1
                ))?,
            ],
            output.clone(),
            BTreeMap::from([
                (id("in_features")?, SemanticValue::Unsigned(c.widths[local])),
                (id("out_features")?, SemanticValue::Unsigned(c.hidden)),
            ]),
        )?);
        specs.insert(
            output.clone(),
            tensor(vec![c.rows[local], c.hidden], ElementType::Bf16),
        );
        outputs.push(output);
    }
    let input_specs = inputs
        .iter()
        .map(|i| (i.clone(), specs[i].clone()))
        .collect();
    let weights = c
        .components
        .iter()
        .enumerate()
        .map(|(local, component)| {
            let i = c.parameter_index(local);
            Ok(WeightReference {
                weight_id: id(&format!("weight.fast_h3.input_projection.{i}"))?,
                value_id: id(&format!("value.fast_h3.input_projection.weight{i}"))?,
                tensor: tensor(
                    component.dimensions.clone(),
                    component.physical_element_type(),
                ),
            })
        })
        .collect::<Result<_, VNextError>>()?;
    Ok((
        ModelProgram::new(
            id(F::FAMILY_ID)?,
            inputs,
            vec![ProgramBlock {
                id: "block.fast_h3.input_projection".into(),
                nodes,
            }],
            vec![],
            weights,
            outputs,
        )?
        .with_input_tensor_specs(input_specs)?,
        specs,
    ))
}
