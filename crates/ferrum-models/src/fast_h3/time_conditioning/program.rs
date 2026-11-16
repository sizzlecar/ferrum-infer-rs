use super::*;

fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
pub(super) fn build(
    c: &FastH3TimeConditioningConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    type F = FastH3TimeConditioningFamily;
    let input = ProgramValueId::new(F::INPUT_ID)?;
    let embedding = ProgramValueId::new(F::EMBEDDING_ID)?;
    let output = ProgramValueId::new(F::OUTPUT_ID)?;
    let input_specs = BTreeMap::from([(
        input.clone(),
        tensor(vec![c.rows, c.frequency], ElementType::F32),
    )]);
    let mut specs = input_specs.clone();
    let weight_values = (0..6)
        .map(|i| value(&format!("weight.{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let weights = c
        .components
        .iter()
        .enumerate()
        .map(|(i, component)| {
            Ok(WeightReference {
                weight_id: weight(i)?,
                value_id: weight_values[i].clone(),
                tensor: tensor(
                    component.dimensions.clone(),
                    component.physical_element_type(),
                ),
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let mut nodes = Vec::new();
    let mut push = |name: &str,
                    op: &str,
                    inputs,
                    out: ProgramValueId,
                    width,
                    dtype,
                    attributes|
     -> Result<(), VNextError> {
        nodes.push(ProgramNode {
            id: NodeId::new(format!("node.fast_h3.time_conditioning.{name}"))?,
            operation_id: OperationId::new(op)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![out.clone()],
            attributes,
        });
        specs.insert(out, tensor(vec![c.rows, width], dtype));
        Ok(())
    };
    let affine_attrs = |k, n| -> Result<_, VNextError> {
        Ok(BTreeMap::from([
            (AttributeId::new("in_features")?, SemanticValue::Unsigned(k)),
            (
                AttributeId::new("out_features")?,
                SemanticValue::Unsigned(n),
            ),
        ]))
    };
    let expanded = value("expanded")?;
    let activated = value("activated")?;
    let modulation_f32 = value("modulation_input_f32")?;
    let modulation_bf16 = value("modulation_input_bf16")?;
    push(
        "expand",
        AFFINE_LINEAR_F32_OPERATION_ID,
        vec![
            input.clone(),
            weight_values[0].clone(),
            weight_values[1].clone(),
        ],
        expanded.clone(),
        c.intermediate,
        ElementType::F32,
        affine_attrs(c.frequency, c.intermediate)?,
    )?;
    push(
        "activate",
        SILU_F32_OPERATION_ID,
        vec![expanded],
        activated.clone(),
        c.intermediate,
        ElementType::F32,
        BTreeMap::new(),
    )?;
    push(
        "embed",
        AFFINE_LINEAR_F32_OPERATION_ID,
        vec![
            activated,
            weight_values[2].clone(),
            weight_values[3].clone(),
        ],
        embedding.clone(),
        c.embedding,
        ElementType::F32,
        affine_attrs(c.intermediate, c.embedding)?,
    )?;
    push(
        "modulation_activate",
        SILU_F32_OPERATION_ID,
        vec![embedding.clone()],
        modulation_f32.clone(),
        c.embedding,
        ElementType::F32,
        BTreeMap::new(),
    )?;
    push(
        "modulation_cast",
        CAST_F32_TO_BF16_OPERATION_ID,
        vec![modulation_f32],
        modulation_bf16.clone(),
        c.embedding,
        ElementType::Bf16,
        BTreeMap::new(),
    )?;
    let width = c
        .hidden
        .checked_mul(18)
        .ok_or_else(|| invalid("modulation width overflow"))?;
    push(
        "modulate",
        AFFINE_LINEAR_BF16_OPERATION_ID,
        vec![
            modulation_bf16,
            weight_values[4].clone(),
            weight_values[5].clone(),
        ],
        output.clone(),
        width,
        ElementType::Bf16,
        affine_attrs(c.embedding, width)?,
    )?;
    let program = ModelProgram::new(
        ModelFamilyId::new(F::FAMILY_ID)?,
        vec![input],
        vec![ProgramBlock {
            id: "block.fast_h3.time_conditioning".into(),
            nodes,
        }],
        vec![],
        weights,
        vec![embedding, output],
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok((program, specs))
}
