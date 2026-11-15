use super::*;
fn tensor(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn node(
    name: &str,
    op: &str,
    inputs: Vec<ProgramValueId>,
    outputs: Vec<ProgramValueId>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.fast_h3.attention_projection.{name}"))?,
        operation_id: OperationId::new(op)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs,
        attributes,
    })
}
pub(super) fn build(
    c: &FastH3AttentionProjectionConfig,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let input = ProgramValueId::new(FastH3AttentionProjectionFamily::INPUT_ID)?;
    let packed = value("activation.packed")?;
    let scales = value("activation.scales")?;
    let input_specs = BTreeMap::from([(
        input.clone(),
        tensor(vec![c.rows, c.hidden], ElementType::Bf16),
    )]);
    let mut specs = input_specs.clone();
    let geometry = Nvfp4MatrixLayout::new(c.rows, c.hidden)?;
    specs.insert(
        packed.clone(),
        tensor(geometry.packed_shape().to_vec(), ElementType::U8),
    );
    specs.insert(
        scales.clone(),
        tensor(geometry.scale_shape().to_vec(), ElementType::U8),
    );
    let schema = c.schema()?;
    let weights = schema
        .tensors
        .iter()
        .map(|w| {
            let suffix =
                w.id.as_str()
                    .strip_prefix("weight.fast_h3.attention_projection.")
                    .ok_or_else(|| invalid("projection weight ID"))?;
            Ok(WeightReference {
                weight_id: w.id.clone(),
                value_id: value(suffix)?,
                tensor: tensor(w.dimensions.clone(), w.logical_element_type),
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    let mut nodes = vec![node(
        "quantize",
        NVFP4_QUANTIZE_BF16_OPERATION_ID,
        vec![input.clone(), value("unit_sf")?],
        vec![packed.clone(), scales.clone()],
        Nvfp4QuantizeProfile::CudaRcpApproxFtzV1.attributes()?,
    )?];
    let mut outputs = Vec::with_capacity(4);
    for p in &c.projections {
        let output = ProgramValueId::new(p.kind.output_id())?;
        let operand = |suffix| value(&format!("{}.{}", p.kind.module(), suffix));
        nodes.push(node(
            p.kind.module(),
            NVFP4_MATMUL_BF16_OPERATION_ID,
            vec![
                packed.clone(),
                operand("packed")?,
                scales.clone(),
                operand("scales")?,
                operand("alpha")?,
            ],
            vec![output.clone()],
            BTreeMap::new(),
        )?);
        specs.insert(
            output.clone(),
            tensor(vec![c.rows, c.output_width()?], ElementType::Bf16),
        );
        outputs.push(output);
    }
    let program = ModelProgram::new(
        ModelFamilyId::new(FastH3AttentionProjectionFamily::FAMILY_ID)?,
        vec![input],
        vec![ProgramBlock {
            id: "block.fast_h3.attention_projection".into(),
            nodes,
        }],
        vec![],
        weights,
        outputs,
    )?
    .with_input_tensor_specs(input_specs)?;
    Ok((program, specs))
}
