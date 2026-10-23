use super::*;
pub(super) fn conv_attributes(
    kernel: u64,
) -> Result<BTreeMap<AttributeId, SemanticValue>, VNextError> {
    let pad = if kernel == 1 { 0 } else { 1 };
    Ok(BTreeMap::from([
        (
            AttributeId::new("padding")?,
            SemanticValue::Integers(vec![pad; 4]),
        ),
        (
            AttributeId::new("stride")?,
            SemanticValue::Integers(vec![1, 1]),
        ),
        (
            AttributeId::new("dilation")?,
            SemanticValue::Integers(vec![1, 1]),
        ),
        (AttributeId::new("groups")?, SemanticValue::Unsigned(1)),
        (
            AttributeId::new("padding_mode")?,
            SemanticValue::Text(Conv2dPaddingMode::ExplicitZero.as_str().to_string()),
        ),
    ]))
}
pub(super) fn build(
    c: &QwenImage21VaeStemConfig,
    family: &ModelFamilyId,
) -> Result<ModelProgram, VNextError> {
    c.validate()?;
    let input = ProgramValueId::new(QwenImage21VaeStemFamily::INPUT_ID)?;
    let out = QwenImage21VaeStemFamily::OUTPUT_IDS
        .into_iter()
        .map(ProgramValueId::new)
        .collect::<Result<Vec<_>, _>>()?;
    let weight = |n| weights::value_id(QwenImage21VaeStemFamily::WEIGHT_NAMES[n]);
    let node = |index: usize,
                name: &str,
                operation: &str,
                inputs: Vec<ProgramValueId>,
                attributes|
     -> Result<ProgramNode, VNextError> {
        Ok(ProgramNode {
            id: NodeId::new(format!("node.qwen_image21.vae.decoder_stem.{name}"))?,
            operation_id: OperationId::new(operation)?,
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::Fixed,
            inputs,
            outputs: vec![out[index].clone()],
            attributes,
        })
    };
    let nodes = vec![
        node(
            0,
            "post_quant",
            CONV2D_BF16_OPERATION_ID,
            vec![input.clone(), weight(0)?, weight(1)?],
            conv_attributes(1)?,
        )?,
        node(
            1,
            "conv_in",
            CONV2D_BF16_OPERATION_ID,
            vec![out[0].clone(), weight(2)?, weight(3)?],
            conv_attributes(3)?,
        )?,
        node(
            2,
            "norm1",
            CHANNEL_L2_NORM_BF16_OPERATION_ID,
            vec![out[1].clone(), weight(4)?],
            c.norm_attributes()?,
        )?,
        node(
            3,
            "silu1",
            SILU_NCHW_BF16_OPERATION_ID,
            vec![out[2].clone()],
            BTreeMap::new(),
        )?,
        node(
            4,
            "conv1",
            CONV2D_BF16_OPERATION_ID,
            vec![out[3].clone(), weight(5)?, weight(6)?],
            conv_attributes(3)?,
        )?,
        node(
            5,
            "norm2",
            CHANNEL_L2_NORM_BF16_OPERATION_ID,
            vec![out[4].clone(), weight(7)?],
            c.norm_attributes()?,
        )?,
        node(
            6,
            "silu2",
            SILU_NCHW_BF16_OPERATION_ID,
            vec![out[5].clone()],
            BTreeMap::new(),
        )?,
        node(
            7,
            "conv2",
            CONV2D_BF16_OPERATION_ID,
            vec![out[6].clone(), weight(8)?, weight(9)?],
            conv_attributes(3)?,
        )?,
        // Upstream returns x + h, where h is the unmodified conv_in output.
        node(
            8,
            "residual",
            PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
            vec![out[7].clone(), out[1].clone()],
            BTreeMap::new(),
        )?,
    ];
    let references = c
        .expected_weights()?
        .into_iter()
        .map(|w| {
            Ok(WeightReference {
                weight_id: w.weight_id()?,
                value_id: w.value_id()?,
                tensor: ProgramTensorSpec {
                    dimensions: w.dimensions().to_vec(),
                    element_type: ElementType::Bf16,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    ModelProgram::new(
        family.clone(),
        vec![input],
        vec![ProgramBlock {
            id: "block.qwen_image21.vae.decoder_stem".into(),
            nodes,
        }],
        vec![],
        references,
        out,
    )?
    .with_input_tensor_specs(c.input_tensor_specs()?)
}
