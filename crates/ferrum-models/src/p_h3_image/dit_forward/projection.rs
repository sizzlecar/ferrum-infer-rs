use super::*;
pub(super) const IMAGE_EMBEDDED: &str = "value.p_h3_image.dit_forward.image_embedded";
pub(super) const AUDIO_EMBEDDED: &str = "value.p_h3_image.dit_forward.audio_embedded";
pub(super) fn entries(c: &PH3ImageDiTConfig) -> Result<Vec<(&'static str, Vec<u64>)>, VNextError> {
    Ok(vec![
        ("video_patch_proj.weight", vec![c.hidden, c.image_width()?]),
        ("video_patch_proj.bias", vec![c.hidden]),
        ("audio_patch_proj.weight", vec![c.hidden, c.audio_channels]),
        ("audio_patch_proj.bias", vec![c.hidden]),
    ])
}
fn weight(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.dit_forward.weight.{name}"))
}
pub(super) fn build(c: &PH3ImageDiTConfig) -> Result<fragment::Fragment, VNextError> {
    let mut components = vec![];
    let mut tensors = vec![];
    let mut weights = vec![];
    for (name, dimensions) in entries(c)? {
        let id = WeightId::new(format!("component.p_h3_image.dit_forward.{name}"))?;
        let tensor_id = WeightId::new(format!("weight.p_h3_image.dit_forward.{name}"))?;
        components.push(WeightComponentSpec {
            id: id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.into()],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: tensor_id.clone(),
            dimensions: dimensions.clone(),
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Dense { component_id: id },
            required: true,
        });
        weights.push(WeightReference {
            weight_id: tensor_id,
            value_id: weight(name)?,
            tensor: config::tensor(dimensions, ElementType::F32),
        });
    }
    let mut nodes = vec![];
    let mut specs = BTreeMap::new();
    let mut boundaries = BTreeMap::new();
    for (name, module, input, output, rows, width) in [
        (
            "image",
            "video_patch_proj",
            PH3ImageDiTFamily::IMAGE_PATCHES_ID,
            IMAGE_EMBEDDED,
            c.image_rows()?,
            c.image_width()?,
        ),
        (
            "audio",
            "audio_patch_proj",
            PH3ImageDiTFamily::AUDIO_PATCHES_ID,
            AUDIO_EMBEDDED,
            c.audio_rows()?,
            c.audio_channels,
        ),
    ] {
        let projected =
            ProgramValueId::new(format!("value.p_h3_image.dit_forward.{name}_projected_f32"))?;
        let input = ProgramValueId::new(input)?;
        let output = ProgramValueId::new(output)?;
        nodes.push(node(
            &format!("{name}_project"),
            AFFINE_LINEAR_F32_OPERATION_ID,
            vec![
                input.clone(),
                weight(&format!("{module}.weight"))?,
                weight(&format!("{module}.bias"))?,
            ],
            projected.clone(),
            BTreeMap::from([
                (
                    AttributeId::new("in_features")?,
                    SemanticValue::Unsigned(width),
                ),
                (
                    AttributeId::new("out_features")?,
                    SemanticValue::Unsigned(c.hidden),
                ),
            ]),
        )?);
        nodes.push(node(
            &format!("{name}_round"),
            CAST_F32_TO_BF16_OPERATION_ID,
            vec![projected.clone()],
            output.clone(),
            BTreeMap::new(),
        )?);
        specs.insert(input, config::tensor(vec![rows, width], ElementType::F32));
        specs.insert(
            projected.clone(),
            config::tensor(vec![rows, c.hidden], ElementType::F32),
        );
        boundaries.insert(projected, ElementType::F32);
        specs.insert(
            output.clone(),
            config::tensor(vec![rows, c.hidden], ElementType::Bf16),
        );
        boundaries.insert(output, ElementType::Bf16);
    }
    let operations = [
        (AFFINE_LINEAR_F32_OPERATION_ID, true),
        (CAST_F32_TO_BF16_OPERATION_ID, false),
    ]
    .into_iter()
    .map(|(id, math)| {
        Ok(NumericalOperationContract {
            operation_id: OperationId::new(id)?,
            version: ContractVersion::new(1, 0),
            multiplication_type: math.then_some(ElementType::F32),
            accumulation_type: math.then_some(ElementType::F32),
            staged_arithmetic: None,
            composite_arithmetic: None,
        })
    })
    .collect::<Result<_, VNextError>>()?;
    Ok(fragment::Fragment {
        schema: WeightSchema {
            format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
            layout_id: WeightLayoutId::new("weight-layout.p_h3_image.dit_forward.patch_f32.v1")?,
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        },
        nodes,
        weights,
        specs,
        operations,
        boundaries,
    })
}
pub(super) fn node(
    name: &str,
    operation: &str,
    inputs: Vec<ProgramValueId>,
    output: ProgramValueId,
    attributes: BTreeMap<AttributeId, SemanticValue>,
) -> Result<ProgramNode, VNextError> {
    Ok(ProgramNode {
        id: NodeId::new(format!("node.p_h3_image.dit_forward.{name}"))?,
        operation_id: OperationId::new(operation)?,
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes,
    })
}
