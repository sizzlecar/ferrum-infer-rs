use super::*;
pub(super) fn entries(c: &PH3ImageFinalLayerConfig) -> Vec<(&'static str, Vec<u64>, ElementType)> {
    vec![
        ("final_layer.norm.weight", vec![c.hidden], ElementType::Bf16),
        (
            "final_layer.adaln_proj.linear.weight",
            vec![2 * c.hidden, c.basis],
            ElementType::F32,
        ),
        (
            "final_layer.adaln_proj.linear.bias",
            vec![2 * c.hidden],
            ElementType::F32,
        ),
        (
            "final_layer.video_out.weight",
            vec![c.image_width, c.hidden],
            ElementType::F32,
        ),
        (
            "final_layer.video_out.bias",
            vec![c.image_width],
            ElementType::F32,
        ),
        (
            "final_layer.audio_out.weight",
            vec![c.audio_width, c.hidden],
            ElementType::F32,
        ),
        (
            "final_layer.audio_out.bias",
            vec![c.audio_width],
            ElementType::F32,
        ),
    ]
}
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.final_layer.weight.{name}"))
}
fn weight(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.p_h3_image.final_layer.{name}"))
}
pub(super) fn schema(
    c: &PH3ImageFinalLayerConfig,
    capacity: u64,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut bytes = 0u64;
    for (name, dimensions, element_type) in entries(c) {
        bytes = bytes
            .checked_add(tensor(dimensions.clone(), element_type).byte_len()?)
            .filter(|n| *n <= capacity)
            .ok_or_else(|| invalid("stored final weights exceed capacity"))?;
        let component_id = WeightId::new(format!("component.p_h3_image.final_layer.{name}"))?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.into()],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense { element_type },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: weight(name)?,
            dimensions,
            logical_element_type: element_type,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.final_layer.comfy_curve.single_head.v1",
        )?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
pub(super) fn references(c: &PH3ImageFinalLayerConfig) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)
        .into_iter()
        .map(|(name, dimensions, element_type)| {
            Ok(WeightReference {
                weight_id: weight(name)?,
                value_id: value(name)?,
                tensor: tensor(dimensions, element_type),
            })
        })
        .collect()
}
