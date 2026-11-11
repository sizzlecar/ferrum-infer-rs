use super::*;
use ferrum_quantization::comfy_convrot::COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID;
pub(super) fn external(c: &PH3ImageVaeDecoderBlockConfig, suffix: &str) -> String {
    format!("decoder.transformer_blocks.{}.{suffix}", c.block_index())
}
pub(super) fn projections(
    c: &PH3ImageVaeDecoderBlockConfig,
) -> Result<[(&'static str, u64, u64); 4], VNextError> {
    let h = c.architecture().hidden_size()?;
    let m = c.architecture().intermediate_size()?;
    Ok([
        (
            "attn.to_qkv",
            h,
            h.checked_mul(3)
                .ok_or_else(|| invalid("architecture", "QKV width overflows"))?,
        ),
        ("attn.to_out", h, h),
        (
            "ff.w1",
            h,
            m.checked_mul(2)
                .ok_or_else(|| invalid("architecture", "gate/up width overflows"))?,
        ),
        ("ff.w2", m, h),
    ])
}
pub(super) fn original_bf16_vectors(
    c: &PH3ImageVaeDecoderBlockConfig,
) -> Result<Vec<(String, Vec<u64>)>, VNextError> {
    let h = c.architecture().hidden_size()?;
    let mut vectors = ["norm1.weight", "norm2.weight", "scale1", "scale2"]
        .into_iter()
        .map(|name| (name.to_owned(), vec![h]))
        .collect::<Vec<_>>();
    for (name, _, output) in projections(c)? {
        vectors.push((format!("{name}.bias"), vec![output]));
    }
    Ok(vectors)
}
pub(super) fn entries(
    c: &PH3ImageVaeDecoderBlockConfig,
) -> Result<Vec<(String, Vec<u64>, ElementType)>, VNextError> {
    let mut result = original_bf16_vectors(c)?
        .into_iter()
        .map(|(name, dims)| (name, dims, ElementType::F32))
        .collect::<Vec<_>>();
    for (module, input, output) in projections(c)? {
        result.push((
            format!("{module}.weight"),
            vec![output, input],
            ElementType::I8,
        ));
        result.push((
            format!("{module}.weight_scale"),
            vec![output, 1],
            ElementType::F32,
        ));
    }
    Ok(result)
}
pub(super) fn value(suffix: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!(
        "value.p_h3_image.vae.decoder_block.weight.{suffix}"
    ))
}
fn weight(suffix: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.p_h3_image.vae.decoder_block.{suffix}"))
}
pub(super) fn schema(
    c: &PH3ImageVaeDecoderBlockConfig,
    capacity: u64,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut bytes = 0u64;
    for (name, dimensions, element_type) in entries(c)? {
        let tensor = ProgramTensorSpec {
            dimensions: dimensions.clone(),
            element_type,
            layout: ResolvedTensorLayout::Contiguous,
        };
        bytes = bytes
            .checked_add(tensor.byte_len()?)
            .filter(|n| *n <= capacity)
            .ok_or_else(|| invalid("weights", "stored weights exceed declared capacity"))?;
        let component_id = WeightId::new(format!("component.p_h3_image.vae.decoder_block.{name}"))?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![external(c, &name)],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense { element_type },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: weight(&name)?,
            dimensions,
            logical_element_type: element_type,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new(COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID)?,
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.vae.decoder_block.comfy_int8.f32_promoted.v1",
        )?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
pub(super) fn references(
    c: &PH3ImageVaeDecoderBlockConfig,
) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)?
        .into_iter()
        .map(|(name, dimensions, element_type)| {
            Ok(WeightReference {
                weight_id: weight(&name)?,
                value_id: value(&name)?,
                tensor: ProgramTensorSpec {
                    dimensions,
                    element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
        })
        .collect()
}
