use super::*;
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.curve_block.weight.{name}"))
}
pub(super) fn external(c: &PH3ImageCurveBlockConfig, name: &str) -> String {
    format!("blocks.{}.{name}", c.block_index)
}
pub(super) fn entries(c: &PH3ImageCurveBlockConfig) -> Vec<(String, Vec<u64>, ElementType)> {
    let h = c.hidden;
    let inner = c.attention_inner_dim();
    let mut entries = vec![
        ("norm1.weight".into(), vec![h], ElementType::Bf16),
        ("norm2.weight".into(), vec![h], ElementType::Bf16),
        (
            "attn.q_norm.weight".into(),
            vec![c.head_dim],
            ElementType::Bf16,
        ),
        (
            "attn.k_norm.weight".into(),
            vec![c.head_dim],
            ElementType::Bf16,
        ),
        (
            "adaln_proj.linear.weight".into(),
            vec![18 * h, c.basis],
            ElementType::F32,
        ),
        (
            "adaln_proj.linear.bias".into(),
            vec![18 * h],
            ElementType::F32,
        ),
    ];
    for (name, input, output) in [
        ("attn.qkv_proj", h, 3 * inner),
        ("attn.out_proj", inner, h),
        ("mlp.fc1", h, 2 * c.ffn),
        ("mlp.fc2", c.ffn, h),
    ] {
        let components = match c
            .projection_storage
            .module(name)
            .expect("declared projection role")
        {
            PH3ImageProjectionStorage::CodebookW4A8 => vec![
                ("weight", vec![output, input / 2], ElementType::I8),
                ("weight_s_rel", vec![output, input / 16], ElementType::U8),
                ("weight_s_channel", vec![output], ElementType::F32),
                ("weight_codebook", vec![16], ElementType::F32),
            ],
            PH3ImageProjectionStorage::Int8W8A8 => vec![
                ("weight", vec![output, input], ElementType::I8),
                ("weight_scale", vec![output, 1], ElementType::F32),
            ],
        };
        for (suffix, dimensions, dtype) in components {
            entries.push((format!("{name}.{suffix}"), dimensions, dtype));
        }
    }
    entries
}
pub(super) fn schema(
    c: &PH3ImageCurveBlockConfig,
    capacity: u64,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut bytes = 0u64;
    for (name, dimensions, dtype) in entries(c) {
        let spec = config::tensor(dimensions.clone(), dtype);
        spec.validate(&name)?;
        bytes = bytes
            .checked_add(spec.byte_len()?)
            .filter(|b| *b <= capacity)
            .ok_or_else(|| invalid("block weights exceed declared capacity"))?;
        let component = WeightId::new(format!("component.p_h3_image.curve_block.{name}"))?;
        components.push(WeightComponentSpec {
            id: component.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![external(c, &name)],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: dtype,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: WeightId::new(format!("weight.p_h3_image.curve_block.{name}"))?,
            dimensions,
            logical_element_type: dtype,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: component,
            },
            required: true,
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new(if c.projection_storage.all_w4() {
            "weight-layout.p_h3_image.curve_block.canonical_w4_f32.v1"
        } else {
            "weight-layout.p_h3_image.curve_block.canonical_mixed_convrot_f32.v1"
        })?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(&ModelFamilyId::new(PH3ImageCurveBlockFamily::FAMILY_ID)?)?;
    Ok(schema)
}
pub(super) fn references(c: &PH3ImageCurveBlockConfig) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)
        .into_iter()
        .map(|(name, dimensions, dtype)| {
            Ok(WeightReference {
                weight_id: WeightId::new(format!("weight.p_h3_image.curve_block.{name}"))?,
                value_id: value(&name)?,
                tensor: config::tensor(dimensions, dtype),
            })
        })
        .collect()
}
