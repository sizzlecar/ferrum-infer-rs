use super::*;
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.token_refiner.weight.{name}"))
}
pub(super) fn entries(c: &PH3ImageTokenRefinerConfig) -> Vec<(String, Vec<u64>, ElementType)> {
    let mut entries = vec![
        (
            "condition_proj.weight".into(),
            vec![c.hidden, c.text_dim],
            ElementType::Bf16,
        ),
        (
            "condition_proj.bias".into(),
            vec![c.hidden],
            ElementType::Bf16,
        ),
        (
            "token_refiner.final_norm.weight".into(),
            vec![c.hidden],
            ElementType::Bf16,
        ),
    ];
    for layer in 0..c.layers {
        let prefix = format!("token_refiner.blocks.{layer}");
        for (name, width) in [
            ("norm1.weight", c.hidden),
            ("norm2.weight", c.hidden),
            ("attn.q_norm.weight", c.head_dim),
            ("attn.k_norm.weight", c.head_dim),
        ] {
            entries.push((format!("{prefix}.{name}"), vec![width], ElementType::Bf16));
        }
        for (name, input, output) in [
            ("attn.qkv_proj", c.hidden, 3 * c.inner()),
            ("attn.out_proj", c.inner(), c.hidden),
            ("mlp.fc1", c.hidden, 2 * c.ffn),
            ("mlp.fc2", c.ffn, c.hidden),
        ] {
            entries.push((
                format!("{prefix}.{name}.weight"),
                vec![output, input],
                ElementType::I8,
            ));
            entries.push((
                format!("{prefix}.{name}.weight_scale"),
                vec![output, 1],
                ElementType::F32,
            ));
        }
    }
    entries
}
pub(super) fn schema(
    c: &PH3ImageTokenRefinerConfig,
    capacity: u64,
) -> Result<WeightSchema, VNextError> {
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    let mut bytes = 0u64;
    for (name, dimensions, dtype) in entries(c) {
        let tensor = config::tensor(dimensions.clone(), dtype);
        tensor.validate(&name)?;
        bytes = bytes
            .checked_add(tensor.byte_len()?)
            .filter(|n| *n <= capacity)
            .ok_or_else(|| invalid("stored weights exceed declared capacity"))?;
        let component = WeightId::new(format!("component.p_h3_image.token_refiner.{name}"))?;
        components.push(WeightComponentSpec {
            id: component.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.clone()],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: dtype,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: WeightId::new(format!("weight.p_h3_image.token_refiner.{name}"))?,
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
        layout_id: WeightLayoutId::new("weight-layout.p_h3_image.token_refiner.canonical_w8.v1")?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(&ModelFamilyId::new(PH3ImageTokenRefinerFamily::FAMILY_ID)?)?;
    Ok(schema)
}
pub(super) fn references(
    c: &PH3ImageTokenRefinerConfig,
) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)
        .into_iter()
        .map(|(name, shape, dtype)| {
            Ok(WeightReference {
                weight_id: WeightId::new(format!("weight.p_h3_image.token_refiner.{name}"))?,
                value_id: value(&name)?,
                tensor: config::tensor(shape, dtype),
            })
        })
        .collect()
}
