use super::*;
pub(super) struct Entry {
    pub name: String,
    pub original: Vec<u64>,
    pub physical: Vec<u64>,
}
pub(super) fn entries(c: &PH3ImageVaeEncoderVerticalConfig) -> Vec<Entry> {
    let (input, output) = c.widths();
    let prefix = format!("encoder.down.{}.block.0", c.block_level());
    let down = format!("encoder.down.{}.downsample.conv", c.block_level());
    let mut entries = Vec::new();
    let mut conv = |name: String, ic, oc, k| {
        entries.push(Entry {
            name: format!("{name}.weight"),
            original: vec![oc, ic, k, k, k],
            physical: vec![oc, ic, k, k],
        });
        entries.push(Entry {
            name: format!("{name}.bias"),
            original: vec![oc],
            physical: vec![oc],
        });
    };
    conv(
        "encoder.conv_in".into(),
        c.architecture().in_channels(),
        input,
        3,
    );
    conv(format!("{prefix}.conv1"), input, output, 3);
    conv(format!("{prefix}.conv2"), output, output, 3);
    conv(format!("{prefix}.nin_shortcut"), input, output, 1);
    conv(down, output, output, 3);
    for (norm, channels) in [("norm1", input), ("norm2", output)] {
        for parameter in ["weight", "bias"] {
            entries.push(Entry {
                name: format!("{prefix}.{norm}.{parameter}"),
                original: vec![channels],
                physical: vec![channels],
            });
        }
    }
    entries
}
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!(
        "value.p_h3_image.vae.encoder_vertical.weight.{name}"
    ))
}
fn weight(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.p_h3_image.vae.encoder_vertical.{name}"))
}
pub(super) fn schema(
    c: &PH3ImageVaeEncoderVerticalConfig,
    capacity: u64,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut total = 0u64;
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    for e in entries(c) {
        let component_id = WeightId::new(format!(
            "component.p_h3_image.vae.encoder_vertical.{}",
            e.name
        ))?;
        total = total
            .checked_add(config::tensor(e.physical.clone()).byte_len()?)
            .filter(|n| *n <= capacity)
            .ok_or_else(|| invalid("physical F32 weights exceed capacity"))?;
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![e.name.clone()],
            dimensions: e.physical.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: weight(&e.name)?,
            dimensions: e.physical,
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
    }
    let s = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.vae.encoder_vertical.last_tap.f32",
        )?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    s.validate(family)?;
    Ok(s)
}
pub(super) fn references(
    c: &PH3ImageVaeEncoderVerticalConfig,
) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)
        .into_iter()
        .map(|e| {
            Ok(WeightReference {
                weight_id: weight(&e.name)?,
                value_id: value(&e.name)?,
                tensor: config::tensor(e.physical),
            })
        })
        .collect()
}
