use super::*;
pub(super) struct Entry {
    pub name: String,
    pub original: Vec<u64>,
    pub physical: Vec<u64>,
}
fn conv(entries: &mut Vec<Entry>, name: String, input: u64, output: u64, kernel: u64) {
    entries.push(Entry {
        name: format!("{name}.weight"),
        original: vec![output, input, kernel, kernel, kernel],
        physical: vec![output, input, kernel, kernel],
    });
    entries.push(Entry {
        name: format!("{name}.bias"),
        original: vec![output],
        physical: vec![output],
    });
}
fn norm(entries: &mut Vec<Entry>, name: String, channels: u64) {
    for parameter in ["weight", "bias"] {
        entries.push(Entry {
            name: format!("{name}.{parameter}"),
            original: vec![channels],
            physical: vec![channels],
        });
    }
}
pub(super) fn entries(c: &PH3ImageVaeEncoderConfig) -> Result<Vec<Entry>, VNextError> {
    c.validate()?;
    let a = c.architecture();
    let mut entries = Vec::new();
    let mut input = a.channels()[0];
    conv(
        &mut entries,
        "encoder.conv_in".into(),
        a.in_channels(),
        input,
        3,
    );
    for (level, &output) in a.channels().iter().enumerate() {
        for block in 0..a.layers_per_block() {
            let prefix = format!("encoder.down.{level}.block.{block}");
            norm(&mut entries, format!("{prefix}.norm1"), input);
            conv(&mut entries, format!("{prefix}.conv1"), input, output, 3);
            norm(&mut entries, format!("{prefix}.norm2"), output);
            conv(&mut entries, format!("{prefix}.conv2"), output, output, 3);
            if input != output {
                conv(
                    &mut entries,
                    format!("{prefix}.nin_shortcut"),
                    input,
                    output,
                    1,
                );
            }
            input = output;
        }
        if a.spatial_downsample_factors()[level] > 1 || a.temporal_downsample_factors()[level] > 1 {
            conv(
                &mut entries,
                format!("encoder.down.{level}.downsample.conv"),
                input,
                input,
                3,
            );
        }
    }
    norm(&mut entries, "encoder.norm_out".into(), input);
    let moments = a
        .latent_channels()
        .checked_mul(2)
        .ok_or_else(|| invalid("moment channels overflow"))?;
    conv(&mut entries, "encoder.conv_out".into(), input, moments, 3);
    conv(&mut entries, "quant_conv".into(), moments, moments, 1);
    Ok(entries)
}
pub(super) fn value(name: &str) -> Result<ProgramValueId, VNextError> {
    ProgramValueId::new(format!("value.p_h3_image.vae.encoder.weight.{name}"))
}
fn weight(name: &str) -> Result<WeightId, VNextError> {
    WeightId::new(format!("weight.p_h3_image.vae.encoder.{name}"))
}
pub(super) fn schema(
    c: &PH3ImageVaeEncoderConfig,
    capacity: u64,
    family: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let mut total = 0u64;
    let mut components = Vec::new();
    let mut tensors = Vec::new();
    for e in entries(c)? {
        let component_id = WeightId::new(format!("component.p_h3_image.vae.encoder.{}", e.name))?;
        total = total
            .checked_add(config::tensor(e.physical.clone()).byte_len()?)
            .filter(|n| *n <= capacity)
            .ok_or_else(|| invalid("physical full CNN weights exceed capacity"))?;
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
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense")?,
        layout_id: WeightLayoutId::new(
            "weight-layout.p_h3_image.vae.encoder.full_cnn.last_tap.f32",
        )?,
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    schema.validate(family)?;
    Ok(schema)
}
pub(super) fn references(c: &PH3ImageVaeEncoderConfig) -> Result<Vec<WeightReference>, VNextError> {
    entries(c)?
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
