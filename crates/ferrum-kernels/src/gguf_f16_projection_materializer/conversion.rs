use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use half::f16;

pub(super) fn supported_format(
    spec: &BlockQuantizationSpec,
) -> Result<GgufBlockFormat, VNextError> {
    let format = GgufBlockFormat::from_spec(spec).map_err(invalid)?;
    if !matches!(
        format,
        GgufBlockFormat::Q4K | GgufBlockFormat::Q5K | GgufBlockFormat::Q6K | GgufBlockFormat::Q8_0
    ) {
        return Err(invalid(
            "GGUF RN-F16 projection format is outside the declared conversion contract",
        ));
    }
    Ok(format)
}

/// A single bounded F32 block is temporary; no second whole F32 matrix is
/// allocated. `half` implements IEEE RN-even, including ties/subnormals.
pub(super) fn convert(format: GgufBlockFormat, source: &[u8]) -> Result<Vec<u8>, VNextError> {
    if source.is_empty() || !source.len().is_multiple_of(format.block_bytes()) {
        return Err(invalid(
            "GGUF RN-F16 source must contain complete nonempty blocks",
        ));
    }
    let bytes = source
        .len()
        .checked_div(format.block_bytes())
        .and_then(|n| n.checked_mul(format.block_values()))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| invalid("GGUF RN-F16 converted allocation overflows"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| invalid("GGUF RN-F16 host allocation unavailable"))?;
    append_converted(format, source, &mut output)?;
    Ok(output)
}
fn append_converted(
    format: GgufBlockFormat,
    source: &[u8],
    output: &mut Vec<u8>,
) -> Result<(), VNextError> {
    let mut decoded = [0_f32; 256];
    for block in source.chunks_exact(format.block_bytes()) {
        let values = &mut decoded[..format.block_values()];
        format.decode(block, values).map_err(invalid)?;
        for &value in values.iter() {
            let converted = round_finite(value)?;
            output.extend_from_slice(&converted.to_le_bytes());
        }
    }
    Ok(())
}
pub(super) fn round_finite(value: f32) -> Result<u16, VNextError> {
    let half = f16::from_f32(value);
    if !value.is_finite() || !half.is_finite() {
        return Err(invalid(
            "GGUF RN-F16 source is nonfinite or overflows binary16",
        ));
    }
    Ok(half.to_bits())
}

pub(super) fn logical_source_shape(source: &WeightComponentSpec) -> Result<Vec<u64>, VNextError> {
    if !(2..=3).contains(&source.dimensions.len()) || source.dimensions.iter().any(|n| *n == 0) {
        return Err(invalid(
            "GGUF RN-F16 source must be a 2D or stacked 3D projection component",
        ));
    }
    let mut dimensions = source.dimensions.clone();
    match &source.encoding {
        WeightEncoding::BlockQuantized(spec)
            if source.role == WeightComponentRole::PackedValues =>
        {
            supported_format(spec)?;
            // The plan has proved row-major storage and a final block axis.
            // A packed gate/up component retains both leading dimensions.
            let width = dimensions.last_mut().unwrap();
            *width = width
                .checked_mul(u64::from(spec.logical_values_per_block))
                .ok_or_else(|| invalid("GGUF RN-F16 width overflow"))?;
        }
        WeightEncoding::Dense {
            element_type: ElementType::F16,
        } if source.role == WeightComponentRole::Values => {}
        _ => {
            return Err(invalid(
                "GGUF RN-F16 projection source encoding is unsupported",
            ))
        }
    }
    source.physical_bytes()?;
    Ok(dimensions)
}
pub(super) fn derived_component(
    sources: &[&WeightComponentSpec],
    dimensions: &[u64],
) -> Result<WeightComponentSpec, VNextError> {
    if sources.is_empty()
        || !(2..=3).contains(&dimensions.len())
        || dimensions.iter().any(|d| *d == 0)
    {
        return Err(invalid(
            "GGUF RN-F16 derived projection shape/source group absent",
        ));
    }
    let mut elements = 0u64;
    let mut unique = BTreeSet::new();
    for source in sources {
        if !unique.insert(&source.id) {
            return Err(invalid("GGUF RN-F16 source group repeats a component"));
        }
        let shape = logical_source_shape(source)?;
        if shape.last() != dimensions.last() {
            return Err(invalid("GGUF RN-F16 source group input widths differ"));
        }
        elements = elements
            .checked_add(
                shape
                    .iter()
                    .try_fold(1_u64, |total, &dimension| total.checked_mul(dimension))
                    .ok_or_else(|| invalid("source elements overflow"))?,
            )
            .ok_or_else(|| invalid("source group elements overflow"))?;
    }
    let target_elements = dimensions
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or_else(|| invalid("target elements overflow"))?;
    if elements != target_elements {
        return Err(invalid("GGUF RN-F16 source group span differs from target"));
    }
    let source_wire = serde_json::to_vec(sources).map_err(|e| invalid(e.to_string()))?;
    let shape_wire = serde_json::to_vec(dimensions).map_err(|e| invalid(e.to_string()))?;
    let digest = fingerprint(&[&source_wire, &shape_wire]);
    let component = WeightComponentSpec {
        id: WeightId::new(format!("component.execution.gguf-rn-f16.{digest}"))?,
        role: WeightComponentRole::Values,
        external_names: sources
            .iter()
            .flat_map(|s| &s.external_names)
            .enumerate()
            .map(|(index, _)| format!("execution.gguf-rn-f16.{digest}.{index}"))
            .collect(),
        dimensions: dimensions.to_vec(),
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F16,
        },
        required: sources.iter().any(|s| s.required),
    };
    component.physical_bytes()?;
    Ok(component)
}
pub(super) fn materialize_group<'source>(
    source: &'source dyn WeightComponentSource,
    source_components: &[&WeightComponentSpec],
    execution_components: &[&WeightComponentSpec],
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    if source_components.is_empty() || execution_components.is_empty() {
        return Err(invalid("GGUF RN-F16 source/target group is empty"));
    }
    let original = if source_components.len() == 1 {
        Some(source_components[0])
    } else {
        None
    };
    let mut converted_target: Option<&WeightComponentSpec> = None;
    let mut seen = BTreeSet::new();
    for &target in execution_components {
        if !seen.insert(&target.id) {
            return Err(invalid(
                "GGUF RN-F16 target group repeats an execution component",
            ));
        }
        if original == Some(target) {
            continue;
        }
        if converted_target.is_some()
            || target != &derived_component(source_components, &target.dimensions)?
        {
            return Err(invalid(
                "GGUF RN-F16 target differs from its deduplicated source-derived spec",
            ));
        }
        converted_target = Some(target);
    }
    // Original retained head/embedding and its converted projection can share
    // one source group. Read that source once; retain the borrowed original and
    // produce only one independent F16 payload.
    let mut payloads = Vec::with_capacity(source_components.len());
    for component in source_components {
        let payload = source.component(component)?;
        if u64::try_from(payload.bytes().len()).ok() != Some(component.physical_bytes()?) {
            return Err(invalid(
                "GGUF RN-F16 source payload length differs from schema",
            ));
        }
        payloads.push(Some(payload));
    }
    let mut converted = if let Some(target) = converted_target {
        let capacity = usize::try_from(target.physical_bytes()?)
            .map_err(|_| invalid("GGUF RN-F16 target exceeds host address space"))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(capacity)
            .map_err(|_| invalid("GGUF RN-F16 host allocation unavailable"))?;
        let mut source_files = Vec::new();
        for (original, payload) in source_components.iter().zip(&payloads) {
            let payload = payload.as_ref().unwrap();
            source_files.extend_from_slice(payload.source_files());
            match &original.encoding {
                WeightEncoding::BlockQuantized(spec) => {
                    append_converted(supported_format(spec)?, payload.bytes(), &mut output)?
                }
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                } => {
                    for bytes in payload.bytes().chunks_exact(2) {
                        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                        if !f16::from_bits(bits).is_finite() {
                            return Err(invalid(
                                "GGUF RN-F16 retained projection contains nonfinite F16",
                            ));
                        }
                    }
                    output.extend_from_slice(payload.bytes());
                }
                _ => unreachable!("source-derived spec validates encoding"),
            }
        }
        if output.len() != capacity {
            return Err(invalid(
                "GGUF RN-F16 converted payload length differs from plan",
            ));
        }
        Some(WeightComponentPayload::from_ordered_sources(
            target,
            target.external_names.clone(),
            source_files,
            target.dimensions.clone(),
            ElementType::F16,
            output,
        )?)
    } else {
        None
    };
    execution_components
        .iter()
        .map(|target| {
            if original == Some(*target) {
                payloads[0]
                    .take()
                    .ok_or_else(|| invalid("GGUF RN-F16 original payload already consumed"))
            } else {
                converted
                    .take()
                    .ok_or_else(|| invalid("GGUF RN-F16 converted payload already consumed"))
            }
        })
        .collect()
}
