use super::*;
use crate::{gguf_blocks::GgufBlockFormat, gguf_f16_projection_materializer as legacy};

pub(super) fn fragment_component(
    sources: &[&WeightComponentSpec],
) -> Result<(RnF16FragmentPlanV1, WeightComponentSpec), VNextError> {
    let first = sources
        .first()
        .ok_or_else(|| invalid("RN fragment source group empty"))?;
    let WeightEncoding::BlockQuantized(first_spec) = &first.encoding else {
        return Err(invalid(
            "RN fragment requires original Q4K/Q5K/Q6K components",
        ));
    };
    let format = match GgufBlockFormat::from_spec(first_spec).map_err(invalid)? {
        GgufBlockFormat::Q4K => RnF16FragmentSourceFormatV1::Q4K,
        GgufBlockFormat::Q5K => RnF16FragmentSourceFormatV1::Q5K,
        GgufBlockFormat::Q6K => RnF16FragmentSourceFormatV1::Q6K,
        _ => return Err(invalid("RN fragment source format is outside Q4K/Q5K/Q6K")),
    };
    let first_shape = legacy::conversion::logical_source_shape(first)?;
    let k = *first_shape
        .last()
        .ok_or_else(|| invalid("RN fragment source width absent"))?;
    let mut n = 0_u64;
    let mut seen = BTreeSet::new();
    for source in sources {
        if !seen.insert(&source.id) || source.encoding != first.encoding {
            return Err(invalid(
                "RN fragment source group repeats a component or mixes encodings",
            ));
        }
        let shape = legacy::conversion::logical_source_shape(source)?;
        if shape.last() != Some(&k) {
            return Err(invalid("RN fragment source widths differ"));
        }
        let rows = shape[..shape.len() - 1]
            .iter()
            .try_fold(1_u64, |a, b| a.checked_mul(*b))
            .ok_or_else(|| invalid("RN fragment source rows overflow"))?;
        n = n
            .checked_add(rows)
            .ok_or_else(|| invalid("RN fragment combined rows overflow"))?;
    }
    let plan = RnF16FragmentPlanV1::new(format, n, k)?;
    if &plan.source_block_spec() != first_spec {
        return Err(invalid("RN fragment source spec differs from checked ABI"));
    }
    let source_wire = serde_json::to_vec(sources).map_err(|e| invalid(e.to_string()))?;
    let digest = fingerprint(&[
        &source_wire,
        &plan.packing_abi().to_le_bytes(),
        &n.to_le_bytes(),
        &k.to_le_bytes(),
    ]);
    let component = WeightComponentSpec {
        id: WeightId::new(format!("component.execution.gguf-rn-f16-fragment.{digest}"))?,
        role: WeightComponentRole::PackedValues,
        external_names: sources
            .iter()
            .flat_map(|s| &s.external_names)
            .enumerate()
            .map(|(i, _)| format!("execution.gguf-rn-f16-fragment.{digest}.{i}"))
            .collect(),
        dimensions: plan.packed_dimensions().to_vec(),
        encoding: plan.packed_encoding(),
        required: sources.iter().any(|s| s.required),
    };
    if component.physical_bytes()? != plan.packed_bytes() {
        return Err(invalid("RN fragment target plan byte mismatch"));
    }
    Ok((plan, component))
}

pub(super) fn materialize_group<'source>(
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
    targets: &[&WeightComponentSpec],
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    if sources.is_empty() || targets.is_empty() {
        return Err(invalid("RN fragment source/target group empty"));
    }
    let original = if sources.len() == 1 {
        Some(sources[0])
    } else {
        None
    };
    let needs_packet = targets.iter().any(|target| {
        original != Some(*target) && matches!(target.encoding, WeightEncoding::BlockQuantized(_))
    });
    if !needs_packet {
        return legacy::conversion::materialize_group(source, sources, targets);
    }
    let (plan, packet_spec) = fragment_component(sources)?;
    let mut packet_target = None;
    let mut dense_target = None;
    let mut seen = BTreeSet::new();
    for &target in targets {
        if !seen.insert(&target.id) {
            return Err(invalid("RN fragment target group repeats a component"));
        }
        if original == Some(target) {
            continue;
        }
        if target == &packet_spec {
            packet_target = Some(target);
        } else if matches!(
            target.encoding,
            WeightEncoding::Dense {
                element_type: ElementType::F16
            }
        ) && dense_target.is_none()
            && target == &legacy::conversion::derived_component(sources, &target.dimensions)?
        {
            dense_target = Some(target);
        } else {
            return Err(invalid(
                "RN fragment target differs from checked source-derived dual group",
            ));
        }
    }
    let packet_target = packet_target.ok_or_else(|| invalid("RN fragment target absent"))?;
    // Registry groups dense + packet by the same ordered source identities.
    // Actual source I/O occurs once per component; both outputs borrow those
    // bytes only during this cold conversion and own their final payloads.
    let mut payloads = Vec::with_capacity(sources.len());
    for component in sources {
        let payload = source.component(component)?;
        if u64::try_from(payload.bytes().len()).ok() != Some(component.physical_bytes()?) {
            return Err(invalid(
                "RN fragment source payload span differs from schema",
            ));
        }
        payloads.push(Some(payload));
    }
    let capacity = usize::try_from(plan.dense_bytes())
        .map_err(|_| invalid("RN fragment dense span exceeds host address space"))?;
    let mut dense = Vec::new();
    dense
        .try_reserve_exact(capacity)
        .map_err(|_| invalid("RN fragment dense allocation unavailable"))?;
    let mut source_files = Vec::new();
    let source_bytes = payloads
        .iter()
        .map(|p| p.as_ref().unwrap().bytes())
        .collect::<Vec<_>>();
    let format = GgufBlockFormat::from_spec(&plan.source_block_spec()).map_err(invalid)?;
    for payload in &payloads {
        let payload = payload.as_ref().unwrap();
        source_files.extend_from_slice(payload.source_files());
        legacy::conversion::append_converted(format, payload.bytes(), &mut dense)?;
    }
    if dense.len() != capacity {
        return Err(invalid("RN fragment dense RN conversion span differs"));
    }
    let packet = crate::gguf_rn_fragment::pack_checked(&plan, &source_bytes, Some(&dense))?;
    let mut dense_payload = if let Some(target) = dense_target {
        Some(WeightComponentPayload::from_ordered_sources(
            target,
            target.external_names.clone(),
            source_files.clone(),
            target.dimensions.clone(),
            target.physical_element_type(),
            dense,
        )?)
    } else {
        None
    };
    let mut packet_payload = Some(WeightComponentPayload::from_ordered_sources(
        packet_target,
        packet_target.external_names.clone(),
        source_files,
        packet_target.dimensions.clone(),
        packet_target.physical_element_type(),
        packet,
    )?);
    targets
        .iter()
        .map(|target| {
            if original == Some(*target) {
                payloads[0]
                    .take()
                    .ok_or_else(|| invalid("RN fragment original payload already consumed"))
            } else if *target == packet_target {
                packet_payload
                    .take()
                    .ok_or_else(|| invalid("RN fragment packet payload already consumed"))
            } else {
                dense_payload
                    .take()
                    .ok_or_else(|| invalid("RN fragment dense payload already consumed"))
            }
        })
        .collect()
}
