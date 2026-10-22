use super::*;

/// Integer rounding of the finite F32 representation. No enumeration/search in
/// the large-matrix path. Subnormals, ties-even and signed zeros are explicit.
pub(super) fn encode(value: f32) -> u8 {
    debug_assert!(value.is_finite());
    let bits = value.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 448.0_f32.to_bits() {
        return sign | 126;
    }
    let exponent = ((magnitude >> 23) & 255) as i32 - 127;
    if exponent < -10 {
        return sign;
    }
    let significand = (magnitude & 0x7f_ffff) | 0x80_0000;
    let (base, remainder, halfway) = if exponent < -6 {
        let shift = (14 - exponent) as u32;
        (
            significand >> shift,
            significand & ((1 << shift) - 1),
            1 << (shift - 1),
        )
    } else {
        (
            (((exponent + 7) as u32) << 3) | ((magnitude >> 20) & 7),
            magnitude & 0x000f_ffff,
            1 << 19,
        )
    };
    let rounded = base + u32::from(remainder > halfway || (remainder == halfway && base & 1 != 0));
    sign | (rounded.min(126) as u8)
}

pub(super) fn decode(bits: u8) -> f32 {
    let exponent = (bits >> 3) & 15;
    let mantissa = bits & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else if exponent == 15 && mantissa == 7 {
        f32::NAN
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    if bits & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub(super) fn scale(maximum: f32) -> Result<(f32, f32), VNextError> {
    if !maximum.is_finite() || maximum < 0.0 {
        return Err(invalid("nonfinite source maximum"));
    }
    let scale = if maximum == 0.0 { 1.0 } else { maximum / 448.0 };
    let inverse = 1.0 / scale;
    if !scale.is_finite() || scale <= 0.0 || !inverse.is_finite() || inverse <= 0.0 {
        return Err(invalid(
            "nonzero group scale or reciprocal is not finite positive",
        ));
    }
    Ok((scale, inverse))
}

pub(super) fn materialize<'source>(
    host: &HostMemoryAccount,
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
    outputs: &[&WeightComponentSpec],
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    if outputs.is_empty() {
        return Err(invalid("empty requested component group"));
    }
    if let ([original], [output]) = (sources, outputs) {
        if original == output {
            return source.component(original).map(|payload| vec![payload]);
        }
    }
    let derived = plan::derive_group(sources)?;
    let mut indices = Vec::with_capacity(outputs.len());
    let mut seen = BTreeSet::new();
    let mut total = 0usize;
    for output in outputs {
        let index = derived
            .iter()
            .position(|derived| derived == *output)
            .ok_or_else(|| invalid("requested component differs from ordered source recipe"))?;
        if !seen.insert(index) {
            return Err(invalid("duplicate requested output component"));
        }
        let bytes = usize::try_from(output.physical_bytes()?)
            .map_err(|_| invalid("payload exceeds usize"))?;
        let start = total;
        total = total
            .checked_add(bytes)
            .filter(|total| *total <= isize::MAX as usize)
            .ok_or_else(|| invalid("group backing overflow"))?;
        indices.push((index, start..total));
    }
    // Retain the exact immutable sources through both passes. This rejects
    // implicit source transcoding into an unaccounted, whole-matrix Vec.
    let payloads = sources
        .iter()
        .map(|spec| {
            let payload = source.component(spec)?;
            if payload.component_id() != &spec.id
                || payload.dimensions() != spec.dimensions
                || payload.element_type() != ElementType::Bf16
                || payload.external_names() != spec.external_names
                || payload.bytes().len() as u64 != spec.physical_bytes()?
                || payload.retained_host_memory().is_none()
            {
                return Err(invalid(
                    "source must expose its exact retained BF16 backing without transcoding",
                ));
            }
            Ok(payload)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut maximum = 0_f32;
    for payload in &payloads {
        for bytes in payload.bytes().chunks_exact(2) {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
            if !value.is_finite() {
                return Err(invalid(
                    "nonfinite BF16 source is outside the conversion profile",
                ));
            }
            maximum = maximum.max(value.abs());
        }
    }
    let (scale, inverse) = scale(maximum)?;
    let mut backing = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, total)
        .map_err(|e| invalid(format!("WeightBacking admission: {e}")))?;
    for (index, range) in &indices {
        let target = &mut backing.as_mut()[range.clone()];
        if *index >= sources.len() {
            target.copy_from_slice(&scale.to_le_bytes());
        } else {
            for (byte, original) in target
                .iter_mut()
                .zip(payloads[*index].bytes().chunks_exact(2))
            {
                let value =
                    f32::from_bits(u32::from(u16::from_le_bytes([original[0], original[1]])) << 16);
                *byte = encode(value * inverse);
            }
        }
    }
    let backing = backing.freeze();
    let files: Vec<_> = payloads
        .iter()
        .map(|payload| payload.source_file().to_owned())
        .collect();
    outputs
        .iter()
        .zip(indices)
        .map(|(component, (_, range))| {
            WeightComponentPayload::from_managed_host(
                component,
                component.external_names.clone(),
                files.clone(),
                component.dimensions.clone(),
                component.physical_element_type(),
                backing.slice(range).map_err(|e| invalid(e.to_string()))?,
            )
        })
        .collect()
}
