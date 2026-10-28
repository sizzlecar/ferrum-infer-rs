use super::*;
use timing::{GroupTiming, Phase};

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

/// Source lengths and BF16 encodings are validated before this scan. Clearing
/// the sign bit gives the same ordering as finite, nonnegative F32 magnitudes.
/// Every NaN/Inf encoding is above every finite magnitude, so reject once after
/// reducing all source slices without a per-element floating-point operation.
pub(super) fn finite_bf16_amax<'a>(
    sources: impl IntoIterator<Item = &'a [u8]>,
) -> Result<f32, VNextError> {
    let mut maximum = 0_u16;
    for source in sources {
        for bytes in source.chunks_exact(2) {
            maximum = maximum.max(u16::from_le_bytes([bytes[0], bytes[1]]) & 0x7fff);
        }
    }
    if maximum >= 0x7f80 {
        return Err(invalid(
            "nonfinite BF16 source is outside the conversion profile",
        ));
    }
    Ok(f32::from_bits(u32::from(maximum) << 16))
}

#[cfg(test)]
pub(super) fn materialize<'source>(
    host: &HostMemoryAccount,
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
    outputs: &[&WeightComponentSpec],
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    materialize_with_cache(host, source, sources, outputs, None)
}

#[cfg(test)]
pub(super) fn materialize_with_cache<'source>(
    host: &HostMemoryAccount,
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
    outputs: &[&WeightComponentSpec],
    cache: Option<&host_cache::HostCache>,
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    materialize_composed(host, source, sources, outputs, cache, None)
}

pub(super) fn materialize_composed<'source>(
    host: &HostMemoryAccount,
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
    outputs: &[&WeightComponentSpec],
    cache: Option<&host_cache::HostCache>,
    composition: Option<&ManagedLinearAdapter>,
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    let mut timing = GroupTiming::new();
    let result = materialize_observed(
        host,
        source,
        sources,
        outputs,
        cache,
        composition,
        &mut timing,
    );
    timing.report(sources, outputs, result.is_ok());
    result
}

fn materialize_observed<'source>(
    host: &HostMemoryAccount,
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
    outputs: &[&WeightComponentSpec],
    cache: Option<&host_cache::HostCache>,
    composition: Option<&ManagedLinearAdapter>,
    timing: &mut GroupTiming,
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    if outputs.is_empty() {
        return Err(invalid("empty requested component group"));
    }
    if let ([original], [output]) = (sources, outputs) {
        if original == output {
            if let Some(composition) = composition.filter(|adapter| adapter.affects(original)) {
                return materialize_bf16_composed(
                    host,
                    source,
                    original,
                    composition,
                    cache,
                    timing,
                );
            }
            timing.mode = "borrowed";
            return timing.measure(Phase::SourceResolve, || {
                source.component(original).map(|payload| vec![payload])
            });
        }
    }
    let derived = plan::derive_group(sources)?;
    let mut indices = Vec::with_capacity(outputs.len());
    let mut seen = BTreeSet::new();
    let mut total = 0usize;
    let mut packed_elements = 0_u64;
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
        if index < sources.len() {
            packed_elements = packed_elements
                .checked_add(bytes as u64)
                .ok_or_else(|| invalid("requested packed elements overflow"))?;
        }
        let start = total;
        total = total
            .checked_add(bytes)
            .filter(|total| *total <= isize::MAX as usize)
            .ok_or_else(|| invalid("group backing overflow"))?;
        indices.push((index, start..total));
    }
    // Retain the exact immutable sources through both passes. This rejects
    // implicit source transcoding into an unaccounted, whole-matrix Vec.
    let mut payloads = timing.measure(Phase::SourceResolve, || resolve_sources(source, sources))?;
    let files = payloads
        .iter()
        .map(|payload| payload.source_file().to_owned())
        .collect::<Vec<_>>();
    let cache = cache.filter(|cache| cache.fits(total));
    let mut key = cache
        .map(|cache| cache.key(sources, outputs, &payloads))
        .transpose()?;
    // One materializer serializes cache misses. No partially initialized entry
    // is visible, duplicate misses cannot exceed its retained-byte allowance,
    // and an unwind poisons this owner rather than publishing partial results.
    let mut state = cache.map(host_cache::HostCache::lock).transpose()?;
    if let (Some(state), Some(key)) = (state.as_mut(), key.as_ref()) {
        if let Some((backing, ranges)) = state.get(key)? {
            timing.mode = "cache_hit";
            return payloads_from_backing(outputs, &files, &backing, ranges);
        }
    }
    // Cache identity uses the retained base owners plus the materializer's
    // adapter/content/strength fingerprint. Composition is explicit and occurs
    // only after a cold miss, never inside source resolution or admission.
    if let Some(composition) = composition {
        payloads = timing.measure(Phase::LoraCompose, || {
            payloads
                .into_iter()
                .zip(sources)
                .map(|(payload, component)| {
                    if !composition.affects(component) {
                        return Ok(payload);
                    }
                    let backing = composition
                        .compose(host, component, payload.bytes())
                        .map_err(|error| invalid(error.to_string()))?;
                    WeightComponentPayload::from_managed_host(
                        component,
                        component.external_names.clone(),
                        vec![payload.source_file().to_owned()],
                        component.dimensions.clone(),
                        ElementType::Bf16,
                        backing,
                    )
                })
                .collect::<Result<Vec<_>, VNextError>>()
        })?;
    }
    let maximum = timing.measure(Phase::Amax, || {
        finite_bf16_amax(payloads.iter().map(|payload| payload.bytes()))
    })?;
    let (scale, inverse) = scale(maximum)?;
    if let (Some(cache), Some(state)) = (cache, state.as_mut()) {
        cache.make_room(state, total);
    }
    let mut backing = timing.measure(Phase::HostAllocateZeroed, || {
        host.allocate_zeroed(HostMemoryClass::WeightBacking, total)
            .map_err(|e| invalid(format!("WeightBacking admission: {e}")))
    })?;
    timing.measure(Phase::Encode, || -> Result<(), VNextError> {
        // Only requested packed outputs amortize lookup construction. In
        // particular scale-only subsets do not allocate this temporary claim.
        // It is dropped before returning/freeze/cache publication, including on
        // failure. All immutable sources still participate in the amax scan.
        let lookup = (lookup::workspace_bytes(packed_elements) != 0)
            .then(|| lookup::LookupTable::new(host, maximum, inverse))
            .transpose()?;
        for (index, range) in &indices {
            let target = &mut backing.as_mut()[range.clone()];
            if *index >= sources.len() {
                target.copy_from_slice(&scale.to_le_bytes());
            } else if let Some(lookup) = &lookup {
                lookup.encode_into(payloads[*index].bytes(), target);
            } else {
                for (byte, original) in target
                    .iter_mut()
                    .zip(payloads[*index].bytes().chunks_exact(2))
                {
                    let value = f32::from_bits(
                        u32::from(u16::from_le_bytes([original[0], original[1]])) << 16,
                    );
                    *byte = encode(value * inverse);
                }
            }
        }
        Ok(())
    })?;
    let backing = backing.freeze();
    let ranges = indices
        .into_iter()
        .map(|(_, range)| range)
        .collect::<Vec<_>>();
    let result = payloads_from_backing(outputs, &files, &backing, ranges.clone())?;
    if let (Some(state), Some(key)) = (state.as_mut(), key.take()) {
        state.publish(key, backing, ranges)?;
    }
    Ok(result)
}

fn materialize_bf16_composed<'source>(
    host: &HostMemoryAccount,
    source: &'source dyn WeightComponentSource,
    component: &WeightComponentSpec,
    composition: &ManagedLinearAdapter,
    cache: Option<&host_cache::HostCache>,
    timing: &mut GroupTiming,
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    let specs = [component];
    let payloads = timing.measure(Phase::SourceResolve, || resolve_sources(source, &specs))?;
    let files = vec![payloads[0].source_file().to_owned()];
    let bytes = usize::try_from(component.physical_bytes()?)
        .map_err(|_| invalid("BF16 composition exceeds usize"))?;
    let cache = cache.filter(|cache| cache.fits(bytes));
    let key = cache
        .map(|cache| cache.key(&specs, &specs, &payloads))
        .transpose()?;
    let mut state = cache.map(host_cache::HostCache::lock).transpose()?;
    if let (Some(state), Some(key)) = (state.as_mut(), key.as_ref()) {
        if let Some((backing, ranges)) = state.get(key)? {
            timing.mode = "lora_bf16_cache_hit";
            return payloads_from_backing(&specs, &files, &backing, ranges);
        }
    }
    if let (Some(cache), Some(state)) = (cache, state.as_mut()) {
        cache.make_room(state, bytes);
    }
    timing.mode = "lora_bf16";
    let backing = timing
        .measure(Phase::LoraCompose, || {
            composition.compose(host, component, payloads[0].bytes())
        })
        .map_err(|error| invalid(error.to_string()))?;
    let ranges = vec![0..bytes];
    let result = payloads_from_backing(&specs, &files, &backing, ranges.clone())?;
    if let (Some(state), Some(key)) = (state.as_mut(), key) {
        state.publish(key, backing, ranges)?;
    }
    Ok(result)
}

/// Resolve immutable owner identity without scanning, decoding or copying the
/// source bytes. Both conversion and cache admission use the same validation.
pub(super) fn resolve_sources<'source>(
    source: &'source dyn WeightComponentSource,
    sources: &[&WeightComponentSpec],
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    sources
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
        .collect()
}

fn payloads_from_backing<'source>(
    outputs: &[&WeightComponentSpec],
    files: &[String],
    backing: &ManagedHostBytes,
    ranges: Vec<std::ops::Range<usize>>,
) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
    outputs
        .iter()
        .zip(ranges)
        .map(|(component, range)| {
            WeightComponentPayload::from_managed_host(
                component,
                component.external_names.clone(),
                files.to_vec(),
                component.dimensions.clone(),
                component.physical_element_type(),
                backing.slice(range).map_err(|e| invalid(e.to_string()))?,
            )
        })
        .collect()
}
