//! The single existing physical source mapping, shared by borrowed checkpoints
//! and opaque prepared components. No quantized matrix is expanded on Host.
use super::*;
use anyhow::ensure;
use ferrum_quantization::SafetensorsArchive;

pub(in crate::p_h3_image_vae) fn component<'s>(
    config: &PH3ImageVaeEncoderConfig,
    source: &'s SafetensorsArchive,
    host: &HostMemoryAccount,
    c: &WeightComponentSpec,
) -> Result<WeightComponentPayload<'s>, VNextError> {
    let e = entry(config, c)?;
    let name = &e.name;
    let t = source.tensor(name).map_err(|e| invalid(e.to_string()))?;
    if t.element_type() != Some(ElementType::F32) || t.shape() != e.original {
        return Err(invalid("original full encoder F32 shape changed"));
    }
    let [outputs, inputs, time, height, width] = e.original.as_slice() else {
        return source.component(c);
    };
    if *time == 1 {
        return WeightComponentPayload::new(
            c,
            name.clone(),
            t.source_file(),
            c.dimensions.clone(),
            ElementType::F32,
            t.bytes(),
        )?
        .with_retained_host_memory(t.retained_host_memory().clone());
    }
    let mut backing = host
        .allocate_zeroed(
            HostMemoryClass::WeightBacking,
            usize::try_from(c.physical_bytes()?)
                .map_err(|_| invalid("packed weights exceed address space"))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
    let plane = usize::try_from(
        height
            .checked_mul(*width)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| invalid("temporal plane overflows"))?,
    )
    .map_err(|_| invalid("temporal plane exceeds address space"))?;
    let pairs = usize::try_from(
        outputs
            .checked_mul(*inputs)
            .ok_or_else(|| invalid("channel pairs overflow"))?,
    )
    .map_err(|_| invalid("channel pairs exceed address space"))?;
    for pair in 0..pairs {
        let start = (pair * (*time as usize) + *time as usize - 1) * plane;
        backing.as_mut()[pair * plane..(pair + 1) * plane]
            .copy_from_slice(&t.bytes()[start..start + plane]);
    }
    WeightComponentPayload::from_managed_host(
        c,
        vec![name.clone()],
        vec![t.source_file().to_owned()],
        c.dimensions.clone(),
        ElementType::F32,
        backing.freeze(),
    )
}

/// Validate the immutable complete source once, before geometry-only prepares.
pub(in crate::p_h3_image_vae) fn validate(
    config: &PH3ImageVaeEncoderConfig,
    source: &SafetensorsArchive,
    limits: PH3ImageVaeEncoderLimits,
) -> anyhow::Result<()> {
    PH3ImageVaeEncoderFamily::new(limits)?.check(config)?;
    for e in weights::entries(config)? {
        let t = source.tensor(&e.name)?;
        ensure!(
            t.element_type() == Some(ElementType::F32) && t.shape() == e.original,
            "full encoder source {} requires its original F32 shape",
            e.name
        );
        ensure!(
            t.bytes()
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "full encoder source {} is nonfinite",
            e.name
        );
    }
    Ok(())
}

fn entry(
    config: &PH3ImageVaeEncoderConfig,
    c: &WeightComponentSpec,
) -> Result<weights::Entry, VNextError> {
    let [name] = c.external_names.as_slice() else {
        return Err(invalid("one exact full encoder source required"));
    };
    let e = weights::entries(config)?
        .into_iter()
        .find(|e| &e.name == name)
        .ok_or_else(|| invalid("unknown full encoder source"))?;
    if c.dimensions != e.physical
        || c.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::F32,
            })
    {
        return Err(invalid(
            "component differs from declared full encoder last-tap mapping",
        ));
    }
    Ok(e)
}

/// Logical bytes of the exact managed source allocation. Original F32 vectors
/// and KT1 convolution views remain borrowed; no payload is accessed here.
pub(in crate::p_h3_image_vae) fn temporary_backing_bytes(
    config: &PH3ImageVaeEncoderConfig,
    c: &WeightComponentSpec,
) -> Result<u64, VNextError> {
    let e = entry(config, c)?;
    match e.original.as_slice() {
        [_, _, time, _, _] if *time > 1 => c.physical_bytes(),
        _ => Ok(0),
    }
}
