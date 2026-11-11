//! The single existing physical source mapping, shared by borrowed checkpoints
//! and opaque prepared components. No quantized matrix is expanded on Host.
use super::*;
use anyhow::ensure;
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use half::bf16;

pub(in crate::p_h3_image_vae) fn component<'s>(
    config: &PH3ImageVaeDecoderExecutionConfig,
    canonical: &'s ComfyCanonicalWeightSource,
    host: &HostMemoryAccount,
    c: &WeightComponentSpec,
) -> Result<WeightComponentPayload<'s>, VNextError> {
    let [name] = c.external_names.as_slice() else {
        return Err(invalid("decoder component requires one exact source"));
    };
    let source = canonical
        .archive()
        .tensor(name)
        .map_err(|e| invalid(e.to_string()))?;
    if source.element_type() != Some(ElementType::Bf16) {
        return canonical.component(c);
    }
    // Only two exact singleton views differ from the declared original
    // shape. Both preserve byte order; arbitrary reshaping is rejected.
    let target = outer::entries(config)?
        .into_iter()
        .find(|(n, _, _)| *n == name.as_str());
    let shape_ok = if let Some((_, original, view)) = target {
        source.shape() == original && c.dimensions == view
    } else {
        source.shape() == c.dimensions
    };
    if !shape_ok
        || c.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::F32,
            })
    {
        return Err(invalid(
            "BF16 source shape/view or F32 promotion contract differs",
        ));
    }
    let bytes = usize::try_from(c.physical_bytes()?)
        .map_err(|_| invalid("static promotion exceeds Host address space"))?;
    let mut backing = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, bytes)
        .map_err(|e| invalid(e.to_string()))?;
    for (src, dst) in source
        .bytes()
        .chunks_exact(2)
        .zip(backing.as_mut().chunks_exact_mut(4))
    {
        dst.copy_from_slice(
            &bf16::from_bits(u16::from_le_bytes(src.try_into().unwrap()))
                .to_f32()
                .to_le_bytes(),
        );
    }
    WeightComponentPayload::from_managed_host(
        c,
        vec![name.clone()],
        vec![source.source_file().to_owned()],
        c.dimensions.clone(),
        ElementType::F32,
        backing.freeze(),
    )
}

/// Validate the immutable complete source once, before geometry-only prepares.
pub(in crate::p_h3_image_vae) fn validate(
    config: &PH3ImageVaeDecoderExecutionConfig,
    source: &ComfyCanonicalWeightSource,
    limits: PH3ImageVaeDecoderLimits,
) -> anyhow::Result<()> {
    compose::admit(config, limits)?;
    for layer in 0..config.architecture().layers() {
        PH3ImageVaeDecoderBlockCheckpoint::new(&config.block(layer)?, source, limits.block())?;
    }
    for (name, shape, _) in outer::entries(config)? {
        let t = source.archive().tensor(name)?;
        ensure!(
            t.element_type() == Some(ElementType::Bf16) && t.shape() == shape,
            "VAE outer parameter {name} original BF16 shape differs"
        );
        ensure!(
            t.bytes()
                .chunks_exact(2)
                .all(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).is_finite()),
            "VAE outer parameter {name} is nonfinite"
        );
    }
    Ok(())
}

/// Header-only query matching the source's actual BF16-promotion allocation.
/// Raw I8/F32 canonical carriers require no temporary source WeightBacking.
pub(in crate::p_h3_image_vae) fn temporary_backing_bytes(
    source: &ComfyCanonicalWeightSource,
    c: &WeightComponentSpec,
) -> anyhow::Result<u64> {
    let [name] = c.external_names.as_slice() else {
        anyhow::bail!("decoder requires one exact source name");
    };
    let original = source.archive().tensor(name)?;
    if original.element_type() == Some(ElementType::Bf16) {
        ensure!(
            c.encoding
                == WeightEncoding::Dense {
                    element_type: ElementType::F32
                },
            "original BF16 source must declare its F32 promotion"
        );
        Ok(c.physical_bytes()?)
    } else {
        Ok(0)
    }
}
