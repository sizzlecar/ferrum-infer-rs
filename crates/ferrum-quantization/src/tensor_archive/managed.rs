//! Explicit cold-path dense normalization using the composition's Host ledger.
use super::*;
use ferrum_runtime::{
    CanonicalRational, HostMemoryAccount, HostMemoryClass, VNextError, WeightComponentPayload,
    WeightComponentSpec,
};
use half::{bf16, f16};
mod bf16_to_f16;

/// Native blocks remain borrowed and unchanged. Dense floating-point values
/// may be rounded to the requested component type, with output reserved before
/// allocation. This is a source representation conversion, not dequantization
/// of a model or an execution-side persistent mirror.
/// Single-source `DenseAffine` applies the archive contract's F64 affine,
/// followed by F32 and the destination rounding. Its destination is always
/// managed, including when source and destination dtypes are equal.
pub struct ManagedDenseArchiveSource<'archive> {
    archive: &'archive dyn TensorArchive,
    host: &'archive HostMemoryAccount,
}
impl<'archive> ManagedDenseArchiveSource<'archive> {
    pub fn new(archive: &'archive dyn TensorArchive, host: &'archive HostMemoryAccount) -> Self {
        Self { archive, host }
    }

    fn composite<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let invalid = |reason: &str| VNextError::InvalidExecutionPlan {
            reason: format!("managed archive component `{}`: {reason}", component.id),
        };
        let WeightEncoding::Dense {
            element_type: target,
        } = &component.encoding
        else {
            return Err(invalid(
                "multi-source normalization requires a dense floating target",
            ));
        };
        if component.external_names.len() < 2
            || component.dimensions.is_empty()
            || component.dimensions.iter().any(|&d| d == 0)
            || !floating(*target)
        {
            return Err(invalid(
                "multi-source dense target needs positive shape and at least two sources",
            ));
        }
        let tensors = component
            .external_names
            .iter()
            .map(|name| {
                self.archive
                    .tensor(name)
                    .map_err(|e| invalid(&e.to_string()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        // Match the existing SafetensorsArchive source contract: an exact
        // leading-axis stack, or an exact axis-zero concatenation. Declared
        // source order is the byte order; tensor names never select a recipe.
        let stacked = component.dimensions.first().copied() == u64::try_from(tensors.len()).ok()
            && tensors
                .iter()
                .all(|tensor| tensor.shape() == &component.dimensions[1..]);
        let concatenated = tensors.iter().all(|tensor| {
            tensor.shape().len() == component.dimensions.len()
                && tensor.shape()[1..] == component.dimensions[1..]
        }) && tensors
            .iter()
            .try_fold(0u64, |rows, tensor| rows.checked_add(tensor.shape()[0]))
            == component.dimensions.first().copied();
        if !stacked && !concatenated {
            return Err(invalid(
                "sources are neither an exact leading-axis stack nor an axis-zero concatenation",
            ));
        }
        let bytes = usize::try_from(component.physical_bytes()?)
            .ok()
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid("target exceeds address space"))?;
        let mut source_types = Vec::with_capacity(tensors.len());
        let mut part_bytes = Vec::with_capacity(tensors.len());
        let mut total = 0usize;
        for tensor in &tensors {
            let WeightEncoding::Dense {
                element_type: source,
            } = tensor.encoding()
            else {
                return Err(invalid(
                    "multi-source normalization requires dense floating sources",
                ));
            };
            if !floating(*source) {
                return Err(invalid(
                    "only dense F32/F16/BF16 normalization is supported",
                ));
            }
            let elements = tensor.shape().iter().try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            });
            let elements =
                elements.ok_or_else(|| invalid("source element count exceeds address space"))?;
            if elements.checked_mul(source.size_bytes() as usize) != Some(tensor.bytes().len()) {
                return Err(invalid(
                    "source payload extent differs from its declared shape",
                ));
            }
            let extent = elements
                .checked_mul(target.size_bytes() as usize)
                .ok_or_else(|| invalid("converted source extent overflows"))?;
            total = total
                .checked_add(extent)
                .ok_or_else(|| invalid("combined source extent overflows"))?;
            source_types.push(*source);
            part_bytes.push(extent);
        }
        if total != bytes {
            return Err(invalid(
                "combined source bytes differ from the exact target extent",
            ));
        }
        // Every source is an immutable archive view. The only payload-sized
        // allocation is this one exact destination on the caller's Host view.
        let mut backing = self
            .host
            .reserve(HostMemoryClass::WeightBacking, bytes)
            .and_then(|reservation| reservation.allocate_zeroed())
            .map_err(|e| invalid(&e.to_string()))?;
        let mut offset = 0usize;
        let mut files = Vec::with_capacity(tensors.len());
        for ((tensor, source), extent) in tensors.iter().zip(source_types).zip(part_bytes) {
            let end = offset + extent; // checked above as part of the exact total
            convert_floating(
                source,
                *target,
                tensor.bytes(),
                &mut backing.as_mut()[offset..end],
                None,
            )
            .map_err(invalid)?;
            offset = end;
            files.push(tensor.source_file().to_owned());
        }
        WeightComponentPayload::from_managed_host(
            component,
            component.external_names.clone(),
            files,
            component.dimensions.clone(),
            *target,
            backing.freeze(),
        )
    }
}
impl WeightComponentSource for ManagedDenseArchiveSource<'_> {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        let invalid = |reason: &str| VNextError::InvalidExecutionPlan {
            reason: format!("managed archive component `{}`: {reason}", component.id),
        };
        let [name] = component.external_names.as_slice() else {
            return self.composite(component);
        };
        let tensor = self
            .archive
            .tensor(name)
            .map_err(|e| invalid(&e.to_string()))?;
        let (source, target, affine) = match (tensor.encoding(), &component.encoding) {
            (
                WeightEncoding::Dense {
                    element_type: source,
                },
                WeightEncoding::Dense {
                    element_type: target,
                },
            ) => (*source, *target, None),
            (
                WeightEncoding::Dense {
                    element_type: source,
                },
                WeightEncoding::DenseAffine {
                    element_type: target,
                    scale,
                    bias,
                },
            ) => (*source, *target, Some((*scale, *bias))),
            _ => {
                if tensor.encoding() != &component.encoding {
                    return Err(invalid("stored encoding differs from the bound component"));
                }
                return self.archive.component(component);
            }
        };
        if tensor.shape() != component.dimensions {
            return Err(invalid("stored dimensions differ from the bound component"));
        }
        if source == target && affine.is_none() {
            return self.archive.component(component);
        }
        if !floating(source) || !floating(target) {
            return Err(invalid(
                "only dense F32/F16/BF16 normalization is supported",
            ));
        }
        let elements = component
            .dimensions
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| invalid("source element count exceeds address space"))?;
        if elements.checked_mul(source.size_bytes() as usize) != Some(tensor.bytes().len()) {
            return Err(invalid(
                "source payload extent differs from its declared shape",
            ));
        }
        let bytes = usize::try_from(component.physical_bytes()?)
            .ok()
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid("target exceeds address space"))?;
        if elements.checked_mul(target.size_bytes() as usize) != Some(bytes) {
            return Err(invalid(
                "target payload extent differs from its declared shape",
            ));
        }
        let mut backing = self
            .host
            .reserve(HostMemoryClass::WeightBacking, bytes)
            .and_then(|reservation| reservation.allocate_zeroed())
            .map_err(|e| invalid(&e.to_string()))?;
        convert_floating(source, target, tensor.bytes(), backing.as_mut(), affine)
            .map_err(invalid)?;
        WeightComponentPayload::from_managed_host(
            component,
            vec![name.clone()],
            vec![tensor.source_file().to_owned()],
            component.dimensions.clone(),
            target,
            backing.freeze(),
        )
    }
}

fn floating(ty: ElementType) -> bool {
    matches!(ty, ElementType::F32 | ElementType::F16 | ElementType::Bf16)
}

fn convert_floating(
    source: ElementType,
    target: ElementType,
    input: &[u8],
    output: &mut [u8],
    affine: Option<(CanonicalRational, CanonicalRational)>,
) -> std::result::Result<(), &'static str> {
    if source == ElementType::Bf16 && target == ElementType::F16 && affine.is_none() {
        return bf16_to_f16::convert(input, output);
    }
    convert_floating_scalar(source, target, input, output, affine)
}

fn convert_floating_scalar(
    source: ElementType,
    target: ElementType,
    input: &[u8],
    output: &mut [u8],
    affine: Option<(CanonicalRational, CanonicalRational)>,
) -> std::result::Result<(), &'static str> {
    if !input.len().is_multiple_of(source.size_bytes() as usize)
        || input.len() / source.size_bytes() as usize != output.len() / target.size_bytes() as usize
        || !output.len().is_multiple_of(target.size_bytes() as usize)
    {
        return Err("source and target floating element counts differ");
    }
    if source == target && affine.is_none() {
        output.copy_from_slice(input);
        return Ok(());
    }
    let affine = affine.map(|(scale, bias)| {
        (
            scale.numerator() as f64 / scale.denominator() as f64,
            bias.numerator() as f64 / bias.denominator() as f64,
        )
    });
    for (input, output) in input
        .chunks_exact(source.size_bytes() as usize)
        .zip(output.chunks_exact_mut(target.size_bytes() as usize))
    {
        let mut value = match source {
            ElementType::F32 => f32::from_le_bytes(input.try_into().unwrap()),
            ElementType::F16 => f16::from_le_bytes(input.try_into().unwrap()).to_f32(),
            ElementType::Bf16 => bf16::from_le_bytes(input.try_into().unwrap()).to_f32(),
            _ => return Err("unsupported dense source type"),
        };
        if let Some((scale, bias)) = affine {
            // Match SafetensorsArchive's declared DenseAffine boundary. An
            // F32 multiply/add or an F16 intermediate would be a different
            // representation contract near cancellation and rounding ties.
            value = (f64::from(value) * scale + bias) as f32;
            if !value.is_finite() {
                return Err("dense affine materialization produced a non-finite value");
            }
        }
        let converted = match target {
            ElementType::F32 => {
                output.copy_from_slice(&value.to_le_bytes());
                value
            }
            ElementType::F16 => {
                let v = f16::from_f32(value);
                output.copy_from_slice(&v.to_le_bytes());
                v.to_f32()
            }
            ElementType::Bf16 => {
                let v = bf16::from_f32(value);
                output.copy_from_slice(&v.to_le_bytes());
                v.to_f32()
            }
            _ => return Err("unsupported dense target type"),
        };
        if value.is_finite() && !converted.is_finite() {
            return Err("finite dense value overflows the target type");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
