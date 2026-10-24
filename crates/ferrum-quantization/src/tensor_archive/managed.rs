//! Explicit cold-path dense normalization using the composition's Host ledger.
use super::*;
use ferrum_runtime::{
    HostMemoryAccount, HostMemoryClass, VNextError, WeightComponentPayload, WeightComponentSpec,
};
use half::{bf16, f16};

/// Native blocks remain borrowed and unchanged. Dense floating-point values
/// may be rounded to the requested component type, with output reserved before
/// allocation. This is a source representation conversion, not dequantization
/// of a model or an execution-side persistent mirror.
pub struct ManagedDenseArchiveSource<'archive> {
    archive: &'archive dyn TensorArchive,
    host: &'archive HostMemoryAccount,
}
impl<'archive> ManagedDenseArchiveSource<'archive> {
    pub fn new(archive: &'archive dyn TensorArchive, host: &'archive HostMemoryAccount) -> Self {
        Self { archive, host }
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
            return Err(invalid("requires exactly one physical tensor"));
        };
        let tensor = self
            .archive
            .tensor(name)
            .map_err(|e| invalid(&e.to_string()))?;
        let (
            WeightEncoding::Dense {
                element_type: source,
            },
            WeightEncoding::Dense {
                element_type: target,
            },
        ) = (tensor.encoding(), &component.encoding)
        else {
            if tensor.encoding() != &component.encoding {
                return Err(invalid("stored encoding differs from the bound component"));
            }
            return self.archive.component(component);
        };
        if tensor.shape() != component.dimensions {
            return Err(invalid("stored dimensions differ from the bound component"));
        }
        if source == target {
            return self.archive.component(component);
        }
        let floating =
            |ty: ElementType| matches!(ty, ElementType::F32 | ElementType::F16 | ElementType::Bf16);
        if !floating(*source) || !floating(*target) {
            return Err(invalid(
                "only dense F32/F16/BF16 normalization is supported",
            ));
        }
        let bytes = usize::try_from(component.physical_bytes()?)
            .map_err(|_| invalid("target exceeds address space"))?;
        let mut backing = self
            .host
            .reserve(HostMemoryClass::WeightBacking, bytes)
            .and_then(|reservation| reservation.allocate_zeroed())
            .map_err(|e| invalid(&e.to_string()))?;
        for (input, output) in tensor
            .bytes()
            .chunks_exact(source.size_bytes() as usize)
            .zip(
                backing
                    .as_mut()
                    .chunks_exact_mut(target.size_bytes() as usize),
            )
        {
            let value = match source {
                ElementType::F32 => f32::from_le_bytes(input.try_into().unwrap()),
                ElementType::F16 => f16::from_le_bytes(input.try_into().unwrap()).to_f32(),
                ElementType::Bf16 => bf16::from_le_bytes(input.try_into().unwrap()).to_f32(),
                _ => unreachable!(),
            };
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
                _ => unreachable!(),
            };
            if value.is_finite() && !converted.is_finite() {
                return Err(invalid("finite dense value overflows the target type"));
            }
        }
        WeightComponentPayload::from_managed_host(
            component,
            vec![name.clone()],
            vec![tensor.source_file().to_owned()],
            component.dimensions.clone(),
            *target,
            backing.freeze(),
        )
    }
}
