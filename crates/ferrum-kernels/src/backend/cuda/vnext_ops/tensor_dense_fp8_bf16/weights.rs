//! Exact WholeTensor storage resolution, independent of container/file names.

use super::super::tensor_storage::component;
pub(super) use super::super::tensor_storage::scratch;
use super::*;
use ferrum_runtime::{
    InvocationWorkProjection, OperationInvocation, PhysicalStorageLayout, PhysicalWeightLayout,
    PhysicalWeightPadding, QuantizationSpec, ResolvedStorageComponent, WeightComponentRole,
    WeightEncoding,
};

pub(super) struct Components<'a> {
    pub packed: &'a ResolvedStorageComponent,
    pub scale: &'a ResolvedStorageComponent,
}

pub(super) fn validate(binding: &ResolvedValueBinding) -> Result<Components<'_>, String> {
    let weight = binding
        .weight()
        .ok_or("FP8 weight lacks typed physical layout")?;
    weight
        .validate_logical(
            binding.tensor().dimensions(),
            binding.tensor().element_type(),
        )
        .map_err(|e| e.to_string())?;
    let PhysicalWeightLayout::QuantizedTensor {
        packed_values,
        packed_dimensions,
        scales,
    } = weight.physical_layout()
    else {
        return Err(
            "FP8 dense requires a WholeTensor quantized leaf, without transforms or wrappers"
                .into(),
        );
    };
    let exact = |storage: &PhysicalStorageLayout| {
        matches!(
            storage,
            PhysicalStorageLayout::Contiguous {
                padding: PhysicalWeightPadding::Exact
            }
        )
    };
    if !exact(&packed_values.storage)
        || !exact(&scales.storage)
        || packed_values.component_id == scales.component_id
        || packed_dimensions != binding.tensor().dimensions()
        || packed_dimensions.len() != 2
        || weight.components().len() != 2
        || binding.storage().components().len() != 2
    {
        return Err(
            "FP8 dense requires exact unpadded OI bytes and one separate scalar scale".into(),
        );
    }
    let packed = weight
        .components()
        .iter()
        .find(|c| c.component_id() == &packed_values.component_id)
        .ok_or("FP8 packed metadata absent")?;
    let scale = weight
        .components()
        .iter()
        .find(|c| c.component_id() == &scales.component_id)
        .ok_or("FP8 scale metadata absent")?;
    if packed.role() != WeightComponentRole::PackedValues
        || packed.physical_dimensions() != packed_dimensions
        || packed.encoding()
            != &WeightEncoding::Quantized(QuantizationSpec::fp8_e4m3fn_whole_tensor_f32())
        || scale.role() != WeightComponentRole::Scales
        || scale.physical_dimensions() != [1]
        || scale.encoding()
            != &(WeightEncoding::Dense {
                element_type: ElementType::F32,
            })
    {
        return Err(
            "FP8 packed/scale dtype, shape or encoding differs from WholeTensor E4M3FN/F32 ABI"
                .into(),
        );
    }
    let stored = |id, dtype, bytes| {
        let mut found = binding
            .storage()
            .components()
            .iter()
            .filter(|s| s.component_id() == Some(id));
        let value = found.next().ok_or("FP8 component storage absent")?;
        if found.next().is_some() || value.element_type() != dtype || value.length_bytes() != bytes
        {
            return Err("FP8 component storage identity/extent/dtype mismatch");
        }
        Ok(value)
    };
    let packed = stored(
        &packed_values.component_id,
        ElementType::U8,
        packed.physical_bytes().map_err(|e| e.to_string())?,
    )?;
    let scale = stored(&scales.component_id, ElementType::F32, 4)?;
    Ok(Components { packed, scale })
}

pub(super) fn retain<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
) -> Result<[CudaBufferRegion; 2], String> {
    let components = validate(binding)?;
    Ok([
        component(participant, components.packed, 16)?,
        component(participant, components.scale, 4)?,
    ])
}

pub(super) fn plain<'a, W: InvocationWorkProjection<'a>>(
    participant: &OperationInvocation<'a, CudaDeviceBuffer, W>,
    binding: &ResolvedValueBinding,
    bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let [stored] = binding.storage().components() else {
        return Err("FP8 X/Y require one BF16 component".into());
    };
    if stored.element_type() != ElementType::Bf16 || stored.length_bytes() != bytes {
        return Err("FP8 X/Y storage differs from the logical BF16 tensor".into());
    }
    component(participant, stored, 16)
}
