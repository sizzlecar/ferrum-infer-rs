//! Dual physical components remain distinct from native GGUF and plain Dense.
use super::*;
use ferrum_interfaces::vnext::{
    PhysicalStorageLayout, PhysicalWeightLayout, PhysicalWeightPadding, WeightComponentRole,
    WeightEncoding, WeightId,
};

pub(super) fn validate(value: &ResolvedValueBinding) -> Result<RnF16FragmentPlanV1, String> {
    let weight = value
        .weight()
        .ok_or("RN fragment weight metadata is absent")?;
    let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
        dense_values,
        fragment_values,
        source_format,
    } = weight.physical_layout()
    else {
        return Err("RN fragment consumer requires explicit dense and fragment layout".into());
    };
    let plan = RnF16FragmentPlanV1::from_dimensions(*source_format, value.tensor().dimensions())
        .map_err(|e| e.to_string())?;
    let exact = PhysicalStorageLayout::Contiguous {
        padding: PhysicalWeightPadding::Exact,
    };
    if !f16_contiguous(value)
        || dense_values.component_id == fragment_values.component_id
        || dense_values.storage != exact
        || fragment_values.storage != exact
        || weight.components().len() != 2
        || value.storage().components().len() != 2
    {
        return Err("RN fragment binding requires two distinct exact physical components".into());
    }
    let packed_dimensions = plan.packed_dimensions();
    for (id, role, dims, encoding, bytes) in [
        (
            &dense_values.component_id,
            WeightComponentRole::Values,
            value.tensor().dimensions(),
            ferrum_interfaces::vnext::WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            plan.dense_bytes(),
        ),
        (
            &fragment_values.component_id,
            WeightComponentRole::PackedValues,
            packed_dimensions.as_slice(),
            plan.packed_encoding(),
            plan.packed_bytes(),
        ),
    ] {
        let mut metadata = weight
            .components()
            .iter()
            .filter(|c| c.component_id() == id);
        let component = metadata
            .next()
            .ok_or("RN fragment component metadata is absent")?;
        let mut storage = value
            .storage()
            .components()
            .iter()
            .filter(|c| c.component_id() == Some(id));
        let stored = storage
            .next()
            .ok_or("RN fragment component storage is absent")?;
        let dtype = match encoding {
            WeightEncoding::Dense { element_type } => element_type,
            _ => ElementType::U8,
        };
        if metadata.next().is_some()
            || storage.next().is_some()
            || component.role() != role
            || component.physical_dimensions() != dims
            || component.encoding() != &encoding
            || component.physical_bytes().map_err(|e| e.to_string())? != bytes
            || stored.length_bytes() != bytes
            || stored.element_type() != dtype
        {
            return Err("RN fragment component differs from its checked physical plan".into());
        }
    }
    Ok(plan)
}
fn selected_id(value: &ResolvedValueBinding, fragment: bool) -> Result<&WeightId, String> {
    let weight = value
        .weight()
        .ok_or("RN fragment weight metadata is absent")?;
    let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
        dense_values,
        fragment_values,
        ..
    } = weight.physical_layout()
    else {
        return Err("RN fragment weight layout is absent".into());
    };
    Ok(if fragment {
        &fragment_values.component_id
    } else {
        &dense_values.component_id
    })
}
fn resolve(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    plan: RnF16FragmentPlanV1,
    fragment: bool,
) -> Result<CudaBufferRegion, String> {
    let value = binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?;
    // prepare has validated both components and all participant plans once.
    let id = selected_id(value, fragment)?;
    let stored = value
        .storage()
        .components()
        .iter()
        .find(|s| s.component_id() == Some(id))
        .ok_or("RN fragment selected component storage is absent")?;
    let mut views = participant
        .views()
        .iter()
        .filter(|v| v.resource_id() == stored.resource_id());
    let view = views.next().ok_or("RN fragment resource view is absent")?;
    if views.next().is_some() {
        return Err("RN fragment resource view is ambiguous".into());
    }
    let translated = view
        .translate(stored.offset_bytes(), stored.length_bytes())
        .map_err(|e| e.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or("RN fragment physical range is absent")?;
    if physical.next().is_some() {
        return Err("RN fragment weight is not physically contiguous".into());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|e| e.to_string())?;
    let (dtype, bytes) = if fragment {
        (ElementType::U8, plan.packed_bytes())
    } else {
        (ElementType::F16, plan.dense_bytes())
    };
    if region.element_type() != dtype || region.length_bytes() != bytes {
        return Err("RN fragment retained range differs from its declared span".into());
    }
    Ok(region)
}
pub(super) fn shared(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    plan: RnF16FragmentPlanV1,
    fragment: bool,
) -> Result<CudaBufferRegion, String> {
    let first = resolve(&invocation.participants()[0], ordinal, plan, fragment)?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve(participant, ordinal, plan, fragment)?;
        if !same_physical_region(&first, &candidate) {
            return Err(
                "RN fragment participants do not share the selected weight representation".into(),
            );
        }
    }
    Ok(first)
}
