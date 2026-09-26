//! Physical consumer boundary for the explicit RN-F16 projection contracts.
//! This validates already approved/materialized weights; it never grants
//! materialization authority or attaches native cost evidence to vendor GEMM.
use super::*;
use ferrum_interfaces::vnext::{
    gguf_f16_projection_role_v1, OperationId, PhysicalWeightLayout, WeightEncoding,
};

pub(super) fn is_operation(operation: &OperationId) -> bool {
    gguf_f16_projection_role_v1(operation, 2).is_some()
}

pub(super) fn provider_formats(
    operation: &OperationId,
    mut formats: BTreeSet<WeightFormatId>,
) -> Result<BTreeSet<WeightFormatId>, VNextError> {
    let mixed = WeightFormatId::new(
        crate::gguf_f16_projection_materializer::GGUF_F16_PROJECTION_FORMAT_ID,
    )?;
    if is_operation(operation) {
        // The global execution schema is mixed: these operands are Dense F16,
        // while unrelated head/embedding operands retain their original ABI.
        return Ok(BTreeSet::from([mixed]));
    }
    if formats.contains(&WeightFormatId::new("weight-format.gguf.native-block")?) {
        // A container label is not permission to reinterpret physical bytes.
        // Existing retained consumers still validate/decode original layouts.
        formats.insert(mixed);
    }
    Ok(formats)
}

pub(super) fn validate_values(
    operation: &OperationId,
    values: &[ResolvedValueBinding],
) -> Result<(), String> {
    if !is_operation(operation) {
        return Ok(());
    }
    for ordinal in 0..=7 {
        if gguf_f16_projection_role_v1(operation, ordinal).is_none() {
            continue;
        }
        let value = binding(values, ResolvedValueRole::Input, ordinal)?;
        let weight = value
            .weight()
            .ok_or("RN-F16 projection lacks physical weight metadata")?;
        let PhysicalWeightLayout::Dense { component_id } = weight.physical_layout() else {
            return Err("RN-F16 projection requires one materialized dense weight".into());
        };
        let [component] = weight.components() else {
            return Err("RN-F16 projection has multiple physical components".into());
        };
        let [storage] = value.storage().components() else {
            return Err("RN-F16 projection has multiple storage spans".into());
        };
        if value.tensor().element_type() != ElementType::F16
            || value.tensor().layout() != &ResolvedTensorLayout::Contiguous
            || component.component_id() != component_id
            || component.physical_dimensions() != value.tensor().dimensions()
            || component.encoding()
                != &(WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
            || storage.component_id() != Some(component_id)
            || storage.element_type() != ElementType::F16
            || storage.length_bytes() != component.physical_bytes().map_err(|e| e.to_string())?
        {
            return Err(
                "RN-F16 projection binding differs from its full dense F16 contract".into(),
            );
        }
    }
    Ok(())
}

pub(super) fn validate_invocation(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<(), String> {
    if is_operation(&invocation.operation().id) {
        if invocation.operation().version != ContractVersion::new(1, 0) {
            return Err("RN-F16 projection requires operation contract 1.0".into());
        }
        for participant in invocation.participants() {
            validate_values(&invocation.operation().id, participant.bindings())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
