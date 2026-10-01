//! Immutable linear metadata owned by one exact bound plan node. Current
//! rows, packed coordinates, selected PSOs and staging are never retained.
use super::*;
use ferrum_interfaces::vnext::{OperationCostPreparationRequest, PreparedOperationCostData};

pub(super) struct PreparedDenseCostData {
    pub(super) input: u64,
    pub(super) output: u64,
    pub(super) part: PreparedLinearPart,
}

impl PreparedDenseCostData {
    pub(super) fn unprepared(
        attributes: &std::collections::BTreeMap<
            ferrum_interfaces::vnext::AttributeId,
            ferrum_interfaces::vnext::SemanticValue,
        >,
        bindings: &[ferrum_interfaces::vnext::ResolvedValueBinding],
    ) -> Result<Option<Self>, String> {
        let input = unsigned_attribute(attributes, "in_features")?;
        let output = unsigned_attribute(attributes, "out_features")?;
        if input == 0 || output == 0 {
            return Err("dense cost route has empty features".into());
        }
        checked_u32(input, "dense cost route input width")?;
        checked_u32(output, "dense cost route output width")?;
        validate_dense_linear_bindings(bindings, input, output)?;
        let binding = binding(bindings, ResolvedValueRole::Input, 1)?;
        let Some((metadata, layout)) = plain_weight(binding, false)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            input,
            output,
            part: prepare_leaf_encoding(&metadata, &layout, output, input, 1, 0)?,
        }))
    }
}

pub(in crate::backend::metal::vnext_ops::linear) fn prepare_dense(
    request: OperationCostPreparationRequest<'_>,
) -> Option<PreparedOperationCostData> {
    if request.operation_id().as_str() != DENSE_LINEAR_OPERATION_ID {
        return None;
    }
    // This recipe belongs only to the exact immutable node/registry binding.
    // Preparation failure preserves the original checked query and its error.
    Some(PreparedOperationCostData::new(
        PreparedDenseCostData::unprepared(request.attributes(), request.bindings()).ok()??,
    ))
}

pub(super) struct PreparedSwiGluCostData {
    pub(super) hidden: u64,
    pub(super) intermediate: u64,
    pub(super) packed: u64,
    pub(super) gate: Vec<PreparedLinearPart>,
    pub(super) down: PreparedLinearPart,
    pub(super) staging_bytes: u64,
    pub(super) classes: Option<selected::prepared::PreparedSwiGluClasses>,
}

impl PreparedSwiGluCostData {
    /// The compatibility path uses the same original static renderer, without
    /// compiling classes for a value that will be dropped after this query.
    pub(super) fn unprepared(
        attributes: &std::collections::BTreeMap<
            ferrum_interfaces::vnext::AttributeId,
            ferrum_interfaces::vnext::SemanticValue,
        >,
        bindings: &[ferrum_interfaces::vnext::ResolvedValueBinding],
    ) -> Result<Option<Self>, String> {
        let hidden = unsigned_attribute(attributes, "hidden_size")?;
        let intermediate = unsigned_attribute(attributes, "intermediate_size")?;
        if hidden == 0 || intermediate == 0 {
            return Err("SwiGLU cost has empty features".into());
        }
        validate_swiglu_bindings(bindings, hidden, intermediate)?;
        let Some((gate_components, gate_layout)) =
            plain_weight(binding(bindings, ResolvedValueRole::Input, 1)?, true)?
        else {
            return Ok(None);
        };
        let Some((down_components, down_layout)) =
            plain_weight(binding(bindings, ResolvedValueRole::Input, 2)?, false)?
        else {
            return Ok(None);
        };
        let packed = intermediate
            .checked_mul(2)
            .ok_or("SwiGLU packed width overflows")?;
        let gate = match &gate_layout {
            MetalResolvedWeightLayout::Composite { parts } => {
                prepare_gate_up_partition(parts, intermediate, hidden, |layout, offset| {
                    prepare_leaf_encoding(&gate_components, layout, intermediate, hidden, 2, offset)
                })?
            }
            layout => vec![prepare_leaf_encoding(
                &gate_components,
                layout,
                packed,
                hidden,
                2,
                0,
            )?],
        };
        let down =
            prepare_leaf_encoding(&down_components, &down_layout, hidden, intermediate, 1, 0)?;
        Ok(Some(Self {
            hidden,
            intermediate,
            packed,
            gate,
            down,
            staging_bytes: staged_prefill::workspace_bytes(bindings, hidden, intermediate)?,
            classes: None,
        }))
    }

    fn compile_classes(&mut self) {
        // Placeholder M is not a selected launch: class preparation binds only
        // fixed matrix ABI. The query independently chooses its real M/PSO.
        self.classes = (|| {
            let gate = self
                .gate
                .iter()
                .map(|&part| linear_launch(part, 0, 0, 1, self.hidden, self.packed, 0, 0).ok())
                .collect::<Option<Vec<_>>>()?;
            let down =
                linear_launch(self.down, 0, 0, 1, self.intermediate, self.hidden, 0, 0).ok()?;
            selected::prepared::PreparedSwiGluClasses::new(&gate, down, self.packed)
        })();
    }
}

pub(in crate::backend::metal::vnext_ops::linear) fn prepare_swiglu(
    request: OperationCostPreparationRequest<'_>,
) -> Option<PreparedOperationCostData> {
    if request.operation_id().as_str() != DENSE_SWIGLU_OPERATION_ID {
        return None;
    }
    // A preparation error never converts unsupported/invalid metadata into a
    // route. None preserves the original checked query and its original error.
    let mut data =
        PreparedSwiGluCostData::unprepared(request.attributes(), request.bindings()).ok()??;
    data.compile_classes();
    Some(PreparedOperationCostData::new(data))
}

#[cfg(test)]
mod tests;
