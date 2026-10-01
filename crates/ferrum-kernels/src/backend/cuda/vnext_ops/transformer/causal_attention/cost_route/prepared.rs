//! Immutable numerical metadata from the exact provider/plan-node binding.
//! Current rows, context, address aliasing, selection and work stay in route().
use super::*;
use ferrum_interfaces::vnext::OperationId;

pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) struct PreparedCostData {
    pub(super) shape: CausalAttentionShape,
    pub(super) projection: CausalProjection,
    pub(super) rounded: bool,
    pub(super) parts: Vec<Vec<weights::MatrixPart>>,
    pub(super) template: selected::CostTemplate,
}

impl PreparedCostData {
    pub(in crate::backend::cuda::vnext_ops::transformer::causal_attention) fn new(
        operation: &OperationId,
        bindings: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        semantics: CausalAttentionSemantics,
        precision: CausalPrecision,
        #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
    ) -> Result<Option<Self>, String> {
        super::super::super::gguf_f16_projection::validate_values(operation, bindings)?;
        let rounded = super::super::super::gguf_f16_projection::is_operation(operation);
        let shape = CausalAttentionShape::from_attributes_for(attributes, semantics)?;
        validate_signature_values(bindings, shape, semantics, precision)?;
        if shape.int8_kv {
            return Ok(None);
        }
        let projection = CausalProjection::from_values(
            bindings,
            #[cfg(feature = "vllm-marlin")]
            projection_runtime,
        )?;
        if !(matches!(projection, CausalProjection::Native { .. }) && !rounded
            || matches!(projection, CausalProjection::F16) && rounded)
        {
            return Ok(None);
        }
        let parts = if rounded {
            Vec::new()
        } else {
            (2..=5)
                .map(|ordinal| {
                    let value = binding(bindings, ResolvedValueRole::Input, ordinal)?;
                    weights::matrix_parts(
                        value.weight().ok_or("causal projection metadata absent")?,
                        value.tensor().dimensions(),
                    )
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        Ok(Some(Self {
            template: selected::CostTemplate::new(shape, precision, projection)?,
            shape,
            projection,
            rounded,
            parts,
        }))
    }
}

#[cfg(test)]
mod tests;
