//! Explicit source-decoded, RN-even F16 projection mathematics.
//!
//! These contracts authorize a separate approximate materializer, never a
//! reinterpretation of strict GGUF weights. Before a projection, decode the
//! source's declared GGUF storage to F32 using that format's decoder, then round
//! each coefficient once to IEEE binary16, round-to-nearest ties-to-even.
//! Preserve signed zero and subnormal rules; reject nonfinite source or rounded
//! coefficients before executable publication. Source identity, decoding and
//! conversion implementation remain part of the materializer's provenance.
//!
//! Every supported invocation width and phase uses these rounded coefficients.
//! Packed/participant organization cannot change the mathematical policy. A
//! projection multiplies the existing F16 input by the stored F16 coefficient,
//! accumulates in F32 (including a vendor GEMM's declared reduction order), then
//! rounds its result to the existing F16 storage boundary. This does not promise
//! bitwise agreement between different F32 reduction trees or TensorCore use.
//! Implementations must qualify against this policy's independent reference and
//! preserve the implementation identity. The inherited oracle tolerance applies
//! to that policy, not to the old unrounded strict weights; it is not a model
//! quality or serving-SLO guarantee.
//!
//! Only explicitly listed weight-input ordinals below may be transformed.
//! Operation selection alone does not grant a numeric quality artifact or
//! permission to transform an aliased weight with another unauthorized consumer.
use super::*;

pub const DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID: &str =
    "operation.dense_swiglu.gguf-f16-weights";
pub const DENSE_SWIGLU_GGUF_F16_WEIGHTS_CAPABILITY_ID: &str =
    "capability.operation.dense_swiglu.gguf-f16-weights";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID: &str =
    "operation.gated_delta_recurrent_attention.f32-master.gguf-f16-projections";
pub const GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID: &str =
    "capability.operation.gated_delta_recurrent_attention.f32-master.gguf-f16-projections";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID: &str =
    "operation.causal_paged_attention.f32-master.gguf-f16-projections";
pub const CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID: &str =
    "capability.operation.causal_paged_attention.f32-master.gguf-f16-projections";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufF16ProjectionRoleV1 {
    FfnGateUp,
    FfnDown,
    GdnInput,
    GdnOutput,
    CausalQuery,
    CausalKey,
    CausalValue,
    CausalOutput,
}

/// Pure classification of version-1 contracts. Callers must still check the
/// node's required version, original weight source and every consumer, then
/// pass the existing approximate-materializer quality verifier. No tensor-name
/// or model-name convention can authorize an additional ordinal.
pub fn gguf_f16_projection_role_v1(
    operation: &OperationId,
    input_ordinal: u32,
) -> Option<GgufF16ProjectionRoleV1> {
    use GgufF16ProjectionRoleV1::*;
    match (operation.as_str(), input_ordinal) {
        (DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID, 1) => Some(FfnGateUp),
        (DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID, 2) => Some(FfnDown),
        (GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID, 2) => {
            Some(GdnInput)
        }
        (GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID, 7) => {
            Some(GdnOutput)
        }
        (CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID, 2) => {
            Some(CausalQuery)
        }
        (CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID, 3) => Some(CausalKey),
        (CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID, 4) => {
            Some(CausalValue)
        }
        (CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID, 5) => {
            Some(CausalOutput)
        }
        _ => None,
    }
}

fn independent(
    mut contract: StandardOperationContract,
    id: &str,
    capability: &str,
) -> Result<StandardOperationContract, VNextError> {
    contract.descriptor.id = OperationId::new(id)?;
    contract.descriptor.version = ContractVersion::new(1, 0);
    contract.descriptor.provider = provider_requirement(capability, ContractVersion::new(1, 0))?;
    contract.descriptor.validate()?;
    Ok(contract)
}

/// Gate/up and down use the module's RN-F16 weight policy. Both projected
/// gate/up results and the SiLU-times-up result retain their F16 boundaries;
/// SiLU and multiplication of the two intermediate values remain F32 before
/// that rounding. Hidden input/output remain F16; residual is a separate op.
pub fn dense_swiglu_gguf_f16_weights_contract() -> Result<StandardOperationContract, VNextError> {
    independent(
        dense_swiglu_contract()?,
        DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID,
        DENSE_SWIGLU_GGUF_F16_WEIGHTS_CAPABILITY_ID,
    )
}

/// Only packed Q/K/V/Z/b/a and output projections change. F32 hidden stream,
/// F32 input normalization with F16 projection input, convolution parameters
/// and history, recurrence/Delta state, gated normalization and residual keep
/// the strict F32-master contract. Projection results remain F16.
pub fn gated_delta_recurrent_attention_f32_master_gguf_f16_projections_contract(
) -> Result<StandardOperationContract, VNextError> {
    independent(
        gated_delta_recurrent_attention_f32_master_contract()?,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
    )
}

/// Only Q, K, V and output projections change. The F32 hidden/residual stream,
/// Q/K normalization, RoPE, F16 KV storage, attention/softmax, optional output
/// gate and all persistent state retain the strict F32-master contract.
pub fn causal_paged_attention_f32_master_gguf_f16_projections_contract(
) -> Result<StandardOperationContract, VNextError> {
    independent(
        causal_paged_attention_f32_master_contract()?,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID,
        CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_CAPABILITY_ID,
    )
}

#[cfg(test)]
mod tests;
