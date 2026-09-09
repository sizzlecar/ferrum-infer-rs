use std::collections::BTreeSet;

use ferrum_interfaces::vnext::*;

use super::super::vnext_runtime::{
    CpuDeviceBuffer, CpuDeviceCommand, CpuDeviceRuntime, CpuRuntimeError,
};
use super::composition::implementation_fingerprint;
use super::lowering;
use super::scalar::CpuFloat;
use crate::gguf_blocks::GgufBlockFormat;

#[derive(Clone, Copy)]
pub(super) enum CpuOperation {
    Embedding(CpuFloat),
    RmsNorm { input: CpuFloat, output: CpuFloat },
    Residual(CpuFloat),
    Linear,
    LastTokenLinear(CpuFloat),
    SwiGlu,
    Argmax(CpuFloat),
    GatedDelta(CpuFloat),
    CausalAttention(CpuFloat),
}

impl CpuOperation {
    pub(super) const IMPLEMENTED: &[Self] = &[
        Self::Embedding(CpuFloat::F16),
        Self::Embedding(CpuFloat::F32),
        Self::RmsNorm {
            input: CpuFloat::F16,
            output: CpuFloat::F16,
        },
        Self::RmsNorm {
            input: CpuFloat::F32,
            output: CpuFloat::F16,
        },
        Self::RmsNorm {
            input: CpuFloat::F32,
            output: CpuFloat::F32,
        },
        Self::Residual(CpuFloat::F16),
        Self::Residual(CpuFloat::F32),
        Self::Linear,
        Self::LastTokenLinear(CpuFloat::F16),
        Self::LastTokenLinear(CpuFloat::F32),
        Self::SwiGlu,
        Self::Argmax(CpuFloat::F16),
        Self::Argmax(CpuFloat::F32),
        Self::GatedDelta(CpuFloat::F16),
        Self::GatedDelta(CpuFloat::F32),
        Self::CausalAttention(CpuFloat::F16),
        Self::CausalAttention(CpuFloat::F32),
    ];

    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Embedding(CpuFloat::F16) => token_embedding_contract(),
            Self::Embedding(CpuFloat::F32) => token_embedding_f32_master_contract(),
            Self::RmsNorm {
                input: CpuFloat::F16,
                output: CpuFloat::F16,
            } => rms_norm_contract(),
            Self::RmsNorm {
                input: CpuFloat::F32,
                output: CpuFloat::F16,
            } => rms_norm_f32_to_f16_contract(),
            Self::RmsNorm {
                input: CpuFloat::F32,
                output: CpuFloat::F32,
            } => rms_norm_f32_contract(),
            Self::RmsNorm { .. } => Err(invalid_plan("CPU has no F16 to F32 norm contract")),
            Self::Residual(CpuFloat::F16) => residual_add_contract(),
            Self::Residual(CpuFloat::F32) => residual_add_f32_f16_contract(),
            Self::Linear => dense_linear_contract(),
            Self::LastTokenLinear(CpuFloat::F16) => last_token_dense_linear_contract(),
            Self::LastTokenLinear(CpuFloat::F32) => last_token_dense_linear_f32_contract(),
            Self::SwiGlu => dense_swiglu_contract(),
            Self::Argmax(CpuFloat::F16) => last_token_masked_argmax_contract(),
            Self::Argmax(CpuFloat::F32) => last_token_masked_argmax_f32_contract(),
            Self::GatedDelta(CpuFloat::F16) => gated_delta_recurrent_attention_contract(),
            Self::GatedDelta(CpuFloat::F32) => {
                gated_delta_recurrent_attention_f32_master_contract()
            }
            Self::CausalAttention(CpuFloat::F16) => causal_paged_attention_contract(),
            Self::CausalAttention(CpuFloat::F32) => causal_paged_attention_f32_master_contract(),
        }
    }

    pub(super) fn native_id(self) -> &'static str {
        match self {
            Self::Embedding(_) => "cpu.token_embedding",
            Self::RmsNorm { .. } => "cpu.rms_norm",
            Self::Residual(_) => "cpu.residual_add",
            Self::Linear => "cpu.dense_linear",
            Self::LastTokenLinear(_) => "cpu.last_token_dense_linear",
            Self::SwiGlu => "cpu.dense_swiglu",
            Self::Argmax(_) => "cpu.last_token_masked_argmax",
            Self::GatedDelta(_) => "cpu.gated_delta_recurrent_attention",
            Self::CausalAttention(_) => "cpu.causal_attention",
        }
    }
}

pub(super) struct CpuOperationProvider {
    operation: CpuOperation,
    descriptor: OperationProviderDescriptor,
}

impl CpuOperationProvider {
    pub(super) fn new(
        runtime: &CpuDeviceRuntime,
        operation: CpuOperation,
    ) -> Result<Self, CpuRuntimeError> {
        let contract = operation.contract()?;
        let descriptor = contract.descriptor();
        let capabilities = descriptor.provider.required_capabilities.clone();
        if !capabilities.is_subset(&runtime.descriptor().capabilities) {
            return Err(CpuRuntimeError::new(
                "CPU operation is absent from its runtime capabilities",
            ));
        }
        let has_matrix = matches!(
            operation,
            CpuOperation::Embedding(_)
                | CpuOperation::Linear
                | CpuOperation::LastTokenLinear(_)
                | CpuOperation::SwiGlu
                | CpuOperation::GatedDelta(_)
                | CpuOperation::CausalAttention(_)
        );
        let has_weights = has_matrix || matches!(operation, CpuOperation::RmsNorm { .. });
        let weight_formats = if has_weights {
            [
                "weight-format.safetensors.dense",
                "weight-format.gguf.native-block",
            ]
            .into_iter()
            .map(WeightFormatId::new)
            .collect::<Result<_, _>>()?
        } else {
            BTreeSet::new()
        };
        let quantization_formats = if has_matrix {
            [
                GgufBlockFormat::Q3K,
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q5K,
                GgufBlockFormat::Q6K,
                GgufBlockFormat::Q8_0,
                GgufBlockFormat::Iq3S,
                GgufBlockFormat::Iq4Nl,
                GgufBlockFormat::Iq4Xs,
            ]
            .into_iter()
            .map(|format| QuantizationFormatId::new(format.format_id()))
            .collect::<Result<_, _>>()?
        } else {
            BTreeSet::new()
        };
        let paged_kv =
            DynamicStorageRequirement::new(vec![super::causal_attention::kv_storage_profile()?])?;
        let storage = (0..descriptor.inputs.len() as u32)
            .map(|ordinal| {
                ProviderStorageBindingRequirement::new(
                    ResolvedValueRole::Input,
                    ordinal,
                    if matches!(operation, CpuOperation::CausalAttention(_)) && ordinal == 8 {
                        paged_kv.clone()
                    } else {
                        DynamicStorageRequirement::contiguous()
                    },
                )
            })
            .chain(std::iter::once(ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                0,
                DynamicStorageRequirement::contiguous(),
            )))
            .collect();
        let identity = descriptor.id.as_str();
        Ok(Self {
            operation,
            descriptor: OperationProviderDescriptor::new(
                ProviderId::new(format!("provider.cpu.{identity}"))?,
                descriptor.id.clone(),
                descriptor.fingerprint()?,
                implementation_fingerprint(),
                ProviderExecutionSemantics::bitwise_eager_only(),
                descriptor.version,
                runtime.descriptor().id.clone(),
                capabilities,
                weight_formats,
                quantization_formats,
                storage,
                format!("resource-estimator.cpu.{identity}"),
                ContractVersion::new(1, 0),
                implementation_fingerprint(),
            )?,
        })
    }
}

impl OperationResourceEstimator for CpuOperationProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "CPU resource estimator received a different operation contract",
            ));
        }
        let formula = match self.operation {
            CpuOperation::SwiGlu => {
                let width = unsigned(request.attributes(), "intermediate_size")?;
                Some(ProviderWorkspaceSizeFormula::tokens(
                    width
                        .checked_mul(6)
                        .ok_or_else(|| invalid_plan("CPU SwiGLU workspace overflows"))?,
                )?)
            }
            CpuOperation::Argmax(_) => Some(ProviderWorkspaceSizeFormula::actual_sequences(
                unsigned(request.attributes(), "vocab_size")?,
            )?),
            CpuOperation::GatedDelta(_) => {
                let shape =
                    super::gated_delta::GatedDeltaShape::from_attributes(request.attributes())
                        .map_err(|error| invalid_plan(error.to_string()))?;
                Some(ProviderWorkspaceSizeFormula::actual_sequences(
                    super::gated_delta_launch::workspace_bytes(shape)
                        .map_err(|error| invalid_plan(error.to_string()))?,
                )?)
            }
            CpuOperation::CausalAttention(_) => {
                let shape =
                    super::causal_attention::CausalShape::from_attributes(request.attributes())
                        .map_err(|error| invalid_plan(error.to_string()))?;
                Some(ProviderWorkspaceSizeFormula::actual_sequences(
                    super::causal_attention_launch::workspace_bytes(shape)
                        .map_err(|error| invalid_plan(error.to_string()))?,
                )?)
            }
            _ => None,
        };
        let scratch = formula
            .map(|formula| {
                ProviderWorkspaceRequirement::from_formula(
                    formula,
                    16,
                    ProviderWorkspaceScope::Invocation,
                    ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                    DynamicStorageRequirement::contiguous(),
                )
            })
            .transpose()?;
        let estimate = OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            scratch,
            None,
        );
        if matches!(self.operation, CpuOperation::CausalAttention(_)) {
            Ok(
                estimate.with_binding(ProviderWorkspaceRequirement::from_formula(
                    ProviderWorkspaceSizeFormula::actual_sequences(16)?,
                    16,
                    ProviderWorkspaceScope::Invocation,
                    ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                    DynamicStorageRequirement::contiguous(),
                )?),
            )
        } else {
            Ok(estimate)
        }
    }
}

impl OperationProvider<CpuDeviceRuntime> for CpuOperationProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CpuDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CpuDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        let result = if invocation.operation().id != *self.descriptor.operation_id()
            || invocation.provider_id() != self.descriptor.provider_id()
        {
            Err(CpuRuntimeError::new(
                "CPU invocation differs from its selected provider",
            ))
        } else {
            lowering::encode(self.operation, &invocation)
        };
        result.map_err(|error| {
            OperationFailure::new(
                identity,
                ProfilePhase::Forward,
                "cpu.operation.encode",
                error.to_string().chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued CPU invocation carries a valid execution identity")
        })
    }
}

pub(super) fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn unsigned(
    attributes: &std::collections::BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, VNextError> {
    match attributes
        .iter()
        .find(|(id, _)| id.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) if *value > 0 => Ok(*value),
        _ => Err(invalid_plan(format!(
            "CPU operation lacks positive unsigned attribute {name}"
        ))),
    }
}
