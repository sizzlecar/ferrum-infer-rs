//! Fixed-tensor grouped-query attention over compact K/V storage.

use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceStructuredGqaAttentionLaunch,
};
use ferrum_runtime::{
    structured_gqa_attention_bf16_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderExecutionSemantics, ProviderId,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, StructuredGqaAttentionBf16Geometry,
    TensorOperationInvocation, VNextError, STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID,
    STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_STRUCTURED_GQA_ATTENTION_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.structured_gqa_attention.bf16.online_f32.libm-0.2.15.v1";
const PROVIDER_ID: &str = "provider.reference.structured_gqa_attention.bf16";
const ESTIMATOR_ID: &str = "resource-estimator.reference.structured_gqa_attention.bf16";

pub(super) struct ReferenceStructuredGqaAttentionProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferenceStructuredGqaAttentionProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = structured_gqa_attention_bf16_contract().map_err(contract_error)?;
        let capability = CapabilityId::new(STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks BF16 structured GQA attention capability",
            ));
        }
        let numerical = super::transcendental::numerical_implementation_fingerprint();
        let implementation = implementation_fingerprint(&[
            numerical.as_bytes(),
            include_str!("structured_gqa_attention.rs").as_bytes(),
            include_str!("runtime/structured_gqa_attention.rs").as_bytes(),
            include_str!("runtime/structured_attention.rs").as_bytes(),
            REFERENCE_STRUCTURED_GQA_ATTENTION_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
            PROVIDER_ID.as_bytes(),
        ]);
        let estimator =
            implementation_fingerprint(&[ESTIMATOR_ID.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::new(),
            BTreeSet::new(),
            contiguous_bindings(3),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor })
    }
}

impl OperationResourceEstimator for ReferenceStructuredGqaAttentionProvider {
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
                "structured GQA attention estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        // Direct admitted storage and scalar FP32 operations; no scratch allocation.
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceStructuredGqaAttentionProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "structured GQA attention requires a fixed tensor invocation".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|error| provider_failure(identity, error))
    }
}

fn encode(
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("structured GQA attention requires one tensor participant".into());
    };
    if invocation.operation().id.as_str() != STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("structured GQA attention requires its fixed tensor contract".into());
    }
    let checked = geometry(participant.bindings(), participant.attributes())?;
    ReferenceDeviceCommand::structured_gqa_attention_tensor(ReferenceStructuredGqaAttentionLaunch {
        query: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::Bf16,
        )?,
        key: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            ElementType::Bf16,
        )?,
        value: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            ElementType::Bf16,
        )?,
        output: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?,
        geometry: checked,
    })
    .map_err(|error| error.to_string())
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<StructuredGqaAttentionBf16Geometry, String> {
    if bindings.len() != 4 {
        return Err("structured GQA attention requires Q/K/V and one output".into());
    }
    let checked = StructuredGqaAttentionBf16Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())?;
    checked
        .element_count()
        .checked_mul(2)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("structured GQA attention byte extent exceeds usize")?;
    checked
        .rows()
        .checked_mul(checked.key_value_columns())
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("structured GQA attention KV byte extent exceeds usize")?;
    Ok(checked)
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.structured_gqa_attention.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded static provider failure metadata")
}
fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}
fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
