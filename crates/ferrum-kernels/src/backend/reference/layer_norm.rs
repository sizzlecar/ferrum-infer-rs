//! Fixed-tensor BF16 LayerNorm provider. No affine weights or token execution.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    layer_norm_bf16_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    LayerNormBf16Geometry, NodeWorkContract, OperationContract, OperationFailure,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase,
    ProviderExecutionSemantics, ProviderId, ResolvedValueBinding, ResolvedValueRole,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    TensorOperationInvocation, VNextError, LAYER_NORM_BF16_CAPABILITY_ID,
    LAYER_NORM_BF16_OPERATION_ID,
};

use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceLayerNormLaunch,
};

const PROVIDER_ID: &str = "provider.reference.layer_norm.bf16";
const ESTIMATOR_ID: &str = "resource-estimator.reference.layer_norm.bf16";
const VALUE_ALIGNMENT_BYTES: u64 = 16;

pub(super) struct ReferenceLayerNormProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferenceLayerNormProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = layer_norm_bf16_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(LAYER_NORM_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime does not advertise BF16 layer-norm capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("layer_norm.rs").as_bytes(),
            include_str!("runtime/layer_norm.rs").as_bytes(),
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
            contiguous_bindings(1),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor })
    }
}

impl OperationResourceEstimator for ReferenceLayerNormProvider {
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
                "reference layer-norm estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        // Reductions use bounded stack scalars and read admitted storage directly.
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            None,
            None,
        ))
    }
}

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceLayerNormProvider {
    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "reference BF16 layer-norm supports fixed tensors only; token invocation is unsupported".to_owned(),
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
            .map_err(|message| provider_failure(identity, message))
    }
}

fn encode(
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("reference BF16 layer-norm requires one tensor participant".to_owned());
    };
    if invocation.operation().id.as_str() != LAYER_NORM_BF16_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("reference BF16 layer-norm requires its fixed tensor contract".to_owned());
    }
    let geometry = geometry(participant.bindings(), participant.attributes())?;
    let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    ReferenceDeviceCommand::layer_norm_tensor(ReferenceLayerNormLaunch {
        input: contiguous_region(participant, input, ElementType::Bf16)?,
        output: contiguous_region(participant, output, ElementType::Bf16)?,
        rows: usize::try_from(geometry.rows())
            .map_err(|_| "reference layer-norm rows exceed usize")?,
        hidden_size: usize::try_from(geometry.hidden_size())
            .map_err(|_| "reference layer-norm width exceeds usize")?,
        epsilon: geometry.epsilon(),
    })
    .map_err(|error| error.to_string())
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<LayerNormBf16Geometry, String> {
    if bindings.len() != 2 {
        return Err("reference layer-norm requires exactly one input and one output".to_owned());
    }
    let get = |name| {
        attributes
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, value)| value)
    };
    let Some(SemanticValue::Unsigned(hidden_size)) = get("hidden_size") else {
        return Err("reference layer-norm requires unsigned hidden_size".to_owned());
    };
    let Some(SemanticValue::Rational(epsilon)) = get("epsilon") else {
        return Err("reference layer-norm requires rational epsilon".to_owned());
    };
    LayerNormBf16Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        *hidden_size,
        *epsilon,
    )
    .map_err(|error| error.to_string())
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.layer_norm.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("reference provider failure metadata is bounded and static")
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
