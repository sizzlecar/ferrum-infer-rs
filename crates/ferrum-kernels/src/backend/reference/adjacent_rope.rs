//! Fixed-tensor adjacent-pair BF16 RoPE with explicit FP32 frequencies.

use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceAdjacentRopeLaunch, ReferenceAdjacentRopeShape, ReferenceDeviceBuffer,
    ReferenceDeviceCommand, ReferenceDeviceRuntime, ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{
    adjacent_rope_bf16_contract, AdjacentRopeBf16Geometry, AttributeId, BatchedOperationInvocation,
    CapabilityId, ContractVersion, DeviceRuntime, ElementType, EncodedDeviceOperation,
    ExecutionIdentityEnvelope, NodeWorkContract, OperationContract, OperationFailure,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase,
    ProviderExecutionSemantics, ProviderId, ResolvedValueBinding, ResolvedValueRole,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    TensorOperationInvocation, VNextError, ADJACENT_ROPE_BF16_CAPABILITY_ID,
    ADJACENT_ROPE_BF16_OPERATION_ID,
};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.adjacent_rope.bf16.f32_fma_rne.v1";
const PROVIDER_ID: &str = "provider.reference.adjacent_rope.bf16";
const ESTIMATOR_ID: &str = "resource-estimator.reference.adjacent_rope.bf16";

pub(super) struct ReferenceAdjacentRopeProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferenceAdjacentRopeProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = adjacent_rope_bf16_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(ADJACENT_ROPE_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks BF16 adjacent RoPE capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("adjacent_rope.rs").as_bytes(),
            include_str!("runtime/adjacent_rope.rs").as_bytes(),
            REFERENCE_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
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
            contiguous_bindings(2),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor })
    }
}

impl OperationResourceEstimator for ReferenceAdjacentRopeProvider {
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
                "adjacent RoPE estimator received another operation",
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

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceAdjacentRopeProvider {
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
            "adjacent RoPE requires a fixed tensor invocation".into(),
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
        return Err("adjacent RoPE requires one tensor participant".into());
    };
    if invocation.operation().id.as_str() != ADJACENT_ROPE_BF16_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("adjacent RoPE requires its fixed tensor contract".into());
    }
    let checked = geometry(participant.bindings(), participant.attributes())?;
    ReferenceDeviceCommand::adjacent_rope_tensor(ReferenceAdjacentRopeLaunch {
        input: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::Bf16,
        )?,
        frequencies: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            ElementType::F32,
        )?,
        output: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::Bf16,
        )?,
        shape: backend_shape(checked)?,
    })
    .map_err(|error| error.to_string())
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<AdjacentRopeBf16Geometry, String> {
    if bindings.len() != 3 || attributes.len() != 1 {
        return Err(
            "adjacent RoPE requires two inputs, one output and exactly the head_dim attribute"
                .into(),
        );
    }
    let get = |name| {
        attributes
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, value)| value)
    };
    let Some(SemanticValue::Unsigned(head_dim)) = get("head_dim") else {
        return Err("adjacent RoPE requires unsigned head_dim".into());
    };
    let checked = AdjacentRopeBf16Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        *head_dim,
    )
    .map_err(|error| error.to_string())?;
    backend_shape(checked)?
        .logical_work()
        .map_err(|error| error.to_string())?;
    Ok(checked)
}

fn backend_shape(geometry: AdjacentRopeBf16Geometry) -> Result<ReferenceAdjacentRopeShape, String> {
    Ok(ReferenceAdjacentRopeShape {
        rows: usize::try_from(geometry.rows()).map_err(|_| "adjacent RoPE rows exceed usize")?,
        columns: usize::try_from(geometry.columns())
            .map_err(|_| "adjacent RoPE columns exceed usize")?,
        head_dim: usize::try_from(geometry.head_dim())
            .map_err(|_| "adjacent RoPE head_dim exceeds usize")?,
        sequence: usize::try_from(geometry.sequence())
            .map_err(|_| "adjacent RoPE sequence exceeds usize")?,
    })
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.adjacent_rope.encode",
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
