//! Same-shape BF16 residual addition, without a gate allocation.

use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceActivationShape, ReferenceDeviceBuffer, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeError, ReferencePlainResidualAddLaunch,
};
use ferrum_runtime::{
    plain_residual_add_bf16_contract, AttributeId, BatchedOperationInvocation,
    Bf16SameShapeGeometry, CapabilityId, ContractVersion, DeviceRuntime, ElementType,
    EncodedDeviceOperation, ExecutionIdentityEnvelope, NodeWorkContract, OperationContract,
    OperationFailure, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase,
    ProviderExecutionSemantics, ProviderId, ResolvedValueBinding, ResolvedValueRole,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue,
    TensorOperationInvocation, VNextError, PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID,
    PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_PLAIN_RESIDUAL_ADD_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.plain_residual_add.bf16.f32_rne.v1";
const PROVIDER_ID: &str = "provider.reference.plain_residual_add.bf16";
const ESTIMATOR_ID: &str = "resource-estimator.reference.plain_residual_add.bf16";

pub(super) struct ReferencePlainResidualAddProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferencePlainResidualAddProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = plain_residual_add_bf16_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(PLAIN_RESIDUAL_ADD_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks BF16 plain residual add capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("plain_residual_add.rs").as_bytes(),
            include_str!("runtime/plain_residual_add.rs").as_bytes(),
            REFERENCE_PLAIN_RESIDUAL_ADD_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
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

impl OperationResourceEstimator for ReferencePlainResidualAddProvider {
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
                "plain residual add estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        // Direct admitted storage and scalar temporaries; no global scratch.
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

impl OperationProvider<ReferenceDeviceRuntime> for ReferencePlainResidualAddProvider {
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
            "plain residual add requires a fixed tensor invocation".into(),
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
        return Err("plain residual add requires one tensor participant".into());
    };
    if invocation.operation().id.as_str() != PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("plain residual add requires its fixed tensor contract".into());
    }
    let checked = geometry(participant.bindings(), participant.attributes())?;
    ReferenceDeviceCommand::plain_residual_add_tensor(ReferencePlainResidualAddLaunch {
        input: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::Bf16,
        )?,
        update: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            ElementType::Bf16,
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
) -> Result<Bf16SameShapeGeometry, String> {
    if bindings.len() != 3 || !attributes.is_empty() {
        return Err("plain residual add requires two inputs, one output and no attributes".into());
    }
    if binding(bindings, ResolvedValueRole::Input, 1)?.tensor()
        != binding(bindings, ResolvedValueRole::Input, 0)?.tensor()
    {
        return Err("plain residual add requires exactly equal input tensor specifications".into());
    }
    let checked = Bf16SameShapeGeometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|error| error.to_string())?;
    backend_shape(checked)?
        .logical_work(1)
        .map_err(|error| error.to_string())?;
    Ok(checked)
}

fn backend_shape(geometry: Bf16SameShapeGeometry) -> Result<ReferenceActivationShape, String> {
    Ok(ReferenceActivationShape {
        rows: usize::try_from(geometry.rows())
            .map_err(|_| "plain residual add rows exceed usize")?,
        width: usize::try_from(geometry.width())
            .map_err(|_| "plain residual add width exceeds usize")?,
    })
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.plain_residual_add.encode",
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
