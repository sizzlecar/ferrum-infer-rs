//! Fixed-tensor whole-row BF16 RMSNorm; independent of token/F16 RMSNorm.

use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceZeroCenteredRmsNormLaunch,
    ReferenceZeroCenteredRmsNormShape,
};
use ferrum_runtime::{
    zero_centered_rms_norm_bf16_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderExecutionSemantics, ProviderId,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, TensorOperationInvocation, VNextError,
    WeightFormatId, ZeroCenteredRmsNormBf16Geometry, ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID,
    ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID,
};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_ZERO_CENTERED_RMS_NORM_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.zero_centered_rms_norm.bf16.pairwise_f32.sqrt_div.single_b.v1";
const PROVIDER_ID: &str = "provider.reference.zero_centered_rms_norm.bf16";
const ESTIMATOR_ID: &str = "resource-estimator.reference.zero_centered_rms_norm.bf16";

pub(super) struct ReferenceZeroCenteredRmsNormProvider {
    descriptor: OperationProviderDescriptor,
}

impl ReferenceZeroCenteredRmsNormProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = zero_centered_rms_norm_bf16_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(ZERO_CENTERED_RMS_NORM_BF16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks BF16 zero-centered RMSNorm capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("zero_centered_rms_norm.rs").as_bytes(),
            include_str!("runtime/zero_centered_rms_norm.rs").as_bytes(),
            REFERENCE_ZERO_CENTERED_RMS_NORM_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
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
            BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?]),
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

impl OperationResourceEstimator for ReferenceZeroCenteredRmsNormProvider {
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
                "zero-centered RMSNorm estimator received another operation",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        // Direct admitted storage, fixed scalar reduction stack, no scratch allocation.
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

impl OperationProvider<ReferenceDeviceRuntime> for ReferenceZeroCenteredRmsNormProvider {
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
            "zero-centered RMSNorm requires a fixed tensor invocation".into(),
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
        return Err("zero-centered RMSNorm requires one tensor participant".into());
    };
    if invocation.operation().id.as_str() != ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("zero-centered RMSNorm requires its fixed tensor contract".into());
    }
    let checked = geometry(participant.bindings(), participant.attributes())?;
    ReferenceDeviceCommand::zero_centered_rms_norm_tensor(ReferenceZeroCenteredRmsNormLaunch {
        input: contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::Bf16,
        )?,
        weight: contiguous_region(
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
) -> Result<ZeroCenteredRmsNormBf16Geometry, String> {
    if bindings.len() != 3 || attributes.len() != 1 {
        return Err(
            "zero-centered RMSNorm requires two inputs, one output and exactly epsilon attribute"
                .into(),
        );
    }
    let get = |name| {
        attributes
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, value)| value)
    };
    let Some(SemanticValue::Rational(epsilon)) = get("epsilon") else {
        return Err("zero-centered RMSNorm requires rational epsilon".into());
    };
    let checked = ZeroCenteredRmsNormBf16Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        *epsilon,
    )
    .map_err(|error| error.to_string())?;
    backend_shape(checked)?
        .logical_work()
        .map_err(|error| error.to_string())?;
    Ok(checked)
}

fn backend_shape(
    geometry: ZeroCenteredRmsNormBf16Geometry,
) -> Result<ReferenceZeroCenteredRmsNormShape, String> {
    Ok(ReferenceZeroCenteredRmsNormShape {
        rows: usize::try_from(geometry.rows())
            .map_err(|_| "zero-centered RMSNorm rows exceed usize")?,
        columns: usize::try_from(geometry.columns())
            .map_err(|_| "zero-centered RMSNorm columns exceed usize")?,
        epsilon: geometry.epsilon(),
    })
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.zero_centered_rms_norm.encode",
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
