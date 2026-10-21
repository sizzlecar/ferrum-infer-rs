//! Independent fixed-tensor BF16 modulation and gated residual providers.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_runtime::{
    gated_residual_bf16_contract, modulation_scale_bf16_contract, AttributeId,
    BatchedOperationInvocation, Bf16RowBroadcastGeometry, CapabilityId, ContractVersion,
    DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderExecutionSemantics, ProviderId,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, RowBroadcast, SemanticValue, TensorOperationInvocation,
    VNextError, GATED_RESIDUAL_BF16_CAPABILITY_ID, GATED_RESIDUAL_BF16_OPERATION_ID,
    MODULATION_SCALE_BF16_CAPABILITY_ID, MODULATION_SCALE_BF16_OPERATION_ID,
};

use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceGatedResidualLaunch, ReferenceModulationScaleLaunch,
    ReferencePointwiseShape,
};

const VALUE_ALIGNMENT_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReferencePointwiseKind {
    ModulationScale,
    GatedResidual,
}

impl ReferencePointwiseKind {
    fn operation_id(self) -> &'static str {
        match self {
            Self::ModulationScale => MODULATION_SCALE_BF16_OPERATION_ID,
            Self::GatedResidual => GATED_RESIDUAL_BF16_OPERATION_ID,
        }
    }

    fn input_count(self) -> u32 {
        match self {
            Self::ModulationScale => 2,
            Self::GatedResidual => 3,
        }
    }
}

pub(super) struct ReferencePointwiseProvider {
    descriptor: OperationProviderDescriptor,
    kind: ReferencePointwiseKind,
}

impl ReferencePointwiseProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: ReferencePointwiseKind,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let (contract, capability, provider_id, estimator_id) = match kind {
            ReferencePointwiseKind::ModulationScale => (
                modulation_scale_bf16_contract(),
                MODULATION_SCALE_BF16_CAPABILITY_ID,
                "provider.reference.modulation_scale.bf16",
                "resource-estimator.reference.modulation_scale.bf16",
            ),
            ReferencePointwiseKind::GatedResidual => (
                gated_residual_bf16_contract(),
                GATED_RESIDUAL_BF16_CAPABILITY_ID,
                "provider.reference.gated_residual.bf16",
                "resource-estimator.reference.gated_residual.bf16",
            ),
        };
        let contract = contract.map_err(contract_error)?;
        let capability = CapabilityId::new(capability).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks selected BF16 pointwise capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("pointwise.rs").as_bytes(),
            include_str!("runtime/pointwise.rs").as_bytes(),
            provider_id.as_bytes(),
        ]);
        let estimator =
            implementation_fingerprint(&[estimator_id.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider_id).map_err(contract_error)?,
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
            contiguous_bindings(kind.input_count()),
            estimator_id,
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor, kind })
    }
}

impl OperationResourceEstimator for ReferencePointwiseProvider {
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
                "reference pointwise estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        // Row parameters are read directly; no broadcast or intermediate payload.
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

impl OperationProvider<ReferenceDeviceRuntime> for ReferencePointwiseProvider {
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
            "reference BF16 pointwise supports fixed tensors only; token invocation is unsupported"
                .to_owned(),
        ))
    }

    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(self.kind, invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|message| provider_failure(identity, message))
    }
}

fn encode(
    kind: ReferencePointwiseKind,
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("reference BF16 pointwise requires one tensor participant".to_owned());
    };
    if invocation.operation().id.as_str() != kind.operation_id()
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("reference BF16 pointwise requires its fixed tensor contract".to_owned());
    }
    let geometry = geometry(kind, participant.bindings(), participant.attributes())?;
    let shape = ReferencePointwiseShape {
        rows: usize::try_from(geometry.rows())
            .map_err(|_| "reference pointwise rows exceed usize")?,
        width: usize::try_from(geometry.width())
            .map_err(|_| "reference pointwise width exceeds usize")?,
        broadcast: geometry.mode(),
    };
    let region = |role, ordinal| {
        contiguous_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            ElementType::Bf16,
        )
    };
    match kind {
        ReferencePointwiseKind::ModulationScale => {
            ReferenceDeviceCommand::modulation_scale_tensor(ReferenceModulationScaleLaunch {
                input: region(ResolvedValueRole::Input, 0)?,
                scale: region(ResolvedValueRole::Input, 1)?,
                output: region(ResolvedValueRole::Output, 0)?,
                shape,
            })
        }
        ReferencePointwiseKind::GatedResidual => {
            ReferenceDeviceCommand::gated_residual_tensor(ReferenceGatedResidualLaunch {
                input: region(ResolvedValueRole::Input, 0)?,
                gate: region(ResolvedValueRole::Input, 1)?,
                update: region(ResolvedValueRole::Input, 2)?,
                output: region(ResolvedValueRole::Output, 0)?,
                shape,
            })
        }
    }
    .map_err(|error| error.to_string())
}

fn geometry(
    kind: ReferencePointwiseKind,
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Bf16RowBroadcastGeometry, String> {
    if bindings.len() != kind.input_count() as usize + 1 {
        return Err("reference pointwise has an incorrect input/output count".to_owned());
    }
    let Some(SemanticValue::Text(mode)) = attributes
        .iter()
        .find(|(id, _)| id.as_str() == "broadcast")
        .map(|(_, value)| value)
    else {
        return Err("reference pointwise requires explicit text broadcast".to_owned());
    };
    let mode = RowBroadcast::parse(mode)
        .ok_or_else(|| "reference pointwise broadcast mode is unsupported".to_owned())?;
    let base = binding(bindings, ResolvedValueRole::Input, 0)?.tensor();
    if kind == ReferencePointwiseKind::GatedResidual
        && binding(bindings, ResolvedValueRole::Input, 2)?.tensor() != base
    {
        return Err("reference gated residual update must match its base tensor".to_owned());
    }
    Bf16RowBroadcastGeometry::new(
        base,
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        mode,
    )
    .map_err(|error| error.to_string())
}

fn provider_failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.pointwise.encode",
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
