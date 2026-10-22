//! Independent fixed-tensor BF16 row gather and per-batch concatenation.
use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    checked_concat_bytes, checked_gather_bytes, ReferenceBatchedRowConcatLaunch,
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceRowGatherLaunch,
};
use ferrum_runtime::{
    batched_row_concat_bf16_contract, row_gather_bf16_contract, AttributeId,
    BatchedOperationInvocation, BatchedRowConcatBf16Geometry, CapabilityId, ContractVersion,
    DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    NodeWorkContract, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, PhysicalWeightLayout, ProfilePhase, ProviderExecutionSemantics,
    ProviderId, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, RowGatherBf16Geometry, SemanticValue,
    TensorOperationInvocation, VNextError, WeightFormatId, BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID,
    BATCHED_ROW_CONCAT_BF16_OPERATION_ID, ROW_GATHER_BF16_CAPABILITY_ID,
    ROW_GATHER_BF16_OPERATION_ID,
};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_ROW_GATHER_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.bitcopy.row_gather.bf16.v1";
pub const REFERENCE_BATCHED_ROW_CONCAT_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.bitcopy.batched_row_concat.bf16.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReferenceRowTransportKind {
    Gather,
    Concat,
}
impl ReferenceRowTransportKind {
    fn operation_id(self) -> &'static str {
        match self {
            Self::Gather => ROW_GATHER_BF16_OPERATION_ID,
            Self::Concat => BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
        }
    }
    fn input_count(self) -> u32 {
        match self {
            Self::Gather => 1,
            Self::Concat => 2,
        }
    }
}
pub(super) struct ReferenceRowTransportProvider {
    descriptor: OperationProviderDescriptor,
    kind: ReferenceRowTransportKind,
}
impl ReferenceRowTransportProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: ReferenceRowTransportKind,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let (contract, capability, provider, estimator, profile) = match kind {
            ReferenceRowTransportKind::Gather => (
                row_gather_bf16_contract(),
                ROW_GATHER_BF16_CAPABILITY_ID,
                "provider.reference.row_gather.bf16",
                "resource-estimator.reference.row_gather.bf16",
                REFERENCE_ROW_GATHER_BF16_NUMERICAL_PROFILE_ID,
            ),
            ReferenceRowTransportKind::Concat => (
                batched_row_concat_bf16_contract(),
                BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID,
                "provider.reference.batched_row_concat.bf16",
                "resource-estimator.reference.batched_row_concat.bf16",
                REFERENCE_BATCHED_ROW_CONCAT_BF16_NUMERICAL_PROFILE_ID,
            ),
        };
        let contract = contract.map_err(contract_error)?;
        let capability = CapabilityId::new(capability).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference runtime lacks selected row-transport capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("row_transport.rs").as_bytes(),
            include_str!("runtime/row_transport.rs").as_bytes(),
            provider.as_bytes(),
            profile.as_bytes(),
        ]);
        let estimator_fingerprint =
            implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider).map_err(contract_error)?,
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
            contiguous_bindings(kind.input_count()),
            estimator,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor, kind })
    }
}
impl OperationResourceEstimator for ReferenceRowTransportProvider {
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
                "row transport estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceRowTransportProvider {
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
            "BF16 row transport requires a fixed tensor invocation".into(),
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
            .map_err(|reason| provider_failure(identity, reason))
    }
}
enum Geometry {
    Gather(RowGatherBf16Geometry),
    Concat(BatchedRowConcatBf16Geometry),
}
fn geometry(
    kind: ReferenceRowTransportKind,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    if values.len() != kind.input_count() as usize + 1 {
        return Err("row transport has incorrect input/output arity".into());
    }
    for ordinal in 0..kind.input_count() {
        if binding(values, ResolvedValueRole::Input, ordinal)?
            .weight()
            .is_some_and(|weight| {
                !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
            })
        {
            return Err("row transport requires unquantized dense weight storage".into());
        }
    }
    let input = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let output = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    match kind {
        ReferenceRowTransportKind::Gather => {
            let g =
                RowGatherBf16Geometry::new(input, output, attributes).map_err(|e| e.to_string())?;
            checked_gather_bytes(&g).map_err(|e| e.to_string())?;
            Ok(Geometry::Gather(g))
        }
        ReferenceRowTransportKind::Concat => {
            batched_row_concat_bf16_contract()
                .map_err(|e| e.to_string())?
                .descriptor()
                .validate_attributes(attributes)
                .map_err(|e| e.to_string())?;
            let Some(SemanticValue::Unsigned(batch)) = attributes
                .iter()
                .find(|(id, _)| id.as_str() == "batch")
                .map(|(_, v)| v)
            else {
                return Err("concat requires unsigned batch".into());
            };
            let g = BatchedRowConcatBf16Geometry::new(
                input,
                binding(values, ResolvedValueRole::Input, 1)?.tensor(),
                output,
                *batch,
            )
            .map_err(|e| e.to_string())?;
            checked_concat_bytes(g).map_err(|e| e.to_string())?;
            Ok(Geometry::Concat(g))
        }
    }
}
fn encode(
    kind: ReferenceRowTransportKind,
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("row transport requires exactly one tensor participant".into());
    };
    if invocation.operation().id.as_str() != kind.operation_id()
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("row transport requires its fixed tensor contract".into());
    }
    let g = geometry(kind, participant.bindings(), participant.attributes())?;
    let region = |role, ordinal| {
        contiguous_region(
            participant,
            binding(participant.bindings(), role, ordinal)?,
            ElementType::Bf16,
        )
    };
    match g {
        Geometry::Gather(geometry) => {
            ReferenceDeviceCommand::row_gather_tensor(ReferenceRowGatherLaunch {
                source: region(ResolvedValueRole::Input, 0)?,
                output: region(ResolvedValueRole::Output, 0)?,
                geometry,
            })
        }
        Geometry::Concat(geometry) => {
            ReferenceDeviceCommand::batched_row_concat_tensor(ReferenceBatchedRowConcatLaunch {
                left: region(ResolvedValueRole::Input, 0)?,
                right: region(ResolvedValueRole::Input, 1)?,
                output: region(ResolvedValueRole::Output, 0)?,
                geometry,
            })
        }
    }
    .map_err(|e| e.to_string())
}
fn provider_failure(identity: ExecutionIdentityEnvelope, reason: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.row_transport.encode",
        reason.chars().take(2048).collect::<String>(),
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
