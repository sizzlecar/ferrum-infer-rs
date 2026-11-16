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
use super::runtime::{ReferenceRowIndexGatherLaunch, ReferenceRowWriteLaunch};
use ferrum_runtime::{
    batched_row_concat_bf16_contract, row_gather_bf16_contract, AttributeId,
    BatchedOperationInvocation, BatchedRowConcatBf16Geometry, CapabilityId, ContractVersion,
    DeviceRuntime, ElementType, EncodedDeviceOperation, ExecutionIdentityEnvelope,
    LeadingBatchBf16Geometry, LeadingBatchBf16Kind, NodeWorkContract, OperationContract,
    OperationFailure, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, PhysicalWeightLayout,
    ProfilePhase, ProviderExecutionSemantics, ProviderId, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, RowGatherBf16Geometry, SemanticValue,
    TensorOperationInvocation, VNextError, WeightFormatId, BATCHED_ROW_CONCAT_BF16_CAPABILITY_ID,
    BATCHED_ROW_CONCAT_BF16_OPERATION_ID, ROW_GATHER_BF16_CAPABILITY_ID,
    ROW_GATHER_BF16_OPERATION_ID,
};
use ferrum_runtime::{
    row_index_gather_or_zero_bf16_contract, RowIndexGatherBf16Geometry,
    ROW_INDEX_GATHER_OR_ZERO_BF16_CAPABILITY_ID, ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID,
};
use ferrum_runtime::{
    row_write_bf16_contract, RowWriteBf16Geometry, ROW_WRITE_BF16_CAPABILITY_ID,
    ROW_WRITE_BF16_OPERATION_ID,
};
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_ROW_GATHER_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.bitcopy.row_gather.bf16.v1";
pub const REFERENCE_BATCHED_ROW_CONCAT_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.bitcopy.batched_row_concat.bf16.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ReferenceRowTransportKind {
    Gather,
    RowWrite,
    Concat,
    IndexGather,
    MergeLeadingBatch,
    SplitLeadingBatch,
    MergeLeadingBatchView,
    SplitLeadingBatchView,
}
impl ReferenceRowTransportKind {
    fn operation_id(self) -> &'static str {
        match self {
            Self::MergeLeadingBatch => LeadingBatchBf16Kind::Merge.operation_id(),
            Self::MergeLeadingBatchView => LeadingBatchBf16Kind::Merge.view_operation_id(),
            Self::SplitLeadingBatch => LeadingBatchBf16Kind::Split.operation_id(),
            Self::SplitLeadingBatchView => LeadingBatchBf16Kind::Split.view_operation_id(),
            Self::Gather => ROW_GATHER_BF16_OPERATION_ID,
            Self::RowWrite => ROW_WRITE_BF16_OPERATION_ID,
            Self::Concat => BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
            Self::IndexGather => ROW_INDEX_GATHER_OR_ZERO_BF16_OPERATION_ID,
        }
    }
    fn input_count(self) -> u32 {
        match self {
            Self::Gather
            | Self::MergeLeadingBatch
            | Self::SplitLeadingBatch
            | Self::MergeLeadingBatchView
            | Self::SplitLeadingBatchView => 1,
            Self::Concat | Self::IndexGather | Self::RowWrite => 2,
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
            ReferenceRowTransportKind::MergeLeadingBatch => (
                LeadingBatchBf16Kind::Merge.contract(),
                LeadingBatchBf16Kind::Merge.capability_id(),
                "provider.reference.merge_leading_batch.bf16",
                "resource-estimator.reference.merge_leading_batch.bf16",
                "reference.bitcopy.merge_leading_batch.bf16.v1",
            ),
            ReferenceRowTransportKind::MergeLeadingBatchView => (
                LeadingBatchBf16Kind::Merge.view_contract(),
                LeadingBatchBf16Kind::Merge.view_capability_id(),
                "provider.reference.view.merge_leading_batch.bf16",
                "resource-estimator.reference.view.merge_leading_batch.bf16",
                "reference.view.bitcopy.merge_leading_batch.bf16.v1",
            ),
            ReferenceRowTransportKind::SplitLeadingBatch => (
                LeadingBatchBf16Kind::Split.contract(),
                LeadingBatchBf16Kind::Split.capability_id(),
                "provider.reference.split_leading_batch.bf16",
                "resource-estimator.reference.split_leading_batch.bf16",
                "reference.bitcopy.split_leading_batch.bf16.v1",
            ),
            ReferenceRowTransportKind::SplitLeadingBatchView => (
                LeadingBatchBf16Kind::Split.view_contract(),
                LeadingBatchBf16Kind::Split.view_capability_id(),
                "provider.reference.view.split_leading_batch.bf16",
                "resource-estimator.reference.view.split_leading_batch.bf16",
                "reference.view.bitcopy.split_leading_batch.bf16.v1",
            ),
            ReferenceRowTransportKind::IndexGather => (
                row_index_gather_or_zero_bf16_contract(),
                ROW_INDEX_GATHER_OR_ZERO_BF16_CAPABILITY_ID,
                "provider.reference.row_index_gather_or_zero.bf16",
                "resource-estimator.reference.row_index_gather_or_zero.bf16",
                "reference.bitcopy.row_index_gather_or_zero.bf16.i32.total.v1",
            ),
            ReferenceRowTransportKind::RowWrite => (
                row_write_bf16_contract(),
                ROW_WRITE_BF16_CAPABILITY_ID,
                "provider.reference.row_write.bf16",
                "resource-estimator.reference.row_write.bf16",
                "reference.bitcopy.row_write.bf16.v1",
            ),
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
        if matches!(
            self.kind,
            ReferenceRowTransportKind::MergeLeadingBatchView
                | ReferenceRowTransportKind::SplitLeadingBatchView
        ) {
            return EncodedDeviceOperation::read_only_view(invocation.participants()[0].bindings())
                .map(Some)
                .map_err(|e| provider_failure(identity, e.to_string()));
        }
        encode(self.kind, invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|reason| provider_failure(identity, reason))
    }
}
enum Geometry {
    RowWrite(RowWriteBf16Geometry),
    IndexGather(RowIndexGatherBf16Geometry),
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
        ReferenceRowTransportKind::MergeLeadingBatch
        | ReferenceRowTransportKind::SplitLeadingBatch
        | ReferenceRowTransportKind::MergeLeadingBatchView
        | ReferenceRowTransportKind::SplitLeadingBatchView => {
            if !attributes.is_empty() {
                return Err("leading batch copy has no attributes".into());
            }
            let direction = if matches!(
                kind,
                ReferenceRowTransportKind::MergeLeadingBatch
                    | ReferenceRowTransportKind::MergeLeadingBatchView
            ) {
                LeadingBatchBf16Kind::Merge
            } else {
                LeadingBatchBf16Kind::Split
            };
            let g = LeadingBatchBf16Geometry::new(direction, input, output)
                .map_err(|e| e.to_string())?;
            // After proving the public rank-changing contract, lower the byte
            // transport to the existing identity row-copy implementation.
            // No model tensor is relabeled and no storage view escapes.
            let flat = ResolvedTensorSpec::new(
                vec![g.flat_rows(), g.columns()],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .map_err(|e| e.to_string())?;
            let a = [
                ("batch", SemanticValue::Unsigned(1)),
                ("column_start", SemanticValue::Unsigned(0)),
                ("column_count", SemanticValue::Unsigned(g.columns())),
                (
                    "run_lengths",
                    SemanticValue::Integers(vec![g.flat_rows() as i64]),
                ),
                ("source_starts", SemanticValue::Integers(vec![0])),
                ("source_batch_strides", SemanticValue::Integers(vec![0])),
                ("source_row_strides", SemanticValue::Integers(vec![1])),
            ]
            .into_iter()
            .map(|(k, v)| Ok((AttributeId::new(k).map_err(|e| e.to_string())?, v)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
            RowGatherBf16Geometry::new(&flat, &flat, &a)
                .map(Geometry::Gather)
                .map_err(|e| e.to_string())
        }
        ReferenceRowTransportKind::IndexGather => {
            if !attributes.is_empty() {
                return Err("indexed gather has no attributes".into());
            }
            RowIndexGatherBf16Geometry::new(
                input,
                binding(values, ResolvedValueRole::Input, 1)?.tensor(),
                output,
            )
            .map(Geometry::IndexGather)
            .map_err(|e| e.to_string())
        }
        ReferenceRowTransportKind::RowWrite => {
            row_write_bf16_contract()
                .map_err(|e| e.to_string())?
                .descriptor()
                .validate_attributes(attributes)
                .map_err(|e| e.to_string())?;
            let Some(SemanticValue::Unsigned(start)) = attributes
                .iter()
                .find(|(k, _)| k.as_str() == "row_start")
                .map(|(_, v)| v)
            else {
                return Err("row write requires row_start".into());
            };
            RowWriteBf16Geometry::new(
                input,
                binding(values, ResolvedValueRole::Input, 1)?.tensor(),
                output,
                *start,
            )
            .map(Geometry::RowWrite)
            .map_err(|e| e.to_string())
        }
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
        Geometry::RowWrite(geometry) => {
            ReferenceDeviceCommand::row_write_tensor(ReferenceRowWriteLaunch {
                base: region(ResolvedValueRole::Input, 0)?,
                updates: region(ResolvedValueRole::Input, 1)?,
                output: region(ResolvedValueRole::Output, 0)?,
                geometry,
            })
        }
        Geometry::IndexGather(geometry) => {
            ReferenceDeviceCommand::row_index_gather_tensor(ReferenceRowIndexGatherLaunch {
                source: region(ResolvedValueRole::Input, 0)?,
                indices: contiguous_region(
                    participant,
                    binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
                    ElementType::I32,
                )?,
                output: region(ResolvedValueRole::Output, 0)?,
                geometry,
            })
        }
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
