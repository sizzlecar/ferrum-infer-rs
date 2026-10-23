//! Direct conceptual grouping, with no materialized padded or shuffled tensor.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceSpatialGroupMeanLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_SPATIAL_GROUP_MEAN_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.spatial_group_mean.bf16.ascending_f32.v1";
pub(super) struct ReferenceSpatialGroupMeanProvider {
    descriptor: OperationProviderDescriptor,
}
impl ReferenceSpatialGroupMeanProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = spatial_group_mean_bf16_contract().map_err(error)?;
        let capability = CapabilityId::new(SPATIAL_GROUP_MEAN_BF16_CAPABILITY_ID).map_err(error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "missing spatial group mean capability",
            ));
        }
        let implementation = implementation_fingerprint(&[
            include_str!("spatial_group_mean.rs").as_bytes(),
            include_str!("runtime/spatial_group_mean.rs").as_bytes(),
            REFERENCE_SPATIAL_GROUP_MEAN_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
        ]);
        let estimator = "resource-estimator.reference.spatial_group_mean.bf16";
        let estimator_fingerprint =
            implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new("provider.reference.spatial_group_mean.bf16").map_err(error)?,
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint().map_err(error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::new(),
            BTreeSet::new(),
            contiguous_bindings(1),
            estimator,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(error)?
        .with_fixed_tensor_work(true);
        Ok(Self { descriptor })
    }
}
impl OperationResourceEstimator for ReferenceSpatialGroupMeanProvider {
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
            return Err(invalid("estimator operation differs"));
        }
        geometry(request.values(), request.attributes()).map_err(invalid)?;
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceSpatialGroupMeanProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        Err(failure(
            i.participants()[0].identity().clone(),
            "requires fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| failure(identity, e))
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<SpatialGroupMeanBf16Geometry, String> {
    if values.len() != 2 {
        return Err("requires one input and one output".into());
    }
    for value in values {
        if value
            .tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?
            > isize::MAX as u64
        {
            return Err("tensor exceeds address space".into());
        }
    }
    SpatialGroupMeanBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
fn encode(
    i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [p] = i.participants() else {
        return Err("requires one participant".into());
    };
    if i.operation().id.as_str() != SPATIAL_GROUP_MEAN_BF16_OPERATION_ID
        || !matches!(p.work(), NodeWorkContract::Fixed)
        || !i.resource_work_shape().resource_work().is_tensor()
    {
        return Err("requires its own fixed tensor contract".into());
    }
    let shape = geometry(p.bindings(), p.attributes())?;
    let region = |role| contiguous_region(p, binding(p.bindings(), role, 0)?, ElementType::Bf16);
    ReferenceDeviceCommand::spatial_group_mean_tensor(ReferenceSpatialGroupMeanLaunch {
        input: region(ResolvedValueRole::Input)?,
        output: region(ResolvedValueRole::Output)?,
        shape,
    })
    .map_err(|e| e.to_string())
}
fn error(e: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.spatial_group_mean.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded static provider failure metadata")
}
