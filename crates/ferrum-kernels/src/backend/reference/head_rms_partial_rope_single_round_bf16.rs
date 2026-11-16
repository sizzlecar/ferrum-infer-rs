use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceHeadRmsPartialRopeSingleRoundLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_PROFILE: &str =
    "reference.head_rms_partial_rope.single_round_bf16.block128_warp4_fma.libm_sqrtf_rne_div.v1";
struct Provider {
    descriptor: OperationProviderDescriptor,
}
pub fn reference_vnext_head_rms_partial_rope_single_round_bf16_provider(
    runtime: &ReferenceDeviceRuntime,
) -> Result<Box<dyn OperationProvider<ReferenceDeviceRuntime>>, ReferenceDeviceRuntimeError> {
    let contract = head_rms_partial_rope_single_round_bf16_contract().map_err(error)?;
    let cap =
        CapabilityId::new(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_CAPABILITY_ID).map_err(error)?;
    if !runtime.descriptor().capabilities.contains(&cap) {
        return Err(error("runtime lacks single-round RMS/RoPE capability"));
    }
    let fingerprint = implementation_fingerprint(&[
        REFERENCE_HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_PROFILE.as_bytes(),
        include_bytes!("head_rms_partial_rope_single_round_bf16.rs"),
        include_bytes!("runtime/head_rms_partial_rope_single_round_bf16.rs"),
    ]);
    let estimator = "resource-estimator.reference.head_rms_partial_rope.single_round_bf16";
    let descriptor = OperationProviderDescriptor::new(
        ProviderId::new("provider.reference.head_rms_partial_rope.single_round_bf16")
            .map_err(error)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint().map_err(error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        runtime.descriptor().id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([WeightFormatId::new(
            super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID,
        )
        .map_err(error)?]),
        BTreeSet::new(),
        contiguous_bindings(4),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map_err(error)?
    .with_fixed_tensor_work(true);
    Ok(Box::new(Provider { descriptor }))
}
impl OperationResourceEstimator for Provider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        r: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if r.operation().id != *self.descriptor.operation_id()
            || r.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid("another operation reached estimator"));
        }
        geometry(r.values(), r.attributes()).map_err(invalid)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            16,
            None,
            None,
        ))
    }
}
impl OperationProvider<ReferenceDeviceRuntime> for Provider {
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
            "requires fixed tensor invocation".into(),
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
fn encode(
    i: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [p] = i.participants() else {
        return Err("requires one fixed participant".into());
    };
    if i.operation().id.as_str() != HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID
        || !matches!(p.work(), NodeWorkContract::Fixed)
        || !i.resource_work_shape().resource_work().is_tensor()
    {
        return Err("requires selected fixed tensor contract".into());
    }
    let geometry = geometry(p.bindings(), p.attributes())?;
    let region =
        |role, index| contiguous_region(p, binding(p.bindings(), role, index)?, ElementType::Bf16);
    ReferenceDeviceCommand::head_rms_partial_rope_single_round_tensor(
        ReferenceHeadRmsPartialRopeSingleRoundLaunch {
            input: region(ResolvedValueRole::Input, 0)?,
            gamma: region(ResolvedValueRole::Input, 1)?,
            cosine: region(ResolvedValueRole::Input, 2)?,
            sine: region(ResolvedValueRole::Input, 3)?,
            output: region(ResolvedValueRole::Output, 0)?,
            geometry,
        },
    )
    .map_err(|e| e.to_string())
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<HeadRmsPartialRopeSingleRoundBf16Geometry, String> {
    if values.len() != 5
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires numerical dense operands".into());
    }
    HeadRmsPartialRopeSingleRoundBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Input, 1)?.tensor(),
        binding(values, ResolvedValueRole::Input, 2)?.tensor(),
        binding(values, ResolvedValueRole::Input, 3)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.single_round_rms_rope.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded failure")
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn error(e: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(e.to_string())
}
