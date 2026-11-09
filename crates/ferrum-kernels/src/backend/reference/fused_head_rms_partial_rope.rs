//! Managed implementation of the independent fused normalization/rotation ABI.
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferenceFusedHeadRmsPartialRopeLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.fused_head_rms_partial_rope.bf16.warp32_precise_single_norm.v1";
pub(super) struct ReferenceFusedHeadRmsPartialRopeProvider {
    descriptor: OperationProviderDescriptor,
}
impl ReferenceFusedHeadRmsPartialRopeProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = fused_head_rms_partial_rope_bf16_contract().map_err(error)?;
        let capability =
            CapabilityId::new(FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_CAPABILITY_ID).map_err(error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(error("reference runtime lacks fused RMS/RoPE capability"));
        }
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("fused_head_rms_partial_rope.rs"),
            include_bytes!("runtime/fused_head_rms_partial_rope.rs"),
            REFERENCE_FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_NUMERICAL_PROFILE_ID.as_bytes(),
        ]);
        let estimator = "resource-estimator.reference.fused_head_rms_partial_rope.bf16";
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new("provider.reference.fused_head_rms_partial_rope.bf16")
                .map_err(error)?,
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint().map_err(error)?,
            fingerprint.clone(),
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
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
        Ok(Self { descriptor })
    }
}
impl OperationResourceEstimator for ReferenceFusedHeadRmsPartialRopeProvider {
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
            return Err(invalid(
                "another operation reached fused RMS/RoPE estimator",
            ));
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceFusedHeadRmsPartialRopeProvider {
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
        Err(failure(
            invocation.participants()[0].identity().clone(),
            "requires fixed tensor invocation".into(),
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
            .map_err(|message| failure(identity, message))
    }
}
fn encode(
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("requires one tensor participant".into());
    };
    if invocation.operation().id.as_str() != FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("requires fused head RMS/RoPE fixed tensor work".into());
    }
    let geometry = geometry(participant.bindings(), participant.attributes())?;
    let input = |ordinal| {
        contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
            ElementType::Bf16,
        )
    };
    ReferenceDeviceCommand::fused_head_rms_partial_rope_tensor(
        ReferenceFusedHeadRmsPartialRopeLaunch {
            input: input(0)?,
            gamma: input(1)?,
            cosine: input(2)?,
            sine: input(3)?,
            output: contiguous_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::Bf16,
            )?,
            geometry,
        },
    )
    .map_err(|e| e.to_string())
}
pub(super) fn geometry(
    bindings: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<FusedHeadRmsPartialRopeBf16Geometry, String> {
    if bindings.len() != 5 || attrs.len() != 3 {
        return Err(
            "fused RMS/RoPE requires four inputs, one output and head_dim/rotary_dim/epsilon"
                .into(),
        );
    }
    let get = |name| {
        attrs
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, value)| value)
    };
    let unsigned = |name| match get(name) {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("requires unsigned {name}")),
    };
    let Some(SemanticValue::Rational(epsilon)) = get("epsilon") else {
        return Err("requires rational epsilon".into());
    };
    let g = FusedHeadRmsPartialRopeBf16Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 3)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        unsigned("head_dim")?,
        unsigned("rotary_dim")?,
        *epsilon,
    )
    .map_err(|e| e.to_string())?;
    for binding in bindings {
        if binding.weight().is_some_and(|weight| {
            !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
        }) {
            return Err(
                "fused RMS/RoPE cannot consume packed or quantized physical weights".into(),
            );
        }
        if binding
            .tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?
            > isize::MAX as u64
        {
            return Err("fused RMS/RoPE operand exceeds address space".into());
        }
    }
    Ok(g)
}
fn error(reason: impl std::fmt::Display) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(reason.to_string())
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
        "reference.fused_head_rms_partial_rope.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded provider failure")
}
