//! Image gamma, fused-QKV spatial attention, and bounded spatial gather providers.
//!
//! Attention uses RN32 products, a power-of-two tree padded with positive zero,
//! scale, ascending-key online softmax with libm 0.2.15 expf, separate RN32
//! multiply/add updates, and RN32 reciprocal/multiply before the final BF16.
//! It has an independent profile: neither SDPA nor CUDA exp bitwise parity is
//! implied. Only bounded 8 KiB stack accumulator state is used, never S² scores.
use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceChannelL2NormLaunch, ReferenceDeviceBuffer, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, ReferenceDeviceRuntimeError, ReferenceSpatialDecoderGeometry,
    ReferenceSpatialDecoderLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};
pub const REFERENCE_CHANNEL_L2_NORM_IMAGE_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.channel_l2_norm_image.bf16.pairwise_f32.staged.v1";
pub const REFERENCE_SPATIAL_ATTENTION_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.spatial_attention.bf16.libm-0.2.15.online_padded_pairwise.v1";
pub const REFERENCE_SPATIAL_UPSAMPLE_GATHER_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.bitcopy.spatial_upsample_gather.bf16.v1";
#[derive(Clone, Copy)]
pub(super) enum DecoderKind {
    Norm,
    Attention,
    Upsample,
}
impl DecoderKind {
    fn suffix(self) -> &'static str {
        match self {
            Self::Norm => "channel_l2_norm_image.bf16",
            Self::Attention => "spatial_attention.bf16",
            Self::Upsample => "spatial_upsample_gather.bf16",
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::Norm => CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
            Self::Attention => SPATIAL_ATTENTION_BF16_OPERATION_ID,
            Self::Upsample => SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::Norm => CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID,
            Self::Attention => SPATIAL_ATTENTION_BF16_CAPABILITY_ID,
            Self::Upsample => SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Norm => REFERENCE_CHANNEL_L2_NORM_IMAGE_BF16_NUMERICAL_PROFILE_ID,
            Self::Attention => REFERENCE_SPATIAL_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
            Self::Upsample => REFERENCE_SPATIAL_UPSAMPLE_GATHER_BF16_NUMERICAL_PROFILE_ID,
        }
    }
    fn inputs(self) -> u32 {
        if matches!(self, Self::Norm) {
            2
        } else {
            1
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Norm => channel_l2_norm_image_bf16_contract(),
            Self::Attention => spatial_attention_bf16_contract(),
            Self::Upsample => spatial_upsample_gather_bf16_contract(),
        }
    }
}
pub(super) struct ReferenceSpatialDecoderProvider {
    kind: DecoderKind,
    descriptor: OperationProviderDescriptor,
}
impl ReferenceSpatialDecoderProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: DecoderKind,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = kind.contract().map_err(error)?;
        let capability = CapabilityId::new(kind.capability()).map_err(error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "missing spatial decoder capability",
            ));
        }
        let provider = format!("provider.reference.{}", kind.suffix());
        let estimator = format!("resource-estimator.reference.{}", kind.suffix());
        let numerical = super::transcendental::numerical_implementation_fingerprint();
        let implementation = implementation_fingerprint(&[
            include_str!("spatial_decoder.rs").as_bytes(),
            include_str!("runtime/spatial_decoder.rs").as_bytes(),
            include_str!("runtime/spatial_bf16.rs").as_bytes(),
            kind.profile().as_bytes(),
            numerical.as_bytes(),
        ]);
        let estimator_fp =
            implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
        let formats = if matches!(kind, DecoderKind::Norm) {
            BTreeSet::from([
                WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID).map_err(error)?
            ])
        } else {
            BTreeSet::new()
        };
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(provider).map_err(error)?,
            contract.descriptor().id.clone(),
            contract.descriptor().fingerprint().map_err(error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            formats,
            BTreeSet::new(),
            contiguous_bindings(kind.inputs()),
            estimator,
            ContractVersion::new(1, 0),
            estimator_fp,
        )
        .map_err(error)?
        .with_fixed_tensor_work(true);
        Ok(Self { kind, descriptor })
    }
}
impl OperationResourceEstimator for ReferenceSpatialDecoderProvider {
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
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid)?;
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceSpatialDecoderProvider {
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
            "spatial decoder requires fixed tensor work".into(),
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
            .map_err(|e| failure(identity, e))
    }
}
enum Geometry {
    Norm(ChannelL2NormImageBf16Geometry),
    Attention(SpatialAttentionBf16Geometry),
    Upsample(SpatialUpsampleGatherBf16Geometry),
}
fn geometry(
    kind: DecoderKind,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    if values.len() != kind.inputs() as usize + 1 {
        return Err("operand count differs".into());
    }
    for v in values {
        if v.weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("actual dense weight storage required".into());
        }
        let bytes = v
            .tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?;
        if bytes > isize::MAX as u64 {
            return Err("tensor exceeds address space".into());
        }
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    match kind {
        DecoderKind::Norm => ChannelL2NormImageBf16Geometry::new(
            x,
            binding(values, ResolvedValueRole::Input, 1)?.tensor(),
            y,
            attrs,
        )
        .map(Geometry::Norm),
        DecoderKind::Attention => {
            SpatialAttentionBf16Geometry::new(x, y, attrs).map(Geometry::Attention)
        }
        DecoderKind::Upsample => {
            SpatialUpsampleGatherBf16Geometry::new(x, y, attrs).map(Geometry::Upsample)
        }
    }
    .map_err(|e| e.to_string())
}
fn encode(
    kind: DecoderKind,
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("requires one participant".into());
    };
    if invocation.operation().id.as_str() != kind.operation()
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("requires its own fixed tensor contract".into());
    }
    let shape = geometry(kind, participant.bindings(), participant.attributes())?;
    let region = |role, index| {
        contiguous_region(
            participant,
            binding(participant.bindings(), role, index)?,
            ElementType::Bf16,
        )
    };
    let input = region(ResolvedValueRole::Input, 0)?;
    let output = region(ResolvedValueRole::Output, 0)?;
    match shape {
        Geometry::Norm(g) => {
            ReferenceDeviceCommand::channel_l2_norm_tensor(ReferenceChannelL2NormLaunch {
                input,
                gamma: region(ResolvedValueRole::Input, 1)?,
                output,
                shape: g.arithmetic_geometry(),
            })
        }
        Geometry::Attention(g) => {
            ReferenceDeviceCommand::spatial_decoder_tensor(ReferenceSpatialDecoderLaunch {
                input,
                output,
                shape: ReferenceSpatialDecoderGeometry::Attention(g),
            })
        }
        Geometry::Upsample(g) => {
            ReferenceDeviceCommand::spatial_decoder_tensor(ReferenceSpatialDecoderLaunch {
                input,
                output,
                shape: ReferenceSpatialDecoderGeometry::Upsample(g),
            })
        }
    }
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
        "reference.spatial_decoder.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded static provider failure metadata")
}
#[cfg(test)]
mod tests;
