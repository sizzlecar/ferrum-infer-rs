//! Independent fixed-tensor Conv2D and channel-L2 providers.
use super::composition::REFERENCE_DENSE_SAFETENSORS_FORMAT_ID;
use super::dense_linear::{
    binding, contiguous_bindings, contiguous_region, implementation_fingerprint,
};
use super::runtime::{
    ReferenceActivationShape, ReferenceChannelL2NormLaunch, ReferenceConv2dLaunch,
    ReferenceDeviceBuffer, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    ReferenceDeviceRuntimeError, ReferencePlainResidualAddLaunch, ReferenceUnaryKind,
    ReferenceUnaryLaunch,
};
use ferrum_runtime::*;
use std::collections::{BTreeMap, BTreeSet};

pub const REFERENCE_CONV2D_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.conv2d.bf16.serial_f32.staged_bias.v1";
pub const REFERENCE_CHANNEL_L2_NORM_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.channel_l2_norm.bf16.pairwise_f32.staged.v1";
pub const REFERENCE_SILU_NCHW_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.silu.nchw.bf16.libm-0.2.15.v1";
pub const REFERENCE_PLAIN_RESIDUAL_ADD_NCHW_BF16_NUMERICAL_PROFILE_ID: &str =
    "reference.plain_residual_add.nchw.bf16.f32_rne.v1";
#[derive(Debug, Clone, Copy)]
pub(super) enum SpatialKind {
    Conv2d,
    ChannelL2Norm,
    SiluNchw,
    ResidualAddNchw,
}
impl SpatialKind {
    fn operation(self) -> &'static str {
        match self {
            Self::Conv2d => CONV2D_BF16_OPERATION_ID,
            Self::ChannelL2Norm => CHANNEL_L2_NORM_BF16_OPERATION_ID,
            Self::SiluNchw => SILU_NCHW_BF16_OPERATION_ID,
            Self::ResidualAddNchw => PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::Conv2d => CONV2D_BF16_CAPABILITY_ID,
            Self::ChannelL2Norm => CHANNEL_L2_NORM_BF16_CAPABILITY_ID,
            Self::SiluNchw => SILU_NCHW_BF16_CAPABILITY_ID,
            Self::ResidualAddNchw => PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID,
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::Conv2d => "provider.reference.conv2d.bf16",
            Self::ChannelL2Norm => "provider.reference.channel_l2_norm.bf16",
            Self::SiluNchw => "provider.reference.silu.nchw.bf16",
            Self::ResidualAddNchw => "provider.reference.plain_residual_add.nchw.bf16",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::Conv2d => "resource-estimator.reference.conv2d.bf16",
            Self::ChannelL2Norm => "resource-estimator.reference.channel_l2_norm.bf16",
            Self::SiluNchw => "resource-estimator.reference.silu.nchw.bf16",
            Self::ResidualAddNchw => "resource-estimator.reference.plain_residual_add.nchw.bf16",
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Conv2d => REFERENCE_CONV2D_BF16_NUMERICAL_PROFILE_ID,
            Self::ChannelL2Norm => REFERENCE_CHANNEL_L2_NORM_BF16_NUMERICAL_PROFILE_ID,
            Self::SiluNchw => REFERENCE_SILU_NCHW_BF16_NUMERICAL_PROFILE_ID,
            Self::ResidualAddNchw => REFERENCE_PLAIN_RESIDUAL_ADD_NCHW_BF16_NUMERICAL_PROFILE_ID,
        }
    }
    fn input_count(self) -> u32 {
        match self {
            Self::Conv2d => 3,
            Self::ChannelL2Norm | Self::ResidualAddNchw => 2,
            Self::SiluNchw => 1,
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Conv2d => conv2d_bf16_contract(),
            Self::ChannelL2Norm => channel_l2_norm_bf16_contract(),
            Self::SiluNchw => silu_nchw_bf16_contract(),
            Self::ResidualAddNchw => plain_residual_add_nchw_bf16_contract(),
        }
    }
}
pub(super) struct ReferenceSpatialProvider {
    kind: SpatialKind,
    descriptor: OperationProviderDescriptor,
}
impl ReferenceSpatialProvider {
    pub(super) fn new(
        runtime: &ReferenceDeviceRuntime,
        kind: SpatialKind,
    ) -> Result<Self, ReferenceDeviceRuntimeError> {
        let contract = kind.contract().map_err(contract_error)?;
        let capability = CapabilityId::new(kind.capability()).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "missing reference spatial capability",
            ));
        }
        let numerical = super::transcendental::numerical_implementation_fingerprint();
        let implementation = implementation_fingerprint(&[
            include_str!("spatial_bf16.rs").as_bytes(),
            include_str!("runtime/spatial_bf16.rs").as_bytes(),
            include_str!("runtime/transcendental.rs").as_bytes(),
            include_str!("runtime/plain_residual_add.rs").as_bytes(),
            numerical.as_bytes(),
            kind.profile().as_bytes(),
            kind.provider().as_bytes(),
        ]);
        let estimator =
            implementation_fingerprint(&[kind.estimator().as_bytes(), implementation.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(kind.provider()).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            implementation,
            ProviderExecutionSemantics::bitwise_eager_only(),
            ContractVersion::new(1, 0),
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?]),
            BTreeSet::new(),
            contiguous_bindings(kind.input_count()),
            kind.estimator(),
            ContractVersion::new(1, 0),
            estimator,
        )
        .map_err(contract_error)?
        .with_fixed_tensor_work(true);
        Ok(Self { kind, descriptor })
    }
}
impl OperationResourceEstimator for ReferenceSpatialProvider {
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
            return Err(invalid("spatial estimator received a different operation"));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid)?;
        // Reference directly indexes conceptual zero padding; it never allocates
        // im2col/padded images or per-spatial-point reduction vectors.
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
impl OperationProvider<ReferenceDeviceRuntime> for ReferenceSpatialProvider {
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
            "spatial BF16 operation requires fixed tensor work".into(),
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
    Conv2d(Conv2dBf16Geometry),
    ChannelL2Norm(ChannelL2NormBf16Geometry),
    SiluNchw(Bf16NchwGeometry),
    ResidualAddNchw(Bf16NchwGeometry),
}
fn geometry(
    kind: SpatialKind,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Geometry, String> {
    if values.len() != kind.input_count() as usize + 1 {
        return Err("spatial operand count differs".into());
    }
    for v in values {
        if v.weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("spatial provider requires actual dense weight storage".into());
        }
        let bytes = v
            .tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?;
        usize::try_from(bytes).map_err(|_| "spatial tensor exceeds address width")?;
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    match kind {
        SpatialKind::Conv2d => {
            let g = Conv2dBf16Geometry::new(
                x,
                binding(values, ResolvedValueRole::Input, 1)?.tensor(),
                binding(values, ResolvedValueRole::Input, 2)?.tensor(),
                y,
                attrs,
            )
            .map_err(|e| e.to_string())?;
            usize::try_from(g.padded_input_bytes())
                .map_err(|_| "padded spatial coordinates exceed address width")?;
            for v in g
                .padding()
                .into_iter()
                .chain(g.stride())
                .chain(g.dilation())
            {
                usize::try_from(v).map_err(|_| "convolution coordinate exceeds address width")?;
            }
            Ok(Geometry::Conv2d(g))
        }
        SpatialKind::ChannelL2Norm => ChannelL2NormBf16Geometry::new(
            x,
            binding(values, ResolvedValueRole::Input, 1)?.tensor(),
            y,
            attrs,
        )
        .map(Geometry::ChannelL2Norm)
        .map_err(|e| e.to_string()),
        SpatialKind::SiluNchw | SpatialKind::ResidualAddNchw => {
            if !attrs.is_empty() {
                return Err("NCHW scalar operations have no attributes".into());
            }
            let g = Bf16NchwGeometry::new(x, y).map_err(|e| e.to_string())?;
            let per_element = if matches!(kind, SpatialKind::SiluNchw) {
                4
            } else {
                1
            };
            flat_shape(g)?
                .logical_work(per_element)
                .map_err(|e| e.to_string())?;
            if matches!(kind, SpatialKind::ResidualAddNchw) {
                if binding(values, ResolvedValueRole::Input, 1)?.tensor() != x {
                    return Err("NCHW residual operands differ in actual shape/dtype/layout".into());
                }
                Ok(Geometry::ResidualAddNchw(g))
            } else {
                Ok(Geometry::SiluNchw(g))
            }
        }
    }
}
fn encode(
    kind: SpatialKind,
    invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
) -> Result<ReferenceDeviceCommand, String> {
    let [participant] = invocation.participants() else {
        return Err("spatial operation needs one participant".into());
    };
    if invocation.operation().id.as_str() != kind.operation()
        || !matches!(participant.work(), NodeWorkContract::Fixed)
        || !invocation.resource_work_shape().resource_work().is_tensor()
    {
        return Err("spatial operation requires its fixed tensor contract".into());
    }
    let checked = geometry(kind, participant.bindings(), participant.attributes())?;
    let region = |role, index| {
        contiguous_region(
            participant,
            binding(participant.bindings(), role, index)?,
            ElementType::Bf16,
        )
    };
    let x = region(ResolvedValueRole::Input, 0)?;
    let y = region(ResolvedValueRole::Output, 0)?;
    match checked {
        Geometry::Conv2d(shape) => ReferenceDeviceCommand::conv2d_tensor(ReferenceConv2dLaunch {
            input: x,
            weight: region(ResolvedValueRole::Input, 1)?,
            bias: region(ResolvedValueRole::Input, 2)?,
            output: y,
            shape,
        }),
        Geometry::ChannelL2Norm(shape) => {
            ReferenceDeviceCommand::channel_l2_norm_tensor(ReferenceChannelL2NormLaunch {
                input: x,
                gamma: region(ResolvedValueRole::Input, 1)?,
                output: y,
                shape,
            })
        }
        Geometry::SiluNchw(g) => {
            ReferenceDeviceCommand::unary_activation_tensor(ReferenceUnaryLaunch {
                input: x,
                output: y,
                kind: ReferenceUnaryKind::Silu,
                shape: flat_shape(g)?,
            })
        }
        Geometry::ResidualAddNchw(g) => {
            ReferenceDeviceCommand::plain_residual_add_tensor(ReferencePlainResidualAddLaunch {
                input: x,
                update: region(ResolvedValueRole::Input, 1)?,
                output: y,
                shape: flat_shape(g)?,
            })
        }
    }
    .map_err(|e| e.to_string())
}
fn flat_shape(g: Bf16NchwGeometry) -> Result<ReferenceActivationShape, String> {
    // A storage-loop tiling of verified contiguous NCHW, not an input ABI view.
    Ok(ReferenceActivationShape {
        rows: usize::try_from(g.element_count() / g.width())
            .map_err(|_| "NCHW outer extent exceeds usize")?,
        width: usize::try_from(g.width()).map_err(|_| "NCHW width exceeds usize")?,
    })
}
fn failure(identity: ExecutionIdentityEnvelope, message: String) -> OperationFailure {
    OperationFailure::new(
        identity,
        ProfilePhase::Forward,
        "reference.spatial_bf16.encode",
        message.chars().take(2048).collect::<String>(),
        false,
    )
    .expect("bounded static provider failure metadata")
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}
fn contract_error(error: VNextError) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(error.to_string())
}

#[cfg(test)]
mod tests;
