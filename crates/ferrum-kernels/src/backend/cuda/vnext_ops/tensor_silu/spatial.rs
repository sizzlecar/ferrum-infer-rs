//! Real NCHW contracts with admitted storage. Unary/add arithmetic shares the
//! flat native primitives without pretending that the logical tensor is rank 2.

use super::*;
use ferrum_runtime::{
    channel_l2_norm_bf16_contract, channel_l2_norm_image_bf16_contract,
    plain_residual_add_nchw_bf16_contract, silu_nchw_bf16_contract, Bf16NchwGeometry,
    ChannelL2NormBf16Geometry, ChannelL2NormImageBf16Geometry, PhysicalWeightLayout,
    WeightFormatId, CHANNEL_L2_NORM_BF16_CAPABILITY_ID, CHANNEL_L2_NORM_BF16_OPERATION_ID,
    CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID, CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
    PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID, PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
    SILU_NCHW_BF16_CAPABILITY_ID, SILU_NCHW_BF16_OPERATION_ID,
};

pub const L2_PROFILE_ID: &str = "cuda.ieee.channel_l2_norm.bf16.padded_pairwise.v1";
pub const L2_IMAGE_PROFILE_ID: &str = "cuda.ieee.channel_l2_norm_image.bf16.padded_pairwise.v1";
pub const SILU_PROFILE_ID: &str = "cuda.libdevice.precise.silu.nchw.bf16.v1";
pub const ADD_PROFILE_ID: &str = "cuda.ieee.plain_residual_add.nchw.bf16.v1";
pub const NUMERICAL_IDENTITY:&str=concat!(
    "NCHW;C=1..4096;L2=zero-padded-next-power-of-two-adjacent-FP32-square-sum-tree;",
    "RN32-sqrt;NaN-propagating-clamp-min-epsilon;B(div_rn)->B(scale_mul)->B(gamma_mul)->B(add_positive_zero);",
    "scale-and-epsilon=(rational-as-f64)-as-f32;gamma-actual[C,1,1,1]-or-independent-image-contract[C,1,1];",
    "silu=existing-precise-FP32-expf-div-profile;add=B(RN32(x+u));",
    "logical-work-L2=6/elements;Silu=4/elements;Add=1/elements;",
    "spatial>=32:one-thread-per-pixel;coalesced-spatial-lanes;12-named-FP32-carry-slots;same-adjacent-tree;",
    "spatial<32:original-CTA-per-pixel;shared=4*nextpow2(C);otherwise-shared=0;",
    "no-global-scratch;no-ATen-reduction-or-deployment-bitwise-claim"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpatialKind {
    ChannelL2,
    ChannelL2Image,
    Silu,
    ResidualAdd,
}
impl SpatialKind {
    fn inputs(self) -> u32 {
        if self == Self::Silu {
            1
        } else {
            2
        }
    }
    fn id(self) -> &'static str {
        match self {
            Self::ChannelL2 => CHANNEL_L2_NORM_BF16_OPERATION_ID,
            Self::ChannelL2Image => CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
            Self::Silu => SILU_NCHW_BF16_OPERATION_ID,
            Self::ResidualAdd => PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::ChannelL2 => CHANNEL_L2_NORM_BF16_CAPABILITY_ID,
            Self::ChannelL2Image => CHANNEL_L2_NORM_IMAGE_BF16_CAPABILITY_ID,
            Self::Silu => SILU_NCHW_BF16_CAPABILITY_ID,
            Self::ResidualAdd => PLAIN_RESIDUAL_ADD_NCHW_BF16_CAPABILITY_ID,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::ChannelL2 => L2_PROFILE_ID,
            Self::ChannelL2Image => L2_IMAGE_PROFILE_ID,
            Self::Silu => SILU_PROFILE_ID,
            Self::ResidualAdd => ADD_PROFILE_ID,
        }
    }
    fn native_operation(self) -> &'static str {
        match self {
            Self::ChannelL2 => "vnext_channel_l2_norm_bf16",
            Self::ChannelL2Image => "vnext_channel_l2_norm_image_bf16",
            Self::Silu => "vnext_silu_nchw_bf16",
            Self::ResidualAdd => "vnext_plain_residual_add_nchw_bf16",
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::ChannelL2 => "provider.cuda.channel_l2_norm.bf16",
            Self::ChannelL2Image => "provider.cuda.channel_l2_norm_image.bf16",
            Self::Silu => "provider.cuda.silu.nchw.bf16",
            Self::ResidualAdd => "provider.cuda.plain_residual_add.nchw.bf16",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::ChannelL2 => "resource-estimator.cuda.channel_l2_norm.bf16",
            Self::ChannelL2Image => "resource-estimator.cuda.channel_l2_norm_image.bf16",
            Self::Silu => "resource-estimator.cuda.silu.nchw.bf16",
            Self::ResidualAdd => "resource-estimator.cuda.plain_residual_add.nchw.bf16",
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::ChannelL2 => channel_l2_norm_bf16_contract(),
            Self::ChannelL2Image => channel_l2_norm_image_bf16_contract(),
            Self::Silu => silu_nchw_bf16_contract(),
            Self::ResidualAdd => plain_residual_add_nchw_bf16_contract(),
        }
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: SpatialKind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks selected NCHW capability",
        ));
    }
    let c = kind.contract().map_err(contract_error)?;
    let fp = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        kind.profile().as_bytes(),
        kind.provider().as_bytes(),
    ]);
    let estimator = implementation_fingerprint(&[kind.estimator().as_bytes(), fp.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(kind.provider()).map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fp,
        ProviderExecutionSemantics::bitwise_eager_only(),
        c.descriptor().version,
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.inputs()),
        kind.estimator(),
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    kind: SpatialKind,
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(SpatialProvider {
        kind,
        descriptor,
        kernels: Arc::clone(kernels),
    })
}
struct SpatialProvider {
    kind: SpatialKind,
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}
impl OperationResourceEstimator for SpatialProvider {
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
            return Err(invalid_plan(
                "CUDA NCHW estimator received another operation",
            ));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            r.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        ))
    }
}
impl OperationProvider<CudaDeviceRuntime> for SpatialProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        i: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            i.participants()[0].identity().clone(),
            "cuda.nchw.encode",
            "CUDA NCHW operations require fixed tensor work".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let id = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(id, "cuda.nchw.encode", e))
    }
}
impl SpatialProvider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("CUDA NCHW requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA NCHW requires its fixed contract".into());
        }
        let s = geometry(self.kind, p.bindings(), p.attributes())?;
        let kind = self.kind;
        let mut regions = Vec::with_capacity(kind.inputs() as usize + 1);
        for ordinal in 0..kind.inputs() {
            let bytes = if ordinal == 1
                && matches!(kind, SpatialKind::ChannelL2 | SpatialKind::ChannelL2Image)
            {
                s.channels * 2
            } else {
                s.bytes
            };
            regions.push(retained_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, ordinal)?,
                bytes,
            )?);
        }
        let output = retained_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            s.bytes,
        )?;
        for r in &regions {
            nonoverlapping_ranges(
                r.device_ptr(),
                r.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(kind.native_operation(), regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let out = regions[kind.inputs() as usize].device_ptr();
            if matches!(kind, SpatialKind::ChannelL2 | SpatialKind::ChannelL2Image) {
                let gamma = regions[1].device_ptr();
                let mut launch = stream.launch_builder(kernels.spatial_function(2));
                launch
                    .arg(&x)
                    .arg(&gamma)
                    .arg(&out)
                    .arg(&s.positions)
                    .arg(&s.channels)
                    .arg(&s.spatial)
                    .arg(&s.tree_width)
                    .arg(&s.epsilon)
                    .arg(&s.scale);
                unsafe { launch.launch(s.channel_norm_launch()) }
                    .map(|_| ())
                    .map_err(|e| CudaDeviceRuntimeError::driver("BF16 channel L2 normalization", e))
            } else {
                let function = if kind == SpatialKind::Silu {
                    kernels.function(Kind::Silu)
                } else {
                    kernels.function(Kind::PlainResidualAdd)
                };
                let second = regions[1].device_ptr();
                let mut launch = stream.launch_builder(function);
                launch.arg(&x);
                if kind == SpatialKind::ResidualAdd {
                    launch.arg(&second);
                }
                launch.arg(&out).arg(&s.elements);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (
                            s.elements.div_ceil(THREADS as u64).min(MAX_BLOCKS) as u32,
                            1,
                            1,
                        ),
                        block_dim: (THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("BF16 NCHW pointwise", e))
            }
        })
        .and_then(|c| c.with_tensor_work(s.elements, s.operations, 1, 0))
        .map_err(|e| e.to_string())
    }
}
#[derive(Clone, Copy, Debug)]
struct Shape {
    channels: u64,
    spatial: u64,
    positions: u64,
    elements: u64,
    bytes: u64,
    operations: u64,
    tree_width: u32,
    epsilon: f32,
    scale: f32,
}
impl Shape {
    fn channel_norm_launch(&self) -> LaunchConfig {
        let spatial_lanes = self.spatial >= 32;
        LaunchConfig {
            grid_dim: (
                if spatial_lanes {
                    self.positions.div_ceil(u64::from(THREADS))
                } else {
                    self.positions
                }
                .min(MAX_BLOCKS) as u32,
                1,
                1,
            ),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: if spatial_lanes {
                0
            } else {
                self.tree_width * 4
            },
        }
    }
}
fn geometry(
    kind: SpatialKind,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if values.len() != kind.inputs() as usize + 1 {
        return Err("CUDA NCHW operand count mismatch".into());
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?;
    let y = binding(values, ResolvedValueRole::Output, 0)?;
    for v in values {
        if v.weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("CUDA NCHW requires dense physical weights".into());
        }
    }
    let g = Bf16NchwGeometry::new(x.tensor(), y.tensor()).map_err(|e| e.to_string())?;
    let elements = g.element_count();
    let bytes = elements
        .checked_mul(2)
        .filter(|&n| n <= isize::MAX as u64)
        .ok_or("CUDA NCHW byte extent exceeds address space")?;
    let spatial = g.height() * g.width();
    let positions = g.batch() * spatial;
    let (epsilon, scale, tree_width, work) =
        if matches!(kind, SpatialKind::ChannelL2 | SpatialKind::ChannelL2Image) {
            let gamma = binding(values, ResolvedValueRole::Input, 1)?;
            let n = if kind == SpatialKind::ChannelL2Image {
                ChannelL2NormImageBf16Geometry::new(x.tensor(), gamma.tensor(), y.tensor(), attrs)
                    .map_err(|e| e.to_string())?
                    .arithmetic_geometry()
            } else {
                ChannelL2NormBf16Geometry::new(x.tensor(), gamma.tensor(), y.tensor(), attrs)
                    .map_err(|e| e.to_string())?
            };
            if g.channels() > 4096 {
                return Err("CUDA channel L2 normalization supports C<=4096".into());
            }
            (
                n.effective_epsilon(),
                n.effective_scale(),
                g.channels().next_power_of_two() as u32,
                6,
            )
        } else {
            if !attrs.is_empty() {
                return Err("CUDA NCHW pointwise accepts no attributes".into());
            }
            if kind == SpatialKind::ResidualAdd
                && binding(values, ResolvedValueRole::Input, 1)?.tensor() != x.tensor()
            {
                return Err(
                    "CUDA NCHW residual requires an exact second input shape/dtype/layout".into(),
                );
            }
            (0.0, 0.0, 0, if kind == SpatialKind::Silu { 4 } else { 1 })
        };
    let operations = elements
        .checked_mul(work)
        .ok_or("CUDA NCHW logical work overflows")?;
    Ok(Shape {
        channels: g.channels(),
        spatial,
        positions,
        elements,
        bytes,
        operations,
        tree_width,
        epsilon,
        scale,
    })
}

#[cfg(test)]
mod tests;
