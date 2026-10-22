//! Single-head spatial attention and exact spatial transport, using the same
//! admitted buffers and one shared native-module owner as the other providers.

use super::*;
use ferrum_runtime::{
    spatial_attention_bf16_contract, spatial_upsample_gather_bf16_contract, PhysicalWeightLayout,
    SpatialAttentionBf16Geometry, SpatialUpsampleGatherBf16Geometry, WeightFormatId,
    SPATIAL_ATTENTION_BF16_CAPABILITY_ID, SPATIAL_ATTENTION_BF16_OPERATION_ID,
    SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID, SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
};

pub const ATTENTION_PROFILE_ID: &str = "cuda.ieee.spatial_attention.bf16.online_padded_pairwise.v1";
pub const GATHER_PROFILE_ID: &str = "cuda.bitcopy.spatial_upsample_gather.bf16.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "attention=single-head-full-spatial;QKV[N,3C,H,W];Y[N,C,H,W];C=1..2048;",
    "256-threads;8-Q-and-accumulator-scalar-slots-per-thread;",
    "FP32-dot-zero-padded-nextpow2-adjacent-tree;scale-mul-rn;ascending-key-online;",
    "first:m=score,l=1,acc=V;later:m=max(m,score);a=expf(sub_rn(old,m));",
    "p=expf(sub_rn(score,m));l=add_rn(mul_rn(l,a),p);",
    "acc=add_rn(mul_rn(acc,a),mul_rn(p,V));Y=B(mul_rn(acc,div_rn(1,l)));",
    "nonfinite-score=whole-query-NaN;V=IEEE-per-channel;NaN-class-only;",
    "no-global-scratch-or-S-squared-storage;dynamic-shared=4*nextpow2(C);",
    "static-shared=6-floats+one-u32-with-target-padding;",
    "finite-path-nominal-logical-work-from-geometry;not-hardware-FLOPs;",
    "general-expf-not-qualified-by-BF16-SiLU-evidence;no-Torch-bitwise-or-throughput-claim;",
    "gather=opaque-u16-copy;tile<=64;by-value-initialized-metadata;NoAlias"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Attention,
    Gather,
}
impl Kind {
    pub(super) fn index(self) -> usize {
        match self {
            Self::Attention => 0,
            Self::Gather => 1,
        }
    }
    pub(super) fn function(self) -> &'static str {
        match self {
            Self::Attention => "ferrum_vnext_spatial_attention_bf16",
            Self::Gather => "ferrum_vnext_spatial_upsample_gather_bf16",
        }
    }
    fn operation(self) -> &'static str {
        match self {
            Self::Attention => SPATIAL_ATTENTION_BF16_OPERATION_ID,
            Self::Gather => SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
        }
    }
    fn capability(self) -> &'static str {
        match self {
            Self::Attention => SPATIAL_ATTENTION_BF16_CAPABILITY_ID,
            Self::Gather => SPATIAL_UPSAMPLE_GATHER_BF16_CAPABILITY_ID,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::Attention => ATTENTION_PROFILE_ID,
            Self::Gather => GATHER_PROFILE_ID,
        }
    }
    fn provider(self) -> &'static str {
        match self {
            Self::Attention => "provider.cuda.spatial_attention.bf16",
            Self::Gather => "provider.cuda.spatial_upsample_gather.bf16",
        }
    }
    fn estimator(self) -> &'static str {
        match self {
            Self::Attention => "resource-estimator.cuda.spatial_attention.bf16",
            Self::Gather => "resource-estimator.cuda.spatial_upsample_gather.bf16",
        }
    }
    fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::Attention => spatial_attention_bf16_contract(),
            Self::Gather => spatial_upsample_gather_bf16_contract(),
        }
    }
}

pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: Kind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(kind.capability()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks selected spatial capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let fp = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        kind.profile().as_bytes(),
        kind.provider().as_bytes(),
    ]);
    let estimator = implementation_fingerprint(&[kind.estimator().as_bytes(), fp.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(kind.provider()).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fp,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(1),
        kind.estimator(),
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    kind: Kind,
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
    kind: Kind,
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}
impl OperationResourceEstimator for SpatialProvider {
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
                "CUDA spatial estimator received another operation",
            ));
        }
        geometry(self.kind, request.values(), request.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
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
            "cuda.spatial.encode",
            "CUDA spatial operations require fixed tensor work".into(),
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
            .map_err(|e| provider_failure(id, "cuda.spatial.encode", e))
    }
}
impl SpatialProvider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("CUDA spatial operation requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA spatial operation requires its fixed tensor contract".into());
        }
        let shape = geometry(self.kind, p.bindings(), p.attributes())?;
        let input = retained_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Input, 0)?,
            shape.input_bytes,
        )?;
        let output = retained_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            shape.output_bytes,
        )?;
        nonoverlapping_ranges(
            input.device_ptr(),
            input.length_bytes(),
            output.device_ptr(),
            output.length_bytes(),
        )?;
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        CudaDeviceCommand::operation(
            kind.function(),
            vec![input, output],
            move |stream, regions| {
                let input = regions[0].device_ptr();
                let output = regions[1].device_ptr();
                let mut launch = stream.launch_builder(kernels.extended_spatial_function(kind));
                launch.arg(&input).arg(&output);
                let config = match &shape.native {
                    NativeShape::Attention {
                        positions,
                        channels,
                        spatial,
                        tree_width,
                        scale,
                    } => {
                        launch
                            .arg(positions)
                            .arg(channels)
                            .arg(spatial)
                            .arg(tree_width)
                            .arg(scale);
                        LaunchConfig {
                            grid_dim: ((*positions).min(MAX_BLOCKS) as u32, 1, 1),
                            block_dim: (THREADS, 1, 1),
                            shared_mem_bytes: *tree_width * 4,
                        }
                    }
                    NativeShape::Gather(metadata) => {
                        launch.arg(&shape.elements).arg(metadata);
                        LaunchConfig {
                            grid_dim: (
                                shape.elements.div_ceil(THREADS as u64).min(MAX_BLOCKS) as u32,
                                1,
                                1,
                            ),
                            block_dim: (THREADS, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    }
                };
                unsafe { launch.launch(config) }
                    .map(|_| ())
                    .map_err(|e| CudaDeviceRuntimeError::driver("BF16 spatial operation", e))
            },
        )
        .and_then(|c| c.with_tensor_work(shape.elements, shape.operations, 1, 0))
        .map_err(|e| e.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    native: NativeShape,
    input_bytes: u64,
    output_bytes: u64,
    elements: u64,
    operations: u64,
}
#[derive(Clone, Copy, Debug)]
enum NativeShape {
    Attention {
        positions: u64,
        channels: u32,
        spatial: u64,
        tree_width: u32,
        scale: f32,
    },
    Gather(NativeGather),
}
/// Exact initialized by-value ABI; 584 bytes plus pointers and element count.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct NativeGather {
    offsets: [u64; 64],
    input_channels: u64,
    input_height: u64,
    input_width: u64,
    output_channels: u64,
    output_height: u64,
    output_width: u64,
    scale_h: u64,
    scale_w: u64,
    channel_stride: u64,
}
// SAFETY: fixed-width initialized scalars only, matching the CUDA repr(C) ABI.
unsafe impl cudarc::driver::DeviceRepr for NativeGather {}
const _: [(); 584] = [(); std::mem::size_of::<NativeGather>()];
impl NativeGather {
    fn new(g: &SpatialUpsampleGatherBf16Geometry) -> Self {
        let mut offsets = [0; 64];
        offsets[..g.channel_offsets().len()].copy_from_slice(g.channel_offsets());
        Self {
            offsets,
            input_channels: g.input_channels(),
            input_height: g.input_height(),
            input_width: g.input_width(),
            output_channels: g.output_channels(),
            output_height: g.output_height(),
            output_width: g.output_width(),
            scale_h: g.scale_h(),
            scale_w: g.scale_w(),
            channel_stride: g.channel_stride(),
        }
    }
}
fn geometry(
    kind: Kind,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if values.len() != 2 {
        return Err("CUDA spatial operation requires one input and one output".into());
    }
    for v in values {
        if v.weight()
            .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        {
            return Err("CUDA spatial operation requires dense physical weights".into());
        }
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let bytes = |t: &ferrum_runtime::ResolvedTensorSpec| {
        t.minimum_storage_bytes()
            .map_err(|e| e.to_string())
            .and_then(|b| {
                if b <= isize::MAX as u64 {
                    Ok(b)
                } else {
                    Err("CUDA spatial tensor exceeds address space".into())
                }
            })
    };
    let (native, elements, operations) = match kind {
        Kind::Attention => {
            let g = SpatialAttentionBf16Geometry::new(x, y, attrs).map_err(|e| e.to_string())?;
            (
                NativeShape::Attention {
                    positions: g.batch() * g.spatial(),
                    channels: g.channels() as u32,
                    spatial: g.spatial(),
                    tree_width: g.channels().next_power_of_two() as u32,
                    scale: g.effective_scale(),
                },
                g.element_count(),
                g.logical_arithmetic_operations(),
            )
        }
        Kind::Gather => {
            let g =
                SpatialUpsampleGatherBf16Geometry::new(x, y, attrs).map_err(|e| e.to_string())?;
            (
                NativeShape::Gather(NativeGather::new(&g)),
                g.element_count(),
                0,
            )
        }
    };
    Ok(Shape {
        native,
        input_bytes: bytes(x)?,
        output_bytes: bytes(y)?,
        elements,
        operations,
    })
}

#[cfg(test)]
mod tests;
