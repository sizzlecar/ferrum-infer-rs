//! One retained precise kernel, four independent typed F32 operation profiles.
use super::*;
use ferrum_runtime::{
    F32ChannelwiseGeometry, F32ChannelwiseOp, PhysicalWeightLayout, WeightFormatId,
};
pub const FUNCTION: &str = "ferrum_vnext_channelwise_f32";
pub fn numerical_profile(kind: F32ChannelwiseOp) -> &'static str {
    match kind {
        F32ChannelwiseOp::GroupedColumnSlice => "cuda.grouped_column_slice.f32.exact_bits.v1",
        F32ChannelwiseOp::BiasAdd => "cuda.channel_bias_add.f32.rne.gradual.v1",
        F32ChannelwiseOp::ScaledResidual => {
            "cuda.channel_scaled_residual.f32.single_fma.gradual.v1"
        }
        F32ChannelwiseOp::NanToNum => "cuda.nan_to_num.f32.exact_finite_bits.v1",
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: F32ChannelwiseOp,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks selected F32 channelwise capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        numerical_profile(kind).as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("channelwise_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_channelwise_f32.cuh"
        )),
    ]);
    let estimator = format!("resource-estimator.cuda.{}", kind.operation_id());
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}", kind.operation_id()))
            .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new("weight-format.safetensors.dense").map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.input_count() as u32),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    kind: F32ChannelwiseOp,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        kind,
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    kind: F32ChannelwiseOp,
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
            return Err(invalid_plan(
                "F32 channelwise estimator received another operation",
            ));
        }
        geometry(self.kind, r.values(), r.attributes()).map_err(invalid_plan)?;
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
impl OperationProvider<CudaDeviceRuntime> for Provider {
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
            "cuda.channelwise_f32.encode",
            "requires fixed tensor invocation".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = i.participants()[0].identity().clone();
        self.encode(i)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, "cuda.channelwise_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one F32 channelwise participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed F32 channelwise contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.input_count() + 1);
        for (role, ordinal) in (0..self.kind.input_count())
            .map(|n| (ResolvedValueRole::Input, n as u32))
            .chain(std::iter::once((ResolvedValueRole::Output, 0)))
        {
            let b = binding(p.bindings(), role, ordinal)?;
            let elements = if role == ResolvedValueRole::Output {
                Some(g.elements())
            } else {
                g.input_elements(ordinal as usize)
            }
            .ok_or("unknown channelwise input ordinal")?;
            regions.push(retained_typed_region(
                p,
                b,
                elements.checked_mul(4).ok_or("byte extent overflows")?,
                ElementType::F32,
            )?);
        }
        let last = regions.len() - 1;
        for input in &regions[..last] {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                regions[last].device_ptr(),
                regions[last].length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        let inputs = self.kind.input_count();
        CudaDeviceCommand::operation(FUNCTION, regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let p1 = if inputs > 1 {
                regions[1].device_ptr()
            } else {
                0
            };
            let p2 = if inputs > 2 {
                regions[2].device_ptr()
            } else {
                0
            };
            let y = regions[inputs].device_ptr();
            let elements = g.elements();
            let source_columns = g.source_columns();
            let columns = g.columns();
            let group_width = g.group_width();
            let column_start = g.column_start();
            let column_count = g.column_count();
            let kind = match g.kind() {
                F32ChannelwiseOp::GroupedColumnSlice => 0u32,
                F32ChannelwiseOp::BiasAdd => 1,
                F32ChannelwiseOp::ScaledResidual => 2,
                F32ChannelwiseOp::NanToNum => 3,
            };
            let mut launch = stream.launch_builder(kernels.channelwise_f32_function());
            launch
                .arg(&x)
                .arg(&p1)
                .arg(&p2)
                .arg(&y)
                .arg(&elements)
                .arg(&source_columns)
                .arg(&columns)
                .arg(&group_width)
                .arg(&column_start)
                .arg(&column_count)
                .arg(&kind);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(MAX_BLOCKS) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 channelwise launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), g.logical_arithmetic_operations(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: F32ChannelwiseOp,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<F32ChannelwiseGeometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact dense F32 channelwise operands".into());
    }
    let inputs = (0..kind.input_count())
        .map(|index| binding(b, ResolvedValueRole::Input, index as u32).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    F32ChannelwiseGeometry::new(
        kind,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
