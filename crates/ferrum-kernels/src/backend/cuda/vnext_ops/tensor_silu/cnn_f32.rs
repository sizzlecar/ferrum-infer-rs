//! Generic F32 NCHW GroupNorm+SiLU and residual providers. Convolution has
//! a separate explicitly configured cuDNN owner in conv2d_f32.
use super::*;
use ferrum_runtime::{F32CnnGeometry, F32CnnOp, PhysicalWeightLayout, WeightFormatId};
pub(super) const FUNCTIONS: [&str; 4] = [
    "ferrum_vnext_nchw_pad_f32",
    "ferrum_vnext_nchw_bias_f32",
    "ferrum_vnext_group_norm_silu_nchw_f32",
    "ferrum_vnext_plain_residual_add_nchw_f32",
];
pub fn numerical_profile(kind: F32CnnOp) -> &'static str {
    match kind {
        F32CnnOp::GroupNormSilu => "cuda.group_norm_silu.nchw.f32.tree256.two_pass.libdevice.rn.v1",
        F32CnnOp::ResidualAdd => "cuda.plain_residual_add.nchw.f32.rn.no_ftz.v1",
        F32CnnOp::Conv2d => super::conv2d_f32::NUMERICAL_PROFILE_ID,
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: F32CnnOp,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    if kind == F32CnnOp::Conv2d {
        return Err(CudaDeviceRuntimeError::contract(
            "F32 convolution requires explicit cuDNN composition",
        ));
    }
    let cap = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks selected F32 CNN capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let fp = implementation_fingerprint(&[
        numerical_profile(kind).as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("cnn_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_cnn_f32.cuh"
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
        fp.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(kind.input_count() as u32),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fp.as_bytes()]),
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    kind: F32CnnOp,
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
    kind: F32CnnOp,
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
            return Err(invalid_plan("CNN estimator received another operation"));
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
            "cuda.cnn_f32.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.cnn_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("CNN requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed CNN contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::new();
        for n in 0..self.kind.input_count() {
            regions.push(retained_typed_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, n as u32)?,
                g.input_elements(n).ok_or("missing CNN input extent")? * 4,
                ElementType::F32,
            )?);
        }
        let output = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            g.elements() * 4,
            ElementType::F32,
        )?;
        for input in &regions {
            nonoverlapping_ranges(
                input.device_ptr(),
                input.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let kernels = Arc::clone(&self.kernels);
        let elements = g.elements();
        let work = g.logical_arithmetic_operations();
        CudaDeviceCommand::operation(g.kind().operation_id(), regions, move |stream, regions| {
            let input = regions[0].device_ptr();
            let parameter = regions[1].device_ptr();
            let out = regions.last().unwrap().device_ptr();
            match g {
                F32CnnGeometry::ResidualAdd(_) => {
                    let mut launch = stream.launch_builder(kernels.cnn_f32_function(3));
                    launch.arg(&input).arg(&parameter).arg(&out).arg(&elements);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (elements.div_ceil(256).min(MAX_BLOCKS) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                F32CnnGeometry::GroupNormSilu(g) => {
                    let beta = regions[2].device_ptr();
                    let [batch, channels, height, width] = g.shape();
                    let groups = batch * g.groups();
                    let group_elements = g.group_elements();
                    let spatial = height * width;
                    let epsilon = g.effective_epsilon();
                    let mut launch = stream.launch_builder(kernels.cnn_f32_function(2));
                    launch
                        .arg(&input)
                        .arg(&parameter)
                        .arg(&beta)
                        .arg(&out)
                        .arg(&groups)
                        .arg(&group_elements)
                        .arg(&channels)
                        .arg(&spatial)
                        .arg(&epsilon);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (groups.min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                F32CnnGeometry::Conv2d(_) => unreachable!("external cuDNN provider required"),
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 NCHW CNN launch", e))
        })
        .and_then(|c| c.with_tensor_work(elements, work, 1, 0))
        .map_err(|e| e.to_string())
    }
}
pub(super) fn geometry(
    kind: F32CnnOp,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<F32CnnGeometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact Dense F32 CNN ports".into());
    }
    let inputs = (0..kind.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n as u32).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    F32CnnGeometry::new(
        kind,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
