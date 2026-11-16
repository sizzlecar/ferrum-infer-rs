use super::*;
use ferrum_runtime::{PhysicalWeightLayout, SignalF32Geometry, SignalF32Op, WeightFormatId};
pub const NUMERICAL_PROFILE: &str =
    "cuda.signal.f32.tap_fma_ascending.libdevice_separate_rn.gradual.v1";
pub(super) const FUNCTIONS: [&str; 4] = [
    "ferrum_vnext_fir_resample1d_f32",
    "ferrum_vnext_snake_beta_log_nchw_f32",
    "ferrum_vnext_clamp_rank4_f32",
    "ferrum_vnext_snake_alpha_nchw_f32",
];
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: SignalF32Op,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "missing F32 signal capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        kind.operation_id().as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("signal_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_signal_f32.cuh"
        )),
    ]);
    let estimator = format!("resource-estimator.cuda.{}", kind.operation_id());
    let fp = implementation_fingerprint(&[estimator.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}", kind.operation_id()))
            .map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
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
        estimator,
        ContractVersion::new(1, 0),
        fp,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    kind: SignalF32Op,
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
    kind: SignalF32Op,
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
            return Err(invalid_plan("signal estimator received another operation"));
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
            "cuda.signal_f32",
            "signal requires fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.signal_f32", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("signal requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed signal contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.input_count() + 1);
        for n in 0..self.kind.input_count() {
            regions.push(retained_typed_region(
                p,
                binding(p.bindings(), ResolvedValueRole::Input, n as u32)?,
                g.input_elements(n).unwrap() * 4,
                ElementType::F32,
            )?);
        }
        let output = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            g.output_elements() * 4,
            ElementType::F32,
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
        CudaDeviceCommand::operation(self.kind.operation_id(), regions, move |stream, regions| {
            let input = regions[0].device_ptr();
            let output = regions.last().unwrap().device_ptr();
            let elements = g.output_elements();
            let function = match g.kind {
                SignalF32Op::FirResample => 0,
                SignalF32Op::SnakeBetaLog => 1,
                SignalF32Op::Clamp => 2,
                SignalF32Op::SnakeAlpha => 3,
            };
            let mut b = stream.launch_builder(kernels.signal_f32_function(function));
            match g.kind {
                SignalF32Op::FirResample => {
                    let taps = regions[1].device_ptr();
                    let transpose = i32::from(g.transpose);
                    b.arg(&input)
                        .arg(&taps)
                        .arg(&output)
                        .arg(&elements)
                        .arg(&g.input_width)
                        .arg(&g.output_width)
                        .arg(&g.taps)
                        .arg(&g.stride)
                        .arg(&g.padding[0])
                        .arg(&g.padding[1])
                        .arg(&g.crop[0])
                        .arg(&transpose);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                SignalF32Op::SnakeBetaLog | SignalF32Op::SnakeAlpha => {
                    let a = regions[1].device_ptr();
                    let beta = regions[2].device_ptr();
                    let spatial = g.height * g.input_width;
                    b.arg(&input).arg(&a);
                    if g.kind == SignalF32Op::SnakeBetaLog {
                        b.arg(&beta);
                    }
                    b.arg(&output).arg(&elements).arg(&g.channels).arg(&spatial);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
                SignalF32Op::Clamp => {
                    b.arg(&input)
                        .arg(&output)
                        .arg(&elements)
                        .arg(&g.minimum)
                        .arg(&g.maximum);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                }
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 signal launch", e))
        })
        .and_then(|c| {
            c.with_tensor_work(g.output_elements(), g.logical_arithmetic_operations(), 1, 0)
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: SignalF32Op,
    b: &[ResolvedValueBinding],
    a: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<SignalF32Geometry, String> {
    if b.len() != kind.input_count() + 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("signal requires exact dense ports".into());
    }
    let inputs = (0..kind.input_count())
        .map(|n| binding(b, ResolvedValueRole::Input, n as u32).map(|v| v.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    SignalF32Geometry::new(
        kind,
        &inputs,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        a,
    )
    .map_err(|e| e.to_string())
}
