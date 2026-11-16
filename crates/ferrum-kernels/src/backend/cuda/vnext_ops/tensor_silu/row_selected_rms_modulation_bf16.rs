//! Fused RMS and compact modulation on the shared admitted CUDA module.
use super::*;
use ferrum_runtime::{
    row_selected_residual_rms_modulation_bf16_contract, row_selected_rms_modulation_bf16_contract,
    DynamicStorageRequirement, PhysicalWeightLayout, ProviderStorageBindingRequirement,
    RowSelectedRmsModulationGeometry, RowSelectedRmsModulationKind, WeightFormatId,
};

pub const NUMERICAL_PROFILE: &str =
    "cuda.row_selected_rms_modulation.bf16.block8192_vec8_warp16_fma.div_full.rsqrt_approx_ftz.v1";
fn contract(kind: RowSelectedRmsModulationKind) -> Result<StandardOperationContract, VNextError> {
    match kind {
        RowSelectedRmsModulationKind::Normalize => row_selected_rms_modulation_bf16_contract(),
        RowSelectedRmsModulationKind::Residual => {
            row_selected_residual_rms_modulation_bf16_contract()
        }
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: RowSelectedRmsModulationKind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks fused RMS modulation capability",
        ));
    }
    let c = contract(kind).map_err(contract_error)?;
    let name = kind.operation_id().trim_start_matches("operation.");
    let estimator = format!("resource-estimator.cuda.{name}");
    let fingerprint = implementation_fingerprint(&[
        kind.operation_id().as_bytes(),
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("row_selected_rms_modulation_bf16.rs"),
        include_bytes!("row_routing.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_row_selected_rms_modulation_bf16.cuh"
        )),
    ]);
    let mut bindings = super::super::transformer::contiguous_bindings(kind.input_count() as u32);
    if kind == RowSelectedRmsModulationKind::Residual {
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
    }
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{name}")).map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        bindings,
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
    kind: RowSelectedRmsModulationKind,
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
    kind: RowSelectedRmsModulationKind,
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
            return Err(invalid_plan("estimator received another operation"));
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
            "cuda.fused_rms_modulation.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.fused_rms_modulation.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one fixed tensor participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed tensor contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(self.kind.input_count() + self.kind.output_count());
        for (role, count) in [
            (ResolvedValueRole::Input, self.kind.input_count()),
            (ResolvedValueRole::Output, self.kind.output_count()),
        ] {
            for n in 0..count {
                let b = binding(p.bindings(), role, n as u32)?;
                regions.push(retained_typed_region(
                    p,
                    b,
                    b.tensor()
                        .minimum_storage_bytes()
                        .map_err(|e| e.to_string())?,
                    ElementType::Bf16,
                )?);
            }
        }
        for out in self.kind.input_count()..regions.len() {
            for other in 0..out {
                nonoverlapping_ranges(
                    regions[out].device_ptr(),
                    regions[out].length_bytes(),
                    regions[other].device_ptr(),
                    regions[other].length_bytes(),
                )?;
            }
        }
        let launches = row_routing::chunks(g.routing()).len() as u64;
        let route = g.routing();
        let (width, table_width) = (route.columns(), route.parameter_columns());
        let (shift, scale, gate) = (route.shift_column(), route.scale_column(), g.gate_column());
        let epsilon = g.effective_epsilon();
        let residual = u32::from(self.kind == RowSelectedRmsModulationKind::Residual);
        let kernels = Arc::clone(&self.kernels);
        let work = g.logical_arithmetic_operations();
        let elements = route.elements();
        CudaDeviceCommand::operation(
            "vnext_row_selected_rms_modulation_bf16",
            regions,
            move |stream, regions| {
                for chunk in row_routing::chunks(g.routing()) {
                    let routing = chunk.routing;
                    let rows = chunk.rows;
                    let x = regions[0].device_ptr() + chunk.byte_offset;
                    let table = regions[1].device_ptr();
                    let gamma = regions[2].device_ptr();
                    let branch = if residual == 1 {
                        regions[3].device_ptr() + chunk.byte_offset
                    } else {
                        x
                    };
                    let hidden = if residual == 1 {
                        regions[4].device_ptr() + chunk.byte_offset
                    } else {
                        regions[3].device_ptr() + chunk.byte_offset
                    };
                    let output =
                        regions.last().expect("checked output").device_ptr() + chunk.byte_offset;
                    let mut launch =
                        stream.launch_builder(kernels.row_selected_rms_modulation_bf16_function());
                    launch
                        .arg(&x)
                        .arg(&table)
                        .arg(&gamma)
                        .arg(&branch)
                        .arg(&hidden)
                        .arg(&output)
                        .arg(&rows)
                        .arg(&width)
                        .arg(&table_width)
                        .arg(&shift)
                        .arg(&scale)
                        .arg(&gate)
                        .arg(&epsilon)
                        .arg(&residual)
                        .arg(&routing);
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (rows.min(65535) as u32, 1, 1),
                            block_dim: (512, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map(|_| ())
                    .map_err(|e| {
                        CudaDeviceRuntimeError::driver("fused RMS modulation launch", e)
                    })?;
                }
                Ok(())
            },
        )
        .and_then(|c| c.with_tensor_work(elements, work, launches, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: RowSelectedRmsModulationKind,
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowSelectedRmsModulationGeometry, String> {
    if b.len() != kind.input_count() + kind.output_count()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires dense numerical operands".into());
    }
    let inputs = (0..kind.input_count())
        .map(|i| binding(b, ResolvedValueRole::Input, i as u32).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = (0..kind.output_count())
        .map(|i| binding(b, ResolvedValueRole::Output, i as u32).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    RowSelectedRmsModulationGeometry::new(kind, &inputs, &outputs, attrs).map_err(|e| e.to_string())
}
