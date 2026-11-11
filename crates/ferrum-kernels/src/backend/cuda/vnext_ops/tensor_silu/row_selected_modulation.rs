use super::*;
use ferrum_runtime::{
    row_selected_modulation_bf16_f32_contract, PhysicalWeightLayout,
    RowSelectedModulationBf16F32Geometry, WeightFormatId,
    ROW_SELECTED_MODULATION_BF16_F32_CAPABILITY_ID, ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str = "cuda.row_selected_modulation.bf16_f32.separate_rne.gradual.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_row_selected_modulation_bf16_f32";
#[repr(C)]
#[derive(Clone, Copy)]
struct Routing {
    ends: [u64; 32],
    rows: [u64; 32],
    count: u64,
}
// SAFETY: repr(C) and fully initialized fixed-width integer fields match
// FerrumRowSelectedModulationRouting. There are no pointers or padding bytes.
unsafe impl cudarc::driver::DeviceRepr for Routing {}
const _: [(); 520] = [(); std::mem::size_of::<Routing>()];
const _: [(); 8] = [(); std::mem::align_of::<Routing>()];
impl Routing {
    fn new(g: &RowSelectedModulationBf16F32Geometry) -> Self {
        let mut r = Self {
            ends: [0; 32],
            rows: [0; 32],
            count: g.run_ends().len() as u64,
        };
        r.ends[..g.run_ends().len()].copy_from_slice(g.run_ends());
        r.rows[..g.parameter_rows().len()].copy_from_slice(g.parameter_rows());
        r
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(ROW_SELECTED_MODULATION_BF16_F32_CAPABILITY_ID)
        .map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks row-selected modulation capability",
        ));
    }
    let contract = row_selected_modulation_bf16_f32_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("row_selected_modulation.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_row_selected_modulation.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.row_selected_modulation.bf16_f32";
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.row_selected_modulation.bf16_f32")
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
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new("weight-format.safetensors.dense").map_err(contract_error)?
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(2),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
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
                "row-selected estimator received another operation",
            ));
        }
        geometry(r.values(), r.attributes()).map_err(invalid_plan)?;
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
            "cuda.row_selected_modulation.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.row_selected_modulation.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one row-selected participant".into());
        };
        if i.operation().id.as_str() != ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed row modulation contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let parameter_bytes = g
            .parameter_count()
            .checked_mul(g.columns())
            .and_then(|n| n.checked_mul(8))
            .ok_or("parameter bytes overflow")?;
        let input = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Input, 0)?,
            g.elements() * 2,
            ElementType::Bf16,
        )?;
        let parameters = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Input, 1)?,
            parameter_bytes,
            ElementType::F32,
        )?;
        let output = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            g.elements() * 4,
            ElementType::F32,
        )?;
        for read in [&input, &parameters] {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        let routing = Routing::new(&g);
        let kernels = Arc::clone(&self.kernels);
        let elements = g.elements();
        let columns = g.columns();
        let work = g.logical_arithmetic_operations();
        CudaDeviceCommand::operation(
            FUNCTION,
            vec![input, parameters, output],
            move |stream, regions| {
                let x = regions[0].device_ptr();
                let p = regions[1].device_ptr();
                let y = regions[2].device_ptr();
                let mut launch = stream.launch_builder(kernels.row_selected_modulation_function());
                launch
                    .arg(&x)
                    .arg(&p)
                    .arg(&y)
                    .arg(&elements)
                    .arg(&columns)
                    .arg(&routing);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (elements.div_ceil(256).min(MAX_BLOCKS) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| {
                    CudaDeviceRuntimeError::driver("row-selected BF16/F32 modulation launch", e)
                })
            },
        )
        .and_then(|c| c.with_tensor_work(elements, work, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowSelectedModulationBf16F32Geometry, String> {
    if b.len() != 3
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("row modulation requires exact dense operands".into());
    }
    RowSelectedModulationBf16F32Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Input, 1)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
