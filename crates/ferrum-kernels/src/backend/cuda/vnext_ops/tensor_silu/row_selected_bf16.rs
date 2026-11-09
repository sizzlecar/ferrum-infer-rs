//! Compact row routing keeps the existing expanded BF16 arithmetic boundaries.
use super::*;
use ferrum_runtime::{
    row_selected_addcmul_bf16_contract, row_selected_modulation_bf16_contract,
    PhysicalWeightLayout, RowSelectedBf16Geometry, RowSelectedBf16Kind, WeightFormatId,
    ROW_SELECTED_ADDCMUL_BF16_CAPABILITY_ID, ROW_SELECTED_BF16_MAX_RUNS,
    ROW_SELECTED_MODULATION_BF16_CAPABILITY_ID,
};

pub const MODULATION_PROFILE: &str = "cuda.row_selected_modulation.bf16.staged_rne.gradual.v1";
pub const ADDCMUL_PROFILE: &str = "cuda.row_selected_addcmul.bf16.separate_rne.gradual.v1";
pub(super) const FUNCTIONS: [&str; 2] = [
    "ferrum_vnext_row_selected_modulation_bf16",
    "ferrum_vnext_row_selected_addcmul_bf16",
];

#[repr(C)]
#[derive(Clone, Copy)]
struct Routing {
    ends: [u64; ROW_SELECTED_BF16_MAX_RUNS],
    rows: [u64; ROW_SELECTED_BF16_MAX_RUNS],
    count: u64,
}
// SAFETY: repr(C), initialized fixed-width integers and no pointers/padding;
// this matches FerrumRowSelectedBf16Routing's checked CUDA ABI exactly.
unsafe impl cudarc::driver::DeviceRepr for Routing {}
const _: [(); 520] = [(); std::mem::size_of::<Routing>()];
const _: [(); 8] = [(); std::mem::align_of::<Routing>()];
impl Routing {
    fn new(g: &RowSelectedBf16Geometry) -> Self {
        let mut result = Self {
            ends: [0; ROW_SELECTED_BF16_MAX_RUNS],
            rows: [0; ROW_SELECTED_BF16_MAX_RUNS],
            count: g.run_ends().len() as u64,
        };
        result.ends[..g.run_ends().len()].copy_from_slice(g.run_ends());
        result.rows[..g.parameter_rows().len()].copy_from_slice(g.parameter_rows());
        result
    }
}
fn name(kind: RowSelectedBf16Kind) -> &'static str {
    match kind {
        RowSelectedBf16Kind::Modulation => "row_selected_modulation",
        RowSelectedBf16Kind::Addcmul => "row_selected_addcmul",
    }
}
fn input_count(kind: RowSelectedBf16Kind) -> u32 {
    match kind {
        RowSelectedBf16Kind::Modulation => 2,
        RowSelectedBf16Kind::Addcmul => 3,
    }
}
pub(super) fn index(kind: RowSelectedBf16Kind) -> usize {
    match kind {
        RowSelectedBf16Kind::Modulation => 0,
        RowSelectedBf16Kind::Addcmul => 1,
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: RowSelectedBf16Kind,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let (contract, capability, profile) = match kind {
        RowSelectedBf16Kind::Modulation => (
            row_selected_modulation_bf16_contract(),
            ROW_SELECTED_MODULATION_BF16_CAPABILITY_ID,
            MODULATION_PROFILE,
        ),
        RowSelectedBf16Kind::Addcmul => (
            row_selected_addcmul_bf16_contract(),
            ROW_SELECTED_ADDCMUL_BF16_CAPABILITY_ID,
            ADDCMUL_PROFILE,
        ),
    };
    let cap = CapabilityId::new(capability).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks selected compact BF16 row operation capability",
        ));
    }
    let contract = contract.map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("row_selected_bf16.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_row_selected_bf16.cuh"
        )),
    ]);
    let estimator = format!("resource-estimator.cuda.{}.bf16", name(kind));
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}.bf16", name(kind))).map_err(contract_error)?,
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
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(input_count(kind)),
        &estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
    )
    .map(|descriptor| descriptor.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    kind: RowSelectedBf16Kind,
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
    kind: RowSelectedBf16Kind,
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
                "compact BF16 row estimator received another contract",
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
            "cuda.row_selected_bf16.encode",
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
            .map_err(|error| provider_failure(identity, "cuda.row_selected_bf16.encode", error))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("compact row operation requires one fixed participant".into());
        };
        if i.operation().id != *self.descriptor.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed compact BF16 row contract".into());
        }
        let g = geometry(self.kind, p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(input_count(self.kind) as usize + 1);
        for ordinal in 0..input_count(self.kind) {
            let b = binding(p.bindings(), ResolvedValueRole::Input, ordinal)?;
            regions.push(retained_typed_region(
                p,
                b,
                b.tensor()
                    .minimum_storage_bytes()
                    .map_err(|e| e.to_string())?,
                ElementType::Bf16,
            )?);
        }
        let b = binding(p.bindings(), ResolvedValueRole::Output, 0)?;
        let output = retained_typed_region(
            p,
            b,
            b.tensor()
                .minimum_storage_bytes()
                .map_err(|e| e.to_string())?,
            ElementType::Bf16,
        )?;
        for read in &regions {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let routing = Routing::new(&g);
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        let (elements, columns, parameter_columns) =
            (g.elements(), g.columns(), g.parameter_columns());
        let (shift, scale, gate) = (g.shift_column(), g.scale_column(), g.gate_column());
        let work = g.logical_arithmetic_operations();
        CudaDeviceCommand::operation(FUNCTIONS[index(kind)], regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let table = regions[1].device_ptr();
            let output = regions[input_count(kind) as usize].device_ptr();
            let mut launch = stream.launch_builder(kernels.row_selected_bf16_function(kind));
            let config = LaunchConfig {
                grid_dim: (elements.div_ceil(256).min(MAX_BLOCKS) as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let launched = match kind {
                RowSelectedBf16Kind::Modulation => {
                    launch
                        .arg(&x)
                        .arg(&table)
                        .arg(&output)
                        .arg(&elements)
                        .arg(&columns)
                        .arg(&parameter_columns)
                        .arg(&shift)
                        .arg(&scale)
                        .arg(&routing);
                    unsafe { launch.launch(config) }
                }
                RowSelectedBf16Kind::Addcmul => {
                    let update = regions[2].device_ptr();
                    launch
                        .arg(&x)
                        .arg(&table)
                        .arg(&update)
                        .arg(&output)
                        .arg(&elements)
                        .arg(&columns)
                        .arg(&parameter_columns)
                        .arg(&gate)
                        .arg(&routing);
                    unsafe { launch.launch(config) }
                }
            };
            launched
                .map(|_| ())
                .map_err(|error| CudaDeviceRuntimeError::driver("compact BF16 row launch", error))
        })
        .and_then(|command| command.with_tensor_work(elements, work, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    kind: RowSelectedBf16Kind,
    bindings: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowSelectedBf16Geometry, String> {
    if bindings.len() != input_count(kind) as usize + 1
        || bindings.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("compact BF16 row operation requires exact dense physical ports".into());
    }
    let inputs = (0..input_count(kind))
        .map(|ordinal| binding(bindings, ResolvedValueRole::Input, ordinal).map(|b| b.tensor()))
        .collect::<Result<Vec<_>, _>>()?;
    RowSelectedBf16Geometry::new(
        kind,
        &inputs,
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|error| error.to_string())
}
