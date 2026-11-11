//! One managed native row-map packet, with independent replacement/addition
//! profiles and exact bit transport outside the selected rows.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    PhysicalWeightLayout, RowScatterF32Geometry, RowScatterF32Op, WeightFormatId,
    ROW_SCATTER_F32_MAX_RUNS,
};
pub const REPLACE_NUMERICAL_PROFILE: &str = "cuda.row_scatter_replace.f32.exact_bits.v1";
pub const ADD_NUMERICAL_PROFILE: &str = "cuda.row_scatter_add.f32.selected_rows.rne.gradual.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_row_scatter_f32";
const ESTIMATOR: &str = "resource-estimator.cuda.row_scatter.f32";
#[repr(C)]
#[derive(Clone, Copy)]
struct Shape {
    rows: u64,
    columns: u64,
    destination: [u64; ROW_SCATTER_F32_MAX_RUNS],
    source: [u64; ROW_SCATTER_F32_MAX_RUNS],
    length: [u64; ROW_SCATTER_F32_MAX_RUNS],
    runs: u32,
    padding: u32,
}
unsafe impl cudarc::driver::DeviceRepr for Shape {}
const _: () = assert!(std::mem::size_of::<Shape>() == 792);
impl Shape {
    fn new(g: &RowScatterF32Geometry) -> Self {
        let mut shape = Self {
            rows: g.rows(),
            columns: g.columns(),
            destination: [0; ROW_SCATTER_F32_MAX_RUNS],
            source: [0; ROW_SCATTER_F32_MAX_RUNS],
            length: [0; ROW_SCATTER_F32_MAX_RUNS],
            runs: g.runs().len() as u32,
            padding: 0,
        };
        for (i, r) in g.runs().iter().enumerate() {
            shape.destination[i] = r.destination_start;
            shape.source[i] = r.source_start;
            shape.length[i] = r.length;
        }
        shape
    }
}
pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: RowScatterF32Op,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 scatter capability",
        ));
    }
    let c = kind.contract().map_err(contract_error)?;
    let profile = match kind {
        RowScatterF32Op::Replace => REPLACE_NUMERICAL_PROFILE,
        RowScatterF32Op::Add => ADD_NUMERICAL_PROFILE,
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("row_scatter_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_row_scatter_f32.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}", kind.operation_id()))
            .map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fingerprint,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(2),
        ESTIMATOR,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map(|d| d.with_fixed_tensor_work(true))
    .map_err(contract_error)
}
pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    kind: RowScatterF32Op,
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
    kind: RowScatterF32Op,
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
            return Err(invalid_plan("scatter estimator received another contract"));
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
            "cuda.row_scatter_f32.encode",
            "scatter requires fixed tensors".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.row_scatter_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("scatter requires one participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("scatter requires its selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(3);
        for (role, ordinal, elements) in [
            (ResolvedValueRole::Input, 0, g.elements()),
            (ResolvedValueRole::Input, 1, g.update_rows() * g.columns()),
            (ResolvedValueRole::Output, 0, g.elements()),
        ] {
            let b = binding(p.bindings(), role, ordinal)?;
            let [component] = b.storage().components() else {
                return Err("scatter requires one physical component per port".into());
            };
            if component.element_type() != ElementType::F32
                || elements.checked_mul(4) != Some(component.length_bytes())
            {
                return Err("scatter physical dtype or extent differs".into());
            }
            regions.push(tensor_storage::component(p, component, 16)?);
        }
        for r in &regions[..2] {
            dense::disjoint(
                r.device_ptr(),
                r.length_bytes(),
                regions[2].device_ptr(),
                regions[2].length_bytes(),
            )?;
        }
        let shape = Shape::new(&g);
        let elements = g.elements();
        let operations = g.logical_operations(self.kind);
        let add = match self.kind {
            RowScatterF32Op::Replace => 0i32,
            RowScatterF32Op::Add => 1,
        };
        let name = if add == 0 {
            "vnext_row_scatter_replace_f32"
        } else {
            "vnext_row_scatter_add_f32"
        };
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(name, regions, move |stream, regions| {
            let base = regions[0].device_ptr();
            let updates = regions[1].device_ptr();
            let output = regions[2].device_ptr();
            let mut b = stream.launch_builder(kernels.row_scatter_f32_function());
            b.arg(&base)
                .arg(&updates)
                .arg(&output)
                .arg(&shape)
                .arg(&add);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 selected row scatter", e))
        })
        .and_then(|c| c.with_tensor_work(elements, operations, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowScatterF32Geometry, String> {
    if values.len() != 3
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("scatter requires three exact dense ports".into());
    }
    RowScatterF32Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Input, 1)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())
}
