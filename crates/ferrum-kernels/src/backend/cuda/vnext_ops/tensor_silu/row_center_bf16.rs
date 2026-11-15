use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    DynamicStorageRequirement, PhysicalWeightLayout, ProviderStorageBindingRequirement,
    RowCenterBf16Geometry, WeightFormatId, ROW_CENTER_BF16_CAPABILITY_ID,
    ROW_CENTER_BF16_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str =
    "cuda.row_center_bf16.f32_stride8_tree_mean_bf16_subtract_rn.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.row_center_bf16";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(ROW_CENTER_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "missing row center capability",
        ));
    }
    let contract = ferrum_runtime::row_center_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("row_center_bf16.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_row_center_bf16.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.row_center.bf16").map_err(contract_error)?,
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
        {
            let mut bindings = super::super::transformer::contiguous_bindings(1);
            bindings.push(ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Output,
                1,
                DynamicStorageRequirement::contiguous(),
            ));
            bindings
        },
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
                "row center estimator received another operation",
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
            "cuda.row_center_bf16.encode",
            "requires fixed tensor execution".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.row_center_bf16.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one tensor participant".into());
        };
        if i.operation().id.as_str() != ROW_CENTER_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires row center fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(3);
        for (role, ordinal, bytes) in [
            (ResolvedValueRole::Input, 0, g.data_bytes()),
            (ResolvedValueRole::Output, 0, g.data_bytes()),
            (ResolvedValueRole::Output, 1, g.mean_bytes()),
        ] {
            let b = binding(p.bindings(), role, ordinal)?;
            let [c] = b.storage().components() else {
                return Err("row center requires one component per port".into());
            };
            if c.element_type() != ElementType::Bf16 || c.length_bytes() != bytes {
                return Err("row center physical dtype/extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        for i in 0..3 {
            for j in i + 1..3 {
                dense::disjoint(
                    regions[i].device_ptr(),
                    regions[i].length_bytes(),
                    regions[j].device_ptr(),
                    regions[j].length_bytes(),
                )?;
            }
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            ROW_CENTER_BF16_OPERATION_ID,
            regions,
            move |stream, regions| {
                let x = regions[0].device_ptr();
                let y = regions[1].device_ptr();
                let mean = regions[2].device_ptr();
                let elements = g.elements();
                let means = g.mean_elements();
                let rows = g.rows();
                let columns = g.columns();
                let factor = g.factor();
                let mut builder = stream.launch_builder(kernels.row_center_bf16_function(0));
                builder
                    .arg(&x)
                    .arg(&mean)
                    .arg(&means)
                    .arg(&rows)
                    .arg(&columns)
                    .arg(&factor);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (means.div_ceil(32).min(65535) as u32, 1, 1),
                        block_dim: (32, 8, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| CudaDeviceRuntimeError::driver("row center mean launch", e))?;
                let mut builder = stream.launch_builder(kernels.row_center_bf16_function(1));
                builder
                    .arg(&x)
                    .arg(&mean)
                    .arg(&y)
                    .arg(&elements)
                    .arg(&rows)
                    .arg(&columns);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("row center subtract launch", e))
            },
        )
        .and_then(|c| c.with_tensor_work(g.elements() + g.mean_elements(), g.arithmetic(), 2, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<RowCenterBf16Geometry, String> {
    if b.len() != 3
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("row center requires three dense ports".into());
    }
    RowCenterBf16Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        binding(b, ResolvedValueRole::Output, 1)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
