//! Exact F32 bit transport over admitted CUDA owners; no arithmetic or cast.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    reshape_copy_f32_contract, PhysicalWeightLayout, ReshapeCopyF32Geometry, WeightFormatId,
    RESHAPE_COPY_F32_CAPABILITY_ID, RESHAPE_COPY_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str = "cuda.reshape_copy.f32.exact_bits.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_reshape_copy_f32";
const ESTIMATOR: &str = "resource-estimator.cuda.reshape_copy.f32";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(RESHAPE_COPY_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 reshape capability",
        ));
    }
    let c = reshape_copy_f32_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("reshape_copy_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_reshape_copy_f32.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.reshape_copy.f32").map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
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
        super::super::transformer::contiguous_bindings(1),
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
            return Err(invalid_plan("reshape estimator received another contract"));
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
            "cuda.reshape_copy_f32.encode",
            "reshape requires fixed tensors".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.reshape_copy_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("reshape requires one participant".into());
        };
        if i.operation().id.as_str() != RESHAPE_COPY_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("reshape requires its selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(2);
        for role in [ResolvedValueRole::Input, ResolvedValueRole::Output] {
            let b = binding(p.bindings(), role, 0)?;
            let [component] = b.storage().components() else {
                return Err("reshape requires one original physical component".into());
            };
            if component.element_type() != ElementType::F32 || component.length_bytes() != g.bytes()
            {
                return Err("reshape physical dtype/extent differs".into());
            }
            regions.push(tensor_storage::component(p, component, 16)?);
        }
        dense::disjoint(
            regions[0].device_ptr(),
            g.bytes(),
            regions[1].device_ptr(),
            g.bytes(),
        )?;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation("vnext_reshape_copy_f32", regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let y = regions[1].device_ptr();
            let elements = g.elements();
            let mut builder = stream.launch_builder(kernels.reshape_copy_f32_function());
            builder.arg(&x).arg(&y).arg(&elements);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 reshape bit copy", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), 0, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<ReshapeCopyF32Geometry, String> {
    if values.len() != 2
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("reshape requires exact dense original ports".into());
    }
    reshape_copy_f32_contract()
        .map_err(|e| e.to_string())?
        .descriptor()
        .validate_attributes(attrs)
        .map_err(|e| e.to_string())?;
    let u = |name: &str| {
        attrs
            .iter()
            .find(|(k, _)| k.as_str() == name)
            .and_then(|(_, v)| {
                if let SemanticValue::Unsigned(n) = v {
                    Some(*n)
                } else {
                    None
                }
            })
            .ok_or_else(|| format!("missing reshape {name}"))
    };
    ReshapeCopyF32Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        u("rows")?,
        u("columns")?,
    )
    .map_err(|e| e.to_string())
}
