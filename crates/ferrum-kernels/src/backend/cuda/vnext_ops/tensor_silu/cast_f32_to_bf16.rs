//! IEEE conversion using admitted owners and no global scratch or workspace.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    cast_f32_to_bf16_contract, CastF32ToBf16Geometry, PhysicalWeightLayout, WeightFormatId,
    CAST_F32_TO_BF16_CAPABILITY_ID, CAST_F32_TO_BF16_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str = "cuda.cast.f32_to_bf16.ieee_rne.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.cast.f32_to_bf16";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(CAST_F32_TO_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks F32 to BF16 cast capability",
        ));
    }
    let contract = cast_f32_to_bf16_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("cast_f32_to_bf16.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_cast_f32_to_bf16.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.cast.f32_to_bf16").map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        fingerprint,
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([capability]),
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
            return Err(invalid_plan("cast estimator received another operation"));
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
            "cuda.cast_f32_to_bf16.encode",
            "cast requires fixed tensor execution".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.cast_f32_to_bf16.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("cast requires one tensor participant".into());
        };
        if i.operation().id.as_str() != CAST_F32_TO_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("cast requires its selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(2);
        for (role, dtype, size) in [
            (ResolvedValueRole::Input, ElementType::F32, 4),
            (ResolvedValueRole::Output, ElementType::Bf16, 2),
        ] {
            let b = binding(p.bindings(), role, 0)?;
            let [c] = b.storage().components() else {
                return Err("cast requires one component per port".into());
            };
            if c.element_type() != dtype || g.elements().checked_mul(size) != Some(c.length_bytes())
            {
                return Err("cast physical dtype or exact byte extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        dense::disjoint(
            regions[0].device_ptr(),
            regions[0].length_bytes(),
            regions[1].device_ptr(),
            regions[1].length_bytes(),
        )?;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation("vnext_cast_f32_to_bf16", regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let y = regions[1].device_ptr();
            let elements = g.elements();
            let mut builder = stream.launch_builder(kernels.cast_f32_to_bf16_function());
            builder.arg(&x).arg(&y).arg(&elements);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 to BF16 cast launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), g.elements(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<CastF32ToBf16Geometry, String> {
    if b.len() != 2
        || !attrs.is_empty()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("cast requires two dense ports and no attributes".into());
    }
    CastF32ToBf16Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}
