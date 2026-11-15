//! Packed SwiGLU using admitted owners and the existing shared CUDA module.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    packed_swiglu_bf16_contract, PackedSwiGluBf16Geometry, PackedSwiGluBf16Profile,
    PhysicalWeightLayout, WeightFormatId, PACKED_SWIGLU_BF16_CAPABILITY_ID,
    PACKED_SWIGLU_BF16_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str = "cuda.packed_swiglu.bf16.exp2-single-round-or-staged-silu.v2";
const ESTIMATOR: &str = "resource-estimator.cuda.packed_swiglu.bf16";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(PACKED_SWIGLU_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks packed BF16 SwiGLU capability",
        ));
    }
    let contract = packed_swiglu_bf16_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("packed_swiglu_bf16.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_packed_swiglu_bf16.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.packed_swiglu.bf16").map_err(contract_error)?,
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
            return Err(invalid_plan(
                "packed SwiGLU estimator received another operation",
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
            "cuda.packed_swiglu_bf16.encode",
            "packed SwiGLU requires fixed tensor execution".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.packed_swiglu_bf16.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("packed SwiGLU requires one tensor participant".into());
        };
        if i.operation().id.as_str() != PACKED_SWIGLU_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("packed SwiGLU requires its selected fixed tensor contract".into());
        }
        let (g, profile) = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(2);
        for (role, dtype, size) in [
            (ResolvedValueRole::Input, ElementType::Bf16, 4),
            (ResolvedValueRole::Output, ElementType::Bf16, 2),
        ] {
            let b = binding(p.bindings(), role, 0)?;
            let [c] = b.storage().components() else {
                return Err("packed SwiGLU requires one component per port".into());
            };
            if c.element_type() != dtype || g.elements().checked_mul(size) != Some(c.length_bytes())
            {
                return Err("packed SwiGLU physical dtype or exact byte extent differs".into());
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
        CudaDeviceCommand::operation(
            "vnext_packed_swiglu_bf16",
            regions,
            move |stream, regions| {
                let x = regions[0].device_ptr();
                let y = regions[1].device_ptr();
                let elements = g.elements();
                let width = g.width();
                let staged = u32::from(profile == PackedSwiGluBf16Profile::SiluBf16ThenMultiply);
                let mut builder = stream.launch_builder(kernels.packed_swiglu_bf16_function());
                builder
                    .arg(&x)
                    .arg(&y)
                    .arg(&elements)
                    .arg(&width)
                    .arg(&staged);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("packed BF16 SwiGLU launch", e))
            },
        )
        .and_then(|c| c.with_tensor_work(g.elements(), g.arithmetic_operations(profile), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<(PackedSwiGluBf16Geometry, PackedSwiGluBf16Profile), String> {
    let profile = PackedSwiGluBf16Profile::from_attributes(attrs).map_err(|e| e.to_string())?;
    if b.len() != 2
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("packed SwiGLU requires two dense ports".into());
    }
    PackedSwiGluBf16Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map(|g| (g, profile))
    .map_err(|e| e.to_string())
}
