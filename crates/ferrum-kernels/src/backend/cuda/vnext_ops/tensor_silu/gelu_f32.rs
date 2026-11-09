//! Full-F32 GELU over managed CUDA bindings. The precise libdevice profile is
//! independent of both the portable libm provider and an upstream Torch binary.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{GeluF32Geometry, GeluF32Variant, PhysicalWeightLayout, WeightFormatId};

pub const TANH_NUMERICAL_PROFILE: &str = "cuda.gelu_tanh.f32.libdevice_precise.separate.gradual.v1";
pub const ERF_NUMERICAL_PROFILE: &str = "cuda.gelu_erf.f32.libdevice_precise.separate.gradual.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_gelu_f32";
const ESTIMATOR: &str = "resource-estimator.cuda.gelu_f32";

pub(super) fn descriptor(
    device: &DeviceDescriptor,
    variant: GeluF32Variant,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(variant.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 GELU capability",
        ));
    }
    let contract = variant.contract().map_err(contract_error)?;
    let profile = match variant {
        GeluF32Variant::Tanh => TANH_NUMERICAL_PROFILE,
        GeluF32Variant::Erf => ERF_NUMERICAL_PROFILE,
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("gelu_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_gelu_f32.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}", variant.operation_id()))
            .map_err(contract_error)?,
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
    variant: GeluF32Variant,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(Provider {
        descriptor,
        kernels: Arc::clone(kernels),
        variant,
    })
}
struct Provider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    variant: GeluF32Variant,
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
                "F32 GELU estimator received another operation",
            ));
        }
        geometry(r.values(), r.attributes())
            .map_err(invalid_plan)?
            .logical_operations(self.variant)?;
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
            "cuda.gelu_f32.encode",
            "F32 GELU requires fixed tensors".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.gelu_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("F32 GELU requires one tensor participant".into());
        };
        if i.operation().id.as_str() != self.variant.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("F32 GELU requires its selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let work = g
            .logical_operations(self.variant)
            .map_err(|e| e.to_string())?;
        let mut regions = Vec::with_capacity(2);
        for role in [ResolvedValueRole::Input, ResolvedValueRole::Output] {
            let b = binding(p.bindings(), role, 0)?;
            let [component] = b.storage().components() else {
                return Err("F32 GELU requires one physical component per port".into());
            };
            if component.element_type() != ElementType::F32
                || g.elements().checked_mul(4) != Some(component.length_bytes())
            {
                return Err("F32 GELU physical dtype or exact byte extent differs".into());
            }
            regions.push(tensor_storage::component(p, component, 16)?);
        }
        dense::disjoint(
            regions[0].device_ptr(),
            regions[0].length_bytes(),
            regions[1].device_ptr(),
            regions[1].length_bytes(),
        )?;
        let kernels = Arc::clone(&self.kernels);
        let (name, variant) = match self.variant {
            GeluF32Variant::Tanh => ("vnext_gelu_tanh_f32", 0i32),
            GeluF32Variant::Erf => ("vnext_gelu_erf_f32", 1i32),
        };
        CudaDeviceCommand::operation(name, regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let y = regions[1].device_ptr();
            let elements = g.elements();
            let mut builder = stream.launch_builder(kernels.gelu_f32_function());
            builder.arg(&x).arg(&y).arg(&elements).arg(&variant);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 GELU launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), work, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<GeluF32Geometry, String> {
    if values.len() != 2
        || !attributes.is_empty()
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 GELU requires two dense ports and no attributes".into());
    }
    GeluF32Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}
