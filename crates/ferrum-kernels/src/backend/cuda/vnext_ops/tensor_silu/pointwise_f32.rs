//! Two ordinary F32 boundaries over admitted storage; no quantization or half
//! conversion. The libdevice exponential profile is qualified independently.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{F32PointwiseOp, F32SameShapeGeometry, PhysicalWeightLayout, WeightFormatId};

pub const RESIDUAL_NUMERICAL_PROFILE: &str = "cuda.plain_residual_add.f32.rne.gradual.v1";
pub const SILU_MUL_NUMERICAL_PROFILE: &str =
    "cuda.silu_mul.f32.libdevice_precise.separate.gradual.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.pointwise_f32";

pub(super) fn descriptor(
    device: &DeviceDescriptor,
    kind: F32PointwiseOp,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(kind.capability_id()).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 pointwise capability",
        ));
    }
    let contract = kind.contract().map_err(contract_error)?;
    let profile = match kind {
        F32PointwiseOp::ResidualAdd => RESIDUAL_NUMERICAL_PROFILE,
        F32PointwiseOp::SiluMul => SILU_MUL_NUMERICAL_PROFILE,
    };
    let fingerprint = implementation_fingerprint(&[
        profile.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("pointwise_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_pointwise_f32.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(format!("provider.cuda.{}", kind.operation_id()))
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
    kind: F32PointwiseOp,
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
    kind: F32PointwiseOp,
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
                "F32 pointwise estimator received another operation",
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
            "cuda.pointwise_f32.encode",
            "F32 pointwise requires fixed tensors".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.pointwise_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("F32 pointwise requires one tensor participant".into());
        };
        if i.operation().id.as_str() != self.kind.operation_id()
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("F32 pointwise requires its selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(3);
        for (role, index) in [
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Output, 0),
        ] {
            let b = binding(p.bindings(), role, index)?;
            let [component] = b.storage().components() else {
                return Err("F32 pointwise requires one physical component per port".into());
            };
            if component.element_type() != ElementType::F32
                || g.elements().checked_mul(4) != Some(component.length_bytes())
            {
                return Err("F32 pointwise physical dtype or exact byte extent differs".into());
            }
            regions.push(tensor_storage::component(p, component, 16)?);
        }
        for input in &regions[..2] {
            dense::disjoint(
                input.device_ptr(),
                input.length_bytes(),
                regions[2].device_ptr(),
                regions[2].length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        let kind = self.kind;
        let (name, operations) = match kind {
            F32PointwiseOp::ResidualAdd => ("vnext_plain_residual_add_f32", 1u64),
            F32PointwiseOp::SiluMul => ("vnext_silu_mul_f32", 5),
        };
        let work = g
            .elements()
            .checked_mul(operations)
            .ok_or("F32 pointwise logical work overflows")?;
        CudaDeviceCommand::operation(name, regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let update = regions[1].device_ptr();
            let output = regions[2].device_ptr();
            let elements = g.elements();
            let kind = match kind {
                F32PointwiseOp::ResidualAdd => 0i32,
                F32PointwiseOp::SiluMul => 1,
            };
            let mut builder = stream.launch_builder(kernels.pointwise_f32_function());
            builder
                .arg(&x)
                .arg(&update)
                .arg(&output)
                .arg(&elements)
                .arg(&kind);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 pointwise launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), work, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<F32SameShapeGeometry, String> {
    if b.len() != 3
        || !attrs.is_empty()
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("F32 pointwise requires three dense ports and no attributes".into());
    }
    F32SameShapeGeometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Input, 1)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
    )
    .map_err(|e| e.to_string())
}
