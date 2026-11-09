//! Full-F32 two-pass centered normalization with bounded static CTA state.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    layer_norm_f32_contract, LayerNormF32Geometry, PhysicalWeightLayout, WeightFormatId,
    LAYER_NORM_F32_CAPABILITY_ID, LAYER_NORM_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.layer_norm.f32.cta256_two_pass.separate_affine.rn_sqrt_div.no_ftz.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_layer_norm_f32";
const IDENTITY:&str="F32-X/Y[R,H];F32-gamma/beta[H];H1..2^24;lane256-strided-ascending-sum;tree128..1;mean=RN32(sum/H);second-pass=RN32(RN32(x-mean)^2);population-variance;inverse=RN32(1/sqrt(RN32(var+eps)));Y=RN32(RN32(RN32(x-mean)*inverse)*gamma)+beta-separate-RN32;no-FMA-no-half-no-Welford;gradual-underflow;static-shared-256F32+2F32;zero-global-scratch;NoAlias;IEEE-nonfinite;not-Torch-binary-parity;target-qualification-required";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(LAYER_NORM_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "runtime lacks F32 LayerNorm capability",
        ));
    }
    let c = layer_norm_f32_contract().map_err(contract_error)?;
    let fp = implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        IDENTITY.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("layer_norm_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_layer_norm_f32.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.layer_norm.f32";
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.layer_norm.f32").map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fp.clone(),
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
        super::super::transformer::contiguous_bindings(3),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fp.as_bytes()]),
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
                "F32 LayerNorm estimator received another operation",
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
            "cuda.layer_norm_f32.encode",
            "F32 LayerNorm requires fixed tensor work".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.layer_norm_f32.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("F32 LayerNorm requires one participant".into());
        };
        if i.operation().id.as_str() != LAYER_NORM_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires F32 LayerNorm fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(4);
        for (role, n) in [
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Input, 2),
            (ResolvedValueRole::Output, 0),
        ] {
            let b = binding(p.bindings(), role, n)?;
            let [c] = b.storage().components() else {
                return Err("requires one exact F32 component per port".into());
            };
            if c.element_type() != ElementType::F32
                || c.length_bytes()
                    != b.tensor()
                        .minimum_storage_bytes()
                        .map_err(|e| e.to_string())?
            {
                return Err("physical dtype or byte extent differs from F32 contract".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        for r in &regions[..3] {
            dense::disjoint(
                r.device_ptr(),
                r.length_bytes(),
                regions[3].device_ptr(),
                regions[3].length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(FUNCTION, regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let gamma = regions[1].device_ptr();
            let beta = regions[2].device_ptr();
            let y = regions[3].device_ptr();
            let rows = g.rows();
            let width = g.columns() as u32;
            let epsilon = g.effective_epsilon();
            let mut launch = stream.launch_builder(kernels.layer_norm_f32_function());
            launch
                .arg(&x)
                .arg(&gamma)
                .arg(&beta)
                .arg(&y)
                .arg(&rows)
                .arg(&width)
                .arg(&epsilon);
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (rows.min(65535) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("F32 LayerNorm launch", e))
        })
        .and_then(|c| c.with_tensor_work(g.elements(), g.logical_arithmetic_operations(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<LayerNormF32Geometry, String> {
    if b.len() != 4
        || attrs.len() != 1
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exact dense F32 LayerNorm operands and epsilon".into());
    }
    let Some(SemanticValue::Rational(epsilon)) = attrs
        .iter()
        .find(|(k, _)| k.as_str() == "epsilon")
        .map(|(_, v)| v)
    else {
        return Err("requires rational epsilon".into());
    };
    let x = |n| binding(b, ResolvedValueRole::Input, n).map(|v| v.tensor());
    LayerNormF32Geometry::new(
        x(0)?,
        x(1)?,
        x(2)?,
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        *epsilon,
    )
    .map_err(|e| e.to_string())
}
