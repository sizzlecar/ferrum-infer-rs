//! Canonical selected-row lookup with explicit source compute-dtype boundary.
use super::super::{tensor_dense_bf16 as dense, tensor_storage};
use super::*;
use ferrum_runtime::{
    comfy_convrot_int8_embedding_f32_contract, ComfyConvRotInt8EmbeddingF32Geometry,
    ComfyEmbeddingDequantizedDtype, PhysicalWeightLayout, WeightFormatId,
    COMFY_CONVROT_INT8_EMBEDDING_F32_CAPABILITY_ID, COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE: &str =
    "cuda.comfy.int8_embedding.scale_before_regular_h256.f32.rn_no_ftz.explicit_source_dtype.v1";
const ESTIMATOR: &str = "resource-estimator.cuda.comfy_convrot_int8_embedding.f32";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(COMFY_CONVROT_INT8_EMBEDDING_F32_CAPABILITY_ID)
        .map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks canonical INT8 ConvRot embedding capability",
        ));
    }
    let contract = comfy_convrot_int8_embedding_f32_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("comfy_embedding.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_comfy_convrot_embedding.cuh"
        )),
    ]);
    let estimator = implementation_fingerprint(&[ESTIMATOR.as_bytes(), fingerprint.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.comfy_convrot_int8_embedding.f32")
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
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
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
                "embedding estimator received another operation",
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
            "cuda.comfy_embedding.encode",
            "embedding requires fixed tensor execution".into(),
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
            .map_err(|e| provider_failure(identity, "cuda.comfy_embedding.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("embedding requires one tensor participant".into());
        };
        if i.operation().id.as_str() != COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("embedding requires its selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(4);
        for (role, ordinal, dtype) in [
            (ResolvedValueRole::Input, 0, ElementType::U32),
            (ResolvedValueRole::Input, 1, ElementType::I8),
            (ResolvedValueRole::Input, 2, ElementType::F32),
            (ResolvedValueRole::Output, 0, ElementType::F32),
        ] {
            let b = binding(p.bindings(), role, ordinal)?;
            let [c] = b.storage().components() else {
                return Err("embedding requires one exact component per port".into());
            };
            if c.element_type() != dtype
                || c.length_bytes()
                    != b.tensor()
                        .minimum_storage_bytes()
                        .map_err(|e| e.to_string())?
            {
                return Err("embedding physical dtype or exact byte extent differs".into());
            }
            regions.push(tensor_storage::component(p, c, 16)?);
        }
        for input in &regions[..3] {
            dense::disjoint(
                input.device_ptr(),
                input.length_bytes(),
                regions[3].device_ptr(),
                regions[3].length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            "vnext_comfy_convrot_int8_embedding_f32",
            regions,
            move |stream, regions| {
                let ids = regions[0].device_ptr();
                let table = regions[1].device_ptr();
                let scales = regions[2].device_ptr();
                let output = regions[3].device_ptr();
                let groups = g.elements() / 256;
                let width = g.width();
                let vocabulary = g.vocabulary();
                let mode =
                    u32::from(g.dequantized_dtype() == ComfyEmbeddingDequantizedDtype::Bf16ThenF32);
                let mut launch = stream.launch_builder(kernels.comfy_convrot_function(8));
                launch
                    .arg(&ids)
                    .arg(&table)
                    .arg(&scales)
                    .arg(&output)
                    .arg(&groups)
                    .arg(&width)
                    .arg(&vocabulary)
                    .arg(&mode);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (groups.min(65535) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| {
                    CudaDeviceRuntimeError::driver("INT8 ConvRot selected-row embedding", e)
                })
            },
        )
        .and_then(|c| c.with_tensor_work(g.elements(), g.logical_arithmetic_operations(), 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<ComfyConvRotInt8EmbeddingF32Geometry, String> {
    if b.len() != 4
        || attrs.len() != 3
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("embedding requires exact dense canonical ports and three attributes".into());
    }
    let mode = attrs
        .iter()
        .find(|(k, _)| k.as_str() == "dequantized_dtype")
        .map(|(_, v)| v);
    let Some(SemanticValue::Text(mode)) = mode else {
        return Err("embedding requires explicit dequantized_dtype".into());
    };
    let mode =
        ComfyEmbeddingDequantizedDtype::parse(mode).ok_or("unsupported embedding source dtype")?;
    ComfyConvRotInt8EmbeddingF32Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Input, 1)?.tensor(),
        binding(b, ResolvedValueRole::Input, 2)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        super::super::unsigned_attribute(attrs, "hidden_size")?,
        super::super::unsigned_attribute(attrs, "vocab_size")?,
        mode,
    )
    .map_err(|e| e.to_string())
}
