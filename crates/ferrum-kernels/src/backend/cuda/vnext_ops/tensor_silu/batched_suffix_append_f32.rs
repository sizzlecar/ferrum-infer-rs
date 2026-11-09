use super::*;
use ferrum_runtime::{
    batched_suffix_append_f32_contract, BatchedSuffixAppendF32Geometry, PhysicalWeightLayout,
    WeightFormatId, BATCHED_SUFFIX_APPEND_F32_CAPABILITY_ID,
    BATCHED_SUFFIX_APPEND_F32_OPERATION_ID,
};
pub const NUMERICAL_PROFILE_ID: &str = "cuda.batched_suffix_append.f32.exact_bits.positive_zero.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_batched_suffix_append_f32";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(BATCHED_SUFFIX_APPEND_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks F32 suffix append",
        ));
    }
    let contract = batched_suffix_append_f32_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("batched_suffix_append_f32.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_batched_suffix_append_f32.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.batched_suffix_append.f32";
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.batched_suffix_append.f32").map_err(contract_error)?,
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
            WeightFormatId::new("weight-format.safetensors.dense").map_err(contract_error)?,
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(2),
        estimator,
        ContractVersion::new(1, 0),
        implementation_fingerprint(&[estimator.as_bytes(), fingerprint.as_bytes()]),
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
            return Err(invalid_plan("suffix estimator received another operation"));
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
            "cuda.batched_suffix_append.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.batched_suffix_append.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("suffix append requires one participant".into());
        };
        if i.operation().id.as_str() != BATCHED_SUFFIX_APPEND_F32_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed suffix append".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let input = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Input, 0)?,
            g.input_elements() * 4,
            ElementType::F32,
        )?;
        let suffix = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Input, 1)?,
            g.suffix_elements() * 4,
            ElementType::F32,
        )?;
        let output = retained_typed_region(
            p,
            binding(p.bindings(), ResolvedValueRole::Output, 0)?,
            g.elements() * 4,
            ElementType::F32,
        )?;
        for r in [&input, &suffix] {
            nonoverlapping_ranges(
                r.device_ptr(),
                r.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        let elements = g.elements();
        let columns = g.columns();
        let patch_rows = g.patch_rows();
        let suffix_rows = g.suffix_rows();
        let zero_rows = g.zero_rows();
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            FUNCTION,
            vec![input, suffix, output],
            move |stream, regions| {
                let input = regions[0].device_ptr();
                let suffix = regions[1].device_ptr();
                let output = regions[2].device_ptr();
                let mut launch =
                    stream.launch_builder(kernels.batched_suffix_append_f32_function());
                launch
                    .arg(&input)
                    .arg(&suffix)
                    .arg(&output)
                    .arg(&elements)
                    .arg(&columns)
                    .arg(&patch_rows)
                    .arg(&suffix_rows)
                    .arg(&zero_rows);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (elements.div_ceil(256).min(MAX_BLOCKS) as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("F32 suffix append", e))
            },
        )
        .and_then(|c| c.with_tensor_work(elements, 0, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    b: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<BatchedSuffixAppendF32Geometry, String> {
    if b.len() != 3
        || attrs.len() != 2
        || b.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires exactly three dense F32 ports and two attributes".into());
    }
    let read = |name: &str| -> Result<u64, String> {
        match attrs.get(&AttributeId::new(name).map_err(|e| e.to_string())?) {
            Some(SemanticValue::Unsigned(v)) => Ok(*v),
            _ => Err("missing unsigned suffix attribute".into()),
        }
    };
    BatchedSuffixAppendF32Geometry::new(
        binding(b, ResolvedValueRole::Input, 0)?.tensor(),
        binding(b, ResolvedValueRole::Input, 1)?.tensor(),
        binding(b, ResolvedValueRole::Output, 0)?.tensor(),
        read("batch")?,
        read("zero_rows")?,
    )
    .map_err(|e| e.to_string())
}
