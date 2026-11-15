use super::*;
use ferrum_runtime::{
    head_rms_partial_rope_single_round_bf16_contract, HeadRmsPartialRopeSingleRoundBf16Geometry,
    PhysicalWeightLayout, WeightFormatId, HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_CAPABILITY_ID,
    HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID,
};

pub const NUMERICAL_PROFILE: &str =
    "cuda.head_rms_partial_rope.single_round_bf16.block128_warp4_fma.div_full.rsqrt_approx_ftz.v1";
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_CAPABILITY_ID)
        .map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks single-round head RMS/RoPE capability",
        ));
    }
    let c = head_rms_partial_rope_single_round_bf16_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("head_rms_partial_rope_single_round_bf16.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_head_rms_partial_rope_single_round_bf16.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.head_rms_partial_rope.single_round_bf16";
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.head_rms_partial_rope.single_round_bf16")
            .map_err(contract_error)?,
        c.descriptor().id.clone(),
        c.descriptor().fingerprint().map_err(contract_error)?,
        fingerprint.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.id.clone(),
        BTreeSet::from([cap]),
        BTreeSet::from([
            WeightFormatId::new(super::super::DENSE_SAFETENSORS_FORMAT_ID)
                .map_err(contract_error)?,
        ]),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(4),
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
            return Err(invalid_plan(
                "another operation reached single-round RMS/RoPE estimator",
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
            "cuda.single_round_rms_rope.encode",
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
            .map_err(|e| provider_failure(identity, "cuda.single_round_rms_rope.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        i: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [p] = i.participants() else {
            return Err("requires one fixed participant".into());
        };
        if i.operation().id.as_str() != HEAD_RMS_PARTIAL_ROPE_SINGLE_ROUND_BF16_OPERATION_ID
            || !matches!(p.work(), NodeWorkContract::Fixed)
            || !i.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires selected fixed tensor contract".into());
        }
        let g = geometry(p.bindings(), p.attributes())?;
        let mut regions = Vec::with_capacity(5);
        for (role, index) in [
            (ResolvedValueRole::Input, 0),
            (ResolvedValueRole::Input, 1),
            (ResolvedValueRole::Input, 2),
            (ResolvedValueRole::Input, 3),
            (ResolvedValueRole::Output, 0),
        ] {
            let b = binding(p.bindings(), role, index)?;
            regions.push(retained_typed_region(
                p,
                b,
                b.tensor()
                    .minimum_storage_bytes()
                    .map_err(|e| e.to_string())?,
                ElementType::Bf16,
            )?);
        }
        for read in &regions[..4] {
            nonoverlapping_ranges(
                regions[4].device_ptr(),
                regions[4].length_bytes(),
                read.device_ptr(),
                read.length_bytes(),
            )?;
        }
        let elements = g.rows() * g.columns();
        let head_rows = elements / g.head_dim();
        let columns = g.columns();
        let head = g.head_dim();
        let rot = g.rotary_dim();
        let eps = g.effective_epsilon();
        let work = g
            .logical_arithmetic_operations()
            .map_err(|e| e.to_string())?;
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            "vnext_head_rms_partial_rope_single_round_bf16",
            regions,
            move |stream, regions| {
                let x = regions[0].device_ptr();
                let gamma = regions[1].device_ptr();
                let cos = regions[2].device_ptr();
                let sin = regions[3].device_ptr();
                let out = regions[4].device_ptr();
                let mut launch =
                    stream.launch_builder(kernels.head_rms_partial_rope_single_round_function());
                launch
                    .arg(&x)
                    .arg(&gamma)
                    .arg(&cos)
                    .arg(&sin)
                    .arg(&out)
                    .arg(&head_rows)
                    .arg(&columns)
                    .arg(&head)
                    .arg(&rot)
                    .arg(&eps);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (head_rows.min(65535) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|e| CudaDeviceRuntimeError::driver("single-round RMS/RoPE launch", e))
            },
        )
        .and_then(|c| c.with_tensor_work(elements, work, 1, 0))
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<HeadRmsPartialRopeSingleRoundBf16Geometry, String> {
    if values.len() != 5
        || values.iter().any(|v| {
            v.weight()
                .is_some_and(|w| !matches!(w.physical_layout(), PhysicalWeightLayout::Dense { .. }))
        })
    {
        return Err("requires numerical dense operands".into());
    }
    HeadRmsPartialRopeSingleRoundBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Input, 1)?.tensor(),
        binding(values, ResolvedValueRole::Input, 2)?.tensor(),
        binding(values, ResolvedValueRole::Input, 3)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())
}
