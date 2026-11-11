//! Independent fused RMS/partial rotation on the existing admitted CUDA lane.
use super::*;
use ferrum_runtime::{
    fused_head_rms_partial_rope_bf16_contract, FusedHeadRmsPartialRopeBf16Geometry,
    PhysicalWeightLayout, WeightFormatId, FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_CAPABILITY_ID,
};
pub const NUMERICAL_PROFILE_ID: &str =
    "cuda.ieee.fused_head_rms_partial_rope.bf16.warp32_precise_single_norm.v1";
pub(super) const FUNCTION: &str = "ferrum_vnext_fused_head_rms_partial_rope_bf16";
const IDENTITY: &str = concat!(
    "warp32-FMA-squares-stride32-shfl-down16-8-4-2-1;",
    "RN32-mean-epsilon-sqrt-reciprocal;B(RN32(RN32(x*r)*gamma));",
    "F32-two-product-pair-separate-add-then-B;BF16-per-row-cos-sin;normalized-tail;",
    "NoAlias;no-global-scratch;no-new-runtime;gradual-underflow;NaN-class-only;",
    "not-upstream-fast-math-rsqrt-or-Torch-binary-parity;requires-target-validation"
);
pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let cap = CapabilityId::new(FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_CAPABILITY_ID)
        .map_err(contract_error)?;
    if !device.capabilities.contains(&cap) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA lacks fused head RMS/RoPE capability",
        ));
    }
    let contract = fused_head_rms_partial_rope_bf16_contract().map_err(contract_error)?;
    let fingerprint = implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        IDENTITY.as_bytes(),
        numerical_fingerprint().as_bytes(),
        include_bytes!("fused_head_rms_partial_rope.rs"),
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/vnext_fused_head_rms_partial_rope.cuh"
        )),
    ]);
    let estimator = "resource-estimator.cuda.fused_head_rms_partial_rope.bf16";
    OperationProviderDescriptor::new(
        ProviderId::new("provider.cuda.fused_head_rms_partial_rope.bf16")
            .map_err(contract_error)?,
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
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id != *self.descriptor.operation_id()
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "another operation reached fused RMS/RoPE estimator",
            ));
        }
        geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
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
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        Err(provider_failure(
            invocation.participants()[0].identity().clone(),
            "cuda.fused_rms_rope.encode",
            "requires fixed tensor invocation".into(),
        ))
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<CudaDeviceCommand>>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        self.encode(invocation)
            .map(EncodedDeviceOperation::compute)
            .map(Some)
            .map_err(|e| provider_failure(identity, "cuda.fused_rms_rope.encode", e))
    }
}
impl Provider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("requires one participant".into());
        };
        if invocation.operation().id != *self.descriptor.operation_id()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("requires its fixed tensor contract".into());
        }
        let g = geometry(participant.bindings(), participant.attributes())?;
        let elements = g.rows() * g.columns();
        let heads = elements / g.head_dim();
        let mut regions = Vec::with_capacity(5);
        for (ordinal, bytes) in [
            elements * 2,
            g.head_dim() * 2,
            g.rows() * g.rotary_dim() * 2,
            g.rows() * g.rotary_dim() * 2,
        ]
        .into_iter()
        .enumerate()
        {
            regions.push(retained_region(
                participant,
                binding(
                    participant.bindings(),
                    ResolvedValueRole::Input,
                    ordinal as u32,
                )?,
                bytes,
            )?);
        }
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            elements * 2,
        )?;
        for read in &regions {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        regions.push(output);
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(FUNCTION, regions, move |stream, regions| {
            let x = regions[0].device_ptr();
            let gamma = regions[1].device_ptr();
            let cos = regions[2].device_ptr();
            let sin = regions[3].device_ptr();
            let y = regions[4].device_ptr();
            let columns = g.columns();
            let head = g.head_dim();
            let rotary = g.rotary_dim();
            let epsilon = g.effective_epsilon();
            let mut builder = stream.launch_builder(kernels.fused_head_rms_partial_rope_function());
            builder
                .arg(&x)
                .arg(&gamma)
                .arg(&cos)
                .arg(&sin)
                .arg(&y)
                .arg(&heads)
                .arg(&columns)
                .arg(&head)
                .arg(&rotary)
                .arg(&epsilon);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (heads.div_ceil(4).min(MAX_BLOCKS) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(|e| CudaDeviceRuntimeError::driver("fused head RMS/partial RoPE launch", e))
        })
        .and_then(|command| {
            command.with_tensor_work(
                elements,
                g.logical_arithmetic_operations().map_err(contract_error)?,
                1,
                0,
            )
        })
        .map_err(|e| e.to_string())
    }
}
fn geometry(
    bindings: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<FusedHeadRmsPartialRopeBf16Geometry, String> {
    if bindings.len() != 5 || attrs.len() != 3 {
        return Err("requires four inputs, one output and head_dim/rotary_dim/epsilon".into());
    }
    let get = |name| {
        attrs
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, value)| value)
    };
    let unsigned = |name| match get(name) {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("requires unsigned {name}")),
    };
    let Some(SemanticValue::Rational(epsilon)) = get("epsilon") else {
        return Err("requires rational epsilon".into());
    };
    let g = FusedHeadRmsPartialRopeBf16Geometry::new(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 3)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        unsigned("head_dim")?,
        unsigned("rotary_dim")?,
        *epsilon,
    )
    .map_err(|e| e.to_string())?;
    for binding in bindings {
        if binding.weight().is_some_and(|weight| {
            !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
        }) {
            return Err(
                "fused RMS/RoPE cannot consume packed or quantized physical weights".into(),
            );
        }
        if binding
            .tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?
            > isize::MAX as u64
        {
            return Err("operand exceeds address space".into());
        }
    }
    Ok(g)
}
