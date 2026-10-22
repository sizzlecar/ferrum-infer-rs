//! Half-split rotary with row-specific BF16 cos/sin and materialized BF16
//! multiplication boundaries. This is distinct from adjacent-pair F32 FMA.

use super::*;
use ferrum_runtime::{
    half_split_rope_bf16_contract, HalfSplitRopeBf16Geometry, ResolvedTensorSpec,
    HALF_SPLIT_ROPE_BF16_CAPABILITY_ID, HALF_SPLIT_ROPE_BF16_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.ieee.half_split_rope.bf16.staged.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "half-split-rotate=(-second,first);x-y-BF16[R,C];cos-sin-BF16[R,D];",
    "frequency-per-row-shared-across-heads;no-half-equality-assumption;",
    "Y=B(RN32(decode(B(RN32(x*cos)))+decode(B(RN32(rotate_half(x)*sin)))));",
    "separate-fmul-rn-and-fadd-rn;no-fma;integer-BF16-RNE;gradual-underflow;",
    "NaN-class-only;NoAlias;no-frequency-generation;zero-global-scratch;",
    "logical-work=7-per-pair(four-mul,two-add,one-negation);",
    "target-validation-required;no-full-model-or-Torch-parity-claim"
);
pub(super) const FUNCTION: &str = "ferrum_vnext_half_split_rope_bf16";
const PROVIDER_ID: &str = "provider.cuda.half_split_rope.bf16.staged";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.half_split_rope.bf16.staged";

pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(HALF_SPLIT_ROPE_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks half-split BF16 RoPE capability",
        ));
    }
    let contract = half_split_rope_bf16_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        PROVIDER_ID.as_bytes(),
    ]);
    let estimator =
        implementation_fingerprint(&[ESTIMATOR_ID.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
        contract.descriptor().id.clone(),
        contract
            .descriptor()
            .fingerprint()
            .map_err(contract_error)?,
        implementation,
        ProviderExecutionSemantics::bitwise_eager_only(),
        contract.descriptor().version,
        device.id.clone(),
        BTreeSet::from([capability]),
        BTreeSet::new(),
        BTreeSet::new(),
        super::super::transformer::contiguous_bindings(3),
        ESTIMATOR_ID,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)
    .map(|descriptor| descriptor.with_fixed_tensor_work(true))
}

pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(HalfSplitRopeProvider {
        descriptor,
        kernels: Arc::clone(kernels),
    })
}

struct HalfSplitRopeProvider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}

impl OperationResourceEstimator for HalfSplitRopeProvider {
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
                "CUDA half-split RoPE estimator received another operation",
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

impl OperationProvider<CudaDeviceRuntime> for HalfSplitRopeProvider {
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
            "cuda.half_split_rope.bf16.encode",
            "CUDA half-split RoPE requires fixed tensor work".into(),
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
            .map_err(|message| {
                provider_failure(identity, "cuda.half_split_rope.bf16.encode", message)
            })
    }
}

impl HalfSplitRopeProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA half-split RoPE requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != HALF_SPLIT_ROPE_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA half-split RoPE requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let input = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.bytes,
        )?;
        let cosine = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            shape.frequency_bytes,
        )?;
        let sine = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            shape.frequency_bytes,
        )?;
        let output = retained_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.bytes,
        )?;
        for read in [&input, &cosine, &sine] {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        CudaDeviceCommand::operation(
            FUNCTION,
            vec![input, cosine, sine, output],
            move |stream, regions| {
                let input = regions[0].device_ptr();
                let cosine = regions[1].device_ptr();
                let sine = regions[2].device_ptr();
                let output = regions[3].device_ptr();
                let mut builder = stream.launch_builder(kernels.half_split_rope_function());
                builder
                    .arg(&input)
                    .arg(&cosine)
                    .arg(&sine)
                    .arg(&output)
                    .arg(&shape.pairs)
                    .arg(&shape.columns)
                    .arg(&shape.head_dim);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (
                            shape.pairs.div_ceil(u64::from(THREADS)).min(MAX_BLOCKS) as u32,
                            1,
                            1,
                        ),
                        block_dim: (THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map(|_| ())
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("BF16 half-split RoPE launch", error)
                })
            },
        )
        .and_then(|command| command.with_tensor_work(shape.elements, shape.operations, 1, 0))
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    columns: u64,
    head_dim: u64,
    elements: u64,
    pairs: u64,
    bytes: u64,
    frequency_bytes: u64,
    operations: u64,
}

impl Shape {
    fn new(checked: HalfSplitRopeBf16Geometry) -> Result<Self, String> {
        let elements = checked
            .rows()
            .checked_mul(checked.columns())
            .ok_or("half-split RoPE element count overflows")?;
        let bytes = elements
            .checked_mul(2)
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("half-split RoPE BF16 byte extent exceeds address space")?;
        let frequency_bytes = checked
            .rows()
            .checked_mul(checked.head_dim())
            .and_then(|n| n.checked_mul(2))
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("half-split RoPE BF16 frequency byte extent exceeds address space")?;
        let pairs = elements / 2;
        let operations = pairs
            .checked_mul(7)
            .ok_or("half-split RoPE logical work overflows")?;
        Ok(Self {
            columns: checked.columns(),
            head_dim: checked.head_dim(),
            elements,
            pairs,
            bytes,
            frequency_bytes,
            operations,
        })
    }
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != 4 || attributes.len() != 1 {
        return Err(
            "CUDA half-split RoPE requires three inputs, one output and only head_dim".into(),
        );
    }
    let Some(SemanticValue::Unsigned(head_dim)) = attributes
        .iter()
        .find(|(key, _)| key.as_str() == "head_dim")
        .map(|(_, value)| value)
    else {
        return Err("CUDA half-split RoPE requires unsigned head_dim".into());
    };
    validate_specs(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        *head_dim,
    )
}

fn validate_specs(
    input: &ResolvedTensorSpec,
    cosine: &ResolvedTensorSpec,
    sine: &ResolvedTensorSpec,
    output: &ResolvedTensorSpec,
    head_dim: u64,
) -> Result<Shape, String> {
    Shape::new(
        HalfSplitRopeBf16Geometry::new(input, cosine, sine, output, head_dim)
            .map_err(|error| error.to_string())?,
    )
}

#[cfg(test)]
mod tests;
