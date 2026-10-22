//! Adjacent-pair BF16 rotary operation with explicit F32 frequency storage.
//! All source values and frequencies remain in their admitted allocations.

use super::*;
use ferrum_runtime::{
    adjacent_rope_bf16_contract, AdjacentRopeBf16Geometry, ResolvedTensorSpec,
    ADJACENT_ROPE_BF16_CAPABILITY_ID, ADJACENT_ROPE_BF16_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.ieee.adjacent_rope.bf16.fma.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "adjacent-pairs;input-output-BF16;frequency-F32[S,D/2,2];",
    "row-mod-sequence;heads-and-batches-share-frequency;",
    "real=B(FMA_RN(a,c,-MUL_RN(b,s)));imag=B(FMA_RN(b,c,MUL_RN(a,s)));",
    "explicit-fmaf-rn;explicit-fmul-rn;integer-BF16-RNE;gradual-underflow;",
    "fmad=false-for-implicit-contraction;NaN-class-only;NoAlias;",
    "no-frequency-generation-or-unit-magnitude-assumption;zero-global-scratch;",
    "logical-work=7-per-pair(two-mul,two-fma-as-four-arithmetic,one-negation);",
    "target-validation-required;no-full-model-or-Torch-parity-claim"
);
pub(super) const FUNCTION: &str = "ferrum_vnext_adjacent_rope_bf16";
const PROVIDER_ID: &str = "provider.cuda.adjacent_rope.bf16.fma";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.adjacent_rope.bf16.fma";

pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability = CapabilityId::new(ADJACENT_ROPE_BF16_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks adjacent BF16 RoPE capability",
        ));
    }
    let contract = adjacent_rope_bf16_contract().map_err(contract_error)?;
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
        super::super::transformer::contiguous_bindings(2),
        ESTIMATOR_ID,
        ContractVersion::new(1, 0),
        estimator,
    )
    .map_err(contract_error)
    .map(|d| d.with_fixed_tensor_work(true))
}

pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(RopeProvider {
        descriptor,
        kernels: Arc::clone(kernels),
    })
}

struct RopeProvider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}

impl OperationResourceEstimator for RopeProvider {
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
                "CUDA adjacent RoPE estimator received another operation",
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

impl OperationProvider<CudaDeviceRuntime> for RopeProvider {
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
            "cuda.adjacent_rope.bf16.encode",
            "CUDA adjacent RoPE requires fixed tensor work".into(),
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
                provider_failure(identity, "cuda.adjacent_rope.bf16.encode", message)
            })
    }
}

impl RopeProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA adjacent RoPE requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != ADJACENT_ROPE_BF16_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA adjacent RoPE requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let input = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.bytes,
            ElementType::Bf16,
        )?;
        let frequencies = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            shape.frequency_bytes,
            ElementType::F32,
        )?;
        let output = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.bytes,
            ElementType::Bf16,
        )?;
        for read in [&input, &frequencies] {
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
            vec![input, frequencies, output],
            move |stream, regions| {
                let input = regions[0].device_ptr();
                let frequencies = regions[1].device_ptr();
                let output = regions[2].device_ptr();
                let mut builder = stream.launch_builder(kernels.rope_function());
                builder
                    .arg(&input)
                    .arg(&frequencies)
                    .arg(&output)
                    .arg(&shape.pairs)
                    .arg(&shape.columns)
                    .arg(&shape.sequence)
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
                .map_err(|error| CudaDeviceRuntimeError::driver("BF16 adjacent RoPE launch", error))
            },
        )
        .and_then(|command| command.with_tensor_work(shape.elements, shape.operations, 1, 0))
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    columns: u64,
    sequence: u64,
    head_dim: u64,
    elements: u64,
    pairs: u64,
    bytes: u64,
    frequency_bytes: u64,
    operations: u64,
}

impl Shape {
    fn new(checked: AdjacentRopeBf16Geometry) -> Result<Self, String> {
        let elements = checked
            .rows()
            .checked_mul(checked.columns())
            .ok_or("RoPE element count overflows")?;
        let bytes = elements
            .checked_mul(2)
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("RoPE BF16 byte extent exceeds address space")?;
        let frequency_bytes = checked
            .sequence()
            .checked_mul(checked.head_dim())
            .and_then(|n| n.checked_mul(4))
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("RoPE F32 frequency byte extent exceeds address space")?;
        let pairs = elements / 2;
        let operations = pairs
            .checked_mul(7)
            .ok_or("RoPE logical arithmetic work overflows")?;
        Ok(Self {
            columns: checked.columns(),
            sequence: checked.sequence(),
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
    if bindings.len() != 3 || attributes.len() != 1 {
        return Err("CUDA adjacent RoPE requires two inputs, one output and only head_dim".into());
    }
    let Some(SemanticValue::Unsigned(head_dim)) = attributes
        .iter()
        .find(|(key, _)| key.as_str() == "head_dim")
        .map(|(_, value)| value)
    else {
        return Err("CUDA adjacent RoPE requires unsigned head_dim".into());
    };
    validate_specs(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        *head_dim,
    )
}

fn validate_specs(
    input: &ResolvedTensorSpec,
    frequencies: &ResolvedTensorSpec,
    output: &ResolvedTensorSpec,
    head_dim: u64,
) -> Result<Shape, String> {
    Shape::new(
        AdjacentRopeBf16Geometry::new(input, frequencies, output, head_dim)
            .map_err(|e| e.to_string())?,
    )
}

#[cfg(test)]
mod tests;
