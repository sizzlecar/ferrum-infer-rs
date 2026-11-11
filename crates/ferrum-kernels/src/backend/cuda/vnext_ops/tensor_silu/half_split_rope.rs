//! Half-split rotary with row-specific BF16 cos/sin and materialized BF16
//! multiplication boundaries. This is distinct from adjacent-pair F32 FMA.

use super::*;
use ferrum_runtime::{
    half_split_rope_bf16_contract, partial_half_split_rope_bf16_contract,
    HalfSplitRopeBf16Geometry, PartialHalfSplitRopeBf16Geometry, ResolvedTensorSpec,
    HALF_SPLIT_ROPE_BF16_CAPABILITY_ID, HALF_SPLIT_ROPE_BF16_OPERATION_ID,
    PARTIAL_HALF_SPLIT_ROPE_BF16_CAPABILITY_ID, PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.ieee.half_split_rope.bf16.staged.v1";
pub const PARTIAL_NUMERICAL_PROFILE_ID: &str = "cuda.ieee.partial_half_split_rope.bf16.staged.v1";
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
    descriptor_kind(device, false)
}
pub(super) fn partial_descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    descriptor_kind(device, true)
}
fn descriptor_kind(
    device: &DeviceDescriptor,
    partial: bool,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let (contract, capability, provider_id, estimator_id, profile) = if partial {
        (
            partial_half_split_rope_bf16_contract(),
            PARTIAL_HALF_SPLIT_ROPE_BF16_CAPABILITY_ID,
            "provider.cuda.partial_half_split_rope.bf16.staged",
            "resource-estimator.cuda.partial_half_split_rope.bf16.staged",
            PARTIAL_NUMERICAL_PROFILE_ID,
        )
    } else {
        (
            half_split_rope_bf16_contract(),
            HALF_SPLIT_ROPE_BF16_CAPABILITY_ID,
            PROVIDER_ID,
            ESTIMATOR_ID,
            NUMERICAL_PROFILE_ID,
        )
    };
    let capability = CapabilityId::new(capability).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks half-split BF16 RoPE capability",
        ));
    }
    let contract = contract.map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        profile.as_bytes(),
        NUMERICAL_IDENTITY.as_bytes(),
        provider_id.as_bytes(),
        if partial {
            b"leading-rotary-dim;tail-u16-bit-copy;work=7-per-rotated-pair"
        } else {
            b"full-head"
        },
    ]);
    let estimator =
        implementation_fingerprint(&[estimator_id.as_bytes(), implementation.as_bytes()]);
    OperationProviderDescriptor::new(
        ProviderId::new(provider_id).map_err(contract_error)?,
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
        estimator_id,
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
        if self.partial() {
            partial_geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        } else {
            geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        }
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
    fn partial(&self) -> bool {
        self.descriptor.operation_id().as_str() == PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID
    }
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA half-split RoPE requires one tensor participant".into());
        };
        if invocation.operation().id != *self.descriptor.operation_id()
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA half-split RoPE requires its fixed tensor contract".into());
        }
        let shape = if self.partial() {
            partial_geometry(participant.bindings(), participant.attributes())?
        } else {
            geometry(participant.bindings(), participant.attributes())?
        };
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
                    .arg(&shape.launch_units)
                    .arg(&shape.columns)
                    .arg(&shape.head_dim)
                    .arg(&shape.rotary_dim);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (
                            shape
                                .launch_units
                                .div_ceil(u64::from(THREADS))
                                .min(MAX_BLOCKS) as u32,
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
    rotary_dim: u64,
    elements: u64,
    pairs: u64,
    launch_units: u64,
    bytes: u64,
    frequency_bytes: u64,
    operations: u64,
}

impl Shape {
    fn new(checked: HalfSplitRopeBf16Geometry) -> Result<Self, String> {
        Self::dimensions(
            checked.rows(),
            checked.columns(),
            checked.head_dim(),
            checked.head_dim(),
        )
    }
    fn new_partial(checked: PartialHalfSplitRopeBf16Geometry) -> Result<Self, String> {
        Self::dimensions(
            checked.rows(),
            checked.columns(),
            checked.head_dim(),
            checked.rotary_dim(),
        )
    }
    fn dimensions(rows: u64, columns: u64, head_dim: u64, rotary_dim: u64) -> Result<Self, String> {
        let elements = rows
            .checked_mul(columns)
            .ok_or("half-split RoPE element count overflows")?;
        let bytes = elements
            .checked_mul(2)
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("half-split RoPE BF16 byte extent exceeds address space")?;
        let frequency_bytes = rows
            .checked_mul(rotary_dim)
            .and_then(|n| n.checked_mul(2))
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("half-split RoPE BF16 frequency byte extent exceeds address space")?;
        let heads = elements / head_dim;
        let pairs = heads
            .checked_mul(rotary_dim / 2)
            .ok_or("rotated pair count overflows")?;
        let launch_units = heads
            .checked_mul(rotary_dim / 2 + head_dim - rotary_dim)
            .ok_or("rotary launch units overflow")?;
        let operations = pairs
            .checked_mul(7)
            .ok_or("half-split RoPE logical work overflows")?;
        Ok(Self {
            columns,
            head_dim,
            rotary_dim,
            elements,
            pairs,
            launch_units,
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

fn partial_geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != 4 || attributes.len() != 2 {
        return Err(
            "partial RoPE requires three inputs, one output, head_dim and rotary_dim".into(),
        );
    }
    let unsigned = |name| match attributes
        .iter()
        .find(|(id, _)| id.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!("partial RoPE requires unsigned {name}")),
    };
    Shape::new_partial(
        PartialHalfSplitRopeBf16Geometry::new(
            binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
            binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
            binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
            binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
            unsigned("head_dim")?,
            unsigned("rotary_dim")?,
        )
        .map_err(|error| error.to_string())?,
    )
}

#[cfg(test)]
mod tests;
