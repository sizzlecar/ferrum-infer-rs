//! Grouped-query structured attention over original Q/K/V allocations.
//! Query heads share KV heads through an index map, never a repeated KV tensor.

use super::*;
use ferrum_runtime::{
    structured_gqa_attention_f32_contract, ResolvedTensorSpec, StructuredGqaAttentionF32Geometry,
    STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID, STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID,
};

pub const NUMERICAL_PROFILE_ID: &str = "cuda.ieee.structured_gqa_attention.f32.online_exp.v1";
pub const NUMERICAL_IDENTITY: &str = concat!(
    "F32-QY[B*S,Hq*D];F32-KV[B*S,Hkv*D];Hq-divisible-by-Hkv;compact-KV;",
    "explicit-structured-key-mask-and-segments;key-only-padding;NoAlias;zero-global-scratch;",
    "D2..256;one-CTA-per-query-head;one-F32-Q-and-acc-scalar-per-thread;",
    "tail-guarded-adjacent-pairwise-dot;unmatched-subtree-carried-without-add-zero;",
    "ascending-key-online-softmax;separate-F32-mul-add;",
    "precise-expf-div-sqrt;fmad=false;ftz=false;no-fast-math;no-BF16-or-F16-boundary;",
    "empty-visible-set=positive-zero-without-QKV-read;masked-KV-never-read;",
    "visible-nonfinite-score=whole-head-NaN;visible-V-IEEE-per-channel;",
    "logical-batch-tiles16;by-value-validated-mask;target-qualification-required;",
    "not-Torch-binary-parity;not-Reference-libm-bitwise-parity"
);
pub(super) const FUNCTION: &str = "ferrum_vnext_structured_gqa_attention_f32";
const PROVIDER_ID: &str = "provider.cuda.structured_gqa_attention.f32.online_exp";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.structured_gqa_attention.f32.online_exp";

use super::attention::{BatchMask, NativeMask};

pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    let capability =
        CapabilityId::new(STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID).map_err(contract_error)?;
    if !device.capabilities.contains(&capability) {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA runtime lacks structured F32 GQA attention capability",
        ));
    }
    let contract = structured_gqa_attention_f32_contract().map_err(contract_error)?;
    let implementation = implementation_fingerprint(&[
        numerical_fingerprint().as_bytes(),
        include_bytes!("gqa_f32.rs"),
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
    .map(|d| d.with_fixed_tensor_work(true))
}

pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
) -> Box<dyn OperationProvider<CudaDeviceRuntime>> {
    Box::new(GqaAttentionProvider {
        descriptor,
        kernels: Arc::clone(kernels),
    })
}

struct GqaAttentionProvider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
}

impl OperationResourceEstimator for GqaAttentionProvider {
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
                "CUDA GQA attention estimator received another operation",
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

impl OperationProvider<CudaDeviceRuntime> for GqaAttentionProvider {
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
            "cuda.structured_gqa_attention.f32.encode",
            "CUDA GQA attention requires fixed tensor work".into(),
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
                provider_failure(
                    identity,
                    "cuda.structured_gqa_attention.f32.encode",
                    message,
                )
            })
    }
}

impl GqaAttentionProvider {
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA GQA attention requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA GQA attention requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        let query = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            shape.query_bytes,
            ElementType::F32,
        )?;
        let key = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            shape.key_value_bytes,
            ElementType::F32,
        )?;
        let value = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            shape.key_value_bytes,
            ElementType::F32,
        )?;
        let output = retained_typed_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            shape.query_bytes,
            ElementType::F32,
        )?;
        for read in [&query, &key, &value] {
            nonoverlapping_ranges(
                read.device_ptr(),
                read.length_bytes(),
                output.device_ptr(),
                output.length_bytes(),
            )?;
        }
        let kernels = Arc::clone(&self.kernels);
        let work = (shape.elements, shape.operations, shape.masks.len() as u64);
        CudaDeviceCommand::operation(
            FUNCTION,
            vec![query, key, value, output],
            move |stream, regions| {
                for tile in &shape.masks {
                    let query_offset = tile.first_batch * (shape.query_bytes / shape.batch);
                    let key_offset = tile.first_batch * (shape.key_value_bytes / shape.batch);
                    let query = regions[0].device_ptr() + query_offset;
                    let key = regions[1].device_ptr() + key_offset;
                    let value = regions[2].device_ptr() + key_offset;
                    let output = regions[3].device_ptr() + query_offset;
                    let groups = (shape.groups / shape.batch) * tile.batches;
                    let mut builder = stream.launch_builder(kernels.gqa_attention_f32_function());
                    builder
                        .arg(&query)
                        .arg(&key)
                        .arg(&value)
                        .arg(&output)
                        .arg(&groups)
                        .arg(&shape.query_heads)
                        .arg(&shape.key_value_heads)
                        .arg(&shape.sequence)
                        .arg(&shape.width)
                        .arg(&shape.scale)
                        .arg(&tile.mask);
                    unsafe { builder.launch(shape.launch_config_for_groups(groups)) }.map_err(
                        |error| {
                            CudaDeviceRuntimeError::driver(
                                "F32 structured GQA attention launch",
                                error,
                            )
                        },
                    )?;
                }
                Ok(())
            },
        )
        .and_then(|command| command.with_tensor_work(work.0, work.1, work.2, 0))
        .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug)]
struct Shape {
    groups: u64,
    query_heads: u64,
    key_value_heads: u64,
    sequence: u32,
    width: u32,
    scale: f32,
    elements: u64,
    operations: u64,
    query_bytes: u64,
    key_value_bytes: u64,
    batch: u64,
    masks: Vec<BatchMask>,
}

impl Shape {
    #[cfg(test)]
    fn launch_config(&self) -> LaunchConfig {
        self.launch_config_for_groups(self.groups)
    }

    fn launch_config_for_groups(&self, groups: u64) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (groups.min(MAX_BLOCKS) as u32, 1, 1),
            block_dim: (self.width, 1, 1),
            shared_mem_bytes: self.width * 4,
        }
    }

    fn new(checked: StructuredGqaAttentionF32Geometry) -> Result<Self, String> {
        let query_bytes = checked
            .element_count()
            .checked_mul(4)
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("CUDA GQA attention byte extent exceeds address space")?;
        let key_value_bytes = checked
            .rows()
            .checked_mul(checked.key_value_columns())
            .and_then(|elements| elements.checked_mul(4))
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or("CUDA GQA KV byte extent exceeds address space")?;
        let groups = checked
            .rows()
            .checked_mul(checked.query_heads())
            .ok_or("CUDA GQA attention group count overflows")?;
        Ok(Self {
            groups,
            query_heads: checked.query_heads(),
            key_value_heads: checked.key_value_heads(),
            sequence: checked.sequence_length() as u32,
            width: checked.head_dim() as u32,
            scale: checked.effective_scale(),
            elements: checked.element_count(),
            operations: checked.logical_arithmetic_operations(),
            query_bytes,
            key_value_bytes,
            batch: checked.batch(),
            masks: NativeMask::tiles(checked.mask())?,
        })
    }
}

fn geometry(
    bindings: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if bindings.len() != 4 {
        return Err("CUDA GQA attention requires exactly Q/K/V and one output".into());
    }
    validate_specs(
        binding(bindings, ResolvedValueRole::Input, 0)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 1)?.tensor(),
        binding(bindings, ResolvedValueRole::Input, 2)?.tensor(),
        binding(bindings, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
}

fn validate_specs(
    q: &ResolvedTensorSpec,
    k: &ResolvedTensorSpec,
    v: &ResolvedTensorSpec,
    y: &ResolvedTensorSpec,
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    Shape::new(
        StructuredGqaAttentionF32Geometry::new(q, k, v, y, attributes)
            .map_err(|e| e.to_string())?,
    )
}

#[cfg(test)]
mod tests;
