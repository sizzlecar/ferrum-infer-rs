//! Independent finite-domain fused F32 attention, with explicit strict-kernel
//! fallback for masks, grouped heads and dimensions outside the native shape.

use super::*;
use ferrum_runtime::{
    ResolvedTensorSpec, StructuredGqaAttentionF32Geometry,
    STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID,
};

pub use crate::native_ops::cutlass_fused_f32_attention::NUMERICAL_PROFILE_ID;
mod native;
pub(crate) use native::source_fingerprint;
use native::{Bindings, Native, Request, Span};

/// Additional static CUDART module allowance, charged to the existing device
/// account and conservatively retained for the process lifetime. No payload
/// scratch, shape cache or new stream is created by this provider.
#[derive(Clone, Debug)]
pub struct CudaCutlassFusedF32Config {
    pub opaque_memory_bytes: u64,
    pub opaque_memory_profile: String,
}
impl CudaCutlassFusedF32Config {
    pub fn validate(&self) -> Result<(), CudaDeviceRuntimeError> {
        if self.opaque_memory_bytes == 0
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(CudaDeviceRuntimeError::contract("fused F32 attention requires a positive opaque allowance and versioned ASCII profile"));
        }
        Ok(())
    }
}

use super::attention::{BatchMask, NativeMask};

pub(super) fn descriptor(
    device: &DeviceDescriptor,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    // CPU package declarations share the exact source-based descriptor. Runtime
    // availability still requires this binary's validated compiled artifact;
    // its actual identity is also bound into every admitted implementation.
    source_fingerprint().map_err(CudaDeviceRuntimeError::contract)?;
    crate::native_ops::cutlass_fused_f32_attention::provider_descriptor(device)
        .map_err(contract_error)
}

pub(super) fn provider(
    descriptor: OperationProviderDescriptor,
    kernels: &Arc<RetainedFunctions>,
    runtime: &CudaDeviceRuntime,
    config: &CudaCutlassFusedF32Config,
) -> Result<Box<dyn OperationProvider<CudaDeviceRuntime>>, CudaDeviceRuntimeError> {
    let native = Native::new(runtime, config)?;
    Ok(Box::new(GqaAttentionProvider {
        descriptor,
        kernels: Arc::clone(kernels),
        native,
    }))
}

struct GqaAttentionProvider {
    descriptor: OperationProviderDescriptor,
    kernels: Arc<RetainedFunctions>,
    native: Arc<Native>,
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
        let shape = geometry(request.values(), request.attributes()).map_err(invalid_plan)?;
        OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            ALIGNMENT,
            None,
            None,
        )
        .with_prepared_implementation_fingerprint(self.prepared_fingerprint(&shape))
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
    fn prepared_fingerprint(&self, shape: &Shape) -> String {
        match &shape.native_request {
            Some(request) => self.native.fingerprint(request),
            None => implementation_fingerprint(&[
                b"strict-online-exp-fallback",
                self.native.artifact_identity.as_bytes(),
                gqa_f32::NUMERICAL_IDENTITY.as_bytes(),
            ]),
        }
    }
    fn encode(
        &self,
        invocation: TensorOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<CudaDeviceCommand, String> {
        let [participant] = invocation.participants() else {
            return Err("CUDA GQA attention requires one tensor participant".into());
        };
        if invocation.operation().id.as_str() != STRUCTURED_GQA_ATTENTION_FUSED_F32_OPERATION_ID
            || !matches!(participant.work(), NodeWorkContract::Fixed)
            || !invocation.resource_work_shape().resource_work().is_tensor()
        {
            return Err("CUDA GQA attention requires its fixed tensor contract".into());
        }
        let shape = geometry(participant.bindings(), participant.attributes())?;
        if invocation.prepared_implementation_fingerprint()
            != Some(self.prepared_fingerprint(&shape).as_str())
        {
            return Err("fused F32 route differs from the admitted implementation".into());
        }
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
        let regions = vec![query, key, value, output];
        if regions
            .iter()
            .any(|r| !Arc::ptr_eq(r.context(), self.native.owner.context()))
        {
            return Err("fused F32 bindings belong to another context".into());
        }
        if let Some(request) = shape.native_request {
            let native = Arc::clone(&self.native);
            spans(&regions)?
                .validate(&request)
                .map_err(|e| e.to_string())?;
            return CudaDeviceCommand::operation(
                native::CUTLASS_FUSED_F32_ATTENTION_OPERATOR,
                regions,
                move |stream, regions| {
                    native
                        .owner
                        .context()
                        .bind_to_thread()
                        .map_err(|e| CudaDeviceRuntimeError::driver("fused F32 context bind", e))?;
                    let bindings = spans(regions).map_err(CudaDeviceRuntimeError::contract)?;
                    // Custom commands retain native/payload owners through the
                    // existing fence/error recovery. Never retry a native error.
                    unsafe { native.run(&request, &bindings, stream.cu_stream().cast()) }
                },
            )
            .and_then(|c| c.with_tensor_work(shape.elements, shape.operations, 1, 0))
            .map_err(|e| e.to_string());
        }
        let kernels = Arc::clone(&self.kernels);
        let work = (shape.elements, shape.operations, shape.masks.len() as u64);
        CudaDeviceCommand::operation(gqa_f32::FUNCTION, regions, move |stream, regions| {
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
                        CudaDeviceRuntimeError::driver("F32 structured GQA attention launch", error)
                    },
                )?;
            }
            Ok(())
        })
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
    native_request: Option<Request>,
}

impl Shape {
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
        let native_request = native_request(&checked);
        Ok(Self {
            native_request,
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

/// Pure selection before enqueue. Structured masks remain exactly on the old
/// strict kernel; unsupported native geometry is not a runtime-error retry.
fn native_request(g: &StructuredGqaAttentionF32Geometry) -> Option<Request> {
    let mask = g.mask();
    if g.query_heads() != g.key_value_heads()
        || mask.segment_ends() != [g.sequence_length() as u32]
        || mask.segment_bidirectional() != [true]
        || (0..mask.batch()).any(|b| {
            mask.key_run_ends(b) != Some(&[g.sequence_length() as u32][..])
                || mask.key_run_valid(b) != Some(&[true][..])
        })
    {
        return None;
    }
    Request::new(
        u32::try_from(g.batch()).ok()?,
        u32::try_from(g.sequence_length()).ok()?,
        u32::try_from(g.query_heads()).ok()?,
        u32::try_from(g.head_dim()).ok()?,
        g.effective_scale(),
    )
    .ok()
}
fn spans(regions: &[CudaBufferRegion]) -> Result<Bindings, String> {
    let [q, k, v, y] = regions else {
        return Err("fused F32 requires four retained spans".into());
    };
    let span = |r: &CudaBufferRegion| Span {
        address: r.device_ptr(),
        elements: r.length_bytes() / 4,
    };
    Ok(Bindings {
        query: span(q),
        key: span(k),
        value: span(v),
        output: span(y),
    })
}
#[cfg(test)]
mod tests;
