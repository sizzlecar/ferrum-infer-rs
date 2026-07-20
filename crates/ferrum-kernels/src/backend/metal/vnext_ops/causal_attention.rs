//! Native Metal provider for the standard fixed-page causal attention operation.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    causal_paged_attention_contract, AttributeId, BatchedOperationInvocation,
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView,
    ElementType, EncodedDeviceOperation, OperationBufferStorageKind, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, ResolvedValueBinding, ResolvedValueRole, SemanticValue,
    VNextError, CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, CAUSAL_PAGED_ATTENTION_OPERATION_ID,
};
use metal::{
    ArgumentEncoder, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Function, MTLArgumentBuffersTier, MTLResourceUsage, MTLSize,
};

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError,
};
use super::linear::{
    append_shared_matrix_weight, dispatch_linear, linear_launch, validate_launch_regions,
    LinearLaunch, MetalLinearPipelines,
};
use super::primitives::{dispatch_residual_add_at, dispatch_rms_norm_at, MetalPrimitivePipelines};
use super::{
    binding, checked_u32, contract_error, ensure_invocation, f16_contiguous,
    implementation_fingerprint, invalid_plan, provider_descriptor, provider_failure,
    rational_attribute, shared_binding_region, shared_full_region, shared_scratch_region,
    token_binding_is_shared, unsigned_attribute, DENSE_SAFETENSORS_FORMAT_ID,
    GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID, Q5_K_FORMAT_ID, Q6_K_FORMAT_ID, Q8_0_FORMAT_ID,
    VALUE_ALIGNMENT_BYTES, VNEXT_KV_PAGE_BYTES,
};

const SHADER_SOURCE: &str = include_str!("causal_attention.metal");
const PROVIDER_ID: &str = "provider.metal.causal_paged_attention.f16.native";
const ESTIMATOR_ID: &str = "resource-estimator.metal.causal_paged_attention.f16.native";
const PREPARE_KERNEL: &str = "vnext_causal_prepare_f16";
const ATTENTION_KERNEL: &str = "vnext_causal_attention_f16";
const PREPARE_PAGE_TABLE_INDEX: u64 = 6;
const ATTENTION_PAGE_TABLE_INDEX: u64 = 3;
const SIMD_THREADS: u64 = 32;
const MAXIMUM_HEAD_DIM: u64 = 256;
const MAXIMUM_KV_PAGES: u64 = 16_384;

pub(super) struct MetalCausalAttentionPipelines {
    prepare: ComputePipelineState,
    attention: ComputePipelineState,
    prepare_function: Function,
    binding_encoded_length: u64,
    binding_alignment: u64,
}

impl MetalCausalAttentionPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        if device.argument_buffers_support() != MTLArgumentBuffersTier::Tier2 {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal causal attention requires argument-buffer tier 2",
            ));
        }
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile Metal vNext causal-attention library: {error}"
                ))
            })?;
        let function = |name: &str| {
            library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "load Metal vNext causal-attention `{name}`: {error}"
                ))
            })
        };
        let prepare_function = function(PREPARE_KERNEL)?;
        let attention_function = function(ATTENTION_KERNEL)?;
        let prepare_encoder = prepare_function.new_argument_encoder(PREPARE_PAGE_TABLE_INDEX);
        let attention_encoder = attention_function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        let binding_encoded_length = prepare_encoder.encoded_length();
        let binding_alignment = prepare_encoder.alignment();
        if binding_encoded_length == 0
            || binding_alignment == 0
            || attention_encoder.encoded_length() != binding_encoded_length
            || attention_encoder.alignment() != binding_alignment
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal causal-attention kernels disagree on the page-table argument layout",
            ));
        }
        let pipeline = |function: &Function| {
            device
                .new_compute_pipeline_state_with_function(function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext causal-attention pipeline: {error}"
                    ))
                })
        };
        let prepare = pipeline(&prepare_function)?;
        let attention = pipeline(&attention_function)?;
        if prepare.thread_execution_width() != SIMD_THREADS
            || attention.thread_execution_width() != SIMD_THREADS
        {
            return Err(MetalDeviceRuntimeError::contract(format!(
                "Metal causal attention requires {SIMD_THREADS}-lane SIMD execution, got prepare={} attention={}",
                prepare.thread_execution_width(),
                attention.thread_execution_width()
            )));
        }
        Ok(Self {
            prepare,
            attention,
            prepare_function,
            binding_encoded_length,
            binding_alignment,
        })
    }

    fn binding_slot_bytes(&self) -> Result<u64, String> {
        align_up(self.binding_encoded_length, self.binding_alignment)
    }

    fn new_binding_encoder(&self) -> ArgumentEncoder {
        self.prepare_function
            .new_argument_encoder(PREPARE_PAGE_TABLE_INDEX)
    }
}

pub(super) struct MetalCausalPagedAttentionProvider {
    descriptor: OperationProviderDescriptor,
    attention: Arc<MetalCausalAttentionPipelines>,
    linear: Arc<MetalLinearPipelines>,
    primitives: Arc<MetalPrimitivePipelines>,
}

impl MetalCausalPagedAttentionProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        attention: Arc<MetalCausalAttentionPipelines>,
        linear: Arc<MetalLinearPipelines>,
        primitives: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = causal_paged_attention_contract().map_err(contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            PROVIDER_ID,
            CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
            ESTIMATOR_ID,
            storage_bindings().map_err(contract_error)?,
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[
                Q4_K_FORMAT_ID,
                Q5_K_FORMAT_ID,
                Q6_K_FORMAT_ID,
                Q8_0_FORMAT_ID,
            ],
            implementation_fingerprint(&[
                include_str!("causal_attention.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                include_str!("linear.rs").as_bytes(),
                include_str!("linear.metal").as_bytes(),
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives.metal").as_bytes(),
                PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            attention,
            linear,
            primitives,
        })
    }
}

fn storage_bindings() -> Result<Vec<ProviderStorageBindingRequirement>, VNextError> {
    let paged = DynamicStorageRequirement::new(vec![DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena {
            block_bytes: VNEXT_KV_PAGE_BYTES,
        },
        DynamicStorageView::PagedRegions {
            block_bytes: VNEXT_KV_PAGE_BYTES,
        },
    )?])?;
    Ok((0..9)
        .map(|ordinal| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal,
                if ordinal == 8 {
                    paged.clone()
                } else {
                    DynamicStorageRequirement::contiguous()
                },
            )
        })
        .chain(std::iter::once(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            0,
            DynamicStorageRequirement::contiguous(),
        )))
        .collect())
}

impl OperationResourceEstimator for MetalCausalPagedAttentionProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != CAUSAL_PAGED_ATTENTION_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "Metal estimator `{}` received another operation",
                self.descriptor.resource_estimator_id()
            )));
        }
        let shape =
            CausalAttentionShape::from_attributes(request.attributes()).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::tokens(
                shape.scratch_bytes_per_token().map_err(invalid_plan)?,
            )?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            DynamicStorageRequirement::contiguous(),
        )?;
        let binding = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::actual_sequences(
                self.attention.binding_slot_bytes().map_err(invalid_plan)?,
            )?,
            self.attention.binding_alignment,
            ProviderWorkspaceScope::Invocation,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            Some(scratch),
            None,
        )
        .with_binding(binding))
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalCausalPagedAttentionProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_attention(
            Arc::clone(&self.attention),
            Arc::clone(&self.linear),
            Arc::clone(&self.primitives),
            invocation,
        )
        .map_err(|message| {
            provider_failure(identity, "metal.causal_paged_attention.encode", message)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CausalAttentionShape {
    hidden_size: u64,
    query_heads: u64,
    key_value_heads: u64,
    head_dim: u64,
    query_features: u64,
    query_projection_features: u64,
    kv_features: u64,
    rope_dim: u64,
    maximum_context_tokens: u64,
    epsilon: f32,
    rope_theta: f32,
    rope_interleaved: bool,
    output_gate: bool,
}

impl CausalAttentionShape {
    fn from_attributes(attributes: &BTreeMap<AttributeId, SemanticValue>) -> Result<Self, String> {
        let shape = Self {
            hidden_size: unsigned_attribute(attributes, "hidden_size")?,
            query_heads: unsigned_attribute(attributes, "query_heads")?,
            key_value_heads: unsigned_attribute(attributes, "key_value_heads")?,
            head_dim: unsigned_attribute(attributes, "head_dim")?,
            query_features: unsigned_attribute(attributes, "query_features")?,
            query_projection_features: unsigned_attribute(attributes, "query_projection_features")?,
            kv_features: unsigned_attribute(attributes, "kv_features")?,
            rope_dim: unsigned_attribute(attributes, "rope_dim")?,
            maximum_context_tokens: unsigned_attribute(attributes, "maximum_context_tokens")?,
            epsilon: rational_attribute(attributes, "epsilon")?,
            rope_theta: rational_attribute(attributes, "rope_theta")?,
            rope_interleaved: bool_attribute(attributes, "rope_interleaved")?,
            output_gate: bool_attribute(attributes, "output_gate")?,
        };
        if !bool_attribute(attributes, "causal")? {
            return Err("Metal causal attention requires causal=true".to_owned());
        }
        let query_features = shape
            .query_heads
            .checked_mul(shape.head_dim)
            .ok_or_else(|| "Metal causal-attention query width overflows".to_owned())?;
        let kv_features = shape
            .key_value_heads
            .checked_mul(shape.head_dim)
            .ok_or_else(|| "Metal causal-attention KV width overflows".to_owned())?;
        let query_projection_features = query_features
            .checked_mul(if shape.output_gate { 2 } else { 1 })
            .ok_or_else(|| "Metal causal-attention query projection width overflows".to_owned())?;
        if shape.hidden_size == 0
            || shape.query_heads == 0
            || shape.key_value_heads == 0
            || shape.head_dim == 0
            || shape.rope_dim == 0
            || shape.maximum_context_tokens == 0
            || shape.query_heads % shape.key_value_heads != 0
            || shape.head_dim > MAXIMUM_HEAD_DIM
            || shape.rope_dim > shape.head_dim
            || !shape.rope_dim.is_multiple_of(2)
            || shape.query_features != query_features
            || shape.kv_features != kv_features
            || shape.query_projection_features != query_projection_features
        {
            return Err("Metal causal-attention attributes are inconsistent".to_owned());
        }
        if shape.maximum_pages()? > MAXIMUM_KV_PAGES {
            return Err(format!(
                "Metal causal attention requires {} pages, exceeding provider limit {}",
                shape.maximum_pages()?,
                MAXIMUM_KV_PAGES
            ));
        }
        shape.params(1, 0, 1)?;
        Ok(shape)
    }

    fn state_bytes_per_token(self) -> Result<u64, String> {
        self.kv_features
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| "Metal causal-attention KV bytes per token overflow".to_owned())
    }

    fn physical_state_bytes(self, tokens: u64) -> Result<u64, String> {
        let logical = self
            .state_bytes_per_token()?
            .checked_mul(tokens)
            .ok_or_else(|| "Metal causal-attention KV state size overflows".to_owned())?;
        align_up(logical, VNEXT_KV_PAGE_BYTES)
    }

    fn physical_state_bytes_for_source_frontier(
        self,
        source_end_tokens: u64,
        full_input_tokens: u64,
    ) -> Result<u64, String> {
        if source_end_tokens == 0 || source_end_tokens > full_input_tokens {
            return Err("Metal causal-attention source frontier exceeds its full input".to_owned());
        }
        self.physical_state_bytes(source_end_tokens)
    }

    fn maximum_pages(self) -> Result<u64, String> {
        Ok(self.physical_state_bytes(self.maximum_context_tokens)? / VNEXT_KV_PAGE_BYTES)
    }

    fn scratch_bytes_per_token(self) -> Result<u64, String> {
        [
            self.hidden_size,
            self.query_projection_features,
            self.kv_features,
            self.kv_features,
            self.query_features,
            self.query_features,
            self.hidden_size,
        ]
        .into_iter()
        .try_fold(0_u64, |total, elements| {
            total
                .checked_add(aligned_bytes(elements, ElementType::F16.size_bytes())?)
                .ok_or_else(|| "Metal causal-attention token scratch size overflows".to_owned())
        })
    }

    fn params(
        self,
        tokens: u64,
        position_start: u64,
        page_count: u64,
    ) -> Result<CausalAttentionParams, String> {
        let query_head_stride = self
            .head_dim
            .checked_mul(if self.output_gate { 2 } else { 1 })
            .ok_or_else(|| "Metal causal-attention query head stride overflows".to_owned())?;
        Ok(CausalAttentionParams {
            page_elements: checked_u32(
                VNEXT_KV_PAGE_BYTES / ElementType::F16.size_bytes(),
                "Metal causal-attention page elements",
            )?,
            page_count: checked_u32(page_count, "Metal causal-attention page count")?,
            position_start: checked_u32(position_start, "Metal causal-attention source position")?,
            tokens: checked_u32(tokens, "Metal causal-attention token count")?,
            query_heads: checked_u32(self.query_heads, "Metal causal-attention query heads")?,
            key_value_heads: checked_u32(
                self.key_value_heads,
                "Metal causal-attention key/value heads",
            )?,
            head_dim: checked_u32(self.head_dim, "Metal causal-attention head dimension")?,
            rope_dim: checked_u32(self.rope_dim, "Metal causal-attention RoPE dimension")?,
            query_projection_stride: checked_u32(
                self.query_projection_features,
                "Metal causal-attention query projection stride",
            )?,
            query_head_stride: checked_u32(
                query_head_stride,
                "Metal causal-attention query head stride",
            )?,
            kv_projection_stride: checked_u32(
                self.kv_features,
                "Metal causal-attention KV projection stride",
            )?,
            output_gate: u32::from(self.output_gate),
            rope_interleaved: u32::from(self.rope_interleaved),
            epsilon: self.epsilon,
            rope_theta: self.rope_theta,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CausalAttentionParams {
    page_elements: u32,
    page_count: u32,
    position_start: u32,
    tokens: u32,
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    rope_dim: u32,
    query_projection_stride: u32,
    query_head_stride: u32,
    kv_projection_stride: u32,
    output_gate: u32,
    rope_interleaved: u32,
    epsilon: f32,
    rope_theta: f32,
}

#[derive(Debug, Clone, Copy)]
struct ScratchLayout {
    required_bytes: u64,
    normalized: u64,
    query_raw: u64,
    key_raw: u64,
    value_raw: u64,
    query: u64,
    context: u64,
    projected: u64,
}

impl ScratchLayout {
    fn new(shape: CausalAttentionShape, total_tokens: u64) -> Result<Self, String> {
        if total_tokens == 0 {
            return Err("Metal causal-attention scratch cannot size empty work".to_owned());
        }
        let mut offset = 0;
        let normalized = reserve_tokens(&mut offset, shape.hidden_size, total_tokens)?;
        let query_raw = reserve_tokens(&mut offset, shape.query_projection_features, total_tokens)?;
        let key_raw = reserve_tokens(&mut offset, shape.kv_features, total_tokens)?;
        let value_raw = reserve_tokens(&mut offset, shape.kv_features, total_tokens)?;
        let query = reserve_tokens(&mut offset, shape.query_features, total_tokens)?;
        let context = reserve_tokens(&mut offset, shape.query_features, total_tokens)?;
        let projected = reserve_tokens(&mut offset, shape.hidden_size, total_tokens)?;
        let expected = shape
            .scratch_bytes_per_token()?
            .checked_mul(total_tokens)
            .ok_or_else(|| "Metal causal-attention scratch size overflows".to_owned())?;
        if offset != expected {
            return Err(
                "Metal causal-attention scratch layout differs from its estimate".to_owned(),
            );
        }
        Ok(Self {
            required_bytes: offset,
            normalized,
            query_raw,
            key_raw,
            value_raw,
            query,
            context,
            projected,
        })
    }

    fn token_offset(self, base: u64, token_start: u64, width: u64) -> Result<u64, String> {
        base.checked_add(
            aligned_bytes(width, ElementType::F16.size_bytes())?
                .checked_mul(token_start)
                .ok_or_else(|| {
                    "Metal causal-attention token scratch offset overflows".to_owned()
                })?,
        )
        .filter(|offset| *offset < self.required_bytes)
        .ok_or_else(|| "Metal causal-attention token scratch range is invalid".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct BindingLayout {
    required_bytes: u64,
    slot_bytes: u64,
}

impl BindingLayout {
    fn new(slot_bytes: u64, participant_count: usize) -> Result<Self, String> {
        if slot_bytes == 0 || participant_count == 0 {
            return Err("Metal causal-attention binding cannot size empty work".to_owned());
        }
        let count = u64::try_from(participant_count)
            .map_err(|_| "Metal causal-attention participant count exceeds u64".to_owned())?;
        let required_bytes = slot_bytes
            .checked_mul(count)
            .ok_or_else(|| "Metal causal-attention binding workspace size overflows".to_owned())?;
        Ok(Self {
            required_bytes,
            slot_bytes,
        })
    }

    fn offset(self, participant: usize) -> Result<u64, String> {
        self.slot_bytes
            .checked_mul(
                u64::try_from(participant).map_err(|_| {
                    "Metal causal-attention participant index exceeds u64".to_owned()
                })?,
            )
            .filter(|offset| *offset < self.required_bytes)
            .ok_or_else(|| "Metal causal-attention binding offset is invalid".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct SharedRegions {
    input_norm: usize,
    query_norm: usize,
    key_norm: usize,
    scratch: usize,
    binding: usize,
}

#[derive(Debug, Clone, Copy)]
struct ParticipantLaunch {
    input: usize,
    output: usize,
    first_page_region: usize,
    page_count: usize,
    binding_offset: u64,
    normalized: u64,
    query_raw: u64,
    key_raw: u64,
    value_raw: u64,
    query: u64,
    context: u64,
    projected: u64,
    hidden_size: u32,
    residual_elements: u32,
    params: CausalAttentionParams,
    query_projection: LinearLaunch,
    key_projection: LinearLaunch,
    value_projection: LinearLaunch,
    output_projection: LinearLaunch,
}

#[derive(Debug, Clone, Copy)]
struct PageBinding {
    first_page_region: usize,
    page_count: usize,
    binding_offset: u64,
}

fn encode_attention(
    attention: Arc<MetalCausalAttentionPipelines>,
    linear: Arc<MetalLinearPipelines>,
    primitives: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, String> {
    ensure_invocation(&invocation, CAUSAL_PAGED_ATTENTION_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let shape = CausalAttentionShape::from_attributes(first.attributes())?;
    validate_signature(first, shape)?;
    for participant in &invocation.participants()[1..] {
        if CausalAttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("Metal causal-attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape)?;
    }

    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::new(shape, total_tokens)?;
    let binding_layout = BindingLayout::new(
        attention.binding_slot_bytes()?,
        invocation.participants().len(),
    )?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal causal-attention participant ranges are incomplete".to_owned());
    }

    let mut regions = Vec::new();
    let query_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        2,
        shape.query_projection_features,
        shape.hidden_size,
        "Metal causal-attention query projection",
    )?;
    let key_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        3,
        shape.kv_features,
        shape.hidden_size,
        "Metal causal-attention key projection",
    )?;
    let value_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        4,
        shape.kv_features,
        shape.hidden_size,
        "Metal causal-attention value projection",
    )?;
    let output_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        5,
        shape.hidden_size,
        shape.query_features,
        "Metal causal-attention output projection",
    )?;
    let shared = SharedRegions {
        input_norm: push_shared_region(&mut regions, &invocation, 1)?,
        query_norm: push_shared_region(&mut regions, &invocation, 6)?,
        key_norm: push_shared_region(&mut regions, &invocation, 7)?,
        scratch: {
            let index = regions.len();
            regions.push(shared_scratch_region(&invocation, layout.required_bytes)?);
            index
        },
        binding: {
            let index = regions.len();
            regions.push(shared_binding_region(
                &invocation,
                binding_layout.required_bytes,
            )?);
            index
        },
    };
    let input_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let output_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Output, 0, ElementType::F16)?;

    let mut binding_regions = vec![regions[shared.binding].clone()];
    let mut page_bindings = Vec::with_capacity(invocation.participants().len());
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant_index, (participant, token_range)) in invocation
        .participants()
        .iter()
        .zip(token_ranges)
        .enumerate()
    {
        let tokens = token_range.immediate_tokens();
        let source = token_range.source_token_range();
        let packed_start = token_range.immediate_token_range().start;
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err(
                "Metal causal-attention token range exceeds its admitted context".to_owned(),
            );
        }
        let input = regions.len();
        regions.push(super::contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::F16,
            if input_shared {
                packed_start
            } else {
                source.start
            },
            tokens,
        )?);
        let output = regions.len();
        regions.push(super::contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F16,
            if output_shared {
                packed_start
            } else {
                source.start
            },
            tokens,
        )?);

        let state = binding(participant.bindings(), ResolvedValueRole::Input, 8)?;
        let pages = paged_state_regions(
            participant,
            state,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
        )?;
        if pages.len() > MAXIMUM_KV_PAGES as usize {
            return Err("Metal causal-attention page table exceeds its provider limit".to_owned());
        }
        let binding_first_page = binding_regions.len();
        binding_regions.extend(pages.iter().cloned());
        let first_page_region = regions.len();
        regions.extend(pages);
        let page_count = regions.len() - first_page_region;
        let page_count_u64 = u64::try_from(page_count)
            .map_err(|_| "Metal causal-attention page count exceeds u64".to_owned())?;
        let binding_offset = binding_layout.offset(participant_index)?;
        page_bindings.push(PageBinding {
            first_page_region: binding_first_page,
            page_count,
            binding_offset,
        });

        let normalized = layout.token_offset(layout.normalized, packed_start, shape.hidden_size)?;
        let query_raw = layout.token_offset(
            layout.query_raw,
            packed_start,
            shape.query_projection_features,
        )?;
        let key_raw = layout.token_offset(layout.key_raw, packed_start, shape.kv_features)?;
        let value_raw = layout.token_offset(layout.value_raw, packed_start, shape.kv_features)?;
        let query = layout.token_offset(layout.query, packed_start, shape.query_features)?;
        let context = layout.token_offset(layout.context, packed_start, shape.query_features)?;
        let projected = layout.token_offset(layout.projected, packed_start, shape.hidden_size)?;

        launches.push(ParticipantLaunch {
            input,
            output,
            first_page_region,
            page_count,
            binding_offset,
            normalized,
            query_raw,
            key_raw,
            value_raw,
            query,
            context,
            projected,
            hidden_size: checked_u32(shape.hidden_size, "Metal causal-attention hidden size")?,
            residual_elements: checked_u32(
                tokens.checked_mul(shape.hidden_size).ok_or_else(|| {
                    "Metal causal-attention residual element count overflows".to_owned()
                })?,
                "Metal causal-attention residual elements",
            )?,
            params: shape.params(tokens, source.start, page_count_u64)?,
            query_projection: linear_launch(
                query_weight,
                shared.scratch,
                shared.scratch,
                tokens,
                shape.hidden_size,
                shape.query_projection_features,
                normalized,
                query_raw,
            )?,
            key_projection: linear_launch(
                key_weight,
                shared.scratch,
                shared.scratch,
                tokens,
                shape.hidden_size,
                shape.kv_features,
                normalized,
                key_raw,
            )?,
            value_projection: linear_launch(
                value_weight,
                shared.scratch,
                shared.scratch,
                tokens,
                shape.hidden_size,
                shape.kv_features,
                normalized,
                value_raw,
            )?,
            output_projection: linear_launch(
                output_weight,
                shared.scratch,
                shared.scratch,
                tokens,
                shape.query_features,
                shape.hidden_size,
                context,
                projected,
            )?,
        });
    }

    for launch in &launches {
        validate_launch_regions(
            &regions,
            &[
                launch.query_projection,
                launch.key_projection,
                launch.value_projection,
                launch.output_projection,
            ],
        )?;
    }

    let argument_encoder = attention.new_binding_encoder();
    let binding_command = MetalDeviceCommand::operation(
        "vnext_causal_paged_attention_bindings",
        binding_regions,
        move |_encoder, regions| {
            encode_page_bindings(&argument_encoder, binding_layout, &page_bindings, regions)
        },
    )
    .map_err(|error| error.to_string())?;

    let compute_command = MetalDeviceCommand::operation(
        "vnext_causal_paged_attention",
        regions,
        move |encoder, regions| {
            for launch in &launches {
                enqueue_attention(
                    &attention,
                    &linear,
                    &primitives,
                    encoder.compute_encoder(),
                    regions,
                    shared,
                    launch,
                );
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?;

    Ok(EncodedDeviceOperation::compute(compute_command).with_dynamic_binding(binding_command))
}

fn encode_page_bindings(
    encoder: &ArgumentEncoder,
    layout: BindingLayout,
    bindings: &[PageBinding],
    regions: &[MetalBufferRegion],
) -> Result<(), MetalDeviceRuntimeError> {
    let workspace = regions.first().ok_or_else(|| {
        MetalDeviceRuntimeError::contract("Metal causal-attention binding command has no workspace")
    })?;
    if workspace.element_type() != ElementType::U8
        || workspace.length_bytes() < layout.required_bytes
    {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal causal-attention binding workspace differs from its estimate",
        ));
    }
    for binding in bindings {
        let page_end = binding
            .first_page_region
            .checked_add(binding.page_count)
            .ok_or_else(|| {
                MetalDeviceRuntimeError::contract(
                    "Metal causal-attention page region range overflows",
                )
            })?;
        let pages = regions
            .get(binding.first_page_region..page_end)
            .ok_or_else(|| {
                MetalDeviceRuntimeError::contract("Metal causal-attention page regions are missing")
            })?;
        if pages.is_empty()
            || pages.iter().any(|page| {
                page.length_bytes() != VNEXT_KV_PAGE_BYTES
                    || page.element_type() != ElementType::F16
            })
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal causal-attention page regions changed after encoding",
            ));
        }
        let argument_offset = workspace
            .offset_bytes()
            .checked_add(binding.binding_offset)
            .ok_or_else(|| {
                MetalDeviceRuntimeError::contract(
                    "Metal causal-attention argument-buffer offset overflows",
                )
            })?;
        encoder.set_argument_buffer(workspace.buffer(), argument_offset);
        let buffers = pages
            .iter()
            .map(MetalBufferRegion::buffer)
            .collect::<Vec<_>>();
        let offsets = pages
            .iter()
            .map(MetalBufferRegion::offset_bytes)
            .collect::<Vec<_>>();
        encoder.set_buffers(0, &buffers, &offsets);
    }
    Ok(())
}

fn enqueue_attention(
    attention: &MetalCausalAttentionPipelines,
    linear: &MetalLinearPipelines,
    primitives: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    launch: &ParticipantLaunch,
) {
    let scratch = &regions[shared.scratch];
    dispatch_rms_norm_at(
        primitives,
        encoder,
        &regions[launch.input],
        0,
        &regions[shared.input_norm],
        scratch,
        launch.normalized,
        launch.params.tokens,
        launch.hidden_size,
        launch.params.epsilon,
    );
    for projection in [
        launch.query_projection,
        launch.key_projection,
        launch.value_projection,
    ] {
        dispatch_linear(linear, encoder, regions, projection);
    }
    dispatch_prepare(attention, encoder, regions, shared, launch);
    dispatch_attention(attention, encoder, regions, shared, launch);
    dispatch_linear(linear, encoder, regions, launch.output_projection);
    dispatch_residual_add_at(
        primitives,
        encoder,
        &regions[launch.input],
        0,
        scratch,
        launch.projected,
        &regions[launch.output],
        0,
        launch.residual_elements,
    );
}

fn dispatch_prepare(
    pipelines: &MetalCausalAttentionPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    launch: &ParticipantLaunch,
) {
    let scratch = &regions[shared.scratch];
    encoder.set_compute_pipeline_state(&pipelines.prepare);
    for (index, offset) in [launch.query_raw, launch.key_raw, launch.value_raw]
        .into_iter()
        .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_region_offset(encoder, 3, &regions[shared.query_norm], 0);
    set_region_offset(encoder, 4, &regions[shared.key_norm], 0);
    set_region_offset(encoder, 5, scratch, launch.query);
    set_region_offset(
        encoder,
        PREPARE_PAGE_TABLE_INDEX,
        &regions[shared.binding],
        launch.binding_offset,
    );
    set_params(encoder, 7, &launch.params);
    use_pages(encoder, regions, launch);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(launch.params.tokens),
            u64::from(launch.params.query_heads) + 2 * u64::from(launch.params.key_value_heads),
            1,
        ),
        MTLSize::new(SIMD_THREADS, 1, 1),
    );
}

fn dispatch_attention(
    pipelines: &MetalCausalAttentionPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    launch: &ParticipantLaunch,
) {
    let scratch = &regions[shared.scratch];
    encoder.set_compute_pipeline_state(&pipelines.attention);
    set_region_offset(encoder, 0, scratch, launch.query);
    set_region_offset(encoder, 1, scratch, launch.query_raw);
    set_region_offset(encoder, 2, scratch, launch.context);
    set_region_offset(
        encoder,
        ATTENTION_PAGE_TABLE_INDEX,
        &regions[shared.binding],
        launch.binding_offset,
    );
    set_params(encoder, 4, &launch.params);
    use_pages(encoder, regions, launch);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(launch.params.tokens),
            u64::from(launch.params.query_heads),
            1,
        ),
        MTLSize::new(SIMD_THREADS, 1, 1),
    );
}

fn use_pages(
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: &ParticipantLaunch,
) {
    let page_end = launch
        .first_page_region
        .checked_add(launch.page_count)
        .expect("validated Metal causal-attention page range overflowed during dispatch");
    let pages = regions
        .get(launch.first_page_region..page_end)
        .expect("validated Metal causal-attention page range changed during dispatch");
    for page in pages {
        encoder.use_resource(
            page.buffer(),
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );
    }
}

fn set_params(encoder: &ComputeCommandEncoderRef, index: u64, params: &CausalAttentionParams) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<CausalAttentionParams>() as u64,
        params as *const _ as *const c_void,
    );
}

fn set_region_offset(
    encoder: &ComputeCommandEncoderRef,
    index: u64,
    region: &MetalBufferRegion,
    extra_offset_bytes: u64,
) {
    encoder.set_buffer(
        index,
        Some(region.buffer()),
        region.offset_bytes() + extra_offset_bytes,
    );
}

fn paged_state_regions(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    state: &ResolvedValueBinding,
    expected_physical_bytes: u64,
) -> Result<Vec<MetalBufferRegion>, String> {
    let [component] = state.storage().components() else {
        return Err(
            "Metal causal-attention state requires one logical storage component".to_owned(),
        );
    };
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "Metal causal-attention state has no resource view".to_owned())?;
    if component.offset_bytes() != 0
        || component.element_type() != ElementType::F16
        || view.descriptor().element_type != ElementType::F16
        || view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().size_bytes != expected_physical_bytes
        || expected_physical_bytes == 0
        || !expected_physical_bytes.is_multiple_of(VNEXT_KV_PAGE_BYTES)
    {
        return Err("Metal causal-attention state is not its admitted fixed-page view".to_owned());
    }
    let translated = view
        .translate(0, expected_physical_bytes)
        .map_err(|error| error.to_string())?;
    let capacity = usize::try_from(expected_physical_bytes / VNEXT_KV_PAGE_BYTES)
        .map_err(|_| "Metal causal-attention page capacity exceeds usize".to_owned())?;
    let mut pages = Vec::with_capacity(capacity);
    let mut next_logical = 0_u64;
    for physical in translated.iter() {
        if physical.logical_offset_bytes() != next_logical
            || physical.length_bytes() == 0
            || !physical.length_bytes().is_multiple_of(VNEXT_KV_PAGE_BYTES)
        {
            return Err("Metal causal-attention paged translation lost block geometry".to_owned());
        }
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let mut offset = 0_u64;
        while offset < physical.length_bytes() {
            let start = range
                .start
                .checked_add(offset)
                .ok_or_else(|| "Metal causal-attention page offset overflows".to_owned())?;
            let end = start
                .checked_add(VNEXT_KV_PAGE_BYTES)
                .ok_or_else(|| "Metal causal-attention page range overflows".to_owned())?;
            let page = buffer
                .retained_region(start..end, retention.clone())
                .map_err(|error| error.to_string())?;
            if page.length_bytes() != VNEXT_KV_PAGE_BYTES || page.element_type() != ElementType::F16
            {
                return Err(
                    "Metal causal-attention physical page differs from its contract".to_owned(),
                );
            }
            pages.push(page);
            offset += VNEXT_KV_PAGE_BYTES;
        }
        next_logical = next_logical
            .checked_add(physical.length_bytes())
            .ok_or_else(|| "Metal causal-attention logical page coverage overflows".to_owned())?;
    }
    if next_logical != expected_physical_bytes || pages.is_empty() {
        return Err("Metal causal-attention pages do not cover admitted state".to_owned());
    }
    Ok(pages)
}

fn validate_signature(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    shape: CausalAttentionShape,
) -> Result<(), String> {
    let value = |ordinal| binding(participant.bindings(), ResolvedValueRole::Input, ordinal);
    let hidden = value(0)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let [tokens, hidden_width] = hidden.tensor().dimensions() else {
        return Err("Metal causal-attention hidden input is not two-dimensional".to_owned());
    };
    let expected = [
        (value(1)?, vec![shape.hidden_size]),
        (
            value(2)?,
            vec![shape.query_projection_features, shape.hidden_size],
        ),
        (value(3)?, vec![shape.kv_features, shape.hidden_size]),
        (value(4)?, vec![shape.kv_features, shape.hidden_size]),
        (value(5)?, vec![shape.hidden_size, shape.query_features]),
        (value(6)?, vec![shape.head_dim]),
        (value(7)?, vec![shape.head_dim]),
        (value(8)?, vec![2, shape.key_value_heads, shape.head_dim]),
    ];
    if *tokens == 0
        || *hidden_width != shape.hidden_size
        || output.tensor().dimensions() != [*tokens, shape.hidden_size]
        || !f16_contiguous(hidden)
        || !f16_contiguous(output)
        || expected.iter().any(|(binding, dimensions)| {
            binding.tensor().dimensions() != dimensions.as_slice() || !f16_contiguous(binding)
        })
    {
        return Err("Metal causal-attention signature differs from its shape".to_owned());
    }
    Ok(())
}

fn push_shared_region(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
) -> Result<usize, String> {
    let index = regions.len();
    regions.push(shared_full_region(
        invocation,
        ResolvedValueRole::Input,
        ordinal,
        ElementType::F16,
    )?);
    Ok(index)
}

fn bool_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<bool, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Bool(value)) => Ok(*value),
        _ => Err(format!(
            "Metal causal-attention provider lacks boolean attribute {name:?}"
        )),
    }
}

fn reserve_tokens(offset: &mut u64, elements: u64, tokens: u64) -> Result<u64, String> {
    let start = *offset;
    let stride = aligned_bytes(elements, ElementType::F16.size_bytes())?;
    *offset = offset
        .checked_add(
            stride
                .checked_mul(tokens)
                .ok_or_else(|| "Metal causal-attention scratch span overflows".to_owned())?,
        )
        .ok_or_else(|| "Metal causal-attention scratch offset overflows".to_owned())?;
    Ok(start)
}

fn aligned_bytes(elements: u64, element_bytes: u64) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| "Metal causal-attention byte count overflows".to_owned())?;
    align_up(bytes, VALUE_ALIGNMENT_BYTES)
}

fn align_up(bytes: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("Metal causal-attention alignment is not a power of two".to_owned());
    }
    bytes
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .filter(|value| *value > 0)
        .ok_or_else(|| "Metal causal-attention alignment overflows".to_owned())
}

#[cfg(test)]
mod shape_tests {
    use super::*;
    use ferrum_interfaces::vnext::CanonicalRational;

    fn qwen35_4b_attributes() -> BTreeMap<AttributeId, SemanticValue> {
        BTreeMap::from([
            (
                AttributeId::new("hidden_size").unwrap(),
                SemanticValue::Unsigned(2560),
            ),
            (
                AttributeId::new("query_heads").unwrap(),
                SemanticValue::Unsigned(16),
            ),
            (
                AttributeId::new("key_value_heads").unwrap(),
                SemanticValue::Unsigned(4),
            ),
            (
                AttributeId::new("head_dim").unwrap(),
                SemanticValue::Unsigned(256),
            ),
            (
                AttributeId::new("query_features").unwrap(),
                SemanticValue::Unsigned(4096),
            ),
            (
                AttributeId::new("query_projection_features").unwrap(),
                SemanticValue::Unsigned(8192),
            ),
            (
                AttributeId::new("kv_features").unwrap(),
                SemanticValue::Unsigned(1024),
            ),
            (
                AttributeId::new("rope_dim").unwrap(),
                SemanticValue::Unsigned(64),
            ),
            (
                AttributeId::new("maximum_context_tokens").unwrap(),
                SemanticValue::Unsigned(262_144),
            ),
            (
                AttributeId::new("epsilon").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
            ),
            (
                AttributeId::new("rope_theta").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(10_000, 1).unwrap()),
            ),
            (
                AttributeId::new("rope_interleaved").unwrap(),
                SemanticValue::Bool(false),
            ),
            (
                AttributeId::new("output_gate").unwrap(),
                SemanticValue::Bool(true),
            ),
            (
                AttributeId::new("causal").unwrap(),
                SemanticValue::Bool(true),
            ),
            (
                AttributeId::new("layer_index").unwrap(),
                SemanticValue::Unsigned(3),
            ),
        ])
    }

    #[test]
    fn qwen35_4b_shape_exactly_fits_fixed_page_capability() {
        let shape = CausalAttentionShape::from_attributes(&qwen35_4b_attributes()).unwrap();
        assert_eq!(shape.state_bytes_per_token().unwrap(), 4096);
        assert_eq!(shape.maximum_pages().unwrap(), MAXIMUM_KV_PAGES);
        assert_eq!(
            shape.physical_state_bytes(17).unwrap(),
            2 * VNEXT_KV_PAGE_BYTES
        );
        assert_eq!(
            ScratchLayout::new(shape, 3).unwrap().required_bytes,
            3 * shape.scratch_bytes_per_token().unwrap()
        );
    }
}

#[cfg(test)]
#[path = "causal_attention_tests.rs"]
mod conformance_tests;
