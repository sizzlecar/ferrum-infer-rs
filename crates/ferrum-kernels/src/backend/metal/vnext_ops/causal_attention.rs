//! Native Metal provider for the standard fixed-page causal attention operation.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    causal_paged_attention_contract, causal_paged_attention_f32_master_contract, AttributeId,
    BatchedOperationInvocation, DeviceBatchingForm, DeviceReusableExecutionTopologyFingerprint,
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView,
    ElementType, EncodedDeviceOperation, OperationBufferStorageKind, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, ResolvedTensorLayout,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, VNextError,
    CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, CAUSAL_PAGED_ATTENTION_F32_MASTER_CAPABILITY_ID,
    CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID, CAUSAL_PAGED_ATTENTION_OPERATION_ID,
};
use metal::{
    ArgumentEncoder, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Function, MTLArgumentBuffersTier, MTLResourceUsage, MTLSize,
};
use sha2::{Digest, Sha256};

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError, MetalSubmissionEncoder,
};
use super::linear::{
    append_shared_matrix_weight, dispatch_linear, linear_launch,
    validate_launch_regions_with_raw_workspace, LinearLaunch, MetalLinearPipelines,
};
use super::primitives::{
    dispatch_residual_add_at, dispatch_residual_add_f32_f16_at, dispatch_rms_norm_at,
    dispatch_rms_norm_f32_to_f16_at, MetalPrimitivePipelines,
};
use super::{
    authorize_reusable_topology, binding, checked_u32, contract_error, ensure_invocation,
    f16_contiguous, implementation_fingerprint, invalid_plan, provider_descriptor,
    provider_failure, rational_attribute, shared_binding_region, shared_full_region,
    shared_scratch_region, shared_token_region, token_binding_is_packed, unsigned_attribute,
    DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID, Q5_K_FORMAT_ID,
    Q6_K_FORMAT_ID, Q8_0_FORMAT_ID, VALUE_ALIGNMENT_BYTES, VNEXT_KV_PAGE_BYTES,
};

const SHADER_SOURCE: &str = include_str!("causal_attention.metal");
const PROVIDER_ID: &str = "provider.metal.causal_paged_attention.f16.native";
const ESTIMATOR_ID: &str = "resource-estimator.metal.causal_paged_attention.f16.native";
const F32_MASTER_PROVIDER_ID: &str = "provider.metal.causal_paged_attention.f32-master.native";
const F32_MASTER_ESTIMATOR_ID: &str =
    "resource-estimator.metal.causal_paged_attention.f32-master.native";
const PREPARE_KERNEL: &str = "vnext_causal_prepare_f16";
const ATTENTION_KERNEL: &str = "vnext_causal_attention_f16";
const DIRECT_DECODE_ATTENTION_KERNEL: &str = "vnext_causal_attention_decode_direct_f16";
const GROUPED_DECODE_PARTIAL_ATTENTION_KERNEL: &str =
    "vnext_causal_attention_decode_grouped_partial_f16";
const GROUPED_DECODE_REDUCE_ATTENTION_KERNEL: &str =
    "vnext_causal_attention_decode_grouped_reduce_f16";
const TILED_PREFILL_ATTENTION_KERNEL: &str = "vnext_causal_attention_prefill_tiled_f16";
const GQA_TILED_PREFILL_ATTENTION_KERNEL: &str = "vnext_causal_attention_prefill_gqa_tiled_f16";
const PREPARE_PAGE_TABLE_INDEX: u64 = 6;
const ATTENTION_PAGE_TABLE_INDEX: u64 = 3;
const SIMD_THREADS: u64 = 32;
const THREADGROUP_MEMORY_ALIGNMENT: u64 = 16;
const MAXIMUM_ATTENTION_SIMDGROUPS: u64 = 16;
const MAXIMUM_HEAD_DIM: u64 = 256;
const MAXIMUM_KV_PAGES: u64 = 16_384;
const TILED_PREFILL_QUERY_TILE: u32 = 8;
const TILED_PREFILL_KEY_TILE: u64 = 32;
const GQA_TILED_PREFILL_KEY_TILE: u64 = 64;
const TILED_PREFILL_SIMDGROUPS: u64 = 4;
const GQA_TILED_PREFILL_QUERY_HEADS: u32 = 2;
const GQA_TILED_PREFILL_SIMDGROUPS: u64 = 8;
const GROUPED_DECODE_PARTITIONS: u64 = 8;
const GROUPED_DECODE_MINIMUM_CONTEXT: u64 = GROUPED_DECODE_PARTITIONS * TILED_PREFILL_KEY_TILE;

pub(super) struct MetalCausalAttentionPipelines {
    prepare: ComputePipelineState,
    attention: ComputePipelineState,
    direct_decode_attention: ComputePipelineState,
    grouped_decode_partial_attention: ComputePipelineState,
    grouped_decode_reduce_attention: ComputePipelineState,
    tiled_prefill_attention: ComputePipelineState,
    gqa_tiled_prefill_attention: ComputePipelineState,
    prepare_function: Function,
    binding_encoded_length: u64,
    binding_alignment: u64,
    maximum_attention_simdgroups: u32,
    maximum_threadgroup_memory_length: u64,
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
        let direct_decode_attention_function = function(DIRECT_DECODE_ATTENTION_KERNEL)?;
        let grouped_decode_partial_attention_function =
            function(GROUPED_DECODE_PARTIAL_ATTENTION_KERNEL)?;
        let grouped_decode_reduce_attention_function =
            function(GROUPED_DECODE_REDUCE_ATTENTION_KERNEL)?;
        let tiled_prefill_attention_function = function(TILED_PREFILL_ATTENTION_KERNEL)?;
        let gqa_tiled_prefill_attention_function = function(GQA_TILED_PREFILL_ATTENTION_KERNEL)?;
        let prepare_encoder = prepare_function.new_argument_encoder(PREPARE_PAGE_TABLE_INDEX);
        let attention_encoder = attention_function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        let direct_decode_attention_encoder =
            direct_decode_attention_function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        let grouped_decode_partial_attention_encoder = grouped_decode_partial_attention_function
            .new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        let tiled_prefill_attention_encoder =
            tiled_prefill_attention_function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        let gqa_tiled_prefill_attention_encoder =
            gqa_tiled_prefill_attention_function.new_argument_encoder(ATTENTION_PAGE_TABLE_INDEX);
        let binding_encoded_length = prepare_encoder.encoded_length();
        let binding_alignment = prepare_encoder.alignment();
        if binding_encoded_length == 0
            || binding_alignment == 0
            || attention_encoder.encoded_length() != binding_encoded_length
            || attention_encoder.alignment() != binding_alignment
            || direct_decode_attention_encoder.encoded_length() != binding_encoded_length
            || direct_decode_attention_encoder.alignment() != binding_alignment
            || grouped_decode_partial_attention_encoder.encoded_length() != binding_encoded_length
            || grouped_decode_partial_attention_encoder.alignment() != binding_alignment
            || tiled_prefill_attention_encoder.encoded_length() != binding_encoded_length
            || tiled_prefill_attention_encoder.alignment() != binding_alignment
            || gqa_tiled_prefill_attention_encoder.encoded_length() != binding_encoded_length
            || gqa_tiled_prefill_attention_encoder.alignment() != binding_alignment
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
        let direct_decode_attention = pipeline(&direct_decode_attention_function)?;
        let grouped_decode_partial_attention =
            pipeline(&grouped_decode_partial_attention_function)?;
        let grouped_decode_reduce_attention = pipeline(&grouped_decode_reduce_attention_function)?;
        let tiled_prefill_attention = pipeline(&tiled_prefill_attention_function)?;
        let gqa_tiled_prefill_attention = pipeline(&gqa_tiled_prefill_attention_function)?;
        if prepare.thread_execution_width() != SIMD_THREADS
            || attention.thread_execution_width() != SIMD_THREADS
            || direct_decode_attention.thread_execution_width() != SIMD_THREADS
            || grouped_decode_partial_attention.thread_execution_width() != SIMD_THREADS
            || grouped_decode_reduce_attention.thread_execution_width() != SIMD_THREADS
            || tiled_prefill_attention.thread_execution_width() != SIMD_THREADS
            || gqa_tiled_prefill_attention.thread_execution_width() != SIMD_THREADS
        {
            return Err(MetalDeviceRuntimeError::contract(format!(
                "Metal causal attention requires {SIMD_THREADS}-lane SIMD execution, got prepare={} attention={} direct_decode={} grouped_decode_partial={} grouped_decode_reduce={} tiled_prefill={} gqa_tiled_prefill={}",
                prepare.thread_execution_width(),
                attention.thread_execution_width(),
                direct_decode_attention.thread_execution_width(),
                grouped_decode_partial_attention.thread_execution_width(),
                grouped_decode_reduce_attention.thread_execution_width(),
                tiled_prefill_attention.thread_execution_width(),
                gqa_tiled_prefill_attention.thread_execution_width()
            )));
        }
        let tiled_prefill_threads = SIMD_THREADS * TILED_PREFILL_SIMDGROUPS;
        if (grouped_decode_partial_attention.max_total_threads_per_threadgroup() as u64)
            < tiled_prefill_threads
        {
            return Err(MetalDeviceRuntimeError::contract(format!(
                "Metal grouped causal decode requires {tiled_prefill_threads} threads per threadgroup, pipeline supports {}",
                grouped_decode_partial_attention.max_total_threads_per_threadgroup()
            )));
        }
        if (grouped_decode_reduce_attention.max_total_threads_per_threadgroup() as u64)
            < SIMD_THREADS
        {
            return Err(MetalDeviceRuntimeError::contract(format!(
                "Metal grouped causal-decode reduction requires {SIMD_THREADS} threads per threadgroup, pipeline supports {}",
                grouped_decode_reduce_attention.max_total_threads_per_threadgroup()
            )));
        }
        if (tiled_prefill_attention.max_total_threads_per_threadgroup() as u64)
            < tiled_prefill_threads
        {
            return Err(MetalDeviceRuntimeError::contract(format!(
                "Metal tiled causal prefill requires {tiled_prefill_threads} threads per threadgroup, pipeline supports {}",
                tiled_prefill_attention.max_total_threads_per_threadgroup()
            )));
        }
        let gqa_tiled_prefill_threads = SIMD_THREADS * GQA_TILED_PREFILL_SIMDGROUPS;
        if (gqa_tiled_prefill_attention.max_total_threads_per_threadgroup() as u64)
            < gqa_tiled_prefill_threads
        {
            return Err(MetalDeviceRuntimeError::contract(format!(
                "Metal GQA tiled causal prefill requires {gqa_tiled_prefill_threads} threads per threadgroup, pipeline supports {}",
                gqa_tiled_prefill_attention.max_total_threads_per_threadgroup()
            )));
        }
        let maximum_attention_simdgroups = (attention
            .max_total_threads_per_threadgroup()
            .min(direct_decode_attention.max_total_threads_per_threadgroup())
            as u64
            / SIMD_THREADS)
            .clamp(1, MAXIMUM_ATTENTION_SIMDGROUPS)
            as u32;
        let maximum_threadgroup_memory_length = device.max_threadgroup_memory_length() as u64;
        Ok(Self {
            prepare,
            attention,
            direct_decode_attention,
            grouped_decode_partial_attention,
            grouped_decode_reduce_attention,
            tiled_prefill_attention,
            gqa_tiled_prefill_attention,
            prepare_function,
            binding_encoded_length,
            binding_alignment,
            maximum_attention_simdgroups,
            maximum_threadgroup_memory_length,
        })
    }

    fn attention_simdgroups_for_context(&self, context_positions: u64) -> u32 {
        let context_limit = u32::try_from(context_positions).unwrap_or(u32::MAX).max(1);
        self.maximum_attention_simdgroups.min(context_limit)
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
    operation_id: &'static str,
    hidden_type: ElementType,
    failure_stage: &'static str,
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
        Self::new_with_hidden_type(runtime, attention, linear, primitives, ElementType::F16)
    }

    pub(super) fn new_f32_master(
        runtime: &MetalDeviceRuntime,
        attention: Arc<MetalCausalAttentionPipelines>,
        linear: Arc<MetalLinearPipelines>,
        primitives: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        Self::new_with_hidden_type(runtime, attention, linear, primitives, ElementType::F32)
    }

    fn new_with_hidden_type(
        runtime: &MetalDeviceRuntime,
        attention: Arc<MetalCausalAttentionPipelines>,
        linear: Arc<MetalLinearPipelines>,
        primitives: Arc<MetalPrimitivePipelines>,
        hidden_type: ElementType,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let (contract, operation_id, provider_id, capability_id, estimator_id, failure_stage) =
            match hidden_type {
                ElementType::F16 => (
                    causal_paged_attention_contract().map_err(contract_error)?,
                    CAUSAL_PAGED_ATTENTION_OPERATION_ID,
                    PROVIDER_ID,
                    CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
                    ESTIMATOR_ID,
                    "metal.causal_paged_attention.encode",
                ),
                ElementType::F32 => (
                    causal_paged_attention_f32_master_contract().map_err(contract_error)?,
                    CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID,
                    F32_MASTER_PROVIDER_ID,
                    CAUSAL_PAGED_ATTENTION_F32_MASTER_CAPABILITY_ID,
                    F32_MASTER_ESTIMATOR_ID,
                    "metal.causal_paged_attention.f32_master.encode",
                ),
                _ => {
                    return Err(MetalDeviceRuntimeError::contract(
                        "Metal causal-attention hidden ABI supports only F16 or F32",
                    ))
                }
            };
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            provider_id,
            capability_id,
            estimator_id,
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
                provider_id.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            operation_id,
            hidden_type,
            failure_stage,
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
        if request.operation().id.as_str() != self.operation_id
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
            ProviderWorkspaceSizeFormula::affine(
                0,
                shape
                    .split_decode_partial_bytes_per_sequence()
                    .map_err(invalid_plan)?,
                shape.scratch_bytes_per_token().map_err(invalid_plan)?,
            )?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        let binding = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::actual_sequences(
                self.attention.binding_slot_bytes().map_err(invalid_plan)?,
            )?,
            self.attention.binding_alignment,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
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
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        authorize_reusable_topology(self.descriptor.execution_semantics(), || {
            if request
                .binding_reusable_address_scope(ResolvedValueRole::Input, 0)?
                .is_none()
                || request
                    .binding_reusable_address_scope(ResolvedValueRole::Output, 0)?
                    .is_none()
            {
                return Ok(ReusableExecutionTopology::EagerBoundary);
            }
            reusable_attention_topology(&request, self.operation_id)
                .map(ReusableExecutionTopology::Dynamic)
                .map_err(invalid_plan)
        })
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_attention(
            Arc::clone(&self.attention),
            Arc::clone(&self.linear),
            Arc::clone(&self.primitives),
            self.operation_id,
            self.hidden_type,
            invocation,
        )
        .map_err(|message| provider_failure(identity, self.failure_stage, message))
    }
}

fn reusable_attention_topology(
    request: &ReusableExecutionTopologyRequest<'_>,
    operation_id: &str,
) -> Result<DeviceReusableExecutionTopologyFingerprint, String> {
    if request.operation_id().as_str() != operation_id {
        return Err("Metal causal topology received another operation".to_owned());
    }
    let shape = CausalAttentionShape::from_attributes(request.attributes())?;
    let ranges = request.work_shape().participant_token_ranges();
    if ranges.is_empty() {
        return Err("Metal causal topology has no participant token ranges".to_owned());
    }

    const DOMAIN: &[u8] = b"ferrum.metal.causal-attention.reusable-topology.v1\0";
    let mut digest = Sha256::new();
    digest.update(DOMAIN);
    digest.update((ranges.len() as u64).to_le_bytes());
    digest.update(request.work_shape().immediate_tokens().to_le_bytes());
    for range in ranges {
        let source = range.source_token_range();
        if source.end > range.full_input_tokens()
            || range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("Metal causal topology exceeds its admitted context".to_owned());
        }
        digest.update(range.immediate_tokens().to_le_bytes());
        digest.update(source.start.to_le_bytes());
        digest.update(source.end.to_le_bytes());
    }
    Ok(DeviceReusableExecutionTopologyFingerprint::from_sha256(
        digest.finalize().into(),
    ))
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
        shape.params(1, 0, 1, 1)?;
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

    fn split_decode_partial_bytes_per_sequence(self) -> Result<u64, String> {
        let elements_per_partial = self
            .head_dim
            .checked_add(2)
            .ok_or_else(|| "Metal grouped-decode partial width overflows".to_owned())?;
        let elements = GROUPED_DECODE_PARTITIONS
            .checked_mul(self.query_heads)
            .and_then(|value| value.checked_mul(elements_per_partial))
            .ok_or_else(|| "Metal grouped-decode partial workspace overflows".to_owned())?;
        aligned_bytes(elements, std::mem::size_of::<f32>() as u64)
    }

    fn params(
        self,
        tokens: u64,
        position_start: u64,
        page_count: u64,
        attention_simdgroups: u32,
    ) -> Result<CausalAttentionParams, String> {
        if attention_simdgroups == 0
            || u64::from(attention_simdgroups) > MAXIMUM_ATTENTION_SIMDGROUPS
        {
            return Err("Metal causal-attention SIMDgroup count is unsupported".to_owned());
        }
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
            attention_simdgroups,
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
    attention_simdgroups: u32,
    epsilon: f32,
    rope_theta: f32,
}

#[derive(Debug, Clone, Copy)]
struct ScratchLayout {
    required_bytes: u64,
    split_decode: u64,
    split_decode_bytes_per_sequence: u64,
    normalized: u64,
    query_raw: u64,
    key_raw: u64,
    value_raw: u64,
    query: u64,
    context: u64,
    projected: u64,
}

impl ScratchLayout {
    fn new(
        shape: CausalAttentionShape,
        total_tokens: u64,
        participant_count: usize,
    ) -> Result<Self, String> {
        if total_tokens == 0 || participant_count == 0 {
            return Err("Metal causal-attention scratch cannot size empty work".to_owned());
        }
        let participant_count = u64::try_from(participant_count)
            .map_err(|_| "Metal causal-attention participant count exceeds u64".to_owned())?;
        let mut offset = 0_u64;
        let split_decode = offset;
        let split_decode_bytes_per_sequence = shape.split_decode_partial_bytes_per_sequence()?;
        offset = offset
            .checked_add(
                split_decode_bytes_per_sequence
                    .checked_mul(participant_count)
                    .ok_or_else(|| {
                        "Metal grouped-decode participant workspace overflows".to_owned()
                    })?,
            )
            .ok_or_else(|| "Metal grouped-decode scratch offset overflows".to_owned())?;
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
            .and_then(|bytes| {
                split_decode_bytes_per_sequence
                    .checked_mul(participant_count)
                    .and_then(|split_bytes| bytes.checked_add(split_bytes))
            })
            .ok_or_else(|| "Metal causal-attention scratch size overflows".to_owned())?;
        if offset != expected {
            return Err(
                "Metal causal-attention scratch layout differs from its estimate".to_owned(),
            );
        }
        Ok(Self {
            required_bytes: offset,
            split_decode,
            split_decode_bytes_per_sequence,
            normalized,
            query_raw,
            key_raw,
            value_raw,
            query,
            context,
            projected,
        })
    }

    fn split_decode_offset(self, participant_index: usize) -> Result<u64, String> {
        let participant_index = u64::try_from(participant_index)
            .map_err(|_| "Metal grouped-decode participant index exceeds u64".to_owned())?;
        let offset = self
            .split_decode
            .checked_add(
                self.split_decode_bytes_per_sequence
                    .checked_mul(participant_index)
                    .ok_or_else(|| {
                        "Metal grouped-decode participant offset overflows".to_owned()
                    })?,
            )
            .ok_or_else(|| "Metal grouped-decode participant offset overflows".to_owned())?;
        offset
            .checked_add(self.split_decode_bytes_per_sequence)
            .filter(|end| *end <= self.normalized)
            .map(|_| offset)
            .ok_or_else(|| "Metal grouped-decode participant range is invalid".to_owned())
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
    split_decode: u64,
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
struct PackedLaunch {
    input: usize,
    output: usize,
    normalized: u64,
    projected: u64,
    tokens: u32,
    hidden_size: u32,
    residual_elements: u32,
    epsilon: f32,
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
    operation_id: &'static str,
    hidden_type: ElementType,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, String> {
    ensure_invocation(&invocation, operation_id)?;
    let first = &invocation.participants()[0];
    let shape = CausalAttentionShape::from_attributes(first.attributes())?;
    validate_signature(first, shape, hidden_type)?;
    for participant in &invocation.participants()[1..] {
        if CausalAttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("Metal causal-attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape, hidden_type)?;
    }

    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::new(shape, total_tokens, invocation.participants().len())?;
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
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;

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
            hidden_type,
            if input_packed {
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
            hidden_type,
            if output_packed {
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
            split_decode: layout.split_decode_offset(participant_index)?,
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
            params: shape.params(
                tokens,
                source.start,
                page_count_u64,
                attention.attention_simdgroups_for_context(
                    source.start.checked_add(tokens).ok_or_else(|| {
                        "Metal causal-attention context extent overflowed".to_owned()
                    })?,
                ),
            )?,
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

    let packed = if input_packed && output_packed && launches.len() > 1 {
        let input = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            hidden_type,
            total_tokens,
        )?);
        let output = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            hidden_type,
            total_tokens,
        )?);
        let packed = PackedLaunch {
            input,
            output,
            normalized: layout.normalized,
            projected: layout.projected,
            tokens: checked_u32(total_tokens, "Metal packed causal-attention token count")?,
            hidden_size: checked_u32(
                shape.hidden_size,
                "Metal packed causal-attention hidden size",
            )?,
            residual_elements: checked_u32(
                total_tokens.checked_mul(shape.hidden_size).ok_or_else(|| {
                    "Metal packed causal-attention residual element count overflows".to_owned()
                })?,
                "Metal packed causal-attention residual elements",
            )?,
            epsilon: shape.epsilon,
            query_projection: linear_launch(
                query_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.query_projection_features,
                layout.normalized,
                layout.query_raw,
            )?,
            key_projection: linear_launch(
                key_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.kv_features,
                layout.normalized,
                layout.key_raw,
            )?,
            value_projection: linear_launch(
                value_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.kv_features,
                layout.normalized,
                layout.value_raw,
            )?,
            output_projection: linear_launch(
                output_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.query_features,
                shape.hidden_size,
                layout.context,
                layout.projected,
            )?,
        };
        validate_launch_regions_with_raw_workspace(
            &regions,
            &[
                packed.query_projection,
                packed.key_projection,
                packed.value_projection,
                packed.output_projection,
            ],
            &[shared.scratch],
        )?;
        Some(packed)
    } else {
        for launch in &launches {
            validate_launch_regions_with_raw_workspace(
                &regions,
                &[
                    launch.query_projection,
                    launch.key_projection,
                    launch.value_projection,
                    launch.output_projection,
                ],
                &[shared.scratch],
            )?;
        }
        None
    };

    let argument_encoder = attention.new_binding_encoder();
    let binding_command = MetalDeviceCommand::operation(
        "vnext_causal_paged_attention_bindings",
        binding_regions,
        move |_encoder, regions| {
            encode_page_bindings(&argument_encoder, binding_layout, &page_bindings, regions)
        },
    )
    .map_err(|error| error.to_string())?;

    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal causal-attention participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let packed_enabled = packed.is_some();
    let grouped_decode_reductions = launches
        .iter()
        .filter(|launch| {
            attention_dispatch_plan(&launch.params).kind == AttentionDispatchKind::GroupedDecode
        })
        .count() as u64;
    let dispatch_count = physical_dispatch_count(launches.len(), packed_enabled)
        .saturating_add(grouped_decode_reductions);
    let operation_label = if hidden_type == ElementType::F32 {
        "vnext_causal_paged_attention_f32_master"
    } else {
        "vnext_causal_paged_attention"
    };
    let compute_command =
        MetalDeviceCommand::operation(operation_label, regions, move |encoder, regions| {
            encoder.record_compute_dispatches(dispatch_count);
            if let Some(packed) = packed.as_ref() {
                enqueue_packed_attention(
                    &attention,
                    &linear,
                    &primitives,
                    hidden_type,
                    encoder,
                    regions,
                    shared,
                    packed,
                    &launches,
                );
            } else {
                for launch in &launches {
                    enqueue_attention(
                        &attention,
                        &linear,
                        &primitives,
                        hidden_type,
                        encoder,
                        regions,
                        shared,
                        launch,
                    );
                }
            }
            Ok(())
        })
        .map_err(|error| error.to_string())?
        .with_work_shape(
            if packed_enabled {
                DeviceBatchingForm::Packed
            } else if participant_count == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participant_count,
            token_count,
        )
        .map_err(|error| error.to_string())?;

    Ok(invocation.attach_binding_command(
        EncodedDeviceOperation::compute(compute_command),
        binding_command,
    ))
}

fn physical_dispatch_count(participant_count: usize, packed: bool) -> u64 {
    let participants = participant_count as u64;
    if packed {
        participants.saturating_mul(2).saturating_add(6)
    } else {
        participants.saturating_mul(8)
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_input_rms_norm(
    primitives: &MetalPrimitivePipelines,
    hidden_type: ElementType,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    input_offset_bytes: u64,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
) {
    match hidden_type {
        ElementType::F16 => dispatch_rms_norm_at(
            primitives,
            encoder,
            input,
            input_offset_bytes,
            weight,
            output,
            output_offset_bytes,
            rows,
            hidden_size,
            epsilon,
        ),
        ElementType::F32 => dispatch_rms_norm_f32_to_f16_at(
            primitives,
            encoder,
            input,
            input_offset_bytes,
            weight,
            output,
            output_offset_bytes,
            rows,
            hidden_size,
            epsilon,
        ),
        _ => unreachable!("validated Metal causal-attention hidden ABI"),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hidden_residual(
    primitives: &MetalPrimitivePipelines,
    hidden_type: ElementType,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    left_offset_bytes: u64,
    right: &MetalBufferRegion,
    right_offset_bytes: u64,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    elements: u32,
) {
    match hidden_type {
        ElementType::F16 => dispatch_residual_add_at(
            primitives,
            encoder,
            left,
            left_offset_bytes,
            right,
            right_offset_bytes,
            output,
            output_offset_bytes,
            elements,
        ),
        ElementType::F32 => dispatch_residual_add_f32_f16_at(
            primitives,
            encoder,
            left,
            left_offset_bytes,
            right,
            right_offset_bytes,
            output,
            output_offset_bytes,
            elements,
        ),
        _ => unreachable!("validated Metal causal-attention hidden ABI"),
    }
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
    hidden_type: ElementType,
    encoder: &mut MetalSubmissionEncoder,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    launch: &ParticipantLaunch,
) {
    let scratch = &regions[shared.scratch];
    dispatch_input_rms_norm(
        primitives,
        hidden_type,
        compute_subwork(encoder, "causal_attention.input_norm"),
        &regions[launch.input],
        0,
        &regions[shared.input_norm],
        scratch,
        launch.normalized,
        launch.params.tokens,
        launch.hidden_size,
        launch.params.epsilon,
    );
    for (projection, subwork_id) in [
        (launch.query_projection, "causal_attention.query_projection"),
        (launch.key_projection, "causal_attention.key_projection"),
        (launch.value_projection, "causal_attention.value_projection"),
    ] {
        dispatch_linear(
            linear,
            compute_subwork(encoder, subwork_id),
            regions,
            projection,
        );
    }
    dispatch_prepare(
        attention,
        compute_subwork(encoder, "causal_attention.prepare"),
        regions,
        shared,
        launch,
    );
    dispatch_attention(
        attention,
        compute_subwork(encoder, "causal_attention.core"),
        regions,
        shared,
        launch,
    );
    dispatch_linear(
        linear,
        compute_subwork(encoder, "causal_attention.output_projection"),
        regions,
        launch.output_projection,
    );
    dispatch_hidden_residual(
        primitives,
        hidden_type,
        compute_subwork(encoder, "causal_attention.residual_add"),
        &regions[launch.input],
        0,
        scratch,
        launch.projected,
        &regions[launch.output],
        0,
        launch.residual_elements,
    );
}

#[allow(clippy::too_many_arguments)]
fn enqueue_packed_attention(
    attention: &MetalCausalAttentionPipelines,
    linear: &MetalLinearPipelines,
    primitives: &MetalPrimitivePipelines,
    hidden_type: ElementType,
    encoder: &mut MetalSubmissionEncoder,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    packed: &PackedLaunch,
    participants: &[ParticipantLaunch],
) {
    let scratch = &regions[shared.scratch];
    dispatch_input_rms_norm(
        primitives,
        hidden_type,
        compute_subwork(encoder, "causal_attention.input_norm"),
        &regions[packed.input],
        0,
        &regions[shared.input_norm],
        scratch,
        packed.normalized,
        packed.tokens,
        packed.hidden_size,
        packed.epsilon,
    );
    for (projection, subwork_id) in [
        (packed.query_projection, "causal_attention.query_projection"),
        (packed.key_projection, "causal_attention.key_projection"),
        (packed.value_projection, "causal_attention.value_projection"),
    ] {
        dispatch_linear(
            linear,
            compute_subwork(encoder, subwork_id),
            regions,
            projection,
        );
    }
    for participant in participants {
        dispatch_prepare(
            attention,
            compute_subwork(encoder, "causal_attention.prepare"),
            regions,
            shared,
            participant,
        );
        dispatch_attention(
            attention,
            compute_subwork(encoder, "causal_attention.core"),
            regions,
            shared,
            participant,
        );
    }
    dispatch_linear(
        linear,
        compute_subwork(encoder, "causal_attention.output_projection"),
        regions,
        packed.output_projection,
    );
    dispatch_hidden_residual(
        primitives,
        hidden_type,
        compute_subwork(encoder, "causal_attention.residual_add"),
        &regions[packed.input],
        0,
        scratch,
        packed.projected,
        &regions[packed.output],
        0,
        packed.residual_elements,
    );
}

fn compute_subwork<'a>(
    encoder: &'a mut MetalSubmissionEncoder,
    subwork_id: &'static str,
) -> &'a ComputeCommandEncoderRef {
    encoder.begin_compute_subwork(subwork_id);
    encoder.compute_encoder()
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
    encoder.set_threadgroup_memory_length(0, 0);
    encoder.set_threadgroup_memory_length(1, 0);
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
    let plan = attention_dispatch_plan_with_memory_limit(
        &launch.params,
        pipelines.maximum_threadgroup_memory_length,
    );
    set_region_offset(encoder, 0, scratch, launch.query);
    set_region_offset(encoder, 1, scratch, launch.query_raw);
    set_region_offset(
        encoder,
        2,
        scratch,
        if plan.kind == AttentionDispatchKind::GroupedDecode {
            launch.split_decode
        } else {
            launch.context
        },
    );
    set_region_offset(
        encoder,
        ATTENTION_PAGE_TABLE_INDEX,
        &regions[shared.binding],
        launch.binding_offset,
    );
    set_params(encoder, 4, &launch.params);
    use_pages(encoder, regions, launch);
    encode_attention_dispatch(pipelines, encoder, plan);
    if plan.kind == AttentionDispatchKind::GroupedDecode {
        encoder.set_compute_pipeline_state(&pipelines.grouped_decode_reduce_attention);
        set_region_offset(encoder, 0, scratch, launch.split_decode);
        set_region_offset(encoder, 1, scratch, launch.query_raw);
        set_region_offset(encoder, 2, scratch, launch.context);
        set_params(encoder, 4, &launch.params);
        encoder.set_threadgroup_memory_length(0, grouped_decode_reduce_threadgroup_memory_bytes());
        encoder.set_threadgroup_memory_length(1, 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(launch.params.query_heads), 1, 1),
            MTLSize::new(SIMD_THREADS, 1, 1),
        );
        encoder.set_threadgroup_memory_length(0, 0);
        encoder.set_threadgroup_memory_length(1, 0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionDispatchKind {
    General,
    DirectDecode,
    GroupedDecode,
    GqaTiledPrefill,
    TiledPrefill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AttentionDispatchPlan {
    kind: AttentionDispatchKind,
    threadgroups: [u64; 3],
    threads_per_threadgroup: [u64; 3],
    threadgroup_memory_bytes: [u64; 2],
}

fn attention_dispatch_plan(params: &CausalAttentionParams) -> AttentionDispatchPlan {
    attention_dispatch_plan_with_memory_limit(params, u64::MAX)
}

fn attention_dispatch_plan_with_memory_limit(
    params: &CausalAttentionParams,
    maximum_threadgroup_memory_length: u64,
) -> AttentionDispatchPlan {
    if uses_grouped_decode(params) {
        grouped_decode_attention_dispatch_plan(params)
    } else if uses_direct_decode(params) {
        direct_decode_attention_dispatch_plan(params)
    } else if uses_gqa_tiled_prefill(params)
        && gqa_tiled_prefill_threadgroup_memory_bytes(params) <= maximum_threadgroup_memory_length
    {
        gqa_tiled_prefill_attention_dispatch_plan(params)
    } else if uses_tiled_prefill(params) {
        tiled_prefill_attention_dispatch_plan(params)
    } else {
        general_attention_dispatch_plan(params)
    }
}

fn grouped_decode_attention_dispatch_plan(params: &CausalAttentionParams) -> AttentionDispatchPlan {
    AttentionDispatchPlan {
        kind: AttentionDispatchKind::GroupedDecode,
        threadgroups: [
            GROUPED_DECODE_PARTITIONS,
            u64::from(params.key_value_heads),
            1,
        ],
        threads_per_threadgroup: [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
        threadgroup_memory_bytes: [
            tiled_prefill_half_threadgroup_bytes(params),
            tiled_prefill_float_threadgroup_bytes(params),
        ],
    }
}

fn direct_decode_attention_dispatch_plan(params: &CausalAttentionParams) -> AttentionDispatchPlan {
    AttentionDispatchPlan {
        kind: AttentionDispatchKind::DirectDecode,
        threadgroups: [u64::from(params.tokens), u64::from(params.query_heads), 1],
        threads_per_threadgroup: [SIMD_THREADS, u64::from(params.attention_simdgroups), 1],
        threadgroup_memory_bytes: [attention_threadgroup_memory_bytes(params), 0],
    }
}

fn general_attention_dispatch_plan(params: &CausalAttentionParams) -> AttentionDispatchPlan {
    AttentionDispatchPlan {
        kind: AttentionDispatchKind::General,
        threadgroups: [u64::from(params.tokens), u64::from(params.query_heads), 1],
        threads_per_threadgroup: [SIMD_THREADS, u64::from(params.attention_simdgroups), 1],
        threadgroup_memory_bytes: [attention_threadgroup_memory_bytes(params), 0],
    }
}

fn tiled_prefill_attention_dispatch_plan(params: &CausalAttentionParams) -> AttentionDispatchPlan {
    AttentionDispatchPlan {
        kind: AttentionDispatchKind::TiledPrefill,
        threadgroups: [
            u64::from(params.tokens).div_ceil(u64::from(TILED_PREFILL_QUERY_TILE)),
            u64::from(params.query_heads),
            1,
        ],
        threads_per_threadgroup: [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
        threadgroup_memory_bytes: [
            tiled_prefill_half_threadgroup_bytes(params),
            tiled_prefill_float_threadgroup_bytes(params),
        ],
    }
}

fn gqa_tiled_prefill_attention_dispatch_plan(
    params: &CausalAttentionParams,
) -> AttentionDispatchPlan {
    AttentionDispatchPlan {
        kind: AttentionDispatchKind::GqaTiledPrefill,
        threadgroups: [
            u64::from(params.tokens).div_ceil(u64::from(TILED_PREFILL_QUERY_TILE)),
            u64::from(params.query_heads / GQA_TILED_PREFILL_QUERY_HEADS),
            1,
        ],
        threads_per_threadgroup: [SIMD_THREADS, GQA_TILED_PREFILL_SIMDGROUPS, 1],
        threadgroup_memory_bytes: [
            gqa_tiled_prefill_half_threadgroup_bytes(params),
            gqa_tiled_prefill_float_threadgroup_bytes(params),
        ],
    }
}

fn encode_attention_dispatch(
    pipelines: &MetalCausalAttentionPipelines,
    encoder: &ComputeCommandEncoderRef,
    plan: AttentionDispatchPlan,
) {
    encoder.set_compute_pipeline_state(match plan.kind {
        AttentionDispatchKind::General => &pipelines.attention,
        AttentionDispatchKind::DirectDecode => &pipelines.direct_decode_attention,
        AttentionDispatchKind::GroupedDecode => &pipelines.grouped_decode_partial_attention,
        AttentionDispatchKind::GqaTiledPrefill => &pipelines.gqa_tiled_prefill_attention,
        AttentionDispatchKind::TiledPrefill => &pipelines.tiled_prefill_attention,
    });
    encoder.set_threadgroup_memory_length(0, plan.threadgroup_memory_bytes[0]);
    encoder.set_threadgroup_memory_length(1, plan.threadgroup_memory_bytes[1]);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            plan.threadgroups[0],
            plan.threadgroups[1],
            plan.threadgroups[2],
        ),
        MTLSize::new(
            plan.threads_per_threadgroup[0],
            plan.threads_per_threadgroup[1],
            plan.threads_per_threadgroup[2],
        ),
    );
    encoder.set_threadgroup_memory_length(0, 0);
    encoder.set_threadgroup_memory_length(1, 0);
}

fn uses_grouped_decode(params: &CausalAttentionParams) -> bool {
    let Some(query_heads_per_kv_head) = query_heads_per_kv_head(params) else {
        return false;
    };
    params.tokens == 1
        && matches!(params.head_dim, 128 | 256)
        && matches!(query_heads_per_kv_head, 4 | TILED_PREFILL_QUERY_TILE)
        && u64::from(params.position_start).saturating_add(u64::from(params.tokens))
            >= GROUPED_DECODE_MINIMUM_CONTEXT
        && page_supports_eight_token_matrix(params)
}

fn uses_direct_decode(params: &CausalAttentionParams) -> bool {
    params.tokens == 1
        && matches!(params.head_dim, 128 | 256)
        && query_heads_per_kv_head(params).is_some()
        && page_holds_whole_token_rows(params)
}

fn query_heads_per_kv_head(params: &CausalAttentionParams) -> Option<u32> {
    (params.key_value_heads != 0
        && params.query_heads != 0
        && params.query_heads.is_multiple_of(params.key_value_heads))
    .then(|| params.query_heads / params.key_value_heads)
}

fn uses_gqa_tiled_prefill(params: &CausalAttentionParams) -> bool {
    let Some(query_heads_per_kv_head) = query_heads_per_kv_head(params) else {
        return false;
    };
    params.tokens >= TILED_PREFILL_QUERY_TILE
        && params.head_dim == 256
        && query_heads_per_kv_head >= GQA_TILED_PREFILL_QUERY_HEADS
        && query_heads_per_kv_head.is_multiple_of(GQA_TILED_PREFILL_QUERY_HEADS)
        && page_supports_eight_token_matrix(params)
}

fn uses_tiled_prefill(params: &CausalAttentionParams) -> bool {
    if !matches!(params.head_dim, 128 | 256)
        || params.tokens < TILED_PREFILL_QUERY_TILE
        || query_heads_per_kv_head(params).is_none()
    {
        return false;
    }
    page_supports_eight_token_matrix(params)
}

fn page_supports_eight_token_matrix(params: &CausalAttentionParams) -> bool {
    let Some(token_stride) = key_value_token_stride(params) else {
        return false;
    };
    let page_elements = u64::from(params.page_elements);
    page_holds_whole_token_rows(params)
        && (page_elements / token_stride).is_multiple_of(TILED_PREFILL_QUERY_TILE.into())
}

fn page_holds_whole_token_rows(params: &CausalAttentionParams) -> bool {
    let Some(token_stride) = key_value_token_stride(params) else {
        return false;
    };
    let page_elements = u64::from(params.page_elements);
    page_elements >= token_stride && page_elements.is_multiple_of(token_stride)
}

fn key_value_token_stride(params: &CausalAttentionParams) -> Option<u64> {
    2_u64
        .checked_mul(u64::from(params.key_value_heads))
        .and_then(|stride| stride.checked_mul(u64::from(params.head_dim)))
        .filter(|stride| *stride != 0)
}

fn tiled_prefill_half_threadgroup_bytes(params: &CausalAttentionParams) -> u64 {
    tiled_prefill_half_threadgroup_bytes_for_key_tile(params, TILED_PREFILL_KEY_TILE)
}

fn tiled_prefill_half_threadgroup_bytes_for_key_tile(
    params: &CausalAttentionParams,
    key_tile: u64,
) -> u64 {
    let head_dim = u64::from(params.head_dim);
    ((u64::from(TILED_PREFILL_QUERY_TILE) * head_dim)
        + (u64::from(TILED_PREFILL_QUERY_TILE) * key_tile))
        * std::mem::size_of::<half::f16>() as u64
}

fn tiled_prefill_float_threadgroup_bytes(params: &CausalAttentionParams) -> u64 {
    tiled_prefill_float_threadgroup_bytes_for_key_tile(params, TILED_PREFILL_KEY_TILE)
}

fn tiled_prefill_float_threadgroup_bytes_for_key_tile(
    params: &CausalAttentionParams,
    key_tile: u64,
) -> u64 {
    let head_dim = u64::from(params.head_dim);
    ((u64::from(TILED_PREFILL_QUERY_TILE) * head_dim)
        + (u64::from(TILED_PREFILL_QUERY_TILE) * key_tile))
        * std::mem::size_of::<f32>() as u64
}

fn gqa_tiled_prefill_half_threadgroup_bytes(params: &CausalAttentionParams) -> u64 {
    u64::from(GQA_TILED_PREFILL_QUERY_HEADS)
        * tiled_prefill_half_threadgroup_bytes_for_key_tile(params, GQA_TILED_PREFILL_KEY_TILE)
}

fn gqa_tiled_prefill_float_threadgroup_bytes(params: &CausalAttentionParams) -> u64 {
    u64::from(GQA_TILED_PREFILL_QUERY_HEADS)
        * tiled_prefill_float_threadgroup_bytes_for_key_tile(params, GQA_TILED_PREFILL_KEY_TILE)
}

fn gqa_tiled_prefill_threadgroup_memory_bytes(params: &CausalAttentionParams) -> u64 {
    gqa_tiled_prefill_half_threadgroup_bytes(params)
        .saturating_add(gqa_tiled_prefill_float_threadgroup_bytes(params))
}

fn grouped_decode_reduce_threadgroup_memory_bytes() -> u64 {
    aligned_threadgroup_memory_bytes(
        (GROUPED_DECODE_PARTITIONS + 1) * std::mem::size_of::<f32>() as u64,
    )
}

fn attention_threadgroup_memory_bytes(params: &CausalAttentionParams) -> u64 {
    let simdgroups = u64::from(params.attention_simdgroups);
    let values = simdgroups * u64::from(params.head_dim) + 3 * simdgroups;
    aligned_threadgroup_memory_bytes(values * std::mem::size_of::<f32>() as u64)
}

fn aligned_threadgroup_memory_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(THREADGROUP_MEMORY_ALIGNMENT) * THREADGROUP_MEMORY_ALIGNMENT
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
    hidden_type: ElementType,
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
        || !contiguous(hidden, hidden_type)
        || !contiguous(output, hidden_type)
        || expected.iter().any(|(binding, dimensions)| {
            binding.tensor().dimensions() != dimensions.as_slice() || !f16_contiguous(binding)
        })
    {
        return Err("Metal causal-attention signature differs from its shape".to_owned());
    }
    Ok(())
}

fn contiguous(binding: &ResolvedValueBinding, element_type: ElementType) -> bool {
    binding.tensor().element_type() == element_type
        && matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
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
            ScratchLayout::new(shape, 3, 1).unwrap().required_bytes,
            shape.split_decode_partial_bytes_per_sequence().unwrap()
                + 3 * shape.scratch_bytes_per_token().unwrap()
        );
    }

    #[test]
    fn split_decode_scratch_scales_per_sequence_without_overlapping_token_scratch() {
        let shape = CausalAttentionShape::from_attributes(&qwen35_4b_attributes()).unwrap();
        let split_stride = shape.split_decode_partial_bytes_per_sequence().unwrap();
        let token_stride = shape.scratch_bytes_per_token().unwrap();
        assert_eq!(split_stride, 132_096);

        for participant_count in [1_usize, 3] {
            for total_tokens in [1_u64, 2_048] {
                let layout = ScratchLayout::new(shape, total_tokens, participant_count).unwrap();
                assert_eq!(
                    layout.required_bytes,
                    participant_count as u64 * split_stride + total_tokens * token_stride,
                );
                assert_eq!(layout.normalized, participant_count as u64 * split_stride,);

                let mut previous_end = layout.split_decode;
                for participant_index in 0..participant_count {
                    let start = layout.split_decode_offset(participant_index).unwrap();
                    let end = start.checked_add(split_stride).unwrap();
                    assert!(start >= previous_end);
                    assert!(end <= layout.normalized);
                    previous_end = end;
                }
                assert_eq!(previous_end, layout.normalized);
                assert!(layout.split_decode_offset(participant_count).is_err());
            }
        }
    }

    #[test]
    fn packed_dispatch_count_keeps_only_sequence_local_kernels_per_participant() {
        assert_eq!(physical_dispatch_count(1, false), 8);
        assert_eq!(physical_dispatch_count(4, false), 32);
        assert_eq!(physical_dispatch_count(4, true), 14);
    }
}

#[cfg(test)]
#[path = "causal_attention_tests.rs"]
mod conformance_tests;
