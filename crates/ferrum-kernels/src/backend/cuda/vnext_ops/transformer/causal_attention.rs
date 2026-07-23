//! CUDA provider for the standard causal paged-attention operation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    causal_paged_attention_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceBatchingForm, DeviceRuntime, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView, ElementType,
    EncodedDeviceOperation, OperationBufferStorageKind, OperationContract, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase, ProviderId,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole,
    SemanticValue, VNextError, WeightFormatId, CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID,
    CAUSAL_PAGED_ATTENTION_OPERATION_ID,
};

use super::{ensure_estimator_request, estimate, launch_gemm_f16};
use crate::backend::cuda::vnext_ops::{
    binding, contiguous_token_region, contract_error, implementation_fingerprint,
    DENSE_SAFETENSORS_FORMAT_ID, THREADS_PER_BLOCK, VNEXT_KV_PAGE_BYTES,
};
use crate::backend::cuda::vnext_replay::CudaCommandReplayKeyBuilder;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};

const PROVIDER_ID: &str = "provider.cuda.causal_paged_attention.f16";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.causal_paged_attention.f16";
const RMS_NORM_FUNCTION: &str = "rms_norm_f16";
const PREPARE_FUNCTION: &str = "vnext_causal_prepare_f16";
const ATTENTION_FUNCTION: &str = "vnext_causal_attention_f16";
const RESIDUAL_ADD_FUNCTION: &str = "residual_add_f16";
const RESIDUAL_ADD_INPLACE_FUNCTION: &str = "residual_add_inplace_f16";
const SCRATCH_ALIGNMENT: u64 = 16;
const POINTER_BYTES: u64 = std::mem::size_of::<u64>() as u64;
const BINDING_CONTROL_WORDS: usize = 4;
const BINDING_CONTROL_BYTES: u64 = (BINDING_CONTROL_WORDS * std::mem::size_of::<i32>()) as u64;
const WARP_THREADS: u32 = 32;
const MAXIMUM_HEAD_DIM: u64 = 256;

pub(in crate::backend::cuda::vnext_ops) struct CudaCausalPagedAttentionProvider {
    descriptor: OperationProviderDescriptor,
    functions: CausalAttentionFunctions,
}

#[derive(Clone)]
struct CausalAttentionFunctions {
    rms_norm: CudaFunction,
    prepare: CudaFunction,
    attention: CudaFunction,
    residual_add: CudaFunction,
    residual_add_inplace: CudaFunction,
}

impl CudaCausalPagedAttentionProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = causal_paged_attention_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID).map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime does not advertise causal paged attention",
            ));
        }

        let source = include_str!("causal_attention.rs");
        let provider_fingerprint = implementation_fingerprint(&[
            source.as_bytes(),
            crate::ptx::RMS_NORM.as_bytes(),
            crate::ptx::VNEXT_CAUSAL_ATTENTION.as_bytes(),
            crate::ptx::RESIDUAL_ADD.as_bytes(),
        ]);
        let estimator_fingerprint =
            implementation_fingerprint(&[source.as_bytes(), ESTIMATOR_ID.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            provider_fingerprint,
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?
            ]),
            BTreeSet::new(),
            storage_bindings().map_err(contract_error)?,
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?;

        let rms_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RMS_NORM.to_owned()))
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("causal attention RMSNorm module", error)
            })?;
        let attention_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::VNEXT_CAUSAL_ATTENTION.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("causal attention module", error))?;
        let residual_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RESIDUAL_ADD.to_owned()))
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("causal attention residual module", error)
            })?;
        let functions = CausalAttentionFunctions {
            rms_norm: load_function(&rms_module, RMS_NORM_FUNCTION, "causal attention RMSNorm")?,
            prepare: load_function(
                &attention_module,
                PREPARE_FUNCTION,
                "causal attention prepare",
            )?,
            attention: load_function(&attention_module, ATTENTION_FUNCTION, "causal attention")?,
            residual_add: load_function(
                &residual_module,
                RESIDUAL_ADD_FUNCTION,
                "causal attention residual",
            )?,
            residual_add_inplace: load_function(
                &residual_module,
                RESIDUAL_ADD_INPLACE_FUNCTION,
                "causal attention in-place residual",
            )?,
        };
        Ok(Self {
            descriptor,
            functions,
        })
    }
}

fn load_function(
    module: &Arc<cudarc::driver::CudaModule>,
    name: &str,
    operation: &'static str,
) -> Result<CudaFunction, CudaDeviceRuntimeError> {
    module
        .load_function(name)
        .map_err(|error| CudaDeviceRuntimeError::driver(operation, error))
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

impl OperationResourceEstimator for CudaCausalPagedAttentionProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(
            &self.descriptor,
            &request,
            CAUSAL_PAGED_ATTENTION_OPERATION_ID,
        )?;
        let shape =
            CausalAttentionShape::from_attributes(request.attributes()).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::tokens(
                shape.scratch_bytes_per_token().map_err(invalid_plan)?,
            )?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            DynamicStorageRequirement::contiguous(),
        )?;
        let binding = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::actual_sequences(
                shape.binding_slot_bytes().map_err(invalid_plan)?,
            )?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(
            estimate(&self.descriptor, request.input_fingerprint(), Some(scratch))
                .with_binding(binding),
        )
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaCausalPagedAttentionProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_attention(
            &self.functions,
            self.descriptor.provider_implementation_fingerprint(),
            invocation,
        )
        .map_err(|message| {
            OperationFailure::new(
                identity,
                ProfilePhase::Forward,
                "cuda.causal_paged_attention.encode",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued CUDA causal attention identity must be valid")
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
            return Err("causal attention requires causal=true".to_owned());
        }
        let query_features = shape
            .query_heads
            .checked_mul(shape.head_dim)
            .ok_or_else(|| "causal attention query width overflows".to_owned())?;
        let kv_features = shape
            .key_value_heads
            .checked_mul(shape.head_dim)
            .ok_or_else(|| "causal attention KV width overflows".to_owned())?;
        let projection_multiplier = if shape.output_gate { 2 } else { 1 };
        let query_projection_features = query_features
            .checked_mul(projection_multiplier)
            .ok_or_else(|| "causal attention query projection width overflows".to_owned())?;
        if shape.hidden_size == 0
            || shape.query_heads == 0
            || shape.key_value_heads == 0
            || shape.head_dim == 0
            || shape.rope_dim == 0
            || shape.maximum_context_tokens == 0
        {
            return Err("causal attention dimensions and context must be non-zero".to_owned());
        }
        checked_i32(
            shape.maximum_context_tokens,
            "causal attention maximum context",
        )?;
        if shape.query_heads % shape.key_value_heads != 0
            || shape.head_dim > MAXIMUM_HEAD_DIM
            || shape.rope_dim > shape.head_dim
            || shape.rope_dim % 2 != 0
            || shape.query_features != query_features
            || shape.kv_features != kv_features
            || shape.query_projection_features != query_projection_features
        {
            return Err("causal attention attributes are inconsistent".to_owned());
        }
        shape.cuda_shape()?;
        shape.maximum_pages()?;
        Ok(shape)
    }

    fn state_bytes_per_token(self) -> Result<u64, String> {
        self.kv_features
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| "causal attention KV bytes per token overflow".to_owned())
    }

    fn physical_state_bytes(self, tokens: u64) -> Result<u64, String> {
        let logical = self
            .state_bytes_per_token()?
            .checked_mul(tokens)
            .ok_or_else(|| "causal attention KV state size overflows".to_owned())?;
        align_up(logical, VNEXT_KV_PAGE_BYTES)
    }

    fn physical_state_bytes_for_source_frontier(
        self,
        source_end_tokens: u64,
        full_input_tokens: u64,
    ) -> Result<u64, String> {
        if source_end_tokens == 0 || source_end_tokens > full_input_tokens {
            return Err("causal attention source frontier exceeds its full input".to_owned());
        }
        self.physical_state_bytes(source_end_tokens)
    }

    fn maximum_pages(self) -> Result<u64, String> {
        Ok(self.physical_state_bytes(self.maximum_context_tokens)? / VNEXT_KV_PAGE_BYTES)
    }

    fn binding_slot_bytes(self) -> Result<u64, String> {
        BINDING_CONTROL_BYTES
            .checked_add(aligned_bytes(self.maximum_pages()?, POINTER_BYTES)?)
            .ok_or_else(|| "causal attention binding slot size overflows".to_owned())
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
                .ok_or_else(|| "causal attention token scratch size overflows".to_owned())
        })
    }

    fn cuda_shape(self) -> Result<CudaCausalAttentionShape, String> {
        Ok(CudaCausalAttentionShape {
            hidden_size: checked_i32(self.hidden_size, "causal attention hidden size")?,
            query_heads: checked_i32(self.query_heads, "causal attention query heads")?,
            key_value_heads: checked_i32(self.key_value_heads, "causal attention key/value heads")?,
            head_dim: checked_i32(self.head_dim, "causal attention head dimension")?,
            query_features: checked_i32(self.query_features, "causal attention query width")?,
            query_projection_features: checked_i32(
                self.query_projection_features,
                "causal attention query projection width",
            )?,
            kv_features: checked_i32(self.kv_features, "causal attention KV width")?,
            rope_dim: checked_i32(self.rope_dim, "causal attention RoPE width")?,
            epsilon: self.epsilon,
            rope_theta: self.rope_theta,
            rope_interleaved: i32::from(self.rope_interleaved),
            output_gate: i32::from(self.output_gate),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct CudaCausalAttentionShape {
    hidden_size: i32,
    query_heads: i32,
    key_value_heads: i32,
    head_dim: i32,
    query_features: i32,
    query_projection_features: i32,
    kv_features: i32,
    rope_dim: i32,
    epsilon: f32,
    rope_theta: f32,
    rope_interleaved: i32,
    output_gate: i32,
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
            return Err("causal attention scratch cannot be sized for empty work".to_owned());
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
            .ok_or_else(|| "causal attention scratch size overflows".to_owned())?;
        if offset != expected {
            return Err("causal attention scratch layout differs from its estimate".to_owned());
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
}

#[derive(Debug, Clone, Copy)]
struct BindingLayout {
    required_bytes: u64,
    slot_bytes: u64,
}

impl BindingLayout {
    fn new(shape: CausalAttentionShape, participant_count: usize) -> Result<Self, String> {
        if participant_count == 0 {
            return Err("causal attention binding cannot be sized for empty work".to_owned());
        }
        let participant_count = u64::try_from(participant_count)
            .map_err(|_| "causal attention participant count exceeds u64".to_owned())?;
        let slot_bytes = shape.binding_slot_bytes()?;
        let required_bytes = slot_bytes
            .checked_mul(participant_count)
            .ok_or_else(|| "causal attention binding workspace size overflows".to_owned())?;
        Ok(Self {
            required_bytes,
            slot_bytes,
        })
    }

    fn binding_offset(self, participant: usize) -> Result<u64, String> {
        self.slot_bytes
            .checked_mul(
                u64::try_from(participant)
                    .map_err(|_| "causal attention participant index exceeds u64".to_owned())?,
            )
            .filter(|offset| *offset < self.required_bytes)
            .ok_or_else(|| "causal attention binding offset exceeds its workspace".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct SharedRegions {
    input_norm: usize,
    query_weight: usize,
    key_weight: usize,
    value_weight: usize,
    output_weight: usize,
    query_norm: usize,
    key_norm: usize,
    scratch: usize,
    binding: usize,
}

#[derive(Debug, Clone, Copy)]
struct CausalAttentionLaunch {
    input_region: usize,
    output_region: usize,
    binding_offset: u64,
    tokens: u64,
    tokens_i32: i32,
}

#[derive(Debug, Clone, Copy)]
struct CausalAttentionBinding {
    first_page_region: usize,
    page_count: usize,
    host_binding: usize,
    binding_offset: u64,
}

fn encode_attention(
    functions: &CausalAttentionFunctions,
    provider_fingerprint: &str,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != CAUSAL_PAGED_ATTENTION_OPERATION_ID
    {
        return Err("CUDA causal attention received another or empty operation".to_owned());
    }
    let first = &invocation.participants()[0];
    let shape = CausalAttentionShape::from_attributes(first.attributes())?;
    validate_signature(first, shape)?;
    for participant in &invocation.participants()[1..] {
        if CausalAttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("CUDA causal attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape)?;
    }

    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::new(shape, total_tokens)?;
    let binding_layout = BindingLayout::new(shape, invocation.participants().len())?;
    let cuda = shape.cuda_shape()?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA causal attention participant ranges are incomplete".to_owned());
    }
    let input_shared =
        super::token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let output_shared = super::token_binding_is_shared(
        &invocation,
        ResolvedValueRole::Output,
        0,
        ElementType::F16,
    )?;

    let mut compute_regions = Vec::new();
    let shared = SharedRegions {
        input_norm: push_shared_weight(&mut compute_regions, &invocation, 1)?,
        query_weight: push_shared_weight(&mut compute_regions, &invocation, 2)?,
        key_weight: push_shared_weight(&mut compute_regions, &invocation, 3)?,
        value_weight: push_shared_weight(&mut compute_regions, &invocation, 4)?,
        output_weight: push_shared_weight(&mut compute_regions, &invocation, 5)?,
        query_norm: push_shared_weight(&mut compute_regions, &invocation, 6)?,
        key_norm: push_shared_weight(&mut compute_regions, &invocation, 7)?,
        scratch: {
            let index = compute_regions.len();
            compute_regions.push(super::shared_scratch_region(
                &invocation,
                layout.required_bytes,
            )?);
            index
        },
        binding: {
            let index = compute_regions.len();
            compute_regions.push(super::shared_binding_region(
                &invocation,
                binding_layout.required_bytes,
            )?);
            index
        },
    };

    let mut binding_regions = vec![compute_regions[shared.binding].clone()];
    let mut host_storage = Vec::with_capacity(invocation.participants().len());
    let mut launches = Vec::with_capacity(invocation.participants().len());
    let mut bindings = Vec::with_capacity(invocation.participants().len());
    for (participant_index, (participant, token_range)) in invocation
        .participants()
        .iter()
        .zip(token_ranges)
        .enumerate()
    {
        let tokens = token_range.immediate_tokens();
        let source = token_range.source_token_range();
        let packed = token_range.immediate_token_range();
        if source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
        {
            return Err("causal attention token range exceeds its admitted context".to_owned());
        }
        let input_region = compute_regions.len();
        compute_regions.push(contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::F16,
            if input_shared {
                packed.start
            } else {
                source.start
            },
            tokens,
        )?);
        let output_region = compute_regions.len();
        compute_regions.push(contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F16,
            if output_shared {
                packed.start
            } else {
                source.start
            },
            tokens,
        )?);

        let first_page_region = binding_regions.len();
        let state = binding(participant.bindings(), ResolvedValueRole::Input, 8)?;
        let pages = paged_state_regions(
            participant,
            state,
            shape.physical_state_bytes_for_source_frontier(
                source.end,
                token_range.full_input_tokens(),
            )?,
        )?;
        let page_count = u64::try_from(pages.len())
            .map_err(|_| "causal attention page count exceeds u64".to_owned())?;
        if page_count > shape.maximum_pages()? {
            return Err("causal attention page table exceeds its admitted maximum".to_owned());
        }
        let tokens_i32 = checked_i32(tokens, "causal attention participant token count")?;
        let position_start = checked_i32(source.start, "causal attention source position")?;
        let page_count_i32 = checked_i32(page_count, "causal attention page count")?;
        let host_binding = host_storage.len();
        host_storage.push(binding_payload(
            page_count_i32,
            position_start,
            tokens_i32,
            &pages,
        ));
        let page_count = pages.len();
        binding_regions.extend(pages);
        let binding_offset = binding_layout.binding_offset(participant_index)?;
        bindings.push(CausalAttentionBinding {
            first_page_region,
            page_count,
            host_binding,
            binding_offset,
        });
        launches.push(CausalAttentionLaunch {
            input_region,
            output_region,
            binding_offset,
            tokens,
            tokens_i32,
        });
    }

    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA causal attention participant count exceeds u32".to_owned())?;
    let binding_command = CudaDeviceCommand::operation_with_host_storage_and_blas(
        "vnext_causal_paged_attention_bindings",
        binding_regions,
        host_storage,
        move |stream, _blas, regions, host_storage| {
            enqueue_bindings(stream, binding_layout, &bindings, regions, host_storage)
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participant_count,
            total_tokens,
            0,
            u64::from(participant_count),
        )
    })
    .map_err(|error| error.to_string())?;

    let mut replay_key = CudaCommandReplayKeyBuilder::new(
        provider_fingerprint,
        "vnext_causal_paged_attention_compute",
    )
    .u64(shape.hidden_size)
    .u64(shape.query_heads)
    .u64(shape.key_value_heads)
    .u64(shape.head_dim)
    .u64(shape.query_features)
    .u64(shape.query_projection_features)
    .u64(shape.kv_features)
    .u64(shape.rope_dim)
    .u64(shape.maximum_context_tokens)
    .f32(shape.epsilon)
    .f32(shape.rope_theta)
    .boolean(shape.rope_interleaved)
    .boolean(shape.output_gate)
    .u64(total_tokens)
    .u64(layout.required_bytes)
    .u64(layout.normalized)
    .u64(layout.query_raw)
    .u64(layout.key_raw)
    .u64(layout.value_raw)
    .u64(layout.query)
    .u64(layout.context)
    .u64(layout.projected)
    .u64(binding_layout.required_bytes)
    .u64(binding_layout.slot_bytes)
    .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.input_region as u64)
            .u64(launch.output_region as u64)
            .u64(launch.binding_offset)
            .u64(launch.tokens)
            .i32(launch.tokens_i32);
    }

    let functions = functions.clone();
    let compute_command = CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_causal_paged_attention_compute",
        compute_regions,
        replay_key.finish(),
        move |stream, blas, regions| {
            for launch in &launches {
                enqueue_attention(
                    stream, blas, &functions, shape, cuda, layout, shared, *launch, regions,
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participant_count,
            total_tokens,
            u64::from(participant_count) * 8,
            0,
        )
    })
    .map_err(|error| error.to_string())?;

    Ok(EncodedDeviceOperation::compute(compute_command).with_dynamic_binding(binding_command))
}

fn enqueue_bindings(
    stream: &CudaStream,
    layout: BindingLayout,
    bindings: &[CausalAttentionBinding],
    regions: &[CudaBufferRegion],
    host_storage: &[Box<[u8]>],
) -> Result<(), CudaDeviceRuntimeError> {
    let binding_workspace = &regions[0];
    if binding_workspace.length_bytes() < layout.required_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "causal attention binding workspace is smaller than its admitted estimate",
        ));
    }
    for binding in bindings {
        let page_region_end = binding
            .first_page_region
            .checked_add(binding.page_count)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("causal attention page region range overflows")
            })?;
        if regions
            .get(binding.first_page_region..page_region_end)
            .is_none_or(|pages| {
                pages.iter().any(|page| {
                    page.length_bytes() != VNEXT_KV_PAGE_BYTES
                        || page.element_type() != ElementType::F16
                })
            })
        {
            return Err(CudaDeviceRuntimeError::contract(
                "causal attention page regions changed after encoding",
            ));
        }
        let payload = host_storage.get(binding.host_binding).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("causal attention binding payload is missing")
        })?;
        if payload.len() as u64 > layout.slot_bytes {
            return Err(CudaDeviceRuntimeError::contract(
                "causal attention binding payload exceeds its admitted slot",
            ));
        }
        let destination = scratch_pointer(binding_workspace.device_ptr(), binding.binding_offset)?;
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                destination,
                payload.as_ref(),
                stream.cu_stream(),
            )
        }
        .map_err(|error| {
            CudaDeviceRuntimeError::driver("causal attention binding upload", error)
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn enqueue_attention(
    stream: &CudaStream,
    blas: &CudaBlas,
    functions: &CausalAttentionFunctions,
    logical: CausalAttentionShape,
    cuda: CudaCausalAttentionShape,
    layout: ScratchLayout,
    shared: SharedRegions,
    launch: CausalAttentionLaunch,
    regions: &[CudaBufferRegion],
) -> Result<(), CudaDeviceRuntimeError> {
    let scratch = &regions[shared.scratch];
    if scratch.length_bytes() < layout.required_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "causal attention scratch is smaller than its admitted estimate",
        ));
    }
    let scratch_base = scratch.device_ptr();
    let binding = &regions[shared.binding];
    let control = scratch_pointer(binding.device_ptr(), launch.binding_offset)?;
    let page_table = control.checked_add(BINDING_CONTROL_BYTES).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("causal attention page-table pointer overflows")
    })?;

    let input = regions[launch.input_region].device_ptr();
    let output = regions[launch.output_region].device_ptr();
    let normalized = scratch_pointer(scratch_base, layout.normalized)?;
    let query_raw = scratch_pointer(scratch_base, layout.query_raw)?;
    let key_raw = scratch_pointer(scratch_base, layout.key_raw)?;
    let value_raw = scratch_pointer(scratch_base, layout.value_raw)?;
    let query = scratch_pointer(scratch_base, layout.query)?;
    let context = scratch_pointer(scratch_base, layout.context)?;
    let projected = scratch_pointer(scratch_base, layout.projected)?;

    launch_rms_norm(
        stream,
        &functions.rms_norm,
        input,
        regions[shared.input_norm].device_ptr(),
        normalized,
        launch.tokens,
        cuda.hidden_size,
        cuda.epsilon,
    )?;
    for (weight, destination, out_features, operation) in [
        (
            shared.query_weight,
            query_raw,
            cuda.query_projection_features,
            "causal attention Q GEMM",
        ),
        (
            shared.key_weight,
            key_raw,
            cuda.kv_features,
            "causal attention K GEMM",
        ),
        (
            shared.value_weight,
            value_raw,
            cuda.kv_features,
            "causal attention V GEMM",
        ),
    ] {
        launch_gemm_f16(
            blas,
            normalized,
            regions[weight].device_ptr(),
            destination,
            launch.tokens_i32,
            out_features,
            cuda.hidden_size,
            operation,
        )?;
    }
    launch_prepare(
        stream,
        &functions.prepare,
        query_raw,
        key_raw,
        value_raw,
        regions[shared.query_norm].device_ptr(),
        regions[shared.key_norm].device_ptr(),
        query,
        control,
        page_table,
        launch,
        cuda,
    )?;
    launch_attention(
        stream,
        &functions.attention,
        query,
        query_raw,
        control,
        page_table,
        context,
        launch,
        cuda,
    )?;
    launch_gemm_f16(
        blas,
        context,
        regions[shared.output_weight].device_ptr(),
        projected,
        launch.tokens_i32,
        cuda.hidden_size,
        cuda.query_features,
        "causal attention output GEMM",
    )?;
    let elements = launch
        .tokens
        .checked_mul(logical.hidden_size)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("causal residual size overflows"))?;
    launch_residual(
        stream,
        &functions.residual_add,
        &functions.residual_add_inplace,
        input,
        projected,
        output,
        elements,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_prepare(
    stream: &CudaStream,
    function: &CudaFunction,
    query_raw: u64,
    key_raw: u64,
    value_raw: u64,
    query_norm: u64,
    key_norm: u64,
    query: u64,
    control: u64,
    page_table: u64,
    launch: CausalAttentionLaunch,
    shape: CudaCausalAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let page_elements = checked_i32_runtime(
        VNEXT_KV_PAGE_BYTES / ElementType::F16.size_bytes(),
        "causal page elements",
    )?;
    let query_head_stride = shape
        .head_dim
        .checked_mul(if shape.output_gate != 0 { 2 } else { 1 })
        .ok_or_else(|| CudaDeviceRuntimeError::contract("causal query head stride overflows"))?;
    let combined_heads = shape
        .key_value_heads
        .checked_mul(2)
        .and_then(|heads| heads.checked_add(shape.query_heads))
        .ok_or_else(|| CudaDeviceRuntimeError::contract("causal prepare head count overflows"))?;
    let mut builder = stream.launch_builder(function);
    let pointers = [
        query_raw, key_raw, value_raw, query_norm, key_norm, query, control, page_table,
    ];
    for pointer in &pointers {
        builder.arg(pointer);
    }
    let dimensions = [
        page_elements,
        shape.query_heads,
        shape.key_value_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.query_projection_features,
        query_head_stride,
        shape.kv_features,
    ];
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    builder.arg(&shape.epsilon);
    builder.arg(&shape.rope_theta);
    builder.arg(&shape.rope_interleaved);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_u32_runtime(launch.tokens, "causal prepare token grid")?,
                u32::try_from(combined_heads).map_err(|_| {
                    CudaDeviceRuntimeError::contract("causal prepare head grid exceeds u32")
                })?,
                1,
            ),
            block_dim: (WARP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("causal attention prepare launch", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention(
    stream: &CudaStream,
    function: &CudaFunction,
    query: u64,
    query_raw: u64,
    control: u64,
    page_table: u64,
    output: u64,
    launch: CausalAttentionLaunch,
    shape: CudaCausalAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let page_elements = checked_i32_runtime(
        VNEXT_KV_PAGE_BYTES / ElementType::F16.size_bytes(),
        "causal page elements",
    )?;
    let mut builder = stream.launch_builder(function);
    let pointers = [query, query_raw, control, page_table, output];
    for pointer in &pointers {
        builder.arg(pointer);
    }
    let dimensions = [
        page_elements,
        shape.query_heads,
        shape.key_value_heads,
        shape.head_dim,
        shape.query_projection_features,
        shape.output_gate,
    ];
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_u32_runtime(launch.tokens, "causal attention token grid")?,
                u32::try_from(shape.query_heads).map_err(|_| {
                    CudaDeviceRuntimeError::contract("causal attention head grid exceeds u32")
                })?,
                1,
            ),
            block_dim: (WARP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("causal attention launch", error))
}

fn launch_rms_norm(
    stream: &CudaStream,
    function: &CudaFunction,
    input: u64,
    weight: u64,
    output: u64,
    tokens: u64,
    hidden_size: i32,
    epsilon: f32,
) -> Result<(), CudaDeviceRuntimeError> {
    let rows = checked_u32_runtime(tokens, "causal RMSNorm rows")?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&input);
    builder.arg(&weight);
    builder.arg(&output);
    builder.arg(&hidden_size);
    builder.arg(&epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: ((hidden_size as u32).min(1024), 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("causal attention RMSNorm launch", error))
}

fn launch_residual(
    stream: &CudaStream,
    function: &CudaFunction,
    inplace_function: &CudaFunction,
    input: u64,
    branch: u64,
    output: u64,
    elements: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let elements_i32 = checked_i32_runtime(elements, "causal residual elements")?;
    let grid = checked_u32_runtime(
        elements.div_ceil(u64::from(THREADS_PER_BLOCK)),
        "causal residual grid",
    )?;
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let result = if input == output {
        let mut builder = stream.launch_builder(inplace_function);
        builder.arg(&output);
        builder.arg(&branch);
        builder.arg(&elements_i32);
        unsafe { builder.launch(config) }
    } else {
        let mut builder = stream.launch_builder(function);
        builder.arg(&input);
        builder.arg(&branch);
        builder.arg(&output);
        builder.arg(&elements_i32);
        unsafe { builder.launch(config) }
    };
    result
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("causal attention residual launch", error))
}

fn paged_state_regions(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    state: &ResolvedValueBinding,
    expected_physical_bytes: u64,
) -> Result<Vec<CudaBufferRegion>, String> {
    let [component] = state.storage().components() else {
        return Err("causal attention state requires one logical storage component".to_owned());
    };
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "causal attention state has no resource view".to_owned())?;
    if component.offset_bytes() != 0
        || component.element_type() != ElementType::F16
        || view.descriptor().element_type != ElementType::F16
        || view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().size_bytes != expected_physical_bytes
        || expected_physical_bytes == 0
        || expected_physical_bytes % VNEXT_KV_PAGE_BYTES != 0
    {
        return Err("causal attention state is not its admitted fixed-block paged view".to_owned());
    }
    let translated = view
        .translate(0, expected_physical_bytes)
        .map_err(|error| error.to_string())?;
    let page_capacity = usize::try_from(expected_physical_bytes / VNEXT_KV_PAGE_BYTES)
        .map_err(|_| "causal attention page capacity exceeds usize".to_owned())?;
    let mut pages = Vec::with_capacity(page_capacity);
    let mut next_logical = 0_u64;
    for physical in translated.iter() {
        if physical.logical_offset_bytes() != next_logical
            || physical.length_bytes() == 0
            || physical.length_bytes() % VNEXT_KV_PAGE_BYTES != 0
        {
            return Err("causal attention paged translation lost block geometry".to_owned());
        }
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let mut offset = 0_u64;
        while offset < physical.length_bytes() {
            let start = range
                .start
                .checked_add(offset)
                .ok_or_else(|| "causal attention page offset overflows".to_owned())?;
            let end = start
                .checked_add(VNEXT_KV_PAGE_BYTES)
                .ok_or_else(|| "causal attention page range overflows".to_owned())?;
            let page = buffer
                .retained_region(start..end, retention.clone())
                .map_err(|error| error.to_string())?;
            if page.length_bytes() != VNEXT_KV_PAGE_BYTES || page.element_type() != ElementType::F16
            {
                return Err("causal attention physical page differs from its contract".to_owned());
            }
            pages.push(page);
            offset += VNEXT_KV_PAGE_BYTES;
        }
        next_logical = next_logical
            .checked_add(physical.length_bytes())
            .ok_or_else(|| "causal attention logical page coverage overflows".to_owned())?;
    }
    if next_logical != expected_physical_bytes || pages.is_empty() {
        return Err("causal attention pages do not cover the admitted state".to_owned());
    }
    Ok(pages)
}

fn binding_payload(
    page_count: i32,
    position_start: i32,
    active_tokens: i32,
    pages: &[CudaBufferRegion],
) -> Box<[u8]> {
    let mut payload = Vec::with_capacity(
        BINDING_CONTROL_BYTES as usize + pages.len() * std::mem::size_of::<u64>(),
    );
    for value in [page_count, position_start, active_tokens, 0] {
        payload.extend_from_slice(&value.to_ne_bytes());
    }
    for page in pages {
        payload.extend_from_slice(&page.device_ptr().to_ne_bytes());
    }
    payload.into_boxed_slice()
}

fn validate_signature(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    shape: CausalAttentionShape,
) -> Result<(), String> {
    let value = |ordinal| binding(participant.bindings(), ResolvedValueRole::Input, ordinal);
    let hidden = value(0)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let [tokens, hidden_width] = hidden.tensor().dimensions() else {
        return Err("causal attention hidden input is not two-dimensional".to_owned());
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
        return Err("causal attention signature differs from its resolved shape".to_owned());
    }
    Ok(())
}

fn push_shared_weight(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
) -> Result<usize, String> {
    let index = regions.len();
    regions.push(super::shared_full_region(
        invocation,
        ResolvedValueRole::Input,
        ordinal,
        ElementType::F16,
    )?);
    Ok(index)
}

fn f16_contiguous(binding: &ResolvedValueBinding) -> bool {
    binding.tensor().element_type() == ElementType::F16
        && matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

fn reserve_tokens(offset: &mut u64, elements: u64, tokens: u64) -> Result<u64, String> {
    let start = *offset;
    let stride = aligned_bytes(elements, ElementType::F16.size_bytes())?;
    *offset = offset
        .checked_add(
            stride
                .checked_mul(tokens)
                .ok_or_else(|| "causal attention scratch span overflows".to_owned())?,
        )
        .ok_or_else(|| "causal attention scratch offset overflows".to_owned())?;
    Ok(start)
}

fn aligned_bytes(elements: u64, element_bytes: u64) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| "causal attention byte count overflows".to_owned())?;
    align_up(bytes, SCRATCH_ALIGNMENT)
}

fn align_up(bytes: u64, alignment: u64) -> Result<u64, String> {
    bytes
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .filter(|value| *value > 0)
        .ok_or_else(|| "causal attention alignment overflows".to_owned())
}

fn scratch_pointer(base: u64, offset: u64) -> Result<u64, CudaDeviceRuntimeError> {
    base.checked_add(offset).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("causal attention scratch pointer overflows")
    })
}

fn checked_i32(value: u64, context: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{context} exceeds i32"))
}

fn checked_i32_runtime(value: u64, context: &'static str) -> Result<i32, CudaDeviceRuntimeError> {
    i32::try_from(value)
        .map_err(|_| CudaDeviceRuntimeError::contract(format!("{context} exceeds i32")))
}

fn checked_u32_runtime(value: u64, context: &'static str) -> Result<u32, CudaDeviceRuntimeError> {
    u32::try_from(value)
        .map_err(|_| CudaDeviceRuntimeError::contract(format!("{context} exceeds u32")))
}

fn unsigned_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<u64, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Unsigned(value)) => Ok(*value),
        _ => Err(format!(
            "CUDA causal attention lacks unsigned attribute {name:?}"
        )),
    }
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
            "CUDA causal attention lacks boolean attribute {name:?}"
        )),
    }
}

fn rational_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<f32, String> {
    let rational = match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Rational(value)) => *value,
        _ => {
            return Err(format!(
                "CUDA causal attention lacks rational attribute {name:?}"
            ))
        }
    };
    let value = (rational.numerator() as f64 / rational.denominator() as f64) as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "CUDA causal attention rational attribute {name:?} is not a positive f32"
        ));
    }
    Ok(value)
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::CanonicalRational;

    fn attributes(output_gate: bool) -> BTreeMap<AttributeId, SemanticValue> {
        BTreeMap::from([
            (
                AttributeId::new("hidden_size").unwrap(),
                SemanticValue::Unsigned(2048),
            ),
            (
                AttributeId::new("query_heads").unwrap(),
                SemanticValue::Unsigned(16),
            ),
            (
                AttributeId::new("key_value_heads").unwrap(),
                SemanticValue::Unsigned(2),
            ),
            (
                AttributeId::new("head_dim").unwrap(),
                SemanticValue::Unsigned(128),
            ),
            (
                AttributeId::new("query_features").unwrap(),
                SemanticValue::Unsigned(2048),
            ),
            (
                AttributeId::new("query_projection_features").unwrap(),
                SemanticValue::Unsigned(if output_gate { 4096 } else { 2048 }),
            ),
            (
                AttributeId::new("kv_features").unwrap(),
                SemanticValue::Unsigned(256),
            ),
            (
                AttributeId::new("rope_dim").unwrap(),
                SemanticValue::Unsigned(64),
            ),
            (
                AttributeId::new("maximum_context_tokens").unwrap(),
                SemanticValue::Unsigned(4096),
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
                SemanticValue::Bool(output_gate),
            ),
            (
                AttributeId::new("causal").unwrap(),
                SemanticValue::Bool(true),
            ),
            (
                AttributeId::new("layer_index").unwrap(),
                SemanticValue::Unsigned(0),
            ),
        ])
    }

    #[test]
    fn scratch_estimator_and_layout_are_identical() {
        let shape = CausalAttentionShape::from_attributes(&attributes(true)).unwrap();
        let layout = ScratchLayout::new(shape, 17).unwrap();
        let bindings = BindingLayout::new(shape, 3).unwrap();
        assert_eq!(
            layout.required_bytes,
            17 * shape.scratch_bytes_per_token().unwrap()
        );
        assert_eq!(
            bindings.required_bytes,
            3 * shape.binding_slot_bytes().unwrap()
        );
        assert_eq!(bindings.binding_offset(0).unwrap(), 0);
        assert_eq!(
            bindings.binding_offset(2).unwrap(),
            2 * shape.binding_slot_bytes().unwrap()
        );
        assert_eq!(shape.maximum_pages().unwrap(), 128);
    }

    #[test]
    fn chunked_prefill_state_tracks_the_source_frontier() {
        let shape = CausalAttentionShape::from_attributes(&attributes(true)).unwrap();
        let chunk_state = shape.physical_state_bytes(3).unwrap();
        let full_prompt_state = shape.physical_state_bytes(128).unwrap();

        assert_ne!(chunk_state, full_prompt_state);
        assert_eq!(
            shape
                .physical_state_bytes_for_source_frontier(3, 128)
                .unwrap(),
            chunk_state
        );
        assert!(shape
            .physical_state_bytes_for_source_frontier(129, 128)
            .is_err());
    }

    #[test]
    fn output_gate_changes_the_typed_query_projection_width() {
        let gated = CausalAttentionShape::from_attributes(&attributes(true)).unwrap();
        let plain = CausalAttentionShape::from_attributes(&attributes(false)).unwrap();
        assert_eq!(
            gated.query_projection_features,
            2 * plain.query_projection_features
        );
        let mut invalid = attributes(true);
        invalid.insert(
            AttributeId::new("query_projection_features").unwrap(),
            SemanticValue::Unsigned(2048),
        );
        assert!(CausalAttentionShape::from_attributes(&invalid).is_err());
    }
}
