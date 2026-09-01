//! CUDA provider for the typed GPT-OSS causal paged-attention operation.
//!
//! GPT-OSS attention is intentionally not routed through the standard causal
//! attention provider. Its projection biases, half-split YaRN rotation,
//! learned sink logits, and alternating local/full window are observable model
//! semantics and therefore have a separate provider identity and CUDA ABI.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    gpt_oss_causal_paged_attention_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceBatchingForm, DeviceReusableExecutionTopologyFingerprint, DeviceRuntime,
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView,
    ElementType, EncodedDeviceOperation, EncodedReusableExecutionBindings,
    OperationBufferStorageKind, OperationContract, OperationFailure, OperationInvocation,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase, ProviderId,
    ProviderStorageBindingRequirement, ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, ResolvedTensorLayout,
    ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, ReusableExecutionValueAddress,
    ReusableExecutionWorkspaceAddress, SemanticValue, VNextError, WeightFormatId,
    GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID, GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use super::{attach_invocation_binding, ensure_estimator_request, estimate, launch_gemm_f16};
use crate::backend::cuda::vnext_ops::{
    binding, contiguous_token_region, contract_error, implementation_fingerprint,
    DENSE_SAFETENSORS_FORMAT_ID, THREADS_PER_BLOCK, VNEXT_KV_PAGE_BYTES,
};
use crate::backend::cuda::vnext_replay::CudaCommandReplayKeyBuilder;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};

pub(in crate::backend::cuda::vnext_ops) const GPT_OSS_ATTENTION_PROVIDER_ID: &str =
    "provider.cuda.gpt_oss.causal_paged_attention.f16";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.gpt_oss.causal_paged_attention.f16";
const RMS_NORM_FUNCTION: &str = "rms_norm_f16";
const PREPARE_FUNCTION: &str = "gpt_oss_prepare_qkv_yarn_f16";
const ATTENTION_FUNCTION: &str = "gpt_oss_paged_attention_sink_f16";
const RESIDUAL_BIAS_FUNCTION: &str = "gpt_oss_residual_output_bias_f16";
const COMPUTE_OPERATION: &str = "vnext.gpt_oss.causal_paged_attention.f16";
const BINDING_OPERATION: &str = "vnext_gpt_oss_causal_paged_attention_bindings";

const SCRATCH_ALIGNMENT: u64 = 16;
const POINTER_BYTES: u64 = std::mem::size_of::<u64>() as u64;
const BINDING_CONTROL_WORDS: usize = 4;
const BINDING_CONTROL_BYTES: u64 = (BINDING_CONTROL_WORDS * std::mem::size_of::<i32>()) as u64;
const WARP_THREADS: u32 = 32;

const GPT_OSS_HEAD_DIM: u64 = 64;
const GPT_OSS_ROPE_DIM: u64 = 64;
const GPT_OSS_MAXIMUM_CONTEXT_TOKENS: u64 = 131_072;
const GPT_OSS_ROPE_THETA: f32 = 150_000.0;
const GPT_OSS_YARN_FACTOR: f32 = 32.0;
const GPT_OSS_YARN_ORIGINAL_CONTEXT: u64 = 4_096;
const GPT_OSS_YARN_BETA_FAST: f32 = 32.0;
const GPT_OSS_YARN_BETA_SLOW: f32 = 1.0;
const GPT_OSS_SLIDING_WINDOW: u64 = 128;

pub(in crate::backend::cuda::vnext_ops) struct CudaGptOssCausalPagedAttentionProvider {
    descriptor: OperationProviderDescriptor,
    functions: GptOssAttentionFunctions,
}

#[derive(Clone)]
struct GptOssAttentionFunctions {
    rms_norm: CudaFunction,
    prepare: CudaFunction,
    attention: CudaFunction,
    residual_bias: CudaFunction,
}

impl CudaGptOssCausalPagedAttentionProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = gpt_oss_causal_paged_attention_contract().map_err(contract_error)?;
        let capability = CapabilityId::new(GPT_OSS_CAUSAL_PAGED_ATTENTION_F16_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime does not advertise GPT-OSS causal paged attention",
            ));
        }

        let source = include_str!("gpt_oss_attention.rs");
        let provider_fingerprint = implementation_fingerprint(&[
            source.as_bytes(),
            crate::ptx::RMS_NORM.as_bytes(),
            crate::ptx::GPT_OSS_ATTENTION.as_bytes(),
            GPT_OSS_ATTENTION_PROVIDER_ID.as_bytes(),
        ]);
        let estimator_fingerprint =
            implementation_fingerprint(&[source.as_bytes(), ESTIMATOR_ID.as_bytes()]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(GPT_OSS_ATTENTION_PROVIDER_ID).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            provider_fingerprint,
            ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
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
                CudaDeviceRuntimeError::driver("GPT-OSS attention RMSNorm module", error)
            })?;
        let attention_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::GPT_OSS_ATTENTION.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS attention module", error))?;
        let functions = GptOssAttentionFunctions {
            rms_norm: load_function(&rms_module, RMS_NORM_FUNCTION, "GPT-OSS attention RMSNorm")?,
            prepare: load_function(
                &attention_module,
                PREPARE_FUNCTION,
                "GPT-OSS attention prepare",
            )?,
            attention: load_function(&attention_module, ATTENTION_FUNCTION, "GPT-OSS attention")?,
            residual_bias: load_function(
                &attention_module,
                RESIDUAL_BIAS_FUNCTION,
                "GPT-OSS attention residual+bias",
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
    Ok((0..12)
        .map(|ordinal| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                ordinal,
                if ordinal == 11 {
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

impl OperationResourceEstimator for CudaGptOssCausalPagedAttentionProvider {
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
            GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID,
        )?;
        let shape =
            GptOssAttentionShape::from_attributes(request.attributes()).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(
                0,
                0,
                shape.scratch_bytes_per_token().map_err(invalid_plan)?,
            )?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        let binding = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::actual_sequences(
                shape.binding_slot_bytes().map_err(invalid_plan)?,
            )?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(
            estimate(&self.descriptor, request.input_fingerprint(), Some(scratch))
                .with_binding(binding),
        )
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaGptOssCausalPagedAttentionProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        let mut values = (0..11)
            .map(|ordinal| {
                ReusableExecutionValueAddress::captured(ResolvedValueRole::Input, ordinal)
            })
            .collect::<Vec<_>>();
        values.extend([
            ReusableExecutionValueAddress::program_binding(ResolvedValueRole::Input, 11),
            ReusableExecutionValueAddress::captured(ResolvedValueRole::Output, 0),
        ]);
        if request
            .reusable_address_scope(
                &values,
                &[
                    ReusableExecutionWorkspaceAddress::Scratch,
                    ReusableExecutionWorkspaceAddress::Binding,
                ],
            )?
            .is_none()
        {
            return Ok(ReusableExecutionTopology::EagerBoundary);
        }
        gpt_oss_reusable_topology(&request).map_err(invalid_plan)
    }

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
                "cuda.gpt_oss.causal_paged_attention.encode",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued GPT-OSS attention identity must be valid")
        })
    }

    fn encode_reusable_execution_bindings(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_reusable_attention_bindings(invocation).map_err(|message| {
            OperationFailure::new(
                identity,
                ProfilePhase::Forward,
                "cuda.gpt_oss.causal_paged_attention.encode_reusable_bindings",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued GPT-OSS attention identity must be valid")
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct GptOssAttentionShape {
    hidden_size: u64,
    query_heads: u64,
    kv_heads: u64,
    head_dim: u64,
    query_features: u64,
    kv_features: u64,
    rope_dim: u64,
    maximum_context_tokens: u64,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_original_context_tokens: u64,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    sliding_window_tokens: u64,
    epsilon: f32,
    layer_index: u64,
}

impl GptOssAttentionShape {
    fn from_attributes(attributes: &BTreeMap<AttributeId, SemanticValue>) -> Result<Self, String> {
        let shape = Self {
            hidden_size: unsigned_attribute(attributes, "hidden_size")?,
            query_heads: unsigned_attribute(attributes, "query_heads")?,
            kv_heads: unsigned_attribute(attributes, "kv_heads")?,
            head_dim: unsigned_attribute(attributes, "head_dim")?,
            query_features: unsigned_attribute(attributes, "query_features")?,
            kv_features: unsigned_attribute(attributes, "kv_features")?,
            rope_dim: unsigned_attribute(attributes, "rope_dim")?,
            maximum_context_tokens: unsigned_attribute(attributes, "maximum_context_tokens")?,
            rope_theta: rational_attribute(attributes, "rope_theta")?,
            yarn_factor: rational_attribute(attributes, "yarn_factor")?,
            yarn_original_context_tokens: unsigned_attribute(
                attributes,
                "yarn_original_context_tokens",
            )?,
            yarn_beta_fast: rational_attribute(attributes, "yarn_beta_fast")?,
            yarn_beta_slow: rational_attribute(attributes, "yarn_beta_slow")?,
            sliding_window_tokens: unsigned_attribute(attributes, "sliding_window_tokens")?,
            epsilon: rational_attribute(attributes, "epsilon")?,
            layer_index: unsigned_attribute(attributes, "layer_index")?,
        };
        if !bool_attribute(attributes, "causal")? {
            return Err("GPT-OSS attention requires causal=true".to_owned());
        }
        if bool_attribute(attributes, "yarn_truncate")? {
            return Err("GPT-OSS attention requires yarn_truncate=false".to_owned());
        }
        let query_features = shape
            .query_heads
            .checked_mul(shape.head_dim)
            .ok_or_else(|| "GPT-OSS attention query width overflows".to_owned())?;
        let kv_features = shape
            .kv_heads
            .checked_mul(shape.head_dim)
            .ok_or_else(|| "GPT-OSS attention KV width overflows".to_owned())?;
        if shape.hidden_size == 0
            || shape.query_heads == 0
            || shape.kv_heads == 0
            || shape.query_heads % shape.kv_heads != 0
            || shape.query_features != query_features
            || shape.kv_features != kv_features
            || shape.hidden_size % (SCRATCH_ALIGNMENT / ElementType::F16.size_bytes()) != 0
        {
            return Err("GPT-OSS attention dimensions are inconsistent".to_owned());
        }
        if shape.head_dim != GPT_OSS_HEAD_DIM || shape.rope_dim != GPT_OSS_ROPE_DIM {
            return Err("GPT-OSS attention requires head_dim=rope_dim=64".to_owned());
        }
        if shape.maximum_context_tokens != GPT_OSS_MAXIMUM_CONTEXT_TOKENS {
            return Err("GPT-OSS attention requires maximum_context_tokens=131072".to_owned());
        }
        if shape.rope_theta.to_bits() != GPT_OSS_ROPE_THETA.to_bits()
            || shape.yarn_factor.to_bits() != GPT_OSS_YARN_FACTOR.to_bits()
            || shape.yarn_original_context_tokens != GPT_OSS_YARN_ORIGINAL_CONTEXT
            || shape.yarn_beta_fast.to_bits() != GPT_OSS_YARN_BETA_FAST.to_bits()
            || shape.yarn_beta_slow.to_bits() != GPT_OSS_YARN_BETA_SLOW.to_bits()
        {
            return Err("GPT-OSS attention YaRN attributes differ from the model ABI".to_owned());
        }
        if !matches!(shape.sliding_window_tokens, 0 | GPT_OSS_SLIDING_WINDOW) {
            return Err("GPT-OSS attention sliding window must be 0 or 128".to_owned());
        }
        if !shape.epsilon.is_finite() || shape.epsilon <= 0.0 {
            return Err("GPT-OSS attention epsilon must be finite and positive".to_owned());
        }
        shape.cuda_shape()?;
        shape.maximum_pages()?;
        Ok(shape)
    }

    fn state_bytes_per_token(self) -> Result<u64, String> {
        self.kv_features
            .checked_mul(2)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| "GPT-OSS attention KV bytes per token overflow".to_owned())
    }

    fn physical_state_bytes(self, tokens: u64) -> Result<u64, String> {
        if tokens == 0 || tokens > self.maximum_context_tokens {
            return Err("GPT-OSS attention state token count is invalid".to_owned());
        }
        let logical = self
            .state_bytes_per_token()?
            .checked_mul(tokens)
            .ok_or_else(|| "GPT-OSS attention KV state size overflows".to_owned())?;
        align_up(logical, VNEXT_KV_PAGE_BYTES)
    }

    fn maximum_pages(self) -> Result<u64, String> {
        Ok(self.physical_state_bytes(self.maximum_context_tokens)? / VNEXT_KV_PAGE_BYTES)
    }

    fn binding_slot_bytes(self) -> Result<u64, String> {
        BINDING_CONTROL_BYTES
            .checked_add(aligned_bytes(self.maximum_pages()?, POINTER_BYTES)?)
            .ok_or_else(|| "GPT-OSS attention binding slot overflows".to_owned())
    }

    fn scratch_bytes_per_token(self) -> Result<u64, String> {
        [
            self.hidden_size,
            self.query_features,
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
                .ok_or_else(|| "GPT-OSS attention token scratch size overflows".to_owned())
        })
    }

    fn cuda_shape(self) -> Result<CudaGptOssAttentionShape, String> {
        Ok(CudaGptOssAttentionShape {
            hidden_size: checked_i32(self.hidden_size, "GPT-OSS attention hidden size")?,
            query_heads: checked_i32(self.query_heads, "GPT-OSS attention query heads")?,
            kv_heads: checked_i32(self.kv_heads, "GPT-OSS attention KV heads")?,
            head_dim: checked_i32(self.head_dim, "GPT-OSS attention head dimension")?,
            query_features: checked_i32(self.query_features, "GPT-OSS attention query width")?,
            kv_features: checked_i32(self.kv_features, "GPT-OSS attention KV width")?,
            rope_dim: checked_i32(self.rope_dim, "GPT-OSS attention RoPE width")?,
            maximum_context_tokens: checked_i32(
                self.maximum_context_tokens,
                "GPT-OSS attention maximum context",
            )?,
            rope_theta: self.rope_theta,
            yarn_factor: self.yarn_factor,
            yarn_original_context_tokens: checked_i32(
                self.yarn_original_context_tokens,
                "GPT-OSS attention YaRN original context",
            )?,
            yarn_beta_fast: self.yarn_beta_fast,
            yarn_beta_slow: self.yarn_beta_slow,
            sliding_window_tokens: checked_i32(
                self.sliding_window_tokens,
                "GPT-OSS attention sliding window",
            )?,
            epsilon: self.epsilon,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct CudaGptOssAttentionShape {
    hidden_size: i32,
    query_heads: i32,
    kv_heads: i32,
    head_dim: i32,
    query_features: i32,
    kv_features: i32,
    rope_dim: i32,
    maximum_context_tokens: i32,
    rope_theta: f32,
    yarn_factor: f32,
    yarn_original_context_tokens: i32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    sliding_window_tokens: i32,
    epsilon: f32,
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
    fn new(shape: GptOssAttentionShape, total_tokens: u64) -> Result<Self, String> {
        if total_tokens == 0 {
            return Err("GPT-OSS attention scratch cannot be sized for empty work".to_owned());
        }
        let mut offset = 0;
        let normalized = reserve_tokens(&mut offset, shape.hidden_size, total_tokens)?;
        let query_raw = reserve_tokens(&mut offset, shape.query_features, total_tokens)?;
        let key_raw = reserve_tokens(&mut offset, shape.kv_features, total_tokens)?;
        let value_raw = reserve_tokens(&mut offset, shape.kv_features, total_tokens)?;
        let query = reserve_tokens(&mut offset, shape.query_features, total_tokens)?;
        let context = reserve_tokens(&mut offset, shape.query_features, total_tokens)?;
        let projected = reserve_tokens(&mut offset, shape.hidden_size, total_tokens)?;
        let expected = shape
            .scratch_bytes_per_token()?
            .checked_mul(total_tokens)
            .ok_or_else(|| "GPT-OSS attention scratch size overflows".to_owned())?;
        if offset != expected {
            return Err("GPT-OSS attention scratch layout differs from estimator".to_owned());
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

    fn token_offset(self, base: u64, token: u64, width: u64) -> Result<u64, String> {
        base.checked_add(
            aligned_bytes(width, ElementType::F16.size_bytes())?
                .checked_mul(token)
                .ok_or_else(|| "GPT-OSS attention token scratch offset overflows".to_owned())?,
        )
        .ok_or_else(|| "GPT-OSS attention token scratch pointer overflows".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct BindingLayout {
    required_bytes: u64,
    slot_bytes: u64,
}

impl BindingLayout {
    fn new(shape: GptOssAttentionShape, participant_count: usize) -> Result<Self, String> {
        if participant_count == 0 {
            return Err("GPT-OSS attention binding cannot be sized for empty work".to_owned());
        }
        let participant_count = u64::try_from(participant_count)
            .map_err(|_| "GPT-OSS attention participant count exceeds u64".to_owned())?;
        let slot_bytes = shape.binding_slot_bytes()?;
        let required_bytes = slot_bytes
            .checked_mul(participant_count)
            .ok_or_else(|| "GPT-OSS attention binding workspace size overflows".to_owned())?;
        Ok(Self {
            required_bytes,
            slot_bytes,
        })
    }

    fn binding_offset(self, participant: usize) -> Result<u64, String> {
        self.slot_bytes
            .checked_mul(
                u64::try_from(participant)
                    .map_err(|_| "GPT-OSS attention participant index exceeds u64".to_owned())?,
            )
            .filter(|offset| *offset < self.required_bytes)
            .ok_or_else(|| "GPT-OSS attention binding offset exceeds workspace".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct SharedRegions {
    input_norm: usize,
    query_weight: usize,
    key_weight: usize,
    value_weight: usize,
    output_weight: usize,
    query_bias: usize,
    key_bias: usize,
    value_bias: usize,
    output_bias: usize,
    sinks: usize,
    scratch: usize,
    binding: usize,
}

#[derive(Debug, Clone, Copy)]
struct AttentionLaunch {
    input_region: usize,
    output_region: usize,
    binding_offset: u64,
    packed_token_start: u64,
    normalized_offset: u64,
    query_raw_offset: u64,
    key_raw_offset: u64,
    value_raw_offset: u64,
    query_offset: u64,
    context_offset: u64,
    projected_offset: u64,
    tokens: u64,
    tokens_i32: i32,
    position_start: u64,
    sequence_tokens: u64,
}

fn bind_launch_replay_key(
    replay_key: CudaCommandReplayKeyBuilder,
    launch: AttentionLaunch,
) -> CudaCommandReplayKeyBuilder {
    // The sequence frontier is runtime data, not executable topology. The
    // eager binding command writes `position_start`, `sequence_tokens`, and
    // the current page table into the stable per-program binding region
    // before the captured compute segment runs. Binding those values into the
    // compute key would compile a fresh executable at every decode position.
    replay_key
        .u64(launch.input_region as u64)
        .u64(launch.output_region as u64)
        .u64(launch.binding_offset)
        .u64(launch.packed_token_start)
        .u64(launch.normalized_offset)
        .u64(launch.query_raw_offset)
        .u64(launch.key_raw_offset)
        .u64(launch.value_raw_offset)
        .u64(launch.query_offset)
        .u64(launch.context_offset)
        .u64(launch.projected_offset)
        .u64(launch.tokens)
        .i32(launch.tokens_i32)
}

#[derive(Debug, Clone, Copy)]
struct PackedLaunch {
    input_region: usize,
    output_region: usize,
    tokens: u64,
    tokens_i32: i32,
}

#[derive(Debug, Clone, Copy)]
struct AttentionBinding {
    first_page_region: usize,
    page_count: usize,
    host_binding: usize,
    binding_offset: u64,
}

fn gpt_oss_reusable_topology(
    request: &ReusableExecutionTopologyRequest<'_>,
) -> Result<ReusableExecutionTopology, String> {
    if request.operation_id().as_str() != GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID {
        return Err("CUDA GPT-OSS attention topology received another operation".to_owned());
    }
    let shape = GptOssAttentionShape::from_attributes(request.attributes())?;
    let ranges = request.work_shape().participant_token_ranges();
    if ranges.is_empty() {
        return Err("CUDA GPT-OSS attention topology has no participants".to_owned());
    }
    let fingerprint = gpt_oss_reusable_topology_fingerprint(
        shape,
        ranges.len(),
        ranges.iter().map(|range| {
            (
                range.source_token_range(),
                range.full_input_tokens(),
                range.immediate_tokens(),
            )
        }),
    )?;
    Ok(ReusableExecutionTopology::Dynamic(fingerprint))
}

fn gpt_oss_reusable_topology_fingerprint(
    shape: GptOssAttentionShape,
    participant_count: usize,
    ranges: impl IntoIterator<Item = (Range<u64>, u64, u64)>,
) -> Result<DeviceReusableExecutionTopologyFingerprint, String> {
    if participant_count == 0 {
        return Err("CUDA GPT-OSS attention topology has no participants".to_owned());
    }
    let mut digest = Sha256::new();
    digest.update(b"ferrum.cuda.gpt-oss-attention.reusable-topology.v2\0");
    hash_shape(&mut digest, shape);
    let participant_count_u64 = u64::try_from(participant_count)
        .map_err(|_| "CUDA GPT-OSS attention participant count exceeds u64".to_owned())?;
    digest.update(participant_count_u64.to_le_bytes());
    let mut observed_participants = 0_usize;
    for (source, full_input_tokens, immediate_tokens) in ranges {
        if source.start >= source.end
            || source.end > full_input_tokens
            || full_input_tokens > shape.maximum_context_tokens
            || immediate_tokens == 0
            || source.end - source.start != immediate_tokens
        {
            return Err("CUDA GPT-OSS attention topology token range is invalid".to_owned());
        }
        // The exact frontier and page table are written into the stable
        // program-binding region before each replay. Only the participant's
        // active token width changes compute launch topology.
        digest.update(immediate_tokens.to_le_bytes());
        observed_participants = observed_participants
            .checked_add(1)
            .ok_or_else(|| "CUDA GPT-OSS attention participant count overflowed".to_owned())?;
    }
    if observed_participants != participant_count {
        return Err("CUDA GPT-OSS attention topology participant count differs".to_owned());
    }
    Ok(DeviceReusableExecutionTopologyFingerprint::from_sha256(
        digest.finalize().into(),
    ))
}

fn hash_shape(digest: &mut Sha256, shape: GptOssAttentionShape) {
    for value in [
        shape.hidden_size,
        shape.query_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.query_features,
        shape.kv_features,
        shape.rope_dim,
        shape.maximum_context_tokens,
        shape.yarn_original_context_tokens,
        shape.sliding_window_tokens,
        shape.layer_index,
    ] {
        digest.update(value.to_le_bytes());
    }
    for value in [
        shape.rope_theta,
        shape.yarn_factor,
        shape.yarn_beta_fast,
        shape.yarn_beta_slow,
        shape.epsilon,
    ] {
        digest.update(value.to_bits().to_le_bytes());
    }
}

fn bind_shape_replay_key(
    replay_key: CudaCommandReplayKeyBuilder,
    shape: GptOssAttentionShape,
) -> CudaCommandReplayKeyBuilder {
    replay_key
        .u64(shape.hidden_size)
        .u64(shape.query_heads)
        .u64(shape.kv_heads)
        .u64(shape.head_dim)
        .u64(shape.query_features)
        .u64(shape.kv_features)
        .u64(shape.rope_dim)
        .u64(shape.maximum_context_tokens)
        .f32(shape.rope_theta)
        .f32(shape.yarn_factor)
        .u64(shape.yarn_original_context_tokens)
        .f32(shape.yarn_beta_fast)
        .f32(shape.yarn_beta_slow)
        .u64(shape.sliding_window_tokens)
        .f32(shape.epsilon)
        .u64(shape.layer_index)
}

fn encode_attention(
    functions: &GptOssAttentionFunctions,
    provider_fingerprint: &str,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID
    {
        return Err("CUDA GPT-OSS attention received another or empty operation".to_owned());
    }
    let first = &invocation.participants()[0];
    let shape = GptOssAttentionShape::from_attributes(first.attributes())?;
    validate_signature(first, shape)?;
    for participant in &invocation.participants()[1..] {
        if GptOssAttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("CUDA GPT-OSS attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape)?;
    }

    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::new(shape, total_tokens)?;
    let binding_layout = BindingLayout::new(shape, invocation.participants().len())?;
    let cuda = shape.cuda_shape()?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA GPT-OSS attention participant ranges are incomplete".to_owned());
    }
    let input_packed = super::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = super::token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;

    let mut compute_regions = Vec::new();
    let shared = SharedRegions {
        input_norm: push_shared_weight(&mut compute_regions, &invocation, 1)?,
        query_weight: push_shared_weight(&mut compute_regions, &invocation, 2)?,
        key_weight: push_shared_weight(&mut compute_regions, &invocation, 3)?,
        value_weight: push_shared_weight(&mut compute_regions, &invocation, 4)?,
        output_weight: push_shared_weight(&mut compute_regions, &invocation, 5)?,
        query_bias: push_shared_weight(&mut compute_regions, &invocation, 6)?,
        key_bias: push_shared_weight(&mut compute_regions, &invocation, 7)?,
        value_bias: push_shared_weight(&mut compute_regions, &invocation, 8)?,
        output_bias: push_shared_weight(&mut compute_regions, &invocation, 9)?,
        sinks: push_shared_weight(&mut compute_regions, &invocation, 10)?,
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

    let packed = if input_packed && output_packed && invocation.participants().len() > 1 {
        let input_region = compute_regions.len();
        compute_regions.push(super::shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            total_tokens,
        )?);
        let output_region = compute_regions.len();
        compute_regions.push(super::shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            total_tokens,
        )?);
        Some(PackedLaunch {
            input_region,
            output_region,
            tokens: total_tokens,
            tokens_i32: checked_i32(total_tokens, "packed GPT-OSS attention token count")?,
        })
    } else {
        None
    };

    let mut binding_regions = vec![compute_regions[shared.binding].clone()];
    let mut compute_fence_dependencies = Vec::new();
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
        let packed_range = token_range.immediate_token_range();
        if tokens == 0
            || source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
            || source.end != source.start.saturating_add(tokens)
        {
            return Err("GPT-OSS attention token range exceeds admitted context".to_owned());
        }
        let input_region = if let Some(packed) = packed {
            packed.input_region
        } else {
            let index = compute_regions.len();
            compute_regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F16,
                if input_packed {
                    packed_range.start
                } else {
                    source.start
                },
                tokens,
            )?);
            index
        };
        let output_region = if let Some(packed) = packed {
            packed.output_region
        } else {
            let index = compute_regions.len();
            compute_regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F16,
                if output_packed {
                    packed_range.start
                } else {
                    source.start
                },
                tokens,
            )?);
            index
        };

        let first_page_region = binding_regions.len();
        let pages = paged_state_regions(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 11)?,
            shape.physical_state_bytes(source.end)?,
        )?;
        if u64::try_from(pages.len())
            .map_err(|_| "GPT-OSS attention page count exceeds u64".to_owned())?
            > shape.maximum_pages()?
        {
            return Err("GPT-OSS attention page table exceeds admitted maximum".to_owned());
        }
        let tokens_i32 = checked_i32(tokens, "GPT-OSS attention participant token count")?;
        let payload = binding_payload(
            checked_i32(source.start, "GPT-OSS attention source position")?,
            tokens_i32,
            checked_i32(source.end, "GPT-OSS attention sequence token count")?,
            &pages,
        )?;
        let host_binding = host_storage.len();
        host_storage.push(payload);
        let page_count = pages.len();
        compute_fence_dependencies.extend(pages.iter().cloned());
        binding_regions.extend(pages);
        let binding_offset = binding_layout.binding_offset(participant_index)?;
        bindings.push(AttentionBinding {
            first_page_region,
            page_count,
            host_binding,
            binding_offset,
        });

        let token_start = if packed.is_some() {
            packed_range.start
        } else {
            0
        };
        launches.push(AttentionLaunch {
            input_region,
            output_region,
            binding_offset,
            packed_token_start: packed_range.start,
            normalized_offset: layout.token_offset(
                layout.normalized,
                token_start,
                shape.hidden_size,
            )?,
            query_raw_offset: layout.token_offset(
                layout.query_raw,
                token_start,
                shape.query_features,
            )?,
            key_raw_offset: layout.token_offset(layout.key_raw, token_start, shape.kv_features)?,
            value_raw_offset: layout.token_offset(
                layout.value_raw,
                token_start,
                shape.kv_features,
            )?,
            query_offset: layout.token_offset(layout.query, token_start, shape.query_features)?,
            context_offset: layout.token_offset(
                layout.context,
                token_start,
                shape.query_features,
            )?,
            projected_offset: layout.token_offset(
                layout.projected,
                token_start,
                shape.hidden_size,
            )?,
            tokens,
            tokens_i32,
            position_start: source.start,
            sequence_tokens: source.end,
        });
    }
    if packed.is_some() {
        validate_packed_token_ranges(
            launches
                .iter()
                .map(|launch| (launch.packed_token_start, launch.tokens)),
            total_tokens,
        )?;
    }

    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA GPT-OSS attention participant count exceeds u32".to_owned())?;
    let (binding_command, has_compiled_program_slot) = if let Some(program_binding) =
        invocation.program_binding().cloned()
    {
        let mut regions = binding_regions.into_iter();
        let destination = regions
            .next()
            .ok_or_else(|| "GPT-OSS attention binding destination is missing".to_owned())?;
        let fence_dependencies = regions.collect::<Vec<_>>();
        let mut writes = Vec::with_capacity(bindings.len());
        for (index, (binding, payload)) in bindings.into_iter().zip(host_storage).enumerate() {
            if binding.host_binding != index {
                return Err("GPT-OSS attention binding payload order is not canonical".to_owned());
            }
            writes.push(
                super::CudaProgramBindingWrite::new(binding.binding_offset, payload)
                    .map_err(|error| error.to_string())?,
            );
        }
        (
            CudaDeviceCommand::program_binding_patch(
                BINDING_OPERATION,
                program_binding,
                destination,
                writes,
                fence_dependencies,
            ),
            true,
        )
    } else {
        (
            CudaDeviceCommand::operation_with_host_storage_and_blas(
                BINDING_OPERATION,
                binding_regions,
                host_storage,
                move |stream, _blas, regions, host_storage| {
                    enqueue_bindings(stream, binding_layout, &bindings, regions, host_storage)
                },
            ),
            false,
        )
    };
    let binding_command = binding_command
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

    let packed_enabled = packed.is_some();
    let dispatch_count = if packed_enabled {
        6_u64.saturating_add((launches.len() as u64).saturating_mul(2))
    } else {
        (launches.len() as u64).saturating_mul(8)
    };
    let mut replay_key = bind_shape_replay_key(
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, COMPUTE_OPERATION),
        shape,
    )
    .u64(total_tokens)
    .boolean(packed_enabled)
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
        replay_key = bind_launch_replay_key(replay_key, *launch);
    }
    let replay_key = replay_key.finish();
    let functions = functions.clone();
    let enqueue_compute = move |stream: &CudaStream,
                                blas: &CudaBlas,
                                regions: &[CudaBufferRegion]| {
        if let Some(packed) = packed {
            enqueue_packed_attention(
                stream, blas, &functions, shape, cuda, layout, shared, packed, &launches, regions,
            )?;
        } else {
            for launch in &launches {
                enqueue_attention(
                    stream, blas, &functions, shape, cuda, layout, shared, *launch, regions,
                )?;
            }
        }
        Ok(())
    };
    let compute_command = CudaDeviceCommand::replayable_operation_with_blas_and_fence_dependencies(
        COMPUTE_OPERATION,
        compute_regions,
        compute_fence_dependencies,
        replay_key,
        enqueue_compute,
    )
    .and_then(|command| {
        command.with_work_attribution(
            if packed_enabled {
                DeviceBatchingForm::Packed
            } else if participant_count == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participant_count,
            total_tokens,
            dispatch_count,
            0,
        )
    })
    .map_err(|error| error.to_string())?;

    Ok(attach_invocation_binding(
        EncodedDeviceOperation::compute(compute_command),
        binding_command,
        has_compiled_program_slot,
    ))
}

fn validate_packed_token_ranges(
    ranges: impl IntoIterator<Item = (u64, u64)>,
    total_tokens: u64,
) -> Result<(), String> {
    let mut next_token = 0_u64;
    for (token_start, tokens) in ranges {
        if tokens == 0 || token_start != next_token {
            return Err("packed GPT-OSS attention ranges are not canonical".to_owned());
        }
        next_token = next_token
            .checked_add(tokens)
            .ok_or_else(|| "packed GPT-OSS attention token range overflows".to_owned())?;
    }
    if next_token != total_tokens {
        return Err("packed GPT-OSS attention ranges do not cover work shape".to_owned());
    }
    Ok(())
}

fn encode_reusable_attention_bindings(
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != GPT_OSS_CAUSAL_PAGED_ATTENTION_OPERATION_ID
    {
        return Err("CUDA GPT-OSS attention received another or empty operation".to_owned());
    }
    let program_binding = invocation.program_binding().cloned().ok_or_else(|| {
        "CUDA GPT-OSS reusable binding requires a compiled program binding".to_owned()
    })?;
    let shape = GptOssAttentionShape::from_attributes(invocation.participants()[0].attributes())?;
    let total_tokens = invocation.work_shape().immediate_tokens();
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA GPT-OSS attention participant count exceeds u32".to_owned())?;
    let binding_layout = BindingLayout::new(shape, invocation.participants().len())?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA GPT-OSS attention participant ranges are incomplete".to_owned());
    }
    let destination = super::shared_binding_region(&invocation, binding_layout.required_bytes)?;
    let mut writes = Vec::with_capacity(invocation.participants().len());
    let mut fence_dependencies = Vec::new();
    for (participant_index, (participant, token_range)) in invocation
        .participants()
        .iter()
        .zip(token_ranges)
        .enumerate()
    {
        if GptOssAttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("CUDA GPT-OSS attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape)?;
        let tokens = token_range.immediate_tokens();
        let source = token_range.source_token_range();
        if tokens == 0
            || source.end > token_range.full_input_tokens()
            || token_range.full_input_tokens() > shape.maximum_context_tokens
            || source.end != source.start.saturating_add(tokens)
        {
            return Err("GPT-OSS attention token range exceeds admitted context".to_owned());
        }
        let pages = paged_state_regions(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 11)?,
            shape.physical_state_bytes(source.end)?,
        )?;
        if u64::try_from(pages.len())
            .map_err(|_| "GPT-OSS attention page count exceeds u64".to_owned())?
            > shape.maximum_pages()?
        {
            return Err("GPT-OSS attention page table exceeds admitted maximum".to_owned());
        }
        let payload = binding_payload(
            checked_i32(source.start, "GPT-OSS attention source position")?,
            checked_i32(tokens, "GPT-OSS attention participant token count")?,
            checked_i32(source.end, "GPT-OSS attention sequence token count")?,
            &pages,
        )?;
        writes.push(
            super::CudaProgramBindingWrite::new(
                binding_layout.binding_offset(participant_index)?,
                payload,
            )
            .map_err(|error| error.to_string())?,
        );
        fence_dependencies.extend(pages);
    }
    let binding_command = CudaDeviceCommand::program_binding_patch(
        BINDING_OPERATION,
        program_binding,
        destination,
        writes,
        fence_dependencies,
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
    Ok(EncodedReusableExecutionBindings::empty().with_program_binding(binding_command))
}

fn enqueue_bindings(
    stream: &CudaStream,
    layout: BindingLayout,
    bindings: &[AttentionBinding],
    regions: &[CudaBufferRegion],
    host_storage: &[Box<[u8]>],
) -> Result<(), CudaDeviceRuntimeError> {
    let binding_workspace = &regions[0];
    if binding_workspace.length_bytes() < layout.required_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "GPT-OSS attention binding workspace is smaller than estimate",
        ));
    }
    for binding in bindings {
        let page_region_end = binding
            .first_page_region
            .checked_add(binding.page_count)
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("GPT-OSS attention page region range overflows")
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
                "GPT-OSS attention page regions changed after encoding",
            ));
        }
        let payload = host_storage.get(binding.host_binding).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("GPT-OSS attention binding payload is missing")
        })?;
        if payload.len() as u64 > layout.slot_bytes {
            return Err(CudaDeviceRuntimeError::contract(
                "GPT-OSS attention binding payload exceeds admitted slot",
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
            CudaDeviceRuntimeError::driver("GPT-OSS attention binding upload", error)
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn enqueue_packed_attention(
    stream: &CudaStream,
    blas: &CudaBlas,
    functions: &GptOssAttentionFunctions,
    logical: GptOssAttentionShape,
    cuda: CudaGptOssAttentionShape,
    layout: ScratchLayout,
    shared: SharedRegions,
    packed: PackedLaunch,
    launches: &[AttentionLaunch],
    regions: &[CudaBufferRegion],
) -> Result<(), CudaDeviceRuntimeError> {
    let scratch = &regions[shared.scratch];
    if scratch.length_bytes() < layout.required_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "packed GPT-OSS attention scratch is smaller than estimate",
        ));
    }
    let scratch_base = scratch.device_ptr();
    let binding = &regions[shared.binding];
    let input = regions[packed.input_region].device_ptr();
    let output = regions[packed.output_region].device_ptr();
    let normalized = scratch_pointer(scratch_base, layout.normalized)?;
    let query_raw = scratch_pointer(scratch_base, layout.query_raw)?;
    let key_raw = scratch_pointer(scratch_base, layout.key_raw)?;
    let value_raw = scratch_pointer(scratch_base, layout.value_raw)?;
    let context = scratch_pointer(scratch_base, layout.context)?;
    let projected = scratch_pointer(scratch_base, layout.projected)?;

    launch_rms_norm(
        stream,
        &functions.rms_norm,
        input,
        regions[shared.input_norm].device_ptr(),
        normalized,
        packed.tokens,
        cuda.hidden_size,
        cuda.epsilon,
    )?;
    for (weight, destination, output_features, operation) in [
        (
            shared.query_weight,
            query_raw,
            cuda.query_features,
            "packed GPT-OSS Q GEMM",
        ),
        (
            shared.key_weight,
            key_raw,
            cuda.kv_features,
            "packed GPT-OSS K GEMM",
        ),
        (
            shared.value_weight,
            value_raw,
            cuda.kv_features,
            "packed GPT-OSS V GEMM",
        ),
    ] {
        launch_gemm_f16(
            blas,
            normalized,
            regions[weight].device_ptr(),
            destination,
            packed.tokens_i32,
            output_features,
            cuda.hidden_size,
            operation,
        )?;
    }
    for launch in launches {
        launch_prepare(
            stream,
            &functions.prepare,
            scratch_base,
            binding.device_ptr(),
            regions,
            shared,
            *launch,
            cuda,
        )?;
        launch_attention(
            stream,
            &functions.attention,
            scratch_base,
            binding.device_ptr(),
            regions[shared.sinks].device_ptr(),
            *launch,
            cuda,
        )?;
    }
    launch_gemm_f16(
        blas,
        context,
        regions[shared.output_weight].device_ptr(),
        projected,
        packed.tokens_i32,
        cuda.hidden_size,
        cuda.query_features,
        "packed GPT-OSS output GEMM",
    )?;
    launch_residual_bias(
        stream,
        &functions.residual_bias,
        input,
        projected,
        regions[shared.output_bias].device_ptr(),
        output,
        logical.hidden_size,
        packed.tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn enqueue_attention(
    stream: &CudaStream,
    blas: &CudaBlas,
    functions: &GptOssAttentionFunctions,
    logical: GptOssAttentionShape,
    cuda: CudaGptOssAttentionShape,
    layout: ScratchLayout,
    shared: SharedRegions,
    launch: AttentionLaunch,
    regions: &[CudaBufferRegion],
) -> Result<(), CudaDeviceRuntimeError> {
    let scratch = &regions[shared.scratch];
    if scratch.length_bytes() < layout.required_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "GPT-OSS attention scratch is smaller than estimate",
        ));
    }
    let scratch_base = scratch.device_ptr();
    let input = regions[launch.input_region].device_ptr();
    let output = regions[launch.output_region].device_ptr();
    let normalized = scratch_pointer(scratch_base, launch.normalized_offset)?;
    let query_raw = scratch_pointer(scratch_base, launch.query_raw_offset)?;
    let key_raw = scratch_pointer(scratch_base, launch.key_raw_offset)?;
    let value_raw = scratch_pointer(scratch_base, launch.value_raw_offset)?;
    let context = scratch_pointer(scratch_base, launch.context_offset)?;
    let projected = scratch_pointer(scratch_base, launch.projected_offset)?;

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
    for (weight, destination, output_features, operation) in [
        (
            shared.query_weight,
            query_raw,
            cuda.query_features,
            "GPT-OSS Q GEMM",
        ),
        (
            shared.key_weight,
            key_raw,
            cuda.kv_features,
            "GPT-OSS K GEMM",
        ),
        (
            shared.value_weight,
            value_raw,
            cuda.kv_features,
            "GPT-OSS V GEMM",
        ),
    ] {
        launch_gemm_f16(
            blas,
            normalized,
            regions[weight].device_ptr(),
            destination,
            launch.tokens_i32,
            output_features,
            cuda.hidden_size,
            operation,
        )?;
    }
    launch_prepare(
        stream,
        &functions.prepare,
        scratch_base,
        regions[shared.binding].device_ptr(),
        regions,
        shared,
        launch,
        cuda,
    )?;
    launch_attention(
        stream,
        &functions.attention,
        scratch_base,
        regions[shared.binding].device_ptr(),
        regions[shared.sinks].device_ptr(),
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
        "GPT-OSS output GEMM",
    )?;
    launch_residual_bias(
        stream,
        &functions.residual_bias,
        input,
        projected,
        regions[shared.output_bias].device_ptr(),
        output,
        logical.hidden_size,
        launch.tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_prepare(
    stream: &CudaStream,
    function: &CudaFunction,
    scratch_base: u64,
    binding_base: u64,
    regions: &[CudaBufferRegion],
    shared: SharedRegions,
    launch: AttentionLaunch,
    shape: CudaGptOssAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let control = scratch_pointer(binding_base, launch.binding_offset)?;
    let page_table = control.checked_add(BINDING_CONTROL_BYTES).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("GPT-OSS attention page-table pointer overflows")
    })?;
    let query_raw = scratch_pointer(scratch_base, launch.query_raw_offset)?;
    let key_raw = scratch_pointer(scratch_base, launch.key_raw_offset)?;
    let value_raw = scratch_pointer(scratch_base, launch.value_raw_offset)?;
    let query = scratch_pointer(scratch_base, launch.query_offset)?;
    let page_elements = checked_i32_runtime(
        VNEXT_KV_PAGE_BYTES / ElementType::F16.size_bytes(),
        "GPT-OSS attention page elements",
    )?;
    let mut builder = stream.launch_builder(function);
    let pointers = [
        query_raw,
        key_raw,
        value_raw,
        regions[shared.query_bias].device_ptr(),
        regions[shared.key_bias].device_ptr(),
        regions[shared.value_bias].device_ptr(),
        query,
        control,
        page_table,
    ];
    for pointer in &pointers {
        builder.arg(pointer);
    }
    let dimensions = [
        page_elements,
        shape.query_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.rope_dim,
        shape.query_features,
        shape.kv_features,
    ];
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    builder.arg(&shape.rope_theta);
    builder.arg(&shape.yarn_factor);
    let original_context = shape.yarn_original_context_tokens as f32;
    builder.arg(&original_context);
    builder.arg(&shape.yarn_beta_fast);
    builder.arg(&shape.yarn_beta_slow);
    let combined_heads = shape
        .query_heads
        .checked_add(shape.kv_heads.checked_mul(2).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("GPT-OSS attention prepare head count overflows")
        })?)
        .ok_or_else(|| {
            CudaDeviceRuntimeError::contract("GPT-OSS attention prepare head count overflows")
        })?;
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_u32_runtime(launch.tokens, "GPT-OSS attention prepare token grid")?,
                u32::try_from(combined_heads).map_err(|_| {
                    CudaDeviceRuntimeError::contract(
                        "GPT-OSS attention prepare head grid exceeds u32",
                    )
                })?,
                1,
            ),
            block_dim: (WARP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS attention prepare launch", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_attention(
    stream: &CudaStream,
    function: &CudaFunction,
    scratch_base: u64,
    binding_base: u64,
    sinks: u64,
    launch: AttentionLaunch,
    shape: CudaGptOssAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let control = scratch_pointer(binding_base, launch.binding_offset)?;
    let page_table = control.checked_add(BINDING_CONTROL_BYTES).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("GPT-OSS attention page-table pointer overflows")
    })?;
    let query = scratch_pointer(scratch_base, launch.query_offset)?;
    let output = scratch_pointer(scratch_base, launch.context_offset)?;
    let page_elements = checked_i32_runtime(
        VNEXT_KV_PAGE_BYTES / ElementType::F16.size_bytes(),
        "GPT-OSS attention page elements",
    )?;
    let mut builder = stream.launch_builder(function);
    let pointers = [query, sinks, control, page_table, output];
    for pointer in &pointers {
        builder.arg(pointer);
    }
    let dimensions = [
        page_elements,
        shape.query_heads,
        shape.kv_heads,
        shape.head_dim,
        shape.sliding_window_tokens,
    ];
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_u32_runtime(launch.tokens, "GPT-OSS attention token grid")?,
                u32::try_from(shape.query_heads).map_err(|_| {
                    CudaDeviceRuntimeError::contract(
                        "GPT-OSS attention query-head grid exceeds u32",
                    )
                })?,
                1,
            ),
            block_dim: (WARP_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS attention launch", error))
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
    let rows = checked_u32_runtime(tokens, "GPT-OSS attention RMSNorm rows")?;
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
    .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS attention RMSNorm launch", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_residual_bias(
    stream: &CudaStream,
    function: &CudaFunction,
    residual: u64,
    branch: u64,
    bias: u64,
    output: u64,
    hidden_size: u64,
    tokens: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let elements = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("GPT-OSS residual size overflows"))?;
    let hidden_size = checked_i32_runtime(hidden_size, "GPT-OSS residual hidden size")?;
    let elements_i32 = checked_i32_runtime(elements, "GPT-OSS residual elements")?;
    let grid = checked_u32_runtime(
        elements.div_ceil(u64::from(THREADS_PER_BLOCK)),
        "GPT-OSS residual grid",
    )?;
    let mut builder = stream.launch_builder(function);
    let pointers = [residual, branch, bias, output];
    for pointer in &pointers {
        builder.arg(pointer);
    }
    builder.arg(&hidden_size);
    builder.arg(&elements_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| {
        CudaDeviceRuntimeError::driver("GPT-OSS attention residual+bias launch", error)
    })
}

fn paged_state_regions(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    state: &ResolvedValueBinding,
    expected_physical_bytes: u64,
) -> Result<Vec<CudaBufferRegion>, String> {
    let [component] = state.storage().components() else {
        return Err("GPT-OSS attention state requires one storage component".to_owned());
    };
    let view = participant
        .views()
        .iter()
        .find(|view| view.resource_id() == component.resource_id())
        .ok_or_else(|| "GPT-OSS attention state has no resource view".to_owned())?;
    if component.offset_bytes() != 0
        || component.element_type() != ElementType::F16
        || view.descriptor().element_type != ElementType::F16
        || view.storage_kind() != OperationBufferStorageKind::DynamicPaged
        || view.descriptor().size_bytes != expected_physical_bytes
        || expected_physical_bytes == 0
        || expected_physical_bytes % VNEXT_KV_PAGE_BYTES != 0
    {
        return Err("GPT-OSS attention state is not its fixed-block paged view".to_owned());
    }
    let translated = view
        .translate(0, expected_physical_bytes)
        .map_err(|error| error.to_string())?;
    let page_capacity = usize::try_from(expected_physical_bytes / VNEXT_KV_PAGE_BYTES)
        .map_err(|_| "GPT-OSS attention page capacity exceeds usize".to_owned())?;
    let mut pages = Vec::with_capacity(page_capacity);
    let mut next_logical = 0_u64;
    for physical in translated.iter() {
        if physical.logical_offset_bytes() != next_logical
            || physical.length_bytes() == 0
            || physical.length_bytes() % VNEXT_KV_PAGE_BYTES != 0
        {
            return Err("GPT-OSS attention paged translation lost geometry".to_owned());
        }
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let mut offset = 0_u64;
        while offset < physical.length_bytes() {
            let start = range
                .start
                .checked_add(offset)
                .ok_or_else(|| "GPT-OSS attention page offset overflows".to_owned())?;
            let end = start
                .checked_add(VNEXT_KV_PAGE_BYTES)
                .ok_or_else(|| "GPT-OSS attention page range overflows".to_owned())?;
            let page = buffer
                .retained_region(start..end, retention.clone())
                .map_err(|error| error.to_string())?;
            if page.length_bytes() != VNEXT_KV_PAGE_BYTES || page.element_type() != ElementType::F16
            {
                return Err("GPT-OSS attention physical page differs from contract".to_owned());
            }
            pages.push(page);
            offset += VNEXT_KV_PAGE_BYTES;
        }
        next_logical = next_logical
            .checked_add(physical.length_bytes())
            .ok_or_else(|| "GPT-OSS attention logical page coverage overflows".to_owned())?;
    }
    if next_logical != expected_physical_bytes || pages.is_empty() {
        return Err("GPT-OSS attention pages do not cover state".to_owned());
    }
    Ok(pages)
}

fn binding_payload(
    position_start: i32,
    active_tokens: i32,
    sequence_tokens: i32,
    pages: &[CudaBufferRegion],
) -> Result<Box<[u8]>, String> {
    binding_payload_from_addresses(
        position_start,
        active_tokens,
        sequence_tokens,
        &pages
            .iter()
            .map(CudaBufferRegion::device_ptr)
            .collect::<Vec<_>>(),
    )
}

fn binding_payload_from_addresses(
    position_start: i32,
    active_tokens: i32,
    sequence_tokens: i32,
    page_addresses: &[u64],
) -> Result<Box<[u8]>, String> {
    if position_start < 0
        || active_tokens <= 0
        || sequence_tokens <= position_start
        || sequence_tokens != position_start.saturating_add(active_tokens)
        || page_addresses.is_empty()
    {
        return Err("GPT-OSS attention binding control is invalid".to_owned());
    }
    let page_count = i32::try_from(page_addresses.len())
        .map_err(|_| "GPT-OSS attention page count exceeds i32".to_owned())?;
    let mut payload = Vec::with_capacity(
        BINDING_CONTROL_BYTES as usize + page_addresses.len() * std::mem::size_of::<u64>(),
    );
    for value in [page_count, position_start, active_tokens, sequence_tokens] {
        payload.extend_from_slice(&value.to_ne_bytes());
    }
    for address in page_addresses {
        payload.extend_from_slice(&address.to_ne_bytes());
    }
    Ok(payload.into_boxed_slice())
}

fn validate_signature(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    shape: GptOssAttentionShape,
) -> Result<(), String> {
    let value = |ordinal| binding(participant.bindings(), ResolvedValueRole::Input, ordinal);
    let hidden = value(0)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let [tokens, hidden_width] = hidden.tensor().dimensions() else {
        return Err("GPT-OSS attention hidden input is not two-dimensional".to_owned());
    };
    let expected = [
        (value(1)?, vec![shape.hidden_size]),
        (value(2)?, vec![shape.query_features, shape.hidden_size]),
        (value(3)?, vec![shape.kv_features, shape.hidden_size]),
        (value(4)?, vec![shape.kv_features, shape.hidden_size]),
        (value(5)?, vec![shape.hidden_size, shape.query_features]),
        (value(6)?, vec![shape.query_features]),
        (value(7)?, vec![shape.kv_features]),
        (value(8)?, vec![shape.kv_features]),
        (value(9)?, vec![shape.hidden_size]),
        (value(10)?, vec![shape.query_heads]),
        (value(11)?, vec![2, shape.kv_heads, shape.head_dim]),
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
        return Err("GPT-OSS attention signature differs from resolved shape".to_owned());
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
                .ok_or_else(|| "GPT-OSS attention scratch span overflows".to_owned())?,
        )
        .ok_or_else(|| "GPT-OSS attention scratch offset overflows".to_owned())?;
    Ok(start)
}

fn aligned_bytes(elements: u64, element_bytes: u64) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| "GPT-OSS attention byte count overflows".to_owned())?;
    align_up(bytes, SCRATCH_ALIGNMENT)
}

fn align_up(bytes: u64, alignment: u64) -> Result<u64, String> {
    bytes
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .filter(|value| *value > 0)
        .ok_or_else(|| "GPT-OSS attention alignment overflows".to_owned())
}

fn scratch_pointer(base: u64, offset: u64) -> Result<u64, CudaDeviceRuntimeError> {
    base.checked_add(offset).ok_or_else(|| {
        CudaDeviceRuntimeError::contract("GPT-OSS attention scratch pointer overflows")
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
            "CUDA GPT-OSS attention lacks unsigned attribute {name:?}"
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
            "CUDA GPT-OSS attention lacks boolean attribute {name:?}"
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
                "CUDA GPT-OSS attention lacks rational attribute {name:?}"
            ))
        }
    };
    let value = (rational.numerator() as f64 / rational.denominator() as f64) as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "CUDA GPT-OSS attention rational attribute {name:?} is not a positive f32"
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

    fn rational(numerator: i64, denominator: u64) -> SemanticValue {
        SemanticValue::Rational(CanonicalRational::new(numerator, denominator).unwrap())
    }

    fn attributes(sliding_window_tokens: u64) -> BTreeMap<AttributeId, SemanticValue> {
        BTreeMap::from([
            (
                AttributeId::new("hidden_size").unwrap(),
                SemanticValue::Unsigned(2880),
            ),
            (
                AttributeId::new("query_heads").unwrap(),
                SemanticValue::Unsigned(64),
            ),
            (
                AttributeId::new("kv_heads").unwrap(),
                SemanticValue::Unsigned(8),
            ),
            (
                AttributeId::new("head_dim").unwrap(),
                SemanticValue::Unsigned(64),
            ),
            (
                AttributeId::new("query_features").unwrap(),
                SemanticValue::Unsigned(4096),
            ),
            (
                AttributeId::new("kv_features").unwrap(),
                SemanticValue::Unsigned(512),
            ),
            (
                AttributeId::new("rope_dim").unwrap(),
                SemanticValue::Unsigned(64),
            ),
            (
                AttributeId::new("maximum_context_tokens").unwrap(),
                SemanticValue::Unsigned(131_072),
            ),
            (
                AttributeId::new("rope_theta").unwrap(),
                rational(150_000, 1),
            ),
            (AttributeId::new("yarn_factor").unwrap(), rational(32, 1)),
            (
                AttributeId::new("yarn_original_context_tokens").unwrap(),
                SemanticValue::Unsigned(4096),
            ),
            (AttributeId::new("yarn_beta_fast").unwrap(), rational(32, 1)),
            (AttributeId::new("yarn_beta_slow").unwrap(), rational(1, 1)),
            (
                AttributeId::new("yarn_truncate").unwrap(),
                SemanticValue::Bool(false),
            ),
            (
                AttributeId::new("sliding_window_tokens").unwrap(),
                SemanticValue::Unsigned(sliding_window_tokens),
            ),
            (
                AttributeId::new("causal").unwrap(),
                SemanticValue::Bool(true),
            ),
            (AttributeId::new("epsilon").unwrap(), rational(1, 100_000)),
            (
                AttributeId::new("layer_index").unwrap(),
                SemanticValue::Unsigned(if sliding_window_tokens == 0 { 1 } else { 0 }),
            ),
        ])
    }

    #[test]
    fn exact_gpt_oss_attributes_admit_full_and_sliding_layers() {
        let sliding = GptOssAttentionShape::from_attributes(&attributes(128)).unwrap();
        let full = GptOssAttentionShape::from_attributes(&attributes(0)).unwrap();
        assert_eq!(sliding.head_dim, 64);
        assert_eq!(sliding.sliding_window_tokens, 128);
        assert_eq!(full.sliding_window_tokens, 0);
        assert_ne!(sliding.layer_index, full.layer_index);
    }

    #[test]
    fn gpt_oss_shape_fails_closed_on_yarn_head_and_window_drift() {
        for (name, value) in [
            ("yarn_factor", rational(16, 1)),
            ("yarn_beta_fast", rational(16, 1)),
            ("yarn_beta_slow", rational(2, 1)),
            ("rope_theta", rational(10_000, 1)),
        ] {
            let mut drift = attributes(128);
            drift.insert(AttributeId::new(name).unwrap(), value);
            assert!(GptOssAttentionShape::from_attributes(&drift).is_err());
        }
        let mut head = attributes(128);
        head.insert(
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(128),
        );
        assert!(GptOssAttentionShape::from_attributes(&head).is_err());
        assert!(GptOssAttentionShape::from_attributes(&attributes(64)).is_err());
    }

    #[test]
    fn storage_bindings_keep_only_kv_state_paged() {
        let bindings = storage_bindings().unwrap();
        assert_eq!(bindings.len(), 13);
        for binding in bindings {
            let profile = binding.storage().accepted_profiles()[0];
            if binding.role() == ResolvedValueRole::Input && binding.ordinal() == 11 {
                assert_eq!(
                    profile.allocator(),
                    DynamicStorageAllocator::FixedBlockArena {
                        block_bytes: VNEXT_KV_PAGE_BYTES,
                    }
                );
                assert_eq!(
                    profile.view(),
                    DynamicStorageView::PagedRegions {
                        block_bytes: VNEXT_KV_PAGE_BYTES,
                    }
                );
            } else {
                assert_eq!(profile.allocator(), DynamicStorageAllocator::LinearArena);
                assert_eq!(profile.view(), DynamicStorageView::Contiguous);
            }
        }
    }

    #[test]
    fn workspace_layout_matches_estimator_and_binding_capacity() {
        let shape = GptOssAttentionShape::from_attributes(&attributes(128)).unwrap();
        let layout = ScratchLayout::new(shape, 17).unwrap();
        let bindings = BindingLayout::new(shape, 3).unwrap();
        assert_eq!(
            layout.required_bytes,
            shape.scratch_bytes_per_token().unwrap() * 17
        );
        assert_eq!(
            bindings.required_bytes,
            shape.binding_slot_bytes().unwrap() * 3
        );
        assert_eq!(bindings.binding_offset(0).unwrap(), 0);
        assert_eq!(
            bindings.binding_offset(2).unwrap(),
            shape.binding_slot_bytes().unwrap() * 2
        );
        assert_eq!(shape.maximum_pages().unwrap(), 4096);
    }

    #[test]
    fn binding_payload_is_exact_and_bounded_by_slot() {
        let shape = GptOssAttentionShape::from_attributes(&attributes(128)).unwrap();
        let pages = [0x1000_u64, 0x20_0000_u64];
        let payload = binding_payload_from_addresses(1023, 2, 1025, &pages).unwrap();
        assert_eq!(payload.len(), BINDING_CONTROL_BYTES as usize + 16);
        assert!((payload.len() as u64) <= shape.binding_slot_bytes().unwrap());
        assert!(binding_payload_from_addresses(10, 2, 13, &pages).is_err());
        assert!(binding_payload_from_addresses(0, 1, 1, &[]).is_err());
    }

    #[test]
    fn replay_identity_covers_window_layer_and_yarn() {
        let sliding = GptOssAttentionShape::from_attributes(&attributes(128)).unwrap();
        let full = GptOssAttentionShape::from_attributes(&attributes(0)).unwrap();
        let key = |shape| {
            bind_shape_replay_key(
                CudaCommandReplayKeyBuilder::new("provider", COMPUTE_OPERATION),
                shape,
            )
            .finish()
        };
        assert_ne!(key(sliding), key(full));

        let mut changed_layer = sliding;
        changed_layer.layer_index = 2;
        assert_ne!(key(sliding), key(changed_layer));
        let mut changed_yarn = sliding;
        changed_yarn.yarn_beta_fast = 16.0;
        assert_ne!(key(sliding), key(changed_yarn));
    }

    #[test]
    fn replay_identity_treats_the_binding_frontier_as_runtime_data() {
        let launch = |position_start| AttentionLaunch {
            input_region: 0,
            output_region: 1,
            binding_offset: 64,
            packed_token_start: 0,
            normalized_offset: 128,
            query_raw_offset: 256,
            key_raw_offset: 384,
            value_raw_offset: 512,
            query_offset: 640,
            context_offset: 768,
            projected_offset: 896,
            tokens: 1,
            tokens_i32: 1,
            position_start,
            sequence_tokens: position_start + 1,
        };
        let key = |launch| {
            bind_launch_replay_key(
                CudaCommandReplayKeyBuilder::new("provider", COMPUTE_OPERATION),
                launch,
            )
            .finish()
        };

        assert_eq!(key(launch(1)), key(launch(127)));

        let mut wider = launch(127);
        wider.tokens = 2;
        wider.tokens_i32 = 2;
        assert_ne!(key(launch(1)), key(wider));
    }

    #[test]
    fn reusable_topology_treats_the_binding_frontier_as_runtime_data() {
        let shape = GptOssAttentionShape::from_attributes(&attributes(128)).unwrap();
        let topology = |source: Range<u64>, full_input_tokens, immediate_tokens| {
            gpt_oss_reusable_topology_fingerprint(
                shape,
                1,
                [(source, full_input_tokens, immediate_tokens)],
            )
            .unwrap()
        };

        assert_eq!(topology(1..2, 2, 1), topology(127..128, 128, 1));
        assert_ne!(topology(1..2, 2, 1), topology(126..128, 128, 2));
        assert!(gpt_oss_reusable_topology_fingerprint(shape, 2, [(1..2, 2, 1)],).is_err());
    }

    #[test]
    fn source_contains_official_sink_window_and_yarn_semantics() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/kernels/gpt_oss_attention.cu"
        ));
        assert!(source.contains("running_sum = 1.0f"));
        assert!(source.contains("absolute_position - sliding_window + 1"));
        assert!(source.contains("0.1f * logf(yarn_factor) + 1.0f"));
        assert!(source.contains("beta_fast * 2.0f * CUDART_PI_F"));
        assert!(source.contains("beta_slow * 2.0f * CUDART_PI_F"));
    }
}
