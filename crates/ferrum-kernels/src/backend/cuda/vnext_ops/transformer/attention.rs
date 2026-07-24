//! CUDA provider for the standard recurrent gated-delta attention operation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    gated_delta_recurrent_attention_contract, AttributeId, BatchedOperationInvocation,
    CapabilityId, ContractVersion, DeviceBatchingForm, DeviceRuntime, DynamicStorageRequirement,
    ElementType, EncodedDeviceOperation, EncodedReusableExecutionBindings,
    GatedDeltaDecayParameterization, GatedDeltaExecutionCapabilities, GatedDeltaExecutionForm,
    GatedDeltaExecutionPreference, GatedDeltaValueHeadMapping, OperationContract, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProfilePhase, ProviderId,
    ProviderWorkspaceRequirement, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, SemanticValue, VNextError,
    WeightFormatId, GATED_DELTA_EXECUTION_FORM_SELECTOR_VERSION,
    GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
};

use super::{contiguous_bindings, ensure_estimator_request, estimate, launch_gemm_f16};
use crate::backend::cuda::vnext_ops::{
    binding, contiguous_region, contiguous_token_region, contract_error,
    implementation_fingerprint, DENSE_SAFETENSORS_FORMAT_ID, THREADS_PER_BLOCK,
    VALUE_ALIGNMENT_BYTES,
};
use crate::backend::cuda::vnext_replay::CudaCommandReplayKeyBuilder;
use crate::backend::cuda::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};

const PROVIDER_ID: &str = "provider.cuda.gated_delta_recurrent_attention.f16";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.gated_delta_recurrent_attention.f16";

const RMS_NORM_FUNCTION: &str = "rms_norm_f16";
const PREPARE_FUNCTION: &str =
    "linear_attention_prepare_varlen_packed_qkvz_ba_f16_params_f32_state_f16_z_f16_indirect";
const CONV_STATE_COMMIT_FUNCTION: &str = "recurrent_conv_state_commit_f16_indirect";
const QK_NORM_FUNCTION: &str = "linear_attention_qk_l2norm_f32";
const DELTA_FUNCTION: &str = "recurrent_gated_delta_rule_varlen_f32_indirect";
const DELTA_TILED_FUNCTION: &str = "recurrent_gated_delta_rule_varlen_tiled16_f32_indirect";
const GATED_NORM_FUNCTION: &str = "gated_rms_norm_f16_z_f32_weight";
const F32_TO_F16_FUNCTION: &str = "f32_to_activation_f16";
const RESIDUAL_ADD_FUNCTION: &str = "residual_add_f16";

const SCRATCH_ALIGNMENT: u64 = 16;
const CONTROL_BYTES: u64 = 16;
const STATE_BINDING_SLOT_BYTES: u64 = 16;

pub(in crate::backend::cuda::vnext_ops) struct CudaGatedDeltaRecurrentAttentionProvider {
    descriptor: OperationProviderDescriptor,
    execution_capabilities: GatedDeltaExecutionCapabilities,
    functions: AttentionFunctions,
}

#[derive(Clone)]
struct AttentionFunctions {
    rms_norm: CudaFunction,
    prepare: CudaFunction,
    conv_state_commit: CudaFunction,
    qk_norm: CudaFunction,
    delta: CudaFunction,
    delta_tiled: CudaFunction,
    gated_norm: CudaFunction,
    f32_to_f16: CudaFunction,
    residual_add: CudaFunction,
}

impl CudaGatedDeltaRecurrentAttentionProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = gated_delta_recurrent_attention_contract().map_err(contract_error)?;
        let execution_capabilities = GatedDeltaExecutionCapabilities::recurrent_only();
        let capability = CapabilityId::new(GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA runtime does not advertise recurrent gated-delta attention",
            ));
        }

        let source = include_str!("attention.rs");
        let provider_fingerprint = implementation_fingerprint(&[
            source.as_bytes(),
            crate::ptx::RMS_NORM.as_bytes(),
            crate::ptx::LINEAR_ATTENTION.as_bytes(),
            crate::ptx::GATED_DELTA_RULE.as_bytes(),
            crate::ptx::SANDWICH_NORM.as_bytes(),
            crate::ptx::RESIDUAL_ADD.as_bytes(),
            GATED_DELTA_EXECUTION_FORM_SELECTOR_VERSION.as_bytes(),
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
            contiguous_bindings(11),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?;

        let rms_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RMS_NORM.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("attention RMSNorm module", error))?;
        let linear_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::LINEAR_ATTENTION.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("linear attention module", error))?;
        let delta_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::GATED_DELTA_RULE.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("gated delta module", error))?;
        let sandwich_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::SANDWICH_NORM.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("attention cast module", error))?;
        let residual_module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::RESIDUAL_ADD.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("attention residual module", error))?;
        let functions = AttentionFunctions {
            rms_norm: load_function(&rms_module, RMS_NORM_FUNCTION, "attention RMSNorm")?,
            prepare: load_function(&linear_module, PREPARE_FUNCTION, "attention prepare")?,
            conv_state_commit: load_function(
                &linear_module,
                CONV_STATE_COMMIT_FUNCTION,
                "attention convolution-state commit",
            )?,
            qk_norm: load_function(&linear_module, QK_NORM_FUNCTION, "attention QK norm")?,
            delta: load_function(&delta_module, DELTA_FUNCTION, "attention delta")?,
            delta_tiled: load_function(
                &delta_module,
                DELTA_TILED_FUNCTION,
                "attention tiled delta",
            )?,
            gated_norm: load_function(&linear_module, GATED_NORM_FUNCTION, "attention gated norm")?,
            f32_to_f16: load_function(&sandwich_module, F32_TO_F16_FUNCTION, "attention cast")?,
            residual_add: load_function(
                &residual_module,
                RESIDUAL_ADD_FUNCTION,
                "attention residual",
            )?,
        };
        Ok(Self {
            descriptor,
            execution_capabilities,
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

impl OperationResourceEstimator for CudaGatedDeltaRecurrentAttentionProvider {
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
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
        )?;
        let shape = AttentionShape::from_attributes(request.attributes()).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(
                shape.fixed_scratch_bytes().map_err(invalid_plan)?,
                0,
                shape.scratch_bytes_per_token().map_err(invalid_plan)?,
            )?,
            SCRATCH_ALIGNMENT,
            ProviderWorkspaceScope::Invocation,
            DynamicStorageRequirement::contiguous(),
        )?;
        let binding = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::actual_sequences(STATE_BINDING_SLOT_BYTES)?,
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

impl OperationProvider<CudaDeviceRuntime> for CudaGatedDeltaRecurrentAttentionProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_attention(
            self.descriptor.provider_implementation_fingerprint(),
            &self.functions,
            self.execution_capabilities,
            invocation,
        )
        .map_err(|message| {
            OperationFailure::new(
                identity,
                ProfilePhase::Forward,
                "cuda.gated_delta_recurrent_attention.encode",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued CUDA attention identity must be valid")
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
                "cuda.gated_delta_recurrent_attention.encode_reusable_bindings",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued CUDA attention identity must be valid")
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AttentionShape {
    hidden_size: u64,
    key_heads: u64,
    value_heads: u64,
    key_head_dim: u64,
    value_head_dim: u64,
    qkv_features: u64,
    value_features: u64,
    qkvz_features: u64,
    ba_features: u64,
    conv_kernel: u64,
    conv_state_width: u64,
    epsilon: f32,
    layer_index: u64,
    decay_parameterization: GatedDeltaDecayParameterization,
    value_head_mapping: GatedDeltaValueHeadMapping,
}

impl AttentionShape {
    fn from_attributes(attributes: &BTreeMap<AttributeId, SemanticValue>) -> Result<Self, String> {
        let shape = Self {
            hidden_size: unsigned_attribute(attributes, "hidden_size")?,
            key_heads: unsigned_attribute(attributes, "key_heads")?,
            value_heads: unsigned_attribute(attributes, "value_heads")?,
            key_head_dim: unsigned_attribute(attributes, "key_head_dim")?,
            value_head_dim: unsigned_attribute(attributes, "value_head_dim")?,
            qkv_features: unsigned_attribute(attributes, "qkv_features")?,
            value_features: unsigned_attribute(attributes, "value_features")?,
            qkvz_features: unsigned_attribute(attributes, "qkvz_features")?,
            ba_features: unsigned_attribute(attributes, "ba_features")?,
            conv_kernel: unsigned_attribute(attributes, "conv_kernel")?,
            conv_state_width: unsigned_attribute(attributes, "conv_state_width")?,
            epsilon: rational_attribute(attributes, "epsilon")?,
            layer_index: unsigned_attribute(attributes, "layer_index")?,
            decay_parameterization: decay_parameterization_attribute(attributes)?,
            value_head_mapping: value_head_mapping_attribute(attributes)?,
        };
        if shape.decay_parameterization != GatedDeltaDecayParameterization::LogRate
            || shape.value_head_mapping != GatedDeltaValueHeadMapping::GroupedByKeyHead
        {
            return Err(
                "CUDA gated-delta safetensors provider requires log-rate decay and value heads grouped by key head"
                    .to_owned(),
            );
        }
        let qk_features = shape.qk_features()?;
        let expected_qkv = qk_features
            .checked_mul(2)
            .and_then(|value| value.checked_add(shape.value_features))
            .ok_or_else(|| "attention QKV width overflows".to_owned())?;
        let expected_value = shape
            .value_heads
            .checked_mul(shape.value_head_dim)
            .ok_or_else(|| "attention value width overflows".to_owned())?;
        if shape.hidden_size == 0
            || shape.key_heads == 0
            || shape.value_heads == 0
            || shape.key_head_dim == 0
            || shape.value_head_dim == 0
            || shape.conv_kernel < 2
            || shape.value_heads % shape.key_heads != 0
            || shape.qkv_features != expected_qkv
            || shape.value_features != expected_value
            || shape.qkvz_features
                != shape
                    .qkv_features
                    .checked_add(shape.value_features)
                    .ok_or_else(|| "attention QKVZ width overflows".to_owned())?
            || shape.ba_features
                != shape
                    .value_heads
                    .checked_mul(2)
                    .ok_or_else(|| "attention BA width overflows".to_owned())?
            || shape.conv_state_width != shape.conv_kernel - 1
        {
            return Err("recurrent attention attributes are inconsistent".to_owned());
        }
        shape.cuda_shape()?;
        Ok(shape)
    }

    fn qk_features(self) -> Result<u64, String> {
        self.key_heads
            .checked_mul(self.key_head_dim)
            .ok_or_else(|| "attention QK width overflows".to_owned())
    }

    fn conv_state_elements(self) -> Result<u64, String> {
        self.qkv_features
            .checked_mul(self.conv_state_width)
            .ok_or_else(|| "attention convolution state size overflows".to_owned())
    }

    fn fixed_scratch_bytes(self) -> Result<u64, String> {
        aligned_bytes(self.conv_state_elements()?, ElementType::F16.size_bytes())?
            .checked_add(CONTROL_BYTES)
            .ok_or_else(|| "attention fixed scratch size overflows".to_owned())
    }

    fn scratch_bytes_per_token(self) -> Result<u64, String> {
        let qk_features = self.qk_features()?;
        [
            (1, ElementType::U32),
            (self.hidden_size, ElementType::F16),
            (self.qkvz_features, ElementType::F16),
            (self.ba_features, ElementType::F16),
            (self.value_features, ElementType::F16),
            (qk_features, ElementType::F32),
            (qk_features, ElementType::F32),
            (self.value_features, ElementType::F32),
            (self.value_heads, ElementType::F32),
            (self.value_heads, ElementType::F32),
            (self.value_features, ElementType::F32),
            (self.value_features, ElementType::F32),
            (self.hidden_size, ElementType::F16),
        ]
        .into_iter()
        .try_fold(0_u64, |total, (elements, element_type)| {
            total
                .checked_add(aligned_bytes(elements, element_type.size_bytes())?)
                .ok_or_else(|| "attention token scratch size overflows".to_owned())
        })
    }

    fn cuda_shape(self) -> Result<CudaAttentionShape, String> {
        Ok(CudaAttentionShape {
            hidden_size: checked_i32(self.hidden_size, "attention hidden size")?,
            key_heads: checked_i32(self.key_heads, "attention key heads")?,
            value_heads: checked_i32(self.value_heads, "attention value heads")?,
            key_head_dim: checked_i32(self.key_head_dim, "attention key head dimension")?,
            value_head_dim: checked_i32(self.value_head_dim, "attention value head dimension")?,
            qkv_features: checked_i32(self.qkv_features, "attention QKV width")?,
            value_features: checked_i32(self.value_features, "attention value width")?,
            conv_kernel: checked_i32(self.conv_kernel, "attention convolution kernel")?,
            epsilon: self.epsilon,
            scale: (self.key_head_dim as f32).sqrt().recip(),
            tiled_delta: self.key_head_dim == 128 && self.value_head_dim == 128,
        })
    }

    fn validate_launch_extents(self, tokens: u64) -> Result<(), String> {
        let qk_features = self.qk_features()?;
        for (name, extent) in [
            ("token count", tokens),
            (
                "hidden activation elements",
                checked_product(&[tokens, self.hidden_size])?,
            ),
            (
                "QKV activation elements",
                checked_product(&[tokens, self.qkv_features])?,
            ),
            (
                "QKVZ activation elements",
                checked_product(&[tokens, self.qkvz_features])?,
            ),
            (
                "BA activation elements",
                checked_product(&[tokens, self.ba_features])?,
            ),
            (
                "QK activation elements",
                checked_product(&[tokens, qk_features])?,
            ),
            (
                "value activation elements",
                checked_product(&[tokens, self.value_features])?,
            ),
            (
                "gate activation elements",
                checked_product(&[tokens, self.value_heads])?,
            ),
            (
                "convolution weight elements",
                checked_product(&[self.qkv_features, self.conv_kernel])?,
            ),
            ("convolution state elements", self.conv_state_elements()?),
            (
                "delta state elements",
                checked_product(&[self.value_heads, self.value_head_dim, self.key_head_dim])?,
            ),
        ] {
            if extent > i32::MAX as u64 {
                return Err(format!("attention {name} exceed CUDA i32 indexing"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct CudaAttentionShape {
    hidden_size: i32,
    key_heads: i32,
    value_heads: i32,
    key_head_dim: i32,
    value_head_dim: i32,
    qkv_features: i32,
    value_features: i32,
    conv_kernel: i32,
    epsilon: f32,
    scale: f32,
    tiled_delta: bool,
}

#[derive(Debug, Clone, Copy)]
struct ScratchLayout {
    required_bytes: u64,
    conv_state: u64,
    token_seq_indices: u64,
    normalized: u64,
    qkvz: u64,
    z_or_activation: u64,
    ba: u64,
    query: u64,
    key: u64,
    value: u64,
    g: u64,
    beta: u64,
    core: u64,
    gated: u64,
    projected: u64,
}

impl ScratchLayout {
    fn new(shape: AttentionShape, total_tokens: u64) -> Result<Self, String> {
        if total_tokens == 0 {
            return Err("attention scratch cannot be sized for zero tokens".to_owned());
        }
        let mut offset = CONTROL_BYTES;
        let conv_state =
            reserve_fixed(&mut offset, shape.conv_state_elements()?, ElementType::F16)?;
        let token_seq_indices = reserve_tokens(&mut offset, 1, ElementType::U32, total_tokens)?;
        let normalized = reserve_tokens(
            &mut offset,
            shape.hidden_size,
            ElementType::F16,
            total_tokens,
        )?;
        let qkvz = reserve_tokens(
            &mut offset,
            shape.qkvz_features,
            ElementType::F16,
            total_tokens,
        )?;
        let ba = reserve_tokens(
            &mut offset,
            shape.ba_features,
            ElementType::F16,
            total_tokens,
        )?;
        let z_or_activation = reserve_tokens(
            &mut offset,
            shape.value_features,
            ElementType::F16,
            total_tokens,
        )?;
        let qk_features = shape.qk_features()?;
        let query = reserve_tokens(&mut offset, qk_features, ElementType::F32, total_tokens)?;
        let key = reserve_tokens(&mut offset, qk_features, ElementType::F32, total_tokens)?;
        let value = reserve_tokens(
            &mut offset,
            shape.value_features,
            ElementType::F32,
            total_tokens,
        )?;
        let g = reserve_tokens(
            &mut offset,
            shape.value_heads,
            ElementType::F32,
            total_tokens,
        )?;
        let beta = reserve_tokens(
            &mut offset,
            shape.value_heads,
            ElementType::F32,
            total_tokens,
        )?;
        let core = reserve_tokens(
            &mut offset,
            shape.value_features,
            ElementType::F32,
            total_tokens,
        )?;
        let gated = reserve_tokens(
            &mut offset,
            shape.value_features,
            ElementType::F32,
            total_tokens,
        )?;
        let projected = reserve_tokens(
            &mut offset,
            shape.hidden_size,
            ElementType::F16,
            total_tokens,
        )?;
        let expected = shape
            .fixed_scratch_bytes()?
            .checked_add(
                shape
                    .scratch_bytes_per_token()?
                    .checked_mul(total_tokens)
                    .ok_or_else(|| "attention scratch size overflows".to_owned())?,
            )
            .ok_or_else(|| "attention scratch size overflows".to_owned())?;
        if offset != expected {
            return Err("attention scratch layout differs from its estimator".to_owned());
        }
        Ok(Self {
            required_bytes: offset,
            conv_state,
            token_seq_indices,
            normalized,
            qkvz,
            z_or_activation,
            ba,
            query,
            key,
            value,
            g,
            beta,
            core,
            gated,
            projected,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct StateBindingLayout {
    required_bytes: u64,
}

impl StateBindingLayout {
    fn new(participant_count: usize) -> Result<Self, String> {
        let participant_count = u64::try_from(participant_count)
            .map_err(|_| "attention state binding participant count exceeds u64".to_owned())?;
        let required_bytes = STATE_BINDING_SLOT_BYTES
            .checked_mul(participant_count)
            .ok_or_else(|| "attention state binding workspace size overflows".to_owned())?;
        if required_bytes == 0 {
            return Err("attention state binding cannot be empty".to_owned());
        }
        Ok(Self { required_bytes })
    }

    fn offset(self, participant_index: usize) -> Result<u64, String> {
        let participant_index = u64::try_from(participant_index)
            .map_err(|_| "attention state binding index exceeds u64".to_owned())?;
        participant_index
            .checked_mul(STATE_BINDING_SLOT_BYTES)
            .filter(|offset| {
                offset
                    .checked_add(STATE_BINDING_SLOT_BYTES)
                    .is_some_and(|end| end <= self.required_bytes)
            })
            .ok_or_else(|| "attention state binding offset exceeds its workspace".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct AttentionLaunch {
    input_region: usize,
    output_region: usize,
    state_binding_offset: u64,
    host_control: usize,
    execution_form: GatedDeltaExecutionForm,
    tokens: u64,
    tokens_i32: i32,
}

#[derive(Debug, Clone, Copy)]
struct AttentionStateBinding {
    first_state_region: usize,
    host_binding: usize,
    binding_offset: u64,
    conv_state_bytes: u64,
    delta_state_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
struct SharedRegions {
    input_norm: usize,
    qkvz: usize,
    ba: usize,
    conv: usize,
    a_log: usize,
    dt_bias: usize,
    norm: usize,
    output: usize,
    scratch: usize,
    binding: usize,
}

fn encode_attention(
    provider_fingerprint: &str,
    functions: &AttentionFunctions,
    execution_capabilities: GatedDeltaExecutionCapabilities,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID
    {
        return Err("CUDA recurrent attention received another or empty operation".to_owned());
    }
    let first = &invocation.participants()[0];
    let shape = AttentionShape::from_attributes(first.attributes())?;
    validate_signature(first, shape)?;
    for participant in &invocation.participants()[1..] {
        if AttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("CUDA recurrent attention participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape)?;
    }
    let program_binding = invocation.program_binding().cloned();

    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::new(shape, total_tokens)?;
    let binding_layout = StateBindingLayout::new(invocation.participants().len())?;
    let cuda_shape = shape.cuda_shape()?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("CUDA recurrent attention participant ranges are incomplete".to_owned());
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
        input_norm: push_shared_weight(&mut compute_regions, &invocation, 1, ElementType::F16)?,
        qkvz: push_shared_weight(&mut compute_regions, &invocation, 2, ElementType::F16)?,
        ba: push_shared_weight(&mut compute_regions, &invocation, 3, ElementType::F16)?,
        conv: push_shared_weight(&mut compute_regions, &invocation, 4, ElementType::F16)?,
        a_log: push_shared_weight(&mut compute_regions, &invocation, 5, ElementType::F32)?,
        dt_bias: push_shared_weight(&mut compute_regions, &invocation, 6, ElementType::F32)?,
        norm: push_shared_weight(&mut compute_regions, &invocation, 7, ElementType::F32)?,
        output: push_shared_weight(&mut compute_regions, &invocation, 8, ElementType::F16)?,
        scratch: {
            let index = compute_regions.len();
            compute_regions.push(shared_scratch_region(&invocation, layout.required_bytes)?);
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
    let mut binding_host_storage = Vec::with_capacity(invocation.participants().len());
    let mut state_bindings = Vec::with_capacity(invocation.participants().len());
    let mut compute_fence_dependencies =
        Vec::with_capacity(invocation.participants().len().saturating_mul(2));
    let mut host_storage = Vec::with_capacity(invocation.participants().len());
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant_index, (participant, token_range)) in invocation
        .participants()
        .iter()
        .zip(token_ranges)
        .enumerate()
    {
        let tokens = token_range.immediate_tokens();
        shape.validate_launch_extents(tokens)?;
        let source = token_range.source_token_range();
        let packed = token_range.immediate_token_range();
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
        let conv_state = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 9)?,
            ElementType::F16,
        )?;
        let delta_state = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 10)?,
            ElementType::F32,
        )?;
        let first_state_region = binding_regions.len();
        binding_regions.push(conv_state.clone());
        binding_regions.push(delta_state.clone());
        compute_fence_dependencies.push(conv_state.clone());
        compute_fence_dependencies.push(delta_state.clone());
        let binding_offset = binding_layout.offset(participant_index)?;
        let host_binding = binding_host_storage.len();
        binding_host_storage.push(state_binding_payload(&conv_state, &delta_state));
        state_bindings.push(AttentionStateBinding {
            first_state_region,
            host_binding,
            binding_offset,
            conv_state_bytes: conv_state.length_bytes(),
            delta_state_bytes: delta_state.length_bytes(),
        });
        let tokens_i32 = checked_i32(tokens, "attention participant token count")?;
        let execution_form = execution_capabilities
            .select(tokens, GatedDeltaExecutionPreference::RecurrentScan)
            .map_err(|error| error.to_string())?;
        if let GatedDeltaExecutionForm::ChunkedScan(plan) = execution_form {
            return Err(format!(
                "CUDA gated-delta provider selected an uninstalled {} form for {} tokens",
                execution_form.as_str(),
                plan.token_count()
            ));
        }
        let host_control = host_storage.len();
        host_storage.push(sequence_control(tokens_i32 as u32));
        launches.push(AttentionLaunch {
            input_region,
            output_region,
            state_binding_offset: binding_offset,
            host_control,
            execution_form,
            tokens,
            tokens_i32,
        });
    }

    let functions = functions.clone();
    let mut replay_key = CudaCommandReplayKeyBuilder::new(
        provider_fingerprint,
        "vnext_gated_delta_recurrent_attention",
    )
    .u64(shape.hidden_size)
    .u64(shape.key_heads)
    .u64(shape.value_heads)
    .u64(shape.key_head_dim)
    .u64(shape.value_head_dim)
    .u64(shape.qkv_features)
    .u64(shape.value_features)
    .u64(shape.qkvz_features)
    .u64(shape.ba_features)
    .u64(shape.conv_kernel)
    .u64(shape.conv_state_width)
    .f32(shape.epsilon)
    .u64(shape.layer_index)
    .u64(total_tokens)
    .u64(layout.required_bytes)
    .u64(binding_layout.required_bytes)
    .u64(STATE_BINDING_SLOT_BYTES)
    .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.input_region as u64)
            .u64(launch.output_region as u64)
            .u64(launch.state_binding_offset)
            .u64(launch.host_control as u64)
            .bytes(launch.execution_form.as_str().as_bytes())
            .u64(launch.tokens)
            .i32(launch.tokens_i32);
    }
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA recurrent attention participant count exceeds u32".to_owned())?;
    let binding_command = if let Some(program_binding) = program_binding {
        let mut regions = binding_regions.into_iter();
        let destination = regions
            .next()
            .ok_or_else(|| "CUDA recurrent binding destination is missing".to_owned())?;
        let fence_dependencies = regions.collect::<Vec<_>>();
        let mut writes = Vec::with_capacity(state_bindings.len());
        for (index, (state_binding, payload)) in state_bindings
            .into_iter()
            .zip(binding_host_storage)
            .enumerate()
        {
            if state_binding.host_binding != index {
                return Err("CUDA recurrent binding payload order is not canonical".to_owned());
            }
            writes.push(
                super::CudaProgramBindingWrite::new(state_binding.binding_offset, payload)
                    .map_err(|error| error.to_string())?,
            );
        }
        CudaDeviceCommand::program_binding_patch(
            "vnext_gated_delta_recurrent_attention_bindings",
            program_binding,
            destination,
            writes,
            fence_dependencies,
        )
    } else {
        CudaDeviceCommand::operation_with_host_storage_and_blas(
            "vnext_gated_delta_recurrent_attention_bindings",
            binding_regions,
            binding_host_storage,
            move |stream, _blas, regions, host_storage| {
                enqueue_state_bindings(
                    stream,
                    binding_layout,
                    &state_bindings,
                    regions,
                    host_storage,
                )
            },
        )
    }
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

    let compute_command =
        CudaDeviceCommand::replayable_operation_with_host_storage_blas_and_fence_dependencies(
            "vnext_gated_delta_recurrent_attention",
            compute_regions,
            host_storage,
            compute_fence_dependencies,
            replay_key.finish(),
            move |stream, blas, regions, host_storage| {
                for launch in &launches {
                    enqueue_attention(
                        stream,
                        blas,
                        &functions,
                        cuda_shape,
                        shape,
                        layout,
                        shared,
                        *launch,
                        regions,
                        host_storage,
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
                u64::from(participant_count) * 11,
                u64::from(participant_count) * 2,
            )
        })
        .map_err(|error| error.to_string())?;

    Ok(EncodedDeviceOperation::compute(compute_command).with_program_binding(binding_command))
}

fn encode_reusable_attention_bindings(
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<EncodedReusableExecutionBindings<CudaDeviceCommand>, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID
    {
        return Err("CUDA recurrent attention received another or empty operation".to_owned());
    }
    let program_binding = invocation.program_binding().cloned().ok_or_else(|| {
        "CUDA recurrent direct execution requires a compiled program binding".to_owned()
    })?;
    let total_tokens = invocation.work_shape().immediate_tokens();
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA recurrent attention participant count exceeds u32".to_owned())?;
    let binding_layout = StateBindingLayout::new(invocation.participants().len())?;
    let destination = super::shared_binding_region(&invocation, binding_layout.required_bytes)?;
    let mut writes = Vec::with_capacity(invocation.participants().len());
    let mut fence_dependencies =
        Vec::with_capacity(invocation.participants().len().saturating_mul(2));

    // The sealed program identity already binds static attributes, weights,
    // scratch, launch topology, and provider implementation. The hot path
    // materializes only live recurrent-state addresses.
    for (participant_index, participant) in invocation.participants().iter().enumerate() {
        let conv_state = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 9)?,
            ElementType::F16,
        )?;
        let delta_state = contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 10)?,
            ElementType::F32,
        )?;
        writes.push(
            super::CudaProgramBindingWrite::new(
                binding_layout.offset(participant_index)?,
                state_binding_payload(&conv_state, &delta_state),
            )
            .map_err(|error| error.to_string())?,
        );
        fence_dependencies.push(conv_state);
        fence_dependencies.push(delta_state);
    }

    let binding_command = CudaDeviceCommand::program_binding_patch(
        "vnext_gated_delta_recurrent_attention_bindings",
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

fn state_binding_payload(
    conv_state: &CudaBufferRegion,
    delta_state: &CudaBufferRegion,
) -> Box<[u8]> {
    let mut payload = Vec::with_capacity(STATE_BINDING_SLOT_BYTES as usize);
    payload.extend_from_slice(&conv_state.device_ptr().to_le_bytes());
    payload.extend_from_slice(&delta_state.device_ptr().to_le_bytes());
    payload.into_boxed_slice()
}

fn enqueue_state_bindings(
    stream: &CudaStream,
    layout: StateBindingLayout,
    bindings: &[AttentionStateBinding],
    regions: &[CudaBufferRegion],
    host_storage: &[Box<[u8]>],
) -> Result<(), CudaDeviceRuntimeError> {
    let workspace = regions.first().ok_or_else(|| {
        CudaDeviceRuntimeError::contract("attention state binding workspace is missing")
    })?;
    if workspace.element_type() != ElementType::U8
        || workspace.length_bytes() < layout.required_bytes
    {
        return Err(CudaDeviceRuntimeError::contract(
            "attention state binding workspace differs from its admitted estimate",
        ));
    }
    for binding in bindings {
        let states = regions
            .get(binding.first_state_region..binding.first_state_region.saturating_add(2))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "attention state binding physical regions are missing",
                )
            })?;
        let payload = host_storage.get(binding.host_binding).ok_or_else(|| {
            CudaDeviceRuntimeError::contract("attention state binding payload is missing")
        })?;
        if states[0].element_type() != ElementType::F16
            || states[0].length_bytes() != binding.conv_state_bytes
            || states[1].element_type() != ElementType::F32
            || states[1].length_bytes() != binding.delta_state_bytes
            || payload.len() != STATE_BINDING_SLOT_BYTES as usize
        {
            return Err(CudaDeviceRuntimeError::contract(
                "attention state binding payload differs from its physical state",
            ));
        }
        let conv_pointer = u64::from_le_bytes(
            payload[0..8]
                .try_into()
                .expect("validated recurrent binding pointer width"),
        );
        let delta_pointer = u64::from_le_bytes(
            payload[8..16]
                .try_into()
                .expect("validated recurrent binding pointer width"),
        );
        if conv_pointer != states[0].device_ptr() || delta_pointer != states[1].device_ptr() {
            return Err(CudaDeviceRuntimeError::contract(
                "attention state binding pointers changed after encoding",
            ));
        }
        let destination = scratch_pointer(workspace.device_ptr(), binding.binding_offset)?;
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                destination,
                payload.as_ref(),
                stream.cu_stream(),
            )
        }
        .map_err(|error| CudaDeviceRuntimeError::driver("attention state binding upload", error))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn enqueue_attention(
    stream: &CudaStream,
    blas: &CudaBlas,
    functions: &AttentionFunctions,
    cuda: CudaAttentionShape,
    shape: AttentionShape,
    layout: ScratchLayout,
    shared: SharedRegions,
    launch: AttentionLaunch,
    regions: &[CudaBufferRegion],
    host_storage: &[Box<[u8]>],
) -> Result<(), CudaDeviceRuntimeError> {
    let scratch = &regions[shared.scratch];
    if scratch.length_bytes() < layout.required_bytes {
        return Err(CudaDeviceRuntimeError::contract(
            "recurrent attention scratch is smaller than its admitted estimate",
        ));
    }
    let binding = &regions[shared.binding];
    let state_binding = scratch_pointer(binding.device_ptr(), launch.state_binding_offset)?;
    let scratch_base = scratch.device_ptr();
    let cu_seqlens = scratch_base;
    let token_seq_indices = scratch_pointer(scratch_base, layout.token_seq_indices)?;
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            cu_seqlens,
            host_storage[launch.host_control].as_ref(),
            stream.cu_stream(),
        )
    }
    .map_err(|error| CudaDeviceRuntimeError::driver("attention sequence upload", error))?;
    let token_index_bytes = usize::try_from(
        launch
            .tokens
            .checked_mul(ElementType::U32.size_bytes())
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("attention token index size overflows")
            })?,
    )
    .map_err(|_| CudaDeviceRuntimeError::contract("attention token index size exceeds usize"))?;
    unsafe {
        cudarc::driver::result::memset_d8_async(
            token_seq_indices,
            0,
            token_index_bytes,
            stream.cu_stream(),
        )
    }
    .map_err(|error| CudaDeviceRuntimeError::driver("attention token index zero", error))?;

    let input = regions[launch.input_region].device_ptr();
    let output = regions[launch.output_region].device_ptr();
    let normalized = scratch_pointer(scratch_base, layout.normalized)?;
    let qkvz = scratch_pointer(scratch_base, layout.qkvz)?;
    let ba = scratch_pointer(scratch_base, layout.ba)?;
    let z = scratch_pointer(scratch_base, layout.z_or_activation)?;
    let query = scratch_pointer(scratch_base, layout.query)?;
    let key = scratch_pointer(scratch_base, layout.key)?;
    let value = scratch_pointer(scratch_base, layout.value)?;
    let g = scratch_pointer(scratch_base, layout.g)?;
    let beta = scratch_pointer(scratch_base, layout.beta)?;
    let core = scratch_pointer(scratch_base, layout.core)?;
    let gated = scratch_pointer(scratch_base, layout.gated)?;
    let projected = scratch_pointer(scratch_base, layout.projected)?;
    let final_conv_state = scratch_pointer(scratch_base, layout.conv_state)?;

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
            shared.qkvz,
            qkvz,
            checked_i32(shape.qkvz_features, "attention QKVZ width")
                .map_err(CudaDeviceRuntimeError::contract)?,
            "attention QKVZ GEMM",
        ),
        (
            shared.ba,
            ba,
            checked_i32(shape.ba_features, "attention BA width")
                .map_err(CudaDeviceRuntimeError::contract)?,
            "attention BA GEMM",
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
        qkvz,
        ba,
        regions[shared.conv].device_ptr(),
        state_binding,
        regions[shared.a_log].device_ptr(),
        regions[shared.dt_bias].device_ptr(),
        cu_seqlens,
        token_seq_indices,
        query,
        key,
        value,
        z,
        g,
        beta,
        final_conv_state,
        launch.tokens,
        launch.tokens_i32,
        cuda,
        shape,
    )?;
    let conv_state_elements = i32::try_from(
        shape
            .conv_state_elements()
            .map_err(CudaDeviceRuntimeError::contract)?,
    )
    .map_err(|_| {
        CudaDeviceRuntimeError::contract("attention convolution state elements exceed i32")
    })?;
    launch_conv_state_commit(
        stream,
        &functions.conv_state_commit,
        final_conv_state,
        state_binding,
        conv_state_elements,
    )?;
    launch_qk_norm(
        stream,
        &functions.qk_norm,
        query,
        key,
        launch.tokens,
        launch.tokens_i32,
        cuda,
    )?;
    launch_delta(
        stream,
        functions,
        query,
        key,
        value,
        g,
        beta,
        state_binding,
        cu_seqlens,
        core,
        launch.tokens_i32,
        cuda,
    )?;
    launch_gated_norm(
        stream,
        &functions.gated_norm,
        core,
        z,
        regions[shared.norm].device_ptr(),
        gated,
        launch.tokens,
        cuda,
    )?;
    launch_cast(
        stream,
        &functions.f32_to_f16,
        gated,
        z,
        launch
            .tokens
            .checked_mul(shape.value_features)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("attention cast size overflows"))?,
    )?;
    launch_gemm_f16(
        blas,
        z,
        regions[shared.output].device_ptr(),
        projected,
        launch.tokens_i32,
        cuda.hidden_size,
        cuda.value_features,
        "attention output GEMM",
    )?;
    launch_residual(
        stream,
        &functions.residual_add,
        input,
        projected,
        output,
        launch
            .tokens
            .checked_mul(shape.hidden_size)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("attention residual size overflows"))?,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_prepare(
    stream: &CudaStream,
    function: &CudaFunction,
    qkvz: u64,
    ba: u64,
    conv_weight: u64,
    state_binding: u64,
    a_log: u64,
    dt_bias: u64,
    cu_seqlens: u64,
    token_seq_indices: u64,
    query: u64,
    key: u64,
    value: u64,
    z: u64,
    g: u64,
    beta: u64,
    final_conv_state: u64,
    tokens: u64,
    tokens_i32: i32,
    shape: CudaAttentionShape,
    logical: AttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let batch = 1_i32;
    let total = tokens
        .checked_mul(logical.qkv_features)
        .and_then(|conv| {
            tokens
                .checked_mul(logical.value_heads)
                .map(|gate| conv.max(gate))
        })
        .and_then(|work| {
            logical
                .conv_state_elements()
                .ok()
                .map(|state| work.max(state))
        })
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention prepare work overflows"))?;
    let grid = checked_grid(total, THREADS_PER_BLOCK, "attention prepare")?;
    let mut builder = stream.launch_builder(function);
    let pointers = [
        qkvz,
        ba,
        conv_weight,
        state_binding,
        a_log,
        dt_bias,
        cu_seqlens,
        token_seq_indices,
        query,
        key,
        value,
        z,
        g,
        beta,
        final_conv_state,
    ];
    for pointer in &pointers {
        builder.arg(pointer);
    }
    let dimensions = [
        batch,
        tokens_i32,
        shape.key_heads,
        shape.value_heads,
        shape.key_head_dim,
        shape.value_head_dim,
        shape.conv_kernel,
    ];
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("attention prepare launch", error))
}

fn launch_conv_state_commit(
    stream: &CudaStream,
    function: &CudaFunction,
    source: u64,
    state_binding: u64,
    elements: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    if elements <= 0 {
        return Err(CudaDeviceRuntimeError::contract(
            "attention convolution state commit is empty",
        ));
    }
    let mut builder = stream.launch_builder(function);
    builder.arg(&source);
    builder.arg(&state_binding);
    builder.arg(&elements);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_grid(
                    elements as u64,
                    THREADS_PER_BLOCK,
                    "attention convolution state commit",
                )?,
                1,
                1,
            ),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| {
        CudaDeviceRuntimeError::driver("attention convolution state commit launch", error)
    })
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
    let rows = checked_u32(tokens, "attention RMSNorm rows")?;
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
    .map_err(|error| CudaDeviceRuntimeError::driver("attention RMSNorm launch", error))
}

fn launch_qk_norm(
    stream: &CudaStream,
    function: &CudaFunction,
    query: u64,
    key: u64,
    tokens: u64,
    tokens_i32: i32,
    shape: CudaAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let epsilon = 1.0e-6_f32;
    let rows = tokens
        .checked_mul(shape.key_heads as u64)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention QK rows overflow"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&query);
    builder.arg(&key);
    builder.arg(&tokens_i32);
    builder.arg(&shape.key_heads);
    builder.arg(&shape.key_head_dim);
    builder.arg(&epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (checked_u32(rows, "attention QK rows")?, 1, 1),
            block_dim: (
                (shape.key_head_dim as u32).next_power_of_two().min(256),
                1,
                1,
            ),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("attention QK norm launch", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_delta(
    stream: &CudaStream,
    functions: &AttentionFunctions,
    query: u64,
    key: u64,
    value: u64,
    g: u64,
    beta: u64,
    state_binding: u64,
    cu_seqlens: u64,
    output: u64,
    tokens: i32,
    shape: CudaAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let batch = 1_i32;
    let function = if shape.tiled_delta {
        &functions.delta_tiled
    } else {
        &functions.delta
    };
    let pointers = [
        query,
        key,
        value,
        g,
        beta,
        state_binding,
        cu_seqlens,
        output,
    ];
    let dimensions = [
        batch,
        tokens,
        shape.key_heads,
        shape.value_heads,
        shape.key_head_dim,
        shape.value_head_dim,
    ];
    let use_qk_l2norm = 0_i32;
    let mut builder = stream.launch_builder(function);
    for pointer in &pointers {
        builder.arg(pointer);
    }
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    let grid_dim = if shape.tiled_delta {
        builder.arg(&shape.scale);
        (
            (shape.value_head_dim as u32).div_ceil(16),
            shape.value_heads as u32,
            1,
        )
    } else {
        builder.arg(&use_qk_l2norm);
        builder.arg(&shape.scale);
        (shape.value_heads as u32, 1, 1)
    };
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim,
            block_dim: (
                if shape.tiled_delta {
                    256
                } else {
                    shape.value_head_dim.min(256) as u32
                },
                1,
                1,
            ),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("attention delta launch", error))
}

#[allow(clippy::too_many_arguments)]
fn launch_gated_norm(
    stream: &CudaStream,
    function: &CudaFunction,
    core: u64,
    z: u64,
    weight: u64,
    output: u64,
    tokens: u64,
    shape: CudaAttentionShape,
) -> Result<(), CudaDeviceRuntimeError> {
    let rows = tokens
        .checked_mul(shape.value_heads as u64)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention gated rows overflow"))?;
    let rows_i32 = i32::try_from(rows)
        .map_err(|_| CudaDeviceRuntimeError::contract("attention gated rows exceed i32"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&core);
    builder.arg(&z);
    builder.arg(&weight);
    builder.arg(&output);
    builder.arg(&rows_i32);
    builder.arg(&shape.value_head_dim);
    builder.arg(&shape.epsilon);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (checked_u32(rows, "attention gated rows")?, 1, 1),
            block_dim: (
                (shape.value_head_dim as u32).next_power_of_two().min(256),
                1,
                1,
            ),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("attention gated norm launch", error))
}

fn launch_cast(
    stream: &CudaStream,
    function: &CudaFunction,
    input: u64,
    output: u64,
    elements: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let elements_i32 = i32::try_from(elements)
        .map_err(|_| CudaDeviceRuntimeError::contract("attention cast size exceeds i32"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&input);
    builder.arg(&output);
    builder.arg(&elements_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_grid(elements, THREADS_PER_BLOCK, "attention cast")?,
                1,
                1,
            ),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("attention cast launch", error))
}

fn launch_residual(
    stream: &CudaStream,
    function: &CudaFunction,
    input: u64,
    branch: u64,
    output: u64,
    elements: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let elements_i32 = i32::try_from(elements)
        .map_err(|_| CudaDeviceRuntimeError::contract("attention residual size exceeds i32"))?;
    let mut builder = stream.launch_builder(function);
    builder.arg(&input);
    builder.arg(&branch);
    builder.arg(&output);
    builder.arg(&elements_i32);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (
                checked_grid(elements, THREADS_PER_BLOCK, "attention residual")?,
                1,
                1,
            ),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|error| CudaDeviceRuntimeError::driver("attention residual launch", error))
}

fn validate_signature(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    shape: AttentionShape,
) -> Result<(), String> {
    let value = |ordinal| binding(participant.bindings(), ResolvedValueRole::Input, ordinal);
    let hidden = value(0)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let [tokens, hidden_width] = hidden.tensor().dimensions() else {
        return Err("recurrent attention hidden input is not two-dimensional".to_owned());
    };
    let expected = [
        (value(1)?, vec![shape.hidden_size], ElementType::F16),
        (
            value(2)?,
            vec![shape.qkvz_features, shape.hidden_size],
            ElementType::F16,
        ),
        (
            value(3)?,
            vec![shape.ba_features, shape.hidden_size],
            ElementType::F16,
        ),
        (
            value(4)?,
            vec![shape.qkv_features, shape.conv_kernel],
            ElementType::F16,
        ),
        (value(5)?, vec![shape.value_heads], ElementType::F32),
        (value(6)?, vec![shape.value_heads], ElementType::F32),
        (value(7)?, vec![shape.value_head_dim], ElementType::F32),
        (
            value(8)?,
            vec![shape.hidden_size, shape.value_features],
            ElementType::F16,
        ),
        (
            value(9)?,
            vec![shape.qkv_features, shape.conv_state_width],
            ElementType::F16,
        ),
        (
            value(10)?,
            vec![shape.value_heads, shape.value_head_dim, shape.key_head_dim],
            ElementType::F32,
        ),
    ];
    if *tokens == 0
        || *hidden_width != shape.hidden_size
        || output.tensor().dimensions() != [*tokens, shape.hidden_size]
        || !f16_contiguous(hidden)
        || !f16_contiguous(output)
        || expected.iter().any(|(binding, dimensions, element_type)| {
            binding.tensor().dimensions() != dimensions.as_slice()
                || !contiguous(binding, *element_type)
        })
    {
        return Err("recurrent attention signature differs from its resolved shape".to_owned());
    }
    Ok(())
}

fn push_shared_weight(
    regions: &mut Vec<CudaBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    element_type: ElementType,
) -> Result<usize, String> {
    let index = regions.len();
    regions.push(super::shared_full_region(
        invocation,
        ResolvedValueRole::Input,
        ordinal,
        element_type,
    )?);
    Ok(index)
}

fn shared_scratch_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    required_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let first = scratch_region(&invocation.participants()[0], required_bytes)?;
    for participant in &invocation.participants()[1..] {
        let candidate = scratch_region(participant, required_bytes)?;
        if first.device_ptr() != candidate.device_ptr()
            || first.length_bytes() != candidate.length_bytes()
        {
            return Err("recurrent attention batch does not share invocation scratch".to_owned());
        }
    }
    Ok(first)
}

fn scratch_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    required_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let view = participant
        .scratch_view()
        .ok_or_else(|| "recurrent attention invocation has no scratch".to_owned())?;
    if view.descriptor().element_type != ElementType::U8
        || view.descriptor().size_bytes < required_bytes
    {
        return Err("recurrent attention scratch differs from its estimate".to_owned());
    }
    let translated = view
        .translate(0, view.descriptor().size_bytes)
        .map_err(|error| error.to_string())?;
    let mut physical = translated.iter();
    let region = physical
        .next()
        .ok_or_else(|| "recurrent attention scratch has no physical region".to_owned())?;
    if physical.next().is_some() {
        return Err("recurrent attention scratch is not physically contiguous".to_owned());
    }
    let (buffer, range, retention) = region.buffer_and_physical_range();
    buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())
}

fn f16_contiguous(binding: &ResolvedValueBinding) -> bool {
    contiguous(binding, ElementType::F16)
}

fn contiguous(binding: &ResolvedValueBinding, element_type: ElementType) -> bool {
    binding.tensor().element_type() == element_type
        && matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

fn reserve_fixed(
    offset: &mut u64,
    elements: u64,
    element_type: ElementType,
) -> Result<u64, String> {
    let start = *offset;
    *offset = offset
        .checked_add(aligned_bytes(elements, element_type.size_bytes())?)
        .ok_or_else(|| "attention fixed scratch offset overflows".to_owned())?;
    Ok(start)
}

fn reserve_tokens(
    offset: &mut u64,
    elements_per_token: u64,
    element_type: ElementType,
    tokens: u64,
) -> Result<u64, String> {
    let start = *offset;
    let stride = aligned_bytes(elements_per_token, element_type.size_bytes())?;
    *offset = offset
        .checked_add(
            stride
                .checked_mul(tokens)
                .ok_or_else(|| "attention token scratch span overflows".to_owned())?,
        )
        .ok_or_else(|| "attention token scratch offset overflows".to_owned())?;
    Ok(start)
}

fn aligned_bytes(elements: u64, element_bytes: u64) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| "attention scratch element count overflows".to_owned())?;
    bytes
        .checked_add(SCRATCH_ALIGNMENT - 1)
        .map(|value| value & !(SCRATCH_ALIGNMENT - 1))
        .filter(|value| *value > 0)
        .ok_or_else(|| "attention scratch alignment overflows".to_owned())
}

fn checked_product(factors: &[u64]) -> Result<u64, String> {
    factors.iter().try_fold(1_u64, |product, factor| {
        product
            .checked_mul(*factor)
            .ok_or_else(|| "attention launch extent overflows".to_owned())
    })
}

fn sequence_control(tokens: u32) -> Box<[u8]> {
    [0_u32, tokens]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn scratch_pointer(base: u64, offset: u64) -> Result<u64, CudaDeviceRuntimeError> {
    base.checked_add(offset)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("attention scratch pointer overflows"))
}

fn checked_grid(
    elements: u64,
    block_size: u32,
    context: &'static str,
) -> Result<u32, CudaDeviceRuntimeError> {
    u32::try_from(elements.div_ceil(u64::from(block_size)))
        .map_err(|_| CudaDeviceRuntimeError::contract(format!("{context} grid exceeds u32")))
}

fn checked_i32(value: u64, context: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{context} exceeds i32"))
}

fn checked_u32(value: u64, context: &str) -> Result<u32, CudaDeviceRuntimeError> {
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
        _ => Err(format!("CUDA attention lacks unsigned attribute {name:?}")),
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
        _ => return Err(format!("CUDA attention lacks rational attribute {name:?}")),
    };
    let value = (rational.numerator() as f64 / rational.denominator() as f64) as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "CUDA attention rational attribute {name:?} is not a positive f32"
        ));
    }
    Ok(value)
}

fn text_attribute<'a>(
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<&'a str, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Text(value)) => Ok(value),
        _ => Err(format!("CUDA attention lacks text attribute {name:?}")),
    }
}

fn decay_parameterization_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<GatedDeltaDecayParameterization, String> {
    let value = text_attribute(attributes, "decay_parameterization")?;
    GatedDeltaDecayParameterization::parse(value)
        .ok_or_else(|| format!("CUDA attention has unsupported decay parameterization {value:?}"))
}

fn value_head_mapping_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<GatedDeltaValueHeadMapping, String> {
    let value = text_attribute(attributes, "value_head_mapping")?;
    GatedDeltaValueHeadMapping::parse(value)
        .ok_or_else(|| format!("CUDA attention has unsupported value-head mapping {value:?}"))
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_extent_validation_covers_packed_projection_strides() {
        let mut shape = AttentionShape {
            hidden_size: 16,
            key_heads: 2,
            value_heads: 8,
            key_head_dim: 16,
            value_head_dim: 16,
            qkv_features: 192,
            value_features: 128,
            qkvz_features: 320,
            ba_features: 16,
            conv_kernel: 4,
            conv_state_width: 3,
            epsilon: 1.0e-6,
            layer_index: 0,
            decay_parameterization: GatedDeltaDecayParameterization::LogRate,
            value_head_mapping: GatedDeltaValueHeadMapping::GroupedByKeyHead,
        };

        let tokens = i32::MAX as u64 / shape.qkvz_features + 1;
        assert!(shape
            .validate_launch_extents(tokens)
            .unwrap_err()
            .contains("QKVZ activation elements"));

        shape.ba_features = i32::MAX as u64;
        assert!(shape
            .validate_launch_extents(2)
            .unwrap_err()
            .contains("BA activation elements"));
    }
}
