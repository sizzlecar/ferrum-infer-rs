//! Native Metal provider for the standard recurrent gated-delta operation.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    gated_delta_recurrent_attention_contract, AttributeId, BatchedOperationInvocation,
    DeviceBatchingForm, DynamicStorageRequirement, ElementType, EncodedDeviceOperation,
    GatedDeltaDecayParameterization, GatedDeltaExecutionCapabilities, GatedDeltaExecutionForm,
    GatedDeltaExecutionPreference, GatedDeltaValueHeadMapping, OperationFailure,
    OperationInvocation, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, ProviderWorkspaceRequirement,
    ProviderWorkspaceScope, ProviderWorkspaceSizeFormula, ResolvedTensorLayout,
    ResolvedValueBinding, ResolvedValueRole, SemanticValue, VNextError,
    GATED_DELTA_EXECUTION_FORM_SELECTOR_VERSION, GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
};
use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, FunctionConstantValues,
    MTLDataType, MTLSize,
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
    binding, checked_u32, contiguous_bindings, contiguous_region, contiguous_token_region,
    ensure_invocation, f16_contiguous, implementation_fingerprint, invalid_plan,
    provider_descriptor, provider_failure, rational_attribute, shared_full_region,
    shared_scratch_region, shared_token_region, token_binding_is_shared, unsigned_attribute,
    DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID, Q5_K_FORMAT_ID,
    Q6_K_FORMAT_ID, Q8_0_FORMAT_ID, THREADS_PER_GROUP, VALUE_ALIGNMENT_BYTES,
};

const SHADER_SOURCE: &str = include_str!("gated_delta_attention.metal");
const PROVIDER_ID: &str = "provider.metal.gated_delta_recurrent_attention.f16.native";
const ESTIMATOR_ID: &str = "resource-estimator.metal.gated_delta_recurrent_attention.f16.native";
const PREPARE_CONV_KERNEL: &str = "vnext_gated_delta_prepare_conv_f16";
const PREPARE_GATES_KERNEL: &str = "vnext_gated_delta_prepare_gates_f16";
const COLLECT_CONV_STATE_KERNEL: &str = "vnext_gated_delta_collect_conv_state_f16";
const COPY_F16_KERNEL: &str = "vnext_gated_delta_copy_f16";
const QK_NORM_KERNEL: &str = "vnext_gated_delta_qk_norm_f32";
const DELTA_KERNEL: &str = "vnext_gated_delta_rule_tiled16_f32_state";
const CHUNK_KKT_INVERSE_KERNEL: &str = "vnext_gated_delta_chunk_kkt_inverse_c64";
const CHUNK_UW_KERNEL: &str = "vnext_gated_delta_chunk_uw_c64";
const CHUNK_QK_KERNEL: &str = "vnext_gated_delta_chunk_qk_c64";
const CHUNK_CARRY_KERNEL: &str = "vnext_gated_delta_chunk_carry_c64";
const CHUNK_OUTPUT_KERNEL: &str = "vnext_gated_delta_chunk_output_c64";
const GATED_NORM_KERNEL: &str = "vnext_gated_delta_gated_norm_f16";
const CONV_STATE_ELEMENT_TYPE: ElementType = ElementType::F16;
const DELTA_STATE_ELEMENT_TYPE: ElementType = ElementType::F32;
const VALUE_TILE: u64 = 16;
const GATED_DELTA_CHUNK_SIZE: u32 = 64;
const GATED_DELTA_CHUNK_KEY_DIM_LIMIT: u64 = 128;
const GATED_DELTA_CHUNK_INITIAL_CROSSOVER_TOKENS: u64 = 64;

#[derive(Debug, Clone, Copy)]
struct MetalGatedDeltaExecutionCostModel {
    chunked_scan_crossover_tokens: u64,
}

impl MetalGatedDeltaExecutionCostModel {
    const fn initial_c64() -> Self {
        Self {
            chunked_scan_crossover_tokens: GATED_DELTA_CHUNK_INITIAL_CROSSOVER_TOKENS,
        }
    }

    const fn preference(self, tokens: u64) -> GatedDeltaExecutionPreference {
        if tokens >= self.chunked_scan_crossover_tokens {
            GatedDeltaExecutionPreference::ChunkedScan
        } else {
            GatedDeltaExecutionPreference::RecurrentScan
        }
    }
}

pub(super) struct MetalGatedDeltaPipelines {
    prepare_conv: ComputePipelineState,
    prepare_gates: ComputePipelineState,
    collect_conv_state: ComputePipelineState,
    copy_f16: ComputePipelineState,
    qk_norm: ComputePipelineState,
    delta: ComputePipelineState,
    chunk_kkt_inverse: ComputePipelineState,
    chunk_uw: ComputePipelineState,
    chunk_qk: ComputePipelineState,
    chunk_carry_generic: ComputePipelineState,
    chunk_carry_k128: ComputePipelineState,
    chunk_output: ComputePipelineState,
    gated_norm: ComputePipelineState,
}

impl MetalGatedDeltaPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile Metal vNext gated-delta library: {error}"
                ))
            })?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "load Metal vNext gated-delta `{name}`: {error}"
                ))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext gated-delta `{name}`: {error}"
                    ))
                })
        };
        let carry_pipeline = |key_dim_128: bool| {
            let constants = FunctionConstantValues::new();
            constants.set_constant_value_at_index(
                &key_dim_128 as *const bool as *const c_void,
                MTLDataType::Bool,
                0,
            );
            let function = library
                .get_function(CHUNK_CARRY_KERNEL, Some(constants))
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "load Metal vNext gated-delta {CHUNK_CARRY_KERNEL} \
                         key_dim_128={key_dim_128}: {error}"
                    ))
                })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext gated-delta {CHUNK_CARRY_KERNEL} \
                         key_dim_128={key_dim_128}: {error}"
                    ))
                })
        };
        Ok(Self {
            prepare_conv: pipeline(PREPARE_CONV_KERNEL)?,
            prepare_gates: pipeline(PREPARE_GATES_KERNEL)?,
            collect_conv_state: pipeline(COLLECT_CONV_STATE_KERNEL)?,
            copy_f16: pipeline(COPY_F16_KERNEL)?,
            qk_norm: pipeline(QK_NORM_KERNEL)?,
            delta: pipeline(DELTA_KERNEL)?,
            chunk_kkt_inverse: pipeline(CHUNK_KKT_INVERSE_KERNEL)?,
            chunk_uw: pipeline(CHUNK_UW_KERNEL)?,
            chunk_qk: pipeline(CHUNK_QK_KERNEL)?,
            chunk_carry_generic: carry_pipeline(false)?,
            chunk_carry_k128: carry_pipeline(true)?,
            chunk_output: pipeline(CHUNK_OUTPUT_KERNEL)?,
            gated_norm: pipeline(GATED_NORM_KERNEL)?,
        })
    }
}

pub(super) struct MetalGatedDeltaRecurrentAttentionProvider {
    descriptor: OperationProviderDescriptor,
    execution_capabilities: GatedDeltaExecutionCapabilities,
    execution_cost_model: MetalGatedDeltaExecutionCostModel,
    attention: Arc<MetalGatedDeltaPipelines>,
    linear: Arc<MetalLinearPipelines>,
    primitives: Arc<MetalPrimitivePipelines>,
}

impl MetalGatedDeltaRecurrentAttentionProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        attention: Arc<MetalGatedDeltaPipelines>,
        linear: Arc<MetalLinearPipelines>,
        primitives: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = gated_delta_recurrent_attention_contract().map_err(super::contract_error)?;
        let execution_capabilities =
            GatedDeltaExecutionCapabilities::with_chunked_scan(GATED_DELTA_CHUNK_SIZE)
                .map_err(super::contract_error)?;
        let execution_cost_model = MetalGatedDeltaExecutionCostModel::initial_c64();
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            PROVIDER_ID,
            GATED_DELTA_RECURRENT_ATTENTION_F16_CAPABILITY_ID,
            ESTIMATOR_ID,
            contiguous_bindings(13),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[
                Q4_K_FORMAT_ID,
                Q5_K_FORMAT_ID,
                Q6_K_FORMAT_ID,
                Q8_0_FORMAT_ID,
            ],
            implementation_fingerprint(&[
                include_str!("gated_delta_attention.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                include_str!("linear.rs").as_bytes(),
                include_str!("linear.metal").as_bytes(),
                include_str!("primitives.rs").as_bytes(),
                include_str!("primitives.metal").as_bytes(),
                GATED_DELTA_EXECUTION_FORM_SELECTOR_VERSION.as_bytes(),
                PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            execution_capabilities,
            execution_cost_model,
            attention,
            linear,
            primitives,
        })
    }
}

impl OperationResourceEstimator for MetalGatedDeltaRecurrentAttentionProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "Metal estimator `{}` received another operation",
                self.descriptor.resource_estimator_id()
            )));
        }
        let shape = AttentionShape::from_attributes(request.attributes()).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(
                shape.fixed_scratch_bytes().map_err(invalid_plan)?,
                0,
                shape.scratch_bytes_per_token().map_err(invalid_plan)?,
            )?,
            VALUE_ALIGNMENT_BYTES,
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
        ))
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalGatedDeltaRecurrentAttentionProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_attention(
            Arc::clone(&self.attention),
            Arc::clone(&self.linear),
            Arc::clone(&self.primitives),
            self.execution_capabilities,
            self.execution_cost_model,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(
                identity,
                "metal.gated_delta_recurrent_attention.encode",
                message,
            )
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AttentionShape {
    hidden_size: u64,
    key_heads: u64,
    value_heads: u64,
    key_dim: u64,
    value_dim: u64,
    qkv_features: u64,
    value_features: u64,
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
            key_dim: unsigned_attribute(attributes, "key_head_dim")?,
            value_dim: unsigned_attribute(attributes, "value_head_dim")?,
            qkv_features: unsigned_attribute(attributes, "qkv_features")?,
            value_features: unsigned_attribute(attributes, "value_features")?,
            conv_kernel: unsigned_attribute(attributes, "conv_kernel")?,
            conv_state_width: unsigned_attribute(attributes, "conv_state_width")?,
            epsilon: rational_attribute(attributes, "epsilon")?,
            layer_index: unsigned_attribute(attributes, "layer_index")?,
            decay_parameterization: decay_parameterization_attribute(attributes)?,
            value_head_mapping: value_head_mapping_attribute(attributes)?,
        };
        let qk_features = shape.qk_features()?;
        let expected_qkv = qk_features
            .checked_mul(2)
            .and_then(|value| value.checked_add(shape.value_features))
            .ok_or_else(|| "Metal gated-delta QKV width overflows".to_owned())?;
        let expected_value = shape
            .value_heads
            .checked_mul(shape.value_dim)
            .ok_or_else(|| "Metal gated-delta value width overflows".to_owned())?;
        if shape.hidden_size == 0
            || shape.key_heads == 0
            || shape.value_heads == 0
            || shape.key_dim == 0
            || shape.value_dim == 0
            || shape.value_heads % shape.key_heads != 0
            || shape.qkv_features != expected_qkv
            || shape.value_features != expected_value
            || shape.conv_kernel < 2
            || shape.conv_state_width != shape.conv_kernel - 1
        {
            return Err("Metal gated-delta attributes are inconsistent".to_owned());
        }
        for (elements, element_type) in shape.scratch_rows()? {
            if elements
                .checked_mul(element_type.size_bytes())
                .is_none_or(|bytes| !bytes.is_multiple_of(VALUE_ALIGNMENT_BYTES))
            {
                return Err(
                    "Metal gated-delta scratch row does not preserve 16-byte packing".to_owned(),
                );
            }
        }
        shape.params(1)?;
        shape.validate_launch_extents(1)?;
        Ok(shape)
    }

    fn qk_features(self) -> Result<u64, String> {
        self.key_heads
            .checked_mul(self.key_dim)
            .ok_or_else(|| "Metal gated-delta QK width overflows".to_owned())
    }

    fn conv_state_elements(self) -> Result<u64, String> {
        self.qkv_features
            .checked_mul(self.conv_state_width)
            .ok_or_else(|| "Metal gated-delta convolution state size overflows".to_owned())
    }

    fn scratch_rows(self) -> Result<Vec<(u64, ElementType)>, String> {
        let qk = self.qk_features()?;
        Ok(vec![
            (self.hidden_size, ElementType::F16),
            (self.qkv_features, ElementType::F16),
            (self.value_features, ElementType::F16),
            (self.value_heads, ElementType::F16),
            (self.value_heads, ElementType::F16),
            (qk, ElementType::F32),
            (qk, ElementType::F32),
            (self.value_features, ElementType::F32),
            (self.value_heads, ElementType::F32),
            (self.value_heads, ElementType::F32),
            (self.value_features, ElementType::F32),
        ])
    }

    fn fixed_scratch_bytes(self) -> Result<u64, String> {
        aligned_bytes(self.conv_state_elements()?, ElementType::F16)
    }

    fn scratch_bytes_per_token(self) -> Result<u64, String> {
        self.scratch_rows()?
            .into_iter()
            .try_fold(0_u64, |total, (elements, element_type)| {
                total
                    .checked_add(aligned_bytes(elements, element_type)?)
                    .ok_or_else(|| "Metal gated-delta token scratch size overflows".to_owned())
            })
    }

    fn params(self, tokens: u64) -> Result<GatedDeltaParams, String> {
        Ok(GatedDeltaParams {
            tokens: checked_u32(tokens, "Metal gated-delta token count")?,
            hidden_size: checked_u32(self.hidden_size, "Metal gated-delta hidden size")?,
            key_heads: checked_u32(self.key_heads, "Metal gated-delta key heads")?,
            value_heads: checked_u32(self.value_heads, "Metal gated-delta value heads")?,
            key_dim: checked_u32(self.key_dim, "Metal gated-delta key dimension")?,
            value_dim: checked_u32(self.value_dim, "Metal gated-delta value dimension")?,
            qkv_features: checked_u32(self.qkv_features, "Metal gated-delta QKV width")?,
            value_features: checked_u32(self.value_features, "Metal gated-delta value width")?,
            conv_kernel: checked_u32(self.conv_kernel, "Metal gated-delta convolution kernel")?,
            epsilon: self.epsilon,
            scale: (self.key_dim as f32).sqrt().recip(),
            decay_parameterization: match self.decay_parameterization {
                GatedDeltaDecayParameterization::LogRate => 0,
                GatedDeltaDecayParameterization::NegativeRate => 1,
            },
            value_head_mapping: match self.value_head_mapping {
                GatedDeltaValueHeadMapping::GroupedByKeyHead => 0,
                GatedDeltaValueHeadMapping::InterleavedByKeyHead => 1,
            },
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
                checked_product(&[self.value_heads, self.value_dim, self.key_dim])?,
            ),
        ] {
            if extent > u64::from(u32::MAX) {
                return Err(format!("Metal gated-delta {name} exceed MSL uint indexing"));
            }
        }
        Ok(())
    }

    fn supports_chunked_scan_c64(self) -> Result<bool, String> {
        if self.key_dim > GATED_DELTA_CHUNK_KEY_DIM_LIMIT {
            return Ok(false);
        }
        let chunk = u64::from(GATED_DELTA_CHUNK_SIZE);
        let inverse_bytes =
            checked_product(&[self.value_heads, chunk, ElementType::F16.size_bytes()])?;
        let core_bytes = checked_product(&[
            self.value_heads,
            self.value_dim,
            ElementType::F32.size_bytes(),
        ])?;
        let uw_bytes = checked_product(&[
            self.value_heads,
            self.value_dim
                .checked_add(self.key_dim)
                .ok_or_else(|| "Metal gated-delta chunk U/W width overflows".to_owned())?,
            ElementType::F16.size_bytes(),
        ])?;
        let qkv_bytes = checked_product(&[self.qkv_features, ElementType::F16.size_bytes()])?;
        let raw_qk_bytes =
            checked_product(&[self.key_heads, chunk, ElementType::F16.size_bytes()])?;
        let value_bytes = checked_product(&[
            self.value_heads,
            self.value_dim,
            ElementType::F32.size_bytes(),
        ])?;
        Ok(inverse_bytes <= core_bytes && uw_bytes <= qkv_bytes && raw_qk_bytes <= value_bytes)
    }

    fn validate_chunked_launch_extents(self, tokens: u64) -> Result<(), String> {
        let chunk = u64::from(GATED_DELTA_CHUNK_SIZE);
        let uw_width = self
            .value_dim
            .checked_add(self.key_dim)
            .ok_or_else(|| "Metal gated-delta chunk U/W width overflows".to_owned())?;
        for (name, extent) in [
            (
                "chunk inverse elements",
                checked_product(&[tokens, self.value_heads, chunk])?,
            ),
            (
                "chunk U/W elements",
                checked_product(&[tokens, self.value_heads, uw_width])?,
            ),
            (
                "chunk QK elements",
                checked_product(&[tokens, self.key_heads, chunk])?,
            ),
        ] {
            if extent > u64::from(u32::MAX) {
                return Err(format!("Metal gated-delta {name} exceed MSL uint indexing"));
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GatedDeltaParams {
    tokens: u32,
    hidden_size: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    qkv_features: u32,
    value_features: u32,
    conv_kernel: u32,
    epsilon: f32,
    scale: f32,
    decay_parameterization: u32,
    value_head_mapping: u32,
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
        _ => Err(format!("Metal gated-delta lacks text attribute {name:?}")),
    }
}

fn decay_parameterization_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<GatedDeltaDecayParameterization, String> {
    let value = text_attribute(attributes, "decay_parameterization")?;
    GatedDeltaDecayParameterization::parse(value).ok_or_else(|| {
        format!("Metal gated-delta has unsupported decay parameterization {value:?}")
    })
}

fn value_head_mapping_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<GatedDeltaValueHeadMapping, String> {
    let value = text_attribute(attributes, "value_head_mapping")?;
    GatedDeltaValueHeadMapping::parse(value)
        .ok_or_else(|| format!("Metal gated-delta has unsupported value-head mapping {value:?}"))
}

#[derive(Debug, Clone, Copy)]
struct ScratchLayout {
    required_bytes: u64,
    conv_state: u64,
    normalized: u64,
    qkv: u64,
    z: u64,
    b: u64,
    a: u64,
    query: u64,
    key: u64,
    value: u64,
    g: u64,
    beta: u64,
    core: u64,
}

impl ScratchLayout {
    fn new(shape: AttentionShape, tokens: u64) -> Result<Self, String> {
        if tokens == 0 {
            return Err("Metal gated-delta scratch cannot represent zero tokens".to_owned());
        }
        let mut offset = 0_u64;
        let conv_state =
            reserve_fixed(&mut offset, shape.conv_state_elements()?, ElementType::F16)?;
        let normalized = reserve_tokens(&mut offset, shape.hidden_size, ElementType::F16, tokens)?;
        let qkv = reserve_tokens(&mut offset, shape.qkv_features, ElementType::F16, tokens)?;
        let z = reserve_tokens(&mut offset, shape.value_features, ElementType::F16, tokens)?;
        let b = reserve_tokens(&mut offset, shape.value_heads, ElementType::F16, tokens)?;
        let a = reserve_tokens(&mut offset, shape.value_heads, ElementType::F16, tokens)?;
        let qk = shape.qk_features()?;
        let query = reserve_tokens(&mut offset, qk, ElementType::F32, tokens)?;
        let key = reserve_tokens(&mut offset, qk, ElementType::F32, tokens)?;
        let value = reserve_tokens(&mut offset, shape.value_features, ElementType::F32, tokens)?;
        let g = reserve_tokens(&mut offset, shape.value_heads, ElementType::F32, tokens)?;
        let beta = reserve_tokens(&mut offset, shape.value_heads, ElementType::F32, tokens)?;
        let core = reserve_tokens(&mut offset, shape.value_features, ElementType::F32, tokens)?;
        let expected = shape
            .fixed_scratch_bytes()?
            .checked_add(
                shape
                    .scratch_bytes_per_token()?
                    .checked_mul(tokens)
                    .ok_or_else(|| "Metal gated-delta scratch size overflows".to_owned())?,
            )
            .ok_or_else(|| "Metal gated-delta scratch size overflows".to_owned())?;
        if offset != expected {
            return Err("Metal gated-delta scratch differs from its estimator".to_owned());
        }
        Ok(Self {
            required_bytes: offset,
            conv_state,
            normalized,
            qkv,
            z,
            b,
            a,
            query,
            key,
            value,
            g,
            beta,
            core,
        })
    }

    fn token_offset(
        self,
        base: u64,
        first_token: u64,
        elements_per_token: u64,
        element_type: ElementType,
    ) -> Result<u64, String> {
        base.checked_add(
            first_token
                .checked_mul(elements_per_token)
                .and_then(|elements| elements.checked_mul(element_type.size_bytes()))
                .ok_or_else(|| "Metal gated-delta packed-token offset overflows".to_owned())?,
        )
        .ok_or_else(|| "Metal gated-delta scratch offset overflows".to_owned())
    }
}

#[derive(Debug, Clone, Copy)]
struct SharedRegions {
    input_norm: usize,
    conv: usize,
    a_log: usize,
    dt_bias: usize,
    norm: usize,
    scratch: usize,
}

struct ParticipantLaunch {
    input: usize,
    output: usize,
    conv_state: usize,
    delta_state: usize,
    normalized: u64,
    qkv: u64,
    z: u64,
    b: u64,
    a: u64,
    query: u64,
    key: u64,
    value: u64,
    g: u64,
    beta: u64,
    core: u64,
    residual_elements: u32,
    conv_state_elements: u32,
    execution_form: GatedDeltaExecutionForm,
    params: GatedDeltaParams,
    qkv_projection: LinearLaunch,
    z_projection: LinearLaunch,
    b_projection: LinearLaunch,
    a_projection: LinearLaunch,
    output_projection: LinearLaunch,
}

#[derive(Debug, Clone, Copy)]
struct PackedLaunch {
    input: usize,
    output: usize,
    residual_elements: u32,
    params: GatedDeltaParams,
    qkv_projection: LinearLaunch,
    z_projection: LinearLaunch,
    b_projection: LinearLaunch,
    a_projection: LinearLaunch,
    output_projection: LinearLaunch,
}

fn encode_attention(
    attention: Arc<MetalGatedDeltaPipelines>,
    linear: Arc<MetalLinearPipelines>,
    primitives: Arc<MetalPrimitivePipelines>,
    execution_capabilities: GatedDeltaExecutionCapabilities,
    execution_cost_model: MetalGatedDeltaExecutionCostModel,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let shape = AttentionShape::from_attributes(first.attributes())?;
    validate_signature(first, shape)?;
    for participant in &invocation.participants()[1..] {
        if AttentionShape::from_attributes(participant.attributes())? != shape {
            return Err("Metal gated-delta participant attributes disagree".to_owned());
        }
        validate_signature(participant, shape)?;
    }
    let total_tokens = invocation.work_shape().immediate_tokens();
    let layout = ScratchLayout::new(shape, total_tokens)?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal gated-delta participant ranges are incomplete".to_owned());
    }

    let mut regions = Vec::new();
    let qkv_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        2,
        shape.qkv_features,
        shape.hidden_size,
        "Metal gated-delta QKV projection",
    )?;
    let z_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        3,
        shape.value_features,
        shape.hidden_size,
        "Metal gated-delta Z projection",
    )?;
    let b_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        4,
        shape.value_heads,
        shape.hidden_size,
        "Metal gated-delta B projection",
    )?;
    let a_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        5,
        shape.value_heads,
        shape.hidden_size,
        "Metal gated-delta A projection",
    )?;
    let output_weight = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        10,
        shape.hidden_size,
        shape.value_features,
        "Metal gated-delta output projection",
    )?;
    let shared = SharedRegions {
        input_norm: push_shared_region(&mut regions, &invocation, 1, ElementType::F16)?,
        conv: push_shared_region(&mut regions, &invocation, 6, ElementType::F16)?,
        a_log: push_shared_region(&mut regions, &invocation, 7, ElementType::F32)?,
        dt_bias: push_shared_region(&mut regions, &invocation, 8, ElementType::F32)?,
        norm: push_shared_region(&mut regions, &invocation, 9, ElementType::F32)?,
        scratch: {
            let index = regions.len();
            regions.push(shared_scratch_region(&invocation, layout.required_bytes)?);
            index
        },
    };
    let input_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let output_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Output, 0, ElementType::F16)?;
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let tokens = token_range.immediate_tokens();
        shape.validate_launch_extents(tokens)?;
        let participant_capabilities = if shape.supports_chunked_scan_c64()? {
            execution_capabilities
        } else {
            GatedDeltaExecutionCapabilities::recurrent_only()
        };
        let execution_form = participant_capabilities
            .select(tokens, execution_cost_model.preference(tokens))
            .map_err(|error| error.to_string())?;
        if matches!(execution_form, GatedDeltaExecutionForm::ChunkedScan(_)) {
            shape.validate_chunked_launch_extents(tokens)?;
        }
        let packed_start = token_range.immediate_token_range().start;
        let input_start = if input_shared {
            packed_start
        } else {
            token_range.source_token_range().start
        };
        let output_start = if output_shared {
            packed_start
        } else {
            token_range.source_token_range().start
        };
        let input = regions.len();
        regions.push(contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::F16,
            input_start,
            tokens,
        )?);
        let output = regions.len();
        regions.push(contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F16,
            output_start,
            tokens,
        )?);
        let conv_state = regions.len();
        regions.push(contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 11)?,
            CONV_STATE_ELEMENT_TYPE,
        )?);
        let delta_state = regions.len();
        regions.push(contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 12)?,
            DELTA_STATE_ELEMENT_TYPE,
        )?);
        let normalized = layout.token_offset(
            layout.normalized,
            packed_start,
            shape.hidden_size,
            ElementType::F16,
        )?;
        let qkv = layout.token_offset(
            layout.qkv,
            packed_start,
            shape.qkv_features,
            ElementType::F16,
        )?;
        let z = layout.token_offset(
            layout.z,
            packed_start,
            shape.value_features,
            ElementType::F16,
        )?;
        let b = layout.token_offset(layout.b, packed_start, shape.value_heads, ElementType::F16)?;
        let a = layout.token_offset(layout.a, packed_start, shape.value_heads, ElementType::F16)?;
        let qk = shape.qk_features()?;
        let query = layout.token_offset(layout.query, packed_start, qk, ElementType::F32)?;
        let key = layout.token_offset(layout.key, packed_start, qk, ElementType::F32)?;
        let value = layout.token_offset(
            layout.value,
            packed_start,
            shape.value_features,
            ElementType::F32,
        )?;
        let g = layout.token_offset(layout.g, packed_start, shape.value_heads, ElementType::F32)?;
        let beta = layout.token_offset(
            layout.beta,
            packed_start,
            shape.value_heads,
            ElementType::F32,
        )?;
        let core = layout.token_offset(
            layout.core,
            packed_start,
            shape.value_features,
            ElementType::F32,
        )?;
        let qkv_projection = linear_launch(
            qkv_weight,
            shared.scratch,
            shared.scratch,
            tokens,
            shape.hidden_size,
            shape.qkv_features,
            normalized,
            qkv,
        )?;
        let z_projection = linear_launch(
            z_weight,
            shared.scratch,
            shared.scratch,
            tokens,
            shape.hidden_size,
            shape.value_features,
            normalized,
            z,
        )?;
        let b_projection = linear_launch(
            b_weight,
            shared.scratch,
            shared.scratch,
            tokens,
            shape.hidden_size,
            shape.value_heads,
            normalized,
            b,
        )?;
        let a_projection = linear_launch(
            a_weight,
            shared.scratch,
            shared.scratch,
            tokens,
            shape.hidden_size,
            shape.value_heads,
            normalized,
            a,
        )?;
        let output_projection = linear_launch(
            output_weight,
            shared.scratch,
            shared.scratch,
            tokens,
            shape.value_features,
            shape.hidden_size,
            qkv,
            normalized,
        )?;
        let residual_elements = checked_u32(
            checked_product(&[tokens, shape.hidden_size])?,
            "Metal gated-delta residual elements",
        )?;
        let conv_state_elements = checked_u32(
            shape.conv_state_elements()?,
            "Metal gated-delta convolution state elements",
        )?;
        launches.push(ParticipantLaunch {
            input,
            output,
            conv_state,
            delta_state,
            normalized,
            qkv,
            z,
            b,
            a,
            query,
            key,
            value,
            g,
            beta,
            core,
            residual_elements,
            conv_state_elements,
            execution_form,
            params: shape.params(tokens)?,
            qkv_projection,
            z_projection,
            b_projection,
            a_projection,
            output_projection,
        });
    }
    let packed = if input_shared && output_shared && launches.len() > 1 {
        shape.validate_launch_extents(total_tokens)?;
        let input = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            total_tokens,
        )?);
        let output = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            total_tokens,
        )?);
        let packed = PackedLaunch {
            input,
            output,
            residual_elements: checked_u32(
                checked_product(&[total_tokens, shape.hidden_size])?,
                "Metal packed gated-delta residual elements",
            )?,
            params: shape.params(total_tokens)?,
            qkv_projection: linear_launch(
                qkv_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.qkv_features,
                layout.normalized,
                layout.qkv,
            )?,
            z_projection: linear_launch(
                z_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.value_features,
                layout.normalized,
                layout.z,
            )?,
            b_projection: linear_launch(
                b_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.value_heads,
                layout.normalized,
                layout.b,
            )?,
            a_projection: linear_launch(
                a_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.hidden_size,
                shape.value_heads,
                layout.normalized,
                layout.a,
            )?,
            output_projection: linear_launch(
                output_weight,
                shared.scratch,
                shared.scratch,
                total_tokens,
                shape.value_features,
                shape.hidden_size,
                layout.qkv,
                layout.normalized,
            )?,
        };
        validate_launch_regions(
            &regions,
            &[
                packed.qkv_projection,
                packed.z_projection,
                packed.b_projection,
                packed.a_projection,
                packed.output_projection,
            ],
        )?;
        Some(packed)
    } else {
        for launch in &launches {
            validate_launch_regions(
                &regions,
                &[
                    launch.qkv_projection,
                    launch.z_projection,
                    launch.b_projection,
                    launch.a_projection,
                    launch.output_projection,
                ],
            )?;
        }
        None
    };
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal gated-delta participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let packed_enabled = packed.is_some();
    let dispatch_count = if packed_enabled {
        launches.iter().fold(10_u64, |total, launch| {
            total
                .saturating_add(3)
                .saturating_add(delta_dispatch_count(launch.execution_form))
        })
    } else {
        launches.iter().fold(0_u64, |total, launch| {
            total
                .saturating_add(13)
                .saturating_add(delta_dispatch_count(launch.execution_form))
        })
    };
    let chunked_count = launches
        .iter()
        .filter(|launch| {
            matches!(
                launch.execution_form,
                GatedDeltaExecutionForm::ChunkedScan(_)
            )
        })
        .count();
    let operation_label = if chunked_count == launches.len() {
        "vnext_gated_delta_chunked_attention"
    } else if chunked_count == 0 {
        "vnext_gated_delta_recurrent_attention"
    } else {
        "vnext_gated_delta_mixed_attention"
    };
    MetalDeviceCommand::operation(operation_label, regions, move |encoder, regions| {
        encoder.record_compute_dispatches(dispatch_count);
        if let Some(packed) = packed.as_ref() {
            enqueue_packed_attention(
                &attention,
                &linear,
                &primitives,
                encoder.compute_encoder(),
                regions,
                shared,
                layout,
                packed,
                &launches,
            );
        } else {
            for launch in &launches {
                enqueue_attention(
                    &attention,
                    &linear,
                    &primitives,
                    encoder.compute_encoder(),
                    regions,
                    shared,
                    layout,
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
    .map_err(|error| error.to_string())
}

const fn delta_dispatch_count(form: GatedDeltaExecutionForm) -> u64 {
    match form {
        GatedDeltaExecutionForm::RecurrentScan => 1,
        GatedDeltaExecutionForm::ChunkedScan(_) => 5,
    }
}

#[allow(clippy::too_many_arguments)]
fn enqueue_attention(
    attention: &MetalGatedDeltaPipelines,
    linear: &MetalLinearPipelines,
    primitives: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    layout: ScratchLayout,
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
        launch.params.hidden_size,
        launch.params.epsilon,
    );
    for projection in [
        launch.qkv_projection,
        launch.z_projection,
        launch.b_projection,
        launch.a_projection,
    ] {
        dispatch_linear(linear, encoder, regions, projection);
    }
    dispatch_prepare_conv_and_state(attention, encoder, regions, shared, layout, launch);
    dispatch_prepare_gates(
        attention,
        encoder,
        scratch,
        regions,
        shared,
        launch.a,
        launch.b,
        launch.g,
        launch.beta,
        &launch.params,
    );
    dispatch_qk_norm(attention, encoder, scratch, launch);
    dispatch_delta(
        attention,
        encoder,
        scratch,
        &regions[launch.delta_state],
        launch,
    );
    dispatch_gated_norm(attention, encoder, scratch, &regions[shared.norm], launch);
    dispatch_linear(linear, encoder, regions, launch.output_projection);
    dispatch_residual_add_at(
        primitives,
        encoder,
        &regions[launch.input],
        0,
        scratch,
        launch.normalized,
        &regions[launch.output],
        0,
        launch.residual_elements,
    );
}

#[allow(clippy::too_many_arguments)]
fn enqueue_packed_attention(
    attention: &MetalGatedDeltaPipelines,
    linear: &MetalLinearPipelines,
    primitives: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    layout: ScratchLayout,
    packed: &PackedLaunch,
    participants: &[ParticipantLaunch],
) {
    let scratch = &regions[shared.scratch];
    dispatch_rms_norm_at(
        primitives,
        encoder,
        &regions[packed.input],
        0,
        &regions[shared.input_norm],
        scratch,
        layout.normalized,
        packed.params.tokens,
        packed.params.hidden_size,
        packed.params.epsilon,
    );
    for projection in [
        packed.qkv_projection,
        packed.z_projection,
        packed.b_projection,
        packed.a_projection,
    ] {
        dispatch_linear(linear, encoder, regions, projection);
    }
    dispatch_prepare_gates(
        attention,
        encoder,
        scratch,
        regions,
        shared,
        layout.a,
        layout.b,
        layout.g,
        layout.beta,
        &packed.params,
    );
    for participant in participants {
        dispatch_prepare_conv_and_state(attention, encoder, regions, shared, layout, participant);
    }
    dispatch_qk_norm_at(
        attention,
        encoder,
        scratch,
        layout.query,
        layout.key,
        &packed.params,
    );
    for participant in participants {
        dispatch_delta(
            attention,
            encoder,
            scratch,
            &regions[participant.delta_state],
            participant,
        );
    }
    dispatch_gated_norm_at(
        attention,
        encoder,
        scratch,
        &regions[shared.norm],
        layout.core,
        layout.z,
        layout.qkv,
        &packed.params,
    );
    dispatch_linear(linear, encoder, regions, packed.output_projection);
    dispatch_residual_add_at(
        primitives,
        encoder,
        &regions[packed.input],
        0,
        scratch,
        layout.normalized,
        &regions[packed.output],
        0,
        packed.residual_elements,
    );
}

fn dispatch_prepare_conv_and_state(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    layout: ScratchLayout,
    launch: &ParticipantLaunch,
) {
    let scratch = &regions[shared.scratch];
    encoder.set_compute_pipeline_state(&pipelines.prepare_conv);
    set_region_offset(encoder, 0, scratch, launch.qkv);
    set_region_offset(encoder, 1, &regions[shared.conv], 0);
    set_region_offset(encoder, 2, &regions[launch.conv_state], 0);
    set_region_offset(encoder, 3, scratch, launch.query);
    set_region_offset(encoder, 4, scratch, launch.key);
    set_region_offset(encoder, 5, scratch, launch.value);
    set_params(encoder, 6, &launch.params);
    dispatch_elements(
        encoder,
        u64::from(launch.params.tokens) * u64::from(launch.params.qkv_features),
    );

    encoder.set_compute_pipeline_state(&pipelines.collect_conv_state);
    set_region_offset(encoder, 0, scratch, launch.qkv);
    set_region_offset(encoder, 1, &regions[launch.conv_state], 0);
    set_region_offset(encoder, 2, scratch, layout.conv_state);
    set_params(encoder, 3, &launch.params);
    let state_elements = u64::from(launch.conv_state_elements);
    dispatch_elements(encoder, state_elements);

    encoder.set_compute_pipeline_state(&pipelines.copy_f16);
    set_region_offset(encoder, 0, scratch, layout.conv_state);
    set_region_offset(encoder, 1, &regions[launch.conv_state], 0);
    encoder.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &launch.conv_state_elements as *const _ as *const c_void,
    );
    dispatch_elements(encoder, state_elements);
}

#[allow(clippy::too_many_arguments)]
fn dispatch_prepare_gates(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    regions: &[MetalBufferRegion],
    shared: SharedRegions,
    a: u64,
    b: u64,
    g: u64,
    beta: u64,
    params: &GatedDeltaParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.prepare_gates);
    set_region_offset(encoder, 0, scratch, a);
    set_region_offset(encoder, 1, scratch, b);
    set_region_offset(encoder, 2, &regions[shared.a_log], 0);
    set_region_offset(encoder, 3, &regions[shared.dt_bias], 0);
    set_region_offset(encoder, 4, scratch, g);
    set_region_offset(encoder, 5, scratch, beta);
    set_params(encoder, 6, params);
    dispatch_elements(
        encoder,
        u64::from(params.tokens) * u64::from(params.value_heads),
    );
}

fn dispatch_qk_norm(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    launch: &ParticipantLaunch,
) {
    dispatch_qk_norm_at(
        pipelines,
        encoder,
        scratch,
        launch.query,
        launch.key,
        &launch.params,
    );
}

fn dispatch_qk_norm_at(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    query: u64,
    key: u64,
    params: &GatedDeltaParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.qk_norm);
    set_region_offset(encoder, 0, scratch, query);
    set_region_offset(encoder, 1, scratch, key);
    set_params(encoder, 2, params);
    encoder.set_threadgroup_memory_length(0, 16 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(params.tokens) * u64::from(params.key_heads), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn dispatch_delta(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    state: &MetalBufferRegion,
    launch: &ParticipantLaunch,
) {
    match launch.execution_form {
        GatedDeltaExecutionForm::RecurrentScan => {
            dispatch_recurrent_delta(pipelines, encoder, scratch, state, launch)
        }
        GatedDeltaExecutionForm::ChunkedScan(plan) => {
            debug_assert_eq!(plan.chunk_size(), GATED_DELTA_CHUNK_SIZE);
            dispatch_chunked_delta_c64(pipelines, encoder, scratch, state, launch);
        }
    }
}

fn dispatch_recurrent_delta(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    state: &MetalBufferRegion,
    launch: &ParticipantLaunch,
) {
    encoder.set_compute_pipeline_state(&pipelines.delta);
    for (index, offset) in [
        launch.query,
        launch.key,
        launch.value,
        launch.g,
        launch.beta,
    ]
    .into_iter()
    .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_region_offset(encoder, 5, state, 0);
    set_region_offset(encoder, 6, scratch, launch.core);
    set_params(encoder, 7, &launch.params);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(launch.params.value_dim).div_ceil(VALUE_TILE),
            u64::from(launch.params.value_heads),
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn dispatch_chunked_delta_c64(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    state: &MetalBufferRegion,
    launch: &ParticipantLaunch,
) {
    let params = &launch.params;
    let chunks = u64::from(params.tokens).div_ceil(u64::from(GATED_DELTA_CHUNK_SIZE));

    encoder.set_compute_pipeline_state(&pipelines.chunk_kkt_inverse);
    for (index, offset) in [launch.key, launch.g, launch.beta, launch.core]
        .into_iter()
        .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_params(encoder, 4, params);
    encoder.dispatch_thread_groups(
        MTLSize::new(chunks, u64::from(params.value_heads), 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.chunk_uw);
    for (index, offset) in [
        launch.key,
        launch.value,
        launch.g,
        launch.beta,
        launch.core,
        launch.qkv,
    ]
    .into_iter()
    .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_params(encoder, 6, params);
    dispatch_elements(
        encoder,
        u64::from(params.tokens)
            * u64::from(params.value_heads)
            * (u64::from(params.value_dim) + u64::from(params.key_dim)),
    );

    encoder.set_compute_pipeline_state(&pipelines.chunk_qk);
    for (index, offset) in [launch.query, launch.key, launch.value]
        .into_iter()
        .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_params(encoder, 3, params);
    dispatch_elements(
        encoder,
        u64::from(params.tokens) * u64::from(params.key_heads) * u64::from(GATED_DELTA_CHUNK_SIZE),
    );

    encoder.set_compute_pipeline_state(
        if params.key_dim == GATED_DELTA_CHUNK_KEY_DIM_LIMIT as u32 {
            &pipelines.chunk_carry_k128
        } else {
            &pipelines.chunk_carry_generic
        },
    );
    for (index, offset) in [launch.query, launch.key, launch.g, launch.qkv]
        .into_iter()
        .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_region_offset(encoder, 4, state, 0);
    set_region_offset(encoder, 5, scratch, launch.core);
    set_params(encoder, 6, params);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(params.value_dim).div_ceil(VALUE_TILE),
            u64::from(params.value_heads),
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );

    encoder.set_compute_pipeline_state(&pipelines.chunk_output);
    for (index, offset) in [launch.value, launch.g, launch.qkv, launch.core]
        .into_iter()
        .enumerate()
    {
        set_region_offset(encoder, index as u64, scratch, offset);
    }
    set_params(encoder, 4, params);
    dispatch_elements(
        encoder,
        u64::from(params.tokens) * u64::from(params.value_heads) * u64::from(params.value_dim),
    );
}

fn dispatch_gated_norm(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    weight: &MetalBufferRegion,
    launch: &ParticipantLaunch,
) {
    dispatch_gated_norm_at(
        pipelines,
        encoder,
        scratch,
        weight,
        launch.core,
        launch.z,
        launch.qkv,
        &launch.params,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gated_norm_at(
    pipelines: &MetalGatedDeltaPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    weight: &MetalBufferRegion,
    core: u64,
    z: u64,
    output: u64,
    params: &GatedDeltaParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.gated_norm);
    set_region_offset(encoder, 0, scratch, core);
    set_region_offset(encoder, 1, scratch, z);
    set_region_offset(encoder, 2, weight, 0);
    set_region_offset(encoder, 3, scratch, output);
    set_params(encoder, 4, params);
    encoder.set_threadgroup_memory_length(0, 8 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(params.tokens) * u64::from(params.value_heads),
            1,
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn set_params(encoder: &ComputeCommandEncoderRef, index: u64, params: &GatedDeltaParams) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<GatedDeltaParams>() as u64,
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

fn dispatch_elements(encoder: &ComputeCommandEncoderRef, elements: u64) {
    encoder.dispatch_thread_groups(
        MTLSize::new(elements.div_ceil(THREADS_PER_GROUP), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn push_shared_region(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
    element_type: ElementType,
) -> Result<usize, String> {
    let index = regions.len();
    regions.push(shared_full_region(
        invocation,
        ResolvedValueRole::Input,
        ordinal,
        element_type,
    )?);
    Ok(index)
}

fn validate_signature(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    shape: AttentionShape,
) -> Result<(), String> {
    let value = |ordinal| binding(participant.bindings(), ResolvedValueRole::Input, ordinal);
    let hidden = value(0)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let [tokens, hidden_width] = hidden.tensor().dimensions() else {
        return Err("Metal gated-delta hidden input is not two-dimensional".to_owned());
    };
    let expected = [
        (value(1)?, vec![shape.hidden_size], ElementType::F16),
        (
            value(2)?,
            vec![shape.qkv_features, shape.hidden_size],
            ElementType::F16,
        ),
        (
            value(3)?,
            vec![shape.value_features, shape.hidden_size],
            ElementType::F16,
        ),
        (
            value(4)?,
            vec![shape.value_heads, shape.hidden_size],
            ElementType::F16,
        ),
        (
            value(5)?,
            vec![shape.value_heads, shape.hidden_size],
            ElementType::F16,
        ),
        (
            value(6)?,
            vec![shape.qkv_features, shape.conv_kernel],
            ElementType::F16,
        ),
        (value(7)?, vec![shape.value_heads], ElementType::F32),
        (value(8)?, vec![shape.value_heads], ElementType::F32),
        (value(9)?, vec![shape.value_dim], ElementType::F32),
        (
            value(10)?,
            vec![shape.hidden_size, shape.value_features],
            ElementType::F16,
        ),
        (
            value(11)?,
            vec![shape.qkv_features, shape.conv_state_width],
            CONV_STATE_ELEMENT_TYPE,
        ),
        (
            value(12)?,
            vec![shape.value_heads, shape.value_dim, shape.key_dim],
            DELTA_STATE_ELEMENT_TYPE,
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
        return Err("Metal gated-delta signature differs from its resolved shape".to_owned());
    }
    Ok(())
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
        .checked_add(aligned_bytes(elements, element_type)?)
        .ok_or_else(|| "Metal gated-delta fixed scratch offset overflows".to_owned())?;
    Ok(start)
}

fn reserve_tokens(
    offset: &mut u64,
    elements_per_token: u64,
    element_type: ElementType,
    tokens: u64,
) -> Result<u64, String> {
    let start = *offset;
    *offset = offset
        .checked_add(
            aligned_bytes(elements_per_token, element_type)?
                .checked_mul(tokens)
                .ok_or_else(|| "Metal gated-delta token scratch span overflows".to_owned())?,
        )
        .ok_or_else(|| "Metal gated-delta token scratch offset overflows".to_owned())?;
    Ok(start)
}

fn aligned_bytes(elements: u64, element_type: ElementType) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(element_type.size_bytes())
        .ok_or_else(|| "Metal gated-delta scratch element count overflows".to_owned())?;
    bytes
        .checked_add(VALUE_ALIGNMENT_BYTES - 1)
        .map(|value| value & !(VALUE_ALIGNMENT_BYTES - 1))
        .filter(|value| *value > 0)
        .ok_or_else(|| "Metal gated-delta scratch alignment overflows".to_owned())
}

fn checked_product(values: &[u64]) -> Result<u64, String> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| "Metal gated-delta launch extent overflows".to_owned())
    })
}

#[cfg(test)]
#[path = "gated_delta_attention_tests.rs"]
mod tests;
