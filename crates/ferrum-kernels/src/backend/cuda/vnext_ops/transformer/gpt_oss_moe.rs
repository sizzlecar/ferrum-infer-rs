//! CUDA provider for the typed GPT-OSS routed, clamped MXFP4 MoE contract.
//!
//! This is intentionally separate from the Qwen routed-MoE providers. GPT-OSS
//! uses BF16 router/expert weights and biases, native E2M1/E8M0 group-32
//! expert matrices, selected-logit softmax, an interleaved clamped activation,
//! and a down bias that must be applied before route reduction.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::{
    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CudaFunction, CudaStream,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    gpt_oss_routed_clamped_swiglu_moe_contract, AttributeId, BatchedOperationInvocation,
    CanonicalRational, CapabilityId, ContractVersion, DeviceBatchingForm, DeviceRuntime,
    DynamicStorageRequirement, ElementType, EncodedDeviceOperation, OperationContract,
    OperationFailure, OperationInvocation, OperationProvider, OperationProviderDescriptor,
    OperationResourceEstimate, OperationResourceEstimateRequest, OperationResourceEstimator,
    PhysicalStorageLayout, PhysicalWeightLayout, PhysicalWeightPadding, ProfilePhase, ProviderId,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, QuantizationFormatId, QuantizationGrouping, QuantizationPacking,
    ResolvedTensorLayout, ResolvedValueBinding, ResolvedValueRole, ResolvedWeightComponentLayout,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest, SemanticValue, VNextError,
    WeightComponentRole, WeightEncoding, WeightFormatId, WeightId,
    GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID,
    GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID,
};

use super::super::super::marlin::{
    launch_marlin_moe_mxfp4_bf16, MarlinMoeMxfp4Bf16LaunchArgs, MarlinMoeMxfp4WeightType,
};
use super::super::super::vnext_replay::CudaCommandReplayKeyBuilder;
use super::super::super::vnext_runtime::{
    CudaBufferRegion, CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime,
    CudaDeviceRuntimeError,
};
use super::super::{binding, contract_error, implementation_fingerprint, same_physical_region};
use super::moe::{checked_i32, invalid_plan, MarlinMoeWorkspacePointers, MoeRoutingPlan};
use super::moe_launch::{region_pointer, zero_region, MoeCudaKernels};
use super::moe_workspace::{
    routed_workspace_formula_terms_with_activation_width, MoeWorkspaceLayout, MAX_ROUTER_EXPERTS,
    MAX_ROUTER_TOP_K, MOE_BLOCK_SIZE,
};
use super::{
    contiguous_bindings, ensure_estimator_request, estimate, f16_contiguous, shared_full_region,
    shared_scratch_region, shared_token_region, static_contiguous_reusable_topology,
    CapturedProviderWorkspace,
};
use crate::mxfp4_marlin_materializer::{
    GPT_OSS_MXFP4_MARLIN_QUANTIZATION_FORMAT_ID, GPT_OSS_MXFP4_MARLIN_WEIGHT_FORMAT_ID,
};

const PROVIDER_ID: &str = "provider.cuda.gpt_oss.routed_clamped_swiglu_moe.mxfp4_bf16";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.gpt_oss.routed_clamped_swiglu_moe.mxfp4_bf16";
const COMMAND_NAME: &str = "vnext_gpt_oss_routed_clamped_swiglu_moe";
const VALUE_ALIGNMENT_BYTES: u64 = 16;
const MXFP4_GROUP_SIZE: u64 = 32;
const MARLIN_DOWN_K_ALIGNMENT: u64 = 128;
const THREADS_PER_BLOCK: u32 = 256;
const F16_TO_BF16_FUNCTION: &str = "gpt_oss_f16_to_bf16";
const ROUTER_LOGITS_FUNCTION: &str = "gpt_oss_router_logits_f16_bf16";
const ROUTER_FUNCTION: &str = "gpt_oss_router_topk_selected_softmax_f16";
const SINGLE_TOKEN_ROUTER_FUNCTION: &str =
    "gpt_oss_router_topk_selected_softmax_f16_single_token_marlin";
const ACTIVATION_FUNCTION: &str = "gpt_oss_clamped_swiglu_interleaved_bf16";
const WEIGHTED_SUM_FUNCTION: &str = "gpt_oss_weighted_sum_bf16_to_f16";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GptOssMoeAttributes {
    hidden_size: u64,
    expert_count: u64,
    experts_per_token: u64,
    intermediate_size: u64,
    gate_up_features: u64,
}

impl GptOssMoeAttributes {
    fn from_values(attributes: &BTreeMap<AttributeId, SemanticValue>) -> Result<Self, String> {
        let values = Self {
            hidden_size: unsigned_attribute(attributes, "hidden_size")?,
            expert_count: unsigned_attribute(attributes, "expert_count")?,
            experts_per_token: unsigned_attribute(attributes, "experts_per_token")?,
            intermediate_size: unsigned_attribute(attributes, "intermediate_size")?,
            gate_up_features: unsigned_attribute(attributes, "gate_up_features")?,
        };
        require_bool(attributes, "normalize_topk", true)?;
        require_rational(
            attributes,
            "swiglu_limit",
            CanonicalRational::new(7, 1).map_err(|error| error.to_string())?,
        )?;
        require_bool(attributes, "gate_up_interleaved", true)?;
        require_bool(attributes, "down_bias_before_route_reduction", true)?;
        values.validate()?;
        Ok(values)
    }

    fn validate(self) -> Result<(), String> {
        let expected_gate_up = self
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| "GPT-OSS gate/up feature count overflows u64".to_owned())?;
        if self.hidden_size == 0
            || self.expert_count == 0
            || self.expert_count > MAX_ROUTER_EXPERTS
            || self.experts_per_token == 0
            || self.experts_per_token > self.expert_count
            || self.experts_per_token > MAX_ROUTER_TOP_K
            || self.intermediate_size == 0
            || self.gate_up_features != expected_gate_up
        {
            return Err(format!(
                "GPT-OSS CUDA MoE attributes violate the routed expert contract: {self:?}"
            ));
        }
        if !self.hidden_size.is_multiple_of(64)
            || !self.intermediate_size.is_multiple_of(64)
            || !self.gate_up_features.is_multiple_of(64)
        {
            return Err(format!(
                "GPT-OSS MXFP4 Marlin widths must be divisible by 64: {self:?}"
            ));
        }
        Ok(())
    }

    fn marlin_intermediate_size(self) -> Result<u64, String> {
        align_up_to(self.intermediate_size, MARLIN_DOWN_K_ALIGNMENT)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GptOssMoeLaunchShape {
    tokens: i32,
    expert_count: i32,
    experts_per_token: i32,
    hidden_size: i32,
    intermediate_size: i32,
    marlin_intermediate_size: i32,
    gate_up_features: i32,
    pair_count: i32,
    sorted_capacity: i32,
    device_ordinal: i32,
}

impl GptOssMoeLaunchShape {
    fn from_layout(
        attributes: GptOssMoeAttributes,
        tokens: u64,
        layout: &GptOssMoeWorkspaceLayout,
        device_ordinal: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            tokens: checked_i32(tokens, "GPT-OSS MoE token count")?,
            expert_count: checked_i32(attributes.expert_count, "GPT-OSS MoE expert count")?,
            experts_per_token: checked_i32(
                attributes.experts_per_token,
                "GPT-OSS MoE experts per token",
            )?,
            hidden_size: checked_i32(attributes.hidden_size, "GPT-OSS MoE hidden size")?,
            intermediate_size: checked_i32(
                attributes.intermediate_size,
                "GPT-OSS MoE intermediate size",
            )?,
            marlin_intermediate_size: checked_i32(
                layout.marlin_intermediate_size,
                "GPT-OSS MoE Marlin intermediate size",
            )?,
            gate_up_features: checked_i32(
                attributes.gate_up_features,
                "GPT-OSS MoE gate/up features",
            )?,
            pair_count: checked_i32(layout.base.pair_count, "GPT-OSS MoE pair count")?,
            sorted_capacity: checked_i32(
                layout.base.sorted_capacity,
                "GPT-OSS MoE sorted capacity",
            )?,
            device_ordinal,
        })
    }

    fn activation_elements(self) -> Result<u64, CudaDeviceRuntimeError> {
        u64::try_from(self.pair_count)
            .ok()
            .and_then(|pairs| pairs.checked_mul(self.marlin_intermediate_size as u64))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract(
                    "GPT-OSS MoE activation element count overflows u64",
                )
            })
    }

    fn input_elements(self) -> Result<u64, CudaDeviceRuntimeError> {
        u64::try_from(self.tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(self.hidden_size as u64))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("GPT-OSS MoE input element count overflows u64")
            })
    }
}

#[derive(Clone)]
struct GptOssMoeKernels {
    f16_to_bf16: CudaFunction,
    router_logits: CudaFunction,
    router: CudaFunction,
    single_token_router: CudaFunction,
    activation: CudaFunction,
    weighted_sum: CudaFunction,
}

impl GptOssMoeKernels {
    fn load(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let module = runtime
            .context()
            .load_module(Ptx::from_src(crate::ptx::GPT_OSS_MOE.to_owned()))
            .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS MoE module load", error))?;
        Ok(Self {
            f16_to_bf16: load_function(&module, F16_TO_BF16_FUNCTION)?,
            router_logits: load_function(&module, ROUTER_LOGITS_FUNCTION)?,
            router: load_function(&module, ROUTER_FUNCTION)?,
            single_token_router: load_function(&module, SINGLE_TOKEN_ROUTER_FUNCTION)?,
            activation: load_function(&module, ACTIVATION_FUNCTION)?,
            weighted_sum: load_function(&module, WEIGHTED_SUM_FUNCTION)?,
        })
    }

    fn launch_f16_to_bf16(
        &self,
        stream: &CudaStream,
        input: u64,
        output: u64,
        elements: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let elements_i64 = checked_i64_runtime(elements, "GPT-OSS F16-to-BF16 element count")?;
        let grid = checked_grid(elements, "GPT-OSS F16-to-BF16")?;
        let mut builder = stream.launch_builder(&self.f16_to_bf16);
        builder.arg(&input);
        builder.arg(&output);
        builder.arg(&elements_i64);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS F16-to-BF16 launch", error))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_router_logits(
        &self,
        stream: &CudaStream,
        input: u64,
        weight: u64,
        bias: u64,
        logits: u64,
        shape: GptOssMoeLaunchShape,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let problems = u64::try_from(shape.tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(shape.expert_count as u64))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("GPT-OSS router problem count overflows u64")
            })?;
        let grid = u32::try_from(problems).map_err(|_| {
            CudaDeviceRuntimeError::contract("GPT-OSS router problem grid exceeds u32")
        })?;
        let mut builder = stream.launch_builder(&self.router_logits);
        builder.arg(&input);
        builder.arg(&weight);
        builder.arg(&bias);
        builder.arg(&logits);
        builder.arg(&shape.tokens);
        builder.arg(&shape.expert_count);
        builder.arg(&shape.hidden_size);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS router logits launch", error))
    }

    fn launch_router(
        &self,
        stream: &CudaStream,
        pointers: GptOssMoeWorkspacePointers,
        shape: GptOssMoeLaunchShape,
        plan: MoeRoutingPlan,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let shared_mem_bytes = u32::try_from(shape.expert_count)
            .ok()
            .and_then(|experts| experts.checked_mul(std::mem::size_of::<f32>() as u32))
            .ok_or_else(|| {
                CudaDeviceRuntimeError::contract("GPT-OSS router shared memory overflows")
            })?;
        let function = match plan {
            MoeRoutingPlan::SingleTokenDirectMarlin => &self.single_token_router,
            MoeRoutingPlan::GenericAlign => &self.router,
        };
        let block_size = MOE_BLOCK_SIZE as i32;
        let mut builder = stream.launch_builder(function);
        builder.arg(&pointers.router_logits);
        builder.arg(&pointers.route_ids);
        builder.arg(&pointers.route_weights);
        if matches!(plan, MoeRoutingPlan::SingleTokenDirectMarlin) {
            builder.arg(&pointers.sorted_token_ids);
            builder.arg(&pointers.expert_block_ids);
            builder.arg(&pointers.total_tokens_post_pad);
        }
        builder.arg(&shape.tokens);
        builder.arg(&shape.expert_count);
        builder.arg(&shape.experts_per_token);
        if matches!(plan, MoeRoutingPlan::SingleTokenDirectMarlin) {
            builder.arg(&block_size);
        }
        let grid = u32::try_from(shape.tokens)
            .map_err(|_| CudaDeviceRuntimeError::contract("GPT-OSS router grid exceeds u32"))?;
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS selected router launch", error))
    }

    fn launch_activation(
        &self,
        stream: &CudaStream,
        gate_up: u64,
        output: u64,
        shape: GptOssMoeLaunchShape,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let elements = shape.activation_elements()?;
        let elements_i64 = checked_i64_runtime(elements, "GPT-OSS activation elements")?;
        let grid = checked_grid(elements, "GPT-OSS activation")?;
        let limit = 7.0_f32;
        let mut builder = stream.launch_builder(&self.activation);
        builder.arg(&gate_up);
        builder.arg(&output);
        builder.arg(&shape.intermediate_size);
        builder.arg(&shape.marlin_intermediate_size);
        builder.arg(&elements_i64);
        builder.arg(&limit);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS activation launch", error))
    }

    fn launch_weighted_sum(
        &self,
        stream: &CudaStream,
        slots: u64,
        route_weights: u64,
        output: u64,
        shape: GptOssMoeLaunchShape,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let elements = shape.input_elements()?;
        let elements_i64 = checked_i64_runtime(elements, "GPT-OSS route reduction elements")?;
        let grid = checked_grid(elements, "GPT-OSS route reduction")?;
        let mut builder = stream.launch_builder(&self.weighted_sum);
        builder.arg(&slots);
        builder.arg(&route_weights);
        builder.arg(&output);
        builder.arg(&shape.tokens);
        builder.arg(&shape.experts_per_token);
        builder.arg(&shape.hidden_size);
        builder.arg(&elements_i64);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS route reduction launch", error))
    }
}

fn load_function(
    module: &Arc<cudarc::driver::CudaModule>,
    name: &'static str,
) -> Result<CudaFunction, CudaDeviceRuntimeError> {
    module
        .load_function(name)
        .map_err(|error| CudaDeviceRuntimeError::driver("GPT-OSS MoE function load", error))
}

pub(in crate::backend::cuda::vnext_ops) struct CudaGptOssRoutedClampedSwiGluMoeProvider {
    descriptor: OperationProviderDescriptor,
    kernels: GptOssMoeKernels,
    routing_kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
}

impl CudaGptOssRoutedClampedSwiGluMoeProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = gpt_oss_routed_clamped_swiglu_moe_contract().map_err(contract_error)?;
        let capability =
            CapabilityId::new(GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID)
                .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "CUDA runtime does not advertise `{GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID}`"
            )));
        }
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("gpt_oss_moe.rs").as_bytes(),
            include_str!("moe_launch.rs").as_bytes(),
            include_str!("moe_workspace.rs").as_bytes(),
            include_str!("../../marlin.rs").as_bytes(),
            include_str!("../../../../../kernels/gpt_oss_moe.cu").as_bytes(),
            crate::native_ops::CUDA_NATIVE_SOURCE_BUNDLE_ID.as_bytes(),
            crate::ptx::GPT_OSS_MOE.as_bytes(),
            crate::ptx::MOE_ALIGN_BLOCK_SIZE_PAIR_IDS.as_bytes(),
        ]);
        let estimator_fingerprint = implementation_fingerprint(&[
            include_str!("gpt_oss_moe.rs").as_bytes(),
            include_str!("moe_workspace.rs").as_bytes(),
            ESTIMATOR_ID.as_bytes(),
            provider_fingerprint.as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(contract_error)?,
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
            BTreeSet::from([WeightFormatId::new(GPT_OSS_MXFP4_MARLIN_WEIGHT_FORMAT_ID)
                .map_err(contract_error)?]),
            BTreeSet::from([QuantizationFormatId::new(
                GPT_OSS_MXFP4_MARLIN_QUANTIZATION_FORMAT_ID,
            )
            .map_err(contract_error)?]),
            contiguous_bindings(7),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(contract_error)?;
        let multiprocessor_count = runtime
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| CudaDeviceRuntimeError::driver("multiprocessor count query", error))?;
        let multiprocessor_count = u64::try_from(multiprocessor_count).map_err(|_| {
            CudaDeviceRuntimeError::contract("CUDA multiprocessor count is not positive")
        })?;
        if multiprocessor_count == 0 {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA multiprocessor count is zero",
            ));
        }
        let device_ordinal = i32::try_from(runtime.descriptor().ordinal)
            .map_err(|_| CudaDeviceRuntimeError::contract("CUDA device ordinal exceeds i32"))?;
        Ok(Self {
            descriptor,
            kernels: GptOssMoeKernels::load(runtime)?,
            routing_kernels: MoeCudaKernels::load(runtime)?,
            multiprocessor_count,
            device_ordinal,
        })
    }
}

impl OperationResourceEstimator for CudaGptOssRoutedClampedSwiGluMoeProvider {
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
            GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID,
        )?;
        let attributes =
            GptOssMoeAttributes::from_values(request.attributes()).map_err(invalid_plan)?;
        let marlin_intermediate_size = attributes
            .marlin_intermediate_size()
            .map_err(invalid_plan)?;
        let (fixed_bytes, bytes_per_token) = gpt_oss_workspace_formula_terms(
            attributes.expert_count,
            attributes.experts_per_token,
            attributes.hidden_size,
            attributes.intermediate_size,
            marlin_intermediate_size,
            self.multiprocessor_count,
        )
        .map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(fixed_bytes, 0, bytes_per_token)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaGptOssRoutedClampedSwiGluMoeProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        static_contiguous_reusable_topology(&request, 7, &[CapturedProviderWorkspace::Scratch])
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_gpt_oss_moe(
            self.descriptor.provider_implementation_fingerprint(),
            self.kernels.clone(),
            self.routing_kernels.clone(),
            self.multiprocessor_count,
            self.device_ordinal,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            OperationFailure::new(
                identity,
                ProfilePhase::Forward,
                "cuda.gpt_oss.routed_clamped_swiglu_moe.encode",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued GPT-OSS MoE identity must form a valid provider failure")
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_gpt_oss_moe(
    provider_fingerprint: &str,
    kernels: GptOssMoeKernels,
    routing_kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_OPERATION_ID
    {
        return Err("GPT-OSS CUDA MoE provider received another or empty operation".to_owned());
    }
    let first = &invocation.participants()[0];
    let attributes = GptOssMoeAttributes::from_values(first.attributes())?;
    let tokens = invocation.work_shape().immediate_tokens();
    if tokens == 0 {
        return Err("GPT-OSS CUDA MoE invocation has no immediate tokens".to_owned());
    }
    for participant in invocation.participants() {
        if GptOssMoeAttributes::from_values(participant.attributes())? != attributes {
            return Err("GPT-OSS CUDA MoE participant attributes disagree".to_owned());
        }
        validate_participant(participant.bindings(), attributes)?;
    }

    let gate_up_dimensions = vec![
        attributes.expert_count,
        attributes.gate_up_features,
        attributes.hidden_size,
    ];
    let down_dimensions = vec![
        attributes.expert_count,
        attributes.hidden_size,
        attributes.intermediate_size,
    ];
    let marlin_intermediate_size = attributes.marlin_intermediate_size()?;
    let down_execution_dimensions = vec![
        attributes.expert_count,
        attributes.hidden_size,
        marlin_intermediate_size,
    ];
    let gate_up =
        resolve_shared_mxfp4_weight(&invocation, 3, &gate_up_dimensions, &gate_up_dimensions)?;
    let down =
        resolve_shared_mxfp4_weight(&invocation, 5, &down_dimensions, &down_execution_dimensions)?;
    if gate_up.expert_count != attributes.expert_count
        || down.expert_count != attributes.expert_count
    {
        return Err("GPT-OSS physical expert count differs from operation attributes".to_owned());
    }

    let layout = GptOssMoeWorkspaceLayout::new(
        tokens,
        attributes.expert_count,
        attributes.experts_per_token,
        attributes.hidden_size,
        attributes.intermediate_size,
        marlin_intermediate_size,
        multiprocessor_count,
    )?;
    let shape = GptOssMoeLaunchShape::from_layout(attributes, tokens, &layout, device_ordinal)?;
    validate_launch_problem(
        shape,
        &gate_up.logical_dimensions,
        &gate_up.execution_dimensions,
        gate_up.group_size,
        &down.logical_dimensions,
        &down.execution_dimensions,
        down.group_size,
    )?;
    let routing_plan = MoeRoutingPlan::for_tokens(shape.tokens);
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::Bf16)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 2, ElementType::Bf16)?,
        gate_up.packed_region,
        gate_up.scales_region,
        shared_full_region(&invocation, ResolvedValueRole::Input, 4, ElementType::Bf16)?,
        down.packed_region,
        down.scales_region,
        shared_full_region(&invocation, ResolvedValueRole::Input, 6, ElementType::Bf16)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_scratch_region(&invocation, layout.total_bytes)?,
    ];
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, COMMAND_NAME)
        .i32(shape.tokens)
        .i32(shape.expert_count)
        .i32(shape.experts_per_token)
        .i32(shape.hidden_size)
        .i32(shape.intermediate_size)
        .i32(shape.marlin_intermediate_size)
        .i32(shape.gate_up_features)
        .i32(gate_up.group_size)
        .i32(down.group_size)
        .i32(shape.device_ordinal)
        .bytes(routing_plan.replay_tag())
        .u64(layout.total_bytes)
        .u64(MOE_BLOCK_SIZE)
        .finish();
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "GPT-OSS CUDA MoE participant count exceeds u32".to_owned())?;

    CudaDeviceCommand::replayable_operation(
        COMMAND_NAME,
        regions,
        replay_key,
        move |stream, regions| {
            let scratch = &regions[10];
            if scratch.length_bytes() < layout.total_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "GPT-OSS MoE scratch is smaller than its admitted estimate",
                ));
            }
            let pointers = GptOssMoeWorkspacePointers::new(scratch.device_ptr(), &layout)?;
            kernels.launch_f16_to_bf16(
                stream,
                regions[0].device_ptr(),
                pointers.input_bf16,
                shape.input_elements()?,
            )?;
            kernels.launch_router_logits(
                stream,
                regions[0].device_ptr(),
                regions[1].device_ptr(),
                regions[2].device_ptr(),
                pointers.router_logits,
                shape,
            )?;
            kernels.launch_router(stream, pointers, shape, routing_plan)?;
            if matches!(routing_plan, MoeRoutingPlan::GenericAlign) {
                routing_kernels.launch_align(
                    stream,
                    pointers.route_ids,
                    pointers.sorted_token_ids,
                    pointers.expert_block_ids,
                    pointers.total_tokens_post_pad,
                    shape.pair_count,
                    shape.expert_count,
                    shape.sorted_capacity,
                )?;
            }

            zero_region(stream, scratch.device_ptr(), layout.base.marlin_workspace)?;
            launch_mxfp4_marlin(
                stream,
                pointers.input_bf16,
                regions[3].device_ptr(),
                regions[4].device_ptr(),
                regions[5].device_ptr(),
                pointers.routed_gate_up,
                pointers.marlin(),
                shape.expert_count,
                shape.tokens,
                shape.experts_per_token,
                shape.gate_up_features,
                shape.hidden_size,
                gate_up.group_size,
                shape.device_ordinal,
            )?;
            kernels.launch_activation(
                stream,
                pointers.routed_gate_up,
                pointers.routed_activation,
                shape,
            )?;
            zero_region(stream, scratch.device_ptr(), layout.base.marlin_workspace)?;
            launch_mxfp4_marlin(
                stream,
                pointers.routed_activation,
                regions[6].device_ptr(),
                regions[7].device_ptr(),
                regions[8].device_ptr(),
                pointers.routed_down_slots,
                pointers.marlin(),
                shape.expert_count,
                shape.pair_count,
                1,
                shape.hidden_size,
                shape.marlin_intermediate_size,
                down.group_size,
                shape.device_ordinal,
            )?;
            kernels.launch_weighted_sum(
                stream,
                pointers.routed_down_slots,
                pointers.route_weights,
                regions[9].device_ptr(),
                shape,
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::Packed,
            participant_count,
            tokens,
            gpt_oss_dispatch_count(routing_plan),
            2,
        )
    })
    .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
fn launch_mxfp4_marlin(
    stream: &CudaStream,
    input: u64,
    packed_weight: u64,
    scales: u64,
    bias: u64,
    output: u64,
    workspace: MarlinMoeWorkspacePointers,
    expert_count: i32,
    prob_m: i32,
    top_k: i32,
    prob_n: i32,
    prob_k: i32,
    group_size: i32,
    device_ordinal: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    launch_marlin_moe_mxfp4_bf16(
        stream,
        MarlinMoeMxfp4Bf16LaunchArgs {
            weight_type: MarlinMoeMxfp4WeightType::E2M1E8M0,
            expert_count,
            a: input,
            b: packed_weight,
            c: output,
            c_tmp: Some(workspace.marlin_c_tmp),
            bias,
            scales,
            workspace: workspace.marlin_workspace,
            sorted_token_ids: workspace.sorted_token_ids,
            expert_ids: workspace.expert_block_ids,
            num_tokens_past_padded: workspace.total_tokens_post_pad,
            topk_weights: None,
            moe_block_size: MOE_BLOCK_SIZE as i32,
            top_k,
            mul_topk_weights: false,
            is_ep: false,
            prob_m,
            prob_n,
            prob_k,
            group_size,
            device_ordinal,
            use_atomic_add: false,
            use_fp32_reduce: true,
        },
    )
    .map_err(|error| {
        CudaDeviceRuntimeError::contract(format!(
            "GPT-OSS MXFP4 Marlin-MoE launch rejected: {error}"
        ))
    })
}

fn validate_participant(
    bindings: &[ResolvedValueBinding],
    attributes: GptOssMoeAttributes,
) -> Result<(), String> {
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let [canonical_tokens, input_hidden] = input.tensor().dimensions() else {
        return Err("GPT-OSS CUDA MoE input is not two-dimensional".to_owned());
    };
    if *input_hidden != attributes.hidden_size || !f16_contiguous(input) {
        return Err("GPT-OSS CUDA MoE input must be contiguous [tokens,H] F16".to_owned());
    }
    let expected = [
        (1, vec![attributes.expert_count, attributes.hidden_size]),
        (2, vec![attributes.expert_count]),
        (
            3,
            vec![
                attributes.expert_count,
                attributes.gate_up_features,
                attributes.hidden_size,
            ],
        ),
        (
            4,
            vec![attributes.expert_count, attributes.gate_up_features],
        ),
        (
            5,
            vec![
                attributes.expert_count,
                attributes.hidden_size,
                attributes.intermediate_size,
            ],
        ),
        (6, vec![attributes.expert_count, attributes.hidden_size]),
    ];
    for (ordinal, dimensions) in expected {
        let value = binding(bindings, ResolvedValueRole::Input, ordinal)?;
        if value.tensor().dimensions() != dimensions || !bf16_contiguous(value) {
            return Err(format!(
                "GPT-OSS CUDA MoE input {ordinal} must be contiguous {dimensions:?} BF16"
            ));
        }
    }
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    if output.tensor().dimensions() != [*canonical_tokens, attributes.hidden_size]
        || !f16_contiguous(output)
    {
        return Err("GPT-OSS CUDA MoE output must be contiguous [tokens,H] F16".to_owned());
    }
    Ok(())
}

fn bf16_contiguous(binding: &ResolvedValueBinding) -> bool {
    binding.tensor().element_type() == ElementType::Bf16
        && matches!(binding.tensor().layout(), ResolvedTensorLayout::Contiguous)
}

struct CudaMxfp4MoeWeight {
    packed_region: CudaBufferRegion,
    scales_region: CudaBufferRegion,
    logical_dimensions: Vec<u64>,
    execution_dimensions: Vec<u64>,
    semantic_packed_dimensions: Vec<u64>,
    physical_packed_dimensions: Vec<u64>,
    physical_scales_dimensions: Vec<u64>,
    expert_count: u64,
    group_size: i32,
}

fn resolve_shared_mxfp4_weight(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
    execution_dimensions: &[u64],
) -> Result<CudaMxfp4MoeWeight, String> {
    let resolve = |participant: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let value = binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?;
        resolve_mxfp4_weight(participant, value, logical_dimensions, execution_dimensions)
    };
    let resolved = resolve(&invocation.participants()[0])?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve(participant)?;
        if resolved.logical_dimensions != candidate.logical_dimensions
            || resolved.execution_dimensions != candidate.execution_dimensions
            || resolved.semantic_packed_dimensions != candidate.semantic_packed_dimensions
            || resolved.physical_packed_dimensions != candidate.physical_packed_dimensions
            || resolved.physical_scales_dimensions != candidate.physical_scales_dimensions
            || resolved.expert_count != candidate.expert_count
            || resolved.group_size != candidate.group_size
            || !same_physical_region(&resolved.packed_region, &candidate.packed_region)
            || !same_physical_region(&resolved.scales_region, &candidate.scales_region)
        {
            return Err(format!(
                "GPT-OSS CUDA MoE input {ordinal} is not one shared physical MXFP4 stack"
            ));
        }
    }
    Ok(resolved)
}

fn resolve_mxfp4_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
    execution_dimensions: &[u64],
) -> Result<CudaMxfp4MoeWeight, String> {
    let [expert_count, output_features, input_features] = logical_dimensions else {
        return Err("GPT-OSS MXFP4 expert weight must have shape [E,N,K]".to_owned());
    };
    let [execution_experts, execution_output_features, execution_input_features] =
        execution_dimensions
    else {
        return Err("GPT-OSS MXFP4 execution weight must have shape [E,N,K]".to_owned());
    };
    if *expert_count == 0
        || *output_features == 0
        || *input_features == 0
        || !output_features.is_multiple_of(64)
        || !input_features.is_multiple_of(64)
        || execution_experts != expert_count
        || execution_output_features != output_features
        || execution_input_features < input_features
        || !execution_input_features.is_multiple_of(64)
    {
        return Err(format!(
            "GPT-OSS MXFP4 logical {logical_dimensions:?} / execution {execution_dimensions:?} shape violates Marlin tiles"
        ));
    }
    if binding.tensor().dimensions() != logical_dimensions
        || binding.tensor().element_type() != ElementType::Bf16
    {
        return Err("GPT-OSS MXFP4 logical binding differs from [E,N,K] BF16".to_owned());
    }
    let weight = binding
        .weight()
        .ok_or_else(|| "GPT-OSS MXFP4 weight lacks its typed physical layout".to_owned())?;
    weight
        .validate_logical(logical_dimensions, ElementType::Bf16)
        .map_err(|error| format!("GPT-OSS MXFP4 logical contract is invalid: {error}"))?;
    let PhysicalWeightLayout::Quantized {
        packed_values,
        packed_dimensions,
        scales,
        zero_points,
        zero_point_packed_dimensions,
        axis_indices,
        permutation,
        codebook,
        group_axis,
        group_padding,
    } = weight.physical_layout()
    else {
        return Err("GPT-OSS MXFP4 requires one quantized physical layout".to_owned());
    };
    if zero_points.is_some()
        || zero_point_packed_dimensions.is_some()
        || axis_indices.is_some()
        || permutation.is_some()
        || codebook.is_some()
        || *group_axis != 2
        || !matches!(group_padding, PhysicalWeightPadding::Exact)
        || packed_values.component_id == scales.component_id
    {
        return Err(
            "GPT-OSS MXFP4 physical layout must be contiguous group-32 with typed execution padding and no auxiliary components"
                .to_owned(),
        );
    }
    let mut components = BTreeMap::new();
    for component in weight.components() {
        if components
            .insert(component.component_id().clone(), component)
            .is_some()
        {
            return Err(format!(
                "GPT-OSS MXFP4 layout duplicates component `{}`",
                component.component_id()
            ));
        }
    }
    if components.len() != 2 {
        return Err("GPT-OSS MXFP4 layout must contain only packed values and scales".to_owned());
    }
    let packed_component = required_component(
        &components,
        &packed_values.component_id,
        WeightComponentRole::PackedValues,
    )?;
    let scales_component = required_component(
        &components,
        &scales.component_id,
        WeightComponentRole::Scales,
    )?;
    let WeightEncoding::Quantized(quantization) = packed_component.encoding() else {
        return Err("GPT-OSS MXFP4 packed component is not quantized".to_owned());
    };
    quantization
        .validate()
        .map_err(|error| format!("GPT-OSS MXFP4 quantization ABI is invalid: {error}"))?;
    if quantization.format_id.as_str() != GPT_OSS_MXFP4_MARLIN_QUANTIZATION_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.grouping != QuantizationGrouping::fixed(MXFP4_GROUP_SIZE as u32)
        || quantization.packing != QuantizationPacking::Tiled
        || quantization.scale_type != ElementType::U8
        || quantization.zero_point_type.is_some()
        || !matches!(
            scales_component.encoding(),
            WeightEncoding::Dense {
                element_type: ElementType::U8
            }
        )
    {
        return Err("GPT-OSS expert components are not tiled E2M1/E8M0 group-32 MXFP4".to_owned());
    }
    let expected_semantic_packed = vec![*expert_count, *output_features, *input_features / 2];
    let expected_physical_packed = vec![
        *expert_count,
        *output_features,
        *execution_input_features / 2,
    ];
    let expected_semantic_scales = vec![
        *expert_count,
        *output_features,
        *input_features / MXFP4_GROUP_SIZE,
    ];
    let expected_physical_scales = vec![
        *expert_count,
        *output_features,
        *execution_input_features / MXFP4_GROUP_SIZE,
    ];
    if packed_dimensions != &expected_semantic_packed
        || packed_component.physical_dimensions() != expected_physical_packed
        || scales_component.physical_dimensions() != expected_physical_scales
        || !matches_execution_storage(
            &packed_values.storage,
            &expected_semantic_packed,
            &expected_physical_packed,
        )
        || !matches_execution_storage(
            &scales.storage,
            &expected_semantic_scales,
            &expected_physical_scales,
        )
    {
        return Err(format!(
            "GPT-OSS MXFP4 physical shapes must preserve semantic packed={expected_semantic_packed:?}, scales={expected_semantic_scales:?} and use execution packed={expected_physical_packed:?}, scales={expected_physical_scales:?}"
        ));
    }
    let packed_bytes = checked_product(&expected_physical_packed, "GPT-OSS packed bytes")?;
    let scales_bytes = checked_product(&expected_physical_scales, "GPT-OSS scale bytes")?;
    if packed_component
        .physical_bytes()
        .map_err(|error| error.to_string())?
        != packed_bytes
        || scales_component
            .physical_bytes()
            .map_err(|error| error.to_string())?
            != scales_bytes
    {
        return Err("GPT-OSS MXFP4 physical byte count differs from its shape".to_owned());
    }
    let packed_region = retain_component_region(
        participant,
        &packed_values.component_id,
        binding,
        ElementType::U8,
        packed_bytes,
        packed_bytes / expert_count,
    )?;
    let scales_region = retain_component_region(
        participant,
        &scales.component_id,
        binding,
        ElementType::U8,
        scales_bytes,
        scales_bytes / expert_count,
    )?;
    Ok(CudaMxfp4MoeWeight {
        packed_region,
        scales_region,
        logical_dimensions: logical_dimensions.to_vec(),
        execution_dimensions: execution_dimensions.to_vec(),
        semantic_packed_dimensions: expected_semantic_packed,
        physical_packed_dimensions: expected_physical_packed,
        physical_scales_dimensions: expected_physical_scales,
        expert_count: *expert_count,
        group_size: MXFP4_GROUP_SIZE as i32,
    })
}

fn required_component<'a>(
    components: &'a BTreeMap<WeightId, &'a ResolvedWeightComponentLayout>,
    id: &WeightId,
    role: WeightComponentRole,
) -> Result<&'a ResolvedWeightComponentLayout, String> {
    let component = components
        .get(id)
        .copied()
        .ok_or_else(|| format!("GPT-OSS MXFP4 component `{id}` is absent"))?;
    if component.role() != role {
        return Err(format!(
            "GPT-OSS MXFP4 component `{id}` has role {:?}, expected {role:?}",
            component.role()
        ));
    }
    Ok(component)
}

fn retain_component_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    component_id: &WeightId,
    binding: &ResolvedValueBinding,
    element_type: ElementType,
    length_bytes: u64,
    expert_stride_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let stored = binding
        .storage()
        .components()
        .iter()
        .find(|stored| stored.component_id() == Some(component_id))
        .ok_or_else(|| format!("GPT-OSS MXFP4 component `{component_id}` has no storage"))?;
    if stored.element_type() != element_type || stored.length_bytes() != length_bytes {
        return Err(format!(
            "GPT-OSS MXFP4 component `{component_id}` storage differs from its ABI"
        ));
    }
    let mut views = participant
        .views()
        .iter()
        .filter(|view| view.resource_id() == stored.resource_id());
    let view = views
        .next()
        .ok_or_else(|| format!("GPT-OSS MXFP4 component `{component_id}` has no committed view"))?;
    if views.next().is_some() {
        return Err(format!(
            "GPT-OSS MXFP4 component `{component_id}` has ambiguous committed views"
        ));
    }
    let translated = view
        .translate(stored.offset_bytes(), stored.length_bytes())
        .map_err(|error| error.to_string())?;
    let mut physical_regions = translated.iter();
    let physical = physical_regions.next().ok_or_else(|| {
        format!("GPT-OSS MXFP4 component `{component_id}` translated to no region")
    })?;
    if physical_regions.next().is_some() {
        return Err(format!(
            "GPT-OSS MXFP4 component `{component_id}` is physically fragmented"
        ));
    }
    let (buffer, range, retention) = physical.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if region.element_type() != element_type
        || region.length_bytes() != length_bytes
        || region.device_ptr() == 0
        || !region.device_ptr().is_multiple_of(VALUE_ALIGNMENT_BYTES)
        || !length_bytes.is_multiple_of(VALUE_ALIGNMENT_BYTES)
        || !expert_stride_bytes.is_multiple_of(VALUE_ALIGNMENT_BYTES)
    {
        return Err(format!(
            "GPT-OSS MXFP4 component `{component_id}` violates aligned contiguous geometry"
        ));
    }
    Ok(region)
}

fn matches_execution_storage(
    storage: &PhysicalStorageLayout,
    semantic_dimensions: &[u64],
    physical_dimensions: &[u64],
) -> bool {
    match storage {
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact,
        } => semantic_dimensions == physical_dimensions,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::ZeroFill { padded_dimensions },
        } => semantic_dimensions != physical_dimensions && padded_dimensions == physical_dimensions,
        PhysicalStorageLayout::Strided { .. } | PhysicalStorageLayout::Tiled { .. } => false,
    }
}

fn checked_product(dimensions: &[u64], label: &str) -> Result<u64, String> {
    dimensions
        .iter()
        .try_fold(1_u64, |product, extent| product.checked_mul(*extent))
        .ok_or_else(|| format!("{label} overflows u64"))
}

#[derive(Debug, Clone)]
struct GptOssMoeWorkspaceLayout {
    base: MoeWorkspaceLayout,
    marlin_intermediate_size: u64,
    input_bf16_offset: u64,
    input_bf16_bytes: u64,
    total_bytes: u64,
}

impl GptOssMoeWorkspaceLayout {
    #[allow(clippy::too_many_arguments)]
    fn new(
        tokens: u64,
        expert_count: u64,
        experts_per_token: u64,
        hidden_size: u64,
        intermediate_size: u64,
        marlin_intermediate_size: u64,
        multiprocessor_count: u64,
    ) -> Result<Self, String> {
        let expected_marlin_intermediate_size =
            align_up_to(intermediate_size, MARLIN_DOWN_K_ALIGNMENT)?;
        if marlin_intermediate_size != expected_marlin_intermediate_size {
            return Err(format!(
                "GPT-OSS Marlin intermediate width {marlin_intermediate_size} is not canonical padding of {intermediate_size} to {MARLIN_DOWN_K_ALIGNMENT}"
            ));
        }
        let base = MoeWorkspaceLayout::routed_only_with_activation_width(
            tokens,
            expert_count,
            experts_per_token,
            hidden_size,
            intermediate_size,
            marlin_intermediate_size,
            multiprocessor_count,
        )?;
        let input_bf16_bytes = tokens
            .checked_mul(hidden_size)
            .and_then(|elements| elements.checked_mul(ElementType::Bf16.size_bytes()))
            .ok_or_else(|| "GPT-OSS BF16 input workspace overflows u64".to_owned())?;
        let input_bf16_offset = base.total_bytes;
        let total_bytes = input_bf16_offset
            .checked_add(input_bf16_bytes)
            .ok_or_else(|| "GPT-OSS total workspace overflows u64".to_owned())?;
        if !input_bf16_offset.is_multiple_of(VALUE_ALIGNMENT_BYTES)
            || !input_bf16_bytes.is_multiple_of(VALUE_ALIGNMENT_BYTES)
            || !total_bytes.is_multiple_of(VALUE_ALIGNMENT_BYTES)
        {
            return Err("GPT-OSS BF16 input workspace is not 16-byte aligned".to_owned());
        }
        let (fixed, per_token) = gpt_oss_workspace_formula_terms(
            expert_count,
            experts_per_token,
            hidden_size,
            intermediate_size,
            marlin_intermediate_size,
            multiprocessor_count,
        )?;
        let admitted = fixed
            .checked_add(
                per_token
                    .checked_mul(tokens)
                    .ok_or_else(|| "GPT-OSS admitted workspace overflows u64".to_owned())?,
            )
            .ok_or_else(|| "GPT-OSS admitted workspace overflows u64".to_owned())?;
        if total_bytes > admitted {
            return Err(format!(
                "GPT-OSS workspace {total_bytes} exceeds affine estimate {admitted}"
            ));
        }
        Ok(Self {
            base,
            marlin_intermediate_size,
            input_bf16_offset,
            input_bf16_bytes,
            total_bytes,
        })
    }
}

fn gpt_oss_workspace_formula_terms(
    expert_count: u64,
    experts_per_token: u64,
    hidden_size: u64,
    intermediate_size: u64,
    marlin_intermediate_size: u64,
    multiprocessor_count: u64,
) -> Result<(u64, u64), String> {
    let expected_marlin_intermediate_size =
        align_up_to(intermediate_size, MARLIN_DOWN_K_ALIGNMENT)?;
    if marlin_intermediate_size != expected_marlin_intermediate_size {
        return Err(format!(
            "GPT-OSS workspace estimator received non-canonical Marlin width {marlin_intermediate_size} for logical width {intermediate_size}"
        ));
    }
    let (fixed, per_token) = routed_workspace_formula_terms_with_activation_width(
        expert_count,
        experts_per_token,
        hidden_size,
        intermediate_size,
        marlin_intermediate_size,
        multiprocessor_count,
    )?;
    let input_bf16_per_token = hidden_size
        .checked_mul(ElementType::Bf16.size_bytes())
        .ok_or_else(|| "GPT-OSS BF16 input bytes per token overflow u64".to_owned())?;
    Ok((
        fixed,
        per_token
            .checked_add(input_bf16_per_token)
            .ok_or_else(|| "GPT-OSS workspace bytes per token overflow u64".to_owned())?,
    ))
}

#[derive(Clone, Copy)]
struct GptOssMoeWorkspacePointers {
    router_logits: u64,
    route_ids: u64,
    route_weights: u64,
    sorted_token_ids: u64,
    expert_block_ids: u64,
    total_tokens_post_pad: u64,
    marlin_workspace: u64,
    marlin_c_tmp: u64,
    routed_gate_up: u64,
    routed_activation: u64,
    routed_down_slots: u64,
    input_bf16: u64,
}

impl GptOssMoeWorkspacePointers {
    fn new(base: u64, layout: &GptOssMoeWorkspaceLayout) -> Result<Self, CudaDeviceRuntimeError> {
        if layout.base.shared().is_some() {
            return Err(CudaDeviceRuntimeError::contract(
                "GPT-OSS routed MoE received a shared-expert workspace",
            ));
        }
        let pointer = |region| region_pointer(base, region);
        let input_bf16 = base
            .checked_add(layout.input_bf16_offset)
            .ok_or_else(|| CudaDeviceRuntimeError::contract("GPT-OSS input pointer overflows"))?;
        input_bf16
            .checked_add(layout.input_bf16_bytes.saturating_sub(1))
            .ok_or_else(|| CudaDeviceRuntimeError::contract("GPT-OSS input span overflows"))?;
        Ok(Self {
            router_logits: pointer(layout.base.router_logits)?,
            route_ids: pointer(layout.base.route_ids)?,
            route_weights: pointer(layout.base.route_weights)?,
            sorted_token_ids: pointer(layout.base.sorted_token_ids)?,
            expert_block_ids: pointer(layout.base.expert_block_ids)?,
            total_tokens_post_pad: pointer(layout.base.total_tokens_post_pad)?,
            marlin_workspace: pointer(layout.base.marlin_workspace)?,
            marlin_c_tmp: pointer(layout.base.marlin_c_tmp)?,
            routed_gate_up: pointer(layout.base.routed_gate_up)?,
            routed_activation: pointer(layout.base.routed_activation)?,
            routed_down_slots: pointer(layout.base.routed_down_slots)?,
            input_bf16,
        })
    }

    const fn marlin(self) -> MarlinMoeWorkspacePointers {
        MarlinMoeWorkspacePointers {
            sorted_token_ids: self.sorted_token_ids,
            expert_block_ids: self.expert_block_ids,
            total_tokens_post_pad: self.total_tokens_post_pad,
            marlin_workspace: self.marlin_workspace,
            marlin_c_tmp: self.marlin_c_tmp,
        }
    }
}

fn validate_launch_problem(
    shape: GptOssMoeLaunchShape,
    gate_up_dimensions: &[u64],
    gate_up_execution_dimensions: &[u64],
    gate_up_group_size: i32,
    down_dimensions: &[u64],
    down_execution_dimensions: &[u64],
    down_group_size: i32,
) -> Result<(), String> {
    if gate_up_group_size != MXFP4_GROUP_SIZE as i32
        || down_group_size != MXFP4_GROUP_SIZE as i32
        || gate_up_dimensions
            != [
                shape.expert_count as u64,
                shape.gate_up_features as u64,
                shape.hidden_size as u64,
            ]
        || gate_up_execution_dimensions != gate_up_dimensions
        || down_dimensions
            != [
                shape.expert_count as u64,
                shape.hidden_size as u64,
                shape.intermediate_size as u64,
            ]
        || down_execution_dimensions
            != [
                shape.expert_count as u64,
                shape.hidden_size as u64,
                shape.marlin_intermediate_size as u64,
            ]
        || shape.tokens <= 0
        || shape.pair_count
            != shape
                .tokens
                .checked_mul(shape.experts_per_token)
                .ok_or_else(|| "GPT-OSS pair count overflows i32".to_owned())?
    {
        return Err("GPT-OSS MXFP4 launch shape differs from its typed weights".to_owned());
    }
    Ok(())
}

const fn gpt_oss_dispatch_count(plan: MoeRoutingPlan) -> u64 {
    match plan {
        MoeRoutingPlan::SingleTokenDirectMarlin => 7,
        MoeRoutingPlan::GenericAlign => 8,
    }
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
            "GPT-OSS CUDA MoE lacks unsigned attribute {name:?}"
        )),
    }
}

fn require_bool(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
    expected: bool,
) -> Result<(), String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Bool(actual)) if *actual == expected => Ok(()),
        _ => Err(format!(
            "GPT-OSS CUDA MoE attribute {name:?} must be {expected}"
        )),
    }
}

fn require_rational(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
    expected: CanonicalRational,
) -> Result<(), String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Rational(actual)) if *actual == expected => Ok(()),
        _ => Err(format!(
            "GPT-OSS CUDA MoE attribute {name:?} must be exactly {}/{}",
            expected.numerator(),
            expected.denominator()
        )),
    }
}

fn checked_i64_runtime(value: u64, label: &str) -> Result<i64, CudaDeviceRuntimeError> {
    i64::try_from(value)
        .map_err(|_| CudaDeviceRuntimeError::contract(format!("{label} exceeds i64")))
}

fn align_up_to(value: u64, alignment: u64) -> Result<u64, String> {
    if value == 0 || alignment == 0 || !alignment.is_power_of_two() {
        return Err("GPT-OSS Marlin alignment requires nonzero power-of-two geometry".to_owned());
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| "GPT-OSS Marlin alignment overflows u64".to_owned())
}

fn checked_grid(elements: u64, label: &str) -> Result<u32, CudaDeviceRuntimeError> {
    u32::try_from(elements.div_ceil(u64::from(THREADS_PER_BLOCK)))
        .map_err(|_| CudaDeviceRuntimeError::contract(format!("{label} grid exceeds u32")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_attributes() -> BTreeMap<AttributeId, SemanticValue> {
        [
            ("hidden_size", SemanticValue::Unsigned(2880)),
            ("expert_count", SemanticValue::Unsigned(32)),
            ("experts_per_token", SemanticValue::Unsigned(4)),
            ("intermediate_size", SemanticValue::Unsigned(2880)),
            ("gate_up_features", SemanticValue::Unsigned(5760)),
            ("normalize_topk", SemanticValue::Bool(true)),
            (
                "swiglu_limit",
                SemanticValue::Rational(CanonicalRational::new(7, 1).unwrap()),
            ),
            ("gate_up_interleaved", SemanticValue::Bool(true)),
            (
                "down_bias_before_route_reduction",
                SemanticValue::Bool(true),
            ),
        ]
        .into_iter()
        .map(|(name, value)| (AttributeId::new(name).unwrap(), value))
        .collect()
    }

    #[test]
    fn accepts_only_the_exact_gpt_oss_moe_attributes() {
        let parsed = GptOssMoeAttributes::from_values(&official_attributes()).unwrap();
        assert_eq!(parsed.hidden_size, 2880);
        assert_eq!(parsed.expert_count, 32);
        assert_eq!(parsed.experts_per_token, 4);
        assert_eq!(parsed.gate_up_features, 5760);

        for name in [
            "normalize_topk",
            "gate_up_interleaved",
            "down_bias_before_route_reduction",
        ] {
            let mut drifted = official_attributes();
            drifted.insert(AttributeId::new(name).unwrap(), SemanticValue::Bool(false));
            assert!(GptOssMoeAttributes::from_values(&drifted).is_err());
        }
        let mut drifted = official_attributes();
        drifted.insert(
            AttributeId::new("swiglu_limit").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(6, 1).unwrap()),
        );
        assert!(GptOssMoeAttributes::from_values(&drifted).is_err());
    }

    #[test]
    fn rejects_non_interleaved_or_non_marlin_geometry_before_launch() {
        let mut attributes = official_attributes();
        attributes.insert(
            AttributeId::new("gate_up_features").unwrap(),
            SemanticValue::Unsigned(2880),
        );
        assert!(GptOssMoeAttributes::from_values(&attributes).is_err());

        let mut attributes = official_attributes();
        attributes.insert(
            AttributeId::new("hidden_size").unwrap(),
            SemanticValue::Unsigned(2879),
        );
        assert!(GptOssMoeAttributes::from_values(&attributes).is_err());
    }

    #[test]
    fn gpt_oss_workspace_adds_only_the_typed_bf16_input_copy() {
        let base =
            MoeWorkspaceLayout::routed_only_with_activation_width(8, 32, 4, 2880, 2880, 2944, 46)
                .unwrap();
        let layout = GptOssMoeWorkspaceLayout::new(8, 32, 4, 2880, 2880, 2944, 46).unwrap();
        assert_eq!(layout.input_bf16_offset, base.total_bytes);
        assert_eq!(layout.input_bf16_bytes, 8 * 2880 * 2);
        assert_eq!(layout.total_bytes, base.total_bytes + 8 * 2880 * 2);
        assert_eq!(layout.marlin_intermediate_size, 2944);
        assert_eq!(layout.base.routed_gate_up.length_bytes(), 8 * 4 * 5760 * 2);
        assert_eq!(
            layout.base.routed_activation.length_bytes(),
            8 * 4 * 2944 * 2
        );
        assert!(layout.total_bytes.is_multiple_of(VALUE_ALIGNMENT_BYTES));
    }

    #[test]
    fn official_down_width_uses_minimal_marlin_k_padding() {
        let attributes = GptOssMoeAttributes::from_values(&official_attributes()).unwrap();
        assert_eq!(attributes.marlin_intermediate_size().unwrap(), 2944);
        assert_eq!(align_up_to(2944, MARLIN_DOWN_K_ALIGNMENT).unwrap(), 2944);
        assert!(GptOssMoeWorkspaceLayout::new(1, 32, 4, 2880, 2880, 2880, 46).is_err());
    }

    #[test]
    fn routing_plan_has_stable_replay_and_dispatch_shapes() {
        let decode = MoeRoutingPlan::for_tokens(1);
        let prefill = MoeRoutingPlan::for_tokens(8);
        assert_ne!(decode.replay_tag(), prefill.replay_tag());
        assert_eq!(gpt_oss_dispatch_count(decode), 7);
        assert_eq!(gpt_oss_dispatch_count(prefill), 8);
    }

    #[test]
    fn launch_validation_rejects_weight_or_pair_geometry_drift() {
        let valid = GptOssMoeLaunchShape {
            tokens: 8,
            expert_count: 32,
            experts_per_token: 4,
            hidden_size: 2880,
            intermediate_size: 2880,
            marlin_intermediate_size: 2944,
            gate_up_features: 5760,
            pair_count: 32,
            sorted_capacity: 544,
            device_ordinal: 0,
        };
        let gate_up = [32, 5760, 2880];
        let down = [32, 2880, 2880];
        let down_execution = [32, 2880, 2944];
        validate_launch_problem(valid, &gate_up, &gate_up, 32, &down, &down_execution, 32).unwrap();

        let mut bad_pairs = valid;
        bad_pairs.pair_count = 31;
        assert!(validate_launch_problem(
            bad_pairs,
            &gate_up,
            &gate_up,
            32,
            &down,
            &down_execution,
            32,
        )
        .is_err());
        assert!(
            validate_launch_problem(valid, &gate_up, &gate_up, 64, &down, &down_execution, 32,)
                .is_err()
        );
        assert!(validate_launch_problem(
            valid,
            &[32, 2880, 2880],
            &gate_up,
            32,
            &down,
            &down_execution,
            32,
        )
        .is_err());
        assert!(validate_launch_problem(valid, &gate_up, &gate_up, 32, &down, &down, 32,).is_err());
    }

    #[test]
    fn workspace_pointer_translation_rejects_address_overflow() {
        let layout = GptOssMoeWorkspaceLayout::new(8, 32, 4, 2880, 2880, 2944, 46).unwrap();
        assert!(GptOssMoeWorkspacePointers::new(u64::MAX - 8, &layout).is_err());
    }

    #[test]
    fn contract_binding_abi_keeps_all_biases_bf16_and_public_io_f16() {
        let contract = gpt_oss_routed_clamped_swiglu_moe_contract().unwrap();
        assert_eq!(contract.descriptor().inputs.len(), 7);
        assert_eq!(contract.descriptor().outputs.len(), 1);
        for ordinal in 1..=6 {
            assert_eq!(
                contract.descriptor().inputs[ordinal].element_types(),
                &BTreeSet::from([ElementType::Bf16])
            );
        }
        assert_eq!(
            contract.descriptor().inputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
        assert_eq!(
            contract.descriptor().outputs[0].element_types(),
            &BTreeSet::from([ElementType::F16])
        );
    }
}
