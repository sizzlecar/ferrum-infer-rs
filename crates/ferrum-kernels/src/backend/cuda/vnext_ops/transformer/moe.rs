//! CUDA provider for the backend-neutral routed/shared SwiGLU MoE contract.

use std::collections::{BTreeMap, BTreeSet};

use cudarc::driver::{
    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CudaFunction,
};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    routed_shared_swiglu_moe_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceBatchingForm, DeviceRuntime, DynamicStorageRequirement, ElementType,
    EncodedDeviceOperation, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderId, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    QuantizationFormatId, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, VNextError, WeightFormatId,
    ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID, ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
};

use super::super::super::marlin::{
    launch_marlin_moe_vllm_raw, MarlinMoeF16WeightType, MarlinMoeRawLaunchArgs,
};
use super::super::super::vnext_replay::CudaCommandReplayKeyBuilder;
use super::super::super::vnext_runtime::{
    CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime, CudaDeviceRuntimeError,
};
use super::super::{binding, contract_error, implementation_fingerprint, same_physical_region};
use super::moe_launch::{region_pointer, zero_region, MoeCudaKernels};
use super::moe_weights::{
    resolve_gptq_marlin_moe_weight, resolve_marlin_fp8_moe_weight, CudaMarlinMoeWeight,
    GPTQ_MARLIN_QUANTIZATION_FORMAT_ID, GPTQ_MARLIN_WEIGHT_FORMAT_ID,
};
use super::moe_workspace::{
    workspace_formula_terms, MoeWorkspaceLayout, WorkspaceRegion, MAX_ROUTER_EXPERTS,
    MAX_ROUTER_TOP_K, MOE_BLOCK_SIZE,
};
use super::{
    contiguous_bindings, ensure_estimator_request, estimate, f16_contiguous, launch_gemm_f16,
    shared_full_region, shared_scratch_region, shared_token_region,
    static_contiguous_reusable_topology, CapturedProviderWorkspace, MarlinProjectionRuntime,
};
use crate::marlin_fp8_materializer::{
    MARLIN_FP8_CAPABILITY_ID, MARLIN_FP8_QUANTIZATION_FORMAT_ID, MARLIN_FP8_WEIGHT_FORMAT_ID,
};

const GPTQ_PROVIDER_ID: &str = "provider.cuda.routed_shared_swiglu_moe.f16.gptq_marlin";
const GPTQ_ESTIMATOR_ID: &str = "resource-estimator.cuda.routed_shared_swiglu_moe.f16.gptq_marlin";
const MARLIN_FP8_PROVIDER_ID: &str = "provider.cuda.routed_shared_swiglu_moe.f16.marlin-fp8-w8a16";
const MARLIN_FP8_ESTIMATOR_ID: &str =
    "resource-estimator.cuda.routed_shared_swiglu_moe.f16.marlin-fp8-w8a16";
const COMMAND_NAME: &str = "vnext_routed_shared_swiglu_moe";
const PLANAR_SILU_MUL_FUNCTION_NAME: &str = "fused_silu_mul_f16";
const VALUE_ALIGNMENT_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoeProviderKind {
    GptqMarlin,
    MarlinFp8,
}

impl MoeProviderKind {
    const fn provider_id(self) -> &'static str {
        match self {
            Self::GptqMarlin => GPTQ_PROVIDER_ID,
            Self::MarlinFp8 => MARLIN_FP8_PROVIDER_ID,
        }
    }

    const fn estimator_id(self) -> &'static str {
        match self {
            Self::GptqMarlin => GPTQ_ESTIMATOR_ID,
            Self::MarlinFp8 => MARLIN_FP8_ESTIMATOR_ID,
        }
    }

    const fn weight_format_id(self) -> &'static str {
        match self {
            Self::GptqMarlin => GPTQ_MARLIN_WEIGHT_FORMAT_ID,
            Self::MarlinFp8 => MARLIN_FP8_WEIGHT_FORMAT_ID,
        }
    }

    const fn quantization_format_id(self) -> &'static str {
        match self {
            Self::GptqMarlin => GPTQ_MARLIN_QUANTIZATION_FORMAT_ID,
            Self::MarlinFp8 => MARLIN_FP8_QUANTIZATION_FORMAT_ID,
        }
    }

    const fn routed_weight_type(self) -> MarlinMoeF16WeightType {
        match self {
            Self::GptqMarlin => MarlinMoeF16WeightType::U4B8,
            Self::MarlinFp8 => MarlinMoeF16WeightType::E4M3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MoeAttributes {
    hidden_size: u64,
    expert_count: u64,
    experts_per_token: u64,
    routed_intermediate_size: u64,
    shared_intermediate_size: u64,
    normalize_topk: bool,
}

impl MoeAttributes {
    fn from_values(attributes: &BTreeMap<AttributeId, SemanticValue>) -> Result<Self, String> {
        let values = Self {
            hidden_size: unsigned_attribute(attributes, "hidden_size")?,
            expert_count: unsigned_attribute(attributes, "expert_count")?,
            experts_per_token: unsigned_attribute(attributes, "experts_per_token")?,
            routed_intermediate_size: unsigned_attribute(attributes, "routed_intermediate_size")?,
            shared_intermediate_size: unsigned_attribute(attributes, "shared_intermediate_size")?,
            normalize_topk: bool_attribute(attributes, "normalize_topk")?,
        };
        values.validate()?;
        Ok(values)
    }

    fn validate(self) -> Result<(), String> {
        if self.hidden_size == 0
            || self.expert_count == 0
            || self.expert_count > MAX_ROUTER_EXPERTS
            || self.experts_per_token == 0
            || self.experts_per_token > self.expert_count
            || self.experts_per_token > MAX_ROUTER_TOP_K
            || self.routed_intermediate_size == 0
            || self.shared_intermediate_size == 0
        {
            return Err(format!(
                "CUDA MoE attributes are outside the current router contract: {self:?}"
            ));
        }
        let gate_up_width = self
            .routed_intermediate_size
            .checked_mul(2)
            .ok_or_else(|| "CUDA MoE routed gate/up width overflows".to_owned())?;
        if !self.hidden_size.is_multiple_of(64)
            || !self.routed_intermediate_size.is_multiple_of(64)
            || !gate_up_width.is_multiple_of(64)
        {
            return Err(format!(
                "CUDA Marlin-MoE hidden/routed widths must be divisible by 64: {self:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MoeLaunchShape {
    tokens: i32,
    expert_count: i32,
    experts_per_token: i32,
    hidden_size: i32,
    routed_intermediate_size: i32,
    shared_intermediate_size: i32,
    pair_count: i32,
    sorted_capacity: i32,
    normalize_topk: bool,
    gate_up_group_size: i32,
    down_group_size: i32,
    device_ordinal: i32,
}

#[derive(Clone, Copy)]
struct MarlinWeightRegions {
    packed: usize,
    scales: usize,
}

#[derive(Clone, Copy)]
enum SharedProjectionRegions {
    F16 {
        gate_up: usize,
        down: usize,
    },
    MarlinFp8 {
        gate: MarlinWeightRegions,
        up: MarlinWeightRegions,
        down: MarlinWeightRegions,
    },
}

#[derive(Clone, Copy)]
struct MoeCommandRegions {
    input: usize,
    router: usize,
    routed_gate_up: MarlinWeightRegions,
    routed_down: MarlinWeightRegions,
    shared_gate: usize,
    shared_projection: SharedProjectionRegions,
    output: usize,
    scratch: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MoeRoutingPlan {
    SingleTokenDirectMarlin,
    GenericAlign,
}

impl MoeRoutingPlan {
    pub(super) fn for_tokens(tokens: i32) -> Self {
        if tokens == 1 {
            Self::SingleTokenDirectMarlin
        } else {
            Self::GenericAlign
        }
    }

    pub(super) fn replay_tag(self) -> &'static [u8] {
        match self {
            Self::SingleTokenDirectMarlin => b"single-token-direct-marlin",
            Self::GenericAlign => b"generic-align",
        }
    }

    fn compute_dispatch_count(self) -> u64 {
        match self {
            Self::SingleTokenDirectMarlin => 11,
            Self::GenericAlign => 12,
        }
    }

    pub(super) fn routed_compute_dispatch_count(self) -> u64 {
        match self {
            Self::SingleTokenDirectMarlin => 6,
            Self::GenericAlign => 7,
        }
    }
}

pub(in crate::backend::cuda::vnext_ops) struct CudaRoutedSharedSwiGluMoeProvider {
    descriptor: OperationProviderDescriptor,
    kind: MoeProviderKind,
    kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
    projection_runtime: Option<MarlinProjectionRuntime>,
    planar_silu_mul: Option<CudaFunction>,
}

impl CudaRoutedSharedSwiGluMoeProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_for_kind(runtime, MoeProviderKind::GptqMarlin)
    }

    pub(in crate::backend::cuda::vnext_ops) fn new_marlin_fp8(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::new_for_kind(runtime, MoeProviderKind::MarlinFp8)
    }

    fn new_for_kind(
        runtime: &CudaDeviceRuntime,
        kind: MoeProviderKind,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = routed_shared_swiglu_moe_contract().map_err(contract_error)?;
        let operation_capability = CapabilityId::new(ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID)
            .map_err(contract_error)?;
        let mut required_capabilities = BTreeSet::from([operation_capability]);
        if kind == MoeProviderKind::MarlinFp8 {
            required_capabilities
                .insert(CapabilityId::new(MARLIN_FP8_CAPABILITY_ID).map_err(contract_error)?);
        }
        for capability in &required_capabilities {
            if !runtime.descriptor().capabilities.contains(capability) {
                return Err(CudaDeviceRuntimeError::contract(format!(
                    "CUDA runtime does not advertise capability `{capability}`"
                )));
            }
        }
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("moe.rs").as_bytes(),
            include_str!("moe_launch.rs").as_bytes(),
            include_str!("moe_weights.rs").as_bytes(),
            include_str!("moe_workspace.rs").as_bytes(),
            include_str!("../../marlin.rs").as_bytes(),
            include_str!("../../../../../kernels/moe_combine.cu").as_bytes(),
            crate::native_ops::CUDA_NATIVE_SOURCE_BUNDLE_ID.as_bytes(),
            crate::ptx::MOE_ROUTER.as_bytes(),
            crate::ptx::MOE_ALIGN_BLOCK_SIZE_PAIR_IDS.as_bytes(),
            crate::ptx::MOE_COMBINE.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
            kind.provider_id().as_bytes(),
        ]);
        let estimator_fingerprint = implementation_fingerprint(&[
            include_str!("moe_workspace.rs").as_bytes(),
            kind.estimator_id().as_bytes(),
            provider_fingerprint.as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(kind.provider_id()).map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            provider_fingerprint,
            ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            required_capabilities,
            BTreeSet::from([WeightFormatId::new(kind.weight_format_id()).map_err(contract_error)?]),
            BTreeSet::from([
                QuantizationFormatId::new(kind.quantization_format_id()).map_err(contract_error)?
            ]),
            contiguous_bindings(7),
            kind.estimator_id(),
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
        let kernels = MoeCudaKernels::load(runtime)?;
        let (projection_runtime, planar_silu_mul) = if kind == MoeProviderKind::MarlinFp8 {
            let module = runtime
                .context()
                .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL.to_owned()))
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("MoE planar SiLU module load", error)
                })?;
            let planar_silu_mul = module
                .load_function(PLANAR_SILU_MUL_FUNCTION_NAME)
                .map_err(|error| {
                    CudaDeviceRuntimeError::driver("MoE planar SiLU function load", error)
                })?;
            (
                Some(MarlinProjectionRuntime::query(runtime)?),
                Some(planar_silu_mul),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            descriptor,
            kind,
            kernels,
            multiprocessor_count,
            device_ordinal,
            projection_runtime,
            planar_silu_mul,
        })
    }
}

impl OperationResourceEstimator for CudaRoutedSharedSwiGluMoeProvider {
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
            ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
        )?;
        let attributes = MoeAttributes::from_values(request.attributes()).map_err(invalid_plan)?;
        let (fixed_bytes, bytes_per_token) = workspace_formula_terms(
            attributes.expert_count,
            attributes.experts_per_token,
            attributes.hidden_size,
            attributes.routed_intermediate_size,
            attributes.shared_intermediate_size,
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

impl OperationProvider<CudaDeviceRuntime> for CudaRoutedSharedSwiGluMoeProvider {
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
        encode_moe(
            self.descriptor.provider_implementation_fingerprint(),
            self.kind,
            self.kernels.clone(),
            self.multiprocessor_count,
            self.device_ordinal,
            self.projection_runtime,
            self.planar_silu_mul.clone(),
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            OperationFailure::new(
                identity,
                ProfilePhase::Forward,
                "cuda.routed_shared_swiglu_moe.encode",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued CUDA operation identity must form a valid provider failure")
        })
    }
}

fn encode_moe(
    provider_fingerprint: &str,
    provider_kind: MoeProviderKind,
    kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
    projection_runtime: Option<MarlinProjectionRuntime>,
    planar_silu_mul: Option<CudaFunction>,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID
    {
        return Err(
            "CUDA routed/shared MoE provider received another or empty operation".to_owned(),
        );
    }
    let first = &invocation.participants()[0];
    let attributes = MoeAttributes::from_values(first.attributes())?;
    let tokens = invocation.work_shape().immediate_tokens();
    if tokens == 0 {
        return Err("CUDA routed/shared MoE invocation has no immediate tokens".to_owned());
    }
    for participant in invocation.participants() {
        if MoeAttributes::from_values(participant.attributes())? != attributes {
            return Err("CUDA routed/shared MoE participant attributes disagree".to_owned());
        }
        validate_participant(participant.bindings(), attributes)?;
    }

    let gate_up_dimensions = vec![
        attributes.expert_count,
        2,
        attributes.routed_intermediate_size,
        attributes.hidden_size,
    ];
    let down_dimensions = vec![
        attributes.expert_count,
        attributes.hidden_size,
        attributes.routed_intermediate_size,
    ];
    let gate_up =
        resolve_shared_marlin_weight_for_kind(&invocation, 2, &gate_up_dimensions, provider_kind)?;
    let down =
        resolve_shared_marlin_weight_for_kind(&invocation, 3, &down_dimensions, provider_kind)?;
    if gate_up.expert_count() != attributes.expert_count
        || down.expert_count() != attributes.expert_count
        || gate_up.weight_type() != provider_kind.routed_weight_type()
        || down.weight_type() != provider_kind.routed_weight_type()
    {
        return Err(
            "CUDA routed/shared MoE physical expert stack differs from provider kind or attributes"
                .to_owned(),
        );
    }

    let layout = MoeWorkspaceLayout::new(
        tokens,
        attributes.expert_count,
        attributes.experts_per_token,
        attributes.hidden_size,
        attributes.routed_intermediate_size,
        attributes.shared_intermediate_size,
        multiprocessor_count,
    )?;
    let mut regions = Vec::new();
    let mut push_region = |region| {
        let index = regions.len();
        regions.push(region);
        index
    };
    let input_region = push_region(shared_token_region(
        &invocation,
        ResolvedValueRole::Input,
        0,
        ElementType::F16,
        tokens,
    )?);
    let router_region = push_region(shared_full_region(
        &invocation,
        ResolvedValueRole::Input,
        1,
        ElementType::F16,
    )?);
    let routed_gate_up = MarlinWeightRegions {
        packed: push_region(gate_up.packed_region().clone()),
        scales: push_region(gate_up.scales_region().clone()),
    };
    let routed_down = MarlinWeightRegions {
        packed: push_region(down.packed_region().clone()),
        scales: push_region(down.scales_region().clone()),
    };
    let shared_gate_region = push_region(shared_full_region(
        &invocation,
        ResolvedValueRole::Input,
        4,
        ElementType::F16,
    )?);
    drop(push_region);
    let shared_projection = match provider_kind {
        MoeProviderKind::GptqMarlin => {
            let gate_up = regions.len();
            regions.push(shared_full_region(
                &invocation,
                ResolvedValueRole::Input,
                5,
                ElementType::F16,
            )?);
            let down = regions.len();
            regions.push(shared_full_region(
                &invocation,
                ResolvedValueRole::Input,
                6,
                ElementType::F16,
            )?);
            SharedProjectionRegions::F16 { gate_up, down }
        }
        MoeProviderKind::MarlinFp8 => {
            if projection_runtime.is_none() || planar_silu_mul.is_none() {
                return Err(
                    "CUDA Marlin FP8 MoE provider lacks its projection runtime or planar SiLU kernel"
                        .to_owned(),
                );
            }
            let gate = super::push_shared_marlin_fp8_weight(
                &mut regions,
                &invocation,
                5,
                &[attributes.shared_intermediate_size, attributes.hidden_size],
                Some(0),
            )?;
            let up = super::push_shared_marlin_fp8_weight(
                &mut regions,
                &invocation,
                5,
                &[attributes.shared_intermediate_size, attributes.hidden_size],
                Some(1),
            )?;
            let down = super::push_shared_marlin_fp8_weight(
                &mut regions,
                &invocation,
                6,
                &[attributes.hidden_size, attributes.shared_intermediate_size],
                None,
            )?;
            SharedProjectionRegions::MarlinFp8 {
                gate: MarlinWeightRegions {
                    packed: gate.packed_region,
                    scales: gate.scales_region,
                },
                up: MarlinWeightRegions {
                    packed: up.packed_region,
                    scales: up.scales_region,
                },
                down: MarlinWeightRegions {
                    packed: down.packed_region,
                    scales: down.scales_region,
                },
            }
        }
    };
    let output_region = regions.len();
    regions.push(shared_token_region(
        &invocation,
        ResolvedValueRole::Output,
        0,
        ElementType::F16,
        tokens,
    )?);
    let scratch_region = regions.len();
    regions.push(shared_scratch_region(&invocation, layout.total_bytes)?);
    let command_regions = MoeCommandRegions {
        input: input_region,
        router: router_region,
        routed_gate_up,
        routed_down,
        shared_gate: shared_gate_region,
        shared_projection,
        output: output_region,
        scratch: scratch_region,
    };
    let shape = MoeLaunchShape {
        tokens: checked_i32(tokens, "MoE token count")?,
        expert_count: checked_i32(attributes.expert_count, "MoE expert count")?,
        experts_per_token: checked_i32(attributes.experts_per_token, "MoE experts per token")?,
        hidden_size: checked_i32(attributes.hidden_size, "MoE hidden size")?,
        routed_intermediate_size: checked_i32(
            attributes.routed_intermediate_size,
            "MoE routed intermediate size",
        )?,
        shared_intermediate_size: checked_i32(
            attributes.shared_intermediate_size,
            "MoE shared intermediate size",
        )?,
        pair_count: checked_i32(layout.pair_count, "MoE pair count")?,
        sorted_capacity: checked_i32(layout.sorted_capacity, "MoE sorted capacity")?,
        normalize_topk: attributes.normalize_topk,
        gate_up_group_size: gate_up.group_size(),
        down_group_size: down.group_size(),
        device_ordinal,
    };
    let routing_plan = MoeRoutingPlan::for_tokens(shape.tokens);
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, COMMAND_NAME)
        .i32(shape.tokens)
        .i32(shape.expert_count)
        .i32(shape.experts_per_token)
        .i32(shape.hidden_size)
        .i32(shape.routed_intermediate_size)
        .i32(shape.shared_intermediate_size)
        .boolean(shape.normalize_topk)
        .i32(shape.gate_up_group_size)
        .i32(shape.down_group_size)
        .bytes(match provider_kind {
            MoeProviderKind::GptqMarlin => b"u4b8",
            MoeProviderKind::MarlinFp8 => b"e4m3",
        })
        .i32(shape.device_ordinal)
        .bytes(routing_plan.replay_tag())
        .u64(layout.total_bytes)
        .u64(MOE_BLOCK_SIZE)
        .finish();
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA MoE participant count exceeds u32".to_owned())?;

    CudaDeviceCommand::replayable_operation_with_blas(
        COMMAND_NAME,
        regions,
        replay_key,
        move |stream, blas, regions| {
            let scratch = &regions[command_regions.scratch];
            if scratch.length_bytes() < layout.total_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "MoE scratch is smaller than its admitted estimate",
                ));
            }
            let pointers = MoeWorkspacePointers::new(scratch.device_ptr(), &layout)?;

            launch_gemm_f16(
                blas,
                regions[command_regions.input].device_ptr(),
                regions[command_regions.router].device_ptr(),
                pointers.router_logits,
                shape.tokens,
                shape.expert_count,
                shape.hidden_size,
                "vNext MoE router GEMM",
            )?;
            match routing_plan {
                MoeRoutingPlan::SingleTokenDirectMarlin => {
                    kernels.launch_single_token_router(
                        stream,
                        pointers.router_logits,
                        pointers.route_ids,
                        pointers.route_weights,
                        pointers.sorted_token_ids,
                        pointers.expert_block_ids,
                        pointers.total_tokens_post_pad,
                        shape.expert_count,
                        shape.experts_per_token,
                        shape.normalize_topk,
                    )?;
                }
                MoeRoutingPlan::GenericAlign => {
                    kernels.launch_router(
                        stream,
                        pointers.router_logits,
                        pointers.route_ids,
                        pointers.route_weights,
                        shape.tokens,
                        shape.expert_count,
                        shape.experts_per_token,
                        shape.normalize_topk,
                    )?;
                    kernels.launch_align(
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
            }

            zero_region(stream, scratch.device_ptr(), layout.marlin_workspace)?;
            launch_marlin_typed(
                stream,
                provider_kind.routed_weight_type(),
                regions[command_regions.input].device_ptr(),
                regions[command_regions.routed_gate_up.packed].device_ptr(),
                regions[command_regions.routed_gate_up.scales].device_ptr(),
                pointers.routed_gate_up,
                pointers.marlin(),
                shape.tokens,
                shape.experts_per_token,
                shape
                    .routed_intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("MoE gate/up width exceeds i32")
                    })?,
                shape.hidden_size,
                shape.gate_up_group_size,
                shape.device_ordinal,
            )?;
            kernels.launch_silu(
                stream,
                pointers.routed_gate_up,
                pointers.routed_activation,
                shape.routed_intermediate_size,
                u64::try_from(shape.pair_count)
                    .ok()
                    .and_then(|pairs| pairs.checked_mul(shape.routed_intermediate_size as u64))
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "MoE routed activation element count overflows",
                        )
                    })?,
            )?;
            zero_region(stream, scratch.device_ptr(), layout.marlin_workspace)?;
            launch_marlin_typed(
                stream,
                provider_kind.routed_weight_type(),
                pointers.routed_activation,
                regions[command_regions.routed_down.packed].device_ptr(),
                regions[command_regions.routed_down.scales].device_ptr(),
                pointers.routed_down_slots,
                pointers.marlin(),
                shape.pair_count,
                1,
                shape.hidden_size,
                shape.routed_intermediate_size,
                shape.down_group_size,
                shape.device_ordinal,
            )?;
            kernels.launch_weighted_sum(
                stream,
                pointers.routed_down_slots,
                pointers.route_weights,
                regions[command_regions.output].device_ptr(),
                shape.tokens,
                shape.experts_per_token,
                shape.hidden_size,
            )?;

            launch_gemm_f16(
                blas,
                regions[command_regions.input].device_ptr(),
                regions[command_regions.shared_gate].device_ptr(),
                pointers.shared_gate,
                shape.tokens,
                1,
                shape.hidden_size,
                "vNext MoE shared gate GEMM",
            )?;
            let shared_activation_elements = u64::try_from(shape.tokens)
                .ok()
                .and_then(|tokens| tokens.checked_mul(shape.shared_intermediate_size as u64))
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "MoE shared activation element count overflows",
                    )
                })?;
            match command_regions.shared_projection {
                SharedProjectionRegions::F16 { gate_up, down } => {
                    launch_gemm_f16(
                        blas,
                        regions[command_regions.input].device_ptr(),
                        regions[gate_up].device_ptr(),
                        pointers.shared_gate_up,
                        shape.tokens,
                        shape
                            .shared_intermediate_size
                            .checked_mul(2)
                            .ok_or_else(|| {
                                CudaDeviceRuntimeError::contract(
                                    "MoE shared gate/up width exceeds i32",
                                )
                            })?,
                        shape.hidden_size,
                        "vNext MoE shared gate/up GEMM",
                    )?;
                    kernels.launch_silu(
                        stream,
                        pointers.shared_gate_up,
                        pointers.shared_activation,
                        shape.shared_intermediate_size,
                        shared_activation_elements,
                    )?;
                    launch_gemm_f16(
                        blas,
                        pointers.shared_activation,
                        regions[down].device_ptr(),
                        pointers.shared_output,
                        shape.tokens,
                        shape.hidden_size,
                        shape.shared_intermediate_size,
                        "vNext MoE shared down GEMM",
                    )?;
                }
                SharedProjectionRegions::MarlinFp8 { gate, up, down } => {
                    let projection_runtime = projection_runtime.ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "Marlin FP8 MoE projection runtime is absent",
                        )
                    })?;
                    let planar_silu_mul = planar_silu_mul.as_ref().ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "Marlin FP8 MoE planar SiLU kernel is absent",
                        )
                    })?;
                    let shared_activation_bytes = shared_activation_elements
                        .checked_mul(ElementType::F16.size_bytes())
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "Marlin FP8 MoE shared activation bytes overflow",
                            )
                        })?;
                    let shared_up = pointers
                        .shared_gate_up
                        .checked_add(shared_activation_bytes)
                        .ok_or_else(|| {
                            CudaDeviceRuntimeError::contract(
                                "Marlin FP8 MoE shared up pointer overflows",
                            )
                        })?;
                    for (weight, output) in [(gate, pointers.shared_gate_up), (up, shared_up)] {
                        projection_runtime.launch(
                            crate::backend::cuda::vllm_marlin::MarlinF16WeightType::E4M3Fn,
                            stream,
                            regions[command_regions.input].device_ptr(),
                            regions[weight.packed].device_ptr(),
                            regions[weight.scales].device_ptr(),
                            None,
                            output,
                            pointers.marlin_c_tmp,
                            layout.marlin_c_tmp.length_bytes(),
                            shape.tokens,
                            shape.shared_intermediate_size,
                            shape.hidden_size,
                            -1,
                            "Marlin FP8 MoE shared gate/up projection",
                        )?;
                    }
                    super::launch_planar_silu_mul(
                        stream,
                        planar_silu_mul,
                        pointers.shared_gate_up,
                        shared_up,
                        pointers.shared_activation,
                        shared_activation_elements,
                    )?;
                    projection_runtime.launch(
                        crate::backend::cuda::vllm_marlin::MarlinF16WeightType::E4M3Fn,
                        stream,
                        pointers.shared_activation,
                        regions[down.packed].device_ptr(),
                        regions[down.scales].device_ptr(),
                        None,
                        pointers.shared_output,
                        pointers.marlin_c_tmp,
                        layout.marlin_c_tmp.length_bytes(),
                        shape.tokens,
                        shape.hidden_size,
                        shape.shared_intermediate_size,
                        -1,
                        "Marlin FP8 MoE shared down projection",
                    )?;
                }
            }
            kernels.launch_token_gate_add(
                stream,
                regions[command_regions.output].device_ptr(),
                pointers.shared_output,
                pointers.shared_gate,
                shape.tokens,
                shape.hidden_size,
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::Packed,
            participant_count,
            tokens,
            routing_plan.compute_dispatch_count()
                + u64::from(provider_kind == MoeProviderKind::MarlinFp8),
            2,
        )
    })
    .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn launch_marlin(
    stream: &cudarc::driver::CudaStream,
    input: u64,
    packed_weight: u64,
    scales: u64,
    output: u64,
    workspace: MarlinMoeWorkspacePointers,
    prob_m: i32,
    top_k: i32,
    prob_n: i32,
    prob_k: i32,
    group_size: i32,
    device_ordinal: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    launch_marlin_typed(
        stream,
        MarlinMoeF16WeightType::U4B8,
        input,
        packed_weight,
        scales,
        output,
        workspace,
        prob_m,
        top_k,
        prob_n,
        prob_k,
        group_size,
        device_ordinal,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_marlin_typed(
    stream: &cudarc::driver::CudaStream,
    weight_type: MarlinMoeF16WeightType,
    input: u64,
    packed_weight: u64,
    scales: u64,
    output: u64,
    workspace: MarlinMoeWorkspacePointers,
    prob_m: i32,
    top_k: i32,
    prob_n: i32,
    prob_k: i32,
    group_size: i32,
    device_ordinal: i32,
) -> Result<(), CudaDeviceRuntimeError> {
    launch_marlin_moe_vllm_raw(
        stream,
        MarlinMoeRawLaunchArgs {
            weight_type,
            a: input,
            b: packed_weight,
            c: output,
            c_tmp: Some(workspace.marlin_c_tmp),
            scales,
            zero_points: None,
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
            has_zero_points: false,
            device_ordinal,
            use_atomic_add: false,
            use_fp32_reduce: true,
        },
    )
    .map_err(|error| {
        CudaDeviceRuntimeError::contract(format!("vNext Marlin-MoE launch rejected: {error}"))
    })
}

#[derive(Clone, Copy)]
pub(super) struct MarlinMoeWorkspacePointers {
    pub(super) sorted_token_ids: u64,
    pub(super) expert_block_ids: u64,
    pub(super) total_tokens_post_pad: u64,
    pub(super) marlin_workspace: u64,
    pub(super) marlin_c_tmp: u64,
}

fn validate_participant(
    bindings: &[ResolvedValueBinding],
    attributes: MoeAttributes,
) -> Result<(), String> {
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let [canonical_tokens, input_hidden] = input.tensor().dimensions() else {
        return Err("CUDA routed/shared MoE input is not two-dimensional".to_owned());
    };
    if *input_hidden != attributes.hidden_size || !f16_contiguous(input) {
        return Err(
            "CUDA routed/shared MoE input differs from [tokens, hidden] F16 contiguous".to_owned(),
        );
    }
    let expected = [
        (1, vec![attributes.expert_count, attributes.hidden_size]),
        (
            2,
            vec![
                attributes.expert_count,
                2,
                attributes.routed_intermediate_size,
                attributes.hidden_size,
            ],
        ),
        (
            3,
            vec![
                attributes.expert_count,
                attributes.hidden_size,
                attributes.routed_intermediate_size,
            ],
        ),
        (4, vec![1, attributes.hidden_size]),
        (
            5,
            vec![
                2,
                attributes.shared_intermediate_size,
                attributes.hidden_size,
            ],
        ),
        (
            6,
            vec![attributes.hidden_size, attributes.shared_intermediate_size],
        ),
    ];
    for (ordinal, dimensions) in expected {
        let value = binding(bindings, ResolvedValueRole::Input, ordinal)?;
        if value.tensor().dimensions() != dimensions || !f16_contiguous(value) {
            return Err(format!(
                "CUDA routed/shared MoE input {ordinal} differs from shape {dimensions:?} F16 contiguous"
            ));
        }
    }
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    if output.tensor().dimensions() != [*canonical_tokens, attributes.hidden_size]
        || !f16_contiguous(output)
    {
        return Err(
            "CUDA routed/shared MoE output differs from [tokens, hidden] F16 contiguous".to_owned(),
        );
    }
    Ok(())
}

pub(super) fn resolve_shared_marlin_weight(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinMoeWeight, String> {
    resolve_shared_marlin_weight_for_kind(
        invocation,
        ordinal,
        logical_dimensions,
        MoeProviderKind::GptqMarlin,
    )
}

fn resolve_shared_marlin_weight_for_kind(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
    provider_kind: MoeProviderKind,
) -> Result<CudaMarlinMoeWeight, String> {
    let resolve =
        |participant: &ferrum_interfaces::vnext::OperationInvocation<'_, CudaDeviceBuffer>| {
            let binding = binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?;
            match provider_kind {
                MoeProviderKind::GptqMarlin => {
                    resolve_gptq_marlin_moe_weight(participant, binding, logical_dimensions)
                }
                MoeProviderKind::MarlinFp8 => {
                    resolve_marlin_fp8_moe_weight(participant, binding, logical_dimensions)
                }
            }
        };
    let first = &invocation.participants()[0];
    let resolved = resolve(first)?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve(participant)?;
        if !same_marlin_weight(&resolved, &candidate) {
            return Err(format!(
                "CUDA routed/shared MoE input {ordinal} is not one shared physical Marlin stack"
            ));
        }
    }
    Ok(resolved)
}

fn same_marlin_weight(left: &CudaMarlinMoeWeight, right: &CudaMarlinMoeWeight) -> bool {
    left.logical_dimensions() == right.logical_dimensions()
        && left.packed_physical_dimensions() == right.packed_physical_dimensions()
        && left.scales_physical_dimensions() == right.scales_physical_dimensions()
        && left.expert_count() == right.expert_count()
        && left.packed_expert_stride_bytes() == right.packed_expert_stride_bytes()
        && left.scales_expert_stride_bytes() == right.scales_expert_stride_bytes()
        && left.group_size() == right.group_size()
        && left.weight_type() == right.weight_type()
        && same_physical_region(left.packed_region(), right.packed_region())
        && same_physical_region(left.scales_region(), right.scales_region())
}

#[derive(Clone, Copy)]
struct MoeWorkspacePointers {
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
    shared_gate: u64,
    shared_gate_up: u64,
    shared_activation: u64,
    shared_output: u64,
}

impl MoeWorkspacePointers {
    fn new(base: u64, layout: &MoeWorkspaceLayout) -> Result<Self, CudaDeviceRuntimeError> {
        let pointer = |region: WorkspaceRegion| region_pointer(base, region);
        let shared = layout.shared().ok_or_else(|| {
            CudaDeviceRuntimeError::contract(
                "routed/shared MoE provider received a routed-only workspace",
            )
        })?;
        Ok(Self {
            router_logits: pointer(layout.router_logits)?,
            route_ids: pointer(layout.route_ids)?,
            route_weights: pointer(layout.route_weights)?,
            sorted_token_ids: pointer(layout.sorted_token_ids)?,
            expert_block_ids: pointer(layout.expert_block_ids)?,
            total_tokens_post_pad: pointer(layout.total_tokens_post_pad)?,
            marlin_workspace: pointer(layout.marlin_workspace)?,
            marlin_c_tmp: pointer(layout.marlin_c_tmp)?,
            routed_gate_up: pointer(layout.routed_gate_up)?,
            routed_activation: pointer(layout.routed_activation)?,
            routed_down_slots: pointer(layout.routed_down_slots)?,
            shared_gate: pointer(shared.gate)?,
            shared_gate_up: pointer(shared.gate_up)?,
            shared_activation: pointer(shared.activation)?,
            shared_output: pointer(shared.output)?,
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

pub(super) fn unsigned_attribute(
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
            "CUDA MoE provider lacks unsigned attribute {name:?}"
        )),
    }
}

pub(super) fn bool_attribute(
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    name: &str,
) -> Result<bool, String> {
    match attributes
        .iter()
        .find(|(attribute, _)| attribute.as_str() == name)
        .map(|(_, value)| value)
    {
        Some(SemanticValue::Bool(value)) => Ok(*value),
        _ => Err(format!("CUDA MoE provider lacks bool attribute {name:?}")),
    }
}

pub(super) fn checked_i32(value: u64, label: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{label} exceeds i32"))
}

pub(super) fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attributes(values: [(&str, SemanticValue); 6]) -> BTreeMap<AttributeId, SemanticValue> {
        values
            .into_iter()
            .map(|(name, value)| (AttributeId::new(name).unwrap(), value))
            .collect()
    }

    #[test]
    fn parses_qwen35_moe_shape_without_model_identity() {
        let parsed = MoeAttributes::from_values(&attributes([
            ("hidden_size", SemanticValue::Unsigned(2048)),
            ("expert_count", SemanticValue::Unsigned(256)),
            ("experts_per_token", SemanticValue::Unsigned(8)),
            ("routed_intermediate_size", SemanticValue::Unsigned(512)),
            ("shared_intermediate_size", SemanticValue::Unsigned(512)),
            ("normalize_topk", SemanticValue::Bool(true)),
        ]))
        .unwrap();
        assert_eq!(parsed.hidden_size, 2048);
        assert_eq!(parsed.expert_count, 256);
        assert_eq!(parsed.experts_per_token, 8);
        assert!(parsed.normalize_topk);
    }

    #[test]
    fn rejects_router_geometry_outside_compiled_kernel_bounds() {
        let error = MoeAttributes::from_values(&attributes([
            ("hidden_size", SemanticValue::Unsigned(2048)),
            ("expert_count", SemanticValue::Unsigned(257)),
            ("experts_per_token", SemanticValue::Unsigned(8)),
            ("routed_intermediate_size", SemanticValue::Unsigned(512)),
            ("shared_intermediate_size", SemanticValue::Unsigned(512)),
            ("normalize_topk", SemanticValue::Bool(true)),
        ]))
        .unwrap_err();
        assert!(error.contains("router contract"), "{error}");
    }

    #[test]
    fn selects_direct_marlin_routing_only_for_single_token_decode() {
        assert_eq!(
            MoeRoutingPlan::for_tokens(1),
            MoeRoutingPlan::SingleTokenDirectMarlin
        );
        for tokens in [2, 32, 1024] {
            assert_eq!(
                MoeRoutingPlan::for_tokens(tokens),
                MoeRoutingPlan::GenericAlign
            );
        }
    }

    #[test]
    fn single_token_routing_removes_exactly_one_compute_dispatch() {
        assert_eq!(
            MoeRoutingPlan::SingleTokenDirectMarlin.compute_dispatch_count(),
            11
        );
        assert_eq!(MoeRoutingPlan::GenericAlign.compute_dispatch_count(), 12);
    }

    #[test]
    fn gptq_and_marlin_fp8_providers_advertise_disjoint_weight_abis() {
        let gptq = MoeProviderKind::GptqMarlin;
        assert_eq!(gptq.provider_id(), GPTQ_PROVIDER_ID);
        assert_eq!(gptq.weight_format_id(), GPTQ_MARLIN_WEIGHT_FORMAT_ID);
        assert_eq!(
            gptq.quantization_format_id(),
            GPTQ_MARLIN_QUANTIZATION_FORMAT_ID
        );
        assert_eq!(gptq.routed_weight_type(), MarlinMoeF16WeightType::U4B8);
        assert_ne!(gptq.weight_format_id(), MARLIN_FP8_WEIGHT_FORMAT_ID);

        let fp8 = MoeProviderKind::MarlinFp8;
        assert_eq!(fp8.provider_id(), MARLIN_FP8_PROVIDER_ID);
        assert_eq!(fp8.weight_format_id(), MARLIN_FP8_WEIGHT_FORMAT_ID);
        assert_eq!(
            fp8.quantization_format_id(),
            MARLIN_FP8_QUANTIZATION_FORMAT_ID
        );
        assert_eq!(fp8.routed_weight_type(), MarlinMoeF16WeightType::E4M3);
        assert_ne!(fp8.weight_format_id(), GPTQ_MARLIN_WEIGHT_FORMAT_ID);
    }
}
