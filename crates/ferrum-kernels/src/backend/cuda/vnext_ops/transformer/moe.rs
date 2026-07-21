//! CUDA provider for the backend-neutral routed/shared SwiGLU MoE contract.

use std::collections::{BTreeMap, BTreeSet};

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use ferrum_interfaces::vnext::{
    routed_shared_swiglu_moe_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceRuntime, DynamicStorageRequirement, ElementType, EncodedDeviceOperation,
    OperationContract, OperationFailure, OperationProvider, OperationProviderDescriptor,
    OperationResourceEstimate, OperationResourceEstimateRequest, OperationResourceEstimator,
    ProfilePhase, ProviderId, ProviderWorkspaceRequirement, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula, QuantizationFormatId, ResolvedValueBinding, ResolvedValueRole,
    SemanticValue, VNextError, WeightFormatId, ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID,
    ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
};

use super::super::super::marlin::{launch_marlin_moe_vllm_raw, MarlinMoeRawLaunchArgs};
use super::super::super::vnext_replay::CudaCommandReplayKeyBuilder;
use super::super::super::vnext_runtime::{
    CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime, CudaDeviceRuntimeError,
};
use super::super::{binding, contract_error, implementation_fingerprint, same_physical_region};
use super::moe_launch::{region_pointer, zero_region, MoeCudaKernels};
use super::moe_weights::{
    resolve_gptq_marlin_moe_weight, CudaMarlinMoeWeight, GPTQ_MARLIN_QUANTIZATION_FORMAT_ID,
    GPTQ_MARLIN_WEIGHT_FORMAT_ID,
};
use super::moe_workspace::{
    workspace_formula_terms, MoeWorkspaceLayout, WorkspaceRegion, MAX_ROUTER_EXPERTS,
    MAX_ROUTER_TOP_K, MOE_BLOCK_SIZE,
};
use super::{
    contiguous_bindings, ensure_estimator_request, estimate, f16_contiguous, launch_gemm_f16,
    shared_full_region, shared_scratch_region, shared_token_region,
};

const PROVIDER_ID: &str = "provider.cuda.routed_shared_swiglu_moe.f16.gptq_marlin";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.routed_shared_swiglu_moe.f16.gptq_marlin";
const COMMAND_NAME: &str = "vnext_routed_shared_swiglu_moe";
const VALUE_ALIGNMENT_BYTES: u64 = 16;

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

pub(in crate::backend::cuda::vnext_ops) struct CudaRoutedSharedSwiGluMoeProvider {
    descriptor: OperationProviderDescriptor,
    kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
}

impl CudaRoutedSharedSwiGluMoeProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = routed_shared_swiglu_moe_contract().map_err(contract_error)?;
        let capability = CapabilityId::new(ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "CUDA runtime does not advertise capability `{ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID}`"
            )));
        }
        let provider_fingerprint = implementation_fingerprint(&[
            include_str!("moe.rs").as_bytes(),
            include_str!("moe_launch.rs").as_bytes(),
            include_str!("moe_weights.rs").as_bytes(),
            include_str!("moe_workspace.rs").as_bytes(),
            include_str!("../../marlin.rs").as_bytes(),
            include_str!("../../../../../kernels/moe_combine.cu").as_bytes(),
            include_str!("../../../../../kernels/vllm_marlin_moe/ops.cu").as_bytes(),
            crate::ptx::MOE_ROUTER.as_bytes(),
            crate::ptx::MOE_ALIGN_BLOCK_SIZE_PAIR_IDS.as_bytes(),
            crate::ptx::MOE_COMBINE.as_bytes(),
            crate::ptx::FUSED_SILU_MUL.as_bytes(),
        ]);
        let estimator_fingerprint = implementation_fingerprint(&[
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
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(GPTQ_MARLIN_WEIGHT_FORMAT_ID).map_err(contract_error)?
            ]),
            BTreeSet::from([
                QuantizationFormatId::new(GPTQ_MARLIN_QUANTIZATION_FORMAT_ID)
                    .map_err(contract_error)?,
            ]),
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
        let kernels = MoeCudaKernels::load(runtime)?;
        Ok(Self {
            descriptor,
            kernels,
            multiprocessor_count,
            device_ordinal,
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
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_moe(
            self.descriptor.provider_implementation_fingerprint(),
            self.kernels.clone(),
            self.multiprocessor_count,
            self.device_ordinal,
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
    kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
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
    let gate_up = resolve_shared_marlin_weight(&invocation, 2, &gate_up_dimensions)?;
    let down = resolve_shared_marlin_weight(&invocation, 3, &down_dimensions)?;
    if gate_up.expert_count() != attributes.expert_count
        || down.expert_count() != attributes.expert_count
    {
        return Err(
            "CUDA routed/shared MoE physical expert count differs from attributes".to_owned(),
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
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::F16)?,
        gate_up.packed_region().clone(),
        gate_up.scales_region().clone(),
        down.packed_region().clone(),
        down.scales_region().clone(),
        shared_full_region(&invocation, ResolvedValueRole::Input, 4, ElementType::F16)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 5, ElementType::F16)?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 6, ElementType::F16)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_scratch_region(&invocation, layout.total_bytes)?,
    ];
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
        gate_up_group_size: i32::try_from(gate_up.group_size())
            .map_err(|_| "MoE gate/up group size exceeds i32".to_owned())?,
        down_group_size: i32::try_from(down.group_size())
            .map_err(|_| "MoE down group size exceeds i32".to_owned())?,
        device_ordinal,
    };
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
        .i32(shape.device_ordinal)
        .u64(layout.total_bytes)
        .u64(MOE_BLOCK_SIZE)
        .finish();

    CudaDeviceCommand::replayable_operation_with_blas(
        COMMAND_NAME,
        regions,
        replay_key,
        move |stream, blas, regions| {
            let scratch = &regions[10];
            if scratch.length_bytes() < layout.total_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "MoE scratch is smaller than its admitted estimate",
                ));
            }
            let pointers = MoeWorkspacePointers::new(scratch.device_ptr(), &layout)?;

            launch_gemm_f16(
                blas,
                regions[0].device_ptr(),
                regions[1].device_ptr(),
                pointers.router_logits,
                shape.tokens,
                shape.expert_count,
                shape.hidden_size,
                "vNext MoE router GEMM",
            )?;
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

            zero_region(stream, scratch.device_ptr(), layout.marlin_workspace)?;
            launch_marlin(
                stream,
                regions[0].device_ptr(),
                regions[2].device_ptr(),
                regions[3].device_ptr(),
                pointers.routed_gate_up,
                pointers,
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
            launch_marlin(
                stream,
                pointers.routed_activation,
                regions[4].device_ptr(),
                regions[5].device_ptr(),
                pointers.routed_down_slots,
                pointers,
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
                regions[9].device_ptr(),
                shape.tokens,
                shape.experts_per_token,
                shape.hidden_size,
            )?;

            launch_gemm_f16(
                blas,
                regions[0].device_ptr(),
                regions[6].device_ptr(),
                pointers.shared_gate,
                shape.tokens,
                1,
                shape.hidden_size,
                "vNext MoE shared gate GEMM",
            )?;
            launch_gemm_f16(
                blas,
                regions[0].device_ptr(),
                regions[7].device_ptr(),
                pointers.shared_gate_up,
                shape.tokens,
                shape
                    .shared_intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("MoE shared gate/up width exceeds i32")
                    })?,
                shape.hidden_size,
                "vNext MoE shared gate/up GEMM",
            )?;
            let shared_activation_elements = u64::try_from(shape.tokens)
                .ok()
                .and_then(|tokens| tokens.checked_mul(shape.shared_intermediate_size as u64))
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "MoE shared activation element count overflows",
                    )
                })?;
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
                regions[8].device_ptr(),
                pointers.shared_output,
                shape.tokens,
                shape.hidden_size,
                shape.shared_intermediate_size,
                "vNext MoE shared down GEMM",
            )?;
            kernels.launch_token_gate_add(
                stream,
                regions[9].device_ptr(),
                pointers.shared_output,
                pointers.shared_gate,
                shape.tokens,
                shape.hidden_size,
            )?;
            Ok(())
        },
    )
    .map_err(|error| error.to_string())
}

#[allow(clippy::too_many_arguments)]
fn launch_marlin(
    stream: &cudarc::driver::CudaStream,
    input: u64,
    packed_weight: u64,
    scales: u64,
    output: u64,
    workspace: MoeWorkspacePointers,
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

fn resolve_shared_marlin_weight(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ordinal: u32,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinMoeWeight, String> {
    let first = &invocation.participants()[0];
    let resolved = resolve_gptq_marlin_moe_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
        logical_dimensions,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve_gptq_marlin_moe_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
            logical_dimensions,
        )?;
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
            shared_gate: pointer(layout.shared_gate)?,
            shared_gate_up: pointer(layout.shared_gate_up)?,
            shared_activation: pointer(layout.shared_activation)?,
            shared_output: pointer(layout.shared_output)?,
        })
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
            "CUDA MoE provider lacks unsigned attribute {name:?}"
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
        _ => Err(format!("CUDA MoE provider lacks bool attribute {name:?}")),
    }
}

fn checked_i32(value: u64, label: &str) -> Result<i32, String> {
    i32::try_from(value).map_err(|_| format!("{label} exceeds i32"))
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
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
}
