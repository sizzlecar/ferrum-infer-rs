//! CUDA provider for the backend-neutral routed-only SwiGLU MoE contract.

use std::collections::{BTreeMap, BTreeSet};

use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use ferrum_interfaces::vnext::{
    routed_swiglu_moe_contract, AttributeId, BatchedOperationInvocation, CapabilityId,
    ContractVersion, DeviceBatchingForm, DeviceRuntime, DynamicStorageRequirement, ElementType,
    EncodedDeviceOperation, OperationContract, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, ProfilePhase, ProviderId, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    QuantizationFormatId, ResolvedValueBinding, ResolvedValueRole, ReusableExecutionTopology,
    ReusableExecutionTopologyRequest, SemanticValue, VNextError, WeightFormatId,
    ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID, ROUTED_SWIGLU_MOE_OPERATION_ID,
};

use super::super::super::vnext_replay::CudaCommandReplayKeyBuilder;
use super::super::super::vnext_runtime::{
    CudaDeviceBuffer, CudaDeviceCommand, CudaDeviceRuntime, CudaDeviceRuntimeError,
};
use super::moe::{
    bool_attribute, checked_i32, invalid_plan, launch_marlin, resolve_shared_marlin_weight,
    unsigned_attribute, MarlinMoeWorkspacePointers, MoeRoutingPlan,
};
use super::moe_launch::{region_pointer, zero_region, MoeCudaKernels};
use super::moe_weights::{GPTQ_MARLIN_QUANTIZATION_FORMAT_ID, GPTQ_MARLIN_WEIGHT_FORMAT_ID};
use super::moe_workspace::{
    routed_workspace_formula_terms, MoeWorkspaceLayout, WorkspaceRegion, MAX_ROUTER_EXPERTS,
    MAX_ROUTER_TOP_K, MOE_BLOCK_SIZE,
};
use super::{
    contiguous_bindings, ensure_estimator_request, estimate, f16_contiguous, launch_gemm_f16,
    shared_full_region, shared_scratch_region, shared_token_region,
};
use crate::marlin_fp8_materializer::MARLIN_FP8_WEIGHT_FORMAT_ID;

const PROVIDER_ID: &str = "provider.cuda.routed_swiglu_moe.f16.gptq_marlin";
const ESTIMATOR_ID: &str = "resource-estimator.cuda.routed_swiglu_moe.f16.gptq_marlin";
const COMMAND_NAME: &str = "vnext_routed_swiglu_moe";
const VALUE_ALIGNMENT_BYTES: u64 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RoutedMoeAttributes {
    hidden_size: u64,
    expert_count: u64,
    experts_per_token: u64,
    routed_intermediate_size: u64,
    normalize_topk: bool,
}

impl RoutedMoeAttributes {
    fn from_values(attributes: &BTreeMap<AttributeId, SemanticValue>) -> Result<Self, String> {
        let values = Self {
            hidden_size: unsigned_attribute(attributes, "hidden_size")?,
            expert_count: unsigned_attribute(attributes, "expert_count")?,
            experts_per_token: unsigned_attribute(attributes, "experts_per_token")?,
            routed_intermediate_size: unsigned_attribute(attributes, "routed_intermediate_size")?,
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
        {
            return Err(format!(
                "CUDA routed-only MoE attributes are outside the current router contract: {self:?}"
            ));
        }
        let gate_up_width = self
            .routed_intermediate_size
            .checked_mul(2)
            .ok_or_else(|| "CUDA routed-only MoE gate/up width overflows".to_owned())?;
        if !self.hidden_size.is_multiple_of(64)
            || !self.routed_intermediate_size.is_multiple_of(64)
            || !gate_up_width.is_multiple_of(64)
        {
            return Err(format!(
                "CUDA routed-only Marlin-MoE widths must be divisible by 64: {self:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct RoutedMoeLaunchShape {
    tokens: i32,
    expert_count: i32,
    experts_per_token: i32,
    hidden_size: i32,
    routed_intermediate_size: i32,
    pair_count: i32,
    sorted_capacity: i32,
    normalize_topk: bool,
    gate_up_group_size: i32,
    down_group_size: i32,
    device_ordinal: i32,
}

pub(in crate::backend::cuda::vnext_ops) struct CudaRoutedSwiGluMoeProvider {
    descriptor: OperationProviderDescriptor,
    kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
}

impl CudaRoutedSwiGluMoeProvider {
    pub(in crate::backend::cuda::vnext_ops) fn new(
        runtime: &CudaDeviceRuntime,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = routed_swiglu_moe_contract().map_err(super::contract_error)?;
        let capability = CapabilityId::new(ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID)
            .map_err(super::contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "CUDA runtime does not advertise capability `{ROUTED_SWIGLU_MOE_F16_CAPABILITY_ID}`"
            )));
        }
        let provider_fingerprint = super::implementation_fingerprint(&[
            include_str!("moe_routed.rs").as_bytes(),
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
        ]);
        let estimator_fingerprint = super::implementation_fingerprint(&[
            include_str!("moe_workspace.rs").as_bytes(),
            ESTIMATOR_ID.as_bytes(),
            provider_fingerprint.as_bytes(),
        ]);
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new(PROVIDER_ID).map_err(super::contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(super::contract_error)?,
            provider_fingerprint,
            ferrum_interfaces::vnext::ProviderExecutionSemantics::bitwise_eager_and_replay(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            [GPTQ_MARLIN_WEIGHT_FORMAT_ID, MARLIN_FP8_WEIGHT_FORMAT_ID]
                .into_iter()
                .map(WeightFormatId::new)
                .collect::<Result<BTreeSet<_>, _>>()
                .map_err(super::contract_error)?,
            BTreeSet::from([
                QuantizationFormatId::new(GPTQ_MARLIN_QUANTIZATION_FORMAT_ID)
                    .map_err(super::contract_error)?,
            ]),
            contiguous_bindings(4),
            ESTIMATOR_ID,
            ContractVersion::new(1, 0),
            estimator_fingerprint,
        )
        .map_err(super::contract_error)?;
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

impl OperationResourceEstimator for CudaRoutedSwiGluMoeProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        ensure_estimator_request(&self.descriptor, &request, ROUTED_SWIGLU_MOE_OPERATION_ID)?;
        let attributes =
            RoutedMoeAttributes::from_values(request.attributes()).map_err(invalid_plan)?;
        let (fixed_bytes, bytes_per_token) = routed_workspace_formula_terms(
            attributes.expert_count,
            attributes.experts_per_token,
            attributes.hidden_size,
            attributes.routed_intermediate_size,
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

impl OperationProvider<CudaDeviceRuntime> for CudaRoutedSwiGluMoeProvider {
    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::Static)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_routed_moe(
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
                "cuda.routed_swiglu_moe.encode",
                message.chars().take(2048).collect::<String>(),
                false,
            )
            .expect("core-issued CUDA operation identity must form a valid provider failure")
        })
    }
}

fn encode_routed_moe(
    provider_fingerprint: &str,
    kernels: MoeCudaKernels,
    multiprocessor_count: u64,
    device_ordinal: i32,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.participants().is_empty()
        || invocation.operation().id.as_str() != ROUTED_SWIGLU_MOE_OPERATION_ID
    {
        return Err("CUDA routed-only MoE provider received another or empty operation".to_owned());
    }
    let first = &invocation.participants()[0];
    let attributes = RoutedMoeAttributes::from_values(first.attributes())?;
    let tokens = invocation.work_shape().immediate_tokens();
    if tokens == 0 {
        return Err("CUDA routed-only MoE invocation has no immediate tokens".to_owned());
    }
    for participant in invocation.participants() {
        if RoutedMoeAttributes::from_values(participant.attributes())? != attributes {
            return Err("CUDA routed-only MoE participant attributes disagree".to_owned());
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
            "CUDA routed-only MoE physical expert count differs from attributes".to_owned(),
        );
    }

    let layout = MoeWorkspaceLayout::routed_only(
        tokens,
        attributes.expert_count,
        attributes.experts_per_token,
        attributes.hidden_size,
        attributes.routed_intermediate_size,
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
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_scratch_region(&invocation, layout.total_bytes)?,
    ];
    let shape = RoutedMoeLaunchShape {
        tokens: checked_i32(tokens, "MoE token count")?,
        expert_count: checked_i32(attributes.expert_count, "MoE expert count")?,
        experts_per_token: checked_i32(attributes.experts_per_token, "MoE experts per token")?,
        hidden_size: checked_i32(attributes.hidden_size, "MoE hidden size")?,
        routed_intermediate_size: checked_i32(
            attributes.routed_intermediate_size,
            "MoE routed intermediate size",
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
    let routing_plan = MoeRoutingPlan::for_tokens(shape.tokens);
    let replay_key = CudaCommandReplayKeyBuilder::new(provider_fingerprint, COMMAND_NAME)
        .i32(shape.tokens)
        .i32(shape.expert_count)
        .i32(shape.experts_per_token)
        .i32(shape.hidden_size)
        .i32(shape.routed_intermediate_size)
        .boolean(shape.normalize_topk)
        .i32(shape.gate_up_group_size)
        .i32(shape.down_group_size)
        .i32(shape.device_ordinal)
        .bytes(routing_plan.replay_tag())
        .u64(layout.total_bytes)
        .u64(MOE_BLOCK_SIZE)
        .finish();
    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "CUDA routed-only MoE participant count exceeds u32".to_owned())?;

    CudaDeviceCommand::replayable_operation_with_blas(
        COMMAND_NAME,
        regions,
        replay_key,
        move |stream, blas, regions| {
            let scratch = &regions[7];
            if scratch.length_bytes() < layout.total_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "routed-only MoE scratch is smaller than its admitted estimate",
                ));
            }
            let pointers = RoutedMoeWorkspacePointers::new(scratch.device_ptr(), &layout)?;

            launch_gemm_f16(
                blas,
                regions[0].device_ptr(),
                regions[1].device_ptr(),
                pointers.router_logits,
                shape.tokens,
                shape.expert_count,
                shape.hidden_size,
                "vNext routed-only MoE router GEMM",
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
            launch_marlin(
                stream,
                regions[0].device_ptr(),
                regions[2].device_ptr(),
                regions[3].device_ptr(),
                pointers.routed_gate_up,
                pointers.marlin(),
                shape.tokens,
                shape.experts_per_token,
                shape
                    .routed_intermediate_size
                    .checked_mul(2)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "routed-only MoE gate/up width exceeds i32",
                        )
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
                            "routed-only MoE activation element count overflows",
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
                regions[6].device_ptr(),
                shape.tokens,
                shape.experts_per_token,
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
            routing_plan.routed_compute_dispatch_count(),
            2,
        )
    })
    .map_err(|error| error.to_string())
}

fn validate_participant(
    bindings: &[ResolvedValueBinding],
    attributes: RoutedMoeAttributes,
) -> Result<(), String> {
    let input = super::binding(bindings, ResolvedValueRole::Input, 0)?;
    let [canonical_tokens, input_hidden] = input.tensor().dimensions() else {
        return Err("CUDA routed-only MoE input is not two-dimensional".to_owned());
    };
    if *input_hidden != attributes.hidden_size || !f16_contiguous(input) {
        return Err(
            "CUDA routed-only MoE input differs from [tokens, hidden] F16 contiguous".to_owned(),
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
    ];
    for (ordinal, dimensions) in expected {
        let value = super::binding(bindings, ResolvedValueRole::Input, ordinal)?;
        if value.tensor().dimensions() != dimensions || !f16_contiguous(value) {
            return Err(format!(
                "CUDA routed-only MoE input {ordinal} differs from shape {dimensions:?} F16 contiguous"
            ));
        }
    }
    let output = super::binding(bindings, ResolvedValueRole::Output, 0)?;
    if output.tensor().dimensions() != [*canonical_tokens, attributes.hidden_size]
        || !f16_contiguous(output)
    {
        return Err(
            "CUDA routed-only MoE output differs from [tokens, hidden] F16 contiguous".to_owned(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct RoutedMoeWorkspacePointers {
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
}

impl RoutedMoeWorkspacePointers {
    fn new(base: u64, layout: &MoeWorkspaceLayout) -> Result<Self, CudaDeviceRuntimeError> {
        if layout.shared().is_some() {
            return Err(CudaDeviceRuntimeError::contract(
                "routed-only MoE provider received a shared-expert workspace",
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn attributes(values: [(&str, SemanticValue); 5]) -> BTreeMap<AttributeId, SemanticValue> {
        values
            .into_iter()
            .map(|(name, value)| (AttributeId::new(name).unwrap(), value))
            .collect()
    }

    #[test]
    fn parses_qwen3_moe_shape_without_shared_expert_identity() {
        let parsed = RoutedMoeAttributes::from_values(&attributes([
            ("hidden_size", SemanticValue::Unsigned(2048)),
            ("expert_count", SemanticValue::Unsigned(128)),
            ("experts_per_token", SemanticValue::Unsigned(8)),
            ("routed_intermediate_size", SemanticValue::Unsigned(768)),
            ("normalize_topk", SemanticValue::Bool(true)),
        ]))
        .unwrap();
        assert_eq!(parsed.expert_count, 128);
        assert_eq!(parsed.experts_per_token, 8);
        assert_eq!(parsed.routed_intermediate_size, 768);
    }

    #[test]
    fn rejects_shared_attribute_in_routed_only_shape() {
        let mut values = attributes([
            ("hidden_size", SemanticValue::Unsigned(2048)),
            ("expert_count", SemanticValue::Unsigned(128)),
            ("experts_per_token", SemanticValue::Unsigned(8)),
            ("routed_intermediate_size", SemanticValue::Unsigned(768)),
            ("normalize_topk", SemanticValue::Bool(true)),
        ]);
        values.insert(
            AttributeId::new("shared_intermediate_size").unwrap(),
            SemanticValue::Unsigned(768),
        );
        let contract = routed_swiglu_moe_contract().unwrap();
        assert!(contract
            .descriptor()
            .attributes
            .validate_values(&values, "routed-only MoE test attributes")
            .is_err());
    }
}
