//! Native Metal provider for the routed/shared SwiGLU MoE contract.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    routed_shared_swiglu_moe_contract, AttributeId, BatchedOperationInvocation, DeviceBatchingForm,
    DynamicStorageRequirement, ElementType, EncodedDeviceOperation, OperationFailure,
    OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, PhysicalWeightPadding,
    ProviderWorkspaceRequirement, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    ResolvedValueBinding, ResolvedValueRole, SemanticValue, VNextError, WeightEncoding,
    ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID, ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID,
};
use metal::{CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError,
};
use super::linear::{
    append_shared_gate_up_weight, append_shared_matrix_weight, dispatch_linear, dispatch_swiglu,
    linear_launch, same_resolved_weight, swiglu_launch, validate_launch_regions,
    MetalLinearPipelines,
};
use super::weights::{
    resolve_weight, MetalResolvedWeight, MetalResolvedWeightComponent, MetalResolvedWeightLayout,
};
use super::{
    binding, checked_u32, contiguous_bindings, ensure_invocation, f16_contiguous,
    implementation_fingerprint, invalid_plan, provider_descriptor, provider_failure,
    shared_scratch_region, shared_token_region, GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID,
    Q8_0_FORMAT_ID, THREADS_PER_GROUP, VALUE_ALIGNMENT_BYTES,
};

const SHADER_SOURCE: &str = include_str!("moe.metal");
const PROVIDER_ID: &str = "provider.metal.routed_shared_swiglu_moe.f16.q4k";
const ESTIMATOR_ID: &str = "resource-estimator.metal.routed_shared_swiglu_moe.f16.q4k";
const ROUTE_KERNEL: &str = "vnext_moe_route_topk_f16";
const ROUTED_GATE_UP_KERNEL: &str = "vnext_moe_q4k_gate_up_silu_f16";
const ROUTED_DOWN_KERNEL: &str = "vnext_moe_q4k_down_f16";
const COMBINE_KERNEL: &str = "vnext_moe_combine_f16";
const Q4_K_VALUES_PER_BLOCK: u64 = 256;
const Q4_K_BYTES_PER_BLOCK: u64 = 144;
const MAX_ROUTER_EXPERTS: u64 = 256;
const MAX_ROUTER_TOP_K: u64 = 32;
const WORKSPACE_REGION_COUNT: u64 = 9;

pub(super) struct MetalMoePipelines {
    route: ComputePipelineState,
    routed_gate_up: ComputePipelineState,
    routed_down: ComputePipelineState,
    combine: ComputePipelineState,
}

impl MetalMoePipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile Metal vNext MoE library: {error}"
                ))
            })?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!("load Metal vNext MoE `{name}`: {error}"))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext MoE `{name}`: {error}"
                    ))
                })
        };
        Ok(Self {
            route: pipeline(ROUTE_KERNEL)?,
            routed_gate_up: pipeline(ROUTED_GATE_UP_KERNEL)?,
            routed_down: pipeline(ROUTED_DOWN_KERNEL)?,
            combine: pipeline(COMBINE_KERNEL)?,
        })
    }
}

pub(super) struct MetalRoutedSharedSwiGluMoeProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalMoePipelines>,
    linear_pipelines: Arc<MetalLinearPipelines>,
}

impl MetalRoutedSharedSwiGluMoeProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalMoePipelines>,
        linear_pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = routed_shared_swiglu_moe_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            PROVIDER_ID,
            ROUTED_SHARED_SWIGLU_MOE_F16_CAPABILITY_ID,
            ESTIMATOR_ID,
            contiguous_bindings(7),
            &[GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[Q4_K_FORMAT_ID, Q8_0_FORMAT_ID],
            implementation_fingerprint(&[
                include_str!("moe.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                include_str!("linear.rs").as_bytes(),
                include_str!("linear.metal").as_bytes(),
                PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
            linear_pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalRoutedSharedSwiGluMoeProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "Metal estimator `{}` received another operation",
                self.descriptor.resource_estimator_id()
            )));
        }
        let attributes = MoeAttributes::from_values(request.attributes()).map_err(invalid_plan)?;
        let (fixed_bytes, bytes_per_token) =
            workspace_formula_terms(attributes).map_err(invalid_plan)?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(fixed_bytes, 0, bytes_per_token)?,
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

impl OperationProvider<MetalDeviceRuntime> for MetalRoutedSharedSwiGluMoeProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_moe(
            Arc::clone(&self.pipelines),
            Arc::clone(&self.linear_pipelines),
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "metal.routed_shared_swiglu_moe.encode", message)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            || !self.hidden_size.is_multiple_of(Q4_K_VALUES_PER_BLOCK)
            || !self
                .routed_intermediate_size
                .is_multiple_of(Q4_K_VALUES_PER_BLOCK)
            || !self.shared_intermediate_size.is_multiple_of(32)
        {
            return Err(format!(
                "Metal Q4_K/Q8_0 MoE attributes are unsupported: {self:?}"
            ));
        }
        for (value, name) in [
            (self.hidden_size, "hidden size"),
            (self.expert_count, "expert count"),
            (self.experts_per_token, "experts per token"),
            (self.routed_intermediate_size, "routed intermediate size"),
            (self.shared_intermediate_size, "shared intermediate size"),
        ] {
            checked_u32(value, &format!("Metal MoE {name}"))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WorkspaceRegion {
    offset_bytes: u64,
    length_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MoeWorkspaceLayout {
    router_logits: WorkspaceRegion,
    route_ids: WorkspaceRegion,
    route_weights: WorkspaceRegion,
    routed_activation: WorkspaceRegion,
    routed_down_slots: WorkspaceRegion,
    shared_gate: WorkspaceRegion,
    shared_gate_up: WorkspaceRegion,
    shared_activation: WorkspaceRegion,
    shared_output: WorkspaceRegion,
    total_bytes: u64,
    pair_count: u64,
}

impl MoeWorkspaceLayout {
    fn new(tokens: u64, attributes: MoeAttributes) -> Result<Self, String> {
        if tokens == 0 {
            return Err("Metal MoE workspace requires at least one token".to_owned());
        }
        attributes.validate()?;
        let pair_count = checked_mul(
            tokens,
            attributes.experts_per_token,
            "Metal MoE routed pair count",
        )?;
        let mut cursor = WorkspaceCursor::default();
        let router_logits = cursor.allocate(elements_bytes(
            checked_mul(tokens, attributes.expert_count, "Metal MoE router logits")?,
            ElementType::F16.size_bytes(),
            "Metal MoE router logits",
        )?)?;
        let route_ids = cursor.allocate(elements_bytes(
            pair_count,
            ElementType::I32.size_bytes(),
            "Metal MoE route IDs",
        )?)?;
        let route_weights = cursor.allocate(elements_bytes(
            pair_count,
            ElementType::F32.size_bytes(),
            "Metal MoE route weights",
        )?)?;
        let routed_activation = cursor.allocate(elements_bytes(
            checked_mul(
                pair_count,
                attributes.routed_intermediate_size,
                "Metal MoE routed activation",
            )?,
            ElementType::F16.size_bytes(),
            "Metal MoE routed activation",
        )?)?;
        let routed_down_slots = cursor.allocate(elements_bytes(
            checked_mul(
                pair_count,
                attributes.hidden_size,
                "Metal MoE routed down slots",
            )?,
            ElementType::F16.size_bytes(),
            "Metal MoE routed down slots",
        )?)?;
        let shared_gate = cursor.allocate(elements_bytes(
            tokens,
            ElementType::F16.size_bytes(),
            "Metal MoE shared gate",
        )?)?;
        let shared_gate_up = cursor.allocate(elements_bytes(
            checked_mul(
                checked_mul(
                    tokens,
                    attributes.shared_intermediate_size,
                    "Metal MoE shared gate/up",
                )?,
                2,
                "Metal MoE shared gate/up",
            )?,
            ElementType::F16.size_bytes(),
            "Metal MoE shared gate/up",
        )?)?;
        let shared_activation = cursor.allocate(elements_bytes(
            checked_mul(
                tokens,
                attributes.shared_intermediate_size,
                "Metal MoE shared activation",
            )?,
            ElementType::F16.size_bytes(),
            "Metal MoE shared activation",
        )?)?;
        let shared_output = cursor.allocate(elements_bytes(
            checked_mul(tokens, attributes.hidden_size, "Metal MoE shared output")?,
            ElementType::F16.size_bytes(),
            "Metal MoE shared output",
        )?)?;
        let total_bytes = align_up(cursor.offset, VALUE_ALIGNMENT_BYTES)?;
        let (fixed_bytes, bytes_per_token) = workspace_formula_terms(attributes)?;
        let admitted = fixed_bytes
            .checked_add(checked_mul(
                bytes_per_token,
                tokens,
                "Metal MoE admitted workspace",
            )?)
            .ok_or_else(|| "Metal MoE admitted workspace overflows".to_owned())?;
        if total_bytes > admitted {
            return Err(format!(
                "Metal MoE workspace {total_bytes} exceeds affine estimate {admitted}"
            ));
        }
        Ok(Self {
            router_logits,
            route_ids,
            route_weights,
            routed_activation,
            routed_down_slots,
            shared_gate,
            shared_gate_up,
            shared_activation,
            shared_output,
            total_bytes,
            pair_count,
        })
    }
}

#[derive(Default)]
struct WorkspaceCursor {
    offset: u64,
}

impl WorkspaceCursor {
    fn allocate(&mut self, length_bytes: u64) -> Result<WorkspaceRegion, String> {
        if length_bytes == 0 {
            return Err("Metal MoE workspace region is empty".to_owned());
        }
        self.offset = align_up(self.offset, VALUE_ALIGNMENT_BYTES)?;
        let region = WorkspaceRegion {
            offset_bytes: self.offset,
            length_bytes,
        };
        self.offset = self
            .offset
            .checked_add(length_bytes)
            .ok_or_else(|| "Metal MoE workspace offset overflows".to_owned())?;
        Ok(region)
    }
}

fn workspace_formula_terms(attributes: MoeAttributes) -> Result<(u64, u64), String> {
    attributes.validate()?;
    let routed_activation = checked_mul(
        checked_mul(
            attributes.experts_per_token,
            attributes.routed_intermediate_size,
            "Metal MoE routed activation bytes per token",
        )?,
        ElementType::F16.size_bytes(),
        "Metal MoE routed activation bytes per token",
    )?;
    let routed_down = checked_mul(
        checked_mul(
            attributes.experts_per_token,
            attributes.hidden_size,
            "Metal MoE routed down bytes per token",
        )?,
        ElementType::F16.size_bytes(),
        "Metal MoE routed down bytes per token",
    )?;
    let shared_gate_up = checked_mul(
        checked_mul(
            attributes.shared_intermediate_size,
            2,
            "Metal MoE shared gate/up bytes per token",
        )?,
        ElementType::F16.size_bytes(),
        "Metal MoE shared gate/up bytes per token",
    )?;
    let terms = [
        checked_mul(
            attributes.expert_count,
            ElementType::F16.size_bytes(),
            "Metal MoE router bytes per token",
        )?,
        checked_mul(
            attributes.experts_per_token,
            ElementType::I32.size_bytes(),
            "Metal MoE route ID bytes per token",
        )?,
        checked_mul(
            attributes.experts_per_token,
            ElementType::F32.size_bytes(),
            "Metal MoE route weight bytes per token",
        )?,
        routed_activation,
        routed_down,
        ElementType::F16.size_bytes(),
        shared_gate_up,
        checked_mul(
            attributes.shared_intermediate_size,
            ElementType::F16.size_bytes(),
            "Metal MoE shared activation bytes per token",
        )?,
        checked_mul(
            attributes.hidden_size,
            ElementType::F16.size_bytes(),
            "Metal MoE shared output bytes per token",
        )?,
    ];
    let bytes_per_token = terms.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| "Metal MoE bytes per token overflow".to_owned())
    })?;
    let fixed_bytes = checked_mul(
        WORKSPACE_REGION_COUNT + 1,
        VALUE_ALIGNMENT_BYTES,
        "Metal MoE workspace alignment reserve",
    )?;
    Ok((fixed_bytes, bytes_per_token))
}

#[derive(Debug, Clone, Copy)]
struct Q4ExpertPart {
    region: usize,
    row_stride_bytes: u32,
    expert_stride_bytes: u32,
}

#[derive(Debug, Clone, Copy)]
struct RoutedGateUp {
    gate: Q4ExpertPart,
    up: Q4ExpertPart,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RouteParams {
    tokens: u32,
    expert_count: u32,
    experts_per_token: u32,
    normalize_topk: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Q4ExpertParams {
    output_features: u32,
    input_features: u32,
    row_stride_bytes: u32,
    expert_stride_bytes: u32,
    expert_count: u32,
    experts_per_token: u32,
    pair_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CombineParams {
    tokens: u32,
    experts_per_token: u32,
    hidden_size: u32,
}

fn encode_moe(
    pipelines: Arc<MetalMoePipelines>,
    linear_pipelines: Arc<MetalLinearPipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, ROUTED_SHARED_SWIGLU_MOE_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let attributes = MoeAttributes::from_values(first.attributes())?;
    let tokens = invocation.work_shape().immediate_tokens();
    if tokens == 0 {
        return Err("Metal routed/shared MoE invocation has no immediate tokens".to_owned());
    }
    for participant in invocation.participants() {
        if MoeAttributes::from_values(participant.attributes())? != attributes {
            return Err("Metal routed/shared MoE participant attributes disagree".to_owned());
        }
        validate_participant(participant.bindings(), attributes)?;
    }

    let layout = MoeWorkspaceLayout::new(tokens, attributes)?;
    let mut regions = Vec::new();
    let routed_gate_up = append_routed_gate_up(&mut regions, &invocation, attributes)?;
    let routed_down = append_routed_down(&mut regions, &invocation, attributes)?;
    let router = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        1,
        attributes.expert_count,
        attributes.hidden_size,
        "Metal MoE router",
    )?;
    let shared_gate = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        4,
        1,
        attributes.hidden_size,
        "Metal MoE shared gate",
    )?;
    let shared_gate_up = append_shared_gate_up_weight(
        &mut regions,
        &invocation,
        5,
        attributes.shared_intermediate_size,
        attributes.hidden_size,
        "Metal MoE shared gate/up",
    )?;
    let shared_down = append_shared_matrix_weight(
        &mut regions,
        &invocation,
        6,
        attributes.hidden_size,
        attributes.shared_intermediate_size,
        "Metal MoE shared down",
    )?;

    let input_region = regions.len();
    regions.push(shared_token_region(
        &invocation,
        ResolvedValueRole::Input,
        0,
        ElementType::F16,
        tokens,
    )?);
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

    let router_launch = linear_launch(
        router,
        input_region,
        scratch_region,
        tokens,
        attributes.hidden_size,
        attributes.expert_count,
        0,
        layout.router_logits.offset_bytes,
    )?;
    let shared_gate_launch = linear_launch(
        shared_gate,
        input_region,
        scratch_region,
        tokens,
        attributes.hidden_size,
        1,
        0,
        layout.shared_gate.offset_bytes,
    )?;
    let shared_gate_up_width = checked_mul(
        attributes.shared_intermediate_size,
        2,
        "Metal MoE shared gate/up width",
    )?;
    let shared_gate_up_launches = shared_gate_up
        .into_iter()
        .map(|part| {
            linear_launch(
                part,
                input_region,
                scratch_region,
                tokens,
                attributes.hidden_size,
                shared_gate_up_width,
                0,
                layout.shared_gate_up.offset_bytes,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shared_down_launch = linear_launch(
        shared_down,
        scratch_region,
        scratch_region,
        tokens,
        attributes.shared_intermediate_size,
        attributes.hidden_size,
        layout.shared_activation.offset_bytes,
        layout.shared_output.offset_bytes,
    )?;
    let mut linear_launches = Vec::with_capacity(shared_gate_up_launches.len() + 3);
    linear_launches.push(router_launch);
    linear_launches.push(shared_gate_launch);
    linear_launches.extend(shared_gate_up_launches.iter().copied());
    linear_launches.push(shared_down_launch);
    validate_launch_regions(&regions, &linear_launches)?;
    validate_workspace_regions(&regions[scratch_region], &layout)?;
    let shared_swiglu = swiglu_launch(
        layout.shared_gate_up.offset_bytes,
        layout.shared_activation.offset_bytes,
        tokens,
        attributes.shared_intermediate_size,
        shared_gate_up_width,
    )?;

    let route_params = RouteParams {
        tokens: checked_u32(tokens, "Metal MoE token count")?,
        expert_count: checked_u32(attributes.expert_count, "Metal MoE expert count")?,
        experts_per_token: checked_u32(
            attributes.experts_per_token,
            "Metal MoE experts per token",
        )?,
        normalize_topk: u32::from(attributes.normalize_topk),
    };
    let gate_up_params = Q4ExpertParams {
        output_features: checked_u32(
            attributes.routed_intermediate_size,
            "Metal MoE routed intermediate size",
        )?,
        input_features: checked_u32(attributes.hidden_size, "Metal MoE hidden size")?,
        row_stride_bytes: routed_gate_up.gate.row_stride_bytes,
        expert_stride_bytes: routed_gate_up.gate.expert_stride_bytes,
        expert_count: route_params.expert_count,
        experts_per_token: route_params.experts_per_token,
        pair_count: checked_u32(layout.pair_count, "Metal MoE routed pair count")?,
    };
    if routed_gate_up.gate.row_stride_bytes != routed_gate_up.up.row_stride_bytes
        || routed_gate_up.gate.expert_stride_bytes != routed_gate_up.up.expert_stride_bytes
    {
        return Err("Metal MoE gate/up Q4_K physical strides disagree".to_owned());
    }
    let down_params = Q4ExpertParams {
        output_features: checked_u32(attributes.hidden_size, "Metal MoE hidden size")?,
        input_features: checked_u32(
            attributes.routed_intermediate_size,
            "Metal MoE routed intermediate size",
        )?,
        row_stride_bytes: routed_down.row_stride_bytes,
        expert_stride_bytes: routed_down.expert_stride_bytes,
        expert_count: route_params.expert_count,
        experts_per_token: route_params.experts_per_token,
        pair_count: gate_up_params.pair_count,
    };
    let combine_params = CombineParams {
        tokens: route_params.tokens,
        experts_per_token: route_params.experts_per_token,
        hidden_size: down_params.output_features,
    };
    let router_threadgroup_bytes = router_threadgroup_bytes(attributes)?;
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal MoE participant count",
    )?;
    let dispatch_count = 8_u64
        .checked_add(shared_gate_up_launches.len() as u64)
        .ok_or_else(|| "Metal MoE dispatch count overflows".to_owned())?;

    MetalDeviceCommand::operation(
        "vnext_routed_shared_swiglu_moe",
        regions,
        move |encoder, regions| {
            encoder.record_compute_dispatches(dispatch_count);
            let compute = encoder.compute_encoder();
            dispatch_linear(&linear_pipelines, compute, regions, router_launch);
            dispatch_route(
                &pipelines,
                compute,
                &regions[scratch_region],
                layout.router_logits,
                layout.route_ids,
                layout.route_weights,
                route_params,
                router_threadgroup_bytes,
            );
            dispatch_routed_gate_up(
                &pipelines,
                compute,
                regions,
                routed_gate_up,
                input_region,
                scratch_region,
                &layout,
                gate_up_params,
            );
            dispatch_routed_down(
                &pipelines,
                compute,
                regions,
                routed_down,
                scratch_region,
                &layout,
                down_params,
            );
            dispatch_linear(&linear_pipelines, compute, regions, shared_gate_launch);
            for launch in &shared_gate_up_launches {
                dispatch_linear(&linear_pipelines, compute, regions, *launch);
            }
            dispatch_swiglu(
                &linear_pipelines,
                compute,
                &regions[scratch_region],
                shared_swiglu,
            );
            dispatch_linear(&linear_pipelines, compute, regions, shared_down_launch);
            dispatch_combine(
                &pipelines,
                compute,
                regions,
                scratch_region,
                output_region,
                &layout,
                combine_params,
            );
            Ok(())
        },
    )
    .map_err(|error| error.to_string())?
    .with_work_shape(
        if participant_count == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::Packed
        },
        participant_count,
        tokens,
    )
    .map_err(|error| error.to_string())
}

fn dispatch_route(
    pipelines: &MetalMoePipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    logits: WorkspaceRegion,
    ids: WorkspaceRegion,
    weights: WorkspaceRegion,
    params: RouteParams,
    threadgroup_bytes: u64,
) {
    encoder.set_compute_pipeline_state(&pipelines.route);
    set_region_offset(encoder, 0, scratch, logits.offset_bytes);
    set_region_offset(encoder, 1, scratch, ids.offset_bytes);
    set_region_offset(encoder, 2, scratch, weights.offset_bytes);
    encoder.set_bytes(
        3,
        std::mem::size_of::<RouteParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.set_threadgroup_memory_length(0, threadgroup_bytes);
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(params.tokens), 1, 1),
        MTLSize::new(32, 1, 1),
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_routed_gate_up(
    pipelines: &MetalMoePipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    weight: RoutedGateUp,
    input_region: usize,
    scratch_region: usize,
    layout: &MoeWorkspaceLayout,
    params: Q4ExpertParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.routed_gate_up);
    set_region_offset(encoder, 0, &regions[weight.gate.region], 0);
    set_region_offset(encoder, 1, &regions[weight.up.region], 0);
    set_region_offset(encoder, 2, &regions[input_region], 0);
    set_region_offset(
        encoder,
        3,
        &regions[scratch_region],
        layout.route_ids.offset_bytes,
    );
    set_region_offset(
        encoder,
        4,
        &regions[scratch_region],
        layout.routed_activation.offset_bytes,
    );
    encoder.set_bytes(
        5,
        std::mem::size_of::<Q4ExpertParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(params.output_features).div_ceil(4),
            1,
            u64::from(params.pair_count),
        ),
        MTLSize::new(32, 2, 1),
    );
}

fn dispatch_routed_down(
    pipelines: &MetalMoePipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    weight: Q4ExpertPart,
    scratch_region: usize,
    layout: &MoeWorkspaceLayout,
    params: Q4ExpertParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.routed_down);
    set_region_offset(encoder, 0, &regions[weight.region], 0);
    set_region_offset(
        encoder,
        1,
        &regions[scratch_region],
        layout.routed_activation.offset_bytes,
    );
    set_region_offset(
        encoder,
        2,
        &regions[scratch_region],
        layout.route_ids.offset_bytes,
    );
    set_region_offset(
        encoder,
        3,
        &regions[scratch_region],
        layout.routed_down_slots.offset_bytes,
    );
    encoder.set_bytes(
        4,
        std::mem::size_of::<Q4ExpertParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(params.output_features).div_ceil(4),
            1,
            u64::from(params.pair_count),
        ),
        MTLSize::new(32, 2, 1),
    );
}

fn dispatch_combine(
    pipelines: &MetalMoePipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    scratch_region: usize,
    output_region: usize,
    layout: &MoeWorkspaceLayout,
    params: CombineParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.combine);
    set_region_offset(
        encoder,
        0,
        &regions[scratch_region],
        layout.routed_down_slots.offset_bytes,
    );
    set_region_offset(
        encoder,
        1,
        &regions[scratch_region],
        layout.route_weights.offset_bytes,
    );
    set_region_offset(
        encoder,
        2,
        &regions[scratch_region],
        layout.shared_gate.offset_bytes,
    );
    set_region_offset(
        encoder,
        3,
        &regions[scratch_region],
        layout.shared_output.offset_bytes,
    );
    set_region_offset(encoder, 4, &regions[output_region], 0);
    encoder.set_bytes(
        5,
        std::mem::size_of::<CombineParams>() as u64,
        &params as *const _ as *const c_void,
    );
    let elements = u64::from(params.tokens) * u64::from(params.hidden_size);
    encoder.dispatch_thread_groups(
        MTLSize::new(elements.div_ceil(THREADS_PER_GROUP), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn set_region_offset(
    encoder: &ComputeCommandEncoderRef,
    index: u64,
    region: &MetalBufferRegion,
    offset_bytes: u64,
) {
    encoder.set_buffer(
        index,
        Some(region.buffer()),
        region.offset_bytes() + offset_bytes,
    );
}

fn append_routed_gate_up(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    attributes: MoeAttributes,
) -> Result<RoutedGateUp, String> {
    let resolved = resolve_shared_weight(invocation, 2, "Metal MoE routed gate/up")?;
    if resolved.format_id().as_str() != GGUF_NATIVE_BLOCK_FORMAT_ID
        || resolved.logical_element_type() != ElementType::F16
        || resolved.logical_dimensions()
            != [
                attributes.expert_count,
                2,
                attributes.routed_intermediate_size,
                attributes.hidden_size,
            ]
    {
        return Err("Metal MoE routed gate/up logical weight differs".to_owned());
    }
    let (prepared_regions, components, layout) = resolved.into_command_parts();
    let MetalResolvedWeightLayout::Composite { parts } = layout else {
        return Err("Metal MoE routed gate/up must be a two-part physical composite".to_owned());
    };
    if parts.len() != 2 {
        return Err("Metal MoE routed gate/up physical composite must have two parts".to_owned());
    }
    let mut prepared = [None, None];
    for part in parts {
        if part.logical_offsets.len() != 4
            || part.extents
                != [
                    attributes.expert_count,
                    1,
                    attributes.routed_intermediate_size,
                    attributes.hidden_size,
                ]
            || part.logical_offsets[0] != 0
            || part.logical_offsets[2..] != [0, 0]
            || part.logical_offsets[1] > 1
        {
            return Err("Metal MoE routed gate/up partition geometry differs".to_owned());
        }
        let slot = part.logical_offsets[1] as usize;
        if prepared[slot].is_some() {
            return Err("Metal MoE routed gate/up partitions overlap".to_owned());
        }
        prepared[slot] = Some(prepare_q4_expert_part(
            &prepared_regions,
            &components,
            &part.layout,
            attributes.expert_count,
            attributes.routed_intermediate_size,
            attributes.hidden_size,
            3,
            "Metal MoE routed gate/up",
        )?);
    }
    let [Some(mut gate), Some(mut up)] = prepared else {
        return Err("Metal MoE routed gate/up partitions leave a gap".to_owned());
    };
    let base = regions.len();
    gate.region = gate
        .region
        .checked_add(base)
        .ok_or_else(|| "Metal MoE gate region index overflows".to_owned())?;
    up.region = up
        .region
        .checked_add(base)
        .ok_or_else(|| "Metal MoE up region index overflows".to_owned())?;
    regions.extend(prepared_regions);
    Ok(RoutedGateUp { gate, up })
}

fn append_routed_down(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    attributes: MoeAttributes,
) -> Result<Q4ExpertPart, String> {
    let resolved = resolve_shared_weight(invocation, 3, "Metal MoE routed down")?;
    if resolved.format_id().as_str() != GGUF_NATIVE_BLOCK_FORMAT_ID
        || resolved.logical_element_type() != ElementType::F16
        || resolved.logical_dimensions()
            != [
                attributes.expert_count,
                attributes.hidden_size,
                attributes.routed_intermediate_size,
            ]
    {
        return Err("Metal MoE routed down logical weight differs".to_owned());
    }
    let (prepared_regions, components, layout) = resolved.into_command_parts();
    let mut part = prepare_q4_expert_part(
        &prepared_regions,
        &components,
        &layout,
        attributes.expert_count,
        attributes.hidden_size,
        attributes.routed_intermediate_size,
        2,
        "Metal MoE routed down",
    )?;
    part.region = part
        .region
        .checked_add(regions.len())
        .ok_or_else(|| "Metal MoE down region index overflows".to_owned())?;
    regions.extend(prepared_regions);
    Ok(part)
}

#[allow(clippy::too_many_arguments)]
fn prepare_q4_expert_part(
    regions: &[MetalBufferRegion],
    components: &[MetalResolvedWeightComponent],
    layout: &MetalResolvedWeightLayout,
    expert_count: u64,
    output_features: u64,
    input_features: u64,
    expected_block_axis: u32,
    context: &str,
) -> Result<Q4ExpertPart, String> {
    let MetalResolvedWeightLayout::BlockQuantized {
        component,
        spec,
        block_axis,
        block_padding,
    } = layout
    else {
        return Err(format!("{context} is not one Q4_K physical block stack"));
    };
    if spec.format_id.as_str() != Q4_K_FORMAT_ID
        || spec.logical_values_per_block != Q4_K_VALUES_PER_BLOCK as u32
        || spec.bytes_per_block != Q4_K_BYTES_PER_BLOCK as u32
        || *block_axis != expected_block_axis
        || block_padding != &PhysicalWeightPadding::Exact
        || !input_features.is_multiple_of(Q4_K_VALUES_PER_BLOCK)
    {
        return Err(format!("{context} Q4_K physical ABI differs"));
    }
    let metadata = components
        .get(*component)
        .ok_or_else(|| format!("{context} component metadata is absent"))?;
    if metadata.encoding() != &WeightEncoding::BlockQuantized(spec.clone()) {
        return Err(format!("{context} component encoding differs"));
    }
    let blocks_per_row = input_features / Q4_K_VALUES_PER_BLOCK;
    let dimensions = metadata.physical_dimensions();
    let element_count = dimensions.iter().try_fold(1_u64, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| format!("{context} physical shape overflows"))
    })?;
    if dimensions.first() != Some(&expert_count)
        || dimensions.last() != Some(&blocks_per_row)
        || element_count
            != expert_count
                .checked_mul(output_features)
                .and_then(|value| value.checked_mul(blocks_per_row))
                .ok_or_else(|| format!("{context} physical element count overflows"))?
    {
        return Err(format!(
            "{context} physical stack is not expert-major row-major Q4_K"
        ));
    }
    let region = regions
        .get(*component)
        .ok_or_else(|| format!("{context} physical region is absent"))?;
    let row_stride_bytes = checked_mul(
        blocks_per_row,
        Q4_K_BYTES_PER_BLOCK,
        &format!("{context} row stride"),
    )?;
    let expert_stride_bytes = checked_mul(
        output_features,
        row_stride_bytes,
        &format!("{context} expert stride"),
    )?;
    let expected_bytes = checked_mul(
        expert_count,
        expert_stride_bytes,
        &format!("{context} total bytes"),
    )?;
    if region.length_bytes() != expected_bytes {
        return Err(format!("{context} physical byte length differs"));
    }
    Ok(Q4ExpertPart {
        region: *component,
        row_stride_bytes: checked_u32(row_stride_bytes, &format!("{context} row stride"))?,
        expert_stride_bytes: checked_u32(expert_stride_bytes, &format!("{context} expert stride"))?,
    })
}

fn resolve_shared_weight(
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
    context: &str,
) -> Result<MetalResolvedWeight, String> {
    let first = &invocation.participants()[0];
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err(format!(
                "{context} is not one shared physical weight across participants"
            ));
        }
    }
    Ok(resolved)
}

fn validate_participant(
    bindings: &[ResolvedValueBinding],
    attributes: MoeAttributes,
) -> Result<(), String> {
    let input = binding(bindings, ResolvedValueRole::Input, 0)?;
    let [canonical_tokens, hidden_size] = input.tensor().dimensions() else {
        return Err("Metal routed/shared MoE input is not two-dimensional".to_owned());
    };
    if *hidden_size != attributes.hidden_size || !f16_contiguous(input) {
        return Err(
            "Metal routed/shared MoE input differs from [tokens, hidden] F16 contiguous".to_owned(),
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
                "Metal routed/shared MoE input {ordinal} differs from shape {dimensions:?} F16 contiguous"
            ));
        }
    }
    let output = binding(bindings, ResolvedValueRole::Output, 0)?;
    if output.tensor().dimensions() != [*canonical_tokens, attributes.hidden_size]
        || !f16_contiguous(output)
    {
        return Err(
            "Metal routed/shared MoE output differs from [tokens, hidden] F16 contiguous"
                .to_owned(),
        );
    }
    Ok(())
}

fn validate_workspace_regions(
    scratch: &MetalBufferRegion,
    layout: &MoeWorkspaceLayout,
) -> Result<(), String> {
    for (name, region) in [
        ("router logits", layout.router_logits),
        ("route IDs", layout.route_ids),
        ("route weights", layout.route_weights),
        ("routed activation", layout.routed_activation),
        ("routed down slots", layout.routed_down_slots),
        ("shared gate", layout.shared_gate),
        ("shared gate/up", layout.shared_gate_up),
        ("shared activation", layout.shared_activation),
        ("shared output", layout.shared_output),
    ] {
        if region.length_bytes == 0
            || region
                .offset_bytes
                .checked_add(region.length_bytes)
                .is_none_or(|end| end > scratch.length_bytes())
        {
            return Err(format!("Metal MoE {name} exceeds admitted scratch"));
        }
    }
    Ok(())
}

fn router_threadgroup_bytes(attributes: MoeAttributes) -> Result<u64, String> {
    let float_count = attributes
        .expert_count
        .checked_add(attributes.experts_per_token)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "Metal MoE router threadgroup float count overflows".to_owned())?;
    let float_bytes = checked_mul(
        float_count,
        ElementType::F32.size_bytes(),
        "Metal MoE router threadgroup float bytes",
    )?;
    let id_bytes = checked_mul(
        attributes.experts_per_token,
        ElementType::I32.size_bytes(),
        "Metal MoE router threadgroup ID bytes",
    )?;
    float_bytes
        .checked_add(id_bytes)
        .ok_or_else(|| "Metal MoE router threadgroup bytes overflow".to_owned())
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
            "Metal MoE provider lacks unsigned attribute {name:?}"
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
        _ => Err(format!("Metal MoE provider lacks bool attribute {name:?}")),
    }
}

fn elements_bytes(elements: u64, bytes: u64, context: &str) -> Result<u64, String> {
    checked_mul(elements, bytes, context)
}

fn checked_mul(left: u64, right: u64, context: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{context} overflows"))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("Metal MoE alignment is invalid".to_owned());
    }
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or_else(|| "Metal MoE alignment overflows".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use half::f16;
    use metal::{MTLCommandBufferStatus, MTLResourceOptions};

    fn large_moe_shape() -> MoeAttributes {
        MoeAttributes {
            hidden_size: 2048,
            expert_count: 256,
            experts_per_token: 8,
            routed_intermediate_size: 512,
            shared_intermediate_size: 512,
            normalize_topk: true,
        }
    }

    #[test]
    fn q4k_workspace_formula_covers_runtime_layout() {
        let attributes = large_moe_shape();
        let (fixed, per_token) = workspace_formula_terms(attributes).unwrap();
        for tokens in [1_u64, 4, 16, 96] {
            let layout = MoeWorkspaceLayout::new(tokens, attributes).unwrap();
            assert!(layout.total_bytes <= fixed + tokens * per_token);
            assert_eq!(layout.pair_count, tokens * attributes.experts_per_token);
        }
    }

    #[test]
    fn q4k_provider_shape_is_attribute_driven() {
        large_moe_shape().validate().unwrap();
        let mut another = large_moe_shape();
        another.expert_count = 128;
        another.experts_per_token = 4;
        another.validate().unwrap();
        another.expert_count = MAX_ROUTER_EXPERTS + 1;
        assert!(another.validate().is_err());
    }

    #[test]
    fn q4k_routed_half_kernels_match_dequantized_cpu() {
        const EXPERTS: usize = 2;
        const TOKENS: usize = 2;
        const TOP_K: usize = 1;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 256;

        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping routed MoE kernel conformance");
            return;
        };
        let cpu = CandleDevice::Cpu;
        let quantized_stack = |seed: usize, rows: usize, columns: usize| {
            let mut bytes = Vec::new();
            let mut dequantized = Vec::new();
            for expert in 0..EXPERTS {
                let values = (0..rows * columns)
                    .map(|index| (((index + expert * 131 + seed) as f32) * 0.017).sin() * 0.08)
                    .collect::<Vec<_>>();
                let dense = Tensor::from_vec(values, (rows, columns), &cpu).unwrap();
                let quantized = QTensor::quantize(&dense, GgmlDType::Q4K).unwrap();
                bytes.extend_from_slice(&quantized.data().unwrap());
                dequantized.push(
                    quantized
                        .dequantize(&cpu)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
            }
            (bytes, dequantized)
        };
        let (gate_bytes, gate) = quantized_stack(7, INTERMEDIATE, HIDDEN);
        let (up_bytes, up) = quantized_stack(23, INTERMEDIATE, HIDDEN);
        let (down_bytes, down) = quantized_stack(41, HIDDEN, INTERMEDIATE);
        let input = (0..TOKENS * HIDDEN)
            .map(|index| f16::from_f32(((index as f32) * 0.031).cos() * 0.12))
            .collect::<Vec<_>>();
        let route_ids = [0_i32, 1_i32];
        let shared_buffer = |bytes: *const c_void, length: usize| {
            device.new_buffer_with_data(bytes, length as u64, MTLResourceOptions::StorageModeShared)
        };
        let gate_buffer = shared_buffer(gate_bytes.as_ptr() as *const c_void, gate_bytes.len());
        let up_buffer = shared_buffer(up_bytes.as_ptr() as *const c_void, up_bytes.len());
        let down_buffer = shared_buffer(down_bytes.as_ptr() as *const c_void, down_bytes.len());
        let input_buffer = shared_buffer(
            input.as_ptr() as *const c_void,
            input.len() * std::mem::size_of::<f16>(),
        );
        let ids_buffer = shared_buffer(
            route_ids.as_ptr() as *const c_void,
            route_ids.len() * std::mem::size_of::<i32>(),
        );
        let activation_buffer = device.new_buffer(
            (TOKENS * TOP_K * INTERMEDIATE * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer(
            (TOKENS * TOP_K * HIDDEN * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pipelines = MetalMoePipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let gate_up_params = Q4ExpertParams {
            output_features: INTERMEDIATE as u32,
            input_features: HIDDEN as u32,
            row_stride_bytes: Q4_K_BYTES_PER_BLOCK as u32,
            expert_stride_bytes: (INTERMEDIATE as u64 * Q4_K_BYTES_PER_BLOCK) as u32,
            expert_count: EXPERTS as u32,
            experts_per_token: TOP_K as u32,
            pair_count: (TOKENS * TOP_K) as u32,
        };
        encoder.set_compute_pipeline_state(&pipelines.routed_gate_up);
        encoder.set_buffer(0, Some(&gate_buffer), 0);
        encoder.set_buffer(1, Some(&up_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&ids_buffer), 0);
        encoder.set_buffer(4, Some(&activation_buffer), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<Q4ExpertParams>() as u64,
            &gate_up_params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (INTERMEDIATE as u64).div_ceil(4),
                1,
                (TOKENS * TOP_K) as u64,
            ),
            MTLSize::new(32, 2, 1),
        );
        let down_params = Q4ExpertParams {
            output_features: HIDDEN as u32,
            input_features: INTERMEDIATE as u32,
            row_stride_bytes: Q4_K_BYTES_PER_BLOCK as u32,
            expert_stride_bytes: (HIDDEN as u64 * Q4_K_BYTES_PER_BLOCK) as u32,
            ..gate_up_params
        };
        encoder.set_compute_pipeline_state(&pipelines.routed_down);
        encoder.set_buffer(0, Some(&down_buffer), 0);
        encoder.set_buffer(1, Some(&activation_buffer), 0);
        encoder.set_buffer(2, Some(&ids_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<Q4ExpertParams>() as u64,
            &down_params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new((HIDDEN as u64).div_ceil(4), 1, (TOKENS * TOP_K) as u64),
            MTLSize::new(32, 2, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f16,
                TOKENS * TOP_K * HIDDEN,
            )
        };
        for token in 0..TOKENS {
            let expert = route_ids[token] as usize;
            let input_row = &input[token * HIDDEN..(token + 1) * HIDDEN];
            let mut activation = vec![0.0_f32; INTERMEDIATE];
            for row in 0..INTERMEDIATE {
                let gate_value = input_row
                    .iter()
                    .zip(&gate[expert][row * HIDDEN..(row + 1) * HIDDEN])
                    .map(|(input, weight)| input.to_f32() * weight)
                    .sum::<f32>();
                let up_value = input_row
                    .iter()
                    .zip(&up[expert][row * HIDDEN..(row + 1) * HIDDEN])
                    .map(|(input, weight)| input.to_f32() * weight)
                    .sum::<f32>();
                activation[row] =
                    f16::from_f32((gate_value / (1.0 + (-gate_value).exp())) * up_value).to_f32();
            }
            for row in 0..HIDDEN {
                let expected = f16::from_f32(
                    activation
                        .iter()
                        .zip(&down[expert][row * INTERMEDIATE..(row + 1) * INTERMEDIATE])
                        .map(|(input, weight)| input * weight)
                        .sum::<f32>(),
                )
                .to_f32();
                let observed = actual[token * HIDDEN + row].to_f32();
                let tolerance = 0.02 + expected.abs() * 0.04;
                assert!(
                    (observed - expected).abs() <= tolerance,
                    "token={token} row={row} observed={observed} expected={expected} tolerance={tolerance}"
                );
            }
        }
    }
}
