//! Native Metal linear providers over typed physical weight layouts.

use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    dense_linear_contract, dense_swiglu_contract, last_token_dense_linear_contract,
    BatchedOperationInvocation, DynamicStorageRequirement, ElementType, EncodedDeviceOperation,
    OperationFailure, OperationProvider, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, PhysicalWeightPadding,
    ProviderWorkspaceRequirement, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    ResolvedTensorLayout, ResolvedValueRole, VNextError, WeightEncoding,
    DENSE_LINEAR_F16_CAPABILITY_ID, DENSE_LINEAR_OPERATION_ID, DENSE_SWIGLU_F16_CAPABILITY_ID,
    DENSE_SWIGLU_OPERATION_ID, LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
};
use metal::{CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError,
};
use super::weights::{
    resolve_weight, MetalResolvedCompositePart, MetalResolvedWeight, MetalResolvedWeightComponent,
    MetalResolvedWeightLayout,
};
use super::{
    binding, checked_u32, contiguous_bindings, contiguous_region, contiguous_token_region,
    ensure_invocation, estimate_without_workspace, f16_contiguous, implementation_fingerprint,
    invalid_plan, provider_descriptor, provider_failure, shared_scratch_region,
    shared_token_region, token_binding_is_shared, unsigned_attribute, DENSE_SAFETENSORS_FORMAT_ID,
    GGUF_NATIVE_BLOCK_FORMAT_ID, Q4_K_FORMAT_ID, Q5_K_FORMAT_ID, Q6_K_FORMAT_ID, Q8_0_FORMAT_ID,
    THREADS_PER_GROUP, VALUE_ALIGNMENT_BYTES,
};

const SHADER_SOURCE: &str = include_str!("linear.metal");
const DENSE_LINEAR_PROVIDER_ID: &str = "provider.metal.dense_linear.f16.native";
const DENSE_LINEAR_ESTIMATOR_ID: &str = "resource-estimator.metal.dense_linear.f16.native";
const DENSE_SWIGLU_PROVIDER_ID: &str = "provider.metal.dense_swiglu.f16.native";
const DENSE_SWIGLU_ESTIMATOR_ID: &str = "resource-estimator.metal.dense_swiglu.f16.native";
const LAST_TOKEN_PROVIDER_ID: &str = "provider.metal.last_token_dense_linear.f16.native";
const LAST_TOKEN_ESTIMATOR_ID: &str = "resource-estimator.metal.last_token_dense_linear.f16.native";
const SWIGLU_SCRATCH_PARTS: u64 = 3;

const LINEAR_DENSE_KERNEL: &str = "vnext_linear_dense_f16";
const LINEAR_Q4_K_KERNEL: &str = "vnext_linear_q4_k_f16";
const LINEAR_Q5_K_KERNEL: &str = "vnext_linear_q5_k_f16";
const LINEAR_Q6_K_KERNEL: &str = "vnext_linear_q6_k_f16";
const LINEAR_Q8_0_KERNEL: &str = "vnext_linear_q8_0_f16";
const SWIGLU_KERNEL: &str = "vnext_swiglu_f16";

pub(super) struct MetalLinearPipelines {
    dense: ComputePipelineState,
    q4_k: ComputePipelineState,
    q5_k: ComputePipelineState,
    q6_k: ComputePipelineState,
    q8_0: ComputePipelineState,
    swiglu: ComputePipelineState,
}

impl MetalLinearPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile Metal vNext linear library: {error}"
                ))
            })?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "load Metal vNext linear `{name}`: {error}"
                ))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext linear `{name}`: {error}"
                    ))
                })
        };
        Ok(Self {
            dense: pipeline(LINEAR_DENSE_KERNEL)?,
            q4_k: pipeline(LINEAR_Q4_K_KERNEL)?,
            q5_k: pipeline(LINEAR_Q5_K_KERNEL)?,
            q6_k: pipeline(LINEAR_Q6_K_KERNEL)?,
            q8_0: pipeline(LINEAR_Q8_0_KERNEL)?,
            swiglu: pipeline(SWIGLU_KERNEL)?,
        })
    }
}

pub(super) struct MetalDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalLinearPipelines>,
}

impl MetalDenseLinearProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = dense_linear_contract().map_err(super::contract_error)?;
        let descriptor = linear_provider_descriptor(
            runtime,
            &contract,
            DENSE_LINEAR_PROVIDER_ID,
            DENSE_LINEAR_F16_CAPABILITY_ID,
            DENSE_LINEAR_ESTIMATOR_ID,
            2,
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, DENSE_LINEAR_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalDenseLinearProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_linear(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.dense_linear.encode", message))
    }
}

pub(super) struct MetalDenseSwiGluProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalLinearPipelines>,
}

impl MetalDenseSwiGluProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = dense_swiglu_contract().map_err(super::contract_error)?;
        let descriptor = linear_provider_descriptor(
            runtime,
            &contract,
            DENSE_SWIGLU_PROVIDER_ID,
            DENSE_SWIGLU_F16_CAPABILITY_ID,
            DENSE_SWIGLU_ESTIMATOR_ID,
            3,
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalDenseSwiGluProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != DENSE_SWIGLU_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "Metal estimator `{}` received another operation",
                self.descriptor.resource_estimator_id()
            )));
        }
        let intermediate_size =
            unsigned_attribute(request.attributes(), "intermediate_size").map_err(invalid_plan)?;
        let bytes_per_token = intermediate_size
            .checked_mul(SWIGLU_SCRATCH_PARTS)
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| invalid_plan("Metal dense SwiGLU scratch size overflows"))?;
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::tokens(bytes_per_token)?,
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

impl OperationProvider<MetalDeviceRuntime> for MetalDenseSwiGluProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_dense_swiglu(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.dense_swiglu.encode", message))
    }
}

pub(super) struct MetalLastTokenDenseLinearProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalLinearPipelines>,
}

impl MetalLastTokenDenseLinearProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalLinearPipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = last_token_dense_linear_contract().map_err(super::contract_error)?;
        let descriptor = linear_provider_descriptor(
            runtime,
            &contract,
            LAST_TOKEN_PROVIDER_ID,
            LAST_TOKEN_DENSE_LINEAR_F16_CAPABILITY_ID,
            LAST_TOKEN_ESTIMATOR_ID,
            2,
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalLastTokenDenseLinearProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(
            &self.descriptor,
            &request,
            LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
        )
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalLastTokenDenseLinearProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_last_token_dense_linear(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| {
                provider_failure(identity, "metal.last_token_dense_linear.encode", message)
            })
    }
}

fn linear_provider_descriptor(
    runtime: &MetalDeviceRuntime,
    contract: &dyn ferrum_interfaces::vnext::OperationContract,
    provider_id: &str,
    capability_id: &str,
    estimator_id: &str,
    input_count: u32,
) -> Result<OperationProviderDescriptor, MetalDeviceRuntimeError> {
    provider_descriptor(
        runtime,
        contract,
        provider_id,
        capability_id,
        estimator_id,
        contiguous_bindings(input_count),
        &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
        &[
            Q4_K_FORMAT_ID,
            Q5_K_FORMAT_ID,
            Q6_K_FORMAT_ID,
            Q8_0_FORMAT_ID,
        ],
        implementation_fingerprint(&[
            include_str!("linear.rs").as_bytes(),
            SHADER_SOURCE.as_bytes(),
            provider_id.as_bytes(),
        ]),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinearPhysicalFormat {
    DenseF16,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PreparedLinearPart {
    region: usize,
    format: LinearPhysicalFormat,
    output_offset: u32,
    out_features: u32,
}

struct PreparedLinearWeight {
    regions: Vec<MetalBufferRegion>,
    parts: Vec<PreparedLinearPart>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LinearParams {
    rows: u32,
    in_features: u32,
    out_features: u32,
    output_stride: u32,
    output_column_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SwiGluParams {
    rows: u32,
    intermediate_size: u32,
    gate_up_stride: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LinearLaunch {
    input_region: usize,
    weight_region: usize,
    output_region: usize,
    input_offset_bytes: u64,
    output_offset_bytes: u64,
    format: LinearPhysicalFormat,
    params: LinearParams,
}

fn encode_dense_linear(
    pipelines: Arc<MetalLinearPipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_LINEAR_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let in_features = unsigned_attribute(first.attributes(), "in_features")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    validate_dense_linear_participant(first, in_features, out_features)?;
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
    )?;
    for participant in &invocation.participants()[1..] {
        if unsigned_attribute(participant.attributes(), "in_features")? != in_features
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("Metal dense linear participant attributes disagree".to_owned());
        }
        validate_dense_linear_participant(participant, in_features, out_features)?;
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err("Metal dense linear participants do not share one weight".to_owned());
        }
    }
    let prepared = prepare_matrix_weight(resolved, out_features, in_features)?;
    let [part] = prepared.parts.as_slice() else {
        return Err("Metal dense linear requires one physical matrix".to_owned());
    };
    let part = *part;
    let mut regions = prepared.regions;
    let input_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let output_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Output, 0, ElementType::F16)?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal dense linear participant ranges are incomplete".to_owned());
    }
    let mut launches = Vec::new();
    if input_shared && output_shared {
        let rows = invocation.work_shape().immediate_tokens();
        let input_region = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            rows,
        )?);
        let output_region = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            rows,
        )?);
        launches.push(linear_launch(
            part,
            input_region,
            output_region,
            rows,
            in_features,
            out_features,
            0,
            0,
        )?);
    } else {
        for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
            let rows = token_range.immediate_tokens();
            let input_start = if input_shared {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            let output_start = if output_shared {
                token_range.immediate_token_range().start
            } else {
                token_range.source_token_range().start
            };
            let input_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
                ElementType::F16,
                input_start,
                rows,
            )?);
            let output_region = regions.len();
            regions.push(contiguous_token_region(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
                ElementType::F16,
                output_start,
                rows,
            )?);
            launches.push(linear_launch(
                part,
                input_region,
                output_region,
                rows,
                in_features,
                out_features,
                0,
                0,
            )?);
        }
    }
    validate_launch_regions(&regions, &launches)?;
    MetalDeviceCommand::operation("vnext_dense_linear", regions, move |encoder, regions| {
        for launch in &launches {
            dispatch_linear(&pipelines, encoder.compute_encoder(), regions, *launch);
        }
        Ok(())
    })
    .map_err(|error| error.to_string())
}

fn encode_last_token_dense_linear(
    pipelines: Arc<MetalLinearPipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, LAST_TOKEN_DENSE_LINEAR_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let out_features = unsigned_attribute(first.attributes(), "out_features")?;
    validate_last_token_participant(first, hidden_size, out_features)?;
    let resolved = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
    )?;
    for participant in &invocation.participants()[1..] {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "out_features")? != out_features
        {
            return Err("Metal last-token linear participant attributes disagree".to_owned());
        }
        validate_last_token_participant(participant, hidden_size, out_features)?;
        let candidate = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        if !same_resolved_weight(&resolved, &candidate) {
            return Err("Metal last-token linear participants do not share one weight".to_owned());
        }
    }
    let prepared = prepare_matrix_weight(resolved, out_features, hidden_size)?;
    let [part] = prepared.parts.as_slice() else {
        return Err("Metal last-token linear requires one physical matrix".to_owned());
    };
    let part = *part;
    let mut regions = prepared.regions;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal last-token linear participant ranges are incomplete".to_owned());
    }
    let input_shared =
        token_binding_is_shared(&invocation, ResolvedValueRole::Input, 0, ElementType::F16)?;
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let selected = if input_shared {
            token_range.immediate_token_range()
        } else {
            token_range.source_token_range()
        };
        if selected.is_empty() {
            return Err("Metal last-token linear cannot select an empty span".to_owned());
        }
        let input_region = regions.len();
        regions.push(contiguous_token_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 0)?,
            ElementType::F16,
            selected.end - 1,
            1,
        )?);
        let output_region = regions.len();
        regions.push(contiguous_region(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Output, 0)?,
            ElementType::F16,
        )?);
        launches.push(linear_launch(
            part,
            input_region,
            output_region,
            1,
            hidden_size,
            out_features,
            0,
            0,
        )?);
    }
    validate_launch_regions(&regions, &launches)?;
    MetalDeviceCommand::operation(
        "vnext_last_token_dense_linear",
        regions,
        move |encoder, regions| {
            for launch in &launches {
                dispatch_linear(&pipelines, encoder.compute_encoder(), regions, *launch);
            }
            Ok(())
        },
    )
    .map_err(|error| error.to_string())
}

fn encode_dense_swiglu(
    pipelines: Arc<MetalLinearPipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_SWIGLU_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate_size = unsigned_attribute(first.attributes(), "intermediate_size")?;
    validate_swiglu_participant(first, hidden_size, intermediate_size)?;
    let gate_up = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
    )?;
    let down = resolve_weight(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 2)?,
    )?;
    for participant in &invocation.participants()[1..] {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || unsigned_attribute(participant.attributes(), "intermediate_size")?
                != intermediate_size
        {
            return Err("Metal dense SwiGLU participant attributes disagree".to_owned());
        }
        validate_swiglu_participant(participant, hidden_size, intermediate_size)?;
        let candidate_gate_up = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
        )?;
        let candidate_down = resolve_weight(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
        )?;
        if !same_resolved_weight(&gate_up, &candidate_gate_up)
            || !same_resolved_weight(&down, &candidate_down)
        {
            return Err("Metal dense SwiGLU participants do not share weights".to_owned());
        }
    }
    let gate_up = prepare_gate_up_weight(gate_up, intermediate_size, hidden_size)?;
    let down = prepare_matrix_weight(down, hidden_size, intermediate_size)?;
    let [down_part] = down.parts.as_slice() else {
        return Err("Metal dense SwiGLU down projection requires one matrix".to_owned());
    };
    let down_part = *down_part;
    let tokens = invocation.work_shape().immediate_tokens();
    let activation_elements = tokens
        .checked_mul(intermediate_size)
        .ok_or_else(|| "Metal dense SwiGLU activation size overflows".to_owned())?;
    let gate_up_bytes = activation_elements
        .checked_mul(2)
        .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
        .ok_or_else(|| "Metal dense SwiGLU gate/up scratch size overflows".to_owned())?;
    let activation_bytes = activation_elements
        .checked_mul(ElementType::F16.size_bytes())
        .ok_or_else(|| "Metal dense SwiGLU activation scratch size overflows".to_owned())?;
    let required_scratch_bytes = gate_up_bytes
        .checked_add(activation_bytes)
        .ok_or_else(|| "Metal dense SwiGLU total scratch size overflows".to_owned())?;

    if gate_up.regions.is_empty() || down.regions.is_empty() {
        return Err("Metal dense SwiGLU resolved empty weight storage".to_owned());
    }
    let mut regions = gate_up.regions;
    let down_region_base = regions.len();
    regions.extend(down.regions);
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
    regions.push(shared_scratch_region(&invocation, required_scratch_bytes)?);

    let packed_width = intermediate_size
        .checked_mul(2)
        .ok_or_else(|| "Metal dense SwiGLU packed width overflows".to_owned())?;
    let mut gate_launches = Vec::with_capacity(gate_up.parts.len());
    for part in gate_up.parts {
        let adjusted = PreparedLinearPart {
            region: part.region,
            ..part
        };
        gate_launches.push(linear_launch(
            adjusted,
            input_region,
            scratch_region,
            tokens,
            hidden_size,
            packed_width,
            0,
            0,
        )?);
    }
    let adjusted_down = PreparedLinearPart {
        region: down_region_base + down_part.region,
        ..down_part
    };
    let down_launch = linear_launch(
        adjusted_down,
        scratch_region,
        output_region,
        tokens,
        intermediate_size,
        hidden_size,
        gate_up_bytes,
        0,
    )?;
    validate_launch_regions(&regions, &gate_launches)?;
    validate_launch_regions(&regions, &[down_launch])?;
    validate_region_span(
        &regions[scratch_region],
        0,
        required_scratch_bytes,
        "Metal dense SwiGLU scratch",
    )?;
    let swiglu = SwiGluParams {
        rows: checked_u32(tokens, "Metal dense SwiGLU row count")?,
        intermediate_size: checked_u32(intermediate_size, "Metal dense SwiGLU intermediate size")?,
        gate_up_stride: checked_u32(packed_width, "Metal dense SwiGLU packed width")?,
    };
    MetalDeviceCommand::operation("vnext_dense_swiglu", regions, move |encoder, regions| {
        for launch in &gate_launches {
            dispatch_linear(&pipelines, encoder.compute_encoder(), regions, *launch);
        }
        dispatch_swiglu(
            &pipelines,
            encoder.compute_encoder(),
            &regions[scratch_region],
            gate_up_bytes,
            swiglu,
        );
        dispatch_linear(&pipelines, encoder.compute_encoder(), regions, down_launch);
        Ok(())
    })
    .map_err(|error| error.to_string())
}

pub(super) fn linear_launch(
    part: PreparedLinearPart,
    input_region: usize,
    output_region: usize,
    rows: u64,
    in_features: u64,
    output_stride: u64,
    input_offset_bytes: u64,
    output_offset_bytes: u64,
) -> Result<LinearLaunch, String> {
    Ok(LinearLaunch {
        input_region,
        weight_region: part.region,
        output_region,
        input_offset_bytes,
        output_offset_bytes,
        format: part.format,
        params: LinearParams {
            rows: checked_u32(rows, "Metal linear row count")?,
            in_features: checked_u32(in_features, "Metal linear input width")?,
            out_features: part.out_features,
            output_stride: checked_u32(output_stride, "Metal linear output stride")?,
            output_column_offset: part.output_offset,
        },
    })
}

pub(super) fn validate_launch_regions(
    regions: &[MetalBufferRegion],
    launches: &[LinearLaunch],
) -> Result<(), String> {
    for launch in launches {
        let input = regions
            .get(launch.input_region)
            .ok_or_else(|| "Metal linear input region index is invalid".to_owned())?;
        let weight = regions
            .get(launch.weight_region)
            .ok_or_else(|| "Metal linear weight region index is invalid".to_owned())?;
        let output = regions
            .get(launch.output_region)
            .ok_or_else(|| "Metal linear output region index is invalid".to_owned())?;
        let input_bytes = u64::from(launch.params.rows)
            .checked_mul(u64::from(launch.params.in_features))
            .and_then(|elements| elements.checked_mul(ElementType::F16.size_bytes()))
            .ok_or_else(|| "Metal linear input byte size overflows".to_owned())?;
        let output_elements = u64::from(launch.params.rows)
            .checked_mul(u64::from(launch.params.output_stride))
            .ok_or_else(|| "Metal linear output byte size overflows".to_owned())?;
        let output_bytes = output_elements
            .checked_mul(ElementType::F16.size_bytes())
            .ok_or_else(|| "Metal linear output byte size overflows".to_owned())?;
        validate_region_span(
            input,
            launch.input_offset_bytes,
            input_bytes,
            "Metal linear input",
        )?;
        validate_region_span(
            output,
            launch.output_offset_bytes,
            output_bytes,
            "Metal linear output",
        )?;
        if weight.length_bytes() == 0 {
            return Err("Metal linear weight region is empty".to_owned());
        }
        if u64::from(launch.params.output_column_offset)
            .checked_add(u64::from(launch.params.out_features))
            .is_none_or(|end| end > u64::from(launch.params.output_stride))
        {
            return Err("Metal linear output columns exceed their stride".to_owned());
        }
    }
    Ok(())
}

fn validate_region_span(
    region: &MetalBufferRegion,
    offset: u64,
    length: u64,
    context: &str,
) -> Result<(), String> {
    if length == 0
        || offset
            .checked_add(length)
            .is_none_or(|end| end > region.length_bytes())
    {
        return Err(format!("{context} exceeds its retained physical region"));
    }
    Ok(())
}

pub(super) fn dispatch_linear(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) {
    encoder.set_compute_pipeline_state(match launch.format {
        LinearPhysicalFormat::DenseF16 => &pipelines.dense,
        LinearPhysicalFormat::Q4K => &pipelines.q4_k,
        LinearPhysicalFormat::Q5K => &pipelines.q5_k,
        LinearPhysicalFormat::Q6K => &pipelines.q6_k,
        LinearPhysicalFormat::Q8_0 => &pipelines.q8_0,
    });
    set_region_offset(
        encoder,
        0,
        &regions[launch.input_region],
        launch.input_offset_bytes,
    );
    set_region_offset(encoder, 1, &regions[launch.weight_region], 0);
    set_region_offset(
        encoder,
        2,
        &regions[launch.output_region],
        launch.output_offset_bytes,
    );
    encoder.set_bytes(
        3,
        std::mem::size_of::<LinearParams>() as u64,
        &launch.params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(launch.params.out_features).div_ceil(4),
            u64::from(launch.params.rows),
            1,
        ),
        MTLSize::new(32, 2, 1),
    );
}

fn dispatch_swiglu(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    scratch: &MetalBufferRegion,
    activation_offset_bytes: u64,
    params: SwiGluParams,
) {
    encoder.set_compute_pipeline_state(&pipelines.swiglu);
    set_region_offset(encoder, 0, scratch, 0);
    set_region_offset(encoder, 1, scratch, activation_offset_bytes);
    encoder.set_bytes(
        2,
        std::mem::size_of::<SwiGluParams>() as u64,
        &params as *const _ as *const c_void,
    );
    let elements = u64::from(params.rows) * u64::from(params.intermediate_size);
    encoder.dispatch_thread_groups(
        MTLSize::new(elements.div_ceil(THREADS_PER_GROUP), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
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

fn prepare_matrix_weight(
    weight: MetalResolvedWeight,
    out_features: u64,
    in_features: u64,
) -> Result<PreparedLinearWeight, String> {
    if weight.logical_element_type() != ElementType::F16
        || weight.logical_dimensions() != [out_features, in_features]
    {
        return Err("Metal linear logical weight differs from its contract".to_owned());
    }
    let format_id = weight.format_id().as_str().to_owned();
    let (regions, components, layout) = weight.into_command_parts();
    let part = prepare_leaf_part(
        &format_id,
        &regions,
        &components,
        &layout,
        out_features,
        in_features,
        1,
        0,
    )?;
    Ok(PreparedLinearWeight {
        regions,
        parts: vec![part],
    })
}

pub(super) fn append_shared_matrix_weight(
    regions: &mut Vec<MetalBufferRegion>,
    invocation: &BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ordinal: u32,
    out_features: u64,
    in_features: u64,
    context: &str,
) -> Result<PreparedLinearPart, String> {
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
            return Err(format!("{context} participants do not share one weight"));
        }
    }
    let prepared = prepare_matrix_weight(resolved, out_features, in_features)?;
    let [part] = prepared.parts.as_slice() else {
        return Err(format!("{context} requires one physical matrix"));
    };
    let mut part = *part;
    part.region = part
        .region
        .checked_add(regions.len())
        .ok_or_else(|| format!("{context} region index overflows"))?;
    regions.extend(prepared.regions);
    Ok(part)
}

fn prepare_gate_up_weight(
    weight: MetalResolvedWeight,
    intermediate_size: u64,
    hidden_size: u64,
) -> Result<PreparedLinearWeight, String> {
    if weight.logical_element_type() != ElementType::F16
        || weight.logical_dimensions() != [2, intermediate_size, hidden_size]
    {
        return Err("Metal dense SwiGLU gate/up logical weight differs".to_owned());
    }
    let format_id = weight.format_id().as_str().to_owned();
    let (regions, components, layout) = weight.into_command_parts();
    let parts = match &layout {
        MetalResolvedWeightLayout::Composite { parts } => prepare_gate_up_composite(
            &format_id,
            &regions,
            &components,
            parts,
            intermediate_size,
            hidden_size,
        )?,
        _ => {
            let packed = intermediate_size
                .checked_mul(2)
                .ok_or_else(|| "Metal dense SwiGLU packed rows overflow".to_owned())?;
            vec![prepare_leaf_part(
                &format_id,
                &regions,
                &components,
                &layout,
                packed,
                hidden_size,
                2,
                0,
            )?]
        }
    };
    Ok(PreparedLinearWeight { regions, parts })
}

fn prepare_gate_up_composite(
    format_id: &str,
    regions: &[MetalBufferRegion],
    components: &[MetalResolvedWeightComponent],
    parts: &[MetalResolvedCompositePart],
    intermediate_size: u64,
    hidden_size: u64,
) -> Result<Vec<PreparedLinearPart>, String> {
    if parts.len() != 2 {
        return Err("Metal dense SwiGLU gate/up composite must have two parts".to_owned());
    }
    let mut prepared = Vec::with_capacity(2);
    for part in parts {
        if part.logical_offsets.len() != 3
            || part.extents != [1, intermediate_size, hidden_size]
            || part.logical_offsets[1..] != [0, 0]
            || part.logical_offsets[0] > 1
        {
            return Err("Metal dense SwiGLU gate/up composite partition differs".to_owned());
        }
        let output_offset = part.logical_offsets[0]
            .checked_mul(intermediate_size)
            .ok_or_else(|| "Metal dense SwiGLU partition offset overflows".to_owned())?;
        prepared.push(prepare_leaf_part(
            format_id,
            regions,
            components,
            &part.layout,
            intermediate_size,
            hidden_size,
            2,
            output_offset,
        )?);
    }
    prepared.sort_by_key(|part| part.output_offset);
    let expected_second = checked_u32(
        intermediate_size,
        "Metal dense SwiGLU second partition offset",
    )?;
    if prepared[0].output_offset != 0 || prepared[1].output_offset != expected_second {
        return Err("Metal dense SwiGLU gate/up partitions overlap or leave a gap".to_owned());
    }
    Ok(prepared)
}

#[allow(clippy::too_many_arguments)]
fn prepare_leaf_part(
    format_id: &str,
    regions: &[MetalBufferRegion],
    components: &[MetalResolvedWeightComponent],
    layout: &MetalResolvedWeightLayout,
    out_features: u64,
    in_features: u64,
    expected_block_axis: u32,
    output_offset: u64,
) -> Result<PreparedLinearPart, String> {
    let (component, format) = match layout {
        MetalResolvedWeightLayout::Dense { component }
        | MetalResolvedWeightLayout::Stored { component } => {
            if format_id != DENSE_SAFETENSORS_FORMAT_ID && format_id != GGUF_NATIVE_BLOCK_FORMAT_ID
            {
                return Err("Metal dense linear uses an unsupported weight format".to_owned());
            }
            let metadata = component_metadata(components, *component)?;
            if metadata.encoding()
                != &(WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
                || !physical_matrix_shape_matches(
                    metadata.physical_dimensions(),
                    out_features,
                    in_features,
                )
            {
                return Err("Metal dense linear physical ABI differs".to_owned());
            }
            (*component, LinearPhysicalFormat::DenseF16)
        }
        MetalResolvedWeightLayout::BlockQuantized {
            component,
            spec,
            block_axis,
            block_padding,
        } => {
            if format_id != GGUF_NATIVE_BLOCK_FORMAT_ID
                || *block_axis != expected_block_axis
                || block_padding != &PhysicalWeightPadding::Exact
                || !in_features.is_multiple_of(u64::from(spec.logical_values_per_block))
            {
                return Err("Metal quantized linear physical ABI differs".to_owned());
            }
            let format = match (
                spec.format_id.as_str(),
                spec.logical_values_per_block,
                spec.bytes_per_block,
            ) {
                (Q4_K_FORMAT_ID, 256, 144) => LinearPhysicalFormat::Q4K,
                (Q5_K_FORMAT_ID, 256, 176) => LinearPhysicalFormat::Q5K,
                (Q6_K_FORMAT_ID, 256, 210) => LinearPhysicalFormat::Q6K,
                (Q8_0_FORMAT_ID, 32, 34) => LinearPhysicalFormat::Q8_0,
                _ => {
                    return Err("Metal linear does not support this quantized block ABI".to_owned())
                }
            };
            let blocks_per_row = in_features / u64::from(spec.logical_values_per_block);
            let metadata = component_metadata(components, *component)?;
            if metadata.encoding() != &WeightEncoding::BlockQuantized(spec.clone())
                || !physical_matrix_shape_matches(
                    metadata.physical_dimensions(),
                    out_features,
                    blocks_per_row,
                )
            {
                return Err("Metal quantized linear component shape differs".to_owned());
            }
            (*component, format)
        }
        _ => return Err("Metal linear weight is not one matrix leaf".to_owned()),
    };
    if component >= regions.len() {
        return Err("Metal linear physical component is absent".to_owned());
    }
    Ok(PreparedLinearPart {
        region: component,
        format,
        output_offset: checked_u32(output_offset, "Metal linear output offset")?,
        out_features: checked_u32(out_features, "Metal linear output width")?,
    })
}

fn component_metadata(
    components: &[MetalResolvedWeightComponent],
    component: usize,
) -> Result<&MetalResolvedWeightComponent, String> {
    components
        .get(component)
        .ok_or_else(|| "Metal linear component metadata is absent".to_owned())
}

fn physical_matrix_shape_matches(dimensions: &[u64], rows: u64, columns: u64) -> bool {
    dimensions.last() == Some(&columns)
        && dimensions
            .iter()
            .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
            == rows.checked_mul(columns)
}

fn same_resolved_weight(left: &MetalResolvedWeight, right: &MetalResolvedWeight) -> bool {
    left.format_id() == right.format_id()
        && left.logical_dimensions() == right.logical_dimensions()
        && left.logical_element_type() == right.logical_element_type()
        && left.components() == right.components()
        && left.layout() == right.layout()
        && left.regions().len() == right.regions().len()
        && left
            .regions()
            .iter()
            .zip(right.regions())
            .all(|(left, right)| left.same_physical_region(right))
}

fn validate_dense_linear_participant(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, MetalDeviceBuffer>,
    in_features: u64,
    out_features: u64,
) -> Result<(), String> {
    let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
    let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let dimensions = input.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[1] != in_features
        || weight.tensor().dimensions() != [out_features, in_features]
        || output.tensor().dimensions() != [dimensions[0], out_features]
        || !f16_contiguous(input)
        || !f16_contiguous(weight)
        || !f16_contiguous(output)
    {
        return Err("Metal dense linear invocation differs from its signature".to_owned());
    }
    Ok(())
}

fn validate_last_token_participant(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, MetalDeviceBuffer>,
    hidden_size: u64,
    out_features: u64,
) -> Result<(), String> {
    let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
    let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let dimensions = input.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[0] == 0
        || dimensions[1] != hidden_size
        || weight.tensor().dimensions() != [out_features, hidden_size]
        || output.tensor().dimensions() != [1, out_features]
        || !f16_contiguous(input)
        || !f16_contiguous(weight)
        || !f16_contiguous(output)
    {
        return Err("Metal last-token linear invocation differs from its signature".to_owned());
    }
    Ok(())
}

fn validate_swiglu_participant(
    participant: &ferrum_interfaces::vnext::OperationInvocation<'_, MetalDeviceBuffer>,
    hidden_size: u64,
    intermediate_size: u64,
) -> Result<(), String> {
    let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
    let gate_up = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
    let down = binding(participant.bindings(), ResolvedValueRole::Input, 2)?;
    let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
    let dimensions = input.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[1] != hidden_size
        || gate_up.tensor().dimensions() != [2, intermediate_size, hidden_size]
        || down.tensor().dimensions() != [hidden_size, intermediate_size]
        || output.tensor().dimensions() != dimensions
        || !f16_contiguous(input)
        || !f16_contiguous(gate_up)
        || !f16_contiguous(down)
        || !f16_contiguous(output)
        || !matches!(input.tensor().layout(), ResolvedTensorLayout::Contiguous)
    {
        return Err("Metal dense SwiGLU invocation differs from its signature".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::numerical_tolerance;
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use half::f16;
    use metal::{BufferRef, MTLCommandBufferStatus, MTLResourceOptions};

    const DENSE_LINEAR_TOLERANCE_ID: &str =
        "runtime-vnext.metal.dense-linear.v1.operation.fp16.gguf-q4-k.hidden-2560";
    const DENSE_LINEAR_TOLERANCE_FINGERPRINT: &str =
        "afde4fbda18b82e0d7dfde8e92a416f1ee30f65a5b5ab0b1e31b27b8d3a27878";
    const DENSE_SWIGLU_TOLERANCE_ID: &str =
        "runtime-vnext.metal.dense-swiglu.v1.operation.fp16.gguf-q4-k-q6-k.full-pipeline";
    const DENSE_SWIGLU_TOLERANCE_FINGERPRINT: &str =
        "42d0496fbb889d9c95a35f151cda9726f7730f42f20efbd61aa593083f80661b";
    const LAST_TOKEN_LINEAR_TOLERANCE_ID: &str =
        "runtime-vnext.metal.last-token-dense-linear.v1.operation.fp16.gguf-q6-k.final-row";
    const LAST_TOKEN_LINEAR_TOLERANCE_FINGERPRINT: &str =
        "5dc080fb15a72c886acf83fc02877265f57359cf1c28c424a6cc1148cb256056";

    fn shared_buffer<T>(device: &Device, values: &[T]) -> metal::Buffer {
        device.new_buffer_with_data(
            values.as_ptr() as *const c_void,
            std::mem::size_of_val(values) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn output_buffer<T>(device: &Device, elements: usize) -> metal::Buffer {
        device.new_buffer(
            (elements * std::mem::size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn read_f16(buffer: &BufferRef, elements: usize) -> Vec<f32> {
        let values: &[f16] =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const f16, elements) };
        values.iter().map(|value| value.to_f32()).collect()
    }

    #[test]
    fn native_linear_formats_match_cpu_oracles_on_real_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping linear conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 2_usize;
        let input_width = 2560_usize;
        let output_width = 32_usize;
        let input = (0..rows * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.0017).sin() * 0.25))
            .collect::<Vec<_>>();
        let dense = (0..output_width * input_width)
            .map(|index| f16::from_f32(((index as f32) * 0.0023).cos() * 0.125))
            .collect::<Vec<_>>();
        let input_f32 = input.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let dense_f32 = dense.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(input_f32, (rows, input_width), &cpu).unwrap();
        let dense_tensor = Tensor::from_vec(dense_f32, (output_width, input_width), &cpu).unwrap();
        let input_buffer = shared_buffer(&device, &input);

        let mut cases = Vec::new();
        cases.push((
            LinearPhysicalFormat::DenseF16,
            shared_buffer(&device, &dense),
            input_tensor
                .matmul(&dense_tensor.transpose(0, 1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        ));
        for (dtype, format) in [
            (GgmlDType::Q4K, LinearPhysicalFormat::Q4K),
            (GgmlDType::Q5K, LinearPhysicalFormat::Q5K),
            (GgmlDType::Q6K, LinearPhysicalFormat::Q6K),
            (GgmlDType::Q8_0, LinearPhysicalFormat::Q8_0),
        ] {
            let quantized = QTensor::quantize(&dense_tensor, dtype).unwrap();
            let reference = input_tensor
                .matmul(&quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            cases.push((
                format,
                shared_buffer(&device, &quantized.data().unwrap()),
                reference,
            ));
        }

        for (format, weight, reference) in cases {
            let output = output_buffer::<f16>(&device, rows * output_width);
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            dispatch_raw_linear(
                &pipelines,
                encoder,
                format,
                &input_buffer,
                &weight,
                &output,
                LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: output_width as u32,
                    output_column_offset: 0,
                },
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let actual = read_f16(&output, rows * output_width);
            if format == LinearPhysicalFormat::Q4K {
                numerical_tolerance::assert_matches(
                    "Metal/CPU Q4_K dense linear",
                    &actual,
                    &[rows, output_width],
                    &reference,
                    &[rows, output_width],
                    numerical_tolerance::LogicalDtype::Fp16,
                    DENSE_LINEAR_TOLERANCE_ID,
                    DENSE_LINEAR_TOLERANCE_FINGERPRINT,
                )
                .expect("reviewed dense-linear numerical contract");
            } else {
                assert_linear_diagnostic_close(&format!("{format:?}"), &actual, &reference);
            }
        }
    }

    #[test]
    fn native_swiglu_uses_packed_gate_then_up_order() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping SwiGLU conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 2_u32;
        let intermediate = 64_u32;
        let stride = intermediate * 2;
        let packed = (0..rows * stride)
            .map(|index| f16::from_f32((index as f32 * 0.013).sin()))
            .collect::<Vec<_>>();
        let input = shared_buffer(&device, &packed);
        let output = output_buffer::<f16>(&device, (rows * intermediate) as usize);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipelines.swiglu);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        let params = SwiGluParams {
            rows,
            intermediate_size: intermediate,
            gate_up_stride: stride,
        };
        encoder.set_bytes(
            2,
            std::mem::size_of::<SwiGluParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (u64::from(rows) * u64::from(intermediate)).div_ceil(THREADS_PER_GROUP),
                1,
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let actual = read_f16(&output, (rows * intermediate) as usize);
        for row in 0..rows as usize {
            for column in 0..intermediate as usize {
                let gate = packed[row * stride as usize + column].to_f32();
                let up = packed[row * stride as usize + intermediate as usize + column].to_f32();
                let expected = f16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
                assert!((actual[row * intermediate as usize + column] - expected).abs() <= 0.002);
            }
        }
    }

    #[test]
    fn native_dense_swiglu_q4k_q6k_matches_full_cpu_oracle_on_real_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping dense SwiGLU conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 2_usize;
        let hidden = 256_usize;
        let intermediate = 256_usize;
        let input = (0..rows * hidden)
            .map(|index| f16::from_f32((index as f32 * 0.017).sin() * 0.125))
            .collect::<Vec<_>>();
        let gate_up = (0..2 * intermediate * hidden)
            .map(|index| (index as f32 * 0.0031).cos() * 0.0625)
            .collect::<Vec<_>>();
        let down = (0..hidden * intermediate)
            .map(|index| (index as f32 * 0.0043).sin() * 0.0625)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let input_tensor = Tensor::from_vec(
            input.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
            (rows, hidden),
            &cpu,
        )
        .unwrap();
        let gate_up_tensor = Tensor::from_vec(gate_up, (2 * intermediate, hidden), &cpu).unwrap();
        let down_tensor = Tensor::from_vec(down, (hidden, intermediate), &cpu).unwrap();
        let gate_up_quantized = QTensor::quantize(&gate_up_tensor, GgmlDType::Q4K).unwrap();
        let down_quantized = QTensor::quantize(&down_tensor, GgmlDType::Q6K).unwrap();

        let cpu_gate_up = input_tensor
            .matmul(
                &gate_up_quantized
                    .dequantize(&cpu)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap(),
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let mut cpu_activated = vec![0.0_f32; rows * intermediate];
        for row in 0..rows {
            for column in 0..intermediate {
                let gate = cpu_gate_up[row * 2 * intermediate + column];
                let up = cpu_gate_up[row * 2 * intermediate + intermediate + column];
                cpu_activated[row * intermediate + column] = gate / (1.0 + (-gate).exp()) * up;
            }
        }
        let cpu_output = Tensor::from_vec(cpu_activated, (rows, intermediate), &cpu)
            .unwrap()
            .matmul(
                &down_quantized
                    .dequantize(&cpu)
                    .unwrap()
                    .transpose(0, 1)
                    .unwrap(),
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let input_buffer = shared_buffer(&device, &input);
        let gate_up_buffer = shared_buffer(&device, &gate_up_quantized.data().unwrap());
        let down_buffer = shared_buffer(&device, &down_quantized.data().unwrap());
        let projected = output_buffer::<f16>(&device, rows * 2 * intermediate);
        let activated = output_buffer::<f16>(&device, rows * intermediate);
        let output = output_buffer::<f16>(&device, rows * hidden);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_linear(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q4K,
            &input_buffer,
            &gate_up_buffer,
            &projected,
            LinearParams {
                rows: rows as u32,
                in_features: hidden as u32,
                out_features: (2 * intermediate) as u32,
                output_stride: (2 * intermediate) as u32,
                output_column_offset: 0,
            },
        );
        encoder.set_compute_pipeline_state(&pipelines.swiglu);
        encoder.set_buffer(0, Some(&projected), 0);
        encoder.set_buffer(1, Some(&activated), 0);
        let swiglu = SwiGluParams {
            rows: rows as u32,
            intermediate_size: intermediate as u32,
            gate_up_stride: (2 * intermediate) as u32,
        };
        encoder.set_bytes(
            2,
            std::mem::size_of::<SwiGluParams>() as u64,
            &swiglu as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (rows as u64 * intermediate as u64).div_ceil(THREADS_PER_GROUP),
                1,
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
        dispatch_raw_linear(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q6K,
            &activated,
            &down_buffer,
            &output,
            LinearParams {
                rows: rows as u32,
                in_features: intermediate as u32,
                out_features: hidden as u32,
                output_stride: hidden as u32,
                output_column_offset: 0,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = read_f16(&output, rows * hidden);
        assert!(actual.iter().any(|value| value.abs() > 1.0e-5));
        numerical_tolerance::assert_matches(
            "Metal/CPU dense SwiGLU Q4_K/Q6_K",
            &actual,
            &[rows, hidden],
            &cpu_output,
            &[rows, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            DENSE_SWIGLU_TOLERANCE_ID,
            DENSE_SWIGLU_TOLERANCE_FINGERPRINT,
        )
        .expect("reviewed dense-SwiGLU numerical contract");
    }

    #[test]
    fn native_last_token_q6k_linear_selects_final_row_on_real_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping last-token linear conformance");
            return;
        };
        let pipelines = MetalLinearPipelines::new(&device).unwrap();
        let queue = device.new_command_queue();
        let rows = 3_usize;
        let hidden = 256_usize;
        let output_width = 64_usize;
        let input = (0..rows * hidden)
            .map(|index| {
                let row = index / hidden;
                f16::from_f32((index as f32 * 0.011).sin() * 0.125 + row as f32 * 0.03125)
            })
            .collect::<Vec<_>>();
        let weight = (0..output_width * hidden)
            .map(|index| (index as f32 * 0.0071).cos() * 0.0625)
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let weight_tensor = Tensor::from_vec(weight, (output_width, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&weight_tensor, GgmlDType::Q6K).unwrap();
        let dequantized = quantized.dequantize(&cpu).unwrap().transpose(0, 1).unwrap();
        let cpu_row = |row: usize| {
            Tensor::from_vec(
                input[row * hidden..(row + 1) * hidden]
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                (1, hidden),
                &cpu,
            )
            .unwrap()
            .matmul(&dequantized)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        };
        let first_row = cpu_row(0);
        let final_row = cpu_row(rows - 1);
        assert!(first_row
            .iter()
            .zip(&final_row)
            .any(|(first, last)| (first - last).abs() > 1.0e-3));

        let input_buffer = shared_buffer(&device, &input);
        let weight_buffer = shared_buffer(&device, &quantized.data().unwrap());
        let output = output_buffer::<f16>(&device, output_width);
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_linear_at(
            &pipelines,
            encoder,
            LinearPhysicalFormat::Q6K,
            &input_buffer,
            ((rows - 1) * hidden * std::mem::size_of::<f16>()) as u64,
            &weight_buffer,
            &output,
            LinearParams {
                rows: 1,
                in_features: hidden as u32,
                out_features: output_width as u32,
                output_stride: output_width as u32,
                output_column_offset: 0,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let actual = read_f16(&output, output_width);
        numerical_tolerance::assert_matches(
            "Metal/CPU last-token Q6_K dense linear",
            &actual,
            &[1, output_width],
            &final_row,
            &[1, output_width],
            numerical_tolerance::LogicalDtype::Fp16,
            LAST_TOKEN_LINEAR_TOLERANCE_ID,
            LAST_TOKEN_LINEAR_TOLERANCE_FINGERPRINT,
        )
        .expect("reviewed last-token dense-linear numerical contract");
    }

    fn assert_linear_diagnostic_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.02_f32.max(expected.abs() * 0.01);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{label}[{index}] {actual} != {expected}"
            );
        }
    }

    fn dispatch_raw_linear(
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        format: LinearPhysicalFormat,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        params: LinearParams,
    ) {
        dispatch_raw_linear_at(pipelines, encoder, format, input, 0, weight, output, params);
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_raw_linear_at(
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        format: LinearPhysicalFormat,
        input: &BufferRef,
        input_offset_bytes: u64,
        weight: &BufferRef,
        output: &BufferRef,
        params: LinearParams,
    ) {
        encoder.set_compute_pipeline_state(match format {
            LinearPhysicalFormat::DenseF16 => &pipelines.dense,
            LinearPhysicalFormat::Q4K => &pipelines.q4_k,
            LinearPhysicalFormat::Q5K => &pipelines.q5_k,
            LinearPhysicalFormat::Q6K => &pipelines.q6_k,
            LinearPhysicalFormat::Q8_0 => &pipelines.q8_0,
        });
        encoder.set_buffer(0, Some(input), input_offset_bytes);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<LinearParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.out_features).div_ceil(4),
                u64::from(params.rows),
                1,
            ),
            MTLSize::new(32, 2, 1),
        );
    }
}
