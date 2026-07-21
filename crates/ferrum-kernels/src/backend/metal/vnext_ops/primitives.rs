//! Native F16 Metal providers for the smallest standard transformer ops.

use std::ffi::c_void;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    residual_add_contract, rms_norm_contract, token_embedding_contract, BatchedOperationInvocation,
    DeviceBatchingForm, ElementType, EncodedDeviceOperation, OperationFailure, OperationProvider,
    OperationProviderDescriptor, OperationResourceEstimate, OperationResourceEstimateRequest,
    OperationResourceEstimator, PhysicalWeightPadding, ResolvedValueRole, VNextError,
    WeightEncoding, RESIDUAL_ADD_F16_CAPABILITY_ID, RESIDUAL_ADD_OPERATION_ID,
    RMS_NORM_F16_CAPABILITY_ID, RMS_NORM_OPERATION_ID, TOKEN_EMBEDDING_F16_CAPABILITY_ID,
    TOKEN_EMBEDDING_OPERATION_ID,
};
use metal::{CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use super::super::vnext_runtime::{
    MetalBufferRegion, MetalDeviceBuffer, MetalDeviceCommand, MetalDeviceRuntime,
    MetalDeviceRuntimeError,
};
use super::weights::{resolve_weight, MetalResolvedWeightLayout};
use super::{
    binding, checked_u32, contiguous_bindings, contiguous_token_region, ensure_invocation,
    estimate_without_workspace, f16_contiguous, implementation_fingerprint, provider_descriptor,
    provider_failure, rational_attribute, shared_full_region, shared_token_region,
    unsigned_attribute, DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID, Q6_K_FORMAT_ID,
    THREADS_PER_GROUP,
};

const SHADER_SOURCE: &str = include_str!("primitives.metal");
const TOKEN_EMBEDDING_PROVIDER_ID: &str = "provider.metal.token_embedding.f16";
const TOKEN_EMBEDDING_ESTIMATOR_ID: &str = "resource-estimator.metal.token_embedding.f16";
const RMS_NORM_PROVIDER_ID: &str = "provider.metal.rms_norm.f16";
const RMS_NORM_ESTIMATOR_ID: &str = "resource-estimator.metal.rms_norm.f16";
const RESIDUAL_ADD_PROVIDER_ID: &str = "provider.metal.residual_add.f16";
const RESIDUAL_ADD_ESTIMATOR_ID: &str = "resource-estimator.metal.residual_add.f16";

const EMBEDDING_DENSE_KERNEL: &str = "vnext_embedding_dense_f16";
const EMBEDDING_Q6_K_KERNEL: &str = "vnext_embedding_q6_k_f16";
const RMS_NORM_KERNEL: &str = "vnext_rms_norm_f16";
const RESIDUAL_ADD_KERNEL: &str = "vnext_residual_add_f16";

pub(super) struct MetalPrimitivePipelines {
    embedding_dense: ComputePipelineState,
    embedding_q6_k: ComputePipelineState,
    rms_norm: ComputePipelineState,
    residual_add: ComputePipelineState,
}

impl MetalPrimitivePipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "compile Metal vNext primitive library: {error}"
                ))
            })?;
        let pipeline = |name: &str| {
            let function = library.get_function(name, None).map_err(|error| {
                MetalDeviceRuntimeError::contract(format!(
                    "load Metal vNext primitive `{name}`: {error}"
                ))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| {
                    MetalDeviceRuntimeError::contract(format!(
                        "build Metal vNext primitive `{name}`: {error}"
                    ))
                })
        };
        Ok(Self {
            embedding_dense: pipeline(EMBEDDING_DENSE_KERNEL)?,
            embedding_q6_k: pipeline(EMBEDDING_Q6_K_KERNEL)?,
            rms_norm: pipeline(RMS_NORM_KERNEL)?,
            residual_add: pipeline(RESIDUAL_ADD_KERNEL)?,
        })
    }
}

pub(super) struct MetalTokenEmbeddingProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalTokenEmbeddingProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = token_embedding_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            TOKEN_EMBEDDING_PROVIDER_ID,
            TOKEN_EMBEDDING_F16_CAPABILITY_ID,
            TOKEN_EMBEDDING_ESTIMATOR_ID,
            contiguous_bindings(2),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[Q6_K_FORMAT_ID],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                TOKEN_EMBEDDING_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalTokenEmbeddingProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, TOKEN_EMBEDDING_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalTokenEmbeddingProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_token_embedding(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.token_embedding.encode", message))
    }
}

pub(super) struct MetalRmsNormProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalRmsNormProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = rms_norm_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            RMS_NORM_PROVIDER_ID,
            RMS_NORM_F16_CAPABILITY_ID,
            RMS_NORM_ESTIMATOR_ID,
            contiguous_bindings(2),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                RMS_NORM_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalRmsNormProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, RMS_NORM_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalRmsNormProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_rms_norm(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.rms_norm.encode", message))
    }
}

pub(super) struct MetalResidualAddProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<MetalPrimitivePipelines>,
}

impl MetalResidualAddProvider {
    pub(super) fn new(
        runtime: &MetalDeviceRuntime,
        pipelines: Arc<MetalPrimitivePipelines>,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = residual_add_contract().map_err(super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            RESIDUAL_ADD_PROVIDER_ID,
            RESIDUAL_ADD_F16_CAPABILITY_ID,
            RESIDUAL_ADD_ESTIMATOR_ID,
            contiguous_bindings(2),
            &[DENSE_SAFETENSORS_FORMAT_ID, GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[],
            implementation_fingerprint(&[
                include_str!("primitives.rs").as_bytes(),
                SHADER_SOURCE.as_bytes(),
                RESIDUAL_ADD_PROVIDER_ID.as_bytes(),
            ]),
        )?;
        Ok(Self {
            descriptor,
            pipelines,
        })
    }
}

impl OperationResourceEstimator for MetalResidualAddProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        estimate_without_workspace(&self.descriptor, &request, RESIDUAL_ADD_OPERATION_ID)
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalResidualAddProvider {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_residual_add(Arc::clone(&self.pipelines), invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|message| provider_failure(identity, "metal.residual_add.encode", message))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingPhysicalFormat {
    DenseF16,
    Q6K,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EmbeddingParams {
    token_count: u32,
    hidden_size: u32,
    vocabulary_size: u32,
}

#[derive(Debug, Clone, Copy)]
struct EmbeddingLaunch {
    first_region: usize,
    format: EmbeddingPhysicalFormat,
    params: EmbeddingParams,
}

fn encode_token_embedding(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, TOKEN_EMBEDDING_OPERATION_ID)?;
    let token_ranges = invocation.participant_token_ranges();
    if token_ranges.len() != invocation.participants().len() {
        return Err("Metal token embedding participant ranges are incomplete".to_owned());
    }
    let mut regions = Vec::with_capacity(invocation.participants().len() * 3);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant, token_range) in invocation.participants().iter().zip(token_ranges) {
        let token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let hidden_size = unsigned_attribute(participant.attributes(), "hidden_size")?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        validate_embedding_signature(token_ids, table, output, vocabulary_size, hidden_size)?;
        let weight = resolve_weight(participant, table)?;
        let format = embedding_weight_format(&weight, vocabulary_size, hidden_size)?;
        let (mut table_regions, _, _) = weight.into_command_parts();
        if table_regions.len() != 1 {
            return Err("Metal token embedding requires one physical table component".to_owned());
        }
        let first_region = regions.len();
        regions.append(&mut table_regions);
        regions.push(contiguous_token_region(
            participant,
            token_ids,
            ElementType::U32,
            token_range.source_token_range().start,
            token_range.immediate_tokens(),
        )?);
        regions.push(contiguous_token_region(
            participant,
            output,
            ElementType::F16,
            token_range.immediate_token_range().start,
            token_range.immediate_tokens(),
        )?);
        launches.push(EmbeddingLaunch {
            first_region,
            format,
            params: EmbeddingParams {
                token_count: checked_u32(
                    token_range.immediate_tokens(),
                    "Metal embedding token count",
                )?,
                hidden_size: checked_u32(hidden_size, "Metal embedding hidden size")?,
                vocabulary_size: checked_u32(vocabulary_size, "Metal embedding vocabulary size")?,
            },
        });
    }
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal embedding participant count",
    )?;
    let token_count = invocation.work_shape().immediate_tokens();
    let dispatch_count = launches.len() as u64;
    MetalDeviceCommand::operation("vnext_token_embedding", regions, move |encoder, regions| {
        encoder.record_compute_dispatches(dispatch_count);
        let compute = encoder.compute_encoder();
        for launch in &launches {
            dispatch_embedding(
                &pipelines,
                compute,
                launch.format,
                &regions[launch.first_region],
                &regions[launch.first_region + 1],
                &regions[launch.first_region + 2],
                launch.params,
            );
        }
        Ok(())
    })
    .map_err(|error| error.to_string())?
    .with_work_shape(
        if participant_count == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        participant_count,
        token_count,
    )
    .map_err(|error| error.to_string())
}

fn embedding_weight_format(
    weight: &super::weights::MetalResolvedWeight,
    vocabulary_size: u64,
    hidden_size: u64,
) -> Result<EmbeddingPhysicalFormat, String> {
    if weight.logical_element_type() != ElementType::F16
        || weight.logical_dimensions() != [vocabulary_size, hidden_size]
    {
        return Err("Metal embedding logical weight differs from its contract".to_owned());
    }
    let (component, format) = match weight.layout() {
        MetalResolvedWeightLayout::Dense { component }
        | MetalResolvedWeightLayout::Stored { component } => {
            let component = *component;
            let metadata = weight
                .components()
                .get(component)
                .ok_or_else(|| "Metal dense embedding component is absent".to_owned())?;
            if weight.format_id().as_str() != DENSE_SAFETENSORS_FORMAT_ID
                && weight.format_id().as_str() != GGUF_NATIVE_BLOCK_FORMAT_ID
            {
                return Err("Metal dense embedding uses an unsupported weight format".to_owned());
            }
            if metadata.encoding()
                != &(WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
                || metadata.physical_dimensions() != [vocabulary_size, hidden_size]
            {
                return Err("Metal dense embedding physical ABI differs".to_owned());
            }
            (component, EmbeddingPhysicalFormat::DenseF16)
        }
        MetalResolvedWeightLayout::BlockQuantized {
            component,
            spec,
            block_axis,
            block_padding,
        } => {
            if weight.format_id().as_str() != GGUF_NATIVE_BLOCK_FORMAT_ID
                || spec.format_id.as_str() != Q6_K_FORMAT_ID
                || spec.logical_values_per_block != 256
                || spec.bytes_per_block != 210
                || *block_axis != 1
                || block_padding != &PhysicalWeightPadding::Exact
                || !hidden_size.is_multiple_of(256)
            {
                return Err("Metal Q6_K embedding physical ABI differs".to_owned());
            }
            let metadata = weight
                .components()
                .get(*component)
                .ok_or_else(|| "Metal Q6_K embedding component is absent".to_owned())?;
            if metadata.physical_dimensions() != [vocabulary_size, hidden_size / 256]
                || metadata.encoding() != &WeightEncoding::BlockQuantized(spec.clone())
            {
                return Err("Metal Q6_K embedding component shape differs".to_owned());
            }
            (*component, EmbeddingPhysicalFormat::Q6K)
        }
        _ => return Err("Metal token embedding does not support this physical layout".to_owned()),
    };
    if component != 0 || weight.regions().len() != 1 {
        return Err("Metal token embedding requires one canonical table component".to_owned());
    }
    Ok(format)
}

fn validate_embedding_signature(
    token_ids: &ferrum_interfaces::vnext::ResolvedValueBinding,
    table: &ferrum_interfaces::vnext::ResolvedValueBinding,
    output: &ferrum_interfaces::vnext::ResolvedValueBinding,
    vocabulary_size: u64,
    hidden_size: u64,
) -> Result<(), String> {
    let token_dimensions = token_ids.tensor().dimensions();
    if token_ids.tensor().element_type() != ElementType::U32
        || table.tensor().element_type() != ElementType::F16
        || output.tensor().element_type() != ElementType::F16
        || token_dimensions.len() != 1
        || table.tensor().dimensions() != [vocabulary_size, hidden_size]
        || output.tensor().dimensions() != [token_dimensions[0], hidden_size]
        || !matches!(
            token_ids.tensor().layout(),
            ferrum_interfaces::vnext::ResolvedTensorLayout::Contiguous
        )
        || !matches!(
            table.tensor().layout(),
            ferrum_interfaces::vnext::ResolvedTensorLayout::Contiguous
        )
        || !matches!(
            output.tensor().layout(),
            ferrum_interfaces::vnext::ResolvedTensorLayout::Contiguous
        )
    {
        return Err("Metal token embedding invocation differs from its signature".to_owned());
    }
    Ok(())
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RmsNormParams {
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
}

fn encode_rms_norm(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, RMS_NORM_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    let epsilon = rational_attribute(first.attributes(), "epsilon")?;
    for participant in invocation.participants() {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || rational_attribute(participant.attributes(), "epsilon")? != epsilon
            || !valid_rms_norm(input, weight, output, hidden_size)
        {
            return Err("Metal RMSNorm participants disagree with the signature".to_owned());
        }
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_full_region(&invocation, ResolvedValueRole::Input, 1, ElementType::F16)?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
    ];
    let params = RmsNormParams {
        rows: checked_u32(tokens, "Metal RMSNorm row count")?,
        hidden_size: checked_u32(hidden_size, "Metal RMSNorm hidden size")?,
        epsilon,
    };
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal RMSNorm participant count",
    )?;
    MetalDeviceCommand::operation("vnext_rms_norm", regions, move |encoder, regions| {
        encoder.record_compute_dispatches(1);
        dispatch_rms_norm(
            &pipelines,
            encoder.compute_encoder(),
            &regions[0],
            &regions[1],
            &regions[2],
            params,
        );
        Ok(())
    })
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

fn valid_rms_norm(
    input: &ferrum_interfaces::vnext::ResolvedValueBinding,
    weight: &ferrum_interfaces::vnext::ResolvedValueBinding,
    output: &ferrum_interfaces::vnext::ResolvedValueBinding,
    hidden_size: u64,
) -> bool {
    let [rows, input_hidden] = input.tensor().dimensions() else {
        return false;
    };
    *input_hidden == hidden_size
        && weight.tensor().dimensions() == [hidden_size]
        && output.tensor().dimensions() == [*rows, hidden_size]
        && f16_contiguous(input)
        && f16_contiguous(weight)
        && f16_contiguous(output)
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ResidualAddParams {
    elements: u32,
}

fn encode_residual_add(
    pipelines: Arc<MetalPrimitivePipelines>,
    invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
) -> Result<MetalDeviceCommand, String> {
    ensure_invocation(&invocation, RESIDUAL_ADD_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden_size = unsigned_attribute(first.attributes(), "hidden_size")?;
    for participant in invocation.participants() {
        let left = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let right = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden_size
            || !valid_residual_add(left, right, output, hidden_size)
        {
            return Err("Metal residual-add participants disagree with the signature".to_owned());
        }
    }
    let tokens = invocation.work_shape().immediate_tokens();
    let elements = tokens
        .checked_mul(hidden_size)
        .ok_or_else(|| "Metal residual-add element count overflows".to_owned())?;
    let regions = vec![
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            1,
            ElementType::F16,
            tokens,
        )?,
        shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?,
    ];
    let params = ResidualAddParams {
        elements: checked_u32(elements, "Metal residual-add element count")?,
    };
    let participant_count = checked_u32(
        invocation.participants().len() as u64,
        "Metal residual-add participant count",
    )?;
    MetalDeviceCommand::operation("vnext_residual_add", regions, move |encoder, regions| {
        encoder.record_compute_dispatches(1);
        dispatch_residual_add(
            &pipelines,
            encoder.compute_encoder(),
            &regions[0],
            &regions[1],
            &regions[2],
            params,
        );
        Ok(())
    })
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

fn valid_residual_add(
    left: &ferrum_interfaces::vnext::ResolvedValueBinding,
    right: &ferrum_interfaces::vnext::ResolvedValueBinding,
    output: &ferrum_interfaces::vnext::ResolvedValueBinding,
    hidden_size: u64,
) -> bool {
    let [tokens, input_hidden] = left.tensor().dimensions() else {
        return false;
    };
    *input_hidden == hidden_size
        && right.tensor().dimensions() == [*tokens, hidden_size]
        && output.tensor().dimensions() == [*tokens, hidden_size]
        && f16_contiguous(left)
        && f16_contiguous(right)
        && f16_contiguous(output)
}

fn set_region(encoder: &ComputeCommandEncoderRef, index: u64, region: &MetalBufferRegion) {
    encoder.set_buffer(index, Some(region.buffer()), region.offset_bytes());
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

fn dispatch_embedding(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    format: EmbeddingPhysicalFormat,
    table: &MetalBufferRegion,
    token_ids: &MetalBufferRegion,
    output: &MetalBufferRegion,
    params: EmbeddingParams,
) {
    let pipeline = match format {
        EmbeddingPhysicalFormat::DenseF16 => &pipelines.embedding_dense,
        EmbeddingPhysicalFormat::Q6K => &pipelines.embedding_q6_k,
    };
    encoder.set_compute_pipeline_state(pipeline);
    set_region(encoder, 0, table);
    set_region(encoder, 1, token_ids);
    set_region(encoder, 2, output);
    encoder.set_bytes(
        3,
        std::mem::size_of::<EmbeddingParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(
            u64::from(params.hidden_size).div_ceil(THREADS_PER_GROUP),
            u64::from(params.token_count),
            1,
        ),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn dispatch_rms_norm(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    params: RmsNormParams,
) {
    dispatch_rms_norm_at(
        pipelines,
        encoder,
        input,
        0,
        weight,
        output,
        0,
        params.rows,
        params.hidden_size,
        params.epsilon,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_rms_norm_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    input: &MetalBufferRegion,
    input_offset_bytes: u64,
    weight: &MetalBufferRegion,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
) {
    let params = RmsNormParams {
        rows,
        hidden_size,
        epsilon,
    };
    encoder.set_compute_pipeline_state(&pipelines.rms_norm);
    set_region_offset(encoder, 0, input, input_offset_bytes);
    set_region(encoder, 1, weight);
    set_region_offset(encoder, 2, output, output_offset_bytes);
    encoder.set_bytes(
        3,
        std::mem::size_of::<RmsNormParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(params.rows), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

fn dispatch_residual_add(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    right: &MetalBufferRegion,
    output: &MetalBufferRegion,
    params: ResidualAddParams,
) {
    dispatch_residual_add_at(
        pipelines,
        encoder,
        left,
        0,
        right,
        0,
        output,
        0,
        params.elements,
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_residual_add_at(
    pipelines: &MetalPrimitivePipelines,
    encoder: &ComputeCommandEncoderRef,
    left: &MetalBufferRegion,
    left_offset_bytes: u64,
    right: &MetalBufferRegion,
    right_offset_bytes: u64,
    output: &MetalBufferRegion,
    output_offset_bytes: u64,
    elements: u32,
) {
    let params = ResidualAddParams { elements };
    encoder.set_compute_pipeline_state(&pipelines.residual_add);
    set_region_offset(encoder, 0, left, left_offset_bytes);
    set_region_offset(encoder, 1, right, right_offset_bytes);
    set_region_offset(encoder, 2, output, output_offset_bytes);
    encoder.set_bytes(
        3,
        std::mem::size_of::<ResidualAddParams>() as u64,
        &params as *const _ as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(params.elements).div_ceil(THREADS_PER_GROUP), 1, 1),
        MTLSize::new(THREADS_PER_GROUP, 1, 1),
    );
}

#[cfg(test)]
mod tests {
    use super::super::numerical_tolerance;
    use super::*;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use half::f16;
    use metal::{BufferRef, MTLCommandBufferStatus, MTLResourceOptions};

    const TOKEN_EMBEDDING_TOLERANCE_ID: &str =
        "runtime-vnext.metal.token-embedding.v1.operation.fp16.gguf-q6-k.padding";
    const TOKEN_EMBEDDING_TOLERANCE_FINGERPRINT: &str =
        "f0b7cf49cf36ae1fb1b351713bcd7d0e7c5ba60940c8789bf83d1c85a42ac9d3";
    const RMS_NORM_TOLERANCE_ID: &str =
        "runtime-vnext.metal.rms-norm.v1.operation.fp16.none.hidden-2560";
    const RMS_NORM_TOLERANCE_FINGERPRINT: &str =
        "fa18de9e42a1a74cdc0fa795a3ce94312ed7c9f8313ee40828f911a0ae89cd07";
    const RESIDUAL_ADD_TOLERANCE_ID: &str =
        "runtime-vnext.metal.residual-add.v1.operation.fp16.none.hidden-2560";
    const RESIDUAL_ADD_TOLERANCE_FINGERPRINT: &str =
        "221c4135f6edba3ad4d8e311a5042dabebcb6d9c55bfcabc2b3bdca6c8b0adba";

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
    fn native_f16_primitives_match_cpu_references_on_real_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping primitive conformance");
            return;
        };
        let pipelines = MetalPrimitivePipelines::new(&device).unwrap();
        let queue = device.new_command_queue();

        let vocabulary = 4_usize;
        let hidden = 2560_usize;
        let raw_table = (0..vocabulary * hidden)
            .map(|index| ((index as f32) * 0.013).sin())
            .collect::<Vec<_>>();
        let cpu = CandleDevice::Cpu;
        let table = Tensor::from_vec(raw_table, (vocabulary, hidden), &cpu).unwrap();
        let quantized = QTensor::quantize(&table, GgmlDType::Q6K).unwrap();
        let reference = quantized
            .dequantize(&cpu)
            .unwrap()
            .get(2)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let quantized_bytes = quantized.data().unwrap();
        let table_buffer = shared_buffer(&device, &quantized_bytes);
        let token_buffer = shared_buffer(&device, &[2_u32, u32::MAX]);
        let embedding_output = output_buffer::<f16>(&device, hidden * 2);

        let rms_input = (0..hidden)
            .map(|index| f16::from_f32((index as f32 + 1.0) / hidden as f32))
            .collect::<Vec<_>>();
        let rms_weight = (0..hidden)
            .map(|index| f16::from_f32(0.75 + index as f32 / (hidden * 2) as f32))
            .collect::<Vec<_>>();
        let rms_input_buffer = shared_buffer(&device, &rms_input);
        let rms_weight_buffer = shared_buffer(&device, &rms_weight);
        let rms_output = output_buffer::<f16>(&device, hidden);

        let residual_right = (0..hidden)
            .map(|index| f16::from_f32(-0.25 + index as f32 / hidden as f32))
            .collect::<Vec<_>>();
        let residual_right_buffer = shared_buffer(&device, &residual_right);
        let residual_output = output_buffer::<f16>(&device, hidden);

        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_raw_embedding(
            &pipelines,
            encoder,
            EmbeddingPhysicalFormat::Q6K,
            &table_buffer,
            &token_buffer,
            &embedding_output,
            EmbeddingParams {
                token_count: 2,
                hidden_size: hidden as u32,
                vocabulary_size: vocabulary as u32,
            },
        );
        dispatch_raw_rms_norm(
            &pipelines,
            encoder,
            &rms_input_buffer,
            &rms_weight_buffer,
            &rms_output,
            RmsNormParams {
                rows: 1,
                hidden_size: hidden as u32,
                epsilon: 1e-6,
            },
        );
        dispatch_raw_residual_add(
            &pipelines,
            encoder,
            &rms_output,
            &residual_right_buffer,
            &residual_output,
            ResidualAddParams {
                elements: hidden as u32,
            },
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let embedding = read_f16(&embedding_output, hidden * 2);
        let mut embedding_reference = Vec::with_capacity(hidden * 2);
        embedding_reference.extend_from_slice(&reference);
        embedding_reference.resize(hidden * 2, 0.0);
        numerical_tolerance::assert_matches(
            "Metal/CPU Q6_K token embedding",
            &embedding,
            &[2, hidden],
            &embedding_reference,
            &[2, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            TOKEN_EMBEDDING_TOLERANCE_ID,
            TOKEN_EMBEDDING_TOLERANCE_FINGERPRINT,
        )
        .expect("reviewed token-embedding numerical contract");
        assert!(embedding[hidden..].iter().all(|value| *value == 0.0));

        let rms = read_f16(&rms_output, hidden);
        let mean_square = rms_input
            .iter()
            .map(|value| value.to_f32().powi(2))
            .sum::<f32>()
            / hidden as f32;
        let inverse_rms = (mean_square + 1e-6).sqrt().recip();
        let rms_reference = (0..hidden)
            .map(|index| rms_input[index].to_f32() * inverse_rms * rms_weight[index].to_f32())
            .collect::<Vec<_>>();
        numerical_tolerance::assert_matches(
            "Metal/CPU RMSNorm",
            &rms,
            &[1, hidden],
            &rms_reference,
            &[1, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            RMS_NORM_TOLERANCE_ID,
            RMS_NORM_TOLERANCE_FINGERPRINT,
        )
        .expect("reviewed RMSNorm numerical contract");
        let residual = read_f16(&residual_output, hidden);
        let residual_reference = (0..hidden)
            .map(|index| rms[index] + residual_right[index].to_f32())
            .collect::<Vec<_>>();
        numerical_tolerance::assert_matches(
            "Metal/CPU residual add",
            &residual,
            &[1, hidden],
            &residual_reference,
            &[1, hidden],
            numerical_tolerance::LogicalDtype::Fp16,
            RESIDUAL_ADD_TOLERANCE_ID,
            RESIDUAL_ADD_TOLERANCE_FINGERPRINT,
        )
        .expect("reviewed residual-add numerical contract");
    }

    fn set_raw(encoder: &ComputeCommandEncoderRef, index: u64, buffer: &BufferRef) {
        encoder.set_buffer(index, Some(buffer), 0);
    }

    fn dispatch_raw_embedding(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        format: EmbeddingPhysicalFormat,
        table: &BufferRef,
        token_ids: &BufferRef,
        output: &BufferRef,
        params: EmbeddingParams,
    ) {
        encoder.set_compute_pipeline_state(match format {
            EmbeddingPhysicalFormat::DenseF16 => &pipelines.embedding_dense,
            EmbeddingPhysicalFormat::Q6K => &pipelines.embedding_q6_k,
        });
        set_raw(encoder, 0, table);
        set_raw(encoder, 1, token_ids);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<EmbeddingParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(
                u64::from(params.hidden_size).div_ceil(THREADS_PER_GROUP),
                u64::from(params.token_count),
                1,
            ),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_rms_norm(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        params: RmsNormParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.rms_norm);
        set_raw(encoder, 0, input);
        set_raw(encoder, 1, weight);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<RmsNormParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.rows), 1, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }

    fn dispatch_raw_residual_add(
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        left: &BufferRef,
        right: &BufferRef,
        output: &BufferRef,
        params: ResidualAddParams,
    ) {
        encoder.set_compute_pipeline_state(&pipelines.residual_add);
        set_raw(encoder, 0, left);
        set_raw(encoder, 1, right);
        set_raw(encoder, 2, output);
        encoder.set_bytes(
            3,
            std::mem::size_of::<ResidualAddParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(params.elements).div_ceil(THREADS_PER_GROUP), 1, 1),
            MTLSize::new(THREADS_PER_GROUP, 1, 1),
        );
    }
}
