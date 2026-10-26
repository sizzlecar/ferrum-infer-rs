//! NCHW providers share dense-port checks and explicit invocation workspace.
use super::*;
use ferrum_runtime::*;

mod attention;
mod channel;
mod conv;
mod mean;
#[cfg(test)]
mod tests;
mod upsample;

pub(super) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("spatial/attention.rs"),
    include_str!("spatial/attention.metal"),
    include_str!("spatial/channel.rs"),
    include_str!("spatial/channel.metal"),
    include_str!("spatial/conv.rs"),
    include_str!("spatial/conv.metal"),
    include_str!("spatial/upsample.rs"),
    include_str!("spatial/upsample.metal"),
    include_str!("spatial/mean.rs"),
    include_str!("spatial/mean.metal")
);

pub(super) struct Pipelines {
    conv: Option<ComputePipelineState>,
    conv_1x1: Option<ComputePipelineState>,
    conv_3x3: Option<ComputePipelineState>,
    conv_3x3_masked: Option<ComputePipelineState>,
    pad: Option<ComputePipelineState>,
    channel: Option<ComputePipelineState>,
    channel_shared_limit: u64,
    attention: Option<ComputePipelineState>,
    upsample: Option<ComputePipelineState>,
    mean: Option<ComputePipelineState>,
}
pub(super) fn append_source(selected: &[MetalTensorOperation], source: &mut String) {
    for (enabled, text) in [
        (
            selected.contains(&MetalTensorOperation::SpatialGroupMean),
            include_str!("spatial/mean.metal"),
        ),
        (
            selected.contains(&MetalTensorOperation::Conv2d),
            include_str!("spatial/conv.metal"),
        ),
        (
            selected.iter().any(|k| k.is_channel_norm()),
            include_str!("spatial/channel.metal"),
        ),
        (
            selected.contains(&MetalTensorOperation::SpatialAttention),
            include_str!("spatial/attention.metal"),
        ),
        (
            selected.contains(&MetalTensorOperation::SpatialUpsampleGather),
            include_str!("spatial/upsample.metal"),
        ),
    ] {
        if enabled {
            source.push_str(text);
        }
    }
}
pub(super) fn pipelines(
    runtime: &MetalDeviceRuntime,
    selected: &[MetalTensorOperation],
    library: &metal::LibraryRef,
) -> Result<Pipelines, MetalDeviceRuntimeError> {
    let create = |enabled: bool, name: &str| -> Result<_, MetalDeviceRuntimeError> {
        if !enabled {
            return Ok(None);
        }
        let function = library
            .get_function(name, Some(metal::FunctionConstantValues::new()))
            .map_err(MetalDeviceRuntimeError::contract)?;
        runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map(Some)
            .map_err(MetalDeviceRuntimeError::contract)
    };
    let conv = create(
        selected.contains(&MetalTensorOperation::Conv2d),
        "ferrum_tensor_conv2d_bf16",
    )?;
    let (conv_1x1, conv_3x3, conv_3x3_masked) = if conv.is_some() {
        (
            conv::direct_square_pipeline(runtime, library, 1)?,
            conv::direct_square_pipeline(runtime, library, 3)?,
            conv::masked_3x3_pipeline(runtime, library)?,
        )
    } else {
        (None, None, None)
    };
    let pad = create(conv.is_some(), "ferrum_tensor_pad_nchw_bf16")?;
    let channel = create(
        selected.iter().any(|k| k.is_channel_norm()),
        "ferrum_tensor_channel_l2_bf16",
    )?;
    let attention = create(
        selected.contains(&MetalTensorOperation::SpatialAttention),
        "ferrum_tensor_spatial_attention_bf16",
    )?;
    let upsample = create(
        selected.contains(&MetalTensorOperation::SpatialUpsampleGather),
        "ferrum_tensor_spatial_upsample_bf16",
    )?;
    for p in [&conv, &attention].into_iter().flatten() {
        if p.thread_execution_width() != 32
            || p.max_total_threads_per_threadgroup() < 128
            || p.static_threadgroup_memory_length()
                > runtime.device().max_threadgroup_memory_length()
        {
            return Err(MetalDeviceRuntimeError::contract(
                "Metal spatial GEMM needs four 32-lane groups and its declared on-chip storage",
            ));
        }
    }
    let channel_shared_limit = channel
        .as_ref()
        .map(|p| {
            runtime
                .device()
                .max_threadgroup_memory_length()
                .checked_sub(p.static_threadgroup_memory_length())
                .ok_or_else(|| {
                    MetalDeviceRuntimeError::contract("Metal channel norm exceeds on-chip memory")
                })
        })
        .transpose()?
        .unwrap_or(0);
    let mean = create(
        selected.contains(&MetalTensorOperation::SpatialGroupMean),
        "ferrum_tensor_spatial_group_mean_bf16",
    )?;
    Ok(Pipelines {
        mean,
        conv,
        conv_1x1,
        conv_3x3,
        conv_3x3_masked,
        pad,
        channel,
        channel_shared_limit,
        attention,
        upsample,
    })
}
pub(super) fn scratch_bytes(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    pipelines: &Pipelines,
) -> Result<u64, String> {
    use MetalTensorOperation::*;
    match kind {
        Conv2d => {
            Ok(conv::geometry(values, attrs, pipelines.conv_3x3_masked.is_some())?.scratch_bytes)
        }
        ChannelL2Norm | ChannelL2NormImage => {
            channel::geometry(kind, values, attrs, pipelines.channel_shared_limit).map(|_| 0)
        }
        SpatialAttention => Ok(attention::geometry(values, attrs)?.scratch_bytes),
        SpatialUpsampleGather => upsample::geometry(values, attrs).map(|_| 0),
        SpatialGroupMean => mean::geometry(values, attrs).map(|_| 0),
        _ => Err("not a spatial tensor operation".into()),
    }
}
pub(super) fn estimate_workspace(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
    estimate: OperationResourceEstimate,
    bytes: u64,
) -> Result<OperationResourceEstimate, VNextError> {
    if bytes == 0 {
        return Ok(estimate);
    }
    let scratch = ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::fixed(bytes)?,
        16,
        ProviderWorkspaceScope::Invocation,
        ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
        DynamicStorageRequirement::contiguous(),
    )?;
    Ok(OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        request.input_fingerprint(),
        16,
        Some(scratch),
        None,
    ))
}
pub(super) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<super::Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    match kind {
        MetalTensorOperation::Conv2d => conv::encode(participant, pipelines),
        MetalTensorOperation::SpatialAttention => attention::encode(participant, pipelines),
        MetalTensorOperation::SpatialUpsampleGather => upsample::encode(participant, pipelines),
        MetalTensorOperation::SpatialGroupMean => mean::encode(participant, pipelines),
        _ if kind.is_channel_norm() => channel::encode(kind, participant, pipelines),
        _ => Err("not a spatial tensor operation".into()),
    }
}

fn dense_ports(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
) -> Result<Vec<u64>, String> {
    if values.len() != kind.inputs() as usize + 1 {
        return Err("Metal spatial port count differs".into());
    }
    (0..=kind.inputs())
        .map(|port| {
            let (role, ordinal) = if port == kind.inputs() {
                (ResolvedValueRole::Output, 0)
            } else {
                (ResolvedValueRole::Input, port)
            };
            let v = binding(values, role, ordinal)?;
            if v.tensor().element_type() != ElementType::Bf16
                || v.tensor().layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err("Metal spatial ports require contiguous BF16".into());
            }
            if let Some(weight) = v.weight() {
                weight
                    .validate_logical(v.tensor().dimensions(), ElementType::Bf16)
                    .map_err(|e| e.to_string())?;
                if !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. }) {
                    return Err("Metal spatial ports cannot read packed weight storage".into());
                }
            }
            let bytes = v
                .tensor()
                .minimum_storage_bytes()
                .map_err(|e| e.to_string())?;
            if bytes > isize::MAX as u64 {
                return Err("Metal spatial byte extent exceeds address space".into());
            }
            Ok(bytes)
        })
        .collect()
}
fn regions(
    participant: &TensorParticipant<'_>,
    bytes: &[u64],
) -> Result<Vec<MetalBufferRegion>, String> {
    let mut regions = Vec::with_capacity(bytes.len() + 1);
    for (port, &bytes) in bytes.iter().enumerate() {
        let (role, ordinal) = if port + 1 == participant.bindings().len() {
            (ResolvedValueRole::Output, 0)
        } else {
            (ResolvedValueRole::Input, port as u32)
        };
        regions.push(tensor_port_region(
            participant,
            role,
            ordinal,
            ElementType::Bf16,
            bytes,
        )?);
    }
    for input in &regions[..regions.len() - 1] {
        disjoint(input, regions.last().unwrap())?;
    }
    Ok(regions)
}
fn append_scratch(
    participant: &TensorParticipant<'_>,
    bytes: u64,
    regions: &mut Vec<MetalBufferRegion>,
) -> Result<(), String> {
    let scratch = contiguous_scratch_region(participant, bytes)?;
    if !scratch.offset_bytes().is_multiple_of(16) {
        return Err("Metal spatial scratch needs 16-byte alignment".into());
    }
    for region in regions.iter() {
        disjoint(region, &scratch)?;
    }
    regions.push(scratch);
    Ok(())
}
