use super::super::{
    classify_device_error, BufferDescriptor, BufferUsage, CompletionReservation,
    DefinitelyNotSubmittedWaveRetryAuthority, DeviceBatchingForm, DeviceCommandBatch,
    DeviceCommandLogicalWork, DeviceRuntime, ExecutablePlanView, HostTransferLayout,
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ResourceWorkShape,
};
use super::buffer_view::OperationBufferView;
use super::dispatch_contract::{
    DispatchRetryAuthority, OperationDispatchError, SubmissionScratchInitialization,
    SubmissionWaveDispatchError,
};
use super::foundation::invalid_operation;
use super::{BatchOperationIdentity, BatchOperationNodeIdentity, ElementType};

pub(super) fn encode_provider_workspace_initialization<R, Retry>(
    runtime: &R,
    node_index: u32,
    node_identity: &BatchOperationNodeIdentity,
    requirement: &ProviderWorkspaceRequirement,
    work: &ResourceWorkShape,
    view: &OperationBufferView<'_, R::Buffer>,
    initialization: SubmissionScratchInitialization,
    commands: &mut DeviceCommandBatch<R::Command>,
) -> Result<usize, OperationDispatchError<R, Retry>>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    if requirement.reuse_policy() == ProviderWorkspaceReusePolicy::Preserve {
        return Err(OperationDispatchError::Contract(invalid_operation(
            "scratch workspace cannot preserve bytes across invocations",
        )));
    }
    if initialization == SubmissionScratchInitialization::ProviderContract
        && requirement.reuse_policy() != ProviderWorkspaceReusePolicy::ZeroBeforeUse
    {
        return Ok(0);
    }

    let required_bytes = requirement
        .evaluate_bytes(work)
        .map_err(OperationDispatchError::Contract)?;
    let descriptor = view.descriptor();
    if descriptor.usage != BufferUsage::Scratch
        || descriptor.element_type != ElementType::U8
        || descriptor.size_bytes != required_bytes
        || descriptor.alignment_bytes < requirement.alignment_bytes()
        || descriptor.alignment_bytes % requirement.alignment_bytes() != 0
    {
        return Err(OperationDispatchError::Contract(invalid_operation(
            "scratch workspace zero range differs from its provider requirement",
        )));
    }
    let regions = view
        .translate(0, required_bytes)
        .map_err(OperationDispatchError::Contract)?;
    let participant = node_identity.participants().first().ok_or_else(|| {
        OperationDispatchError::Contract(invalid_operation(
            "scratch workspace initialization has no participant identity",
        ))
    })?;
    let participant_count = u32::try_from(node_identity.participants().len()).map_err(|_| {
        OperationDispatchError::Contract(invalid_operation(
            "scratch workspace participant count exceeds u32",
        ))
    })?;
    if participant_count != work.immediate_sequences() {
        return Err(OperationDispatchError::Contract(invalid_operation(
            "scratch workspace logical participants differ from its resource work",
        )));
    }
    let logical_work = DeviceCommandLogicalWork::new(
        DeviceBatchingForm::Packed,
        participant_count,
        work.immediate_tokens(),
    )
    .map_err(OperationDispatchError::Contract)?;
    let identity = participant.identity().clone();
    let mut encoded_bytes = 0_u64;
    let mut command_count = 0_usize;
    for region in regions.iter() {
        let (buffer, physical_range, _retention) = region.buffer_and_physical_range();
        let actual = runtime.buffer_descriptor(buffer);
        if actual.usage != BufferUsage::Scratch
            || actual.element_type != ElementType::U8
            || physical_range.end > actual.size_bytes
            || physical_range.start >= physical_range.end
        {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "scratch workspace physical zero range drifted",
            )));
        }
        let length_bytes = physical_range.end - physical_range.start;
        match initialization {
            SubmissionScratchInitialization::ProviderContract
            | SubmissionScratchInitialization::FillByte(0) => {
                let command = runtime
                    .encode_zero(buffer, physical_range.start, length_bytes)
                    .map_err(|error| {
                        classify_device_error(runtime, identity.clone(), &error)
                            .map(OperationDispatchError::Initialization)
                            .unwrap_or_else(OperationDispatchError::Contract)
                    })?;
                commands.push_node_initialization(node_index, logical_work, command);
                command_count = command_count.checked_add(1).ok_or_else(|| {
                    OperationDispatchError::Contract(invalid_operation(
                        "scratch workspace initialization command count overflows usize",
                    ))
                })?;
            }
            SubmissionScratchInitialization::FillByte(value) => {
                const FILL_CHUNK_BYTES: u64 = 1024 * 1024;
                let chunk_len =
                    usize::try_from(length_bytes.min(FILL_CHUNK_BYTES)).map_err(|_| {
                        OperationDispatchError::Contract(invalid_operation(
                            "scratch workspace fill chunk exceeds host address space",
                        ))
                    })?;
                let fill = vec![value; chunk_len];
                let mut offset = physical_range.start;
                let end = physical_range.end;
                while offset < end {
                    let piece_bytes = (end - offset).min(FILL_CHUNK_BYTES);
                    let piece_len = usize::try_from(piece_bytes).map_err(|_| {
                        OperationDispatchError::Contract(invalid_operation(
                            "scratch workspace fill piece exceeds host address space",
                        ))
                    })?;
                    let layout = HostTransferLayout::new(ElementType::U8, piece_bytes)
                        .map_err(OperationDispatchError::Contract)?;
                    let command = runtime
                        .encode_upload(&fill[..piece_len], layout, buffer, offset)
                        .map_err(|error| {
                            classify_device_error(runtime, identity.clone(), &error)
                                .map(OperationDispatchError::Initialization)
                                .unwrap_or_else(OperationDispatchError::Contract)
                        })?;
                    commands.push_node_initialization(node_index, logical_work, command);
                    command_count = command_count.checked_add(1).ok_or_else(|| {
                        OperationDispatchError::Contract(invalid_operation(
                            "scratch workspace initialization command count overflows usize",
                        ))
                    })?;
                    offset = offset.checked_add(piece_bytes).ok_or_else(|| {
                        OperationDispatchError::Contract(invalid_operation(
                            "scratch workspace fill offset overflows u64",
                        ))
                    })?;
                }
            }
        }
        encoded_bytes = encoded_bytes.checked_add(length_bytes).ok_or_else(|| {
            OperationDispatchError::Contract(invalid_operation(
                "scratch workspace initialization byte count overflows u64",
            ))
        })?;
    }
    if encoded_bytes != required_bytes {
        return Err(OperationDispatchError::Contract(invalid_operation(
            "scratch workspace initialization commands do not cover the logical workspace",
        )));
    }
    Ok(command_count)
}

pub(super) fn encode_submission_wave_workspace_initializations<R>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    batch_identity: &BatchOperationIdentity,
    scratch_initialization: SubmissionScratchInitialization,
    completion: &CompletionReservation<R>,
    commands: &mut DeviceCommandBatch<R::Command>,
) -> Result<usize, SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
{
    let plan_nodes = resolved.execution_plan().payload().nodes();
    if batch_identity.node_count() != completion.wave().nodes().len() {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            "workspace initialization topology differs from the prepared wave",
        )));
    }
    let mut command_count = 0_usize;
    for (node_index, prepared_node) in completion.wave().nodes().iter().enumerate() {
        let plan_node = plan_nodes
            .iter()
            .find(|node| node.id() == prepared_node.node_id())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(
                    "workspace initialization node is absent from the immutable plan",
                ))
            })?;
        let node_identity = batch_identity.nodes().get(node_index).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "workspace initialization node has no batch identity",
            ))
        })?;
        if plan_node.id() != prepared_node.node_id()
            || node_identity.node_id() != prepared_node.node_id()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "workspace initialization node differs from the prepared wave",
            )));
        }
        let Some(requirement) = plan_node.provider_resources().scratch() else {
            continue;
        };
        if scratch_initialization == SubmissionScratchInitialization::ProviderContract
            && requirement.reuse_policy() != ProviderWorkspaceReusePolicy::ZeroBeforeUse
        {
            continue;
        }
        let resource_id = plan_node.scratch_resource().ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "scratch workspace initialization has no base resource",
            ))
        })?;
        let backing = completion
            .wave()
            .backing_view(node_index, resource_id)
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let descriptor = BufferDescriptor {
            resource_id: resource_id.clone(),
            size_bytes: backing.size_bytes(),
            alignment_bytes: backing.alignment_bytes(),
            usage: backing.usage(),
            element_type: backing.element_type(),
        };
        let view = if backing.capacity_size_bytes() > backing.size_bytes() {
            OperationBufferView::from_backing_prefix(descriptor, backing)
        } else {
            OperationBufferView::from_backing_exact(descriptor, backing)
        };
        let node_index = u32::try_from(node_index).map_err(|_| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "workspace initialization node index exceeds u32",
            ))
        })?;
        let encoded = encode_provider_workspace_initialization::<
            R,
            DefinitelyNotSubmittedWaveRetryAuthority<R>,
        >(
            runtime,
            node_index,
            node_identity,
            requirement,
            prepared_node.work_shape().resource_work(),
            &view,
            scratch_initialization,
            commands,
        )?;
        command_count = command_count.checked_add(encoded).ok_or_else(|| {
            SubmissionWaveDispatchError::Contract(invalid_operation(
                "workspace initialization command count overflows usize",
            ))
        })?;
    }
    Ok(command_count)
}
