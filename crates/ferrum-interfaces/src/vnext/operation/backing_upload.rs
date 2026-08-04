use super::super::{
    classify_device_error, BufferUsage, DeviceRuntime, ExecutionIdentityEnvelope,
    HostTransferLayout, LogicalBackingBufferView,
};
use super::dispatch_contract::SubmissionWaveDispatchError;
use super::foundation::invalid_operation;
use super::storage_profile::ElementType;

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_submission_wave_backing_upload<R>(
    runtime: &R,
    identity: &ExecutionIdentityEnvelope,
    backing: &LogicalBackingBufferView<'_, R::Buffer>,
    expected_usage: BufferUsage,
    element_type: ElementType,
    logical_offset_bytes: u64,
    bytes: &[u8],
    context: &'static str,
    mut push: impl FnMut(R::Command),
) -> Result<usize, SubmissionWaveDispatchError<R>>
where
    R: DeviceRuntime,
{
    let byte_len = u64::try_from(bytes.len()).map_err(|_| {
        SubmissionWaveDispatchError::Contract(invalid_operation(format!(
            "{context} byte length exceeds u64"
        )))
    })?;
    let element_bytes = element_type.size_bytes();
    let destination_end = logical_offset_bytes.checked_add(byte_len).ok_or_else(|| {
        SubmissionWaveDispatchError::Contract(invalid_operation(format!(
            "{context} destination range overflows"
        )))
    })?;
    if byte_len == 0
        || byte_len % element_bytes != 0
        || backing.usage() != expected_usage
        || backing.element_type() != element_type
        || destination_end > backing.size_bytes()
    {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            format!("{context} differs from its resolved logical backing"),
        )));
    }

    let mut logical_cursor = 0_u64;
    let mut encoded_bytes = 0_u64;
    let mut command_count = 0_usize;
    for segment in backing.segment_bindings() {
        let segment_end = logical_cursor
            .checked_add(segment.segment().length_bytes())
            .ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "{context} backing coverage overflows"
                )))
            })?;
        let overlap_start = logical_cursor.max(logical_offset_bytes);
        let overlap_end = segment_end.min(destination_end);
        if overlap_start < overlap_end {
            let source_start =
                usize::try_from(overlap_start - logical_offset_bytes).map_err(|_| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} source offset exceeds host address space"
                    )))
                })?;
            let piece_bytes = overlap_end - overlap_start;
            let source_end = source_start
                .checked_add(usize::try_from(piece_bytes).map_err(|_| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} piece exceeds host address space"
                    )))
                })?)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} source range overflows"
                    )))
                })?;
            let destination_offset = segment
                .segment()
                .offset_bytes()
                .checked_add(overlap_start - logical_cursor)
                .ok_or_else(|| {
                    SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                        "{context} physical offset overflows"
                    )))
                })?;
            if piece_bytes % element_bytes != 0
                || destination_offset % element_bytes != 0
                || source_end > bytes.len()
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    format!("{context} splits an element or exceeds its source"),
                )));
            }
            let actual = runtime.buffer_descriptor(segment.buffer());
            if &actual != segment.descriptor()
                || destination_offset
                    .checked_add(piece_bytes)
                    .is_none_or(|end| end > actual.size_bytes)
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    format!("{context} backing descriptor drifted"),
                )));
            }
            let layout = HostTransferLayout::new(element_type, piece_bytes / element_bytes)
                .map_err(SubmissionWaveDispatchError::Contract)?;
            let command = runtime
                .encode_upload(
                    &bytes[source_start..source_end],
                    layout,
                    segment.buffer(),
                    destination_offset,
                )
                .map_err(|error| {
                    classify_device_error(runtime, identity.clone(), &error)
                        .map(SubmissionWaveDispatchError::InputUpload)
                        .unwrap_or_else(SubmissionWaveDispatchError::Contract)
                })?;
            push(command);
            command_count = command_count.checked_add(1).ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "{context} command count overflows usize"
                )))
            })?;
            encoded_bytes = encoded_bytes.checked_add(piece_bytes).ok_or_else(|| {
                SubmissionWaveDispatchError::Contract(invalid_operation(format!(
                    "{context} encoded byte count overflows"
                )))
            })?;
        }
        logical_cursor = segment_end;
    }
    if encoded_bytes != byte_len {
        return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
            format!("{context} backing does not cover its complete range"),
        )));
    }
    Ok(command_count)
}
