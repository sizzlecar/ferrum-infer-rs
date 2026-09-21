//! Per-submission device inputs. Copies deliberately remain outside reusable
//! compute: each invocation binds its exact retained parent output.
use super::{foundation::invalid_operation, translate_step_participant_upload_range};
use crate::vnext::{
    classify_device_error, AllocationKind, AllocationLifetime, BackingChunkIdentity,
    BatchOperationIdentity, BufferUsage, CompletionReservation, CopyRegion, DeviceCommandBatch,
    DeviceRuntime, ElementType, ExecutablePlanView, LogicalBackingBufferView, NodeId,
    ResolvedValueRole, SubmissionWaveDispatchError, SubmissionWaveInputUpload, TensorAccess,
    VNextError,
};
use std::ops::Range;

struct PhysicalRange {
    chunk: BackingChunkIdentity,
    range: Range<u64>,
}

impl PhysicalRange {
    fn overlaps(&self, other: &Self) -> bool {
        self.chunk == other.chunk
            && self.range.start < other.range.end
            && other.range.start < self.range.end
    }
}

/// Resolves a declared input using exactly the host-upload coordinate rules.
#[allow(clippy::too_many_arguments)]
fn input_destination<'a, R: DeviceRuntime>(
    resolved: &dyn ExecutablePlanView,
    completion: &'a CompletionReservation<R>,
    node_id: &NodeId,
    participant_index: u32,
    ordinal: u32,
    offset: u64,
    element_type: ElementType,
    byte_len: u64,
) -> Result<(LogicalBackingBufferView<'a, R::Buffer>, Range<u64>), VNextError> {
    let node = resolved
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|node| node.id() == node_id)
        .ok_or_else(|| invalid_operation("forwarded input references an unknown plan node"))?;
    let value = node
        .values()
        .iter()
        .find(|value| value.role() == ResolvedValueRole::Input && value.ordinal() == ordinal)
        .ok_or_else(|| invalid_operation("forwarded input references an unknown node input"))?;
    let [component] = value.storage().components() else {
        return Err(invalid_operation(
            "forwarded input requires one activation storage component",
        ));
    };
    if byte_len == 0
        || value.usage() != BufferUsage::Activations
        || !matches!(value.access(), TensorAccess::Read | TensorAccess::ReadWrite)
        || value.tensor().element_type() != element_type
        || component.element_type() != element_type
        || offset
            .checked_add(byte_len)
            .is_none_or(|end| end > component.length_bytes())
    {
        return Err(invalid_operation(
            "forwarded input differs from its resolved activation binding",
        ));
    }
    let descriptor = completion
        .wave()
        .step_resources()
        .dynamic_descriptor(component.resource_id())?;
    let start = component
        .offset_bytes()
        .checked_add(offset)
        .ok_or_else(|| invalid_operation("forwarded input destination offset overflows"))?;
    let end = start
        .checked_add(byte_len)
        .ok_or_else(|| invalid_operation("forwarded input destination extent overflows"))?;
    let destination = if descriptor.lifetime() == AllocationLifetime::Step
        && descriptor.kind() == &AllocationKind::Value
    {
        let work = completion
            .wave()
            .nodes()
            .iter()
            .find(|node| node.node_id() == node_id)
            .ok_or_else(|| invalid_operation("forwarded input has no prepared node"))?
            .work_shape();
        translate_step_participant_upload_range(
            descriptor.demand(),
            work,
            participant_index as usize,
            start..end,
        )?
    } else {
        start..end
    };
    let view = completion.backing_view(node_id, participant_index, component.resource_id())?;
    if view.usage() != BufferUsage::Activations
        || view.element_type() != element_type
        || destination.end.checked_sub(destination.start) != Some(byte_len)
        || destination.end > view.size_bytes()
    {
        return Err(invalid_operation(
            "forwarded input backing differs from its resolved activation",
        ));
    }
    Ok((view, destination))
}

fn physical_ranges<R: DeviceRuntime>(
    runtime: &R,
    view: &LogicalBackingBufferView<'_, R::Buffer>,
    range: Range<u64>,
) -> Result<Vec<PhysicalRange>, VNextError> {
    let mut result = Vec::new();
    let mut logical_start = 0_u64;
    let mut covered = 0_u64;
    for binding in view.segment_bindings() {
        let segment = binding.segment();
        let logical_end = logical_start
            .checked_add(segment.length_bytes())
            .ok_or_else(|| invalid_operation("forwarded input backing coverage overflows"))?;
        let start = logical_start.max(range.start);
        let end = logical_end.min(range.end);
        if start < end {
            let physical_start = segment
                .offset_bytes()
                .checked_add(start - logical_start)
                .ok_or_else(|| invalid_operation("forwarded input physical offset overflows"))?;
            let physical_end = physical_start
                .checked_add(end - start)
                .ok_or_else(|| invalid_operation("forwarded input physical extent overflows"))?;
            let actual = runtime.buffer_descriptor(binding.buffer());
            if &actual != binding.descriptor()
                || actual.usage != view.usage()
                || actual.element_type != view.element_type()
                || physical_end > actual.size_bytes
                || physical_start % view.element_type().size_bytes() != 0
                || (end - start) % view.element_type().size_bytes() != 0
            {
                return Err(invalid_operation(
                    "forwarded input backing descriptor or alignment drifted",
                ));
            }
            result.push(PhysicalRange {
                chunk: binding.chunk().clone(),
                range: physical_start..physical_end,
            });
            covered += end - start;
        }
        logical_start = logical_end;
    }
    if range.end.checked_sub(range.start) != Some(covered) || covered == 0 {
        return Err(invalid_operation(
            "forwarded input backing does not cover the complete range",
        ));
    }
    Ok(result)
}

pub(super) fn encode_submission_wave_forwarded_inputs<R: DeviceRuntime>(
    runtime: &R,
    resolved: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    completion: &CompletionReservation<R>,
    uploads: &[SubmissionWaveInputUpload],
    commands: &mut DeviceCommandBatch<R::Command>,
) -> Result<(), SubmissionWaveDispatchError<R>> {
    let contract = SubmissionWaveDispatchError::Contract;
    let wave = completion.wave();
    let work = wave.step_resources().work_shape().participant_work();
    let forwards = wave.forwarded_inputs();
    if forwards.is_empty() {
        return if work
            .iter()
            .any(|work| work.token_span().submitted_token_source().is_some())
        {
            Err(contract(invalid_operation(
                "each device-source work row requires exactly one matching forwarded input",
            )))
        } else {
            Ok(())
        };
    }
    let mut rows = vec![false; work.len()];
    for forward in forwards {
        forward.validate_for_wave(wave).map_err(contract)?;
        let row = forward.source().participant_index() as usize;
        if rows.get(row).is_none_or(|seen| *seen) {
            return Err(contract(invalid_operation(
                "forwarded inputs repeat or exceed the admitted cohort",
            )));
        }
        rows[row] = true;
        if uploads.iter().any(|upload| {
            upload.participant_index() as usize == row
                && upload.node_id() == forward.node_id()
                && upload.input_ordinal() == forward.input_ordinal()
        }) {
            return Err(contract(invalid_operation(
                "one child token input cannot have both host and device sources",
            )));
        }
    }
    if work
        .iter()
        .zip(&rows)
        .any(|(work, forwarded)| work.token_span().submitted_token_source().is_some() != *forwarded)
    {
        return Err(contract(invalid_operation(
            "each device-source work row requires exactly one matching forwarded input",
        )));
    }

    // Keep views alive until all copy commands are encoded. Their underlying
    // claims remain owned by the wave/predecessor through terminal completion.
    let mut bindings = Vec::with_capacity(forwards.len());
    let mut sources = Vec::with_capacity(forwards.len());
    let mut destinations = Vec::with_capacity(forwards.len());
    for forward in forwards {
        let row = forward.source().participant_index();
        let node_index = identity.node_index(forward.node_id()).ok_or_else(|| {
            contract(invalid_operation(
                "forwarded input has no physical node identity",
            ))
        })?;
        let node_identity = identity.materialize_node(node_index).map_err(contract)?;
        if node_identity.participants().get(row as usize).is_none() {
            return Err(contract(invalid_operation(
                "forwarded input has no physical participant",
            )));
        }
        let (source, source_range) = forward.source_view().map_err(contract)?;
        let (destination, destination_range) = input_destination(
            resolved,
            completion,
            forward.node_id(),
            row,
            forward.input_ordinal(),
            forward.logical_offset_bytes(),
            ElementType::U32,
            4,
        )
        .map_err(contract)?;
        let mut physical_source =
            physical_ranges(runtime, &source, source_range).map_err(contract)?;
        let mut physical_destination =
            physical_ranges(runtime, &destination, destination_range).map_err(contract)?;
        if physical_source.len() != 1 || physical_destination.len() != 1 {
            return Err(contract(invalid_operation(
                "one forwarded U32 token cannot straddle physical allocations",
            )));
        }
        sources.push(physical_source.remove(0));
        destinations.push(physical_destination.remove(0));
        bindings.push((source, destination, node_identity));
    }
    for (index, destination) in destinations.iter().enumerate() {
        if sources.iter().any(|source| source.overlaps(destination))
            || destinations[..index]
                .iter()
                .any(|previous| previous.overlaps(destination))
        {
            return Err(contract(invalid_operation(
                "forwarded inputs alias a source or another destination",
            )));
        }
    }
    // Compare physical ranges as well as semantic ordinals: two declared
    // inputs can alias the same allocation through different storage views.
    for upload in uploads {
        let (view, range) = input_destination(
            resolved,
            completion,
            upload.node_id(),
            upload.participant_index(),
            upload.input_ordinal(),
            upload.logical_offset_bytes(),
            upload.source_layout().element_type(),
            upload.source_layout().byte_len().map_err(contract)?,
        )
        .map_err(contract)?;
        for host_range in physical_ranges(runtime, &view, range).map_err(contract)? {
            if sources
                .iter()
                .chain(&destinations)
                .any(|device_range| device_range.overlaps(&host_range))
            {
                return Err(contract(invalid_operation(
                    "host input upload aliases a forwarded token source or destination",
                )));
            }
        }
    }
    for (index, ((source, destination, node_identity), forward)) in
        bindings.iter().zip(forwards).enumerate()
    {
        let source_range = &sources[index];
        let destination_range = &destinations[index];
        let source_buffer = source
            .segment_bindings()
            .iter()
            .find(|binding| {
                binding.chunk() == &source_range.chunk
                    && binding.segment().offset_bytes() <= source_range.range.start
                    && binding
                        .segment()
                        .offset_bytes()
                        .checked_add(binding.segment().length_bytes())
                        .is_some_and(|end| end >= source_range.range.end)
            })
            .ok_or_else(|| {
                contract(invalid_operation(
                    "forwarded source lost its physical binding",
                ))
            })?;
        let destination_buffer = destination
            .segment_bindings()
            .iter()
            .find(|binding| {
                binding.chunk() == &destination_range.chunk
                    && binding.segment().offset_bytes() <= destination_range.range.start
                    && binding
                        .segment()
                        .offset_bytes()
                        .checked_add(binding.segment().length_bytes())
                        .is_some_and(|end| end >= destination_range.range.end)
            })
            .ok_or_else(|| {
                contract(invalid_operation(
                    "forwarded destination lost its physical binding",
                ))
            })?;
        let region = CopyRegion::new(source_range.range.start, destination_range.range.start, 4)
            .map_err(contract)?;
        region
            .validate_bounds(source_buffer.descriptor(), destination_buffer.descriptor())
            .map_err(contract)?;
        let participant =
            &node_identity.participants()[forward.source().participant_index() as usize];
        let command = runtime
            .encode_copy(source_buffer.buffer(), destination_buffer.buffer(), region)
            .map_err(|error| {
                classify_device_error(runtime, participant.identity().clone(), &error)
                    .map(SubmissionWaveDispatchError::InputUpload)
                    .unwrap_or_else(contract)
            })?;
        commands.push_dynamic_binding(command);
    }
    Ok(())
}
