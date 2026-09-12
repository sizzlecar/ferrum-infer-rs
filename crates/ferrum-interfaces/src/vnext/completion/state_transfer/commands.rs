//! Native state-copy encoding without submission or completion authority.

use super::super::CheckpointCopyGeometry;
use super::{invalid_completion, StateTransferKind};
use crate::vnext::{
    AllocationLifetime, BufferDescriptor, CheckpointBackingOwner, CopyRegion,
    DeviceBufferRetention, DeviceCommandBatch, DeviceRuntime, ExecutionLane,
    LogicalBackingBufferView, OperationBufferView, PreparedSequenceStateTransfer, ResourceId,
    SequenceCheckpointBytePlan, VNextError,
};
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum StateTransferCopyEncodeError<E> {
    Contract(VNextError),
    Runtime {
        resource_id: ResourceId,
        region: CopyRegion,
        error: E,
    },
}

impl<E> From<VNextError> for StateTransferCopyEncodeError<E> {
    fn from(error: VNextError) -> Self {
        Self::Contract(error)
    }
}

/// Physical segments and both logical backing owners survive borrowed views
/// and encoding. The native completion lease separately owns the exclusive
/// sequence guard and capture-destination permission until known quiescence.
#[must_use = "copy retentions must remain in the native completion lease until quiescence"]
pub(super) struct StateTransferCopyRetentions {
    _owners: Vec<DeviceBufferRetention>,
}

#[must_use = "encoded copies must be retained through submission or discarded before submission"]
pub(super) struct PreparedStateTransferCopies<R: DeviceRuntime> {
    commands: Vec<R::Command>,
    geometry: CheckpointCopyGeometry,
    kind: StateTransferKind,
    retentions: StateTransferCopyRetentions,
}

impl<R: DeviceRuntime> PreparedStateTransferCopies<R> {
    /// Validates every endpoint and range before encoding the first command.
    /// No caller batch is changed on failure and this function never submits.
    /// A checkpoint Arc grants retention, not permission to overwrite a
    /// previously published capture; the parent must own that exclusive permit.
    pub(super) fn encode(
        guard: &PreparedSequenceStateTransfer<R>,
        checkpoint: &Arc<CheckpointBackingOwner<R>>,
        byte_plan: &SequenceCheckpointBytePlan,
        lane: &ExecutionLane<R>,
    ) -> Result<Self, StateTransferCopyEncodeError<R::Error>> {
        checkpoint.validate_transfer_binding(guard, byte_plan, lane)?;
        if !lane.current_descriptor_matches_snapshot() {
            return Err(invalid_completion(
                "state-copy runtime differs from its execution lane snapshot",
            )
            .into());
        }
        let runtime = lane.runtime();
        let kind = guard.kind().into();
        let mut owners = vec![DeviceBufferRetention::pair(
            Arc::clone(guard.backing()),
            Arc::clone(checkpoint),
        )];
        let mut views = Vec::with_capacity(byte_plan.resources().len());
        for resource in byte_plan.resources() {
            let sequence = guard.backing_view(resource.resource_id())?;
            let compact = checkpoint.view(resource.resource_id())?;
            if sequence.usage() != compact.usage()
                || sequence.element_type() != compact.element_type()
                || compact.size_bytes() != resource.logical_bytes()
            {
                return Err(invalid_completion(
                    "sequence and checkpoint copy views have incompatible logical storage",
                )
                .into());
            }
            let sequence =
                checked_copy_view(runtime, resource.resource_id(), sequence, &mut owners)?;
            let compact = checked_copy_view(runtime, resource.resource_id(), compact, &mut owners)?;
            views.push((resource, sequence, compact));
        }

        // Retain the views while all translated copies borrow their buffers.
        // The byte plan's sequence and compact offsets are distinct coordinate
        // systems; pairing their physical regions preserves both origins.
        let mut planned = Vec::new();
        for (resource, sequence, compact) in &views {
            for range in resource.ranges() {
                let source = range.source();
                let length = source
                    .end
                    .checked_sub(source.start)
                    .filter(|length| *length > 0)
                    .ok_or_else(|| invalid_completion("state-copy source range is empty"))?;
                let sequence_regions = sequence.translate(source.start, length)?;
                let compact_regions = compact.translate(range.checkpoint_offset(), length)?;
                for copy in sequence_regions.copies_to(&compact_regions)? {
                    let (sequence, compact, region) = copy.buffers_and_region();
                    planned.push(PlannedStateTransferCopy::from_sequence_to_checkpoint(
                        kind,
                        resource.resource_id(),
                        sequence,
                        compact,
                        region,
                    )?);
                }
            }
        }
        if planned.is_empty() || u32::try_from(planned.len()).is_err() {
            return Err(invalid_completion(
                "state copy is empty or its physical command count exceeds u32",
            )
            .into());
        }
        let commands = encode_planned_copies(&planned, |source, destination, region| {
            runtime.encode_copy(source, destination, region)
        })?;
        let geometry = CheckpointCopyGeometry {
            bytes: planned.iter().fold(0_u64, |bytes, copy| {
                bytes.saturating_add(copy.region.length_bytes())
            }),
            commands: commands.len() as u64,
        };
        Ok(Self {
            commands,
            geometry,
            kind,
            retentions: StateTransferCopyRetentions { _owners: owners },
        })
    }

    pub(super) fn len(&self) -> usize {
        self.commands.len()
    }

    pub(super) fn geometry(&self) -> CheckpointCopyGeometry {
        self.geometry
    }

    /// Restore initialization must already precede these copies in `batch`.
    /// This adds no model node, participant, compute command or completion.
    pub(super) fn append_to(
        self,
        batch: &mut DeviceCommandBatch<R::Command>,
    ) -> StateTransferCopyRetentions {
        append_copy_commands(self.kind, self.commands, batch);
        self.retentions
    }
}

fn checked_copy_view<'a, R: DeviceRuntime>(
    runtime: &R,
    resource_id: &ResourceId,
    backing: LogicalBackingBufferView<'a, R::Buffer>,
    owners: &mut Vec<DeviceBufferRetention>,
) -> Result<OperationBufferView<'a, R::Buffer>, VNextError> {
    for binding in backing.segment_bindings() {
        let actual = runtime.buffer_descriptor(binding.buffer());
        let segment = binding.segment();
        if &actual != binding.descriptor()
            || segment
                .offset_bytes()
                .checked_add(segment.length_bytes())
                .is_none_or(|end| end > actual.size_bytes)
        {
            return Err(invalid_completion(format!(
                "state-copy resource `{resource_id}` has a drifted physical buffer descriptor",
            )));
        }
        owners.push(binding.retention());
    }
    let descriptor = BufferDescriptor {
        resource_id: resource_id.clone(),
        size_bytes: backing.size_bytes(),
        alignment_bytes: backing.alignment_bytes(),
        usage: backing.usage(),
        element_type: backing.element_type(),
    };
    // Sequence is the underlying storage lifetime for both endpoints. This
    // view only translates bytes; the distinct checkpoint owner above carries
    // the checkpoint capacity claim and never becomes a model participant.
    Ok(OperationBufferView::from_backing_prefix(
        descriptor,
        backing,
        AllocationLifetime::Sequence,
    ))
}

struct PlannedStateTransferCopy<'a, B> {
    resource_id: &'a ResourceId,
    source: &'a B,
    destination: &'a B,
    region: CopyRegion,
}

impl<'a, B> PlannedStateTransferCopy<'a, B> {
    fn from_sequence_to_checkpoint(
        kind: StateTransferKind,
        resource_id: &'a ResourceId,
        sequence: &'a B,
        checkpoint: &'a B,
        region: CopyRegion,
    ) -> Result<Self, VNextError> {
        let (source, destination, region) = match kind {
            StateTransferKind::Capture => (sequence, checkpoint, region),
            StateTransferKind::Restore => (
                checkpoint,
                sequence,
                CopyRegion::new(
                    region.destination_offset_bytes(),
                    region.source_offset_bytes(),
                    region.length_bytes(),
                )?,
            ),
        };
        Ok(Self {
            resource_id,
            source,
            destination,
            region,
        })
    }
}

fn encode_planned_copies<B, C, E>(
    planned: &[PlannedStateTransferCopy<'_, B>],
    mut encode: impl FnMut(&B, &B, CopyRegion) -> Result<C, E>,
) -> Result<Vec<C>, StateTransferCopyEncodeError<E>> {
    planned
        .iter()
        .map(|copy| {
            encode(copy.source, copy.destination, copy.region).map_err(|error| {
                StateTransferCopyEncodeError::Runtime {
                    resource_id: copy.resource_id.clone(),
                    region: copy.region,
                    error,
                }
            })
        })
        .collect()
}

fn append_copy_commands<C>(
    kind: StateTransferKind,
    commands: Vec<C>,
    batch: &mut DeviceCommandBatch<C>,
) {
    for command in commands {
        match kind {
            StateTransferKind::Capture => batch.push_result_binding(command),
            StateTransferKind::Restore => batch.push_dynamic_binding(command),
        }
    }
}

#[cfg(test)]
mod tests;
