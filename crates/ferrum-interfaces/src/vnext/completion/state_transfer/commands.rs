//! Native state-copy encoding without submission or completion authority.

use super::super::CheckpointCopyGeometry;
use super::{invalid_completion, StateTransferKind};
use crate::vnext::{
    split_strided_copy, AllocationLifetime, BufferDescriptor, CheckpointBackingOwner, CopyRegion,
    DeviceBufferRetention, DeviceCommandBatch, DeviceRuntime, ExecutionLane,
    LogicalBackingBufferView, NativeCheckpointTransferGeometryBuilder, OperationBufferView,
    PreparedSequenceStateTransfer, ResourceId, SequenceCheckpointBytePlan, StridedCopyRegion,
    StridedCopySplitError, VNextError,
};
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum StateTransferCopyEncodeError<E> {
    Contract(VNextError),
    StridedRuntime {
        resource_id: ResourceId,
        region: StridedCopyRegion,
        error: E,
    },
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
    cost_geometry: Option<NativeCheckpointTransferGeometryBuilder>,
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
        Self::encode_with_geometry(guard, checkpoint, byte_plan, lane, false)
    }

    pub(super) fn encode_with_geometry(
        guard: &PreparedSequenceStateTransfer<R>,
        checkpoint: &Arc<CheckpointBackingOwner<R>>,
        byte_plan: &SequenceCheckpointBytePlan,
        lane: &ExecutionLane<R>,
        observe_geometry: bool,
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
            let sequence = checked_copy_view(
                runtime,
                resource.resource_id(),
                sequence,
                resource.physical_copy_extent(),
                &mut owners,
            )?;
            let compact =
                checked_copy_view(runtime, resource.resource_id(), compact, None, &mut owners)?;
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
        let mut rectangles = Vec::new();
        for (resource, sequence, compact) in &views {
            for region in resource.strided_ranges() {
                rectangles.extend(plan_strided_copies(
                    kind,
                    resource.resource_id(),
                    sequence,
                    compact,
                    *region,
                )?);
            }
        }
        let command_count = planned
            .len()
            .checked_add(rectangles.len())
            .ok_or_else(|| invalid_completion("checkpoint physical command count overflows"))?;
        if command_count == 0 || command_count > crate::execution_cost::MAX_COST_COMMANDS {
            return Err(invalid_completion(
                "state copy is empty or its physical command count exceeds the cost command limit",
            )
            .into());
        }
        let mut cost_geometry = observe_geometry.then(NativeCheckpointTransferGeometryBuilder::new);
        let mut copy_bytes = 0_u64;
        for copy in &planned {
            copy_bytes = copy_bytes
                .checked_add(copy.region.length_bytes())
                .ok_or_else(|| invalid_completion("checkpoint copy bytes overflow"))?;
            if let Some(geometry) = &mut cost_geometry {
                geometry.push_copy(copy.resource_id, copy.region.length_bytes())?;
            }
        }
        for copy in &rectangles {
            copy_bytes = copy_bytes
                .checked_add(copy.region.length_bytes()?)
                .ok_or_else(|| invalid_completion("checkpoint strided bytes overflow"))?;
            if let Some(geometry) = &mut cost_geometry {
                geometry.push_strided_copy(copy.resource_id, copy.region)?;
            }
        }
        let mut commands = encode_planned_copies(&planned, |source, destination, region| {
            runtime.encode_copy(source, destination, region)
        })?;
        for copy in &rectangles {
            let encoded = runtime
                .encode_strided_copy(copy.source, copy.destination, copy.region)
                .ok_or_else(|| {
                    invalid_completion("runtime does not support native strided checkpoint copies")
                })?
                .map_err(|error| StateTransferCopyEncodeError::StridedRuntime {
                    resource_id: copy.resource_id.clone(),
                    region: copy.region,
                    error,
                })?;
            commands.push(encoded);
        }
        let geometry = CheckpointCopyGeometry {
            bytes: copy_bytes,
            commands: commands.len() as u64,
        };
        Ok(Self {
            commands,
            geometry,
            cost_geometry,
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

    pub(super) fn cost_geometry(&self) -> Option<NativeCheckpointTransferGeometryBuilder> {
        self.cost_geometry.clone()
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
    physical_extent: Option<u64>,
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
    // Only the trusted byte plan can supply provider physical coordinates.
    // Segment identities and descriptors above still prove the complete live
    // backing, and the bound is its actual committed capacity, not free padding.
    let size_bytes = physical_extent.unwrap_or(backing.size_bytes());
    if size_bytes == 0 || size_bytes > backing.capacity_size_bytes() {
        return Err(invalid_completion(
            "state-copy extent exceeds its committed backing",
        ));
    }
    let descriptor = BufferDescriptor {
        resource_id: resource_id.clone(),
        size_bytes,
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

struct PlannedStridedStateTransferCopy<'a, B> {
    resource_id: &'a ResourceId,
    source: &'a B,
    destination: &'a B,
    region: StridedCopyRegion,
}

fn plan_strided_copies<'a, B>(
    kind: StateTransferKind,
    resource_id: &'a ResourceId,
    sequence: &'a OperationBufferView<'_, B>,
    compact: &'a OperationBufferView<'_, B>,
    region: StridedCopyRegion,
) -> Result<Vec<PlannedStridedStateTransferCopy<'a, B>>, VNextError> {
    let source = sequence.translate(region.source_offset_bytes(), region.source_extent_bytes()?)?;
    let destination = compact.translate(
        region.destination_offset_bytes(),
        region.destination_extent_bytes()?,
    )?;
    let a = source
        .iter()
        .map(|r| r.buffer_and_physical_range())
        .collect::<Vec<_>>();
    let b = destination
        .iter()
        .map(|r| r.buffer_and_physical_range())
        .collect::<Vec<_>>();
    for (source, range_a, _) in &a {
        for (destination, range_b, _) in &b {
            if std::ptr::eq(*source, *destination)
                && range_a.start < range_b.end
                && range_b.start < range_a.end
            {
                return Err(invalid_completion(
                    "strided checkpoint source and destination alias",
                ));
            }
        }
    }
    let shape = StridedCopyRegion::new(
        0,
        0,
        region.width_bytes(),
        region.height(),
        region.source_pitch_bytes(),
        region.destination_pitch_bytes(),
    )?;
    let fragments = split_strided_copy(
        shape,
        &a.iter()
            .map(|(_, r, _)| r.end - r.start)
            .collect::<Vec<_>>(),
        &b.iter()
            .map(|(_, r, _)| r.end - r.start)
            .collect::<Vec<_>>(),
        crate::execution_cost::MAX_COST_COMMANDS,
        || Ok::<(), std::convert::Infallible>(()),
    )
    .map_err(|error| match error {
        StridedCopySplitError::LimitExceeded => {
            invalid_completion("strided copy physical command limit exceeded")
        }
        StridedCopySplitError::Contract(error) => error,
        StridedCopySplitError::Poll(never) => match never {},
    })?;
    fragments
        .into_iter()
        .map(|f| {
            let (sequence, source, _) = &a[f.source_segment];
            let (checkpoint, destination, _) = &b[f.destination_segment];
            let region = StridedCopyRegion::new(
                source
                    .start
                    .checked_add(f.region.source_offset_bytes())
                    .ok_or_else(|| invalid_completion("strided source offset overflows"))?,
                destination
                    .start
                    .checked_add(f.region.destination_offset_bytes())
                    .ok_or_else(|| invalid_completion("strided destination offset overflows"))?,
                f.region.width_bytes(),
                f.region.height(),
                f.region.source_pitch_bytes(),
                f.region.destination_pitch_bytes(),
            )?;
            let (source, destination, region) = match kind {
                StateTransferKind::Capture => (*sequence, *checkpoint, region),
                StateTransferKind::Restore => (*checkpoint, *sequence, region.reversed()),
            };
            Ok(PlannedStridedStateTransferCopy {
                resource_id,
                source,
                destination,
                region,
            })
        })
        .collect()
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
