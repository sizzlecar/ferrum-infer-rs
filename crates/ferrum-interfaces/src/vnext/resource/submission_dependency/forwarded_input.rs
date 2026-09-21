use super::{
    invalid_resource, Arc, DeviceRuntime, PreparedStepSubmissionWave, SubmittedWavePredecessor,
    VNextError,
};
use crate::vnext::{
    translate_step_participant_readback_range, AllocationKind, AllocationLifetime, BufferUsage,
    CompletionReadbackRequest, ElementType, LogicalBackingBufferView, NodeId, ResolvedValueRole,
    TensorAccess,
};
use std::ops::Range;

/// A device token from an exact submitted predecessor, bound to one child
/// input. This owns the source claims, not the parent's Step lease; retiring
/// that Step therefore cannot recycle the source before the child completes.
pub struct SubmissionWaveInputForward<R: DeviceRuntime> {
    predecessor: Arc<SubmittedWavePredecessor<R>>,
    source: CompletionReadbackRequest,
    destination_node: NodeId,
    destination_input_ordinal: u32,
    destination_logical_offset_bytes: u64,
}

impl<R: DeviceRuntime> SubmittedWavePredecessor<R> {
    pub fn forward_token(
        self: &Arc<Self>,
        source: CompletionReadbackRequest,
        destination_node: NodeId,
        destination_input_ordinal: u32,
        destination_logical_offset_bytes: u64,
    ) -> Result<SubmissionWaveInputForward<R>, VNextError> {
        self.validate_token_source(&source)?;
        if destination_logical_offset_bytes % 4 != 0
            || destination_logical_offset_bytes.checked_add(4).is_none()
        {
            return Err(invalid_resource(
                "forwarded token input range is unaligned or overflows",
            ));
        }
        Ok(SubmissionWaveInputForward {
            predecessor: Arc::clone(self),
            source,
            destination_node,
            destination_input_ordinal,
            destination_logical_offset_bytes,
        })
    }

    pub(super) fn validate_token_source(
        &self,
        source: &CompletionReadbackRequest,
    ) -> Result<(), VNextError> {
        let (view, range) = self.token_source_view(source)?;
        // Validate the actual descriptors and every covered segment now, and
        // again at dispatch. The pool view itself checks live claim generation.
        let runtime = self.lane.runtime();
        let mut logical_start = 0_u64;
        let mut covered = 0_u64;
        for binding in view.segment_bindings() {
            let segment = binding.segment();
            let logical_end = logical_start
                .checked_add(segment.length_bytes())
                .ok_or_else(|| invalid_resource("forwarded source logical extent overflows"))?;
            let start = range.start.max(logical_start);
            let end = range.end.min(logical_end);
            if start < end {
                let actual = runtime.buffer_descriptor(binding.buffer());
                let physical = segment
                    .offset_bytes()
                    .checked_add(start - logical_start)
                    .ok_or_else(|| {
                        invalid_resource("forwarded source physical offset overflows")
                    })?;
                if &actual != binding.descriptor()
                    || actual.element_type != ElementType::U32
                    || actual.usage != BufferUsage::Activations
                    || physical % 4 != 0
                    || (end - start) % 4 != 0
                    || physical
                        .checked_add(end - start)
                        .is_none_or(|end| end > actual.size_bytes)
                {
                    return Err(invalid_resource(
                        "forwarded source physical descriptor or extent changed",
                    ));
                }
                covered += end - start;
            }
            logical_start = logical_end;
        }
        if covered != 4 {
            return Err(invalid_resource(
                "forwarded source does not cover one physical U32 token",
            ));
        }
        Ok(())
    }

    fn token_source_view(
        &self,
        source: &CompletionReadbackRequest,
    ) -> Result<(LogicalBackingBufferView<'_, R::Buffer>, Range<u64>), VNextError> {
        if source.output_layout().element_type() != ElementType::U32
            || source.output_layout().element_count() != 1
            || source.expected_usage() != BufferUsage::Activations
            || self.lane.reusable_execution_epoch() != self.lane_epoch
            || !self.lane.is_reusable()
            || !self.lane.current_descriptor_matches_snapshot()
        {
            return Err(invalid_resource(
                "forwarded source requires a live predecessor lane and one activation U32",
            ));
        }
        let participant = self
            .participants
            .get(source.participant_index() as usize)
            .ok_or_else(|| invalid_resource("forwarded source participant is absent"))?;
        let plan = &participant.resources.request.plan;
        let node = plan
            .nodes()
            .iter()
            .find(|node| node.id() == source.node_id())
            .filter(|_| {
                self.receipt
                    .batch_identity()
                    .node_index(source.node_id())
                    .is_some()
            })
            .ok_or_else(|| invalid_resource("forwarded source is not a submitted plan node"))?;
        let mut outputs = node.values().iter().filter(|value| {
            value.role() == ResolvedValueRole::Output
                && value
                    .storage()
                    .components()
                    .iter()
                    .any(|component| component.resource_id() == source.resource_id())
        });
        let output = outputs
            .next()
            .ok_or_else(|| invalid_resource("forwarded source is not a declared node output"))?;
        let [component] = output.storage().components() else {
            return Err(invalid_resource(
                "forwarded token output requires one storage component",
            ));
        };
        let end = source
            .logical_offset_bytes()
            .checked_add(4)
            .ok_or_else(|| invalid_resource("forwarded source logical range overflows"))?;
        let component_end = component
            .offset_bytes()
            .checked_add(component.length_bytes())
            .ok_or_else(|| invalid_resource("forwarded source component range overflows"))?;
        if outputs.next().is_some()
            || output.usage() != BufferUsage::Activations
            || !matches!(
                output.access(),
                TensorAccess::Write | TensorAccess::ReadWrite
            )
            || output.tensor().element_type() != ElementType::U32
            || component.element_type() != ElementType::U32
            || source.logical_offset_bytes() < component.offset_bytes()
            || end > component_end
        {
            return Err(invalid_resource(
                "forwarded token differs from its declared output binding",
            ));
        }
        let pools = plan.dynamic_pools();
        let mut descriptors = pools.domains.iter().filter_map(|domain| {
            domain
                .descriptors
                .iter()
                .find(|descriptor| descriptor.base_resource_id() == source.resource_id())
        });
        let descriptor = descriptors
            .next()
            .ok_or_else(|| invalid_resource("forwarded source has no dynamic descriptor"))?;
        if descriptors.next().is_some()
            || !matches!(
                descriptor.lifetime(),
                AllocationLifetime::Step | AllocationLifetime::Invocation
            )
            || descriptor.kind() != &AllocationKind::Value
            || (descriptor.lifetime() == AllocationLifetime::Invocation
                && self.participants.len() != 1)
        {
            return Err(invalid_resource(
                "forwarded token requires an immutable submitted Step or Invocation value",
            ));
        }
        let authority = self
            .wave_backing
            .backing_slices()
            .iter()
            .chain(self.step_backing.backing_slices())
            .find(|authority| authority.resource_id() == source.resource_id())
            .ok_or_else(|| {
                invalid_resource("predecessor does not retain the forwarded source claim")
            })?;
        let view = pools.view(authority)?;
        let range = if descriptor.lifetime() == AllocationLifetime::Step {
            translate_step_participant_readback_range(
                descriptor.demand(),
                self.work_shape(),
                source.participant_index() as usize,
                source.logical_offset_bytes()..end,
            )?
        } else {
            source.logical_offset_bytes()..end
        };
        if view.usage() != BufferUsage::Activations
            || view.element_type() != ElementType::U32
            || range.end.checked_sub(range.start) != Some(4)
            || range.end > view.size_bytes()
        {
            return Err(invalid_resource(
                "forwarded source backing differs from its declared token range",
            ));
        }
        Ok((view, range))
    }
}

impl<R: DeviceRuntime> SubmissionWaveInputForward<R> {
    pub fn source(&self) -> &CompletionReadbackRequest {
        &self.source
    }
    pub fn node_id(&self) -> &NodeId {
        &self.destination_node
    }
    pub fn input_ordinal(&self) -> u32 {
        self.destination_input_ordinal
    }
    pub fn logical_offset_bytes(&self) -> u64 {
        self.destination_logical_offset_bytes
    }

    pub(crate) fn validate_for_wave(
        &self,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<(), VNextError> {
        let step = wave.step_resources();
        if !step
            .predecessor()
            .is_some_and(|predecessor| Arc::ptr_eq(predecessor, &self.predecessor))
            || wave.execution_lane_id() != self.predecessor.lane.id()
        {
            return Err(invalid_resource(
                "forwarded input is not the child's admitted predecessor",
            ));
        }
        // Admission already checked the whole immutable work shape against
        // this exact predecessor. Bind this declaration to its own row without
        // rescanning the cohort once for every forwarded token.
        let work = step
            .work_shape()
            .participant_work()
            .get(self.source.participant_index() as usize)
            .ok_or_else(|| invalid_resource("forwarded input participant is absent from child"))?;
        let parent = self
            .predecessor
            .work_shape()
            .participant_work()
            .get(self.source.participant_index() as usize)
            .ok_or_else(|| {
                invalid_resource("forwarded source participant is absent from parent")
            })?;
        if !work
            .token_span()
            .submitted_token_source()
            .is_some_and(|metadata| {
                metadata.source() == &self.source
                    && metadata.submission_fingerprint() == self.predecessor.receipt.fingerprint()
                    && metadata.parent_work_fingerprint() == parent.token_span().fingerprint()
            })
        {
            return Err(invalid_resource(
                "forwarded input differs from admitted device token work",
            ));
        }
        self.predecessor.validate_token_source(&self.source)
    }

    pub(crate) fn source_view(
        &self,
    ) -> Result<(LogicalBackingBufferView<'_, R::Buffer>, Range<u64>), VNextError> {
        self.predecessor.token_source_view(&self.source)
    }
}
