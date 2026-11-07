//! Selected copies share the ordinary completion registry and recovery owner.
use super::*;
use crate::{
    BufferDescriptor, DeviceBufferRetention, OperationBufferView, PreparedSelectedStateCommit,
    SelectedPrefix,
};

pub enum SelectedStateSubmission<R: DeviceRuntime> {
    Submitted(CompletionHandle<R>),
    ContractAfterSubmission {
        error: VNextError,
        handle: CompletionHandle<R>,
    },
    Indeterminate(IndeterminateSubmissionHandle<R>),
}
impl<R: DeviceRuntime> PreparedSelectedStateCommit<R> {
    fn encode(&mut self) -> Result<DeviceCommandBatch<R::Command>, VNextError> {
        let lane = self.step.execution_lane();
        if !lane.current_descriptor_matches_snapshot() {
            return Err(invalid_completion(
                "selected-copy runtime descriptor drifted",
            ));
        }
        let runtime = lane.runtime();
        let layout = self
            .step
            .candidate_layout()
            .ok_or_else(|| invalid_completion("selected-copy layout absent"))?;
        let mut owners = Vec::new();
        let mut views = Vec::new();
        for (index, (participant, selection)) in self
            .step
            .work_shape()
            .participants()
            .iter()
            .zip(&self.selections)
            .enumerate()
        {
            for copy in &layout.copies {
                let source = self.step.backing_view(&copy.candidate_resource)?;
                let destination = self
                    .step
                    .participant_backing_view(*participant, &copy.destination)?;
                let source = copy_view(
                    runtime,
                    &copy.candidate_resource,
                    source,
                    AllocationLifetime::Step,
                    &mut owners,
                )?;
                let destination = copy_view(
                    runtime,
                    &copy.destination,
                    destination,
                    AllocationLifetime::Sequence,
                    &mut owners,
                )?;
                let slot = if *selection == SelectedPrefix::One {
                    0
                } else {
                    1
                };
                let start = copy
                    .candidate_offset
                    .checked_add(
                        copy.bytes
                            .checked_mul(slot)
                            .ok_or_else(|| invalid_completion("candidate offset overflow"))?,
                    )
                    .ok_or_else(|| invalid_completion("candidate offset overflow"))?;
                let end = start
                    .checked_add(copy.bytes)
                    .ok_or_else(|| invalid_completion("candidate range overflow"))?;
                let descriptor = self.step.dynamic_descriptor(&copy.candidate_resource)?;
                let range = translate_step_participant_readback_range(
                    descriptor.demand(),
                    self.step.work_shape(),
                    index,
                    start..end,
                )?;
                // Translate all sources/destinations before encoding any command.
                source.translate(range.start, copy.bytes)?;
                destination.translate(copy.destination_offset, copy.bytes)?;
                views.push((
                    source,
                    destination,
                    range.start,
                    copy.destination_offset,
                    copy.bytes,
                ));
            }
        }
        let mut commands =
            DeviceCommandBatch::with_capacity_and_timing(views.len(), DeviceTimingMode::Off);
        for (src, dst, src_start, dst_start, bytes) in &views {
            for copy in src
                .translate(*src_start, *bytes)?
                .copies_to(&dst.translate(*dst_start, *bytes)?)?
            {
                let (a, b, region) = copy.buffers_and_region();
                let command = runtime.encode_copy(a, b, region).map_err(|e| {
                    invalid_completion(format!("selected-copy encoding failed: {e}"))
                })?;
                commands.push_dynamic_binding(command);
            }
        }
        self.retentions = owners;
        Ok(commands)
    }
}
fn copy_view<'a, R: DeviceRuntime>(
    runtime: &R,
    id: &ResourceId,
    view: LogicalBackingBufferView<'a, R::Buffer>,
    lifetime: AllocationLifetime,
    owners: &mut Vec<DeviceBufferRetention>,
) -> Result<OperationBufferView<'a, R::Buffer>, VNextError> {
    for b in view.segment_bindings() {
        let actual = runtime.buffer_descriptor(b.buffer());
        if &actual != b.descriptor()
            || b.segment()
                .offset_bytes()
                .checked_add(b.segment().length_bytes())
                .is_none_or(|end| end > actual.size_bytes)
        {
            return Err(invalid_completion(
                "selected-copy physical descriptor changed",
            ));
        }
        owners.push(b.retention());
    }
    let descriptor = BufferDescriptor {
        resource_id: id.clone(),
        size_bytes: view.size_bytes(),
        alignment_bytes: view.alignment_bytes(),
        usage: view.usage(),
        element_type: view.element_type(),
    };
    Ok(OperationBufferView::from_backing_prefix(
        descriptor, view, lifetime,
    ))
}
impl<R: DeviceRuntime> CompletionReaper<R> {
    pub fn submit_selected_state_commit(
        self: &Arc<Self>,
        mut prepared: PreparedSelectedStateCommit<R>,
    ) -> Result<SelectedStateSubmission<R>, VNextError> {
        let commands = prepared.encode()?;
        let lane = Arc::clone(prepared.step.execution_lane());
        let identity = prepared.identity.clone();
        let mut reservation = Self::reserve_resources(
            self,
            CompletionResourceLease::Selected(prepared),
            Arc::clone(&lane),
            identity,
        )?;
        let mut enqueue = lane.reserve_enqueue()?;
        reservation.mark_submission_started();
        let result = enqueue.submit(commands);
        drop(enqueue);
        Ok(match result {
            LaneSubmitOutcome::Submitted(fence) => {
                match reservation.arm(fence, DeviceTimingMode::Off) {
                    Ok(handle) => SelectedStateSubmission::Submitted(handle),
                    Err((error, handle)) => {
                        SelectedStateSubmission::ContractAfterSubmission { error, handle }
                    }
                }
            }
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                reservation.submission_may_have_happened = false;
                return Err(invalid_completion(format!(
                    "selected copy definitely not submitted: {error}"
                )));
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                SelectedStateSubmission::Indeterminate(reservation.submission_indeterminate())
            }
        })
    }
}
