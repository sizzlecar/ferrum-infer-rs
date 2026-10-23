//! Accounted synchronous readback. The existing completion slot remains the
//! sole owner of device leases until the terminal action has finished.

use super::*;
use crate::{HostMemoryClass, ManagedHostBuffer, ManagedHostBytes};

#[derive(Debug)]
#[must_use = "nonterminal observations retain the existing completion ownership"]
pub struct ManagedCompletionReadbackObservation {
    completion: CompletionObservation,
    outputs: Option<Result<Vec<ManagedHostBytes>, VNextError>>,
}

impl ManagedCompletionReadbackObservation {
    pub fn into_parts(
        self,
    ) -> (
        CompletionObservation,
        Option<Result<Vec<ManagedHostBytes>, VNextError>>,
    ) {
        (self.completion, self.outputs)
    }
}

impl<R: DeviceRuntime> CompletionHandle<R> {
    /// Buffers must be reserved from the executor's Output budget before
    /// submission. The synchronous backend contract keeps borrowed storage
    /// valid on every return; unsupported backends reject without DMA.
    /// A nonterminal observation consumes and releases these output buffers.
    /// The handle retains completion/recovery authority, not these destinations
    /// or a promise to resume output production after recovery.
    pub fn wait_with_managed_readbacks(
        &self,
        requests: Vec<(CompletionReadbackRequest, ManagedHostBuffer)>,
    ) -> Result<ManagedCompletionReadbackObservation, VNextError> {
        self.reaper
            .upgrade()
            .ok_or_else(|| invalid_completion("completion reaper owner was dropped"))?
            .wait_bound_with_managed_readbacks(self.slot_id(), requests)
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    fn wait_bound_with_managed_readbacks(
        &self,
        slot_id: CompletionSlotId,
        requests: Vec<(CompletionReadbackRequest, ManagedHostBuffer)>,
    ) -> Result<ManagedCompletionReadbackObservation, VNextError> {
        if requests.is_empty() {
            return Err(invalid_completion(
                "managed readback requires at least one output",
            ));
        }
        for (request, output) in &requests {
            if output.class() != HostMemoryClass::Output {
                return Err(invalid_completion(
                    "managed readback requires Output host allocation",
                ));
            }
            request.output_layout().validate_bytes(output.len())?;
        }
        {
            let record = self.lookup(slot_id)?;
            let guard = record
                .lock()
                .map_err(|_| invalid_completion("completion slot mutex is poisoned"))?;
            if matches!(&*guard, CompletionRecord::InFlight { resources, .. }
                | CompletionRecord::SubmissionIndeterminate { resources, .. }
                if resources.prepared_terminal_request().is_some())
            {
                return Err(invalid_completion(
                    "managed readback cannot replace a prepared terminal packet",
                ));
            }
        }
        let observation =
            self.observe_bound_with(slot_id, true, |resources, lane, _, disposition, _| {
                if !matches!(disposition, OperationCompletionDisposition::Succeeded) {
                    return Err(invalid_completion(
                        "managed output unavailable after unsuccessful compute",
                    ));
                }
                let mut outputs = Vec::with_capacity(requests.len());
                for (request, mut output) in requests {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let length = request.output_layout().byte_len()?;
                        let end = request
                            .logical_offset_bytes()
                            .checked_add(length)
                            .ok_or_else(|| {
                                invalid_completion("managed readback semantic range overflows")
                            })?;
                        let range = resources.readback_range(
                            request.node_id(),
                            request.participant_index(),
                            request.resource_id(),
                            request.logical_offset_bytes()..end,
                        )?;
                        let backing = resources.backing_view(
                            request.node_id(),
                            request.participant_index(),
                            request.resource_id(),
                        )?;
                        lane.readback_managed_buffer(
                            &backing,
                            request.expected_usage(),
                            range.start,
                            request.output_layout(),
                            &mut output,
                        )
                    }));
                    match result {
                        Ok(Ok(())) => outputs.push(output.freeze()),
                        Ok(Err(error)) => return Err(error),
                        Err(_) => {
                            lane.fail_closed();
                            return Err(invalid_completion(
                                "backend panicked during synchronous managed readback",
                            ));
                        }
                    }
                }
                Ok(outputs)
            })?;
        let (completion, outputs) = match observation {
            BoundCompletionObservation::Pending => (CompletionObservation::Pending, None),
            BoundCompletionObservation::Terminal {
                completion,
                terminal,
            } => (CompletionObservation::Terminal(completion), Some(terminal)),
            BoundCompletionObservation::Indeterminate(failures) => {
                (CompletionObservation::Indeterminate(failures), None)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                (CompletionObservation::SubmissionIndeterminate, None)
            }
            BoundCompletionObservation::ObservationPanicked => {
                (CompletionObservation::ObservationPanicked, None)
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                (CompletionObservation::Quarantined(receipt), None)
            }
        };
        Ok(ManagedCompletionReadbackObservation {
            completion,
            outputs,
        })
    }
}

impl<R: DeviceRuntime> ExecutionLane<R> {
    fn readback_managed_buffer(
        &self,
        backing: &LogicalBackingBufferView<'_, R::Buffer>,
        expected_usage: BufferUsage,
        logical_offset: u64,
        layout: HostTransferLayout,
        output: &mut ManagedHostBuffer,
    ) -> Result<(), VNextError> {
        let length = layout.byte_len()?;
        let end = logical_offset
            .checked_add(length)
            .ok_or_else(|| invalid_completion("managed readback range overflows"))?;
        let element_bytes = layout.element_type().size_bytes();
        if !self.runtime.supports_managed_tensor_io()
            || backing.usage() != expected_usage
            || backing.element_type() != layout.element_type()
            || end > backing.size_bytes()
            || logical_offset % element_bytes != 0
            || output.len() as u64 != length
            || self.fail_closed.load(Ordering::Acquire)
            || !self.current_descriptor_matches_snapshot()
        {
            return Err(invalid_completion(
                "managed readback backing, destination or runtime contract differs",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_completion("managed readback lane mutex poisoned"))?;
        if state.fail_closed
            || !matches!(
                self.runtime.stream_state(&state.stream),
                StreamState::Ready | StreamState::Submitted
            )
        {
            return Err(invalid_completion(
                "managed readback lane is failed or unreadable",
            ));
        }
        let mut cursor = 0_u64;
        let mut covered = 0_u64;
        for binding in backing.segment_bindings() {
            let segment = binding.segment();
            let segment_end = cursor
                .checked_add(segment.length_bytes())
                .ok_or_else(|| invalid_completion("managed readback backing extent overflows"))?;
            let start = cursor.max(logical_offset);
            let stop = segment_end.min(end);
            if start < stop {
                let source_offset = segment
                    .offset_bytes()
                    .checked_add(start - cursor)
                    .ok_or_else(|| {
                        invalid_completion("managed readback physical offset overflows")
                    })?;
                let piece_bytes = stop - start;
                let actual = self.runtime.buffer_descriptor(binding.buffer());
                if &actual != binding.descriptor()
                    || source_offset % element_bytes != 0
                    || piece_bytes % element_bytes != 0
                    || source_offset
                        .checked_add(piece_bytes)
                        .is_none_or(|end| end > actual.size_bytes)
                {
                    return Err(invalid_completion(
                        "managed readback physical binding drifted or splits an element",
                    ));
                }
                let region = CopyRegion::new(source_offset, start - logical_offset, piece_bytes)?;
                let piece_layout =
                    HostTransferLayout::new(layout.element_type(), piece_bytes / element_bytes)?;
                match self.runtime.readback_managed_into(
                    &mut state.stream,
                    binding.buffer(),
                    region,
                    piece_layout,
                    output,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        return Err(invalid_completion(
                            "backend declined synchronous managed readback",
                        ))
                    }
                    Err(error) => {
                        state.fail_closed = true;
                        self.fail_closed.store(true, Ordering::Release);
                        return Err(invalid_completion(format!(
                            "managed readback failed after quiescence: {error}"
                        )));
                    }
                }
                covered = covered
                    .checked_add(piece_bytes)
                    .ok_or_else(|| invalid_completion("managed readback coverage overflows"))?;
            }
            cursor = segment_end;
        }
        if covered != length || !self.current_descriptor_matches_snapshot() {
            state.fail_closed = true;
            self.fail_closed.store(true, Ordering::Release);
            return Err(invalid_completion(
                "managed readback lacks exact output coverage",
            ));
        }
        Ok(())
    }
}
