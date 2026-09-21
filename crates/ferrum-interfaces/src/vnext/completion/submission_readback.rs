use super::*;
use crate::vnext::{BackingChunkIdentity, BufferDescriptor};

#[derive(PartialEq, Eq)]
struct PhysicalReadbackRange {
    chunk: BackingChunkIdentity,
    descriptor: BufferDescriptor,
    offset: u64,
    length: u64,
}

struct PreparedReadbackPiece<E> {
    physical: PhysicalReadbackRange,
    snapshot: DeviceReadbackSnapshot<E>,
}

struct PreparedReadbackOutput<E> {
    request: CompletionReadbackRequest,
    pieces: Vec<PreparedReadbackPiece<E>>,
}

pub(super) struct PreparedCompletionReadbacks<E> {
    slot_id: CompletionSlotId,
    batch_identity_fingerprint: String,
    request: CompletionReadbackBatchRequest,
    outputs: Vec<PreparedReadbackOutput<E>>,
}

/// Reuses the same semantic translation and live backing checks at preparation
/// and terminal collection. Retaining an allocation alone never authorizes a
/// stale pool/chunk/slot or a changed typed physical range.
fn visit_readback_pieces<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    runtime: &R,
    request: &CompletionReadbackRequest,
    mut visit: impl FnMut(
        PhysicalReadbackRange,
        &R::Buffer,
        super::super::DeviceBufferRetention,
    ) -> Result<(), VNextError>,
) -> Result<(), VNextError> {
    let bytes = request.output_layout().byte_len()?;
    let semantic_end = request
        .logical_offset_bytes()
        .checked_add(bytes)
        .ok_or_else(|| invalid_completion("submission readback semantic range overflows"))?;
    let range = resources.readback_range(
        request.node_id(),
        request.participant_index(),
        request.resource_id(),
        request.logical_offset_bytes()..semantic_end,
    )?;
    let backing = resources.backing_view(
        request.node_id(),
        request.participant_index(),
        request.resource_id(),
    )?;
    let element_bytes = request.output_layout().element_type().size_bytes();
    if backing.usage() != request.expected_usage()
        || backing.element_type() != request.output_layout().element_type()
        || range.end > backing.size_bytes()
        || range.end.checked_sub(range.start) != Some(bytes)
        || range.start % element_bytes != 0
    {
        return Err(invalid_completion(
            "submission readback differs from its typed backing range",
        ));
    }
    let mut cursor = 0_u64;
    let mut covered = 0_u64;
    for binding in backing.segment_bindings() {
        let segment = binding.segment();
        let end = cursor
            .checked_add(segment.length_bytes())
            .ok_or_else(|| invalid_completion("submission readback segment coverage overflows"))?;
        let start = cursor.max(range.start);
        let stop = end.min(range.end);
        if start < stop {
            let offset = segment
                .offset_bytes()
                .checked_add(start - cursor)
                .ok_or_else(|| {
                    invalid_completion("submission readback physical offset overflows")
                })?;
            let length = stop - start;
            let descriptor = runtime.buffer_descriptor(binding.buffer());
            if &descriptor != binding.descriptor()
                || offset
                    .checked_add(length)
                    .is_none_or(|end| end > descriptor.size_bytes)
                || offset % element_bytes != 0
                || length % element_bytes != 0
            {
                return Err(invalid_completion(
                    "submission readback descriptor or alignment changed",
                ));
            }
            visit(
                PhysicalReadbackRange {
                    chunk: binding.chunk().clone(),
                    descriptor,
                    offset,
                    length,
                },
                binding.buffer(),
                binding.retention(),
            )?;
            covered = covered
                .checked_add(length)
                .ok_or_else(|| invalid_completion("submission readback coverage overflows"))?;
        }
        cursor = end;
    }
    if covered != bytes {
        return Err(invalid_completion(
            "submission readback has incomplete physical coverage",
        ));
    }
    Ok(())
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    pub(crate) fn prepare_submission_readbacks(
        &mut self,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<(), VNextError> {
        let Some(request) = self.wave().submission_readbacks().cloned() else {
            return Ok(());
        };
        let identity = self.batch_identity.as_ref().expect("reserved identity");
        request.validate_for(identity)?;
        let resources = self.resources.as_ref().expect("reserved resources");
        let lane = self.lane.as_ref().expect("reserved lane");
        let runtime = lane.runtime();
        let mut staged_commands = Vec::new();
        let mut outputs = Vec::with_capacity(request.len());
        let mut unsupported = false;
        for item in request.requests() {
            let mut pieces = Vec::new();
            visit_readback_pieces(resources, runtime, item, |physical, source, retention| {
                if unsupported {
                    return Ok(());
                }
                let Some(staging) = lane.readback_staging.reserve(physical.length) else {
                    unsupported = true;
                    return Ok(());
                };
                let layout = HostTransferLayout::new(
                    item.output_layout().element_type(),
                    physical.length / item.output_layout().element_type().size_bytes(),
                )?;
                let region = CopyRegion::new(physical.offset, 0, physical.length)?;
                let prepared =
                    runtime.prepare_submission_readback(DeviceSubmissionReadbackRequest {
                        source,
                        region,
                        layout,
                        retention,
                        staging,
                    });
                let Some(prepared) = prepared else {
                    unsupported = true;
                    return Ok(());
                };
                let (command, snapshot) = prepared
                    .map_err(|error| {
                        invalid_completion(format!(
                            "submission readback preparation failed: {error}"
                        ))
                    })?
                    .into_parts();
                if let Some(command) = command {
                    staged_commands.push(command);
                }
                pieces.push(PreparedReadbackPiece { physical, snapshot });
                Ok(())
            })?;
            if unsupported {
                // No command has reached the submission batch, so all partial
                // staging allocations and source retentions can roll back.
                return Ok(());
            }
            outputs.push(PreparedReadbackOutput {
                request: item.clone(),
                pieces,
            });
        }
        self.readbacks = Some(Arc::new(PreparedCompletionReadbacks {
            slot_id: self.slot_id,
            batch_identity_fingerprint: identity.fingerprint().to_owned(),
            request,
            outputs,
        }));
        for command in staged_commands {
            commands.push_result_binding(command);
        }
        Ok(())
    }
}

impl<E> PreparedReadbackOutput<E> {
    fn read<R: DeviceRuntime<Error = E>>(
        &self,
        resources: &CompletionResourceLease<R>,
        lane: &Arc<ExecutionLane<R>>,
        timing_mode: DeviceTimingMode,
    ) -> Result<LaneReadback, LaneReadbackError<E>> {
        let started = timing_mode.completion_enabled().then(Instant::now);
        if lane.fail_closed.load(Ordering::Acquire) || !lane.current_descriptor_matches_snapshot() {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "staged readback lane is no longer valid",
            )));
        }
        let mut index = 0_usize;
        visit_readback_pieces(
            resources,
            lane.runtime(),
            &self.request,
            |physical, _, _| {
                if self
                    .pieces
                    .get(index)
                    .is_none_or(|piece| piece.physical != physical)
                {
                    return Err(invalid_completion(
                        "staged readback source differs from its submitted extent",
                    ));
                }
                index += 1;
                Ok(())
            },
        )
        .map_err(LaneReadbackError::Contract)?;
        if index != self.pieces.len() {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "staged readback segment count changed",
            )));
        }
        let mut bytes = Vec::new();
        for piece in &self.pieces {
            let output = piece.snapshot.read_at_terminal().map_err(|error| {
                lane.fail_closed();
                LaneReadbackError::Device(error)
            })?;
            if u64::try_from(output.len()).ok() != Some(piece.physical.length) {
                lane.fail_closed();
                return Err(LaneReadbackError::Contract(invalid_completion(
                    "staged readback returned a wrong byte count",
                )));
            }
            bytes.extend(output);
        }
        if !lane.current_descriptor_matches_snapshot() {
            lane.fail_closed();
            return Err(LaneReadbackError::Contract(invalid_completion(
                "runtime descriptor drifted during staged readback",
            )));
        }
        let timing = match started {
            None => DeviceTimingMeasurement::NotRequested,
            Some(started) => DeviceTimingMeasurement::Measured(CompletionReadbackTiming::new(
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                u32::try_from(self.pieces.len()).unwrap_or(u32::MAX),
                bytes.len() as u64,
            )),
        };
        Ok(LaneReadback { bytes, timing })
    }
}

impl<R: DeviceRuntime> CompletionReaper<R> {
    pub(super) fn wait_bound_with_submission_readbacks(
        &self,
        slot_id: CompletionSlotId,
        request: CompletionReadbackBatchRequest,
        prepared: &PreparedCompletionReadbacks<R::Error>,
    ) -> Result<CompletionReadbackBatchObservation, VNextError> {
        if prepared.slot_id != slot_id || request != prepared.request {
            return Err(invalid_completion(
                "staged readback request differs from its exact submitted snapshot",
            ));
        }
        self.validate_bound_readback_batch(slot_id, &request)?;
        let observation = self.observe_bound_with(
            slot_id,
            true,
            |resources, lane, identity, disposition, timing_mode| {
                prepared
                    .outputs
                    .iter()
                    .map(|output| {
                        let request = output.request.clone();
                        if !matches!(disposition, OperationCompletionDisposition::Succeeded) {
                            return CompletionReadbackDisposition::NotAttempted(request);
                        }
                        if identity.fingerprint() != prepared.batch_identity_fingerprint {
                            return CompletionReadbackDisposition::ContractFailedButQuiescent {
                                request,
                                failure: QuiescentCompletionContractFailure::new(
                                    "staged readback belongs to another submission",
                                ),
                            };
                        }
                        let readback = catch_unwind(AssertUnwindSafe(|| {
                            output.read(resources, lane, timing_mode)
                        }));
                        match readback {
                            Ok(Ok(readback)) => {
                                match CompletionReadbackOutput::new(request.clone(), readback) {
                                    Ok(output) => CompletionReadbackDisposition::Succeeded(output),
                                    Err(error) => {
                                        CompletionReadbackDisposition::ContractFailedButQuiescent {
                                            request,
                                            failure: QuiescentCompletionContractFailure::new(
                                                error.to_string(),
                                            ),
                                        }
                                    }
                                }
                            }
                            Ok(Err(LaneReadbackError::Contract(error))) => {
                                CompletionReadbackDisposition::ContractFailedButQuiescent {
                                    request,
                                    failure: QuiescentCompletionContractFailure::new(
                                        error.to_string(),
                                    ),
                                }
                            }
                            Ok(Err(LaneReadbackError::Device(error))) => {
                                match catch_unwind(AssertUnwindSafe(|| {
                                    classify_batch_device_error(lane.runtime(), identity, &error)
                                })) {
                                    Ok(Ok(failures)) => {
                                        CompletionReadbackDisposition::FailedButQuiescent {
                                            request,
                                            failures,
                                        }
                                    }
                                    Ok(Err(error)) => {
                                        CompletionReadbackDisposition::ContractFailedButQuiescent {
                                            request,
                                            failure: QuiescentCompletionContractFailure::new(
                                                error.to_string(),
                                            ),
                                        }
                                    }
                                    Err(_) => {
                                        lane.fail_closed();
                                        CompletionReadbackDisposition::ContractFailedButQuiescent {
                                            request,
                                            failure: QuiescentCompletionContractFailure::new(
                                                "device runtime panicked while classifying staged readback failure",
                                            ),
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                lane.fail_closed();
                                CompletionReadbackDisposition::ContractFailedButQuiescent {
                                    request,
                                    failure: QuiescentCompletionContractFailure::new(
                                        "device runtime panicked during staged readback",
                                    ),
                                }
                            }
                        }
                    })
                    .collect()
            },
        )?;
        Ok(match observation {
            BoundCompletionObservation::Pending => CompletionReadbackBatchObservation::Pending,
            BoundCompletionObservation::Terminal {
                completion,
                terminal,
            } => CompletionReadbackBatchObservation::Terminal(CompletionReadbackBatchReceipt::new(
                completion, terminal,
            )),
            BoundCompletionObservation::Indeterminate(failures) => {
                CompletionReadbackBatchObservation::Indeterminate(failures)
            }
            BoundCompletionObservation::SubmissionIndeterminate => {
                CompletionReadbackBatchObservation::SubmissionIndeterminate
            }
            BoundCompletionObservation::ObservationPanicked => {
                CompletionReadbackBatchObservation::ObservationPanicked
            }
            BoundCompletionObservation::Quarantined(receipt) => {
                CompletionReadbackBatchObservation::Quarantined(receipt)
            }
        })
    }
}
