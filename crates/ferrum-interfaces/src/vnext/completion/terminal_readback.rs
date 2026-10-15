use super::*;
use crate::vnext::{DeviceScalarU32ReadbackBatch, DeviceScalarU32ReadbackSource, ElementType};

pub(crate) enum TerminalReadbackPrepareError<E> {
    Contract(VNextError),
    Runtime(E),
}

impl<E> From<VNextError> for TerminalReadbackPrepareError<E> {
    fn from(error: VNextError) -> Self {
        Self::Contract(error)
    }
}

impl<R: DeviceRuntime> CompletionResourceLease<R> {
    pub(super) fn prepared_terminal_request(&self) -> Option<&CompletionReadbackBatchRequest> {
        match self {
            Self::Resident(use_) if use_.has_prepared_terminal_readbacks() => {
                use_.terminal_readback_request()
            }
            Self::Wave(wave) if wave.has_prepared_terminal_readbacks() => {
                wave.terminal_readback_request()
            }
            _ => None,
        }
    }

    pub(super) fn take_terminal_readbacks(
        &mut self,
    ) -> Option<(
        CompletionReadbackBatchRequest,
        Arc<dyn std::any::Any + Send + Sync>,
    )> {
        match self {
            Self::Wave(wave) => wave.take_terminal_readbacks(),
            Self::Resident(use_) => use_.take_terminal_readbacks(),
            Self::Invocation(_) => None,
        }
    }
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    /// Resolve from this reservation's current wave, never from a cached pointer.
    /// This runs under the existing lane enqueue reservation, before submission.
    pub(crate) fn prepare_terminal_readbacks(
        &mut self,
        enqueue: &mut ExecutionLaneEnqueue<'_, R>,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<(), TerminalReadbackPrepareError<R::Error>> {
        let resources = self
            .resources
            .as_ref()
            .expect("live reservation owns resources");
        let request = match resources {
            CompletionResourceLease::Wave(wave) => wave.terminal_readback_request(),
            CompletionResourceLease::Resident(use_) => use_.terminal_readback_request(),
            CompletionResourceLease::Invocation(_) => None,
        };
        let Some(request) = request else {
            return Ok(());
        };
        request.validate_for(self.batch_identity.as_ref().expect("reservation identity"))?;
        if request.len() > 32
            || request.requests().iter().any(|item| {
                item.output_layout().element_type() != ElementType::U32
                    || item.output_layout().byte_len().ok() != Some(4)
            })
        {
            return Ok(());
        }
        let runtime = enqueue.lane.runtime.as_ref();
        let mut views = Vec::with_capacity(request.len());
        let mut ranges = Vec::with_capacity(request.len());
        for item in request.requests() {
            let end = item
                .logical_offset_bytes()
                .checked_add(4)
                .ok_or_else(|| invalid_completion("terminal scalar range overflow"))?;
            let range = resources.readback_range(
                item.node_id(),
                item.participant_index(),
                item.resource_id(),
                item.logical_offset_bytes()..end,
            )?;
            let view = resources.backing_view(
                item.node_id(),
                item.participant_index(),
                item.resource_id(),
            )?;
            if view.usage() != item.expected_usage()
                || view.element_type() != ElementType::U32
                || range.end > view.size_bytes()
                || range.start % 4 != 0
                || range.end - range.start != 4
            {
                return Err(invalid_completion(
                    "terminal scalar range is not an aligned current typed output",
                )
                .into());
            }
            views.push(view);
            ranges.push(range);
        }
        let mut sources = Vec::with_capacity(request.len());
        for (view, range) in views.iter().zip(&ranges) {
            let mut cursor = 0_u64;
            let mut found = None;
            for binding in view.segment_bindings() {
                let segment = binding.segment();
                let end = cursor.checked_add(segment.length_bytes()).ok_or_else(|| {
                    invalid_completion("terminal scalar backing coverage overflow")
                })?;
                if range.start < end && cursor < range.end {
                    if range.start < cursor || range.end > end {
                        return Err(invalid_completion(
                            "terminal scalar backing splits one element",
                        )
                        .into());
                    }
                    let offset = segment
                        .offset_bytes()
                        .checked_add(range.start - cursor)
                        .ok_or_else(|| {
                            invalid_completion("terminal scalar physical offset overflow")
                        })?;
                    let actual = runtime.buffer_descriptor(binding.buffer());
                    if &actual != binding.descriptor()
                        || offset % 4 != 0
                        || offset
                            .checked_add(4)
                            .is_none_or(|end| end > actual.size_bytes)
                        || found.is_some()
                    {
                        return Err(invalid_completion(
                            "terminal scalar backing descriptor or extent changed",
                        )
                        .into());
                    }
                    found = Some(DeviceScalarU32ReadbackSource::new(
                        binding.buffer(),
                        binding.descriptor(),
                        CopyRegion::new(offset, 0, 4)?,
                        binding.retention(),
                    ));
                }
                cursor = end;
            }
            sources.push(
                found.ok_or_else(|| invalid_completion("terminal scalar backing is incomplete"))?,
            );
        }
        // No backend call may enqueue in this phase. A panic fails the lane
        // closed rather than being interpreted as an unsupported capability.
        let prepared = catch_unwind(AssertUnwindSafe(|| {
            runtime.prepare_terminal_u32_readbacks(
                &mut enqueue.state.stream,
                DeviceScalarU32ReadbackBatch::new(sources),
            )
        }));
        let prepared = match prepared {
            Ok(result) => result.map_err(TerminalReadbackPrepareError::Runtime)?,
            Err(_) => {
                enqueue.lane.fail_closed();
                return Err(invalid_completion(
                    "terminal readback preparation panicked before submission",
                )
                .into());
            }
        };
        drop(views);
        let Some(prepared) = prepared else {
            return Ok(());
        };
        let (command, payload) = prepared.into_parts();
        commands.set_terminal_readback(command)?;
        match self.resources.as_mut().expect("reservation resources") {
            CompletionResourceLease::Wave(wave) => wave.install_terminal_readback_payload(payload),
            CompletionResourceLease::Resident(use_) => {
                use_.install_terminal_readback_payload(payload)
            }
            CompletionResourceLease::Invocation(_) => unreachable!(),
        }
        Ok(())
    }
}

fn contract_dispositions(
    request: CompletionReadbackBatchRequest,
    message: String,
) -> Vec<CompletionReadbackDisposition> {
    request
        .into_requests()
        .into_iter()
        .map(
            |request| CompletionReadbackDisposition::ContractFailedButQuiescent {
                request,
                failure: QuiescentCompletionContractFailure::new(message.clone()),
            },
        )
        .collect()
}

pub(super) fn finish_prepared_terminal_readbacks<R: DeviceRuntime>(
    lane: &Arc<ExecutionLane<R>>,
    identity: &BatchOperationIdentity,
    disposition: &OperationCompletionDisposition,
    timing_mode: DeviceTimingMode,
    request: CompletionReadbackBatchRequest,
    prepared_request: CompletionReadbackBatchRequest,
    payload: Arc<dyn std::any::Any + Send + Sync>,
) -> Vec<CompletionReadbackDisposition> {
    if request != prepared_request {
        lane.fail_closed();
        return contract_dispositions(
            request,
            "prepared terminal readback request differs from submission".to_owned(),
        );
    }
    if !matches!(disposition, OperationCompletionDisposition::Succeeded) {
        return request
            .into_requests()
            .into_iter()
            .map(CompletionReadbackDisposition::NotAttempted)
            .collect();
    }
    if !lane.current_descriptor_matches_snapshot() || lane.fail_closed.load(Ordering::Acquire) {
        lane.fail_closed();
        return contract_dispositions(
            request,
            "prepared terminal readback lane changed or failed".to_owned(),
        );
    }
    let output = catch_unwind(AssertUnwindSafe(|| {
        lane.runtime().finish_terminal_u32_readbacks(payload)
    }));
    let values = match output {
        Ok(Some(Ok(values))) if values.len() == request.len() => values,
        Ok(Some(Err(error))) => {
            lane.fail_closed();
            return match catch_unwind(AssertUnwindSafe(|| {
                classify_batch_device_error(lane.runtime(), identity, &error)
            })) {
                Ok(Ok(failures)) => request
                    .into_requests()
                    .into_iter()
                    .map(
                        |request| CompletionReadbackDisposition::FailedButQuiescent {
                            request,
                            failures: failures.clone(),
                        },
                    )
                    .collect(),
                _ => contract_dispositions(
                    request,
                    "prepared terminal readback error classification failed".to_owned(),
                ),
            };
        }
        _ => {
            lane.fail_closed();
            return contract_dispositions(
                request,
                "prepared terminal readback extraction failed, panicked or returned a wrong count"
                    .to_owned(),
            );
        }
    };
    let timing = if timing_mode.completion_enabled() {
        DeviceTimingMeasurement::Unavailable(DeviceTimingUnavailableReason::BackendUnsupported)
    } else {
        DeviceTimingMeasurement::NotRequested
    };
    request
        .into_requests()
        .into_iter()
        .zip(values)
        .map(|(request, value)| {
            match CompletionReadbackOutput::new(
                request.clone(),
                LaneReadback {
                    bytes: value.to_le_bytes().to_vec(),
                    timing,
                },
            ) {
                Ok(output) => CompletionReadbackDisposition::Succeeded(output),
                Err(error) => CompletionReadbackDisposition::ContractFailedButQuiescent {
                    request,
                    failure: QuiescentCompletionContractFailure::new(error.to_string()),
                },
            }
        })
        .collect()
}
