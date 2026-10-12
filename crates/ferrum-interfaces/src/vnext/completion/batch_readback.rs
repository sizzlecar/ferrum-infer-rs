//! One successful-terminal lane transaction. Views and every exact lease stay
//! owned until synchronous copying and output assembly finish. No cross-wave
//! view or completed-fence-as-stream-idle authority is stored here.
use super::*;
use crate::vnext::{DeviceReadbackBatchOutcome, DeviceReadbackPiece};

struct Backing<'a, B> {
    view: LogicalBackingBufferView<'a, B>,
    ends: Vec<u64>,
}

#[derive(Clone, Copy)]
struct Piece {
    backing: usize,
    binding: usize,
    source_offset: u64,
    length: u64,
    layout: HostTransferLayout,
}

struct Plan {
    request: usize,
    pieces: Range<usize>,
    bytes: usize,
}

fn failed(
    request: CompletionReadbackRequest,
    reason: impl Into<String>,
) -> CompletionReadbackDisposition {
    CompletionReadbackDisposition::ContractFailedButQuiescent {
        request,
        failure: QuiescentCompletionContractFailure::new(reason),
    }
}

fn group_key<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    request: &CompletionReadbackRequest,
) -> Result<(NodeId, ResourceId, Option<u32>), VNextError> {
    let shared = match resources {
        CompletionResourceLease::Invocation(invocation) => {
            if invocation.node_id() != request.node_id() {
                return Err(invalid_completion(
                    "completion readback node differs from its invocation",
                ));
            }
            invocation
                .backing_slices()
                .iter()
                .chain(invocation.step_resources().backing_slices())
                .any(|a| a.resource_id() == request.resource_id())
        }
        CompletionResourceLease::Wave(wave) => {
            if !wave
                .nodes()
                .iter()
                .any(|node| node.node_id() == request.node_id())
            {
                return Err(invalid_completion(
                    "completion readback node is absent from its wave",
                ));
            }
            wave.claimed_backing()
                .backing_slices()
                .iter()
                .chain(wave.step_resources().backing_slices())
                .any(|a| a.resource_id() == request.resource_id())
        }
    };
    Ok((
        request.node_id().clone(),
        request.resource_id().clone(),
        (!shared).then_some(request.participant_index()),
    ))
}

fn prepare<'a, R: DeviceRuntime>(
    resources: &'a CompletionResourceLease<R>,
    requests: &[CompletionReadbackRequest],
) -> (
    Vec<Backing<'a, R::Buffer>>,
    Vec<Piece>,
    Vec<Plan>,
    Vec<Option<CompletionReadbackDisposition>>,
) {
    let mut backings = Vec::<Backing<'a, R::Buffer>>::new();
    let mut groups = BTreeMap::new();
    let mut pieces = Vec::new();
    let mut plans = Vec::new();
    let mut dispositions = (0..requests.len()).map(|_| None).collect::<Vec<_>>();
    for (index, request) in requests.iter().enumerate() {
        let start = pieces.len();
        let result = (|| -> Result<usize, VNextError> {
            let bytes = request.output_layout().byte_len()?;
            let end = request
                .logical_offset_bytes()
                .checked_add(bytes)
                .ok_or_else(|| {
                    invalid_completion("completion readback semantic range overflows u64")
                })?;
            let range = resources.readback_range(
                request.node_id(),
                request.participant_index(),
                request.resource_id(),
                request.logical_offset_bytes()..end,
            )?;
            let key = group_key(resources, request)?;
            let backing_index = match groups.get(&key).copied() {
                Some(index) => index,
                None => {
                    let view = resources.backing_view(
                        request.node_id(),
                        request.participant_index(),
                        request.resource_id(),
                    )?;
                    let mut cursor = 0u64;
                    let ends = view
                        .segment_bindings()
                        .iter()
                        .map(|binding| {
                            cursor = cursor
                                .checked_add(binding.segment().length_bytes())
                                .ok_or_else(|| {
                                    invalid_completion(
                                        "completion readback backing coverage overflows u64",
                                    )
                                })?;
                            Ok(cursor)
                        })
                        .collect::<Result<Vec<_>, VNextError>>()?;
                    let index = backings.len();
                    backings.push(Backing { view, ends });
                    groups.insert(key, index);
                    index
                }
            };
            let backing = &backings[backing_index];
            let element = request.output_layout().element_type().size_bytes();
            let logical_end = range.start.checked_add(bytes).ok_or_else(|| {
                invalid_completion("completion readback logical range overflows u64")
            })?;
            if backing.view.usage() != request.expected_usage()
                || backing.view.element_type() != request.output_layout().element_type()
                || logical_end > backing.view.size_bytes()
                || range.start % element != 0
                || bytes % element != 0
            {
                return Err(invalid_completion("completion readback must select an aligned typed backing range with matching usage and element type"));
            }
            let mut binding = backing.ends.partition_point(|end| *end <= range.start);
            let mut logical = binding
                .checked_sub(1)
                .map_or(0, |previous| backing.ends[previous]);
            let mut covered = 0u64;
            while let Some(&segment_end) = backing.ends.get(binding) {
                if logical >= logical_end {
                    break;
                }
                let begin = logical.max(range.start);
                let finish = segment_end.min(logical_end);
                if begin < finish {
                    let segment = backing.view.segment_bindings()[binding].segment();
                    let source_offset = segment
                        .offset_bytes()
                        .checked_add(begin - logical)
                        .ok_or_else(|| {
                            invalid_completion("completion readback physical offset overflows u64")
                        })?;
                    let length = finish - begin;
                    if source_offset % element != 0 || length % element != 0 {
                        return Err(invalid_completion(
                            "completion readback backing segments split an element",
                        ));
                    }
                    let layout = HostTransferLayout::new(
                        request.output_layout().element_type(),
                        length / element,
                    )?;
                    pieces.push(Piece {
                        backing: backing_index,
                        binding,
                        source_offset,
                        length,
                        layout,
                    });
                    covered = covered.checked_add(length).ok_or_else(|| {
                        invalid_completion("completion readback coverage overflows")
                    })?;
                }
                logical = segment_end;
                binding += 1;
            }
            if covered != bytes {
                return Err(invalid_completion(
                    "completion readback did not cover its exact logical output",
                ));
            }
            usize::try_from(bytes).map_err(|_| {
                invalid_completion("completion readback output exceeds host address space")
            })
        })();
        match result {
            Ok(bytes) => plans.push(Plan {
                request: index,
                pieces: start..pieces.len(),
                bytes,
            }),
            Err(error) => {
                pieces.truncate(start);
                dispositions[index] = Some(failed(request.clone(), error.to_string()));
            }
        }
    }
    (backings, pieces, plans, dispositions)
}

fn output_prefix_is_valid<E>(outcome: &DeviceReadbackBatchOutcome<E>, pieces: &[Piece]) -> bool {
    let (outputs, failed) = match outcome {
        DeviceReadbackBatchOutcome::Complete(outputs) => (outputs, None),
        DeviceReadbackBatchOutcome::Failed {
            completed,
            failed_input,
            ..
        }
        | DeviceReadbackBatchOutcome::Panicked {
            completed,
            failed_input,
        }
        | DeviceReadbackBatchOutcome::ContractFailed {
            completed,
            failed_input,
            ..
        } => (completed, Some(*failed_input)),
    };
    if match failed {
        Some(index) => index >= pieces.len() || outputs.len() != index,
        None => outputs.len() != pieces.len(),
    } {
        return false;
    }
    outputs.iter().enumerate().all(|(index, output)| {
        output.input_index == index
            && u64::try_from(output.bytes.len()).ok() == Some(pieces[index].length)
    })
}

pub(super) fn attempt<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    identity: &BatchOperationIdentity,
    completion: &OperationCompletionDisposition,
    timing_mode: DeviceTimingMode,
    requests: Vec<CompletionReadbackRequest>,
) -> Vec<CompletionReadbackDisposition> {
    if !matches!(completion, OperationCompletionDisposition::Succeeded) {
        return requests
            .into_iter()
            .map(CompletionReadbackDisposition::NotAttempted)
            .collect();
    }
    match catch_unwind(AssertUnwindSafe(|| {
        attempt_inner(resources, lane, identity, timing_mode, &requests)
    })) {
        Ok(dispositions) => dispositions,
        Err(_) => {
            lane.fail_closed();
            requests
                .into_iter()
                .map(|request| failed(request, "completion batch readback panicked"))
                .collect()
        }
    }
}

fn attempt_inner<R: DeviceRuntime>(
    resources: &CompletionResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    identity: &BatchOperationIdentity,
    timing_mode: DeviceTimingMode,
    requests: &[CompletionReadbackRequest],
) -> Vec<CompletionReadbackDisposition> {
    let started = timing_mode.completion_enabled().then(Instant::now);
    let (backings, mut pieces, mut plans, mut dispositions) = prepare(resources, requests);
    let copy = (|| -> Result<_, LaneReadbackError<R::Error>> {
        if lane.fail_closed.load(Ordering::Acquire) || !lane.current_descriptor_matches_snapshot() {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "completion readback requires its original reusable execution lane",
            )));
        }
        let mut state = lane.state.lock().map_err(|_| {
            LaneReadbackError::Contract(invalid_completion(
                "execution lane state mutex is poisoned during completion readback",
            ))
        })?;
        if state.fail_closed
            || !matches!(
                lane.runtime.stream_state(&state.stream),
                StreamState::Ready | StreamState::Submitted
            )
        {
            return Err(LaneReadbackError::Contract(invalid_completion(
                "completion readback lane is failed or not readable",
            )));
        }
        let mut checked = BTreeMap::new();
        let old_pieces = std::mem::take(&mut pieces);
        for mut plan in std::mem::take(&mut plans) {
            let valid = old_pieces[plan.pieces.clone()].iter().all(|piece| {
                let binding = &backings[piece.backing].view.segment_bindings()[piece.binding];
                let actual = checked
                    .entry((piece.backing, piece.binding))
                    .or_insert_with(|| lane.runtime.buffer_descriptor(binding.buffer()));
                &*actual == binding.descriptor()
                    && piece
                        .source_offset
                        .checked_add(piece.length)
                        .is_some_and(|end| end <= actual.size_bytes)
            });
            if !valid {
                dispositions[plan.request] = Some(failed(
                    requests[plan.request].clone(),
                    "completion readback backing descriptor differs from its committed extent",
                ));
                continue;
            }
            let start = pieces.len();
            pieces.extend_from_slice(&old_pieces[plan.pieces]);
            plan.pieces = start..pieces.len();
            plans.push(plan);
        }
        let inputs = pieces
            .iter()
            .map(|piece| {
                let buffer =
                    backings[piece.backing].view.segment_bindings()[piece.binding].buffer();
                DeviceReadbackPiece::new(
                    buffer,
                    CopyRegion::new(piece.source_offset, 0, piece.length)
                        .expect("checked readback piece"),
                    piece.layout,
                )
            })
            .collect::<Vec<_>>();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            lane.runtime.readback_batch(&mut state.stream, &inputs)
        }))
        .map_err(|_| {
            LaneReadbackError::Contract(invalid_completion(
                "device runtime panicked during batch completion readback",
            ))
        })?;
        if !output_prefix_is_valid(&outcome, &pieces) || !lane.current_descriptor_matches_snapshot()
        {
            state.fail_closed = true;
            lane.fail_closed.store(true, Ordering::Release);
            return Err(LaneReadbackError::Contract(invalid_completion(
                "device runtime returned invalid batch readback order, count, length or descriptor",
            )));
        }
        if !matches!(outcome, DeviceReadbackBatchOutcome::Complete(_)) {
            state.fail_closed = true;
            lane.fail_closed.store(true, Ordering::Release);
        }
        Ok(outcome)
    })();
    let (outputs, failure) = match copy {
        Ok(DeviceReadbackBatchOutcome::Complete(outputs)) => (outputs, None),
        Ok(DeviceReadbackBatchOutcome::Failed {
            completed,
            failed_input,
            error,
        }) => (
            completed,
            Some((failed_input, LaneReadbackError::Device(error))),
        ),
        Ok(DeviceReadbackBatchOutcome::Panicked {
            completed,
            failed_input,
        }) => (
            completed,
            Some((
                failed_input,
                LaneReadbackError::Contract(invalid_completion(
                    "device runtime panicked during completion readback",
                )),
            )),
        ),
        Ok(DeviceReadbackBatchOutcome::ContractFailed {
            completed,
            failed_input,
            reason,
        }) => (
            completed,
            Some((
                failed_input,
                LaneReadbackError::Contract(invalid_completion(reason)),
            )),
        ),
        Err(error) => {
            lane.fail_closed();
            (Vec::new(), Some((0, error)))
        }
    };
    let completed = outputs.len();
    let total_bytes = outputs.iter().fold(0u64, |sum, output| {
        sum.saturating_add(output.bytes.len() as u64)
    });
    // Exactly one success carries the entire batch measurement. Calls counts
    // the one Runtime batch API, not per-request CUDA driver transfers.
    let mut timing =
        started.map_or(
            DeviceTimingMeasurement::NotRequested,
            |start| match u64::try_from(start.elapsed().as_nanos()) {
                Ok(ns) => DeviceTimingMeasurement::Measured(CompletionReadbackTiming::batch(
                    ns,
                    u32::from(!pieces.is_empty()),
                    total_bytes,
                )),
                Err(_) => DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::DurationOverflow,
                ),
            },
        );
    let mut outputs = outputs.into_iter().peekable();
    let mut failure = failure;
    for plan in plans {
        let request = requests[plan.request].clone();
        let disposition = if plan.pieces.end <= completed {
            let bytes = if plan.pieces.len() == 1 {
                outputs.next().expect("validated completed prefix").bytes
            } else {
                let mut bytes = Vec::with_capacity(plan.bytes);
                for _ in plan.pieces.clone() {
                    bytes.extend_from_slice(
                        &outputs.next().expect("validated completed prefix").bytes,
                    );
                }
                bytes
            };
            let output =
                CompletionReadbackOutput::new(request.clone(), LaneReadback { bytes, timing });
            timing = DeviceTimingMeasurement::NotRequested;
            match output {
                Ok(output) => CompletionReadbackDisposition::Succeeded(output),
                Err(error) => {
                    lane.fail_closed();
                    failed(request, error.to_string())
                }
            }
        } else if failure
            .as_ref()
            .is_some_and(|(index, _)| plan.pieces.contains(index))
        {
            let (_, error) = failure.take().expect("current first failure");
            // Discard any partial bytes from the request which failed.
            while outputs
                .peek()
                .is_some_and(|output| plan.pieces.contains(&output.input_index))
            {
                outputs.next();
            }
            match error {
                LaneReadbackError::Contract(error) => failed(request, error.to_string()),
                LaneReadbackError::Device(error) => {
                    match classify_batch_device_error(lane.runtime(), identity, &error) {
                        Ok(failures) => {
                            CompletionReadbackDisposition::FailedButQuiescent { request, failures }
                        }
                        Err(error) => failed(request, error.to_string()),
                    }
                }
            }
        } else {
            failed(
                request,
                "completion readback lane is failed or not readable",
            )
        };
        dispositions[plan.request] = Some(disposition);
    }
    dispositions
        .into_iter()
        .map(|output| output.expect("each readback request receives a disposition"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::{DeviceReadbackPieceOutput, ElementType};

    #[test]
    fn completion_batch_readback_rejects_wrong_output_ordinals_lengths_and_failure_prefixes() {
        let pieces = (0..2)
            .map(|_| Piece {
                backing: 0,
                binding: 0,
                source_offset: 0,
                length: 4,
                layout: HostTransferLayout::new(ElementType::U32, 1).unwrap(),
            })
            .collect::<Vec<_>>();
        let output = |input_index, bytes| DeviceReadbackPieceOutput {
            input_index,
            bytes: vec![0; bytes],
        };
        assert!(output_prefix_is_valid(
            &DeviceReadbackBatchOutcome::<()>::Complete(vec![output(0, 4), output(1, 4)]),
            &pieces
        ));
        for outputs in [
            vec![output(1, 4), output(0, 4)],
            vec![output(0, 4), output(0, 4)],
            vec![output(0, 3), output(1, 4)],
            vec![output(0, 4)],
            vec![output(0, 4), output(1, 4), output(2, 4)],
        ] {
            assert!(!output_prefix_is_valid(
                &DeviceReadbackBatchOutcome::<()>::Complete(outputs),
                &pieces
            ));
        }
        assert!(output_prefix_is_valid(
            &DeviceReadbackBatchOutcome::Failed {
                completed: vec![output(0, 4)],
                failed_input: 1,
                error: ()
            },
            &pieces
        ));
        assert!(!output_prefix_is_valid(
            &DeviceReadbackBatchOutcome::Failed {
                completed: vec![output(0, 4)],
                failed_input: 0,
                error: ()
            },
            &pieces
        ));
        assert!(!output_prefix_is_valid(
            &DeviceReadbackBatchOutcome::Failed {
                completed: vec![output(0, 4), output(1, 4)],
                failed_input: 2,
                error: ()
            },
            &pieces
        ));
        assert!(!output_prefix_is_valid(
            &DeviceReadbackBatchOutcome::<()>::Panicked {
                completed: vec![output(0, 3)],
                failed_input: 1
            },
            &pieces
        ));
    }
}
