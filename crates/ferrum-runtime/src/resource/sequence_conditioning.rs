//! Immutable, accounted sequence inputs used by prefix-causal state execution.
//! Only metadata is copied. Payloads retain their original Host scope and charge.
use super::*;
use crate::{
    HostMemoryAccount, ManagedHostBytes, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
    SubmissionWaveInputUpload,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceConditioningCoverage {
    Exact,
    ZeroOutsideSegments,
}

#[derive(Debug, Clone)]
pub struct SequenceConditioningSegment {
    start_row: usize,
    bytes: ManagedHostBytes,
}
impl SequenceConditioningSegment {
    pub fn new(start_row: usize, bytes: ManagedHostBytes) -> Result<Self, VNextError> {
        if bytes.is_empty() {
            return Err(invalid_resource("conditioning segment is empty"));
        }
        Ok(Self { start_row, bytes })
    }
}

#[derive(Debug, Clone)]
pub struct SequenceConditioningInput {
    value: ProgramValueId,
    tensor: ResolvedTensorSpec,
    coverage: SequenceConditioningCoverage,
    segments: Vec<SequenceConditioningSegment>,
    row_bytes: usize,
}
impl SequenceConditioningInput {
    /// `tensor` covers the complete immutable source, including its leading row
    /// axis. Exact inputs cover every row; sparse inputs define all gaps as zero.
    pub fn new(
        host: &HostMemoryAccount,
        value: ProgramValueId,
        tensor: ResolvedTensorSpec,
        coverage: SequenceConditioningCoverage,
        mut segments: Vec<SequenceConditioningSegment>,
    ) -> Result<Self, VNextError> {
        let dims = tensor.dimensions();
        if !matches!(tensor.layout(), ResolvedTensorLayout::Contiguous)
            || dims.is_empty()
            || dims.iter().any(|&d| d == 0)
        {
            return Err(invalid_resource(
                "conditioning requires a nonempty contiguous row schema",
            ));
        }
        let rows =
            usize::try_from(dims[0]).map_err(|_| invalid_resource("conditioning rows overflow"))?;
        let row_bytes = dims[1..]
            .iter()
            .try_fold(tensor.element_type().size_bytes(), |n, &d| n.checked_mul(d))
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid_resource("conditioning row extent overflows"))?;
        let _ = rows
            .checked_mul(row_bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| invalid_resource("conditioning source extent exceeds address space"))?;
        segments.sort_by_key(|s| s.start_row);
        let mut end = 0;
        for segment in &segments {
            if !host.owns(&segment.bytes)
                || segment.bytes.len() % row_bytes != 0
                || segment.start_row < end
                || (coverage == SequenceConditioningCoverage::Exact && segment.start_row != end)
            {
                return Err(invalid_resource(
                    "conditioning source owners or segment coverage disagree",
                ));
            }
            end = segment
                .start_row
                .checked_add(segment.bytes.len() / row_bytes)
                .filter(|&n| n <= rows)
                .ok_or_else(|| invalid_resource("conditioning segment exceeds source rows"))?;
        }
        if coverage == SequenceConditioningCoverage::Exact && end != rows {
            return Err(invalid_resource(
                "exact conditioning must cover all source rows",
            ));
        }
        Ok(Self {
            value,
            tensor,
            coverage,
            segments,
            row_bytes,
        })
    }

    fn matches_bytes(&self, start: usize, actual: &[u8]) -> Result<bool, VNextError> {
        let end = start
            .checked_add(actual.len())
            .ok_or_else(|| invalid_resource("conditioning comparison range overflows"))?;
        let total = usize::try_from(self.tensor.minimum_storage_bytes()?)
            .map_err(|_| invalid_resource("conditioning source extent overflows"))?;
        if end > total {
            return Ok(false);
        }
        let mut cursor = start;
        for segment in &self.segments {
            let begin = segment.start_row * self.row_bytes;
            let stop = begin + segment.bytes.len();
            if stop <= cursor || begin >= end {
                continue;
            }
            if begin > cursor {
                let gap_end = begin.min(end);
                if self.coverage != SequenceConditioningCoverage::ZeroOutsideSegments
                    || actual[cursor - start..gap_end - start]
                        .iter()
                        .any(|&b| b != 0)
                {
                    return Ok(false);
                }
                cursor = gap_end;
            }
            let overlap_end = stop.min(end);
            if actual[cursor - start..overlap_end - start]
                != segment.bytes.bytes()[cursor - begin..overlap_end - begin]
            {
                return Ok(false);
            }
            cursor = overlap_end;
        }
        Ok(cursor == end
            || (self.coverage == SequenceConditioningCoverage::ZeroOutsideSegments
                && actual[cursor - start..].iter().all(|&b| b == 0)))
    }

    fn matches_owner_or_bytes(
        &self,
        start: usize,
        actual: &ManagedHostBytes,
    ) -> Result<bool, VNextError> {
        let end = start
            .checked_add(actual.len())
            .ok_or_else(|| invalid_resource("conditioning comparison range overflows"))?;
        for segment in &self.segments {
            let begin = segment.start_row * self.row_bytes;
            if begin <= start && end <= begin + segment.bytes.len() {
                let region = segment
                    .bytes
                    .slice(start - begin..end - begin)
                    .map_err(|e| invalid_resource(e.to_string()))?;
                if region.same_region(actual) {
                    return Ok(true);
                }
                break;
            }
        }
        self.matches_bytes(start, actual.bytes())
    }
}

#[derive(Debug)]
pub struct SequenceConditioningInputs {
    inputs: BTreeMap<ProgramValueId, SequenceConditioningInput>,
}
impl SequenceConditioningInputs {
    pub fn new(inputs: Vec<SequenceConditioningInput>) -> Result<Self, VNextError> {
        let mut values = BTreeMap::new();
        for input in inputs {
            if values.insert(input.value.clone(), input).is_some() {
                return Err(invalid_resource("duplicate sequence conditioning input"));
            }
        }
        Ok(Self { inputs: values })
    }
    fn validate_layout(&self, layout: &crate::CandidateStateLayout) -> Result<(), VNextError> {
        if self.inputs.keys().collect::<BTreeSet<_>>()
            != layout.inputs.conditioning_inputs().iter().collect()
        {
            return Err(invalid_resource(
                "sequence conditioning differs from the complete declared input set",
            ));
        }
        for port in &layout.input_ports {
            if &port.value == layout.inputs.token_input() {
                continue;
            }
            let input = &self.inputs[&port.value];
            let matches = input.tensor.element_type() == port.tensor.element_type()
                && if port.token_rows {
                    input.tensor.dimensions().get(1..) == port.tensor.dimensions().get(1..)
                } else {
                    input.tensor == port.tensor
                };
            if !matches {
                return Err(invalid_resource(
                    "sequence conditioning schema differs from its compiled port",
                ));
            }
        }
        Ok(())
    }
}

impl<R: DeviceRuntime> SequenceSession<R> {
    /// Bind once, before any frame. The original accounted payload owners are
    /// retained through cancellation, native completion and terminal release.
    pub fn bind_conditioning_inputs(
        &self,
        inputs: Arc<SequenceConditioningInputs>,
    ) -> Result<(), VNextError> {
        let layout = self
            .resources()
            .request
            .plan
            .dynamic_pools()
            .candidate_state_layout
            .as_ref()
            .ok_or_else(|| {
                invalid_resource("sequence plan has no selected-prefix conditioning contract")
            })?;
        inputs.validate_layout(layout)?;
        let mut slot = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence state mutex poisoned"))?;
        let SequenceSessionSlotState::Active(active) = &mut *slot else {
            return Err(invalid_resource("conditioning requires an active sequence"));
        };
        if active.epoch != self.epoch
            || active.fingerprint != self.fingerprint
            || active.phase != SequenceSessionPhase::Open
            || active.active_frame.is_some()
            || active.has_participant_flights()
            || active.state_transfer.is_reserved()
            || active.retired_frames != 0
            || !matches!(active.completed_boundary, SequenceCompletedFrontier::Fresh)
            || active.conditioning_inputs.is_some()
        {
            return Err(invalid_resource(
                "conditioning can only bind once before the first sequence frame",
            ));
        }
        active.conditioning_inputs = Some(inputs);
        Ok(())
    }
}

impl<R: DeviceRuntime> StepResourceLease<R> {
    /// Validate the complete immutable input set before encoding even a backing
    /// initialization. Matching is by schema and actual contents, never hashes.
    pub(crate) fn validate_candidate_inputs(
        &self,
        uploads: &[SubmissionWaveInputUpload],
    ) -> Result<(), VNextError> {
        let Some(layout) = self.candidate_layout() else {
            return Ok(());
        };
        for upload in uploads {
            if upload.participant_index() as usize >= self.participants.len()
                || (!layout
                    .input_ports
                    .iter()
                    .any(|p| &p.node == upload.node_id() && p.ordinal == upload.input_ordinal())
                    && !layout
                        .output_only_ports
                        .iter()
                        .any(|(n, o)| n == upload.node_id() && *o == upload.input_ordinal()))
            {
                return Err(invalid_resource(
                    "candidate submission upload is outside declared external input ports",
                ));
            }
        }
        let work = self.work_shape().participant_work()?;
        for (index, (participant, work)) in self.participants.iter().zip(work).enumerate() {
            let slot = participant
                .session
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence state mutex poisoned"))?;
            let SequenceSessionSlotState::Active(active) = &*slot else {
                return Err(invalid_resource("candidate upload sequence is inactive"));
            };
            let bound = active.conditioning_inputs.clone();
            drop(slot);
            if !layout.inputs.conditioning_inputs().is_empty() && bound.is_none() {
                return Err(invalid_resource(
                    "candidate sequence lacks immutable conditioning evidence",
                ));
            }
            let span = work.token_span();
            let range = span.immediate_token_range();
            for port in &layout.input_ports {
                let is_token = &port.value == layout.inputs.token_input();
                let row = port.tensor.dimensions()[1..]
                    .iter()
                    .try_fold(port.tensor.element_type().size_bytes(), |n, &d| {
                        n.checked_mul(d)
                    })
                    .ok_or_else(|| invalid_resource("candidate input row overflows"))?;
                let expected = if port.token_rows {
                    range
                        .start
                        .checked_mul(row)
                        .zip(range.end.checked_mul(row))
                        .map(|(s, e)| s..e)
                        .ok_or_else(|| invalid_resource("candidate source span overflows"))?
                } else {
                    0..port.tensor.minimum_storage_bytes()?
                };
                let mut parts = uploads
                    .iter()
                    .filter(|u| {
                        u.participant_index() as usize == index
                            && u.node_id() == &port.node
                            && u.input_ordinal() == port.ordinal
                    })
                    .collect::<Vec<_>>();
                parts.sort_by_key(|u| u.logical_offset_bytes());
                let mut cursor = expected.start;
                for upload in parts {
                    let end = upload
                        .logical_offset_bytes()
                        .checked_add(upload.source_layout().byte_len()?)
                        .ok_or_else(|| invalid_resource("candidate input range overflows"))?;
                    if upload.logical_offset_bytes() != cursor
                        || end > expected.end
                        || end == cursor
                        || upload.source_layout().element_type() != port.tensor.element_type()
                    {
                        return Err(invalid_resource("candidate inputs must cover their exact source span without gaps or overlaps"));
                    }
                    if is_token {
                        let tokens = span.checkpoint_tokens().ok_or_else(|| {
                            invalid_resource("candidate input lacks token evidence")
                        })?;
                        let start = usize::try_from(cursor / 4)
                            .map_err(|_| invalid_resource("token offset overflows"))?;
                        for (offset, bytes) in upload.bytes().chunks_exact(4).enumerate() {
                            if tokens.get(start + offset).copied()
                                != Some(u32::from_le_bytes(bytes.try_into().unwrap()))
                            {
                                return Err(invalid_resource(
                                    "uploaded tokens differ from admitted exact token evidence",
                                ));
                            }
                        }
                    } else {
                        let source = &bound.as_ref().unwrap().inputs[&port.value];
                        let bytes = upload.managed_bytes().ok_or_else(|| {
                            invalid_resource(
                                "conditioning upload requires accounted immutable Host storage",
                            )
                        })?;
                        let offset = usize::try_from(cursor)
                            .map_err(|_| invalid_resource("conditioning offset overflows"))?;
                        if !source.matches_owner_or_bytes(offset, bytes)? {
                            return Err(invalid_resource("uploaded conditioning differs from the original immutable sequence source"));
                        }
                    }
                    cursor = end;
                }
                if cursor != expected.end {
                    return Err(invalid_resource(
                        "candidate submission omits a required input range",
                    ));
                }
            }
        }
        Ok(())
    }
}
