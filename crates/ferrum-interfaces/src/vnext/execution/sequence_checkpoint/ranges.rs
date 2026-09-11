use std::collections::BTreeMap;
use std::ops::Range;

use super::*;
use crate::vnext::{CheckpointBackingRequest, CheckpointBackingRequests};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceCheckpointCopyRange {
    source: Range<u64>,
    checkpoint_offset: u64,
}

impl SequenceCheckpointCopyRange {
    pub fn source(&self) -> Range<u64> {
        self.source.clone()
    }
    pub fn checkpoint_offset(&self) -> u64 {
        self.checkpoint_offset
    }
    pub fn length_bytes(&self) -> u64 {
        self.source.end - self.source.start
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceCheckpointResourceRanges {
    resource_id: ResourceId,
    logical_bytes: u64,
    ranges: Vec<SequenceCheckpointCopyRange>,
}

impl SequenceCheckpointResourceRanges {
    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }
    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }
    pub fn ranges(&self) -> &[SequenceCheckpointCopyRange] {
        &self.ranges
    }
}

/// Compact bytes to allocate and copy, derived from a trusted layout. Source
/// ranges must still be checked against the actual session backing under its
/// state-transfer guard. This is not a completed-boundary or submit authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceCheckpointBytePlan {
    plan_hash: PlanHash,
    layout_fingerprint: String,
    boundary: u64,
    logical_bytes: u64,
    resources: Vec<SequenceCheckpointResourceRanges>,
}

impl SequenceCheckpointBytePlan {
    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }
    pub fn layout_fingerprint(&self) -> &str {
        &self.layout_fingerprint
    }
    pub fn boundary(&self) -> u64 {
        self.boundary
    }
    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }
    pub fn resources(&self) -> &[SequenceCheckpointResourceRanges] {
        &self.resources
    }

    pub(crate) fn backing_requests(&self) -> Result<CheckpointBackingRequests, VNextError> {
        CheckpointBackingRequests::new(
            self.plan_hash.clone(),
            self.resources
                .iter()
                .map(|resource| {
                    CheckpointBackingRequest::new(
                        resource.resource_id.clone(),
                        resource.logical_bytes,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }
}

impl SequenceCheckpointState {
    fn source_range(&self, boundary: u64) -> Result<Range<u64>, VNextError> {
        let length = match self.layout {
            ProviderCheckpointStateLayout::ContiguousBoundaryValue => {
                self.tensor.minimum_storage_bytes()?
            }
            ProviderCheckpointStateLayout::TokenMajorPrefix => self
                .tensor
                .minimum_storage_bytes()?
                .checked_mul(boundary)
                .ok_or_else(|| invalid_plan("checkpoint prefix byte length overflows u64"))?,
        };
        let end = self
            .offset_bytes
            .checked_add(length)
            .ok_or_else(|| invalid_plan("checkpoint byte end overflows u64"))?;
        let capacity = self.descriptor.evaluate_logical_request_bytes_for_shape(
            self.descriptor.demand().theoretical_maximum_shape(),
        )?;
        if length == 0 || end > capacity {
            return Err(invalid_plan(
                "checkpoint state range exceeds its proven resource capacity",
            ));
        }
        Ok(self.offset_bytes..end)
    }

    fn maximum_source_range(&self) -> Result<Range<u64>, VNextError> {
        match self.layout {
            ProviderCheckpointStateLayout::ContiguousBoundaryValue => self.source_range(1),
            ProviderCheckpointStateLayout::TokenMajorPrefix => {
                let capacity = self.descriptor.evaluate_logical_request_bytes_for_shape(
                    self.descriptor.demand().theoretical_maximum_shape(),
                )?;
                self.source_range(capacity / self.tensor.minimum_storage_bytes()?)
            }
        }
    }
}

impl SequenceCheckpointLayout {
    pub(super) fn validate_aliases(&self) -> Result<(), VNextError> {
        let mut groups = BTreeMap::<&ResourceId, Vec<&SequenceCheckpointState>>::new();
        for state in &self.data.states {
            groups.entry(&state.resource_id).or_default().push(state);
        }
        for states in groups.values() {
            let mut ranges = states
                .iter()
                .map(|state| Ok((state.maximum_source_range()?, *state)))
                .collect::<Result<Vec<_>, VNextError>>()?;
            ranges.sort_by_key(|(range, _)| (range.start, range.end));
            for pair in ranges.windows(2) {
                let (left_range, left) = &pair[0];
                let (right_range, right) = &pair[1];
                if left_range.end > right_range.start
                    && !(left_range == right_range
                        && left.tensor == right.tensor
                        && left.layout == right.layout
                        && left.semantics == right.semantics
                        && left.storage == right.storage
                        && left.initialization == right.initialization)
                {
                    return Err(invalid_plan(
                        "checkpoint state aliases do not prove identical content and ABI",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn byte_plan(
        &self,
        plan_hash: PlanHash,
        boundary: u64,
    ) -> Result<SequenceCheckpointBytePlan, VNextError> {
        if boundary == 0 {
            return Err(invalid_plan(
                "checkpoint requires a positive completed boundary",
            ));
        }
        let mut groups = BTreeMap::<ResourceId, Vec<Range<u64>>>::new();
        for state in &self.data.states {
            groups
                .entry(state.resource_id.clone())
                .or_default()
                .push(state.source_range(boundary)?);
        }
        let mut total = 0_u64;
        let mut resources = Vec::new();
        for (resource_id, mut ranges) in groups {
            ranges.sort_by_key(|range| (range.start, range.end));
            let mut merged: Vec<Range<u64>> = Vec::new();
            for range in ranges {
                if let Some(previous) = merged
                    .last_mut()
                    .filter(|previous| range.start <= previous.end)
                {
                    previous.end = previous.end.max(range.end);
                } else {
                    merged.push(range);
                }
            }
            let mut logical_bytes = 0_u64;
            let ranges = merged
                .into_iter()
                .map(|source| {
                    let checkpoint_offset = logical_bytes;
                    logical_bytes = logical_bytes
                        .checked_add(source.end - source.start)
                        .ok_or_else(|| {
                            invalid_plan("compact checkpoint resource size overflows u64")
                        })?;
                    Ok(SequenceCheckpointCopyRange {
                        source,
                        checkpoint_offset,
                    })
                })
                .collect::<Result<Vec<_>, VNextError>>()?;
            total = total
                .checked_add(logical_bytes)
                .ok_or_else(|| invalid_plan("checkpoint total bytes overflow u64"))?;
            resources.push(SequenceCheckpointResourceRanges {
                resource_id,
                logical_bytes,
                ranges,
            });
        }
        Ok(SequenceCheckpointBytePlan {
            plan_hash,
            layout_fingerprint: self.fingerprint()?,
            boundary,
            logical_bytes: total,
            resources,
        })
    }
}
