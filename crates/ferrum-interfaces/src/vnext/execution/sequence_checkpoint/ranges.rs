use std::collections::BTreeMap;
use std::ops::Range;

use super::*;
use crate::vnext::{CheckpointBackingRequest, CheckpointBackingRequests, StridedCopyRegion};

mod key_value_block;
fn is_false(value: &bool) -> bool {
    !*value
}

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
    #[serde(skip_serializing_if = "is_false")]
    physical_prefix_coordinates: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    strided_ranges: Vec<StridedCopyRegion>,
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

    pub fn strided_ranges(&self) -> &[StridedCopyRegion] {
        &self.strided_ranges
    }

    pub(crate) fn sequence_copy_bound(&self, logical_bytes: u64, capacity_bytes: u64) -> u64 {
        if self.physical_prefix_coordinates {
            capacity_bytes
        } else {
            logical_bytes
        }
    }

    pub(crate) fn physical_copy_extent(&self) -> Option<u64> {
        self.physical_prefix_coordinates.then(|| {
            self.ranges
                .iter()
                .map(|range| range.source.end)
                .chain(
                    self.strided_ranges
                        .iter()
                        .map(|range| range.source_end_bytes().expect("checked copy extent")),
                )
                .max()
                .unwrap_or(0)
        })
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
    fn source_copy_plan(
        &self,
        boundary: u64,
    ) -> Result<(Vec<Range<u64>>, Vec<StridedCopyRegion>), VNextError> {
        if let ProviderCheckpointStateLayout::PagedKeyValueBlockPrefix {
            tokens_per_block,
            key_pack_elements,
        } = self.layout
        {
            let shape = self.descriptor.demand().theoretical_maximum_shape();
            let logical_capacity = self
                .descriptor
                .evaluate_logical_request_bytes_for_shape(shape)?;
            let physical_capacity = self.descriptor.evaluate_request_bytes_for_shape(shape)?;
            return key_value_block::source_copy_plan(
                &self.tensor,
                tokens_per_block.get(),
                key_pack_elements.get(),
                self.offset_bytes,
                boundary,
                logical_capacity,
                physical_capacity,
            );
        }
        Ok((vec![self.source_range(boundary)?], Vec::new()))
    }

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
            ProviderCheckpointStateLayout::PagedKeyValueBlockPrefix { .. } => {
                return Err(invalid_plan(
                    "blocked key/value prefix requires its exact copy ranges",
                ));
            }
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
            ProviderCheckpointStateLayout::TokenMajorPrefix
            | ProviderCheckpointStateLayout::PagedKeyValueBlockPrefix { .. } => {
                let capacity = self.descriptor.evaluate_logical_request_bytes_for_shape(
                    self.descriptor.demand().theoretical_maximum_shape(),
                )?;
                let (ranges, strided) =
                    self.source_copy_plan(capacity / self.tensor.minimum_storage_bytes()?)?;
                let end = ranges
                    .iter()
                    .map(|range| range.end)
                    .chain(
                        strided
                            .iter()
                            .map(|range| range.source_end_bytes().expect("checked copy extent")),
                    )
                    .max()
                    .ok_or_else(|| invalid_plan("checkpoint maximum prefix is empty"))?;
                Ok(self.offset_bytes..end)
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

    // Only the owning ExecutionPlan entrypoint may attach a plan hash. Resource
    // and completion code must not relabel another plan's trusted layout.
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
        let mut groups =
            BTreeMap::<ResourceId, (Vec<Range<u64>>, Vec<StridedCopyRegion>, bool)>::new();
        for state in &self.data.states {
            let (ranges, rectangles, physical_coordinates) =
                groups.entry(state.resource_id.clone()).or_default();
            let (contiguous, strided) = state.source_copy_plan(boundary)?;
            ranges.extend(contiguous);
            for rectangle in strided {
                if !rectangles.contains(&rectangle) {
                    rectangles.push(rectangle);
                }
            }
            *physical_coordinates |= matches!(
                state.layout,
                ProviderCheckpointStateLayout::PagedKeyValueBlockPrefix { .. }
            );
        }
        let mut total = 0_u64;
        let mut resources = Vec::new();
        for (resource_id, (mut ranges, mut strided, physical_prefix_coordinates)) in groups {
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
            strided.sort_by_key(|range| range.source_offset_bytes());
            let mut strided_ranges = Vec::with_capacity(strided.len());
            for range in strided {
                strided_ranges.push(StridedCopyRegion::new(
                    range.source_offset_bytes(),
                    logical_bytes,
                    range.width_bytes(),
                    range.height(),
                    range.source_pitch_bytes(),
                    range.width_bytes(),
                )?);
                logical_bytes = logical_bytes
                    .checked_add(range.length_bytes()?)
                    .ok_or_else(|| invalid_plan("compact strided checkpoint bytes overflow u64"))?;
            }
            total = total
                .checked_add(logical_bytes)
                .ok_or_else(|| invalid_plan("checkpoint total bytes overflow u64"))?;
            resources.push(SequenceCheckpointResourceRanges {
                resource_id,
                logical_bytes,
                ranges,
                physical_prefix_coordinates,
                strided_ranges,
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
