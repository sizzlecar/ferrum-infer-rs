use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use super::{invalid_resource, DynamicBackingPoolId, VNextError};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BackingChunkIdentity {
    pool_id: DynamicBackingPoolId,
    ordinal: u32,
    generation: u64,
}

impl BackingChunkIdentity {
    pub(super) fn from_parts(
        pool_id: DynamicBackingPoolId,
        ordinal: u32,
        generation: u64,
    ) -> Result<Self, VNextError> {
        if ordinal == 0 || generation == 0 {
            return Err(invalid_resource("backing chunk identity is invalid"));
        }
        Ok(Self {
            pool_id,
            ordinal,
            generation,
        })
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackingSegment {
    chunk: BackingChunkIdentity,
    offset_bytes: u64,
    length_bytes: u64,
}

impl BackingSegment {
    pub(crate) fn new(
        chunk: BackingChunkIdentity,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, VNextError> {
        Self::from_chunk(
            &chunk.pool_id,
            chunk.ordinal,
            chunk.generation,
            offset_bytes,
            length_bytes,
        )
    }

    pub(super) fn from_chunk(
        pool_id: &DynamicBackingPoolId,
        chunk_ordinal: u32,
        chunk_generation: u64,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, VNextError> {
        if chunk_ordinal == 0
            || chunk_generation == 0
            || length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
        {
            return Err(invalid_resource(
                "backing segment has invalid chunk identity or physical range",
            ));
        }
        Ok(Self {
            chunk: BackingChunkIdentity::from_parts(
                pool_id.clone(),
                chunk_ordinal,
                chunk_generation,
            )?,
            offset_bytes,
            length_bytes,
        })
    }

    pub fn chunk(&self) -> &BackingChunkIdentity {
        &self.chunk
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        self.chunk.pool_id()
    }

    pub const fn chunk_ordinal(&self) -> u32 {
        self.chunk.ordinal()
    }

    pub const fn chunk_generation(&self) -> u64 {
        self.chunk.generation()
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }
}

pub(super) fn backing_segment_range(
    segments: &[BackingSegment],
    physical_offset_bytes: u64,
    size_bytes: u64,
) -> Result<Vec<BackingSegment>, VNextError> {
    let physical_end = physical_offset_bytes
        .checked_add(size_bytes)
        .ok_or_else(|| invalid_resource("logical backing projection range overflows u64"))?;
    if size_bytes == 0 {
        return Err(invalid_resource(
            "logical backing projection must have non-zero size",
        ));
    }
    let mut physical_cursor = 0_u64;
    let mut covered = 0_u64;
    let mut projection = Vec::new();
    for segment in segments {
        if physical_cursor >= physical_end {
            break;
        }
        let segment_end = physical_cursor
            .checked_add(segment.length_bytes())
            .ok_or_else(|| invalid_resource("physical backing extent range overflows u64"))?;
        let overlap_start = physical_cursor.max(physical_offset_bytes);
        let overlap_end = segment_end.min(physical_end);
        if overlap_start < overlap_end {
            let within_segment = overlap_start - physical_cursor;
            let translated_offset = segment
                .offset_bytes()
                .checked_add(within_segment)
                .ok_or_else(|| invalid_resource("backing projection offset overflows u64"))?;
            let length = overlap_end - overlap_start;
            projection.push(BackingSegment::from_chunk(
                segment.pool_id(),
                segment.chunk_ordinal(),
                segment.chunk_generation(),
                translated_offset,
                length,
            )?);
            covered = covered.checked_add(length).ok_or_else(|| {
                invalid_resource("logical backing projection coverage overflows u64")
            })?;
        }
        physical_cursor = segment_end;
    }
    if covered != size_bytes {
        return Err(invalid_resource(
            "logical backing projection exceeds its physical extent",
        ));
    }
    Ok(projection)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FreeExtent {
    pub(super) chunk_generation: u64,
    pub(super) length_bytes: u64,
}

#[derive(Debug, Default)]
pub(super) struct FreeExtentIndex {
    pub(super) by_offset: BTreeMap<(u32, u64), FreeExtent>,
    pub(super) by_size: BTreeSet<(u64, u32, u64, u64)>,
    pub(super) free_bytes: u64,
    pub(super) search_probes: u64,
}

impl FreeExtentIndex {
    fn rollback_segments(&mut self, segments: &[BackingSegment]) -> Result<(), VNextError> {
        for segment in segments.iter().rev() {
            self.release(segment)?;
        }
        Ok(())
    }

    fn with_rollback_context(
        &mut self,
        segments: &[BackingSegment],
        error: VNextError,
    ) -> VNextError {
        match self.rollback_segments(segments) {
            Ok(()) => error,
            Err(rollback) => invalid_resource(format!(
                "dynamic allocator failed and its journal rollback also failed: {error}; rollback: {rollback}"
            )),
        }
    }

    pub(super) fn insert_extent(
        &mut self,
        chunk_ordinal: u32,
        chunk_generation: u64,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<(), VNextError> {
        if chunk_ordinal == 0
            || chunk_generation == 0
            || length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
            || self.by_offset.contains_key(&(chunk_ordinal, offset_bytes))
        {
            return Err(invalid_resource("free extent identity or range is invalid"));
        }
        let end = offset_bytes + length_bytes;
        if self
            .by_offset
            .range(..(chunk_ordinal, offset_bytes))
            .next_back()
            .is_some_and(|(&(ordinal, previous_offset), previous)| {
                ordinal == chunk_ordinal && previous_offset + previous.length_bytes > offset_bytes
            })
            || self
                .by_offset
                .range((chunk_ordinal, offset_bytes)..)
                .next()
                .is_some_and(|(&(ordinal, next_offset), _)| {
                    ordinal == chunk_ordinal && next_offset < end
                })
        {
            return Err(invalid_resource("free extent overlaps an existing extent"));
        }
        let next_free = self
            .free_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_resource("free extent bytes overflow u64"))?;
        self.by_offset.insert(
            (chunk_ordinal, offset_bytes),
            FreeExtent {
                chunk_generation,
                length_bytes,
            },
        );
        assert!(self
            .by_size
            .insert((length_bytes, chunk_ordinal, chunk_generation, offset_bytes,)));
        self.free_bytes = next_free;
        Ok(())
    }

    pub(super) fn remove_extent(
        &mut self,
        chunk_ordinal: u32,
        offset_bytes: u64,
    ) -> Result<FreeExtent, VNextError> {
        let extent = *self
            .by_offset
            .get(&(chunk_ordinal, offset_bytes))
            .ok_or_else(|| invalid_resource("free extent journal references a missing range"))?;
        let size_key = (
            extent.length_bytes,
            chunk_ordinal,
            extent.chunk_generation,
            offset_bytes,
        );
        if !self.by_size.contains(&size_key) {
            return Err(invalid_resource("free extent indexes diverged"));
        }
        let next_free_bytes = self
            .free_bytes
            .checked_sub(extent.length_bytes)
            .ok_or_else(|| invalid_resource("free extent bytes underflowed"))?;
        self.by_offset.remove(&(chunk_ordinal, offset_bytes));
        assert!(self.by_size.remove(&size_key));
        self.free_bytes = next_free_bytes;
        Ok(extent)
    }

    pub(super) fn allocate_contiguous(
        &mut self,
        pool_id: &DynamicBackingPoolId,
        size_bytes: u64,
    ) -> Result<Option<BackingSegment>, VNextError> {
        self.search_probes = self.search_probes.saturating_add(1);
        let selected = self.by_size.range((size_bytes, 0, 0, 0)..).next().copied();
        let Some((length_bytes, chunk_ordinal, chunk_generation, offset_bytes)) = selected else {
            return Ok(None);
        };
        let segment = BackingSegment::from_chunk(
            pool_id,
            chunk_ordinal,
            chunk_generation,
            offset_bytes,
            size_bytes,
        )?;
        let removed = self.remove_extent(chunk_ordinal, offset_bytes)?;
        debug_assert_eq!(removed.length_bytes, length_bytes);
        debug_assert_eq!(removed.chunk_generation, chunk_generation);
        if size_bytes < length_bytes {
            if let Err(error) = self.insert_extent(
                chunk_ordinal,
                chunk_generation,
                offset_bytes + size_bytes,
                length_bytes - size_bytes,
            ) {
                let restore =
                    self.insert_extent(chunk_ordinal, chunk_generation, offset_bytes, length_bytes);
                return Err(match restore {
                    Ok(()) => error,
                    Err(rollback) => invalid_resource(format!(
                        "contiguous allocator failed and could not restore its selected extent: {error}; rollback: {rollback}"
                    )),
                });
            }
        }
        Ok(Some(segment))
    }

    pub(super) fn allocate_paged(
        &mut self,
        pool_id: &DynamicBackingPoolId,
        size_bytes: u64,
        block_bytes: u64,
    ) -> Result<Option<Vec<BackingSegment>>, VNextError> {
        if size_bytes == 0 || size_bytes % block_bytes != 0 {
            return Err(invalid_resource(
                "paged backing reservation is not block aligned",
            ));
        }
        if self.free_bytes < size_bytes {
            return Ok(None);
        }
        let mut remaining = size_bytes;
        let mut segments = Vec::new();
        while remaining != 0 {
            self.search_probes = self.search_probes.saturating_add(1);
            let Some((&(chunk_ordinal, offset_bytes), &extent)) = self.by_offset.first_key_value()
            else {
                self.rollback_segments(&segments)?;
                return Ok(None);
            };
            if extent.length_bytes % block_bytes != 0 {
                let error = invalid_resource("paged free extent lost fixed-block alignment");
                return Err(self.with_rollback_context(&segments, error));
            }
            let take = extent.length_bytes.min(remaining);
            let segment = match BackingSegment::from_chunk(
                pool_id,
                chunk_ordinal,
                extent.chunk_generation,
                offset_bytes,
                take,
            ) {
                Ok(segment) => segment,
                Err(error) => return Err(self.with_rollback_context(&segments, error)),
            };
            if let Err(error) = self.remove_extent(chunk_ordinal, offset_bytes) {
                return Err(self.with_rollback_context(&segments, error));
            }
            if take < extent.length_bytes {
                if let Err(error) = self.insert_extent(
                    chunk_ordinal,
                    extent.chunk_generation,
                    offset_bytes + take,
                    extent.length_bytes - take,
                ) {
                    let restore = self.insert_extent(
                        chunk_ordinal,
                        extent.chunk_generation,
                        offset_bytes,
                        extent.length_bytes,
                    );
                    let error = match restore {
                        Ok(()) => error,
                        Err(rollback) => invalid_resource(format!(
                            "paged allocator failed and could not restore its selected extent: {error}; rollback: {rollback}"
                        )),
                    };
                    return Err(self.with_rollback_context(&segments, error));
                }
            }
            segments.push(segment);
            remaining -= take;
        }
        Ok(Some(segments))
    }

    pub(super) fn release(&mut self, segment: &BackingSegment) -> Result<(), VNextError> {
        let chunk_ordinal = segment.chunk_ordinal();
        let chunk_generation = segment.chunk_generation();
        let mut offset_bytes = segment.offset_bytes();
        let mut length_bytes = segment.length_bytes();
        if let Some((&(ordinal, previous_offset), &previous)) = self
            .by_offset
            .range(..(chunk_ordinal, offset_bytes))
            .next_back()
        {
            if ordinal == chunk_ordinal
                && previous.chunk_generation == chunk_generation
                && previous_offset + previous.length_bytes == offset_bytes
            {
                self.remove_extent(ordinal, previous_offset)?;
                offset_bytes = previous_offset;
                length_bytes = length_bytes
                    .checked_add(previous.length_bytes)
                    .ok_or_else(|| invalid_resource("coalesced free extent overflows u64"))?;
            }
        }
        if let Some((&(ordinal, next_offset), &next)) =
            self.by_offset.range((chunk_ordinal, offset_bytes)..).next()
        {
            if ordinal == chunk_ordinal
                && next.chunk_generation == chunk_generation
                && offset_bytes + length_bytes == next_offset
            {
                self.remove_extent(ordinal, next_offset)?;
                length_bytes = length_bytes
                    .checked_add(next.length_bytes)
                    .ok_or_else(|| invalid_resource("coalesced free extent overflows u64"))?;
            }
        }
        self.insert_extent(chunk_ordinal, chunk_generation, offset_bytes, length_bytes)
    }

    pub(super) fn largest_contiguous_bytes(&self) -> u64 {
        self.by_size
            .last()
            .map_or(0, |(length_bytes, _, _, _)| *length_bytes)
    }
}
