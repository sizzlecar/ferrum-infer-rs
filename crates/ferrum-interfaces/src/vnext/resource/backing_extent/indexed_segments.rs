//! A search index owned by the immutable physical segment list. It contains no
//! lease or allocation authority and cannot outlive or be paired with a different
//! segment list. Logical projections still compare every overlapping segment.
use super::*;

#[derive(Debug)]
pub(in crate::vnext::resource) struct IndexedBackingSegments {
    segments: Vec<BackingSegment>,
    // Failure to allocate or validate this optional index retains the original
    // bounded linear validator. It cannot turn invalid evidence into a match.
    ends: Option<Box<[u64]>>,
}

impl IndexedBackingSegments {
    pub(in crate::vnext::resource) fn new(segments: Vec<BackingSegment>) -> Self {
        let ends = Self::index(&segments);
        Self { segments, ends }
    }

    fn index(segments: &[BackingSegment]) -> Option<Box<[u64]>> {
        if segments.len() <= 1 {
            return None;
        }
        let mut ends = Vec::new();
        ends.try_reserve_exact(segments.len()).ok()?;
        let mut end = 0_u64;
        for segment in segments {
            validate_segment_range(
                segment.chunk_ordinal(),
                segment.chunk_generation(),
                segment.offset_bytes(),
                segment.length_bytes(),
            )
            .ok()?;
            end = end.checked_add(segment.length_bytes())?;
            ends.push(end);
        }
        Some(ends.into_boxed_slice())
    }

    pub(in crate::vnext::resource) fn range_matches_with_poll(
        &self,
        physical_offset_bytes: u64,
        size_bytes: u64,
        evidence: &[BackingSegment],
        mut poll: impl FnMut() -> bool,
    ) -> Result<Option<bool>, VNextError> {
        let Some(ends) = &self.ends else {
            return backing_segment_range_matches_with_poll(
                &self.segments,
                physical_offset_bytes,
                size_bytes,
                evidence,
                poll,
            );
        };
        // Validate in the original coordinate system before translating it.
        // Subtraction must not hide an overflowing original window.
        let physical_end = physical_offset_bytes
            .checked_add(size_bytes)
            .ok_or_else(|| invalid_resource("logical backing projection range overflows u64"))?;
        if size_bytes == 0 {
            return Err(invalid_resource(
                "logical backing projection must have non-zero size",
            ));
        }
        if !poll() {
            return Ok(None);
        }
        if physical_end > *ends.last().expect("nonempty immutable index") {
            return Err(invalid_resource(
                "logical backing projection exceeds its physical extent",
            ));
        }
        // At most usize::BITS comparisons; the original per-overlap polling
        // remains in the validator below. No work scales with skipped extents.
        let start = ends.partition_point(|end| *end <= physical_offset_bytes);
        let origin = start.checked_sub(1).map_or(0, |i| ends[i]);
        backing_segment_range_matches_with_poll(
            &self.segments[start..],
            physical_offset_bytes - origin,
            size_bytes,
            evidence,
            poll,
        )
    }
}

impl std::ops::Deref for IndexedBackingSegments {
    type Target = [BackingSegment];

    fn deref(&self) -> &Self::Target {
        &self.segments
    }
}

impl<'a> IntoIterator for &'a IndexedBackingSegments {
    type Item = &'a BackingSegment;
    type IntoIter = std::slice::Iter<'a, BackingSegment>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.iter()
    }
}

#[cfg(test)]
mod tests;
