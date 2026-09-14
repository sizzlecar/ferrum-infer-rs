use super::{
    invalid_operation, DeviceBufferRetention, OperationBufferRegions, OperationPhysicalRegion,
};
use crate::vnext::{CopyRegion, VNextError};

/// One copy between translated logical views. Transfer ownership must retain
/// both endpoint claims through submission completion; this is no execution
/// authority and does not authorize reading uninitialized state.
pub(crate) struct OperationBufferCopy<'a, B> {
    source: &'a B,
    destination: &'a B,
    region: CopyRegion,
    _source_retention: DeviceBufferRetention,
    _destination_retention: DeviceBufferRetention,
}

impl<'a, B> OperationBufferCopy<'a, B> {
    pub(crate) fn buffers_and_region(&self) -> (&'a B, &'a B, CopyRegion) {
        (self.source, self.destination, self.region)
    }
}

impl<'a, B> OperationBufferRegions<'a, B> {
    /// Pairs exactly these logical ranges despite different physical segment
    /// boundaries. No page padding or unused capacity is added to the copy.
    pub(crate) fn copies_to(
        &self,
        destination: &Self,
    ) -> Result<Vec<OperationBufferCopy<'a, B>>, VNextError> {
        if self.logical_length_bytes != destination.logical_length_bytes {
            return Err(invalid_operation(
                "copy views have different logical lengths",
            ));
        }
        pair_regions(
            self.iter().collect(),
            destination.iter().collect(),
            self.logical_offset_bytes,
            destination.logical_offset_bytes,
            self.logical_length_bytes,
        )
    }
}

fn validate_coverage<B>(
    regions: &[OperationPhysicalRegion<'_, B>],
    start: u64,
    length: u64,
) -> Result<(), VNextError> {
    let end = start
        .checked_add(length)
        .filter(|_| length > 0)
        .ok_or_else(|| invalid_operation("copy logical range is empty or overflows"))?;
    let mut cursor = start;
    for region in regions {
        if region.logical_offset_bytes != cursor || region.length_bytes == 0 {
            return Err(invalid_operation(
                "copy regions have a gap, overlap, or empty segment",
            ));
        }
        region
            .physical_offset_bytes
            .checked_add(region.length_bytes)
            .ok_or_else(|| invalid_operation("copy physical range overflows"))?;
        cursor = cursor
            .checked_add(region.length_bytes)
            .ok_or_else(|| invalid_operation("copy logical coverage overflows"))?;
        if cursor > end {
            return Err(invalid_operation("copy regions exceed the logical range"));
        }
    }
    if cursor != end {
        return Err(invalid_operation(
            "copy regions do not cover the logical range",
        ));
    }
    Ok(())
}

fn pair_regions<'a, B>(
    source: Vec<OperationPhysicalRegion<'a, B>>,
    destination: Vec<OperationPhysicalRegion<'a, B>>,
    source_start: u64,
    destination_start: u64,
    length: u64,
) -> Result<Vec<OperationBufferCopy<'a, B>>, VNextError> {
    validate_coverage(&source, source_start, length)?;
    validate_coverage(&destination, destination_start, length)?;
    // Fresh checkpoint/restore backing must be disjoint. Reject even a
    // cross-segment alias before returning any executable copy regions.
    for src in &source {
        for dst in &destination {
            if std::ptr::eq(src.buffer, dst.buffer)
                && src.physical_offset_bytes < dst.physical_offset_bytes + dst.length_bytes
                && dst.physical_offset_bytes < src.physical_offset_bytes + src.length_bytes
            {
                return Err(invalid_operation("copy source and destination overlap"));
            }
        }
    }

    let mut copies = Vec::new();
    let (mut src_index, mut dst_index) = (0, 0);
    let (mut src_used, mut dst_used) = (0, 0);
    let mut copied = 0;
    while copied < length {
        let src = &source[src_index];
        let dst = &destination[dst_index];
        let bytes = (src.length_bytes - src_used).min(dst.length_bytes - dst_used);
        copies.push(OperationBufferCopy {
            source: src.buffer,
            destination: dst.buffer,
            region: CopyRegion::new(
                src.physical_offset_bytes + src_used,
                dst.physical_offset_bytes + dst_used,
                bytes,
            )?,
            _source_retention: src.retention.clone(),
            _destination_retention: dst.retention.clone(),
        });
        copied += bytes;
        src_used += bytes;
        dst_used += bytes;
        if src_used == src.length_bytes {
            src_index += 1;
            src_used = 0;
        }
        if dst_used == dst.length_bytes {
            dst_index += 1;
            dst_used = 0;
        }
    }
    Ok(copies)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::Arc;

    fn region<'a, B>(
        buffer: &'a B,
        logical: u64,
        physical: u64,
        length: u64,
        owner: &Arc<()>,
    ) -> OperationPhysicalRegion<'a, B> {
        OperationPhysicalRegion {
            buffer,
            logical_offset_bytes: logical,
            physical_offset_bytes: physical,
            length_bytes: length,
            retention: DeviceBufferRetention::plan(Arc::clone(owner)),
        }
    }

    #[test]
    fn copies_fragmented_views_without_reading_or_writing_capacity_tails() {
        let owner = Arc::new(());
        let first = RefCell::new(vec![99, 99, 1, 2, 3, 99]);
        let second = RefCell::new(vec![99, 4, 5, 6, 7, 8, 99]);
        let target_a = RefCell::new(vec![77; 7]);
        let target_b = RefCell::new(vec![77; 8]);
        let copies = pair_regions(
            vec![
                region(&first, 5, 2, 3, &owner),
                region(&second, 8, 1, 5, &owner),
            ],
            vec![
                region(&target_a, 11, 1, 5, &owner),
                region(&target_b, 16, 3, 3, &owner),
            ],
            5,
            11,
            8,
        )
        .unwrap();
        for copy in &copies {
            let (src, dst, range) = copy.buffers_and_region();
            let src_start = range.source_offset_bytes() as usize;
            let dst_start = range.destination_offset_bytes() as usize;
            let length = range.length_bytes() as usize;
            dst.borrow_mut()[dst_start..dst_start + length]
                .copy_from_slice(&src.borrow()[src_start..src_start + length]);
        }
        assert_eq!(*target_a.borrow(), [77, 1, 2, 3, 4, 5, 77]);
        assert_eq!(*target_b.borrow(), [77, 77, 77, 6, 7, 8, 77, 77]);
        assert_eq!(*first.borrow(), [99, 99, 1, 2, 3, 99]);
        assert_eq!(*second.borrow(), [99, 4, 5, 6, 7, 8, 99]);
    }

    #[test]
    fn translated_windows_keep_each_views_physical_origin() {
        use super::super::{OperationBufferStorageKind, OperationRegionSource};

        let source_buffer = 1;
        let destination_buffer = 2;
        let owner = Arc::new(());
        let source = OperationBufferRegions {
            storage_kind: OperationBufferStorageKind::DynamicContiguous,
            logical_offset_bytes: 6,
            logical_length_bytes: 4,
            source: OperationRegionSource::Contiguous {
                buffer: &source_buffer,
                physical_base_offset_bytes: 32,
                retention: DeviceBufferRetention::plan(Arc::clone(&owner)),
            },
        };
        let mut destination = OperationBufferRegions {
            storage_kind: OperationBufferStorageKind::DynamicContiguous,
            logical_offset_bytes: 3,
            logical_length_bytes: 4,
            source: OperationRegionSource::Contiguous {
                buffer: &destination_buffer,
                physical_base_offset_bytes: 64,
                retention: DeviceBufferRetention::plan(owner),
            },
        };
        let copies = source.copies_to(&destination).unwrap();
        let (actual_source, actual_destination, region) = copies[0].buffers_and_region();
        assert!(std::ptr::eq(actual_source, &source_buffer));
        assert!(std::ptr::eq(actual_destination, &destination_buffer));
        assert_eq!(region.source_offset_bytes(), 38);
        assert_eq!(region.destination_offset_bytes(), 67);
        assert_eq!(region.length_bytes(), 4);
        destination.logical_length_bytes = 5;
        assert!(source.copies_to(&destination).is_err());
    }

    #[test]
    fn rejects_incomplete_or_overlapping_logical_coverage() {
        let owner = Arc::new(());
        let source = 1;
        let destination = 2;
        for segments in [
            vec![],
            vec![region(&source, 0, 0, 3, &owner)],
            vec![region(&source, 0, 0, 5, &owner)],
            vec![region(&source, 0, 0, 0, &owner)],
            vec![
                region(&source, 0, 0, 2, &owner),
                region(&source, 3, 2, 1, &owner),
            ],
            vec![
                region(&source, 0, 0, 3, &owner),
                region(&source, 2, 3, 1, &owner),
            ],
            vec![region(&source, 0, u64::MAX, 4, &owner)],
        ] {
            assert!(pair_regions(
                segments,
                vec![region(&destination, 0, 0, 4, &owner)],
                0,
                0,
                4,
            )
            .is_err());
        }
    }

    #[test]
    fn rejects_cross_segment_source_destination_alias_before_copy() {
        let owner = Arc::new(());
        let buffer = 0;
        assert!(pair_regions(
            vec![
                region(&buffer, 0, 0, 4, &owner),
                region(&buffer, 4, 8, 4, &owner)
            ],
            vec![
                region(&buffer, 0, 8, 4, &owner),
                region(&buffer, 4, 0, 4, &owner)
            ],
            0,
            0,
            8,
        )
        .is_err());
        assert!(pair_regions(
            vec![region(&buffer, 0, 0, 4, &owner)],
            vec![region(&buffer, 0, 4, 4, &owner)],
            0,
            0,
            4,
        )
        .is_ok());
    }

    #[test]
    fn planned_copies_retain_both_endpoints_until_dropped() {
        let source_owner = Arc::new(());
        let destination_owner = Arc::new(());
        let source_weak = Arc::downgrade(&source_owner);
        let destination_weak = Arc::downgrade(&destination_owner);
        let source = 1;
        let destination = 2;
        let copies = pair_regions(
            vec![region(&source, 0, 0, 4, &source_owner)],
            vec![region(&destination, 0, 0, 4, &destination_owner)],
            0,
            0,
            4,
        )
        .unwrap();
        drop(source_owner);
        drop(destination_owner);
        assert!(source_weak.upgrade().is_some());
        assert!(destination_weak.upgrade().is_some());
        drop(copies);
        assert!(source_weak.upgrade().is_none());
        assert!(destination_weak.upgrade().is_none());
    }
}
