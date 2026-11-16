//! Bounded contiguous partitions for explicitly row-independent programs.
//! This declares geometry only; it owns no buffers and schedules no work.
use std::{num::NonZeroU64, ops::Range};

#[derive(Clone, Copy, Debug)]
pub struct RowChunks {
    rows: NonZeroU64,
    maximum_rows: NonZeroU64,
}

impl RowChunks {
    pub fn new(rows: NonZeroU64, maximum_rows: NonZeroU64) -> Self {
        Self { rows, maximum_rows }
    }
    pub fn count(self) -> u64 {
        self.rows.get().div_ceil(self.maximum_rows.get())
    }
    pub fn ranges(self) -> impl Iterator<Item = Range<u64>> {
        let mut start = 0;
        std::iter::from_fn(move || {
            if start == self.rows.get() {
                return None;
            }
            let length = self.maximum_rows.get().min(self.rows.get() - start);
            let range = start..start + length;
            start = range.end;
            Some(range)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partitions_exact_rows_including_tails_and_extreme_extents() {
        for (rows, width, expected) in [
            (7, 3, vec![0..3, 3..6, 6..7]),
            (6, 3, vec![0..3, 3..6]),
            (2, 9, vec![0..2]),
            (
                u64::MAX,
                u64::MAX - 1,
                vec![0..u64::MAX - 1, u64::MAX - 1..u64::MAX],
            ),
        ] {
            let chunks = RowChunks::new(
                NonZeroU64::new(rows).unwrap(),
                NonZeroU64::new(width).unwrap(),
            );
            assert_eq!(chunks.count(), expected.len() as u64);
            assert_eq!(chunks.ranges().collect::<Vec<_>>(), expected);
        }
    }
}
